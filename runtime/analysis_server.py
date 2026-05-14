"""KataGo Analysis Server.

Async pipelined wrapper around the KataGo analysis CLI subprocess.

KataGo's analysis mode reads JSON queries from stdin and writes JSON responses
to stdout, each tagged with the query's `id` field. Multiple queries can be
in flight simultaneously; responses may arrive in any order.

This server exposes that protocol over HTTP:
    POST /analysis          — raw KataGo query passthrough
    POST /move              — convenience: return best move + winrate
    POST /analyze_position  — convenience: return winrate only
    GET  /health            — liveness check

Concurrency model: a single background `_read_loop` task continuously reads
KataGo stdout and dispatches each response to its waiting request's
asyncio.Future, keyed by `id`. An asyncio.Lock guards only the stdin write
so multiple in-flight writes don't interleave. Endpoints await their future
and return the collected responses.

Usage:
    uvicorn runtime.analysis_server:app --host 0.0.0.0 --port 9200
    MODEL_PATH=path/to/model.bin.gz uvicorn ...
"""

import asyncio
import fcntl
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional

import yaml
from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import BaseModel


ROOT_DIR = Path(__file__).parent.parent.resolve()
CONFIG_PATH = ROOT_DIR / "configs" / "config.yaml"
DEFAULT_ANALYSIS_CONFIG = ROOT_DIR / "assets" / "bin" / "katago-trt" / "analysis_example.cfg"
STDERR_LOG = ROOT_DIR / "runtime" / "analysis_server.stderr.log"
ANALYSIS_TIMEOUT_SEC = 120
STARTUP_WAIT_SEC = 3
STDOUT_PIPE_SIZE = 1 << 20  # 1 MB — attempted stdout buffer expansion (best-effort)
# Safe to keep high now that the asyncio StreamReader buffer is enlarged (see
# `limit=4 * 1024 * 1024` in create_subprocess_exec). 40 is the original target;
# overridable via env for stress testing.
MAX_IN_FLIGHT_PER_WORKER = int(os.environ.get("MAX_IN_FLIGHT_PER_WORKER", "40"))


# =============================================================================
# Pydantic models (used by /move and /analyze_position)
# =============================================================================

class MoveRequest(BaseModel):
    moves: List[List[str]]
    rules: str
    komi: float
    color: str
    board_size: int = 19


class MoveResponse(BaseModel):
    move: str
    win_rate: Optional[float] = None


class AnalysisPositionRequest(BaseModel):
    moves: List[List[str]]
    rules: str
    komi: float
    color: str
    board_size: int = 19


class AnalysisPositionResponse(BaseModel):
    win_rate: float


# =============================================================================
# Config / command building
# =============================================================================

def load_config() -> dict:
    if not CONFIG_PATH.is_file():
        return {}
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve_path(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT_DIR / p


def build_command(model_path: Optional[str] = None) -> List[str]:
    cfg = load_config().get("katago") or {}
    binary = resolve_path(cfg.get("binary_path", "assets/bin/katago-trt/katago"))
    if model_path:
        model = resolve_path(model_path)
    else:
        env_model = os.environ.get("MODEL_PATH")
        default_model = cfg.get("model_path",
                                "assets/models/kata1-b28c512nbt-s12223529728-d5663671073.bin.gz")
        model = resolve_path(env_model or default_model)

    assert binary.is_file(), f"Missing KataGo binary: {binary}"
    assert model.is_file(), f"Missing model: {model}"
    assert DEFAULT_ANALYSIS_CONFIG.is_file(), f"Missing analysis config: {DEFAULT_ANALYSIS_CONFIG}"

    return [str(binary), "analysis", "-config", str(DEFAULT_ANALYSIS_CONFIG), "-model", str(model)]


# =============================================================================
# Async analysis engine
# =============================================================================

class AnalysisEngine:
    """Async wrapper around the KataGo analysis subprocess.

    Multiple `query()` calls can be in flight simultaneously; responses are
    routed back by KataGo's `id` field via a single background reader task.
    """

    def __init__(self, cmd: List[str], gpu_id: Optional[int] = None, worker_id: int = 0) -> None:
        self._cmd = cmd
        self._model_path = cmd[-1]
        self._gpu_id = gpu_id
        self._worker_id = worker_id
        self._proc: Optional[asyncio.subprocess.Process] = None
        self._reader_task: Optional[asyncio.Task] = None
        self._stderr_task: Optional[asyncio.Task] = None
        self._write_lock = asyncio.Lock()

        # Per-request state, keyed by KataGo query id
        self._pending: Dict[str, asyncio.Future] = {}
        self._responses: Dict[str, List[str]] = {}
        self._expected: Dict[str, int] = {}
        self._done_count: Dict[str, int] = {}

        # Counter for routing decisions (number of in-flight requests on this worker)
        self.in_flight: int = 0

        # For internally-generated queries (/move, /analyze_position)
        self._internal_id = 0

    # ---------------- lifecycle ----------------

    async def start(self) -> None:
        env = os.environ.copy()
        if self._gpu_id is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(self._gpu_id)
        print(f"[worker {self._worker_id}] Starting KataGo on GPU "
              f"{self._gpu_id if self._gpu_id is not None else 'default'}", flush=True)
        # limit=4 MB raises the StreamReader internal buffer threshold. With the
        # default 64 KB, a burst of 16+ KataGo responses (~8 KB each) overflows
        # the buffer → asyncio calls pause_reading() → uvloop stops draining
        # the socket → kernel socket buffer fills → KataGo's write() blocks →
        # analysis threads back up on the stdout mutex → futex deadlock.
        # 4 MB easily absorbs even a 40-response burst.
        self._proc = await asyncio.create_subprocess_exec(
            *self._cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            limit=4 * 1024 * 1024,
            env=env,
        )
        # Increase stdout I/O buffer to 1 MB to prevent KataGo stdout-write
        # deadlock when many large responses (4–8 KB each) are in flight.
        try:
            self._expand_stdout_buf(self._proc, STDOUT_PIPE_SIZE)
            print(f"[worker {self._worker_id}] stdout buffer expanded to {STDOUT_PIPE_SIZE // 1024} KB", flush=True)
        except Exception as e:
            print(f"[worker {self._worker_id}] Warning: could not expand stdout buffer: {e}", flush=True)
        self._reader_task = asyncio.create_task(self._read_loop(), name="katago_reader")
        self._stderr_task = asyncio.create_task(self._stderr_loop(), name="katago_stderr")

        # Give KataGo time to load the model.
        await asyncio.sleep(STARTUP_WAIT_SEC)
        if self._proc.returncode is not None:
            raise RuntimeError(f"KataGo exited at startup with code {self._proc.returncode}")
        print("KataGo analysis engine ready", flush=True)

    async def shutdown(self) -> None:
        for task in (self._reader_task, self._stderr_task):
            if task and not task.done():
                task.cancel()
        if self._proc and self._proc.returncode is None:
            try:
                self._proc.terminate()
                await asyncio.wait_for(self._proc.wait(), timeout=5)
            except (asyncio.TimeoutError, ProcessLookupError):
                if self._proc.returncode is None:
                    self._proc.kill()

    def is_alive(self) -> bool:
        return self._proc is not None and self._proc.returncode is None

    @staticmethod
    def _expand_stdout_buf(proc: asyncio.subprocess.Process, size: int) -> None:
        """Increase the stdout I/O buffer of a KataGo subprocess (Linux only).

        Tries three strategies in order:
        1. fileno() on the asyncio transport (uvloop's ReadUnixTransport exposes this)
        2. _pipe.fileno() on the transport (CPython asyncio ReadPipeTransport)
        3. /proc filesystem: works for pipes (same inode at both ends)
        """
        import socket as _socket

        # Strategy 1 & 2: get fd from the asyncio/uvloop transport
        our_fd = None
        try:
            pipe_transport = proc._transport.get_pipe_transport(1)  # type: ignore[attr-defined]
            if hasattr(pipe_transport, 'fileno'):
                our_fd = pipe_transport.fileno()  # uvloop path
            elif hasattr(pipe_transport, '_pipe'):
                our_fd = pipe_transport._pipe.fileno()  # CPython asyncio path
        except Exception:
            pass

        # Strategy 3: /proc scan (reliable for pipes, fails for socket pairs)
        if our_fd is None:
            stdout_link = os.readlink(f"/proc/{proc.pid}/fd/1")
            m = re.match(r'(pipe|socket):\[(\d+)\]', stdout_link)
            if m:
                target = f"{m.group(1)}:[{m.group(2)}]"
                for name in os.listdir("/proc/self/fd"):
                    try:
                        fd = int(name)
                        if os.readlink(f"/proc/self/fd/{fd}") == target:
                            our_fd = fd
                            break
                    except (ValueError, OSError):
                        continue

        if our_fd is None:
            raise RuntimeError("Could not determine stdout fd")

        # Apply the buffer expansion based on fd type
        stdout_link = os.readlink(f"/proc/{proc.pid}/fd/1")
        if stdout_link.startswith("pipe:"):
            F_SETPIPE_SZ = 1031  # Linux constant, not always in fcntl module
            fcntl.fcntl(our_fd, F_SETPIPE_SZ, size)
        else:
            # Socket (uvloop socketpairs): set SO_RCVBUF on our read end.
            # Linux doubles the value internally; requesting 1 MB → ~2 MB actual.
            duped = os.dup(our_fd)
            sock = _socket.socket(fileno=duped)
            try:
                sock.setsockopt(_socket.SOL_SOCKET, _socket.SO_RCVBUF, size)
            finally:
                sock.close()  # closes duped; original our_fd is unaffected

    # ---------------- background tasks ----------------

    async def _read_loop(self) -> None:
        """Read KataGo stdout and dispatch responses to pending futures by id."""
        assert self._proc and self._proc.stdout
        while True:
            line = await self._proc.stdout.readline()
            if not line:  # EOF — KataGo died
                self._fail_all_pending(RuntimeError("KataGo stdout closed unexpectedly"))
                return

            stripped = line.decode("utf-8", errors="replace").strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError:
                continue  # KataGo only emits JSON on stdout in analysis mode; skip oddities

            req_id = payload.get("id")
            if req_id is None or req_id not in self._pending:
                continue  # untagged, or request already timed out / cancelled

            self._responses[req_id].append(stripped)
            fut = self._pending[req_id]
            if fut.done():
                continue

            if payload.get("error"):
                fut.set_result(self._responses[req_id])
            elif payload.get("warning"):
                pass  # warnings don't count toward completion
            elif payload.get("isDuringSearch") is False:
                self._done_count[req_id] += 1
                if self._done_count[req_id] >= self._expected[req_id]:
                    fut.set_result(self._responses[req_id])

    async def _stderr_loop(self) -> None:
        """Drain KataGo stderr to a log file."""
        assert self._proc and self._proc.stderr
        STDERR_LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(STDERR_LOG, "ab") as f:
            while True:
                line = await self._proc.stderr.readline()
                if not line:
                    return
                f.write(line)
                f.flush()

    def _fail_all_pending(self, exc: BaseException) -> None:
        for fut in self._pending.values():
            if not fut.done():
                fut.set_exception(exc)

    # ---------------- public API ----------------

    async def query(
        self,
        line: str,
        req_id: str,
        expected: int = 1,
        timeout: float = ANALYSIS_TIMEOUT_SEC,
    ) -> List[str]:
        """Send one query to KataGo and await its completion.

        Args:
            line:     single-line JSON query (no embedded newlines)
            req_id:   the `id` field inside the query (must be unique among in-flight)
            expected: number of `isDuringSearch:false` responses to wait for
                      (equals len(analyzeTurns) in the query, or 1 if absent)
            timeout:  seconds to wait before raising asyncio.TimeoutError

        Returns:
            List of raw response lines from KataGo (typically one JSON line per
            analyzeTurn, plus any warnings).
        """
        if not self.is_alive():
            raise RuntimeError("KataGo process is not running")
        if req_id in self._pending:
            raise ValueError(f"Duplicate in-flight request id: {req_id}")

        loop = asyncio.get_event_loop()
        fut: asyncio.Future = loop.create_future()
        self._pending[req_id] = fut
        self._expected[req_id] = expected
        self._responses[req_id] = []
        self._done_count[req_id] = 0
        self.in_flight += 1

        try:
            async with self._write_lock:
                assert self._proc and self._proc.stdin
                self._proc.stdin.write((line + "\n").encode())
                await self._proc.stdin.drain()
            return await asyncio.wait_for(fut, timeout=timeout)
        finally:
            self._pending.pop(req_id, None)
            self._expected.pop(req_id, None)
            self._responses.pop(req_id, None)
            self._done_count.pop(req_id, None)
            self.in_flight -= 1

    async def get_move(
        self,
        moves: List[List[str]],
        rules: str,
        komi: float,
        color: str,
        board_size: int = 19,
    ) -> tuple[str, Optional[float]]:
        """Return (best move in GTP form, winrate) for the given position."""
        self._internal_id += 1
        req_id = f"internal_{self._internal_id}"

        katago_moves = []
        for c, m in moves:
            mv = m.upper()
            katago_moves.append([c.upper(), "pass" if mv == "PASS" else mv])

        request = {
            "id": req_id,
            "moves": katago_moves,
            "rules": rules,
            "komi": komi,
            "boardXSize": board_size,
            "boardYSize": board_size,
            "analyzeTurns": [len(katago_moves)],
        }
        responses = await self.query(json.dumps(request), req_id, expected=1)
        if not responses:
            return "", None
        try:
            result = json.loads(responses[-1])
            move_infos = result.get("moveInfos") or []
            win_rate = result.get("rootInfo", {}).get("winrate")
            if not move_infos:
                return "pass", win_rate
            return move_infos[0].get("move", "pass").upper(), win_rate
        except (json.JSONDecodeError, KeyError, IndexError) as e:
            print(f"Error parsing KataGo response: {e}", flush=True)
            return "", None

    async def get_analysis(self, *args, **kwargs) -> float:
        _, win_rate = await self.get_move(*args, **kwargs)
        return win_rate if win_rate is not None else 0.5


# =============================================================================
# Multi-GPU pool — round-robin / least-loaded across N AnalysisEngine workers
# =============================================================================

class AnalysisEnginePool:
    """Pool of AnalysisEngine workers, each pinned to a different GPU.

    Routes incoming queries to the worker with the fewest in-flight requests
    (least-loaded), which naturally balances under varying request sizes.
    """

    def __init__(self, cmd: List[str], num_workers: int) -> None:
        assert num_workers >= 1
        self._workers: List[AnalysisEngine] = [
            AnalysisEngine(cmd, gpu_id=i, worker_id=i) for i in range(num_workers)
        ]
        self._internal_id = 0

    async def start(self) -> None:
        # Start workers in parallel; their TensorRT engine builds run concurrently.
        await asyncio.gather(*(w.start() for w in self._workers))

    async def shutdown(self) -> None:
        await asyncio.gather(*(w.shutdown() for w in self._workers), return_exceptions=True)

    def is_alive(self) -> bool:
        return all(w.is_alive() for w in self._workers)

    @property
    def model_path(self) -> str:
        return self._workers[0]._model_path

    def _pick_worker(self) -> "AnalysisEngine":
        # Least-loaded by in-flight count; ties broken by worker order.
        return min(self._workers, key=lambda w: w.in_flight)

    async def query(self, line: str, req_id: str, expected: int = 1,
                    timeout: float = ANALYSIS_TIMEOUT_SEC) -> List[str]:
        worker = self._pick_worker()
        if worker.in_flight >= MAX_IN_FLIGHT_PER_WORKER:
            raise OverflowError(
                f"All workers at capacity (min in_flight={worker.in_flight} "
                f">= {MAX_IN_FLIGHT_PER_WORKER})"
            )
        return await worker.query(line, req_id, expected=expected, timeout=timeout)

    async def get_move(self, *args, **kwargs):
        # /move and /analyze_position generate their own request ids; route the
        # whole call to one worker so it stays self-consistent.
        worker = self._pick_worker()
        return await worker.get_move(*args, **kwargs)

    async def get_analysis(self, *args, **kwargs) -> float:
        worker = self._pick_worker()
        return await worker.get_analysis(*args, **kwargs)

    async def monitor_loop(self, interval: float = 10.0, stall_threshold: int = 20) -> None:
        """Periodically probe workers whose in_flight exceeds stall_threshold.

        When a worker looks stalled, reads /proc/<pid>/wchan (what kernel call
        it's sleeping in) and /proc/<pid>/status (thread/memory snapshot).
        This pinpoints whether the deadlock is a pipe/socket write-block,
        a mutex wait, a CUDA sync, or something else.
        """
        # Track how many consecutive polls each worker has been above threshold.
        consecutive: Dict[int, int] = {w._worker_id: 0 for w in self._workers}

        while True:
            await asyncio.sleep(interval)
            for w in self._workers:
                if w.in_flight < stall_threshold:
                    consecutive[w._worker_id] = 0
                    continue

                consecutive[w._worker_id] += 1
                pid = w._proc.pid if w._proc else None

                if pid is None:
                    print(
                        f"[monitor] worker {w._worker_id} in_flight={w.in_flight} "
                        f"(consecutive={consecutive[w._worker_id]}) — process is None",
                        flush=True,
                    )
                    continue

                # Read wchan (the kernel wait channel — what syscall it's blocked in).
                try:
                    wchan = Path(f"/proc/{pid}/wchan").read_text().strip()
                except OSError:
                    wchan = "unreadable"

                # Read the first few lines of /proc/<pid>/status for thread/vm info.
                try:
                    status_lines = Path(f"/proc/{pid}/status").read_text().splitlines()
                    keep = {"State", "Threads", "VmRSS", "voluntary_ctxt_switches",
                            "nonvoluntary_ctxt_switches"}
                    status_snippet = {
                        line.split(":")[0].strip(): line.split(":", 1)[1].strip()
                        for line in status_lines
                        if line.split(":")[0].strip() in keep
                    }
                except OSError:
                    status_snippet = {}

                # Aggregate per-thread wchans (shows how many threads are blocked where).
                thread_wchans: Dict[str, int] = {}
                try:
                    for tid in os.listdir(f"/proc/{pid}/task"):
                        tw = Path(f"/proc/{pid}/task/{tid}/wchan").read_text().strip()
                        thread_wchans[tw] = thread_wchans.get(tw, 0) + 1
                except OSError:
                    pass

                pending_count = len(w._pending)

                print(
                    f"[monitor] worker {w._worker_id} STALLED "
                    f"in_flight={w.in_flight} pending={pending_count} "
                    f"consecutive={consecutive[w._worker_id]} "
                    f"pid={pid} wchan={wchan!r} "
                    f"thread_wchans={dict(sorted(thread_wchans.items(), key=lambda x: -x[1]))} "
                    f"status={status_snippet}",
                    flush=True,
                )


# =============================================================================
# FastAPI app
# =============================================================================

app = FastAPI(title="KataGo Analysis Server")
engine: Optional["AnalysisEnginePool"] = None


@app.on_event("startup")
async def _startup() -> None:
    global engine
    num_workers = int(os.environ.get("NUM_WORKERS", "1"))
    print(f"Launching {num_workers} worker(s)", flush=True)
    engine = AnalysisEnginePool(build_command(), num_workers=num_workers)
    await engine.start()
    asyncio.create_task(engine.monitor_loop(), name="worker_monitor")


@app.on_event("shutdown")
async def _shutdown() -> None:
    if engine is not None:
        await engine.shutdown()


@app.post("/analysis")
async def analysis(request: Request) -> Response:
    """Raw KataGo analysis API. Body is a single-line JSON query."""
    if engine is None or not engine.is_alive():
        raise HTTPException(status_code=503, detail="Engine not ready")

    body = await request.body()
    line = body.decode("utf-8").rstrip("\r\n")
    if "\n" in line or "\r" in line:
        raise HTTPException(status_code=400, detail="JSON must be a single line")

    try:
        payload = json.loads(line)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {exc}") from exc

    req_id = payload.get("id")
    if not req_id:
        raise HTTPException(status_code=400, detail="Missing required field: id")

    analyze_turns = payload.get("analyzeTurns")
    if analyze_turns is None:
        expected = 1
    elif isinstance(analyze_turns, list):
        expected = len(analyze_turns)
    else:
        raise HTTPException(status_code=400, detail="analyzeTurns must be a list if provided")

    try:
        responses = await engine.query(line, req_id, expected=expected)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Timed out waiting for KataGo response")
    except OverflowError as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return Response(content="\n".join(responses), media_type="text/plain")


@app.post("/move", response_model=MoveResponse)
async def generate_move(request: MoveRequest) -> MoveResponse:
    if engine is None or not engine.is_alive():
        raise HTTPException(status_code=503, detail="Engine not ready")
    move, win_rate = await engine.get_move(
        moves=request.moves,
        rules=request.rules,
        komi=request.komi,
        color=request.color,
        board_size=request.board_size,
    )
    if not move:
        raise HTTPException(status_code=500, detail="Failed to generate move")
    return MoveResponse(move=move, win_rate=win_rate)


@app.post("/analyze_position", response_model=AnalysisPositionResponse)
async def analyze_position(request: AnalysisPositionRequest) -> AnalysisPositionResponse:
    if engine is None or not engine.is_alive():
        raise HTTPException(status_code=503, detail="Engine not ready")
    win_rate = await engine.get_analysis(
        moves=request.moves,
        rules=request.rules,
        komi=request.komi,
        color=request.color,
        board_size=request.board_size,
    )
    return AnalysisPositionResponse(win_rate=win_rate)


@app.get("/health")
async def health_check():
    if engine is None or not engine.is_alive():
        raise HTTPException(status_code=503, detail="Engine not ready")
    return {"status": "healthy", "model": engine.model_path,
            "num_workers": len(engine._workers),
            "in_flight": [w.in_flight for w in engine._workers]}


if __name__ == "__main__":
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="KataGo Analysis Server")
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()

    cfg = load_config()
    port = args.port or cfg.get("server", {}).get("port", 9200)
    uvicorn.run(app, host=args.host, port=port, log_level="warning")
