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
import json
import os
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

    def __init__(self, cmd: List[str]) -> None:
        self._cmd = cmd
        self._model_path = cmd[-1]
        self._proc: Optional[asyncio.subprocess.Process] = None
        self._reader_task: Optional[asyncio.Task] = None
        self._stderr_task: Optional[asyncio.Task] = None
        self._write_lock = asyncio.Lock()

        # Per-request state, keyed by KataGo query id
        self._pending: Dict[str, asyncio.Future] = {}
        self._responses: Dict[str, List[str]] = {}
        self._expected: Dict[str, int] = {}
        self._done_count: Dict[str, int] = {}

        # For internally-generated queries (/move, /analyze_position)
        self._internal_id = 0

    # ---------------- lifecycle ----------------

    async def start(self) -> None:
        print(f"Starting KataGo: {' '.join(self._cmd)}", flush=True)
        self._proc = await asyncio.create_subprocess_exec(
            *self._cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
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
# FastAPI app
# =============================================================================

app = FastAPI(title="KataGo Analysis Server")
engine: Optional[AnalysisEngine] = None


@app.on_event("startup")
async def _startup() -> None:
    global engine
    engine = AnalysisEngine(build_command())
    await engine.start()


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
    return {"status": "healthy", "model": engine._model_path}


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
