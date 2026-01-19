import json
import os
import selectors
import signal
import subprocess
import threading
from typing import List, Optional

import yaml
from fastapi import FastAPI, HTTPException, Request, Response


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CONFIG_PATH = os.path.join(ROOT_DIR, "runtime", "config.yaml")
DEFAULT_ANALYSIS_CONFIG = os.path.join(ROOT_DIR, "cpp", "configs", "analysis_example.cfg")
STDERR_LOG = os.path.join(ROOT_DIR, "runtime", "analysis_server.stderr.log")
ANALYSIS_TIMEOUT_SEC = 120


def load_config() -> dict:
    if not os.path.isfile(CONFIG_PATH):
        raise RuntimeError(f"Missing config: {CONFIG_PATH}")
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_command(cfg: dict) -> List[str]:
    katago_cfg = cfg.get("katago") or {}
    backend = katago_cfg.get("backend")
    if backend == "cuda":
        binary = katago_cfg.get("cuda_binary_path")
    elif backend == "trt":
        binary = katago_cfg.get("trt_binary_path")
    else:
        raise RuntimeError("katago.backend must be 'cuda' or 'trt'")

    model_path = katago_cfg.get("model_path")
    if not model_path:
        raise RuntimeError("katago.model_path is required")

    config_path = katago_cfg.get("config_path") or DEFAULT_ANALYSIS_CONFIG
    if not os.path.isfile(config_path):
        raise RuntimeError(f"Missing analysis config: {config_path}")

    if not os.path.isfile(binary):
        raise RuntimeError(f"Missing KataGo binary: {binary}")
    if not os.path.isfile(model_path):
        raise RuntimeError(f"Missing model: {model_path}")

    return [binary, "analysis", "-config", config_path, "-model", model_path]


class AnalysisEngine:
    def __init__(self, cmd: List[str]) -> None:
        self._cmd = cmd
        self._lock = threading.Lock()
        self._selector = selectors.DefaultSelector()
        self._proc = self._start_process()
        self._start_stderr_thread()

    def _start_process(self) -> subprocess.Popen:
        proc = subprocess.Popen(
            self._cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            preexec_fn=os.setsid,
        )
        if proc.stdout is None:
            raise RuntimeError("Failed to open stdout for KataGo process")
        self._selector.register(proc.stdout, selectors.EVENT_READ)
        return proc

    def _start_stderr_thread(self) -> None:
        if self._proc.stderr is None:
            return
        os.makedirs(os.path.dirname(STDERR_LOG), exist_ok=True)
        stderr = self._proc.stderr

        def _drain() -> None:
            with open(STDERR_LOG, "a", encoding="utf-8") as f:
                for line in stderr:
                    f.write(line)

        thread = threading.Thread(target=_drain, daemon=True)
        thread.start()

    def _readline(self, timeout: float) -> Optional[str]:
        events = self._selector.select(timeout)
        if not events:
            return None
        if self._proc.stdout is None:
            return None
        return self._proc.stdout.readline()

    def query(self, line: str, expected: int) -> List[str]:
        if self._proc.poll() is not None:
            raise RuntimeError("KataGo process is not running")
        if self._proc.stdin is None:
            raise RuntimeError("KataGo stdin is not available")

        with self._lock:
            self._proc.stdin.write(line + "\n")
            self._proc.stdin.flush()

            responses: List[str] = []
            matched = 0
            while matched < expected:
                next_line = self._readline(ANALYSIS_TIMEOUT_SEC)
                if next_line is None:
                    raise TimeoutError("Timed out waiting for KataGo response")
                stripped = next_line.strip()
                if not stripped:
                    continue
                responses.append(stripped)
                try:
                    payload = json.loads(stripped)
                except json.JSONDecodeError:
                    continue
                if payload.get("error"):
                    break
                if payload.get("warning"):
                    continue
                if payload.get("id") is not None and payload.get("isDuringSearch") is False:
                    matched += 1
            return responses

    def shutdown(self) -> None:
        if self._proc.poll() is None:
            try:
                os.killpg(self._proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                return


app = FastAPI()
engine: Optional[AnalysisEngine] = None


@app.on_event("startup")
def _startup() -> None:
    global engine
    cfg = load_config()
    cmd = build_command(cfg)
    engine = AnalysisEngine(cmd)


@app.on_event("shutdown")
def _shutdown() -> None:
    if engine is not None:
        engine.shutdown()


@app.post("/analysis")
async def analysis(request: Request) -> Response:
    if engine is None:
        raise HTTPException(status_code=500, detail="Analysis engine not ready")

    body = await request.body()
    raw = body.decode("utf-8")
    line = raw.rstrip("\r\n")
    if "\n" in line or "\r" in line:
        raise HTTPException(status_code=400, detail="JSON must be a single line")

    try:
        payload = json.loads(line)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {exc}") from exc

    if "id" not in payload:
        raise HTTPException(status_code=400, detail="Missing required field: id")
    analyze_turns = payload.get("analyzeTurns")
    if analyze_turns is None:
        expected = 1
    elif isinstance(analyze_turns, list):
        expected = len(analyze_turns)
    else:
        raise HTTPException(status_code=400, detail="analyzeTurns must be a list if provided")

    try:
        responses = engine.query(line, expected)
    except TimeoutError as exc:
        raise HTTPException(status_code=504, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return Response(content="\n".join(responses), media_type="text/plain")
