"""KataGo Analysis Server.

Provides HTTP endpoints for Go position analysis and move generation.
Supports both analysis mode (position evaluation) and game playing.

Usage:
    # Start server (uses config.yaml for default model)
    uvicorn runtime.analysis_server:app --host 0.0.0.0 --port 9200

    # Start with custom model
    MODEL_PATH=assets/models/level_01_kata1-b6c96-s175395328-d26788732.txt.gz \
        uvicorn runtime.analysis_server:app --host 0.0.0.0 --port 8100

Endpoints:
    POST /analysis - KataGo analysis API (raw JSON query)
    POST /move - Generate best move for a position (game playing)
    GET /health - Health check
"""

import json
import os
import re
import selectors
import signal
import subprocess
import threading
import time
from pathlib import Path
from typing import List, Optional

import yaml
from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import BaseModel


ROOT_DIR = Path(__file__).parent.parent.resolve()
CONFIG_PATH = ROOT_DIR / "runtime" / "config.yaml"
DEFAULT_ANALYSIS_CONFIG = ROOT_DIR / "assets" / "bin" / "katago-trt" / "analysis_example.cfg"
STDERR_LOG = ROOT_DIR / "runtime" / "analysis_server.stderr.log"
ANALYSIS_TIMEOUT_SEC = 120


class MoveRequest(BaseModel):
    """Request for generating a move."""
    moves: List[List[str]]  # [[color, move], ...] e.g. [["B", "D4"], ["W", "Q16"]]
    rules: str  # KataGo rule string
    komi: float
    color: str  # "B" or "W" - which color to play
    board_size: int = 19


class MoveResponse(BaseModel):
    """Response containing the generated move."""
    move: str  # GTP format move (e.g., "D4", "pass", "resign")
    win_rate: Optional[float] = None


class AnalysisPositionRequest(BaseModel):
    """Request for position analysis (win rate)."""
    moves: List[List[str]]
    rules: str
    komi: float
    color: str  # Perspective for win rate
    board_size: int = 19


class AnalysisPositionResponse(BaseModel):
    """Response containing position analysis."""
    win_rate: float


def load_config() -> dict:
    if not CONFIG_PATH.is_file():
        return {}
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve_path(path: str) -> Path:
    """Resolve a path relative to repo root."""
    p = Path(path)
    if p.is_absolute():
        return p
    return ROOT_DIR / p


def build_command(model_path: Optional[str] = None) -> List[str]:
    """Build KataGo command line."""
    cfg = load_config()
    katago_cfg = cfg.get("katago") or {}
    
    # Get binary path
    binary_path = katago_cfg.get("binary_path", "assets/bin/katago-trt/katago")
    binary = resolve_path(binary_path)
    
    # Get model path (can be overridden by environment variable)
    if model_path:
        model = resolve_path(model_path)
    else:
        env_model = os.environ.get("MODEL_PATH")
        if env_model:
            model = resolve_path(env_model)
        else:
            default_model = katago_cfg.get("model_path", "assets/models/kata1-b28c512nbt-s12223529728-d5663671073.bin.gz")
            model = resolve_path(default_model)
    
    # Get config path
    config_path = DEFAULT_ANALYSIS_CONFIG
    
    assert binary.is_file(), f"Missing KataGo binary: {binary}"
    assert model.is_file(), f"Missing model: {model}"
    assert config_path.is_file(), f"Missing analysis config: {config_path}"
    
    return [str(binary), "analysis", "-config", str(config_path), "-model", str(model)]


class AnalysisEngine:
    """KataGo analysis engine subprocess manager."""
    
    def __init__(self, cmd: List[str]) -> None:
        self._cmd = cmd
        self._lock = threading.Lock()
        self._selector = selectors.DefaultSelector()
        self._request_id = 0
        self._proc = self._start_process()
        self._start_stderr_thread()
        self._model_path = cmd[-1]  # Last arg is model path

    def _start_process(self) -> subprocess.Popen:
        print(f"Starting KataGo: {' '.join(self._cmd)}")
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
        
        # Wait for startup
        time.sleep(3)
        if proc.poll() is not None:
            stderr = proc.stderr.read() if proc.stderr else ""
            raise RuntimeError(f"KataGo failed to start: {stderr}")
        
        print("KataGo analysis engine ready")
        return proc

    def _start_stderr_thread(self) -> None:
        if self._proc.stderr is None:
            return
        STDERR_LOG.parent.mkdir(parents=True, exist_ok=True)
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
        """Send raw analysis query."""
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

    def get_move(self, moves: List[List[str]], rules: str, komi: float,
                 color: str, board_size: int = 19) -> tuple[str, Optional[float]]:
        """Get best move for a position using analysis API."""
        self._request_id += 1
        
        # Build analysis request
        # Convert moves to KataGo format
        katago_moves = []
        for move_color, move in moves:
            # KataGo uses lowercase color
            c = move_color.upper()
            m = move.upper()
            if m == "PASS":
                katago_moves.append([c, "pass"])
            else:
                katago_moves.append([c, m])
        
        request = {
            "id": str(self._request_id),
            "moves": katago_moves,
            "rules": rules,
            "komi": komi,
            "boardXSize": board_size,
            "boardYSize": board_size,
            "analyzeTurns": [len(katago_moves)],  # Analyze current position
        }
        
        responses = self.query(json.dumps(request), 1)
        
        if not responses:
            return "", None
        
        try:
            result = json.loads(responses[-1])
            
            # Get best move from moveInfos
            move_infos = result.get("moveInfos", [])
            if not move_infos:
                return "pass", result.get("rootInfo", {}).get("winrate")
            
            best_move = move_infos[0]
            move = best_move.get("move", "pass")
            
            # Get win rate from rootInfo
            root_info = result.get("rootInfo", {})
            win_rate = root_info.get("winrate")
            
            return move.upper(), win_rate
        
        except (json.JSONDecodeError, KeyError, IndexError) as e:
            print(f"Error parsing KataGo response: {e}")
            return "", None

    def get_analysis(self, moves: List[List[str]], rules: str, komi: float,
                     color: str, board_size: int = 19) -> float:
        """Get win rate for a position."""
        _, win_rate = self.get_move(moves, rules, komi, color, board_size)
        return win_rate if win_rate is not None else 0.5

    def shutdown(self) -> None:
        if self._proc.poll() is None:
            try:
                os.killpg(self._proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                return


# Global engine instance
app = FastAPI(title="KataGo Analysis Server")
engine: Optional[AnalysisEngine] = None


@app.on_event("startup")
def _startup() -> None:
    global engine
    cmd = build_command()
    engine = AnalysisEngine(cmd)


@app.on_event("shutdown")
def _shutdown() -> None:
    if engine is not None:
        engine.shutdown()


@app.post("/analysis")
async def analysis(request: Request) -> Response:
    """Raw KataGo analysis API endpoint."""
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


@app.post("/move", response_model=MoveResponse)
async def generate_move(request: MoveRequest) -> MoveResponse:
    """Generate the best move for a position (for game playing)."""
    if engine is None:
        raise HTTPException(status_code=500, detail="Analysis engine not ready")
    
    move, win_rate = engine.get_move(
        moves=request.moves,
        rules=request.rules,
        komi=request.komi,
        color=request.color,
        board_size=request.board_size
    )
    
    if not move:
        raise HTTPException(status_code=500, detail="Failed to generate move")
    
    return MoveResponse(move=move, win_rate=win_rate)


@app.post("/analyze_position", response_model=AnalysisPositionResponse)
async def analyze_position(request: AnalysisPositionRequest) -> AnalysisPositionResponse:
    """Get win rate for a position."""
    if engine is None:
        raise HTTPException(status_code=500, detail="Analysis engine not ready")
    
    win_rate = engine.get_analysis(
        moves=request.moves,
        rules=request.rules,
        komi=request.komi,
        color=request.color,
        board_size=request.board_size
    )
    
    return AnalysisPositionResponse(win_rate=win_rate)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    if engine is None:
        raise HTTPException(status_code=500, detail="Engine not initialized")
    return {"status": "healthy", "model": engine._model_path}


if __name__ == "__main__":
    import argparse
    import uvicorn
    
    parser = argparse.ArgumentParser(description="KataGo Analysis Server")
    parser.add_argument("--port", type=int, default=None, help="Server port")
    parser.add_argument("--host", default="0.0.0.0", help="Server host")
    args = parser.parse_args()
    
    cfg = load_config()
    port = args.port or cfg.get("port", 9200)
    uvicorn.run(app, host=args.host, port=port, log_level="warning")
