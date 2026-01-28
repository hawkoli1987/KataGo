"""KataGo player using async HTTP to the analysis server.

Two modes of operation:
1. Remote mode: Connect to an already-running server (endpoint parameter)
2. Local mode: Spawn analysis server subprocess (requires TensorRT)

Remote mode is preferred for testing and production when KataGo servers
are managed separately (via PBS jobs).
"""

import asyncio
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional

import aiohttp

from eval.players.base import Player


# Paths relative to repo root
REPO_ROOT = Path(__file__).parent.parent.parent.resolve()
DEFAULT_BINARY_PATH = REPO_ROOT / "assets" / "bin" / "katago-trt" / "katago"
DEFAULT_CONFIG_PATH = REPO_ROOT / "assets" / "bin" / "katago-trt" / "default_gtp.cfg"


class KataGoPlayer(Player):
    """KataGo player using HTTP to communicate with analysis server.
    
    Can either:
    - Connect to an existing remote server (endpoint mode)
    - Spawn a local server subprocess (local mode, requires TensorRT)
    """
    
    def __init__(
        self,
        endpoint: Optional[str] = None,
        model_path: Optional[str] = None,
        gpu_id: int = 0,
        port: int = 8100,
        name: Optional[str] = None
    ):
        """Initialize KataGo player.
        
        Args:
            endpoint: URL of already-running KataGo server (e.g., "http://localhost:9200")
                      If provided, uses remote mode (no subprocess spawned)
            model_path: Path to KataGo model weights (for local mode only)
            gpu_id: GPU device ID to use (for local mode only)
            port: Port for the analysis server (for local mode only)
            name: Player name
        """
        self._remote_mode = endpoint is not None
        
        if self._remote_mode:
            # Remote mode: connect to existing server
            assert endpoint is not None  # Type narrowing for type checker
            self._base_url = endpoint.rstrip("/")
            player_name = name or "katago_remote"
            self.model_path: Optional[str] = None
            self.gpu_id: Optional[int] = None
            self.port: Optional[int] = None
        else:
            # Local mode: spawn subprocess
            assert model_path is not None, "model_path required for local mode"
            model_p = Path(model_path)
            if not model_p.is_absolute():
                model_p = REPO_ROOT / model_path
            
            assert model_p.exists(), f"Model not found: {model_p}"
            
            player_name = name or model_p.stem
            self.model_path = str(model_p.resolve())
            self.gpu_id = gpu_id
            self.port = port
            self._base_url = f"http://127.0.0.1:{port}"
        
        super().__init__(name=player_name)
        
        self._process: Optional[subprocess.Popen] = None
        self._session: Optional[aiohttp.ClientSession] = None
    
    @property
    def requires_logging(self) -> bool:
        """KataGo players don't need LLM-style logging."""
        return False
    
    async def start(self):
        """Start the player (and optionally spawn server subprocess)."""
        if self._session is not None:
            return
        
        if self._remote_mode:
            # Remote mode: just create HTTP session and verify server is reachable
            self._session = aiohttp.ClientSession()
            print(f"Connecting to remote KataGo server: {self._base_url}")
            await self._wait_for_server(timeout=30)
            print(f"Connected to KataGo server: {self._base_url}")
        else:
            # Local mode: spawn subprocess
            if self._process is not None:
                self._session = aiohttp.ClientSession()
                return
            
            # Build command - run the analysis server script directly
            server_script = REPO_ROOT / "runtime" / "analysis_server.py"
            
            cmd = [
                sys.executable, str(server_script),
                "--port", str(self.port)
            ]
            
            # Set environment
            env = os.environ.copy()
            assert self.gpu_id is not None and self.model_path is not None  # Local mode
            env["CUDA_VISIBLE_DEVICES"] = str(self.gpu_id)
            env["MODEL_PATH"] = self.model_path
            
            print(f"Starting KataGo server on GPU {self.gpu_id}, port {self.port}: {self.name}")
            
            self._process = subprocess.Popen(
                cmd,
                env=env,
                cwd=str(REPO_ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True
            )
            
            # Wait for server to be ready
            await self._wait_for_server(timeout=60)
            
            # Create HTTP session
            self._session = aiohttp.ClientSession()
            
            print(f"KataGo server ready: {self._base_url}")
    
    async def _wait_for_server(self, timeout: float = 60):
        """Wait for the server to become available."""
        start_time = time.time()
        last_error = None
        
        while time.time() - start_time < timeout:
            # Check if local process crashed
            if not self._remote_mode and self._process is not None and self._process.poll() is not None:
                stderr = self._process.stderr.read().decode() if self._process.stderr else ""
                raise RuntimeError(f"KataGo server failed to start: {stderr[:1000]}")
            
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(f"{self._base_url}/health", timeout=aiohttp.ClientTimeout(total=2)) as resp:
                        if resp.status == 200:
                            return
            except Exception as e:
                last_error = e
            
            await asyncio.sleep(1)
        
        raise RuntimeError(f"KataGo server at {self._base_url} not reachable within {timeout}s: {last_error}")
    
    async def stop(self):
        """Stop the player (and subprocess if in local mode)."""
        if self._session is not None:
            await self._session.close()
            self._session = None
        
        if self._remote_mode:
            # Remote mode: just disconnect, don't stop server
            print(f"Disconnected from KataGo server: {self.name}")
            return
        
        if self._process is None:
            return
        
        try:
            # Send SIGTERM to process group
            os.killpg(os.getpgid(self._process.pid), signal.SIGTERM)
            
            # Wait for graceful shutdown
            for _ in range(10):
                if self._process.poll() is not None:
                    break
                await asyncio.sleep(0.5)
            
            # Force kill if still running
            if self._process.poll() is None:
                os.killpg(os.getpgid(self._process.pid), signal.SIGKILL)
                self._process.wait(timeout=2)
        except Exception as e:
            print(f"Warning: Error stopping KataGo server: {e}")
        finally:
            self._process = None
        
        print(f"KataGo server stopped: {self.name}")
    
    async def get_move(
        self,
        move_history: List[List[str]],
        rules: str,
        komi: float,
        color: str,
        win_rate: Optional[float] = None
    ) -> str:
        """Get move from KataGo server via HTTP."""
        assert self._session is not None, "Player not started"
        
        payload = {
            "moves": move_history,
            "rules": rules,
            "komi": komi,
            "color": color,
            "board_size": 19
        }
        
        try:
            async with self._session.post(
                f"{self._base_url}/move",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=120)
            ) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    print(f"KataGo server error: {text}")
                    return ""
                
                data = await resp.json()
                return data.get("move", "")
        
        except Exception as e:
            print(f"Error getting move from KataGo: {e}")
            return ""
    
    async def get_win_rate(
        self,
        move_history: List[List[str]],
        rules: str,
        komi: float,
        color: str
    ) -> Optional[float]:
        """Get win rate from KataGo server."""
        assert self._session is not None, "Player not started"
        
        payload = {
            "moves": move_history,
            "rules": rules,
            "komi": komi,
            "color": color,
            "board_size": 19
        }
        
        try:
            async with self._session.post(
                f"{self._base_url}/analyze_position",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as resp:
                if resp.status != 200:
                    return None
                
                data = await resp.json()
                return data.get("win_rate")
        
        except Exception:
            return None
