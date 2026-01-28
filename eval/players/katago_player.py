"""KataGo neural network player for Go evaluation.

Uses KataGo GTP to generate moves, allowing evaluation of KataGo models.
"""

import re
import subprocess
import time
from pathlib import Path
from typing import Optional

from eval.players.base import Player


class KataGoPlayer(Player):
    """KataGo neural network as a candidate player.
    
    Runs KataGo in GTP mode and generates moves using genmove command.
    """
    
    def __init__(
        self,
        katago_path: str,
        model_path: str,
        config_path: str,
        name: Optional[str] = None,
        log_dir: str = "log"
    ):
        """Initialize KataGo player.
        
        Args:
            katago_path: Path to KataGo executable
            model_path: Path to KataGo model weights
            config_path: Path to KataGo GTP config file
            name: Player name (defaults to model filename)
            log_dir: Directory for KataGo logs
        """
        # Validate paths
        assert Path(katago_path).exists(), f"KataGo executable not found: {katago_path}"
        assert Path(model_path).exists(), f"Model not found: {model_path}"
        assert Path(config_path).exists(), f"Config not found: {config_path}"
        
        player_name = name or Path(model_path).stem
        super().__init__(name=player_name)
        
        self.katago_path = katago_path
        self.model_path = model_path
        self.config_path = config_path
        self.log_dir = log_dir
        
        self.process: Optional[subprocess.Popen] = None
        self._current_rules: Optional[str] = None
        self._current_komi: Optional[float] = None
    
    @property
    def requires_logging(self) -> bool:
        """KataGo players don't need LLM-style logging."""
        return False
    
    def start(self):
        """Start KataGo process."""
        if self.process is not None:
            return
        
        cmd = [
            self.katago_path, "gtp",
            "-model", self.model_path,
            "-config", self.config_path,
            "-override-config", f"logDir={self.log_dir}"
        ]
        
        print(f"Starting KataGo player: {' '.join(cmd)}")
        
        self.process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )
        
        # Wait for startup
        time.sleep(2)
        
        if self.process.poll() is not None:
            stderr = self.process.stderr.read()
            raise RuntimeError(f"KataGo failed to start: {stderr}")
    
    def stop(self):
        """Stop KataGo process."""
        if self.process is None:
            return
        
        try:
            if self.process.poll() is None:
                self._send_command("quit")
                self.process.wait(timeout=5)
        except Exception:
            try:
                self.process.kill()
                self.process.wait(timeout=2)
            except Exception:
                pass
        finally:
            self.process = None
    
    def _send_command(self, cmd: str) -> str:
        """Send a GTP command and return the response."""
        assert self.process is not None, "KataGo not started"
        
        self.process.stdin.write(cmd + "\n")
        self.process.stdin.flush()
        
        response_lines = []
        while True:
            line = self.process.stdout.readline()
            if line.strip() == "":
                if response_lines:
                    break
            else:
                response_lines.append(line.rstrip())
        
        return "\n".join(response_lines)
    
    def _setup_game(self, rules: str, komi: float):
        """Setup game state if rules/komi changed."""
        if rules != self._current_rules or komi != self._current_komi:
            self._send_command("clear_board")
            self._send_command("boardsize 19")
            self._send_command(f"komi {komi}")
            self._send_command(f"kata-set-rules {rules}")
            self._current_rules = rules
            self._current_komi = komi
    
    def _replay_moves(self, move_history: list):
        """Replay move history on the board."""
        self._send_command("clear_board")
        for color, move in move_history:
            self._send_command(f"play {color} {move}")
    
    def get_move(
        self, 
        move_history: list, 
        rules: str, 
        komi: float, 
        color: str,
        win_rate: Optional[float] = None
    ) -> str:
        """Get move from KataGo.
        
        Note: win_rate is ignored for KataGo players since they compute their own.
        """
        # Ensure process is running
        if self.process is None:
            self.start()
        
        # Setup game state
        self._setup_game(rules, komi)
        self._replay_moves(move_history)
        
        # Generate move
        response = self._send_command(f"genmove {color}")
        
        # Response format: "= D4" or "= pass" or "= resign"
        if response.startswith("="):
            return response[1:].strip().upper()
        
        return ""
    
    def play_move(self, color: str, move: str) -> str:
        """Play a move on the board (used by reference engine)."""
        assert self.process is not None, "KataGo not started"
        return self._send_command(f"play {color} {move}")
    
    def genmove(self, color: str) -> str:
        """Generate a move (simpler interface for reference engine)."""
        assert self.process is not None, "KataGo not started"
        response = self._send_command(f"genmove {color}")
        if response.startswith("="):
            return response[1:].strip().upper()
        return ""
    
    def clear_board(self):
        """Clear the board."""
        assert self.process is not None, "KataGo not started"
        return self._send_command("clear_board")
    
    def set_boardsize(self, size: int = 19):
        """Set board size."""
        assert self.process is not None, "KataGo not started"
        return self._send_command(f"boardsize {size}")
    
    def set_komi(self, komi: float):
        """Set komi."""
        assert self.process is not None, "KataGo not started"
        self._current_komi = komi
        return self._send_command(f"komi {komi}")
    
    def set_rules(self, rules: str):
        """Set rules."""
        assert self.process is not None, "KataGo not started"
        self._current_rules = rules
        return self._send_command(f"kata-set-rules {rules}")
    
    def final_score(self) -> str:
        """Get final score."""
        assert self.process is not None, "KataGo not started"
        return self._send_command("final_score")
    
    def get_win_rate(self, color: str, visits: int = 100) -> Optional[float]:
        """Get KataGo's win rate estimation for the specified color."""
        assert self.process is not None, "KataGo not started"
        
        cmd = f"kata-analyze {color} {visits} rootInfo true"
        self.process.stdin.write(cmd + "\n")
        self.process.stdin.flush()
        
        response_lines = []
        max_wait = 30
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            line = self.process.stdout.readline()
            if line:
                response_lines.append(line.strip())
                if "rootInfo" in line:
                    self.process.stdin.write("\n")
                    self.process.stdin.flush()
                    break
        
        full_response = " ".join(response_lines)
        match = re.search(r'rootInfo.*?winrate\s+([\d.]+)', full_response)
        
        if match:
            return float(match.group(1))
        
        return None
