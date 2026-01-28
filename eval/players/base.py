"""Base Player class for Go evaluation.

All player types (LLM, KataGo) implement this interface.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

# GTP coordinate system: A-T (skipping I), 1-19
GTP_COLS = "ABCDEFGHJKLMNOPQRST"  # Note: no 'I'


def validate_gtp_move(move: str, board_size: int = 19) -> bool:
    """Validate a move is in valid GTP format for the given board size."""
    move = move.strip().upper()
    
    if move == "PASS" or move == "RESIGN":
        return True
    
    if len(move) < 2 or len(move) > 3:
        return False
    
    col = move[0]
    if col not in GTP_COLS[:board_size]:
        return False
    
    try:
        row = int(move[1:])
        if row < 1 or row > board_size:
            return False
    except ValueError:
        return False
    
    return True


class Player(ABC):
    """Abstract base class for Go players.
    
    All player types must implement get_move() to generate moves
    given the current game state.
    """
    
    def __init__(self, name: str):
        """Initialize player.
        
        Args:
            name: Human-readable name for this player
        """
        self.name = name
        self._log_path: Optional[Path] = None
        self._game_id: int = 0
    
    def set_log_path(self, log_path: Optional[Path]):
        """Set the path for logging interactions.
        
        Args:
            log_path: Path to JSONL log file, or None to disable logging
        """
        self._log_path = log_path
        if log_path:
            log_path.parent.mkdir(parents=True, exist_ok=True)
    
    def reset_game(self, game_id: int = 0):
        """Reset player state for a new game.
        
        Args:
            game_id: Game identifier for logging
        """
        self._game_id = game_id
    
    @abstractmethod
    def get_move(
        self, 
        move_history: list, 
        rules: str, 
        komi: float, 
        color: str,
        win_rate: Optional[float] = None
    ) -> str:
        """Get next move from the player.
        
        Args:
            move_history: List of [color, coord] pairs in KataGo format
            rules: Explicit rule string (e.g., "koSIMPLEscoreTERRITORYtaxSEKIsui0")
            komi: Komi value
            color: Current player color ("B" or "W")
            win_rate: Reference engine's win rate for current player (optional)
        
        Returns:
            Move in GTP format (e.g., "D4", "Q16", "pass")
        """
        pass
    
    @property
    def requires_logging(self) -> bool:
        """Whether this player type should log interactions.
        
        Override to return False for players that don't need logging (e.g., KataGo).
        """
        return True
    
    def start(self):
        """Start the player (e.g., launch process). Override if needed."""
        pass
    
    def stop(self):
        """Stop the player (e.g., terminate process). Override if needed."""
        pass
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False
