"""Base player interface for Go evaluation.

All players implement an async HTTP-based interface for stateless move generation.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional


# GTP column labels (I is skipped)
GTP_COLS = "ABCDEFGHJKLMNOPQRST"


def validate_gtp_move(move: str) -> bool:
    """Validate a GTP format move.
    
    Valid moves: column letter (A-T, no I) + row number (1-19), or 'pass'/'resign'
    """
    if not move:
        return False
    
    move = move.upper().strip()
    
    if move in ("PASS", "RESIGN"):
        return True
    
    if len(move) < 2 or len(move) > 3:
        return False
    
    col = move[0]
    if col not in GTP_COLS:
        return False
    
    try:
        row = int(move[1:])
        return 1 <= row <= 19
    except ValueError:
        return False


class Player(ABC):
    """Abstract base class for Go players.
    
    All players use a stateless async HTTP interface where each request
    includes the full game state (move history, rules, komi).
    """
    
    def __init__(self, name: str):
        self.name = name
        self._log_path: Optional[Path] = None
        self._current_game_id: int = 0
    
    @property
    def requires_logging(self) -> bool:
        """Whether this player requires LLM-style logging.
        
        Returns True for LLM players, False for KataGo players.
        """
        return False
    
    def set_log_path(self, path: Path):
        """Set the path for logging (if applicable)."""
        self._log_path = path
    
    def reset_game(self, game_id: int = 0):
        """Reset for a new game."""
        self._current_game_id = game_id
    
    @abstractmethod
    async def start(self):
        """Start the player (e.g., spawn server subprocess)."""
        pass
    
    @abstractmethod
    async def stop(self):
        """Stop the player (e.g., terminate server)."""
        pass
    
    @abstractmethod
    async def get_move(
        self,
        move_history: List[List[str]],
        rules: str,
        komi: float,
        color: str,
        win_rate: Optional[float] = None
    ) -> str:
        """Get the next move for the given position.
        
        This is a stateless call - full game state is passed each time.
        
        Args:
            move_history: List of [color, move] pairs, e.g. [["B", "D4"], ["W", "Q16"]]
            rules: KataGo rule string
            komi: Komi value
            color: Color to play ("B" or "W")
            win_rate: Optional win rate from opponent's perspective (for logging)
        
        Returns:
            Move in GTP format (e.g., "D4", "pass", "resign")
        """
        pass
    
    async def get_win_rate(
        self,
        move_history: List[List[str]],
        rules: str,
        komi: float,
        color: str
    ) -> Optional[float]:
        """Get win rate for the given position.
        
        Default implementation returns None. Override for players that support this.
        """
        return None
