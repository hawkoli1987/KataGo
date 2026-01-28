"""Player implementations for Go evaluation.

This module provides different player types:
- OpenAIPlayer: LLM via OpenAI-compatible API (vLLM, etc.)
- KataGoPlayer: KataGo neural network via FastAPI server

All players use async HTTP for stateless move generation.
"""

from eval.players.base import Player, validate_gtp_move, GTP_COLS
from eval.players.openai_player import OpenAIPlayer
from eval.players.katago_player import KataGoPlayer

__all__ = [
    "Player",
    "OpenAIPlayer",
    "KataGoPlayer",
    "validate_gtp_move",
    "GTP_COLS",
]
