"""Player implementations for Go evaluation.

This module provides different player types:
- OpenAIPlayer: LLM via OpenAI-compatible API (vLLM, etc.)
- HuggingFacePlayer: LLM via direct HuggingFace model loading
- KataGoPlayer: KataGo neural network as candidate
"""

from eval.players.base import Player, validate_gtp_move, GTP_COLS
from eval.players.openai_player import OpenAIPlayer
from eval.players.huggingface_player import HuggingFacePlayer
from eval.players.katago_player import KataGoPlayer

__all__ = [
    "Player",
    "OpenAIPlayer", 
    "HuggingFacePlayer",
    "KataGoPlayer",
    "validate_gtp_move",
    "GTP_COLS",
]
