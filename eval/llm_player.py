#!/usr/bin/env python3
"""LLM player abstraction for Go game evaluation.

This module provides LLM player classes that can generate Go moves given
game state information. Supports OpenAI-compatible APIs (vLLM, OpenAI) and
direct HuggingFace model loading.

Usage:
    # Test with OpenAI-compatible API
    python eval/llm_player.py --type openai --endpoint http://localhost:8001/v1 --model deepseek-ai/DeepSeek-V3.2
    
    # Test with HuggingFace model
    python eval/llm_player.py --type huggingface --model path/to/checkpoint
"""

import argparse
import re
from abc import ABC, abstractmethod
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


def parse_llm_response(response: str) -> str:
    """Extract a GTP move from LLM response text.
    
    Tries various patterns to find a valid move coordinate.
    Returns the extracted move or empty string if none found.
    """
    response = response.strip()
    
    # Try to find common patterns
    # Pattern 1: Just the move (e.g., "D4", "Q16", "pass")
    simple_match = re.match(r'^([A-HJ-T]\d{1,2}|pass|resign)$', response, re.IGNORECASE)
    if simple_match:
        return simple_match.group(1).upper()
    
    # Pattern 2: "Move: D4" or "My move is D4"
    move_pattern = re.search(r'(?:move[:\s]+|play[:\s]+)([A-HJ-T]\d{1,2}|pass|resign)', response, re.IGNORECASE)
    if move_pattern:
        return move_pattern.group(1).upper()
    
    # Pattern 3: Find any coordinate-like pattern
    coord_pattern = re.search(r'\b([A-HJ-T]\d{1,2})\b', response, re.IGNORECASE)
    if coord_pattern:
        return coord_pattern.group(1).upper()
    
    # Pattern 4: Check for pass/resign
    if re.search(r'\bpass\b', response, re.IGNORECASE):
        return "PASS"
    if re.search(r'\bresign\b', response, re.IGNORECASE):
        return "RESIGN"
    
    return ""


def format_move_history(moves: list) -> str:
    """Format move history for the prompt.
    
    Args:
        moves: List of [color, coord] pairs, e.g., [["B", "D4"], ["W", "Q16"], ...]
    
    Returns:
        Formatted string representation
    """
    if not moves:
        return "(empty - you play first)"
    
    # Format as: B:D4, W:Q16, B:D16, ...
    formatted = ", ".join(f"{color}:{coord}" for color, coord in moves)
    return formatted


class LLMPlayer(ABC):
    """Abstract base class for LLM Go players."""
    
    @abstractmethod
    def get_move(self, move_history: list, rules: str, komi: float, color: str) -> str:
        """Get next move from the LLM.
        
        Args:
            move_history: List of [color, coord] pairs in KataGo format
            rules: Explicit rule string (e.g., "koSIMPLEscoreTERRITORYtaxSEKIsui0")
            komi: Komi value
            color: Current player color ("B" or "W")
        
        Returns:
            Move in GTP format (e.g., "D4", "Q16", "pass")
        """
        pass
    
    def build_prompt(self, move_history: list, rules: str, komi: float, color: str) -> str:
        """Build the prompt for the LLM."""
        color_name = "Black" if color == "B" else "White"
        history_str = format_move_history(move_history)
        
        prompt = f"""You are playing Go as {color_name}. The board is 19x19.

Rules: {rules}
Komi: {komi}

Move history: {history_str}

Your turn. Output ONLY your move in GTP coordinate format (e.g., "D4", "Q16", "pass").
Move:"""
        return prompt


class OpenAICompatiblePlayer(LLMPlayer):
    """LLM player using OpenAI-compatible API (vLLM, OpenAI, etc.)."""
    
    def __init__(self, api_base: str, model: str, api_key: Optional[str] = None,
                 temperature: float = 0.1, max_tokens: int = 10):
        self.api_base = api_base.rstrip("/")
        self.model = model
        self.api_key = api_key or "EMPTY"
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Import here to avoid dependency issues
        import urllib.request
        import json
        self._urllib = urllib
        self._json = json
    
    def get_move(self, move_history: list, rules: str, komi: float, color: str) -> str:
        """Get move from OpenAI-compatible API."""
        prompt = self.build_prompt(move_history, rules, komi, color)
        
        # Build request
        url = f"{self.api_base}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        
        req = self._urllib.request.Request(
            url,
            data=self._json.dumps(data).encode("utf-8"),
            headers=headers,
            method="POST"
        )
        
        with self._urllib.request.urlopen(req, timeout=60) as resp:
            result = self._json.loads(resp.read().decode("utf-8"))
        
        response_text = result["choices"][0]["message"]["content"]
        move = parse_llm_response(response_text)
        
        return move


class HuggingFacePlayer(LLMPlayer):
    """LLM player using direct HuggingFace model loading."""
    
    def __init__(self, model_name_or_path: str, device: str = "cuda",
                 temperature: float = 0.1, max_new_tokens: int = 10):
        self.model_name_or_path = model_name_or_path
        self.device = device
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        
        self._model = None
        self._tokenizer = None
    
    def _load_model(self):
        """Lazy load the model."""
        if self._model is not None:
            return
        
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        
        print(f"Loading model from {self.model_name_or_path}...")
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path,
            torch_dtype=torch.float16,
            device_map=self.device
        )
        print("Model loaded.")
    
    def get_move(self, move_history: list, rules: str, komi: float, color: str) -> str:
        """Get move from HuggingFace model."""
        self._load_model()
        
        prompt = self.build_prompt(move_history, rules, komi, color)
        
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self.device)
        
        outputs = self._model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            do_sample=self.temperature > 0,
            pad_token_id=self._tokenizer.eos_token_id
        )
        
        response_text = self._tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )
        
        move = parse_llm_response(response_text)
        return move


def create_player(player_type: str, **kwargs) -> LLMPlayer:
    """Factory function to create an LLM player.
    
    Args:
        player_type: "openai" or "huggingface"
        **kwargs: Arguments passed to the player constructor
    
    Returns:
        LLMPlayer instance
    """
    if player_type == "openai":
        return OpenAICompatiblePlayer(**kwargs)
    elif player_type == "huggingface":
        return HuggingFacePlayer(**kwargs)
    else:
        raise ValueError(f"Unknown player type: {player_type}")


def main():
    """Test the LLM player with a sample position."""
    parser = argparse.ArgumentParser(description="Test LLM Go player")
    parser.add_argument("--type", choices=["openai", "huggingface"], required=True,
                        help="Player type")
    parser.add_argument("--endpoint", type=str, default="http://localhost:8001/v1",
                        help="API endpoint for OpenAI-compatible player")
    parser.add_argument("--model", type=str, required=True,
                        help="Model name or path")
    parser.add_argument("--api-key", type=str, default=None,
                        help="API key (optional)")
    args = parser.parse_args()
    
    # Create player
    if args.type == "openai":
        player = OpenAICompatiblePlayer(
            api_base=args.endpoint,
            model=args.model,
            api_key=args.api_key
        )
    else:
        player = HuggingFacePlayer(
            model_name_or_path=args.model
        )
    
    # Test with sample positions
    test_cases = [
        # Empty board - first move
        {
            "move_history": [],
            "rules": "koSIMPLEscoreTERRITORYtaxSEKIsui0",
            "komi": 6.5,
            "color": "B"
        },
        # After a few moves
        {
            "move_history": [["B", "D4"], ["W", "Q16"], ["B", "D16"]],
            "rules": "koPOSITIONALscoreAREAtaxNONEsui1",
            "komi": 7.5,
            "color": "W"
        },
    ]
    
    print(f"Testing {args.type} player with model: {args.model}")
    print("=" * 60)
    
    for i, tc in enumerate(test_cases):
        print(f"\nTest case {i + 1}:")
        print(f"  Rules: {tc['rules']}")
        print(f"  Komi: {tc['komi']}")
        print(f"  Color: {tc['color']}")
        print(f"  Move history: {tc['move_history']}")
        
        try:
            move = player.get_move(**tc)
            valid = validate_gtp_move(move) if move else False
            print(f"  LLM move: {move}")
            print(f"  Valid: {valid}")
        except Exception as e:
            print(f"  ERROR: {e}")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
