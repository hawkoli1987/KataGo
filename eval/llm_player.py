#!/usr/bin/env python3
"""LLM player abstraction for Go game evaluation.

This module provides LLM player classes that can generate Go moves given
game state information. Supports OpenAI-compatible APIs (vLLM, OpenAI) and
direct HuggingFace model loading.

Features:
- JSON-structured output with Pydantic validation
- DeepSeek-V3 and Qwen3 reasoning support
- Automatic retry on invalid responses (max 5 attempts)
- Auto-detect model name from vLLM API endpoint

Usage:
    # Test with OpenAI-compatible API (auto-detect model)
    python eval/llm_player.py --type openai --endpoint http://localhost:8001/v1
    
    # Test with HuggingFace model
    python eval/llm_player.py --type huggingface --model path/to/checkpoint
"""

import argparse
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, ValidationError, field_validator

# GTP coordinate system: A-T (skipping I), 1-19
GTP_COLS = "ABCDEFGHJKLMNOPQRST"  # Note: no 'I'

# Path to prompts configuration
PROMPTS_PATH = Path(__file__).parent / "prompts.json"


def load_prompt_template() -> str:
    """Load the Go move prompt template from prompts.json."""
    with open(PROMPTS_PATH) as f:
        prompts = json.load(f)
    return prompts["go_move"]["template"]


class MoveResponse(BaseModel):
    """Pydantic model for LLM move response validation."""
    reasoning: Optional[str] = None
    move: str
    
    @field_validator('move')
    @classmethod
    def validate_move_format(cls, v: str) -> str:
        """Validate move is in GTP format."""
        v = v.strip().upper()
        
        if v in ("PASS", "RESIGN"):
            return v
        
        if len(v) < 2 or len(v) > 3:
            raise ValueError(f"Invalid move length: {v}")
        
        col = v[0]
        if col not in GTP_COLS:
            raise ValueError(f"Invalid column: {col}")
        
        try:
            row = int(v[1:])
            if row < 1 or row > 19:
                raise ValueError(f"Invalid row: {row}")
        except ValueError as e:
            raise ValueError(f"Invalid row format: {v[1:]}") from e
        
        return v


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


def parse_json_response(response: str) -> Optional[MoveResponse]:
    """Parse and validate JSON response from LLM.
    
    Args:
        response: Raw response text from LLM
        
    Returns:
        MoveResponse if valid, None otherwise
    """
    response = response.strip()
    
    # Find JSON object boundaries
    start_idx = response.find('{')
    end_idx = response.rfind('}')
    
    if start_idx == -1 or end_idx == -1 or end_idx <= start_idx:
        return None
    
    json_str = response[start_idx:end_idx + 1]
    
    try:
        data = json.loads(json_str)
        return MoveResponse(**data)
    except (json.JSONDecodeError, ValidationError):
        return None


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


def is_deepseek_model(model_name: str) -> bool:
    """Check if model is from DeepSeek family (supports json_object response format)."""
    model_lower = model_name.lower()
    deepseek_patterns = [
        "deepseek-v3",
        "deepseek-r1",
        "deepseekv3",
        "deepseekr1",
    ]
    return any(pattern in model_lower for pattern in deepseek_patterns)


def is_reasoning_model(model_name: str) -> bool:
    """Check if model supports reasoning mode.
    
    Supported model families:
    - DeepSeek V3 family: V3, V3.1, V3.2, R1, R1-Distill, etc.
    - Qwen3 family: Qwen3, Qwen3-VL, etc.
    """
    model_lower = model_name.lower()
    reasoning_patterns = [
        # DeepSeek V3 family
        "deepseek-v3",
        "deepseek-r1",
        "deepseekv3",
        "deepseekr1",
        # Qwen3 family
        "qwen3",
        "qwen-3",
        "qwen2.5",
    ]
    return any(pattern in model_lower for pattern in reasoning_patterns)


def fetch_model_name_from_api(api_base: str, api_key: str = "EMPTY") -> Optional[str]:
    """Fetch the model name from the vLLM /v1/models endpoint.
    
    Args:
        api_base: The API base URL (e.g., "http://localhost:8001/v1")
        api_key: API key for authentication
        
    Returns:
        Model name if found, None otherwise
    """
    import urllib.request
    
    url = f"{api_base.rstrip('/')}/models"
    headers = {
        "Authorization": f"Bearer {api_key}"
    }
    
    try:
        req = urllib.request.Request(url, headers=headers, method="GET")
        with urllib.request.urlopen(req, timeout=10) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            
        models = result.get("data", [])
        if models:
            # Return the first model's ID
            return models[0].get("id")
    except Exception as e:
        print(f"WARNING: Failed to fetch model list from {url}: {e}")
    
    return None


class LLMPlayer(ABC):
    """Abstract base class for LLM Go players."""
    
    MAX_RETRIES = 5
    
    def __init__(self):
        self._log_path: Optional[Path] = None
        self._game_id: int = 0
        self._prompt_template: str = load_prompt_template()
    
    def set_log_path(self, log_path: Optional[Path]):
        """Set the path for logging prompts and responses.
        
        Args:
            log_path: Path to JSONL log file, or None to disable logging
        """
        self._log_path = log_path
        if log_path:
            log_path.parent.mkdir(parents=True, exist_ok=True)
    
    def reset_move_counter(self, game_id: Optional[int] = None):
        """Reset for a new game.
        
        Args:
            game_id: Optional game identifier for logging
        """
        if game_id is not None:
            self._game_id = game_id
    
    def _log_interaction(self, prompt: str, response: str, parsed_move: str, 
                         move_history: list, rules: str, komi: float, color: str,
                         reasoning: Optional[str] = None, retries: int = 0,
                         win_rate: Optional[float] = None):
        """Log an LLM interaction to the log file.
        
        Args:
            prompt: The prompt sent to the LLM
            response: Raw response from the LLM
            parsed_move: The parsed move from the response
            move_history: Current move history
            rules: Rule string
            komi: Komi value
            color: Player color
            reasoning: Reasoning/analysis from the LLM
            retries: Number of retry attempts
            win_rate: KataGo's win rate for the current player before the move
        """
        if not self._log_path:
            return
        
        # Use ply (total moves played) as per KataGo convention
        ply = len(move_history)
        
        log_entry = {
            "game_id": self._game_id,
            "ply": ply,
            "color": color,
            "rules": rules,
            "komi": komi,
            "prompt": prompt,
            "raw_response": response,
            "parsed_move": parsed_move,
            "retries": retries,
        }
        
        if reasoning:
            log_entry["reasoning"] = reasoning
        
        if win_rate is not None:
            log_entry["win_rate"] = win_rate
        
        with open(self._log_path, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
    
    @abstractmethod
    def get_move(self, move_history: list, rules: str, komi: float, color: str,
                 win_rate: Optional[float] = None) -> str:
        """Get next move from the LLM.
        
        Args:
            move_history: List of [color, coord] pairs in KataGo format
            rules: Explicit rule string (e.g., "koSIMPLEscoreTERRITORYtaxSEKIsui0")
            komi: Komi value
            color: Current player color ("B" or "W")
            win_rate: KataGo's win rate for the current player (from reference model's perspective)
        
        Returns:
            Move in GTP format (e.g., "D4", "Q16", "pass")
        """
        pass
    
    def build_prompt(self, move_history: list, rules: str, komi: float, color: str) -> str:
        """Build the prompt for the LLM with JSON output format.
        
        Args:
            move_history: List of [color, coord] pairs
            rules: Explicit rule string
            komi: Komi value
            color: Current player color ("B" or "W")
        """
        color_name = "Black" if color == "B" else "White"
        history_str = format_move_history(move_history)
        
        return self._prompt_template.format(
            color_name=color_name,
            rules=rules,
            komi=komi,
            history_str=history_str
        )


class OpenAICompatiblePlayer(LLMPlayer):
    """LLM player using OpenAI-compatible API (vLLM, OpenAI, etc.)."""
    
    def __init__(self, api_base: str, model: Optional[str] = None, 
                 api_key: Optional[str] = None,
                 temperature: float = 0.1, max_tokens: int = 1024):
        super().__init__()
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key or "EMPTY"
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Auto-detect model name if not provided
        if model is None:
            detected_model = fetch_model_name_from_api(self.api_base, self.api_key)
            if detected_model:
                self.model = detected_model
                print(f"Auto-detected model: {self.model}")
            else:
                raise ValueError("Could not auto-detect model name. Please provide --candidate-model")
        else:
            self.model = model
        
        self.enable_reasoning = is_reasoning_model(self.model)
        self.use_json_response_format = is_deepseek_model(self.model)
        
        # Import here to avoid dependency issues
        import urllib.request
        self._urllib = urllib
    
    def _make_request(self, prompt: str) -> dict:
        """Make API request and return the result."""
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
        
        # Enable JSON response format for DeepSeek models (not all models support this)
        if self.use_json_response_format:
            data["response_format"] = {"type": "json_object"}
        
        req = self._urllib.request.Request(
            url,
            data=json.dumps(data).encode("utf-8"),
            headers=headers,
            method="POST"
        )
        
        with self._urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read().decode("utf-8"))
    
    def get_move(self, move_history: list, rules: str, komi: float, color: str,
                 win_rate: Optional[float] = None) -> str:
        """Get move from OpenAI-compatible API with retry logic."""
        prompt = self.build_prompt(move_history, rules, komi, color)
        
        last_response = ""
        last_error = ""
        reasoning = None
        
        for attempt in range(self.MAX_RETRIES):
            try:
                result = self._make_request(prompt)
            except Exception as e:
                last_error = str(e)
                last_response = f"API_ERROR: {e}"
                continue
            
            # Extract response content
            choice = result["choices"][0]
            response_text = choice["message"]["content"]
            last_response = response_text
            
            # Extract reasoning from API response if present (DeepSeek R1 style)
            if "reasoning_content" in choice["message"]:
                reasoning = choice["message"]["reasoning_content"]
            
            # Parse JSON response
            move_response = parse_json_response(response_text)
            
            if move_response is not None:
                move = move_response.move
                # Get reasoning from JSON if not already from API
                if reasoning is None and move_response.reasoning:
                    reasoning = move_response.reasoning
                
                # Log the successful interaction
                self._log_interaction(
                    prompt, response_text, move, move_history, rules, komi, color,
                    reasoning=reasoning, retries=attempt, win_rate=win_rate
                )
                return move
        
        # All retries exhausted - log and return empty string
        self._log_interaction(
            prompt, last_response, "", move_history, rules, komi, color,
            reasoning=reasoning, retries=self.MAX_RETRIES, win_rate=win_rate
        )
        return ""


class HuggingFacePlayer(LLMPlayer):
    """LLM player using direct HuggingFace model loading."""
    
    def __init__(self, model_name_or_path: str, device: str = "cuda",
                 temperature: float = 0.1, max_new_tokens: int = 1024):
        super().__init__()
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
    
    def _generate(self, prompt: str) -> str:
        """Generate response from the model."""
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
        
        return response_text
    
    def get_move(self, move_history: list, rules: str, komi: float, color: str,
                 win_rate: Optional[float] = None) -> str:
        """Get move from HuggingFace model with retry logic."""
        self._load_model()
        
        prompt = self.build_prompt(move_history, rules, komi, color)
        
        last_response = ""
        reasoning = None
        
        for attempt in range(self.MAX_RETRIES):
            response_text = self._generate(prompt)
            last_response = response_text
            
            # Parse JSON response
            move_response = parse_json_response(response_text)
            
            if move_response is not None:
                move = move_response.move
                reasoning = move_response.reasoning
                
                # Log the successful interaction
                self._log_interaction(
                    prompt, response_text, move, move_history, rules, komi, color,
                    reasoning=reasoning, retries=attempt, win_rate=win_rate
                )
                return move
        
        # All retries exhausted - log and return empty string
        self._log_interaction(
            prompt, last_response, "", move_history, rules, komi, color,
            reasoning=reasoning, retries=self.MAX_RETRIES, win_rate=win_rate
        )
        return ""


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
    parser.add_argument("--model", type=str, default=None,
                        help="Model name or path (auto-detected for openai type)")
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
        print(f"Model: {player.model}")
        print(f"Reasoning mode enabled: {player.enable_reasoning}")
    else:
        assert args.model is not None, "Model path required for huggingface type"
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
    
    print(f"Testing {args.type} player")
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
