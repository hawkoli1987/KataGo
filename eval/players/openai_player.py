"""OpenAI-compatible API player for Go evaluation.

Supports vLLM, OpenAI, and other OpenAI-compatible endpoints.
"""

import json
import urllib.request
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, ValidationError, field_validator

from eval.players.base import Player, GTP_COLS


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


def parse_json_response(response: str) -> Optional[MoveResponse]:
    """Parse and validate JSON response from LLM."""
    response = response.strip()
    
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
    """Format move history for the prompt."""
    if not moves:
        return "(empty - you play first)"
    return ", ".join(f"{color}:{coord}" for color, coord in moves)


def is_deepseek_model(model_name: str) -> bool:
    """Check if model is from DeepSeek family."""
    model_lower = model_name.lower()
    return any(p in model_lower for p in ["deepseek-v3", "deepseek-r1", "deepseekv3", "deepseekr1"])


def fetch_model_name_from_api(api_base: str, api_key: str = "EMPTY") -> Optional[str]:
    """Fetch the model name from the vLLM /v1/models endpoint."""
    url = f"{api_base.rstrip('/')}/models"
    headers = {"Authorization": f"Bearer {api_key}"}
    
    try:
        req = urllib.request.Request(url, headers=headers, method="GET")
        with urllib.request.urlopen(req, timeout=10) as resp:
            result = json.loads(resp.read().decode("utf-8"))
        models = result.get("data", [])
        if models:
            return models[0].get("id")
    except Exception as e:
        print(f"WARNING: Failed to fetch model list from {url}: {e}")
    
    return None


# Default prompt template
DEFAULT_PROMPT_TEMPLATE = """You are playing Go as {color_name}. The board is 19x19.

Rules: {rules}
Komi: {komi}

Move history: {history_str}

Your turn. Analyze the position and provide your move.

You MUST respond with a JSON object in this exact format:
{{"reasoning": "<your_analysis>", "move": "<your_move>"}}

Where <your_move> is in GTP coordinate format:
- Column letter A-T (skipping I)
- Row number 1-19
- Examples: "D4", "Q16", "C3", "pass", "resign"

Respond with ONLY the JSON object."""


class OpenAIPlayer(Player):
    """LLM player using OpenAI-compatible API (vLLM, OpenAI, etc.)."""
    
    MAX_RETRIES = 5
    
    def __init__(
        self, 
        api_base: str, 
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.1, 
        max_tokens: int = 1024,
        prompt_template: Optional[str] = None
    ):
        """Initialize OpenAI-compatible player.
        
        Args:
            api_base: API base URL (e.g., "http://localhost:8001/v1")
            model: Model name (auto-detected if None)
            api_key: API key for authentication
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            prompt_template: Custom prompt template (uses default if None)
        """
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key or "EMPTY"
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.prompt_template = prompt_template or DEFAULT_PROMPT_TEMPLATE
        
        # Auto-detect model name if not provided
        if model is None:
            detected_model = fetch_model_name_from_api(self.api_base, self.api_key)
            assert detected_model is not None, \
                f"Could not auto-detect model from {self.api_base}. Provide model explicitly."
            self.model = detected_model
            print(f"Auto-detected model: {self.model}")
        else:
            self.model = model
        
        super().__init__(name=self.model)
        
        self.use_json_response_format = is_deepseek_model(self.model)
    
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
        
        if self.use_json_response_format:
            data["response_format"] = {"type": "json_object"}
        
        req = urllib.request.Request(
            url,
            data=json.dumps(data).encode("utf-8"),
            headers=headers,
            method="POST"
        )
        
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read().decode("utf-8"))
    
    def _build_prompt(self, move_history: list, rules: str, komi: float, color: str) -> str:
        """Build the prompt for the LLM."""
        color_name = "Black" if color == "B" else "White"
        history_str = format_move_history(move_history)
        
        return self.prompt_template.format(
            color_name=color_name,
            rules=rules,
            komi=komi,
            history_str=history_str
        )
    
    def _log_interaction(
        self, 
        prompt: str, 
        response: str, 
        parsed_move: str,
        move_history: list, 
        rules: str, 
        komi: float, 
        color: str,
        reasoning: Optional[str] = None, 
        retries: int = 0,
        win_rate: Optional[float] = None
    ):
        """Log an LLM interaction to the log file."""
        if not self._log_path:
            return
        
        log_entry = {
            "game_id": self._game_id,
            "ply": len(move_history),
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
    
    def get_move(
        self, 
        move_history: list, 
        rules: str, 
        komi: float, 
        color: str,
        win_rate: Optional[float] = None
    ) -> str:
        """Get move from OpenAI-compatible API with retry logic."""
        prompt = self._build_prompt(move_history, rules, komi, color)
        
        last_response = ""
        reasoning = None
        
        for attempt in range(self.MAX_RETRIES):
            try:
                result = self._make_request(prompt)
            except Exception as e:
                last_response = f"API_ERROR: {e}"
                continue
            
            choice = result["choices"][0]
            response_text = choice["message"]["content"]
            last_response = response_text
            
            # Extract reasoning from API response if present
            if "reasoning_content" in choice["message"]:
                reasoning = choice["message"]["reasoning_content"]
            
            move_response = parse_json_response(response_text)
            
            if move_response is not None:
                move = move_response.move
                if reasoning is None and move_response.reasoning:
                    reasoning = move_response.reasoning
                
                self._log_interaction(
                    prompt, response_text, move, move_history, rules, komi, color,
                    reasoning=reasoning, retries=attempt, win_rate=win_rate
                )
                return move
        
        # All retries exhausted
        self._log_interaction(
            prompt, last_response, "", move_history, rules, komi, color,
            reasoning=reasoning, retries=self.MAX_RETRIES, win_rate=win_rate
        )
        return ""
