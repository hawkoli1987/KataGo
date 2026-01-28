"""OpenAI-compatible LLM player using async HTTP.

Supports vLLM and other OpenAI-compatible API endpoints.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import aiohttp
from pydantic import BaseModel, ValidationError

from eval.players.base import Player, validate_gtp_move


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


class LLMResponse(BaseModel):
    """Expected response format from LLM."""
    reasoning: str
    move: str


class OpenAIPlayer(Player):
    """LLM player using OpenAI-compatible API (async HTTP).
    
    Works with vLLM, OpenAI, and other compatible endpoints.
    """
    
    def __init__(
        self,
        api_base: str,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 1024,
        prompt_template: Optional[str] = None,
        name: Optional[str] = None
    ):
        """Initialize OpenAI player.
        
        Args:
            api_base: API base URL (e.g., "http://localhost:8002/v1")
            model: Model name (auto-detected if None)
            api_key: API key (optional for local vLLM)
            temperature: Sampling temperature
            max_tokens: Max tokens to generate
            prompt_template: Custom prompt template
            name: Player name
        """
        self.api_base = api_base.rstrip("/")
        self.model = model
        self.api_key = api_key or "dummy"
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.prompt_template = prompt_template or DEFAULT_PROMPT_TEMPLATE
        
        self._session: Optional[aiohttp.ClientSession] = None
        self._ply = 0
        
        super().__init__(name=name or model or "openai-player")
    
    @property
    def requires_logging(self) -> bool:
        """LLM players require logging."""
        return True
    
    async def start(self):
        """Start the player (create HTTP session, detect model)."""
        self._session = aiohttp.ClientSession()
        
        # Auto-detect model if not specified
        if self.model is None:
            self.model = await self._detect_model()
            self.name = self.model or "openai-player"
        
        print(f"OpenAI player ready: {self.name} @ {self.api_base}")
    
    async def stop(self):
        """Stop the player (close HTTP session)."""
        if self._session is not None:
            await self._session.close()
            self._session = None
    
    async def _detect_model(self) -> Optional[str]:
        """Auto-detect model from the API."""
        assert self._session is not None
        
        try:
            async with self._session.get(
                f"{self.api_base}/models",
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=aiohttp.ClientTimeout(total=10)
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    models = data.get("data", [])
                    if models:
                        model_id = models[0].get("id")
                        print(f"Auto-detected model: {model_id}")
                        return model_id
        except Exception as e:
            print(f"Failed to auto-detect model: {e}")
        
        return None
    
    def reset_game(self, game_id: int = 0):
        """Reset for a new game."""
        super().reset_game(game_id)
        self._ply = 0
    
    async def get_move(
        self,
        move_history: List[List[str]],
        rules: str,
        komi: float,
        color: str,
        win_rate: Optional[float] = None
    ) -> str:
        """Get move from LLM via API."""
        assert self._session is not None, "Player not started"
        
        self._ply = len(move_history) + 1
        
        # Build prompt
        color_name = "Black" if color == "B" else "White"
        history_str = ", ".join(f"{c}:{m}" for c, m in move_history) if move_history else "none"
        
        prompt = self.prompt_template.format(
            color_name=color_name,
            rules=rules,
            komi=komi,
            history_str=history_str
        )
        
        # Try up to 5 times
        max_retries = 5
        last_error = None
        raw_response = ""
        parsed_move = ""
        reasoning = ""
        
        for attempt in range(max_retries):
            try:
                raw_response = await self._call_api(prompt)
                
                # Parse JSON response
                response_data = self._parse_response(raw_response)
                parsed_move = response_data.move.upper().strip()
                reasoning = response_data.reasoning
                
                # Validate move
                if validate_gtp_move(parsed_move):
                    break
                else:
                    last_error = f"Invalid GTP move: {parsed_move}"
            
            except ValidationError as e:
                last_error = f"JSON validation error: {e}"
            except json.JSONDecodeError as e:
                last_error = f"JSON parse error: {e}"
            except Exception as e:
                last_error = str(e)
            
            if attempt < max_retries - 1:
                print(f"  Retry {attempt + 1}/{max_retries}: {last_error}")
        
        # Log the interaction
        if self._log_path is not None:
            self._log_move(prompt, raw_response, parsed_move, reasoning, win_rate, last_error)
        
        if not validate_gtp_move(parsed_move):
            print(f"  LLM failed after {max_retries} attempts: {last_error}")
            return ""
        
        return parsed_move
    
    async def _call_api(self, prompt: str) -> str:
        """Make API call and return raw response."""
        assert self._session is not None
        
        # Check if model supports JSON mode (DeepSeek)
        is_deepseek = self.model and "deepseek" in self.model.lower()
        
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are an expert Go player."},
                {"role": "user", "content": prompt}
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        
        if is_deepseek:
            payload["response_format"] = {"type": "json_object"}
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        async with self._session.post(
            f"{self.api_base}/chat/completions",
            json=payload,
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=120)
        ) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise RuntimeError(f"API error {resp.status}: {text}")
            
            data = await resp.json()
            
        choices = data.get("choices", [])
        if not choices:
            raise RuntimeError("No choices in API response")
        
        message = choices[0].get("message", {})
        content = message.get("content", "")
        
        return content
    
    def _parse_response(self, raw_response: str) -> LLMResponse:
        """Parse LLM response into structured format."""
        # Try direct JSON parse
        try:
            data = json.loads(raw_response)
            return LLMResponse(**data)
        except (json.JSONDecodeError, ValidationError):
            pass
        
        # Try to extract JSON from response
        import re
        json_match = re.search(r'\{[^{}]*"move"[^{}]*\}', raw_response, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
            return LLMResponse(**data)
        
        raise json.JSONDecodeError("No valid JSON found", raw_response, 0)
    
    def _log_move(
        self,
        prompt: str,
        raw_response: str,
        parsed_move: str,
        reasoning: str,
        win_rate: Optional[float],
        error: Optional[str]
    ):
        """Log the move to JSONL file."""
        if self._log_path is None:
            return
        
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        
        entry = {
            "timestamp": datetime.now().isoformat(),
            "game_id": self._current_game_id,
            "ply": self._ply,
            "prompt": prompt,
            "raw_response": raw_response,
            "reasoning": reasoning,
            "parsed_move": parsed_move,
            "win_rate": win_rate,
            "error": error
        }
        
        with open(self._log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")
