"""HuggingFace model player for Go evaluation.

Loads models directly from HuggingFace checkpoints.
"""

import json
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


class HuggingFacePlayer(Player):
    """LLM player using direct HuggingFace model loading."""
    
    MAX_RETRIES = 5
    
    def __init__(
        self, 
        model_name_or_path: str, 
        device: str = "cuda",
        temperature: float = 0.1, 
        max_new_tokens: int = 1024,
        prompt_template: Optional[str] = None
    ):
        """Initialize HuggingFace player.
        
        Args:
            model_name_or_path: HuggingFace model name or local checkpoint path
            device: Device to load model on ("cuda", "cpu", or "auto")
            temperature: Sampling temperature
            max_new_tokens: Maximum new tokens to generate
            prompt_template: Custom prompt template (uses default if None)
        """
        # Extract model name from path
        model_name = Path(model_name_or_path).name if "/" in model_name_or_path else model_name_or_path
        super().__init__(name=model_name)
        
        self.model_name_or_path = model_name_or_path
        self.device = device
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.prompt_template = prompt_template or DEFAULT_PROMPT_TEMPLATE
        
        self._model = None
        self._tokenizer = None
    
    def start(self):
        """Load the model."""
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
    
    def stop(self):
        """Unload the model to free memory."""
        if self._model is not None:
            del self._model
            del self._tokenizer
            self._model = None
            self._tokenizer = None
            
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def _generate(self, prompt: str) -> str:
        """Generate response from the model."""
        assert self._model is not None, "Model not loaded. Call start() first."
        
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        
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
        """Get move from HuggingFace model with retry logic."""
        # Ensure model is loaded
        if self._model is None:
            self.start()
        
        prompt = self._build_prompt(move_history, rules, komi, color)
        
        last_response = ""
        reasoning = None
        
        for attempt in range(self.MAX_RETRIES):
            response_text = self._generate(prompt)
            last_response = response_text
            
            move_response = parse_json_response(response_text)
            
            if move_response is not None:
                move = move_response.move
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
