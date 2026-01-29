"""
Custom reward function for KataGo offline training.

This function computes a weighted sum of three rewards:
1. r_wr: negative MSE on root winrate prediction (range [-1, 0])
2. r_move: 1 if LLM move matches KataGo top move, 0 otherwise
3. r_legal: 0 if move is legal, -1 if illegal

Ground truth data comes from extra_info (pre-populated by preprocessor):
- root_winrate: Ground truth winrate from KataGo analysis
- top_move: Ground truth top move from KataGo analysis
- move_error: Error message if move was illegal, None otherwise

The reward function is called by veRL during training to score each generated response.
"""

import json
import re
import sys
from pathlib import Path
from typing import Optional, Tuple

import yaml

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

REPO_ROOT = Path(__file__).parent.parent.parent.resolve()
CONFIG_PATH = REPO_ROOT / "configs" / "config.yaml"

# Cache for reward weights
_reward_weights_cache = None


def load_reward_weights(config_path: Path = CONFIG_PATH) -> dict:
    """Load reward weights from config.yaml."""
    global _reward_weights_cache
    
    if _reward_weights_cache is not None:
        return _reward_weights_cache
    
    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        rewards_cfg = config.get("training", {}).get("rewards", {})
        _reward_weights_cache = {
            "r_wr_weight": rewards_cfg.get("r_wr_weight", 1.0),
            "r_move_weight": rewards_cfg.get("r_move_weight", 1.0),
            "r_legal_weight": rewards_cfg.get("r_legal_weight", 1.0),
        }
        return _reward_weights_cache
    except Exception as e:
        # Fallback to defaults
        print(f"WARNING: Failed to load reward weights from config: {e}")
        return {"r_wr_weight": 1.0, "r_move_weight": 1.0, "r_legal_weight": 1.0}


def normalize_move(move: str) -> str:
    """Normalize GTP coordinate to lowercase."""
    if not move:
        return ""
    return move.lower().strip()


def extract_root_winrate_and_move(
    response_str: str,
    model_family: Optional[str] = None
) -> Tuple[Optional[float], Optional[str]]:
    """
    Extract root_winrate and top_move from LLM response.
    
    Handles various output formats:
    - Clean JSON: {"root_winrate": 0.65, "top_move": "D4"}
    - JSON with reasoning: "Analysis... {"root_winrate": 0.65, "top_move": "D4"}"
    - Thinking format: <think>...</think>{"root_winrate": 0.65, "top_move": "D4"}
    
    Supports different model families:
    - Qwen3: <think>...</think> tags
    - DeepSeekv3.2: <think>...</think> tags
    - Gemma3: <think>...</think> tags
    
    Args:
        response_str: Model's generated response
        model_family: Model family name (optional, for future model-specific parsing)
    
    Returns:
        Tuple of (root_winrate, top_move) or (None, None) if extraction fails
    """
    if not response_str or not isinstance(response_str, str):
        return (None, None)
    
    response_str = response_str.strip()
    
    # Remove thinking blocks if present (extract content after </think>)
    # This handles Qwen3, DeepSeekv3.2, Gemma3 thinking formats
    think_pattern = r'<think>.*?</think>'
    if re.search(think_pattern, response_str, flags=re.DOTALL):
        # Extract content after </think> tag
        parts = re.split(r'</think>', response_str, flags=re.IGNORECASE)
        if len(parts) > 1:
            response_str = parts[-1].strip()
        else:
            # If no closing tag, remove thinking block
            response_str = re.sub(think_pattern, '', response_str, flags=re.DOTALL)
    
    # Try to find JSON object with root_winrate and top_move
    # Look for the last JSON object that contains both fields
    json_pattern = r'\{[^{}]*(?:"root_winrate"|"top_move")[^{}]*\}'
    matches = re.findall(json_pattern, response_str, re.IGNORECASE | re.DOTALL)
    
    # Try to find a larger JSON object that might span multiple lines
    if not matches:
        # Try multiline JSON pattern
        json_pattern = r'\{[^{}]*(?:"root_winrate"|"top_move")[^}]*\}'
        matches = re.findall(json_pattern, response_str, re.IGNORECASE | re.DOTALL)
    
    # If still no matches, try to find any JSON object
    if not matches:
        json_pattern = r'\{[^{}]*\}'
        matches = re.findall(json_pattern, response_str, re.IGNORECASE | re.DOTALL)
    
    root_winrate = None
    top_move = None
    
    # Try to parse each match (starting from the last one)
    for json_str in reversed(matches):
        try:
            data = json.loads(json_str)
            
            # Extract root_winrate (case-insensitive)
            root_winrate = (
                data.get("root_winrate") or
                data.get("rootWinrate") or
                data.get("root_win_rate") or
                data.get("winrate") or
                data.get("win_rate")
            )
            
            # Extract top_move (case-insensitive)
            top_move = (
                data.get("top_move") or
                data.get("topMove") or
                data.get("top_move") or
                data.get("move") or
                data.get("best_move") or
                data.get("bestMove")
            )
            
            # If we found both, we're done
            if root_winrate is not None and top_move is not None:
                break
            
        except (json.JSONDecodeError, ValueError, TypeError):
            continue
    
    # Validate and normalize
    if root_winrate is not None:
        try:
            root_winrate = float(root_winrate)
            root_winrate = max(0.0, min(1.0, root_winrate))  # Clamp to [0, 1]
        except (ValueError, TypeError):
            root_winrate = None
    
    if top_move is not None:
        top_move = normalize_move(str(top_move))
        if not top_move or top_move == "none" or top_move == "null":
            top_move = None
    
    return (root_winrate, top_move)


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: Optional[dict] = None
) -> float:
    """
    Compute weighted reward for offline KataGo training.
    
    This is the main entry point called by veRL's RewardManager.
    
    Args:
        data_source: Dataset identifier (e.g., "katago/offline")
        solution_str: Model's generated response
        ground_truth: Pre-computed ground truth winrate (from parquet, string format)
        extra_info: Contains ground truth data from preprocessor:
            - root_winrate: Ground truth winrate (float or None)
            - top_move: Ground truth top move (string or None)
            - move_error: Error message if move was illegal (string or None)
            - model_family: Optional model family name for parsing
    
    Returns:
        float: Weighted reward value
    """
    # Load reward weights
    weights = load_reward_weights()
    
    # Extract predictions from LLM response
    model_family = extra_info.get("model_family") if extra_info else None
    predicted_winrate, predicted_move = extract_root_winrate_and_move(solution_str, model_family)
    
    # Get ground truth from extra_info
    if not extra_info:
        # Fallback: no extra_info means we can't compute rewards
        return -1.0
    
    gt_root_winrate = extra_info.get("root_winrate")
    gt_top_move = extra_info.get("top_move")
    move_error = extra_info.get("move_error")
    
    # Check if move is illegal (error present in KataGo response)
    is_illegal = move_error is not None
    
    # Compute r_wr: negative MSE on root winrate (range [-1, 0])
    # If move is illegal, return -1
    if is_illegal:
        r_wr = -1.0
    elif predicted_winrate is None or gt_root_winrate is None:
        r_wr = -1.0  # Format error or missing ground truth
    else:
        try:
            gt_winrate = float(gt_root_winrate)
            mse = (predicted_winrate - gt_winrate) ** 2
            r_wr = -mse  # Negative MSE, range [-1, 0]
        except (ValueError, TypeError):
            r_wr = -1.0
    
    # Compute r_move: 1 if LLM move matches KataGo top move, 0 otherwise
    # If move is illegal, return 0
    if is_illegal:
        r_move = 0.0
    elif predicted_move is None or gt_top_move is None:
        r_move = 0.0  # Missing prediction or ground truth
    else:
        # Normalize both moves for comparison
        pred_move_norm = normalize_move(predicted_move)
        gt_move_norm = normalize_move(gt_top_move)
        r_move = 1.0 if pred_move_norm == gt_move_norm else 0.0
    
    # Compute r_legal: 0 if move is legal, -1 if illegal
    r_legal = -1.0 if is_illegal else 0.0
    
    # Compute weighted sum
    reward_total = (
        weights["r_wr_weight"] * r_wr +
        weights["r_move_weight"] * r_move +
        weights["r_legal_weight"] * r_legal
    )
    
    # Store individual rewards in extra_info for logging (if veRL supports it)
    # Note: veRL may not support modifying extra_info, but we try anyway
    if extra_info:
        extra_info["_reward_components"] = {
            "r_wr": r_wr,
            "r_move": r_move,
            "r_legal": r_legal,
            "reward_total": reward_total,
            "predicted_winrate": predicted_winrate,
            "predicted_move": predicted_move,
        }
    
    return reward_total


# For testing
if __name__ == "__main__":
    print("Testing offline reward function:\n")
    
    test_cases = [
        # (response, extra_info, expected_approx_reward_total)
        (
            '{"root_winrate": 0.65, "top_move": "D4"}',
            {"root_winrate": 0.65, "top_move": "D4", "move_error": None},
            0.0,  # Perfect match: r_wr=0, r_move=1, r_legal=0 -> total=1.0
        ),
        (
            '{"root_winrate": 0.60, "top_move": "D4"}',
            {"root_winrate": 0.65, "top_move": "D4", "move_error": None},
            -0.0025,  # r_wr=-0.0025, r_move=1, r_legal=0 -> total=0.9975
        ),
        (
            '{"root_winrate": 0.65, "top_move": "E4"}',
            {"root_winrate": 0.65, "top_move": "D4", "move_error": None},
            -1.0,  # r_wr=0, r_move=0, r_legal=0 -> total=0.0
        ),
        (
            '{"root_winrate": 0.65, "top_move": "XX"}',
            {"root_winrate": 0.65, "top_move": "D4", "move_error": "Illegal move"},
            -3.0,  # Illegal: r_wr=-1, r_move=0, r_legal=-1 -> total=-3.0
        ),
        (
            '<think>Let me analyze...</think>{"root_winrate": 0.55, "top_move": "d4"}',
            {"root_winrate": 0.55, "top_move": "D4", "move_error": None},
            0.0,  # Perfect match with thinking mode
        ),
    ]
    
    print(f"{'Response':<60} {'GT WR':>8} {'GT Move':>10} {'Pred WR':>8} {'Pred Move':>10} {'Reward':>8}")
    print("-" * 110)
    
    for response, extra_info, expected in test_cases:
        score = compute_score(
            data_source="katago/offline",
            solution_str=response,
            ground_truth=str(extra_info.get("root_winrate", "")),
            extra_info=extra_info
        )
        
        pred_wr, pred_move = extract_root_winrate_and_move(response)
        gt_wr = extra_info.get("root_winrate")
        gt_move = extra_info.get("top_move")
        
        pred_wr_str = f"{pred_wr:.2f}" if pred_wr is not None else "None"
        pred_move_str = pred_move if pred_move else "None"
        gt_wr_str = f"{gt_wr:.2f}" if gt_wr is not None else "None"
        gt_move_str = gt_move if gt_move else "None"
        
        print(f"{response[:58]:<60} {gt_wr_str:>8} {gt_move_str:>10} {pred_wr_str:>8} {pred_move_str:>10} {score:>8.4f}")
    
    print("\nAll tests use pre-computed ground truth from extra_info.")
