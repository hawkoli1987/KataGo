"""
Custom reward function for KataGo winrate prediction training.

This function computes -MSE (negative mean squared error) between the model's
predicted winrate and the ground truth winrate.

Ground truth source (in order of preference):
1. Pre-computed winrate stored in reward_model.ground_truth (from SGF data)
2. Pre-computed winrate stored in extra_info.winrate
3. Real-time API call to KataGo engine (fallback, slower)

The reward function is called by veRL during training to score each generated response.
"""

import json
import re
import requests
from typing import Optional

# KataGo Analysis Engine endpoint (used only as fallback)
KATAGO_ENDPOINT = "http://hopper-34:9000/analysis"
REQUEST_TIMEOUT = 30

# Whether to use real-time API calls as fallback
ENABLE_API_FALLBACK = True


def extract_winrate_from_response(response_str: str) -> Optional[float]:
    """
    Extract winrate value from model response.
    
    Handles various output formats:
    - Clean JSON: {"winrate": 0.56}
    - JSON with reasoning: "The position looks balanced... {"winrate": 0.52}"
    - Markdown code blocks: ```json\n{"winrate": 0.55}\n```
    - Qwen3 thinking format: <think>...</think>{"winrate": 0.54}
    """
    if not response_str or not isinstance(response_str, str):
        return None
    
    response_str = response_str.strip()
    
    # Remove thinking blocks if present
    response_str = re.sub(r'<think>.*?</think>', '', response_str, flags=re.DOTALL)
    
    # Try to find the last JSON object in the response
    json_pattern = r'\{[^{}]*"winrate"[^{}]*\}'
    matches = re.findall(json_pattern, response_str, re.IGNORECASE)
    
    if matches:
        json_str = matches[-1]
        try:
            data = json.loads(json_str)
            winrate = data.get("winrate") or data.get("Winrate") or data.get("WINRATE")
            if winrate is not None:
                winrate = float(winrate)
                return max(0.0, min(1.0, winrate))
        except (json.JSONDecodeError, ValueError, TypeError):
            pass
    
    # Fallback: extract any float that looks like a winrate (0.XX)
    float_pattern = r'\b0\.\d+\b'
    matches = re.findall(float_pattern, response_str)
    if matches:
        try:
            winrate = float(matches[-1])
            if 0.0 <= winrate <= 1.0:
                return winrate
        except ValueError:
            pass
    
    # Also try percentage format (e.g., "56%" -> 0.56)
    percent_pattern = r'(\d+(?:\.\d+)?)\s*%'
    matches = re.findall(percent_pattern, response_str)
    if matches:
        try:
            percent = float(matches[-1])
            if 0 <= percent <= 100:
                return percent / 100.0
        except ValueError:
            pass
    
    return None


def get_ground_truth_from_api(katago_query: str) -> Optional[float]:
    """
    Call KataGo engine to get the ground truth winrate (fallback only).
    
    Args:
        katago_query: JSON string containing the KataGo analysis query
    
    Returns:
        Ground truth winrate for the current player, or None on error
    """
    if not ENABLE_API_FALLBACK:
        return None
    
    try:
        response = requests.post(
            KATAGO_ENDPOINT,
            data=katago_query,
            headers={"Content-Type": "application/json"},
            timeout=REQUEST_TIMEOUT
        )
        response.raise_for_status()
        
        result = json.loads(response.text.strip())
        winrate = result.get("rootInfo", {}).get("winrate")
        
        if winrate is not None:
            return float(winrate)
        return None
        
    except Exception as e:
        # Silently fail - this is just a fallback
        return None


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: Optional[dict] = None
) -> float:
    """
    Compute reward for winrate prediction.
    
    This is the main entry point called by veRL's RewardManager.
    
    Args:
        data_source: Dataset identifier (e.g., "katago/winrate")
        solution_str: Model's generated response
        ground_truth: Pre-computed ground truth winrate (from parquet)
        extra_info: Contains fallback data (katago_query, winrate)
    
    Returns:
        float: Reward value (-MSE):
            - 0.0: Perfect prediction
            - -0.01: Off by 0.1
            - -0.25: Off by 0.5
            - -1.0: Maximum error or invalid format
    """
    FORMAT_ERROR_PENALTY = -1.0
    ENGINE_ERROR_PENALTY = -0.25
    
    # 1. Extract predicted winrate from model response
    predicted_winrate = extract_winrate_from_response(solution_str)
    
    if predicted_winrate is None:
        return FORMAT_ERROR_PENALTY
    
    # 2. Get ground truth winrate (try multiple sources)
    ground_truth_winrate = None
    
    # Source 1: From ground_truth parameter (stored in parquet)
    if ground_truth and ground_truth.strip():
        try:
            ground_truth_winrate = float(ground_truth)
        except ValueError:
            pass
    
    # Source 2: From extra_info.root_winrate (or legacy 'winrate')
    if ground_truth_winrate is None and extra_info:
        winrate = extra_info.get("root_winrate") or extra_info.get("winrate")
        if winrate is not None:
            try:
                ground_truth_winrate = float(winrate)
            except (ValueError, TypeError):
                pass
    
    # Source 3: From KataGo API (fallback, slower)
    if ground_truth_winrate is None and extra_info:
        katago_query = extra_info.get("katago_query", "")
        if katago_query:
            ground_truth_winrate = get_ground_truth_from_api(katago_query)
    
    # If we still don't have ground truth, return penalty
    if ground_truth_winrate is None:
        return ENGINE_ERROR_PENALTY
    
    # 3. Compute -MSE reward
    mse = (predicted_winrate - ground_truth_winrate) ** 2
    reward = -mse
    
    return reward


# For testing
if __name__ == "__main__":
    print("Testing reward function with pre-computed ground truth:\n")
    
    test_cases = [
        # (model_response, ground_truth, expected_approx_reward)
        ('{"winrate": 0.56}', "0.56", 0.0),
        ('{"winrate": 0.50}', "0.56", -0.0036),
        ('{"winrate": 0.60}', "0.50", -0.01),
        ('The position is balanced. {"winrate": 0.55}', "0.55", 0.0),
        ('<think>Let me analyze...</think>{"winrate": 0.48}', "0.50", -0.0004),
        ('The winrate is approximately 52%.', "0.52", 0.0),
        ('Invalid response', "0.50", -1.0),
    ]
    
    print(f"{'Response':<50} {'GT':>6} {'Pred':>6} {'Reward':>8}")
    print("-" * 75)
    
    for response, gt, expected in test_cases:
        score = compute_score(
            data_source="katago/winrate",
            solution_str=response,
            ground_truth=gt,
            extra_info={}
        )
        pred = extract_winrate_from_response(response)
        pred_str = f"{pred:.2f}" if pred else "None"
        print(f"{response[:48]:<50} {gt:>6} {pred_str:>6} {score:>8.4f}")
    
    print("\nAll tests use pre-computed ground truth (no API calls).")
