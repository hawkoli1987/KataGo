"""
Custom reward function for KataGo winrate prediction training.

This function computes -MSE (negative mean squared error) between the model's
predicted winrate and the ground truth from the KataGo analysis engine.

The reward function is called by veRL during training to score each generated response.
"""

import json
import re
import requests
from typing import Optional

# KataGo Analysis Engine endpoint
KATAGO_ENDPOINT = "http://hopper-34:9000/analysis"

# Request timeout in seconds
REQUEST_TIMEOUT = 30


def extract_winrate_from_response(response_str: str) -> Optional[float]:
    """
    Extract winrate value from model response.
    
    Handles various output formats:
    - Clean JSON: {"winrate": 0.56}
    - JSON with reasoning: "The position looks balanced... {"winrate": 0.52}"
    - Markdown code blocks: ```json\n{"winrate": 0.55}\n```
    """
    if not response_str or not isinstance(response_str, str):
        return None
    
    response_str = response_str.strip()
    
    # Try to find the last JSON object in the response
    # This handles cases where the model outputs reasoning before the JSON
    json_pattern = r'\{[^{}]*"winrate"[^{}]*\}'
    matches = re.findall(json_pattern, response_str, re.IGNORECASE)
    
    if matches:
        # Use the last match (most likely to be the final answer)
        json_str = matches[-1]
        try:
            data = json.loads(json_str)
            winrate = data.get("winrate") or data.get("Winrate") or data.get("WINRATE")
            if winrate is not None:
                winrate = float(winrate)
                # Clamp to valid range
                return max(0.0, min(1.0, winrate))
        except (json.JSONDecodeError, ValueError, TypeError):
            pass
    
    # Fallback: try to extract any float that looks like a winrate
    # Pattern: standalone decimal between 0 and 1
    float_pattern = r'\b0\.\d+\b'
    matches = re.findall(float_pattern, response_str)
    if matches:
        # Use the last match
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


def get_ground_truth_winrate(katago_query: str) -> Optional[float]:
    """
    Call KataGo engine to get the ground truth winrate.
    
    Args:
        katago_query: JSON string containing the KataGo analysis query
    
    Returns:
        Ground truth winrate for the current player, or None on error
    """
    try:
        response = requests.post(
            KATAGO_ENDPOINT,
            data=katago_query,
            headers={"Content-Type": "application/json"},
            timeout=REQUEST_TIMEOUT
        )
        response.raise_for_status()
        
        # Parse response
        result = json.loads(response.text.strip())
        
        # Extract winrate from rootInfo
        # Note: KataGo reports winrate for the player to move
        winrate = result.get("rootInfo", {}).get("winrate")
        
        if winrate is not None:
            return float(winrate)
        
        return None
        
    except requests.exceptions.Timeout:
        print(f"[KataGo Reward] Request timeout after {REQUEST_TIMEOUT}s")
        return None
    except requests.exceptions.RequestException as e:
        print(f"[KataGo Reward] Request error: {e}")
        return None
    except (json.JSONDecodeError, ValueError, KeyError) as e:
        print(f"[KataGo Reward] Parse error: {e}")
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
        ground_truth: Not used (we call the engine in real-time)
        extra_info: Contains katago_query for engine call
    
    Returns:
        float: Reward value. For -MSE:
            - 0.0: Perfect prediction
            - -0.25: Off by 0.5 (50% error)
            - -1.0: Maximum error (completely wrong)
    """
    # Penalty values
    FORMAT_ERROR_PENALTY = -1.0
    ENGINE_ERROR_PENALTY = -0.25
    
    # 1. Extract predicted winrate from model response
    predicted_winrate = extract_winrate_from_response(solution_str)
    
    if predicted_winrate is None:
        # Model failed to produce a valid winrate
        return FORMAT_ERROR_PENALTY
    
    # 2. Get ground truth from KataGo engine
    if extra_info is None:
        return ENGINE_ERROR_PENALTY
    
    katago_query = extra_info.get("katago_query", "")
    if not katago_query:
        return ENGINE_ERROR_PENALTY
    
    ground_truth_winrate = get_ground_truth_winrate(katago_query)
    
    if ground_truth_winrate is None:
        # Engine call failed - use neutral penalty
        return ENGINE_ERROR_PENALTY
    
    # 3. Compute -MSE reward
    mse = (predicted_winrate - ground_truth_winrate) ** 2
    reward = -mse
    
    return reward


# For testing
if __name__ == "__main__":
    # Test the reward function
    test_extra_info = {
        "katago_query": json.dumps({
            "id": "test",
            "moves": [["B", "D4"], ["W", "Q16"], ["B", "D16"]],
            "rules": "tromp-taylor",
            "komi": 7.5,
            "boardXSize": 19,
            "boardYSize": 19,
            "analyzeTurns": [3]
        })
    }
    
    # Test with different model outputs
    test_cases = [
        '{"winrate": 0.56}',
        'Based on my analysis, the position is slightly favorable for Black. {"winrate": 0.55}',
        'The winrate is approximately 52%.',
        '```json\n{"winrate": 0.48}\n```',
        'Invalid response without winrate',
    ]
    
    print("Testing reward function:")
    for i, response in enumerate(test_cases):
        score = compute_score(
            data_source="katago/winrate",
            solution_str=response,
            ground_truth="",
            extra_info=test_extra_info
        )
        extracted = extract_winrate_from_response(response)
        print(f"  Test {i+1}: extracted={extracted}, reward={score:.4f}")
