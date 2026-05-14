#!/usr/bin/env python3
"""
Preprocess position and move JSONL files into parquet format for veRL training.

Combines position_dir and move_dir JSONL files, matching by id, and creates
parquet files with prompts and ground truth data ready for training.

Usage:
    python train/preprocess.py [--limit N]
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Retry suffix that analyze_offline.py adds to query ids when retrying after a
# timeout (to avoid HTTP 409 collisions with the still-pending first attempt).
# The suffix may end up persisted in the moves jsonl when the response from a
# retry attempt is the one that succeeds; we strip it here for id matching.
_RETRY_SUFFIX_RE = re.compile(r'_r\d+$')

import pandas as pd
import yaml

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

REPO_ROOT = Path(__file__).parent.parent.resolve()
CONFIG_PATH = REPO_ROOT / "configs" / "config.yaml"
PROMPTS_PATH = REPO_ROOT / "configs" / "prompts.json"


def load_config() -> dict:
    """Load configuration from config.yaml."""
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def load_prompts() -> dict:
    """Load prompt templates from prompts.json."""
    with open(PROMPTS_PATH) as f:
        return json.load(f)


def format_move_history(moves: List[List[str]]) -> str:
    """Format move history as a readable string."""
    if not moves:
        return "No moves yet"
    
    formatted = []
    for i, (color, coord) in enumerate(moves, start=1):
        formatted.append(f"{i}. {color} {coord}")
    
    return "\n".join(formatted)


def determine_current_player(moves: List[List[str]], root_info: Optional[dict]) -> str:
    """Determine current player from moves or rootInfo."""
    if root_info and "currentPlayer" in root_info:
        return root_info["currentPlayer"]
    
    # If no moves, Black starts
    if not moves:
        return "B"
    
    # Last move determines next player
    last_color = moves[-1][0]
    return "W" if last_color == "B" else "B"


def extract_move_data(move_data: dict) -> tuple:
    """
    Extract root_winrate, top_move, and error from move data.
    
    Returns:
        (root_winrate, top_move, move_error)
    """
    # Check for error response
    if "error" in move_data:
        error_msg = move_data.get("error", "Unknown error")
        return (None, None, error_msg)
    
    # Extract root winrate
    root_info = move_data.get("rootInfo", {})
    root_winrate = root_info.get("winrate")
    
    # Extract top move (first move in moveInfos by order)
    move_infos = move_data.get("moveInfos", [])
    top_move = None
    if move_infos and len(move_infos) > 0:
        top_move = move_infos[0].get("move")
    
    return (root_winrate, top_move, None)


def build_prompt(
    position_data: dict,
    prompt_template: dict,
    current_player: str
) -> List[dict]:
    """
    Build prompt messages from position data using template.
    
    Returns:
        List of message dicts: [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]
    """
    template_str = prompt_template["template"]
    system_msg = prompt_template.get("system", "")
    
    # Format move history
    moves = position_data.get("moves", [])
    move_history_str = format_move_history(moves)
    
    # Get board size
    board_size = position_data.get("boardXSize", 19)
    board_size_str = f"{board_size}x{board_size}"
    
    # Format template variables
    user_content = template_str.format(
        current_player=current_player,
        rule_string=position_data.get("rules", ""),
        komi=position_data.get("komi", 0.0),
        move_history=move_history_str,
        board_size=board_size_str
    )
    
    messages = []
    if system_msg:
        messages.append({"role": "system", "content": system_msg})
    messages.append({"role": "user", "content": user_content})
    
    return messages


def process_split(
    position_file: Path,
    move_file: Path,
    prompt_template: dict,
    split_name: str,
    limit: Optional[int] = None
) -> pd.DataFrame:
    """
    Process a single split (train or test) and return DataFrame.
    
    Args:
        position_file: Path to position JSONL file
        move_file: Path to move JSONL file
        prompt_template: Prompt template dict from prompts.json
        split_name: Name of split ("train" or "test")
    
    Returns:
        DataFrame with columns: prompt, extra_info, ground_truth
    """
    print(f"\nProcessing {split_name} split...")
    print(f"  Position file: {position_file}")
    print(f"  Move file: {move_file}")
    
    rows = []
    position_count = 0
    move_count = 0
    error_count = 0
    
    # Read both files line by line, matching by id
    # Note: We iterate based on move_file length since it's smaller (analysis results)
    with open(position_file, 'r') as pos_f, open(move_file, 'r') as move_f:
        pos_lines = iter(pos_f)
        move_lines = iter(move_f)
        
        for line_num in range(1, 1000000):  # Large upper bound
            # Apply limit if specified
            if limit is not None and len(rows) >= limit:
                break
            
            try:
                move_line = next(move_lines)
            except StopIteration:
                # End of move file
                break
            
            try:
                pos_line = next(pos_lines)
            except StopIteration:
                print(f"  WARNING: Position file ended before move file at line {line_num}")
                break
            try:
                position_data = json.loads(pos_line.strip())
                move_data = json.loads(move_line.strip())
                
                # Verify id match — tolerate a "_r<N>" retry suffix on move_id
                # left behind when a retried query was the one that succeeded.
                pos_id = position_data.get("id")
                move_id = move_data.get("id")
                move_id_normalized = (_RETRY_SUFFIX_RE.sub('', move_id)
                                      if move_id else None)

                if pos_id != move_id_normalized:
                    print(f"  WARNING: Line {line_num}: ID mismatch: {pos_id} != {move_id}")
                    error_count += 1
                    continue
                
                # Extract move data
                root_winrate, top_move, move_error = extract_move_data(move_data)
                
                # Determine current player
                root_info = move_data.get("rootInfo", {})
                current_player = determine_current_player(
                    position_data.get("moves", []),
                    root_info
                )
                
                # Build prompt
                prompt_messages = build_prompt(position_data, prompt_template, current_player)
                
                # Build extra_info
                extra_info = {
                    "position_id": pos_id,
                    "root_winrate": root_winrate,
                    "top_move": top_move,
                    "move_error": move_error,
                    "rule_string": position_data.get("rules", ""),
                    "komi": position_data.get("komi", 0.0),
                    "current_player": current_player,
                    "move_history": position_data.get("moves", [])
                }
                
                # Ground truth (string representation of root_winrate)
                ground_truth = str(root_winrate) if root_winrate is not None else ""
                
                # veRL naive reward manager expects reward_model and data_source
                reward_model = {"ground_truth": ground_truth}
                rows.append({
                    "prompt": prompt_messages,
                    "extra_info": extra_info,
                    "ground_truth": ground_truth,
                    "data_source": "katago/offline",
                    "reward_model": reward_model
                })
                
                position_count += 1
                move_count += 1
                
                if line_num % 10000 == 0:
                    print(f"  Processed {line_num} rows...")
                    
            except json.JSONDecodeError as e:
                print(f"  ERROR: Line {line_num}: JSON decode error: {e}")
                error_count += 1
                continue
            except Exception as e:
                print(f"  ERROR: Line {line_num}: Unexpected error: {e}")
                error_count += 1
                continue
    
    print(f"\n  Summary:")
    print(f"    Position rows: {position_count}")
    print(f"    Move rows: {move_count}")
    print(f"    Errors: {error_count}")
    print(f"    Valid rows: {len(rows)}")
    
    if len(rows) == 0:
        raise ValueError(f"No valid rows processed for {split_name} split!")
    
    return pd.DataFrame(rows)


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Preprocess KataGo training data")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of rows to process (for testing)"
    )
    args = parser.parse_args()
    
    print("=" * 60)
    print("KataGo Offline Training Data Preprocessor")
    if args.limit:
        print(f"TEST MODE: Limiting to {args.limit} rows per split")
    print("=" * 60)
    
    # Load configuration - use data section as single source of truth
    config = load_config()
    data_cfg = config.get("data", {})
    
    # Use data section directly (already includes board size in path)
    position_dir = REPO_ROOT / data_cfg.get("position_dir", "data/positions/19x19")
    move_dir = REPO_ROOT / data_cfg.get("move_dir", "data/moves/19x19")
    output_dir = REPO_ROOT / data_cfg.get("preprocessed_dir", "data/RL/katago_offline/19x19")
    
    # File names are always train.jsonl and test.jsonl (no board size prefix needed)
    train_file = "train.jsonl"
    test_file = "test.jsonl"
    
    # Load prompt template
    prompts = load_prompts()
    prompt_template = prompts.get("go_move", {})
    
    if not prompt_template:
        print("ERROR: 'go_move' template not found in prompts.json")
        return 1
    
    # Resolve file paths (position_dir and move_dir already include board size)
    position_train_file = position_dir / train_file
    position_test_file = position_dir / test_file
    move_train_file = move_dir / train_file
    move_test_file = move_dir / test_file
    
    # Verify files exist
    for file_path, name in [
        (position_train_file, "Position train"),
        (position_test_file, "Position test"),
        (move_train_file, "Move train"),
        (move_test_file, "Move test")
    ]:
        if not file_path.exists():
            print(f"ERROR: {name} file not found: {file_path}")
            return 1
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process train split
    print("\n" + "=" * 60)
    train_df = process_split(
        position_train_file,
        move_train_file,
        prompt_template,
        "train",
        limit=args.limit
    )
    
    # Process test split
    print("\n" + "=" * 60)
    test_df = process_split(
        position_test_file,
        move_test_file,
        prompt_template,
        "test",
        limit=args.limit
    )
    
    # Save parquet files
    print("\n" + "=" * 60)
    print("Saving parquet files...")
    
    train_output = output_dir / "train.parquet"
    test_output = output_dir / "test.parquet"
    
    train_df.to_parquet(train_output, index=False)
    print(f"  Saved: {train_output} ({len(train_df)} rows)")
    
    test_df.to_parquet(test_output, index=False)
    print(f"  Saved: {test_output} ({len(test_df)} rows)")
    
    print("\n" + "=" * 60)
    print("Preprocessing complete!")
    print(f"  Output directory: {output_dir}")
    print(f"  Train rows: {len(train_df)}")
    print(f"  Test rows: {len(test_df)}")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
