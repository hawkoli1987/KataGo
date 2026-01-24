#!/usr/bin/env python3
"""
Convert KataGo SGF files to veRL parquet format for winrate prediction training.

Usage:
    python prepare_katago_data.py --input_dir ./assets/analysis/inputs/2025-01-20sgfs \
                                   --output_dir /scratch_aisg/SPEC-SF-AISG/data_yuli/RL/katago \
                                   --max_games 10000 \
                                   --positions_per_game 3 \
                                   --train_ratio 0.9
"""

import argparse
import json
import os
import re
import random
from pathlib import Path
from typing import Optional

import pandas as pd
from tqdm import tqdm


# System prompt for winrate estimation
SYSTEM_PROMPT = """You are a Go game position analyst. Given a game position in JSON format, estimate the current winrate for the player to move.

The position includes:
- moves: List of moves played so far, each as [color, coordinate] where color is "B" or "W"
- rules: The ruleset being used
- komi: The komi value
- boardXSize, boardYSize: Board dimensions

Output your answer as a JSON object with a single key "winrate" containing a float between 0 and 1, representing the probability that the current player will win.

Example output: {"winrate": 0.56}"""


def parse_sgf_moves(sgf_content: str) -> list:
    """Extract moves from SGF content."""
    moves = []
    # Match move patterns like ;B[dc] or ;W[dp]
    move_pattern = re.compile(r';([BW])\[([a-s]{2})\]')
    
    for match in move_pattern.finditer(sgf_content):
        color = match.group(1)
        coord = match.group(2)
        # Convert SGF coordinates to KataGo format (e.g., 'dc' -> 'D3')
        col = coord[0].upper()
        row = ord(coord[1]) - ord('a') + 1
        katago_coord = f"{col}{row}"
        moves.append([color, katago_coord])
    
    return moves


def parse_sgf_metadata(sgf_content: str) -> dict:
    """Extract metadata from SGF header."""
    metadata = {
        "boardXSize": 19,
        "boardYSize": 19,
        "komi": 7.5,
        "rules": "tromp-taylor"
    }
    
    # Parse board size
    size_match = re.search(r'SZ\[(\d+)\]', sgf_content)
    if size_match:
        size = int(size_match.group(1))
        metadata["boardXSize"] = size
        metadata["boardYSize"] = size
    
    # Parse komi
    komi_match = re.search(r'KM\[([0-9.]+)\]', sgf_content)
    if komi_match:
        metadata["komi"] = float(komi_match.group(1))
    
    # Parse rules (simplified)
    rules_match = re.search(r'RU\[([^\]]+)\]', sgf_content)
    if rules_match:
        rules_str = rules_match.group(1).lower()
        if 'chinese' in rules_str:
            metadata["rules"] = "chinese"
        elif 'japanese' in rules_str:
            metadata["rules"] = "japanese"
        else:
            metadata["rules"] = "tromp-taylor"
    
    return metadata


def parse_embedded_winrate(comment: str) -> Optional[float]:
    """
    Extract winrate from SGF comment like "0.39 0.61 0.00 -1.7 v=240 weight=0.00"
    The first value is Black winrate, second is White winrate.
    """
    if not comment:
        return None
    
    # Pattern: "0.39 0.61 0.00 -1.7 v=240 weight=0.00"
    parts = comment.strip().split()
    if len(parts) >= 2:
        try:
            black_winrate = float(parts[0])
            white_winrate = float(parts[1])
            # Sanity check
            if 0 <= black_winrate <= 1 and 0 <= white_winrate <= 1:
                return black_winrate  # Return Black's winrate
        except ValueError:
            pass
    return None


def parse_sgf_with_winrates(sgf_content: str) -> list:
    """
    Parse SGF and extract positions with embedded winrate evaluations.
    Returns list of (moves_so_far, current_player, winrate) tuples.
    """
    positions = []
    
    # Parse all moves with their comments
    # Pattern matches: ;B[dc]C[...] or ;W[dp]C[...]
    move_with_comment_pattern = re.compile(
        r';([BW])\[([a-s]{2})\](?:C\[([^\]]*)\])?'
    )
    
    all_moves = []
    for match in move_with_comment_pattern.finditer(sgf_content):
        color = match.group(1)
        coord = match.group(2)
        comment = match.group(3) if match.group(3) else ""
        
        # Convert coordinates
        col = coord[0].upper()
        row = ord(coord[1]) - ord('a') + 1
        katago_coord = f"{col}{row}"
        
        all_moves.append({
            "color": color,
            "coord": katago_coord,
            "comment": comment
        })
    
    # Extract positions with winrates
    for i, move_info in enumerate(all_moves):
        winrate = parse_embedded_winrate(move_info["comment"])
        if winrate is not None:
            # Build moves list up to this point (inclusive of this move)
            moves_so_far = [[m["color"], m["coord"]] for m in all_moves[:i+1]]
            # Next player to move
            current_player = "W" if move_info["color"] == "B" else "B"
            # Adjust winrate for current player's perspective
            if current_player == "W":
                winrate = 1.0 - winrate  # Convert to White's perspective
            
            positions.append({
                "moves": moves_so_far,
                "current_player": current_player,
                "winrate": winrate,
                "move_number": i + 1
            })
    
    return positions


def create_verl_record(
    position_id: str,
    moves: list,
    metadata: dict,
    current_player: str,
    move_number: int
) -> dict:
    """Create a veRL-compatible data record."""
    
    # Build the user prompt with position info
    position_data = {
        "moves": moves,
        "rules": metadata["rules"],
        "komi": metadata["komi"],
        "boardXSize": metadata["boardXSize"],
        "boardYSize": metadata["boardYSize"],
        "currentPlayer": current_player,
        "moveNumber": move_number
    }
    
    user_content = json.dumps(position_data, separators=(',', ':'))
    
    # Build the KataGo query for reward function
    katago_query = {
        "id": position_id,
        "moves": moves,
        "rules": metadata["rules"],
        "komi": metadata["komi"],
        "boardXSize": metadata["boardXSize"],
        "boardYSize": metadata["boardYSize"],
        "analyzeTurns": [len(moves)]  # Analyze the current position
    }
    
    record = {
        "data_source": "katago/winrate",
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content}
        ],
        "ability": "go_analysis",
        "reward_model": {
            "style": "rule",
            "ground_truth": ""  # Will be computed at runtime via API
        },
        "extra_info": {
            "katago_query": json.dumps(katago_query),
            "id": position_id,
            "move_number": move_number,
            "current_player": current_player
        }
    }
    
    return record


def process_sgf_file(
    sgf_path: Path,
    positions_per_game: int = 3,
    min_move: int = 20,
    max_move: int = 200
) -> list:
    """Process a single SGF file and extract training positions."""
    
    try:
        with open(sgf_path, 'r', encoding='utf-8', errors='ignore') as f:
            sgf_content = f.read()
    except Exception as e:
        print(f"Error reading {sgf_path}: {e}")
        return []
    
    # Parse metadata
    metadata = parse_sgf_metadata(sgf_content)
    
    # Only process 19x19 games
    if metadata["boardXSize"] != 19 or metadata["boardYSize"] != 19:
        return []
    
    # Parse positions with winrates
    positions = parse_sgf_with_winrates(sgf_content)
    
    # Filter by move range (mid-game positions are most interesting)
    valid_positions = [
        p for p in positions 
        if min_move <= p["move_number"] <= max_move
    ]
    
    if not valid_positions:
        return []
    
    # Sample positions
    if len(valid_positions) > positions_per_game:
        selected = random.sample(valid_positions, positions_per_game)
    else:
        selected = valid_positions
    
    # Create veRL records
    records = []
    game_id = sgf_path.stem
    
    for i, pos in enumerate(selected):
        position_id = f"{game_id}_m{pos['move_number']}"
        record = create_verl_record(
            position_id=position_id,
            moves=pos["moves"],
            metadata=metadata,
            current_player=pos["current_player"],
            move_number=pos["move_number"]
        )
        records.append(record)
    
    return records


def main():
    parser = argparse.ArgumentParser(
        description="Convert KataGo SGF files to veRL parquet format"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing SGF files"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for parquet files"
    )
    parser.add_argument(
        "--max_games",
        type=int,
        default=10000,
        help="Maximum number of games to process"
    )
    parser.add_argument(
        "--positions_per_game",
        type=int,
        default=3,
        help="Number of positions to sample per game"
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.9,
        help="Ratio of data for training (rest for validation)"
    )
    parser.add_argument(
        "--min_move",
        type=int,
        default=20,
        help="Minimum move number to sample"
    )
    parser.add_argument(
        "--max_move",
        type=int,
        default=200,
        help="Maximum move number to sample"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    args = parser.parse_args()
    random.seed(args.seed)
    
    # Find all SGF files
    input_path = Path(args.input_dir)
    sgf_files = list(input_path.rglob("*.sgf"))
    print(f"Found {len(sgf_files)} SGF files")
    
    if len(sgf_files) == 0:
        print("No SGF files found!")
        return
    
    # Limit number of games
    if len(sgf_files) > args.max_games:
        sgf_files = random.sample(sgf_files, args.max_games)
        print(f"Sampled {args.max_games} games")
    
    # Process SGF files
    all_records = []
    for sgf_file in tqdm(sgf_files, desc="Processing SGF files"):
        records = process_sgf_file(
            sgf_file,
            positions_per_game=args.positions_per_game,
            min_move=args.min_move,
            max_move=args.max_move
        )
        all_records.extend(records)
    
    print(f"Extracted {len(all_records)} positions")
    
    if len(all_records) == 0:
        print("No positions extracted!")
        return
    
    # Shuffle and split
    random.shuffle(all_records)
    split_idx = int(len(all_records) * args.train_ratio)
    train_records = all_records[:split_idx]
    val_records = all_records[split_idx:]
    
    print(f"Train: {len(train_records)}, Validation: {len(val_records)}")
    
    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save as parquet
    train_df = pd.DataFrame(train_records)
    val_df = pd.DataFrame(val_records)
    
    train_path = output_path / "train.parquet"
    val_path = output_path / "test.parquet"
    
    train_df.to_parquet(train_path, index=False)
    val_df.to_parquet(val_path, index=False)
    
    print(f"Saved train data to {train_path}")
    print(f"Saved validation data to {val_path}")
    
    # Print sample record
    print("\nSample record:")
    sample = train_records[0]
    print(f"  ID: {sample['extra_info']['id']}")
    print(f"  Move: {sample['extra_info']['move_number']}")
    print(f"  Current player: {sample['extra_info']['current_player']}")
    print(f"  Prompt length: {len(sample['prompt'][1]['content'])} chars")


if __name__ == "__main__":
    main()
