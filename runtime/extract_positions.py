#!/usr/bin/env python3
"""
Extract KataGo positions from SGF files to JSONL format.

Two modes:
1. --filter: Remove rectangular boards from input directory (destructive, run once)
2. --extract: Extract positions to JSONL (skips rectangular boards automatically)

Output structure (grouped by board size only):
    output_dir/
    ├── 9x9/
    │   ├── train.jsonl
    │   └── test.jsonl
    ├── 13x13/
    │   ├── train.jsonl
    │   └── test.jsonl
    └── 19x19/
        ├── train.jsonl
        └── test.jsonl
"""

import argparse
import json
import random
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from tqdm import tqdm
from sgfmill import sgf


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class GameProperties:
    """Game properties for grouping."""
    board_size: int
    rules: str
    komi: float
    
    def to_dirname(self) -> str:
        return f"{self.board_size}x{self.board_size}_{self.rules}_{self.komi}"
    
    def __hash__(self):
        return hash((self.board_size, self.rules, self.komi))
    
    def __eq__(self, other):
        if not isinstance(other, GameProperties):
            return False
        return (self.board_size == other.board_size and 
                self.rules == other.rules and 
                self.komi == other.komi)


# =============================================================================
# SGF Utilities
# =============================================================================

def is_rectangular_board(sgf_path: Path) -> bool:
    """Check if SGF file has rectangular board (not supported by sgfmill)."""
    with open(sgf_path, 'rb') as f:
        content = f.read()
    return re.search(rb'SZ\[(\d+):(\d+)\]', content) is not None


def sgfmill_coord_to_katago(row: int, col: int, board_size: int) -> str:
    """
    Convert sgfmill coordinates to KataGo format.
    
    sgfmill: (row, col) where row 0 is top-left
    KataGo: "A1" where A1 is bottom-left, skipping 'I'
    """
    col_char = chr(ord('A') + col)
    if col_char >= 'I':
        col_char = chr(ord(col_char) + 1)
    katago_row = board_size - row
    return f"{col_char}{katago_row}"


def is_evaluation_comment(comment: str) -> bool:
    """Check if comment is a KataGo evaluation (not metadata)."""
    if not comment or not comment.strip():
        return False
    if 'gameHash=' in comment or 'startTurnIdx=' in comment:
        return False
    parts = comment.strip().split()
    if len(parts) < 6:
        return False
    first_part = parts[0]
    return first_part.replace('.', '').isdigit()


# =============================================================================
# Step 1: Filter Rectangular Boards
# =============================================================================

def discover_sgf_files(input_dir: Path) -> List[Path]:
    """Find all SGF files in directory."""
    files = list(input_dir.rglob('*.sgf'))
    assert len(files) > 0, f"No SGF files found in {input_dir}"
    return files


def filter_rectangular_boards(input_dir: Path) -> Tuple[int, int]:
    """
    Remove all rectangular board SGF files from input directory.
    
    This is a destructive operation - files are permanently deleted.
    
    Returns:
        (total_files, deleted_count)
    """
    sgf_files = discover_sgf_files(input_dir)
    total = len(sgf_files)
    print(f"Found {total} SGF files in {input_dir}")
    
    deleted = 0
    for sgf_path in tqdm(sgf_files, desc="Filtering rectangular boards"):
        if is_rectangular_board(sgf_path):
            sgf_path.unlink()
            deleted += 1
    
    print(f"Deleted {deleted} rectangular board files")
    print(f"Remaining: {total - deleted} files")
    
    return total, deleted


# =============================================================================
# Step 2: Parse SGF File
# =============================================================================

def parse_sgf_file(sgf_path: Path) -> Tuple[List[dict], GameProperties]:
    """
    Parse SGF file and extract positions as KataGo queries.
    
    Preconditions:
        - File must be a square board SGF (run filter step first)
        - File must have KM (komi) and RU (rules) properties
    
    Returns:
        (list of KataGo query dicts, GameProperties)
    """
    with open(sgf_path, 'rb') as f:
        sgf_content = f.read()
    
    game = sgf.Sgf_game.from_bytes(sgf_content)
    root = game.get_root()
    board_size = game.get_size()
    
    komi_val = root.get('KM')
    assert komi_val is not None, f"Missing KM (komi) in {sgf_path}"
    komi = float(komi_val)
    
    rules_val = root.get('RU')
    assert rules_val is not None, f"Missing RU (rules) in {sgf_path}"
    rules = str(rules_val)
    
    game_props = GameProperties(board_size=board_size, rules=rules, komi=komi)
    game_id = sgf_path.stem
    
    # Get all moves from main sequence
    main_sequence = game.get_main_sequence()
    all_moves = []
    
    for node in main_sequence:
        color, move = node.get_move()
        if color is None or move is None:
            continue
        
        row, col = move
        katago_coord = sgfmill_coord_to_katago(row, col, board_size)
        color_str = 'B' if color == 'b' else 'W'
        comment = node.get('C') if 'C' in node.properties() else None
        
        all_moves.append({
            'color': color_str,
            'coord': katago_coord,
            'comment': str(comment) if comment else None
        })
    
    # Extract positions with evaluations
    queries = []
    for i, move_info in enumerate(all_moves):
        comment = move_info['comment']
        if comment is None or not is_evaluation_comment(comment):
            continue
        
        moves_so_far = [[m['color'], m['coord']] for m in all_moves[:i+1]]
        position_id = f"{game_id}_m{i+1}"
        
        query = {
            'id': position_id,
            'moves': moves_so_far,
            'rules': rules,
            'komi': komi,
            'boardXSize': board_size,
            'boardYSize': board_size,
            'analyzeTurns': [len(moves_so_far)]
        }
        queries.append(query)
    
    return queries, game_props


# =============================================================================
# Step 3: Extract All Positions
# =============================================================================

def extract_positions(sgf_files: List[Path]) -> Dict[int, List[dict]]:
    """
    Extract positions from all SGF files, grouped by board size.
    
    Automatically skips:
    - Rectangular board files
    - Files with no evaluation comments
    
    Returns:
        Dict mapping board_size (int) to list of KataGo queries
    """
    queries_by_size: Dict[int, List[dict]] = defaultdict(list)
    skipped_no_evals = 0
    skipped_rectangular = 0
    
    for sgf_file in tqdm(sgf_files, desc="Extracting positions"):
        # Skip rectangular boards
        if is_rectangular_board(sgf_file):
            skipped_rectangular += 1
            continue
        
        queries, game_props = parse_sgf_file(sgf_file)
        
        if len(queries) == 0:
            skipped_no_evals += 1
            continue
        queries_by_size[game_props.board_size].extend(queries)
    
    if skipped_rectangular > 0:
        print(f"Skipped {skipped_rectangular} rectangular board files")
    
    if skipped_no_evals > 0:
        print(f"Skipped {skipped_no_evals} files with no evaluations")
    
    return queries_by_size


# =============================================================================
# Step 4: Save JSONL Files
# =============================================================================

def save_jsonl_files(
    queries_by_size: Dict[int, List[dict]],
    output_dir: Path,
    train_ratio: float
) -> int:
    """
    Save queries to JSONL files, split into train/test.
    
    Output structure:
        output_dir/<boardsize>/train.jsonl
        output_dir/<boardsize>/test.jsonl
    
    Returns:
        Total number of positions saved
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    total_positions = 0
    
    for board_size, queries in sorted(queries_by_size.items()):
        dirname = f"{board_size}x{board_size}"
        group_dir = output_dir / dirname
        group_dir.mkdir(parents=True, exist_ok=True)
        
        random.shuffle(queries)
        split_idx = int(len(queries) * train_ratio)
        train_queries = queries[:split_idx]
        test_queries = queries[split_idx:]
        
        with open(group_dir / 'train.jsonl', 'w') as f:
            for q in train_queries:
                f.write(json.dumps(q) + '\n')
        
        with open(group_dir / 'test.jsonl', 'w') as f:
            for q in test_queries:
                f.write(json.dumps(q) + '\n')
        
        print(f"  {dirname}: {len(train_queries)} train, {len(test_queries)} test")
        total_positions += len(queries)
    
    return total_positions


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Extract KataGo positions from SGF files to JSONL format.'
    )
    parser.add_argument(
        '--input-dir', '-i',
        type=Path,
        required=True,
        help='Directory containing SGF files (searches recursively)'
    )
    parser.add_argument(
        '--output-dir', '-o',
        type=Path,
        help='Output directory for JSONL files (required for extraction)'
    )
    parser.add_argument(
        '--filter',
        action='store_true',
        help='Filter mode: permanently delete rectangular board SGFs from input-dir'
    )
    parser.add_argument(
        '--train-ratio',
        type=float,
        default=0.9,
        help='Ratio of data for training (default: 0.9)'
    )
    parser.add_argument(
        '--max-games',
        type=int,
        default=None,
        help='Maximum number of games to process'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for shuffling (default: 42)'
    )
    
    return parser.parse_args()


def run_filter(args: argparse.Namespace) -> int:
    """Run filter mode: remove rectangular boards."""
    print(f"Filter mode: removing rectangular boards from {args.input_dir}")
    total, deleted = filter_rectangular_boards(args.input_dir)
    return 0


def run_extract(args: argparse.Namespace) -> int:
    """Run extraction mode: convert SGFs to JSONL."""
    assert args.output_dir is not None, "--output-dir required for extraction"
    
    random.seed(args.seed)
    
    # Discover SGF files
    sgf_files = discover_sgf_files(args.input_dir)
    print(f"Found {len(sgf_files)} SGF files")
    
    # Sample if requested
    if args.max_games and len(sgf_files) > args.max_games:
        sgf_files = random.sample(sgf_files, args.max_games)
        print(f"Sampled {args.max_games} games")
    
    # Extract positions grouped by board size
    queries_by_size = extract_positions(sgf_files)
    
    print(f"\nExtraction complete:")
    print(f"  Board sizes: {sorted(queries_by_size.keys())}")
    
    assert len(queries_by_size) > 0, "No positions extracted!"
    
    # Save JSONL files
    print(f"\nSaving to {args.output_dir}:")
    total = save_jsonl_files(queries_by_size, args.output_dir, args.train_ratio)
    
    print(f"\nTotal positions: {total}")
    return 0


def main() -> int:
    args = parse_args()
    
    if args.filter:
        return run_filter(args)
    else:
        return run_extract(args)


if __name__ == '__main__':
    sys.exit(main())

# =============================================================================
# Usage Examples
# =============================================================================
#
# # Step 1: Filter rectangular boards (run once, destructive)
# python runtime/extract_positions.py \
#     --input-dir ./raw_sgfs \
#     --filter
#
# # Step 2: Extract positions to JSONL
# python runtime/extract_positions.py \
#     --input-dir ./raw_sgfs \
#     --output-dir ./data \
#     --train-ratio 0.9
#
# # With sampling
# python runtime/extract_positions.py \
#     --input-dir ./raw_sgfs \
#     --output-dir ./data \
#     --max-games 1000
