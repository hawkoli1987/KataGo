#!/usr/bin/env python3
"""
Convert KataGo SGF self-play files to veRL parquet format for winrate prediction training.

Uses sgfmill for SGF parsing. Splits output by game properties (board_size, rules, komi).

Usage:
    python3 prepare_katago_data.py --input_dir ./assets/analysis/inputs/2025-01-20sgfs \
                                   --output_dir /scratch_aisg/SPEC-SF-AISG/data_yuli/RL/katago \
                                   --train_ratio 0.9

Output structure:
    output_dir/
    ├── 19x19_tromp-taylor_7.5/
    │   ├── train.parquet
    │   └── test.parquet
    └── extraction_report.json

Requirements:
    pip install sgfmill pandas pyarrow tqdm
"""

import argparse
import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import pandas as pd
from tqdm import tqdm
from sgfmill import sgf


# System prompt for winrate estimation
SYSTEM_PROMPT = """You are a Go game position analyst. Given a game position in JSON format, estimate the current winrate for the player to move.

The position includes:
- moves: List of moves played so far, each as [color, coordinate] where color is "B" or "W"
- rules: The ruleset being used
- komi: The komi value
- boardXSize, boardYSize: Board dimensions

Output your answer as a JSON object with a single key "winrate" containing a float between 0 and 1, representing the probability that the current player will win.

Example output: {"winrate": 0.56}"""


@dataclass
class GameProperties:
    """Immutable game properties for grouping.
    
    rules: Full KataGo rule string (e.g., koPOSITIONALscoreTERRITORYtaxSEKIsui0)
    """
    board_size: int
    rules: str  # Full KataGo rule string
    komi: float
    
    def to_dirname(self) -> str:
        """Use full rule string for folder naming."""
        return f"{self.board_size}x{self.board_size}_{self.rules}_{self.komi}"
    
    def __hash__(self):
        return hash((self.board_size, self.rules, self.komi))
    
    def __eq__(self, other):
        if not isinstance(other, GameProperties):
            return False
        return (self.board_size == other.board_size and 
                self.rules == other.rules and 
                self.komi == other.komi)


@dataclass
class PositionEval:
    """Evaluation data for a position from KataGo self-play.
    
    Note: The winrate is recorded AFTER the move was played.
    current_player indicates who is to move NEXT.
    """
    move_number: int
    moves: List[List[str]]
    current_player: str
    black_winrate: float
    white_winrate: float
    draw_prob: float
    score_lead: float
    visits: int
    weight: float
    result: Optional[str] = None


def is_evaluation_comment(comment: str) -> bool:
    """
    Check if a comment is a KataGo evaluation (not metadata).
    
    Evaluation format: "0.39 0.61 0.00 -1.7 v=240 weight=0.00"
    Metadata format: "startTurnIdx=20,initTurnNum=0,gameHash=..."
    """
    if not comment or not comment.strip():
        return False
    
    # Metadata comments contain these markers
    if 'gameHash=' in comment or 'startTurnIdx=' in comment:
        return False
    
    # Evaluation comments start with a float (winrate)
    parts = comment.strip().split()
    if len(parts) < 6:
        return False
    
    # First part should be a float between 0 and 1
    first_part = parts[0]
    if not first_part.replace('.', '').isdigit():
        return False
    
    return True


def parse_eval_comment(comment: str) -> Dict[str, float]:
    """
    Parse KataGo self-play evaluation comment.
    
    Format: "0.39 0.61 0.00 -1.7 v=240 weight=0.00"
    Optional: "... result=W+4.5" on final position
    
    Caller must verify is_evaluation_comment() first.
    """
    parts = comment.strip().split()
    
    black_winrate = float(parts[0])
    white_winrate = float(parts[1])
    draw_prob = float(parts[2])
    score_lead = float(parts[3])
    
    assert 0 <= black_winrate <= 1, f"Invalid black_winrate: {black_winrate}"
    assert 0 <= white_winrate <= 1, f"Invalid white_winrate: {white_winrate}"
    
    visits = 0
    weight = 0.0
    result = None
    
    for part in parts[4:]:
        if part.startswith('v='):
            visits = int(part[2:])
        elif part.startswith('weight='):
            weight = float(part[7:])
        elif part.startswith('result='):
            result = part[7:]
    
    return {
        'black_winrate': black_winrate,
        'white_winrate': white_winrate,
        'draw_prob': draw_prob,
        'score_lead': score_lead,
        'visits': visits,
        'weight': weight,
        'result': result
    }


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


def is_rectangular_board(sgf_content: bytes) -> bool:
    """
    Check if SGF has rectangular board (e.g., SZ[14:13]).
    sgfmill doesn't support rectangular boards.
    """
    import re
    # Look for SZ[N:M] pattern (rectangular)
    match = re.search(rb'SZ\[(\d+):(\d+)\]', sgf_content)
    return match is not None


def parse_sgf_file(sgf_path: Path) -> Tuple[List[PositionEval], GameProperties]:
    """
    Parse a single SGF file using sgfmill.
    
    Uses get_main_sequence() since self-play games are linear (no variations).
    
    All errors propagate as exceptions (no silent failures).
    """
    with open(sgf_path, 'rb') as f:
        sgf_content = f.read()
    
    # Skip rectangular boards (sgfmill limitation)
    assert not is_rectangular_board(sgf_content), "Rectangular boards not supported"
    
    game = sgf.Sgf_game.from_bytes(sgf_content)
    root = game.get_root()
    board_size = game.get_size()
    
    # Extract komi - REQUIRED
    komi_val = root.get('KM')
    assert komi_val is not None, "Missing KM (komi) property"
    komi = float(komi_val)
    
    # Extract rules - REQUIRED (use full KataGo rule string)
    rules_val = root.get('RU')
    assert rules_val is not None, "Missing RU (rules) property"
    rules = str(rules_val)
    
    game_props = GameProperties(
        board_size=board_size,
        rules=rules,
        komi=komi
    )
    
    # Get main sequence (linear, no variations)
    main_sequence = game.get_main_sequence()
    
    positions = []
    all_moves = []
    
    for node in main_sequence:
        # Get move (root node has no move)
        color, move = node.get_move()
        if color is None or move is None:
            continue
        
        # Convert coordinates
        row, col = move
        katago_coord = sgfmill_coord_to_katago(row, col, board_size)
        color_str = 'B' if color == 'b' else 'W'
        
        # Get comment if exists
        comment = node.get('C') if 'C' in node.properties() else None
        
        all_moves.append({
            'color': color_str,
            'coord': katago_coord,
            'comment': str(comment) if comment else None
        })
    
    # Extract positions with evaluations
    for i, move_info in enumerate(all_moves):
        comment = move_info['comment']
        
        # Skip moves without evaluation comments
        if comment is None or not is_evaluation_comment(comment):
            continue
        
        eval_data = parse_eval_comment(comment)
        
        moves_so_far = [[m['color'], m['coord']] for m in all_moves[:i+1]]
        current_player = 'W' if move_info['color'] == 'B' else 'B'
        
        positions.append(PositionEval(
            move_number=i + 1,
            moves=moves_so_far,
            current_player=current_player,
            black_winrate=eval_data['black_winrate'],
            white_winrate=eval_data['white_winrate'],
            draw_prob=eval_data['draw_prob'],
            score_lead=eval_data['score_lead'],
            visits=eval_data['visits'],
            weight=eval_data['weight'],
            result=eval_data['result']
        ))
    
    # Return empty if no evaluations (e.g., game ended within opening book)
    return positions, game_props


def create_verl_record(
    position: PositionEval,
    game_id: str,
    game_props: GameProperties
) -> dict:
    """Create a veRL-compatible data record."""
    
    position_id = f"{game_id}_m{position.move_number}"
    
    # Position data for LLM prompt (uses moves for ko handling)
    position_data = {
        'moves': position.moves,
        'rules': game_props.rules,  # Full KataGo rule string
        'komi': game_props.komi,
        'boardXSize': game_props.board_size,
        'boardYSize': game_props.board_size,
        'currentPlayer': position.current_player,
        'moveNumber': position.move_number
    }
    
    user_content = json.dumps(position_data, separators=(',', ':'))
    
    # KataGo query for API verification (uses moves for ko/superko handling)
    # Note: KataGo accepts both short names and full rule strings
    katago_query = {
        'id': position_id,
        'moves': position.moves,
        'rules': game_props.rules,  # Full KataGo rule string
        'komi': game_props.komi,
        'boardXSize': game_props.board_size,
        'boardYSize': game_props.board_size,
        'analyzeTurns': [len(position.moves)]
    }
    
    # Winrate from current player's perspective
    if position.current_player == 'W':
        player_winrate = position.white_winrate
        player_score = -position.score_lead
    else:
        player_winrate = position.black_winrate
        player_score = position.score_lead
    
    record = {
        'data_source': 'katago/winrate',
        'prompt': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': user_content}
        ],
        'ability': 'go_analysis',
        'reward_model': {
            'style': 'rule',
            'ground_truth': str(player_winrate)
        },
        'extra_info': {
            'id': position_id,
            'move_number': position.move_number,
            'current_player': position.current_player,
            'root_winrate': player_winrate,  # Winrate AFTER the last move, for current player
            'black_winrate': position.black_winrate,
            'white_winrate': position.white_winrate,
            'score_lead': player_score,
            'score_lead_black': position.score_lead,
            'visits': position.visits,
            'weight': position.weight,
            'rules': game_props.rules,  # Full KataGo rule string
            'katago_query': json.dumps(katago_query),
        }
    }
    
    if position.result:
        record['extra_info']['result'] = position.result
    
    return record


@dataclass
class ExtractionStats:
    """Statistics from extraction process."""
    total_files: int
    skipped_rectangular: int
    skipped_no_evals: int
    processed_files: int
    total_positions: int
    filtered_by_weight: int


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Convert KataGo SGF files to veRL parquet format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Output is split by game properties (board_size, rules, komi).

Examples:
    python prepare_katago_data.py --input_dir ./sgfs --output_dir ./output
    python prepare_katago_data.py --input_dir ./sgfs --output_dir ./output --max_games 100
    python prepare_katago_data.py --input_dir ./sgfs --output_dir ./output --min_weight 0.01
"""
    )
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--max_games', type=int, default=None)
    parser.add_argument('--train_ratio', type=float, default=0.9)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--min_weight', type=float, default=0.0,
                        help='Minimum training weight (use 0.01 to match KataGo training)')
    return parser.parse_args()


def discover_sgf_files(input_dir: Path) -> List[Path]:
    """Find all SGF files and filter out rectangular boards."""
    all_files = list(input_dir.rglob('*.sgf'))
    print(f"Found {len(all_files)} SGF files")
    assert len(all_files) > 0, "No SGF files found!"
    
    def is_square_board(path: Path) -> bool:
        with open(path, 'rb') as f:
            return not is_rectangular_board(f.read())
    
    print("Filtering rectangular boards...")
    square_files = [f for f in tqdm(all_files, desc="Filtering") if is_square_board(f)]
    skipped = len(all_files) - len(square_files)
    print(f"Skipped {skipped} rectangular boards, {len(square_files)} remaining")
    
    return square_files


def extract_positions(
    sgf_files: List[Path],
    min_weight: float
) -> Tuple[Dict[GameProperties, List[dict]], ExtractionStats]:
    """Extract positions from SGF files and group by game properties."""
    records_by_props: Dict[GameProperties, List[dict]] = defaultdict(list)
    total_positions = 0
    filtered_by_weight = 0
    skipped_no_evals = 0
    
    for sgf_file in tqdm(sgf_files, desc="Processing SGF files"):
        positions, game_props = parse_sgf_file(sgf_file)
        
        # Skip files with no evaluations (e.g., game ended within opening book)
        if len(positions) == 0:
            skipped_no_evals += 1
            continue
        
        game_id = sgf_file.stem
        
        for pos in positions:
            if pos.weight < min_weight:
                filtered_by_weight += 1
                continue
            
            record = create_verl_record(pos, game_id, game_props)
            records_by_props[game_props].append(record)
            total_positions += 1
    
    stats = ExtractionStats(
        total_files=0,  # Will be set by caller
        skipped_rectangular=0,  # Will be set by caller
        skipped_no_evals=skipped_no_evals,
        processed_files=len(sgf_files) - skipped_no_evals,
        total_positions=total_positions,
        filtered_by_weight=filtered_by_weight
    )
    
    return records_by_props, stats


def save_parquet_files(
    records_by_props: Dict[GameProperties, List[dict]],
    output_dir: Path,
    train_ratio: float
) -> List[dict]:
    """Save records to parquet files, split by game properties."""
    output_dir.mkdir(parents=True, exist_ok=True)
    group_stats = []
    
    for game_props, records in records_by_props.items():
        dirname = game_props.to_dirname()
        group_dir = output_dir / dirname
        group_dir.mkdir(parents=True, exist_ok=True)
        
        random.shuffle(records)
        split_idx = int(len(records) * train_ratio)
        train_records = records[:split_idx]
        val_records = records[split_idx:]
        
        pd.DataFrame(train_records).to_parquet(group_dir / 'train.parquet', index=False)
        pd.DataFrame(val_records).to_parquet(group_dir / 'test.parquet', index=False)
        
        group_stats.append({
            'directory': dirname,
            'board_size': game_props.board_size,
            'rules': game_props.rules,
            'komi': game_props.komi,
            'train_count': len(train_records),
            'val_count': len(val_records)
        })
        print(f"  {dirname}: {len(train_records)} train, {len(val_records)} val")
    
    return group_stats


def save_report(
    output_dir: Path,
    args: argparse.Namespace,
    stats: ExtractionStats,
    group_stats: List[dict]
) -> Path:
    """Save extraction report as JSON."""
    report = {
        'input_dir': str(args.input_dir),
        'output_dir': str(args.output_dir),
        'args': vars(args),
        'stats': {
            'total_files': stats.total_files,
            'skipped_rectangular': stats.skipped_rectangular,
            'skipped_no_evals': stats.skipped_no_evals,
            'processed_files': stats.processed_files,
            'total_positions': stats.total_positions,
            'filtered_by_weight': stats.filtered_by_weight,
        },
        'groups': group_stats
    }
    
    report_path = output_dir / 'extraction_report.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    return report_path


def main():
    args = parse_args()
    random.seed(args.seed)
    
    # Discover and filter SGF files
    input_path = Path(args.input_dir)
    all_sgf_files = list(input_path.rglob('*.sgf'))
    sgf_files = discover_sgf_files(input_path)
    skipped_rectangular = len(all_sgf_files) - len(sgf_files)
    
    # Sample if requested
    if args.max_games and len(sgf_files) > args.max_games:
        sgf_files = random.sample(sgf_files, args.max_games)
        print(f"Sampled {args.max_games} games")
    
    # Extract positions
    records_by_props, stats = extract_positions(sgf_files, args.min_weight)
    stats.total_files = len(all_sgf_files)
    stats.skipped_rectangular = skipped_rectangular
    
    print(f"\nExtraction complete:")
    print(f"  Total files: {stats.total_files}")
    print(f"  Skipped (rectangular): {stats.skipped_rectangular}")
    print(f"  Skipped (no evaluations): {stats.skipped_no_evals}")
    print(f"  Processed: {stats.processed_files}")
    print(f"  Positions: {stats.total_positions}")
    print(f"  Filtered by weight: {stats.filtered_by_weight}")
    print(f"  Groups: {len(records_by_props)}")
    
    assert stats.total_positions > 0, "No positions extracted!"
    
    # Save parquet files
    output_path = Path(args.output_dir)
    group_stats = save_parquet_files(records_by_props, output_path, args.train_ratio)
    
    # Save report
    report_path = save_report(output_path, args, stats, group_stats)
    print(f"\nReport: {report_path}")
    
    return 0


if __name__ == '__main__':
    exit(main())
