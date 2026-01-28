"""Game logic for Go evaluation.

Handles game flow, SGF generation, and game result tracking.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from eval.players.base import Player, validate_gtp_move, GTP_COLS
from eval.players.katago_player import KataGoPlayer


# Game rule configurations (KataGo format)
RULE_CONFIGS = [
    ("Japanese", "koSIMPLEscoreTERRITORYtaxSEKIsui0"),
    ("Chinese", "koSIMPLEscoreAREAtaxNONEsui0whbN"),
    ("Korean", "koPOSITIONALscoreAREAtaxNONEsui0whbN"),
    ("AGA", "koSITUATIONALscoreAREAtaxNONEsui0whbN-1"),
    ("NewZealand", "koSITUATIONALscoreAREAtaxNONEsui1"),
    ("TrompTaylor", "koPOSITIONALscoreAREAtaxNONEsui1"),
    ("StoneScoring", "koSIMPLEscoreAREAtaxALLsui0"),
    ("AncientTerritory", "koSIMPLEscoreTERRITORYtaxALLsui0"),
]

KOMI_VALUES = [5.5, 6.5, 7.5]
SIDES = ["B", "W"]

# Total combinations = 8 rules × 3 komis × 2 sides = 48
TOTAL_COMBINATIONS = len(RULE_CONFIGS) * len(KOMI_VALUES) * len(SIDES)


@dataclass
class GameResult:
    """Result of a single game."""
    game_id: int
    rule_name: str
    rule_string: str
    komi: float
    candidate_color: str
    winner: str  # "B", "W", or "draw"
    win_reason: str  # "score", "resign", "forfeit", "timeout", "max_moves"
    candidate_won: bool
    move_count: int
    sgf: str
    timestamp: str


def gtp_to_sgf(move: str) -> str:
    """Convert GTP coordinate to SGF coordinate."""
    move = move.upper()
    if move in ("PASS", "RESIGN"):
        return ""
    
    col = move[0]
    row = int(move[1:])
    
    gtp_col_idx = GTP_COLS.index(col)
    sgf_col = chr(ord('a') + gtp_col_idx)
    sgf_row = chr(ord('a') + (19 - row))
    
    return sgf_col + sgf_row


def build_sgf(rule_name: str, komi: float, moves: list, result: str) -> str:
    """Build an SGF string from game data."""
    header = f"(;GM[1]FF[4]SZ[19]KM[{komi}]RU[{rule_name}]RE[{result}]"
    moves_str = "".join(moves)
    return header + moves_str + ")"


def generate_game_variations(games_per_level: int) -> list:
    """Generate game configuration variations.
    
    Returns list of (rule_name, rule_string, komi, candidate_color) tuples.
    """
    base_combinations = [
        (rule_name, rule_string, komi, side)
        for rule_name, rule_string in RULE_CONFIGS
        for komi in KOMI_VALUES
        for side in SIDES
    ]
    
    full_rounds = games_per_level // TOTAL_COMBINATIONS
    remainder = games_per_level % TOTAL_COMBINATIONS
    
    variations = base_combinations * full_rounds + base_combinations[:remainder]
    return variations


def play_single_game(
    game_id: int,
    candidate: Player,
    reference: KataGoPlayer,
    rule_name: str,
    rule_string: str,
    komi: float,
    candidate_color: str,
    max_moves: int = 500,
    verbose: bool = False
) -> GameResult:
    """Play a single game between candidate and reference.
    
    Args:
        game_id: Unique game identifier
        candidate: The candidate player being evaluated
        reference: KataGo reference player (also used for board state)
        rule_name: Human-readable rule name
        rule_string: KataGo rule string
        komi: Komi value
        candidate_color: Color for candidate ("B" or "W")
        max_moves: Maximum moves before declaring draw
        verbose: Print detailed output
    
    Returns:
        GameResult with game outcome
    """
    # Setup game on reference engine
    reference.clear_board()
    reference.set_boardsize(19)
    reference.set_komi(komi)
    reference.set_rules(rule_string)
    
    move_history = []
    sgf_moves = []
    current_color = "B"  # Black always starts
    winner = None
    win_reason = "score"
    
    reference_color = "W" if candidate_color == "B" else "B"
    
    for move_num in range(max_moves):
        if current_color == candidate_color:
            # Candidate's turn
            win_rate = None
            try:
                win_rate = reference.get_win_rate(current_color, visits=50)
            except Exception:
                pass
            
            move = candidate.get_move(
                move_history, rule_string, komi, current_color, win_rate=win_rate
            )
            
            if not move or not validate_gtp_move(move):
                if verbose:
                    print(f"  Candidate invalid move: '{move}'")
                winner = reference_color
                win_reason = "forfeit"
                break
            
            # Verify move is legal on reference board
            result = reference.play_move(current_color, move)
            if "illegal" in result.lower() or "?" in result:
                if verbose:
                    print(f"  Candidate illegal move: {move}")
                winner = reference_color
                win_reason = "forfeit"
                break
        else:
            # Reference's turn
            move = reference.genmove(current_color)
        
        if verbose:
            print(f"  Move {move_num + 1}: {current_color} {move}")
        
        # Record move
        move_history.append([current_color, move])
        sgf_moves.append(f";{current_color}[{gtp_to_sgf(move)}]")
        
        # Check for game end
        if move.upper() == "RESIGN":
            winner = "W" if current_color == "B" else "B"
            win_reason = "resign"
            break
        
        # Check for two consecutive passes
        if len(move_history) >= 2:
            if (move_history[-1][1].upper() == "PASS" and 
                move_history[-2][1].upper() == "PASS"):
                score_result = reference.final_score()
                if "B+" in score_result:
                    winner = "B"
                elif "W+" in score_result:
                    winner = "W"
                else:
                    winner = "draw"
                break
        
        # Switch color
        current_color = "W" if current_color == "B" else "B"
    
    # Max moves reached
    if winner is None:
        winner = "draw"
        win_reason = "max_moves"
    
    candidate_won = (winner == candidate_color)
    
    sgf = build_sgf(
        rule_name=rule_name,
        komi=komi,
        moves=sgf_moves,
        result=f"{winner}+{win_reason}" if winner != "draw" else "0"
    )
    
    return GameResult(
        game_id=game_id,
        rule_name=rule_name,
        rule_string=rule_string,
        komi=komi,
        candidate_color=candidate_color,
        winner=winner,
        win_reason=win_reason,
        candidate_won=candidate_won,
        move_count=len(move_history),
        sgf=sgf,
        timestamp=datetime.now().isoformat()
    )
