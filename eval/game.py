"""Game logic for Go evaluation.

Contains async game playing logic and SGF generation.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

from eval.players.base import Player, validate_gtp_move


# Go rules to test
GO_RULES = [
    ("Japanese", "koSIMPLEscoreTERRITORYtaxSEKIsui0"),
    ("Chinese", "koSIMPLEscoreAREAtaxNONEsui0whbN"),
    ("Korean", "koSIMPLEscoreTERRITORYtaxNONEsui0"),
    ("AGA", "koSITUATIONALscoreAREAtaxNONEsui0whbN"),
    ("NZ", "koSITUATIONALscoreAREAtaxNONEsui0"),
    ("Tromp-Taylor", "koSITUATIONALscoreAREAtaxALLsui0"),
    ("Stone Scoring", "koSIMPLEscoreAREAtaxALLsui1"),
    ("Ancient", "koSIMPLEscoreAREAtaxNONEsui0"),
]

KOMI_VALUES = [5.5, 6.5, 7.5]
CANDIDATE_COLORS = ["B", "W"]


@dataclass
class GameResult:
    """Result of a single game."""
    game_id: int
    rule_name: str
    rule_string: str
    komi: float
    candidate_color: str
    winner: str  # "B", "W", or "draw"
    win_reason: str  # "resign", "score", "forfeit", "timeout"
    move_count: int
    candidate_won: bool
    sgf: str
    moves: List[List[str]]


def gtp_to_sgf(gtp_move: str) -> str:
    """Convert GTP move to SGF format."""
    if not gtp_move or gtp_move.upper() in ("PASS", "RESIGN"):
        return ""
    
    gtp_move = gtp_move.upper()
    col_letter = gtp_move[0]
    row_num = int(gtp_move[1:])
    
    # GTP columns: A-T (no I), SGF columns: a-s
    gtp_cols = "ABCDEFGHJKLMNOPQRST"
    col_idx = gtp_cols.index(col_letter)
    
    # SGF: a=1, s=19, but row is inverted
    sgf_col = chr(ord('a') + col_idx)
    sgf_row = chr(ord('a') + (19 - row_num))
    
    return sgf_col + sgf_row


def build_sgf(
    moves: List[List[str]],
    rule_name: str,
    komi: float,
    result: str,
    black_player: str = "Unknown",
    white_player: str = "Unknown"
) -> str:
    """Build SGF string from move list."""
    header = (
        f"(;GM[1]FF[4]CA[UTF-8]"
        f"SZ[19]KM[{komi}]"
        f"RU[{rule_name}]"
        f"PB[{black_player}]PW[{white_player}]"
        f"RE[{result}]"
    )
    
    move_str = ""
    for color, move in moves:
        sgf_move = gtp_to_sgf(move)
        if sgf_move:
            move_str += f";{color}[{sgf_move}]"
        elif move.upper() == "PASS":
            move_str += f";{color}[]"
    
    return header + move_str + ")"


def generate_game_variations(games_per_level: int) -> List[Tuple[str, str, float, str]]:
    """Generate game variations covering rules, komi, and colors.
    
    Returns list of (rule_name, rule_string, komi, candidate_color) tuples.
    """
    variations = []
    
    for rule_name, rule_string in GO_RULES:
        for komi in KOMI_VALUES:
            for color in CANDIDATE_COLORS:
                variations.append((rule_name, rule_string, komi, color))
    
    # Repeat to fill games_per_level
    result = []
    for i in range(games_per_level):
        result.append(variations[i % len(variations)])
    
    return result


async def play_single_game(
    game_id: int,
    candidate: Player,
    reference: Player,
    rule_name: str,
    rule_string: str,
    komi: float,
    candidate_color: str,
    max_moves: int = 500,
    verbose: bool = False
) -> GameResult:
    """Play a single game between candidate and reference.
    
    Both players are queried via async HTTP. The full move history is passed
    to each player on every move (stateless).
    
    Args:
        game_id: Game identifier
        candidate: Candidate player (being evaluated)
        reference: Reference player (KataGo)
        rule_name: Human-readable rule name
        rule_string: KataGo rule string
        komi: Komi value
        candidate_color: Color the candidate plays ("B" or "W")
        max_moves: Maximum moves before draw
        verbose: Print move-by-move output
    
    Returns:
        GameResult with outcome and SGF
    """
    reference_color = "W" if candidate_color == "B" else "B"
    moves: List[List[str]] = []
    
    winner = ""
    win_reason = ""
    
    # Determine who plays first (Black always starts)
    current_color = "B"
    
    for move_num in range(max_moves):
        # Determine whose turn
        is_candidate_turn = (current_color == candidate_color)
        player = candidate if is_candidate_turn else reference
        
        # Get win rate from reference's perspective (before candidate's move)
        win_rate = None
        if is_candidate_turn and candidate.requires_logging:
            win_rate = await reference.get_win_rate(moves, rule_string, komi, reference_color)
        
        # Get move
        move = await player.get_move(moves, rule_string, komi, current_color, win_rate)
        
        if verbose:
            player_name = "candidate" if is_candidate_turn else "reference"
            print(f"    {move_num + 1}. {current_color}: {move} ({player_name})")
        
        # Check for invalid move
        if not move or not validate_gtp_move(move):
            # Forfeit
            winner = reference_color if is_candidate_turn else candidate_color
            win_reason = "forfeit"
            break
        
        # Check for resign
        if move.upper() == "RESIGN":
            winner = reference_color if is_candidate_turn else candidate_color
            win_reason = "resign"
            break
        
        # Record move
        moves.append([current_color, move])
        
        # Check for consecutive passes (game end)
        if len(moves) >= 2:
            if moves[-1][1].upper() == "PASS" and moves[-2][1].upper() == "PASS":
                # Game ends - would need scoring, but we'll call it a draw for now
                # In practice, KataGo should resign when losing
                winner = "draw"
                win_reason = "passes"
                break
        
        # Switch color
        current_color = "W" if current_color == "B" else "B"
    
    else:
        # Max moves reached
        winner = "draw"
        win_reason = "max_moves"
    
    # Determine result string for SGF
    if winner == "draw":
        result_str = "0"
    elif win_reason == "resign":
        result_str = f"{winner}+R"
    elif win_reason == "forfeit":
        result_str = f"{winner}+F"
    else:
        result_str = f"{winner}+?"
    
    # Build SGF
    black_player = candidate.name if candidate_color == "B" else reference.name
    white_player = reference.name if candidate_color == "B" else candidate.name
    
    sgf = build_sgf(moves, rule_name, komi, result_str, black_player, white_player)
    
    # Determine if candidate won
    candidate_won = (winner == candidate_color)
    
    return GameResult(
        game_id=game_id,
        rule_name=rule_name,
        rule_string=rule_string,
        komi=komi,
        candidate_color=candidate_color,
        winner=winner,
        win_reason=win_reason,
        move_count=len(moves),
        candidate_won=candidate_won,
        sgf=sgf,
        moves=moves
    )
