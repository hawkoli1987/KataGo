#!/usr/bin/env python3
"""Main evaluation script: ladder-style competition pipeline.

This script evaluates an LLM's Go-playing strength by having it play against
KataGo reference models in a ladder competition with Elo rating.

Usage:
    python3 eval/eval.py \
      --candidate-type openai \
      --candidate-endpoint http://localhost:8002/v1 \
      --model-name "qwen3-test" \
      --games-per-level 48 \
      --promotion-threshold 0.55
      
    Note: 
    - Use http:// (not https://) for local vLLM endpoints
    - --candidate-model is auto-detected from the vLLM API if not provided
"""

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from eval.llm_player import (
    LLMPlayer, OpenAICompatiblePlayer, HuggingFacePlayer,
    validate_gtp_move, GTP_COLS
)

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
SIDES = ["B", "W"]  # Candidate plays as Black or White

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
    win_reason: str  # "score", "resign", "forfeit", "timeout"
    candidate_won: bool
    move_count: int
    sgf: str
    timestamp: str


@dataclass
class LevelResult:
    """Result of playing games at one level."""
    level: int
    reference_model: str
    reference_elo: int
    games_played: int
    wins: int
    losses: int
    draws: int
    win_rate: float
    promoted: bool
    candidate_elo_after: float
    games: list = field(default_factory=list)


class KataGoGTP:
    """Wrapper for KataGo in GTP mode."""
    
    def __init__(self, katago_path: str, model_path: str, config_path: Optional[str] = None,
                 log_dir: str = "log"):
        self.katago_path = katago_path
        self.model_path = model_path
        self.config_path = config_path
        self.log_dir = log_dir
        self.process = None
        self._stderr_output = []
    
    def start(self):
        """Start KataGo process."""
        cmd = [self.katago_path, "gtp"]
        if self.model_path:
            cmd.extend(["-model", self.model_path])
        if self.config_path:
            cmd.extend(["-config", self.config_path])
        # Override log directory
        cmd.extend(["-override-config", f"logDir={self.log_dir}"])
        
        print(f"Starting KataGo: {' '.join(cmd)}")
        
        self.process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )
        
        # Wait a moment for startup and check if process is still running
        time.sleep(2)
        
        if self.process.poll() is not None:
            # Process exited - read stderr
            stderr = self.process.stderr.read()
            raise RuntimeError(f"KataGo failed to start (exit code {self.process.returncode}):\n{stderr}")
    
    def send_command(self, cmd: str) -> str:
        """Send a GTP command and return the response."""
        assert self.process is not None, "KataGo not started"
        
        self.process.stdin.write(cmd + "\n")
        self.process.stdin.flush()
        
        response_lines = []
        while True:
            line = self.process.stdout.readline()
            if line.strip() == "":
                if response_lines:
                    break
            else:
                response_lines.append(line.rstrip())
        
        return "\n".join(response_lines)
    
    def stop(self):
        """Stop KataGo process."""
        if self.process:
            try:
                if self.process.poll() is None:  # Still running
                    self.send_command("quit")
                    self.process.wait(timeout=5)
            except Exception:
                # Force kill if quit doesn't work
                try:
                    self.process.kill()
                    self.process.wait(timeout=2)
                except Exception:
                    pass
            finally:
                self.process = None
    
    def clear_board(self):
        """Clear the board."""
        return self.send_command("clear_board")
    
    def set_boardsize(self, size: int = 19):
        """Set board size."""
        return self.send_command(f"boardsize {size}")
    
    def set_komi(self, komi: float):
        """Set komi."""
        return self.send_command(f"komi {komi}")
    
    def set_rules(self, rules: str):
        """Set rules using kata-set-rules."""
        return self.send_command(f"kata-set-rules {rules}")
    
    def play(self, color: str, move: str):
        """Play a move."""
        return self.send_command(f"play {color} {move}")
    
    def genmove(self, color: str) -> str:
        """Generate a move for the given color."""
        response = self.send_command(f"genmove {color}")
        # Response format: "= D4" or "= pass" or "= resign"
        if response.startswith("="):
            return response[1:].strip().upper()
        return ""
    
    def showboard(self) -> str:
        """Show the current board state."""
        return self.send_command("showboard")
    
    def final_score(self) -> str:
        """Get final score."""
        return self.send_command("final_score")
    
    def get_win_rate(self, color: str, visits: int = 100) -> Optional[float]:
        """Get KataGo's win rate estimation for the specified color.
        
        Uses kata-analyze to get the root win rate from KataGo's perspective.
        
        Args:
            color: The color to get win rate for ("B" or "W")
            visits: Number of visits for analysis (higher = more accurate but slower)
        
        Returns:
            Win rate as float [0, 1], or None if analysis failed
        """
        # Send kata-analyze command with rootInfo
        cmd = f"kata-analyze {color} {visits} rootInfo true"
        self.process.stdin.write(cmd + "\n")
        self.process.stdin.flush()
        
        # Read until we get the analysis output
        response_lines = []
        max_wait = 30  # Maximum seconds to wait
        import time
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            line = self.process.stdout.readline()
            if line:
                response_lines.append(line.strip())
                # Look for rootInfo in the response
                if "rootInfo" in line:
                    # Send newline to stop the analysis
                    self.process.stdin.write("\n")
                    self.process.stdin.flush()
                    break
        
        # Parse the response to extract win rate
        full_response = " ".join(response_lines)
        
        # Extract winrate from rootInfo section
        # Format: ... rootInfo visits 100 winrate 0.523 ...
        import re
        match = re.search(r'rootInfo.*?winrate\s+([\d.]+)', full_response)
        if match:
            win_rate = float(match.group(1))
            return win_rate
        
        return None


def compute_elo_update(candidate_elo: float, opponent_elo: float, 
                       candidate_won: bool, k_factor: float = 32) -> float:
    """Compute new Elo rating after a game."""
    expected = 1 / (1 + 10 ** ((opponent_elo - candidate_elo) / 400))
    actual = 1.0 if candidate_won else 0.0
    new_elo = candidate_elo + k_factor * (actual - expected)
    return new_elo


def generate_game_variations(games_per_level: int) -> list:
    """Generate game configuration variations."""
    variations = []
    
    # Generate all 48 base combinations
    base_combinations = []
    for rule_name, rule_string in RULE_CONFIGS:
        for komi in KOMI_VALUES:
            for side in SIDES:
                base_combinations.append((rule_name, rule_string, komi, side))
    
    # Repeat to fill games_per_level
    full_rounds = games_per_level // TOTAL_COMBINATIONS
    remainder = games_per_level % TOTAL_COMBINATIONS
    
    for _ in range(full_rounds):
        variations.extend(base_combinations)
    
    # Add remainder (first N combinations)
    variations.extend(base_combinations[:remainder])
    
    return variations


def play_single_game(
    game_id: int,
    llm_player: LLMPlayer,
    katago: KataGoGTP,
    rule_name: str,
    rule_string: str,
    komi: float,
    candidate_color: str,
    max_moves: int = 500,
    verbose: bool = False
) -> GameResult:
    """Play a single game between LLM and KataGo."""
    # Setup game
    katago.clear_board()
    katago.set_boardsize(19)
    katago.set_komi(komi)
    katago.set_rules(rule_string)
    
    move_history = []
    sgf_moves = []
    current_color = "B"  # Black always starts
    winner = None
    win_reason = "score"
    
    katago_color = "W" if candidate_color == "B" else "B"
    
    for move_num in range(max_moves):
        if current_color == candidate_color:
            # LLM's turn
            # Get KataGo's win rate estimation for the candidate
            # Note: win_rate is from the candidate's perspective
            win_rate = None
            try:
                win_rate = katago.get_win_rate(current_color, visits=50)
            except Exception:
                pass  # Continue without win rate if analysis fails
            
            try:
                move = llm_player.get_move(move_history, rule_string, komi, current_color,
                                          win_rate=win_rate)
            except Exception as e:
                if verbose:
                    print(f"  LLM error: {e}")
                winner = katago_color
                win_reason = "forfeit"
                break
            
            if not move or not validate_gtp_move(move):
                if verbose:
                    print(f"  LLM invalid move: '{move}'")
                winner = katago_color
                win_reason = "forfeit"
                break
            
            # Check if it's a valid move on the board
            result = katago.play(current_color, move)
            if "illegal" in result.lower() or "?" in result:
                if verbose:
                    print(f"  LLM illegal move: {move}")
                winner = katago_color
                win_reason = "forfeit"
                break
        else:
            # KataGo's turn
            move = katago.genmove(current_color)
        
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
                # Game ended by passing
                score_result = katago.final_score()
                if "B+" in score_result:
                    winner = "B"
                elif "W+" in score_result:
                    winner = "W"
                else:
                    winner = "draw"
                break
        
        # Switch color
        current_color = "W" if current_color == "B" else "B"
    
    # If no winner determined, it's a draw (max moves reached)
    if winner is None:
        winner = "draw"
        win_reason = "max_moves"
    
    candidate_won = (winner == candidate_color)
    
    # Build SGF
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


def gtp_to_sgf(move: str) -> str:
    """Convert GTP coordinate to SGF coordinate."""
    move = move.upper()
    if move == "PASS":
        return ""
    if move == "RESIGN":
        return ""
    
    col = move[0]
    row = int(move[1:])
    
    # GTP column to SGF column
    gtp_col_idx = GTP_COLS.index(col)
    sgf_col = chr(ord('a') + gtp_col_idx)
    
    # GTP row (1-19 from bottom) to SGF row (a-s from top)
    sgf_row = chr(ord('a') + (19 - row))
    
    return sgf_col + sgf_row


def build_sgf(rule_name: str, komi: float, moves: list, result: str) -> str:
    """Build an SGF string from game data."""
    header = f"(;GM[1]FF[4]SZ[19]KM[{komi}]RU[{rule_name}]RE[{result}]"
    moves_str = "".join(moves)
    return header + moves_str + ")"


def test_llm_only(llm_player: LLMPlayer, num_positions: int = 5):
    """Test LLM player without KataGo."""
    print("\n" + "=" * 60)
    print("LLM-ONLY TEST MODE")
    print("Testing LLM player can generate valid moves...")
    print("=" * 60)
    
    test_cases = [
        {"move_history": [], "rules": "koSIMPLEscoreTERRITORYtaxSEKIsui0", "komi": 6.5, "color": "B"},
        {"move_history": [], "rules": "koPOSITIONALscoreAREAtaxNONEsui1", "komi": 7.5, "color": "B"},
        {"move_history": [["B", "D4"], ["W", "Q16"]], "rules": "koSIMPLEscoreAREAtaxNONEsui0whbN", "komi": 7.5, "color": "B"},
        {"move_history": [["B", "D4"], ["W", "Q16"], ["B", "D16"]], "rules": "koSITUATIONALscoreAREAtaxNONEsui0whbN-1", "komi": 7.5, "color": "W"},
        {"move_history": [["B", "D4"], ["W", "Q16"], ["B", "D16"], ["W", "Q4"], ["B", "C6"], ["W", "R14"]], 
         "rules": "koPOSITIONALscoreAREAtaxNONEsui1", "komi": 7.5, "color": "B"},
    ]
    
    valid_count = 0
    total_count = min(num_positions, len(test_cases))
    
    for i, tc in enumerate(test_cases[:num_positions]):
        print(f"\nTest {i+1}/{total_count}:")
        print(f"  Rules: {tc['rules']}")
        print(f"  Komi: {tc['komi']}")
        print(f"  Color: {tc['color']}")
        print(f"  Move history: {tc['move_history'][:3]}{'...' if len(tc['move_history']) > 3 else ''}")
        
        try:
            move = llm_player.get_move(**tc)
            valid = validate_gtp_move(move) if move else False
            print(f"  LLM move: {move}")
            print(f"  Valid: {valid}")
            if valid:
                valid_count += 1
        except Exception as e:
            print(f"  ERROR: {e}")
    
    print(f"\n{'='*60}")
    print(f"RESULT: {valid_count}/{total_count} valid moves generated")
    print(f"{'='*60}")
    
    return valid_count == total_count


def run_ladder_evaluation(
    llm_player: LLMPlayer,
    katago_path: str,
    katago_config: str,
    manifest_path: Path,
    output_dir: Path,
    model_name: str,
    games_per_level: int = 48,
    promotion_threshold: float = 0.55,
    starting_elo: float = 1000.0,
    verbose: bool = False
) -> dict:
    """Run the full ladder evaluation."""
    # Load manifest
    with open(manifest_path) as f:
        manifest = json.load(f)
    
    models = sorted(manifest["models"], key=lambda x: x["level"])
    
    # Setup output directory
    run_dir = output_dir / model_name
    run_dir.mkdir(parents=True, exist_ok=True)
    games_dir = run_dir / "games"
    games_dir.mkdir(exist_ok=True)
    
    # Save config
    config = {
        "model_name": model_name,
        "games_per_level": games_per_level,
        "promotion_threshold": promotion_threshold,
        "starting_elo": starting_elo,
        "timestamp": datetime.now().isoformat()
    }
    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    results = {
        "candidate": {"model_name": model_name},
        "levels": [],
        "final_elo": starting_elo,
        "highest_level": 0,
        "total_games": 0,
        "stopped_reason": None
    }
    
    candidate_elo = starting_elo
    
    for model_info in models:
        level = model_info["level"]
        reference_model = model_info["filename"]
        reference_elo = model_info["approx_elo"]
        model_path = manifest_path.parent / reference_model
        
        print(f"\n{'='*60}")
        print(f"Level {level}: {reference_model} (~{reference_elo} Elo)")
        print(f"{'='*60}")
        
        if not model_path.exists():
            print(f"WARNING: Model not found: {model_path}, skipping level")
            continue
        
        # Create level output directory
        level_dir = games_dir / f"level_{level:02d}"
        level_dir.mkdir(exist_ok=True)
        
        # Set LLM log path for this level
        llm_log_path = level_dir / "llm_log.jsonl"
        llm_player.set_log_path(llm_log_path)
        
        # Start KataGo with this model
        katago = KataGoGTP(katago_path, str(model_path), katago_config)
        try:
            katago.start()
        except Exception as e:
            print(f"ERROR: Failed to start KataGo: {e}")
            continue
        
        # Generate game variations
        variations = generate_game_variations(games_per_level)
        
        wins = 0
        losses = 0
        draws = 0
        
        try:
            for game_idx, (rule_name, rule_string, komi, candidate_color) in enumerate(variations):
                print(f"  Game {game_idx + 1}/{len(variations)}: "
                      f"{rule_name}, komi={komi}, LLM={candidate_color}", end="")
                
                # Reset move counter for new game
                llm_player.reset_move_counter(game_id=game_idx + 1)
                
                result = play_single_game(
                    game_id=game_idx + 1,
                    llm_player=llm_player,
                    katago=katago,
                    rule_name=rule_name,
                    rule_string=rule_string,
                    komi=komi,
                    candidate_color=candidate_color,
                    verbose=verbose
                )
                
                # Update stats
                if result.candidate_won:
                    wins += 1
                    print(f" -> WIN ({result.win_reason})")
                elif result.winner == "draw":
                    draws += 1
                    print(f" -> DRAW")
                else:
                    losses += 1
                    print(f" -> LOSS ({result.win_reason})")
                
                # Update Elo
                if result.winner != "draw":
                    candidate_elo = compute_elo_update(
                        candidate_elo, reference_elo, result.candidate_won
                    )
                
                # Save SGF
                sgf_path = level_dir / f"game_{game_idx + 1:03d}.sgf"
                with open(sgf_path, "w") as f:
                    f.write(result.sgf)
        
        finally:
            katago.stop()
        
        # Compute level results
        games_played = wins + losses + draws
        win_rate = wins / games_played if games_played > 0 else 0.0
        promoted = win_rate >= promotion_threshold
        
        results["levels"].append({
            "level": level,
            "reference_model": reference_model,
            "reference_elo": reference_elo,
            "games_played": games_played,
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "win_rate": win_rate,
            "promoted": promoted,
            "candidate_elo_after": candidate_elo
        })
        results["total_games"] += games_played
        results["highest_level"] = level
        results["final_elo"] = candidate_elo
        
        print(f"\nLevel {level} summary:")
        print(f"  Wins: {wins}, Losses: {losses}, Draws: {draws}")
        print(f"  Win rate: {win_rate:.1%}")
        print(f"  Candidate Elo: {candidate_elo:.0f}")
        
        if promoted:
            print(f"  -> PROMOTED to level {level + 1}")
        else:
            print(f"  -> STOPPED at level {level}")
            results["stopped_reason"] = "win_rate_below_threshold"
            break
        
        # Save intermediate results
        with open(run_dir / "results.json", "w") as f:
            json.dump(results, f, indent=2)
    
    # Check if we completed all levels
    if results["stopped_reason"] is None:
        if results["highest_level"] == models[-1]["level"]:
            results["stopped_reason"] = "completed_all_levels"
        else:
            results["stopped_reason"] = "unknown"
    
    # Save final results
    with open(run_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Save summary
    summary = {
        "model_name": model_name,
        "final_elo": results["final_elo"],
        "highest_level": results["highest_level"],
        "total_games": results["total_games"],
        "stopped_reason": results["stopped_reason"],
        "timestamp": datetime.now().isoformat()
    }
    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"Model: {model_name}")
    print(f"Final Elo: {results['final_elo']:.0f}")
    print(f"Highest Level: {results['highest_level']}")
    print(f"Total Games: {results['total_games']}")
    print(f"Stopped: {results['stopped_reason']}")
    print(f"\nResults saved to: {run_dir}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate LLM Go-playing strength via ladder competition"
    )
    
    # Required arguments
    parser.add_argument("--model-name", type=str, required=True,
                        help="Name for this evaluation run (used as output folder)")
    parser.add_argument("--candidate-type", choices=["openai", "huggingface"], required=True,
                        help="LLM player type")
    parser.add_argument("--candidate-model", type=str, default=None,
                        help="Model name or path (auto-detected from vLLM API if not provided)")
    
    # Optional arguments
    parser.add_argument("--candidate-endpoint", type=str, default="http://localhost:8001/v1",
                        help="API endpoint for OpenAI-compatible player")
    parser.add_argument("--candidate-api-key", type=str, default=None,
                        help="API key for OpenAI-compatible player")
    parser.add_argument("--games-per-level", type=int, default=48,
                        help="Games to play per level (default: 48, should be multiple of 48)")
    parser.add_argument("--promotion-threshold", type=float, default=0.55,
                        help="Win rate needed for promotion (default: 0.55)")
    parser.add_argument("--starting-elo", type=float, default=1000.0,
                        help="Starting Elo estimate (default: 1000)")
    parser.add_argument("--katago-path", type=str, 
                        default="/scratch/Projects/SPEC-SF-AISG/katago/bin/katago-cuda/katago",
                        help="Path to KataGo executable")
    parser.add_argument("--katago-config", type=str,
                        default="/scratch/Projects/SPEC-SF-AISG/katago/bin/katago-cuda/default_gtp.cfg",
                        help="Path to KataGo GTP config file")
    parser.add_argument("--manifest-path", type=Path, 
                        default=Path("assets/models/manifest.json"),
                        help="Path to reference models manifest")
    parser.add_argument("--output-dir", type=Path, default=Path("data/eval"),
                        help="Output directory")
    parser.add_argument("--verbose", action="store_true",
                        help="Print detailed output")
    parser.add_argument("--test-llm-only", action="store_true",
                        help="Test LLM player only (no KataGo required)")
    
    args = parser.parse_args()
    
    # Create LLM player
    if args.candidate_type == "openai":
        llm_player = OpenAICompatiblePlayer(
            api_base=args.candidate_endpoint,
            model=args.candidate_model,
            api_key=args.candidate_api_key
        )
    else:  # huggingface
        llm_player = HuggingFacePlayer(
            model_name_or_path=args.candidate_model
        )
    
    # Test LLM only mode
    if args.test_llm_only:
        success = test_llm_only(llm_player)
        sys.exit(0 if success else 1)
    
    # Check manifest exists
    if not args.manifest_path.exists():
        print(f"ERROR: Manifest not found: {args.manifest_path}")
        print("Run 'python eval/download_reference_models.py' first to download models.")
        sys.exit(1)
    
    # Run evaluation
    results = run_ladder_evaluation(
        llm_player=llm_player,
        katago_path=args.katago_path,
        katago_config=args.katago_config,
        manifest_path=args.manifest_path,
        output_dir=args.output_dir,
        model_name=args.model_name,
        games_per_level=args.games_per_level,
        promotion_threshold=args.promotion_threshold,
        starting_elo=args.starting_elo,
        verbose=args.verbose
    )
    
    print("\nDone!")


if __name__ == "__main__":
    main()
