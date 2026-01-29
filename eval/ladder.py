"""Ladder evaluation logic for Go.

Implements the ladder-style competition pipeline with Elo rating.
Games within a level run in parallel using async I/O.

Architecture:
- Candidate and reference servers run as separate PBS jobs
- This script connects to them via HTTP endpoints
- Reference server is restarted with new model between levels
"""

import asyncio
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import yaml

from eval.players.base import Player
from eval.players.katago_player import KataGoPlayer
from eval.game import play_single_game, generate_game_variations, GameResult


# Repository root
REPO_ROOT = Path(__file__).parent.parent.resolve()
CONFIG_PATH = REPO_ROOT / "configs" / "config.yaml"
MANAGE_SCRIPT = REPO_ROOT / "runtime" / "manage_servers.sh"


def load_config() -> dict:
    """Load configuration from config.yaml."""
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def compute_elo_update(
    candidate_elo: float, 
    opponent_elo: float, 
    candidate_won: bool, 
    k_factor: float = 32
) -> float:
    """Compute new Elo rating after a game."""
    expected = 1 / (1 + 10 ** ((opponent_elo - candidate_elo) / 400))
    actual = 1.0 if candidate_won else 0.0
    return candidate_elo + k_factor * (actual - expected)


def restart_reference_server(model_path: str, timeout: int = 120) -> str:
    """Restart the reference KataGo server with a new model.
    
    Args:
        model_path: Path to the model weights
        timeout: Timeout in seconds to wait for server to be ready
        
    Returns:
        Endpoint URL for the reference server
    """
    print(f"  Restarting reference server with model: {model_path}")
    
    # Restart reference server with new model
    result = subprocess.run(
        [str(MANAGE_SCRIPT), "restart", "reference", "--model", model_path],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT)
    )
    
    if result.returncode != 0:
        raise RuntimeError(f"Failed to restart reference server: {result.stderr}")
    
    # Wait for server to be ready and get endpoint
    result = subprocess.run(
        [str(MANAGE_SCRIPT), "wait", "reference", str(timeout)],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT)
    )
    
    if result.returncode != 0:
        raise RuntimeError(f"Reference server not ready: {result.stderr}")
    
    # Parse endpoint from output (last line should be the URL)
    # Filter for lines that look like URLs
    lines = result.stdout.strip().split("\n")
    endpoint = None
    for line in reversed(lines):
        line = line.strip()
        if line.startswith("http://") or line.startswith("https://"):
            endpoint = line
            break
    
    if not endpoint:
        raise RuntimeError(f"Could not parse endpoint from output: {result.stdout}")
    
    print(f"  Reference server ready at: {endpoint}")
    
    return endpoint


def get_server_endpoints() -> dict:
    """Get current server endpoints from manage_servers.sh."""
    result = subprocess.run(
        [str(MANAGE_SCRIPT), "endpoints"],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT)
    )
    
    endpoints = {}
    for line in result.stdout.strip().split("\n"):
        if "=" in line:
            key, value = line.split("=", 1)
            endpoints[key.lower()] = value
    
    return endpoints


async def play_game_with_id(
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
    """Play a single game (wrapper for parallel execution)."""
    candidate.reset_game(game_id=game_id)
    
    return await play_single_game(
        game_id=game_id,
        candidate=candidate,
        reference=reference,
        rule_name=rule_name,
        rule_string=rule_string,
        komi=komi,
        candidate_color=candidate_color,
        max_moves=max_moves,
        verbose=verbose
    )


async def run_level_games(
    candidate: Player,
    reference: Player,
    variations: List,
    max_parallel: int = 4,
    max_moves: int = 500,
    verbose: bool = False
) -> List[GameResult]:
    """Run all games for a level in parallel.
    
    Args:
        candidate: Candidate player
        reference: Reference player
        variations: List of (rule_name, rule_string, komi, candidate_color)
        max_parallel: Max concurrent games
        max_moves: Maximum moves per game
        verbose: Print verbose output
    
    Returns:
        List of game results
    """
    semaphore = asyncio.Semaphore(max_parallel)
    
    async def run_game(game_idx: int, variation: tuple) -> GameResult:
        async with semaphore:
            rule_name, rule_string, komi, candidate_color = variation
            print(f"  Game {game_idx + 1}/{len(variations)}: "
                  f"{rule_name}, komi={komi}, candidate={candidate_color}", 
                  flush=True)
            
            result = await play_game_with_id(
                game_id=game_idx + 1,
                candidate=candidate,
                reference=reference,
                rule_name=rule_name,
                rule_string=rule_string,
                komi=komi,
                candidate_color=candidate_color,
                max_moves=max_moves,
                verbose=verbose
            )
            
            # Print result
            if result.candidate_won:
                outcome = f"WIN ({result.win_reason})"
            elif result.winner == "draw":
                outcome = "DRAW"
            else:
                outcome = f"LOSS ({result.win_reason})"
            
            print(f"    Game {game_idx + 1} -> {outcome}", flush=True)
            
            return result
    
    # Create all tasks
    tasks = [
        run_game(i, var) 
        for i, var in enumerate(variations)
    ]
    
    # Run all games in parallel (limited by semaphore)
    results = await asyncio.gather(*tasks)
    
    return list(results)


async def run_ladder_evaluation_async(
    candidate: Player,
    manifest_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    model_name: Optional[str] = None,
    games_per_level: Optional[int] = None,
    promotion_threshold: Optional[float] = None,
    starting_elo: Optional[float] = None,
    max_parallel: Optional[int] = None,
    max_moves_per_game: Optional[int] = None,
    verbose: bool = False,
    max_levels: Optional[int] = None,
    reference_endpoint: Optional[str] = None
) -> dict:
    """Run the full ladder evaluation asynchronously.
    
    Uses configuration from configs/config.yaml for defaults.
    
    Args:
        candidate: The candidate player to evaluate
        manifest_path: Path to reference models manifest (default: from config)
        output_dir: Base output directory (default: from config)
        model_name: Name for this evaluation run (default: candidate.name)
        games_per_level: Games per level (default: from config)
        promotion_threshold: Win rate needed for promotion (default: from config)
        starting_elo: Starting Elo estimate (default: from config)
        max_parallel: Max parallel games per level (default: from config)
        max_moves_per_game: Max moves per game (default: from config)
        verbose: Print detailed output
        max_levels: Maximum levels to evaluate (None = all)
        reference_endpoint: Initial reference server endpoint (if already running)
    
    Returns:
        Results dictionary
    """
    # Load configuration
    cfg = load_config()
    eval_cfg = cfg.get("eval", {}).get("ladder", {})
    
    # Apply defaults from config
    manifest_path = manifest_path or Path(cfg["eval"]["manifest_path"])
    output_dir = output_dir or Path(cfg["eval"]["output_dir"])
    model_name = model_name or candidate.name
    games_per_level = games_per_level or eval_cfg.get("games_per_level", 48)
    promotion_threshold = promotion_threshold or eval_cfg.get("promotion_threshold", 0.55)
    starting_elo = starting_elo or eval_cfg.get("starting_elo", 1000)
    max_parallel = max_parallel or eval_cfg.get("max_parallel", 4)
    max_moves_per_game = max_moves_per_game or eval_cfg.get("max_moves_per_game", 500)
    
    # Make paths absolute
    if not manifest_path.is_absolute():
        manifest_path = REPO_ROOT / manifest_path
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    
    # Load manifest
    assert manifest_path.exists(), f"Manifest not found: {manifest_path}"
    with open(manifest_path) as f:
        manifest = json.load(f)
    
    models = sorted(manifest["models"], key=lambda x: x["level"])
    
    # Limit levels if specified
    if max_levels is not None:
        models = models[:max_levels]
    
    # Setup output directory
    run_dir = output_dir / model_name
    run_dir.mkdir(parents=True, exist_ok=True)
    games_dir = run_dir / "games"
    games_dir.mkdir(exist_ok=True)
    
    # Initialize results with config merged in
    results = {
        "config": {
            "model_name": model_name,
            "candidate_name": candidate.name,
            "games_per_level": games_per_level,
            "promotion_threshold": promotion_threshold,
            "starting_elo": starting_elo,
            "max_moves_per_game": max_moves_per_game,
            "timestamp": datetime.now().isoformat()
        },
        "candidate": {"model_name": model_name, "candidate_name": candidate.name},
        "levels": [],
        "final_elo": starting_elo,
        "highest_level": 0,
        "total_games": 0,
        "stopped_reason": None
    }
    
    candidate_elo = starting_elo
    
    # Start candidate
    await candidate.start()
    
    try:
        for model_info in models:
            level = model_info["level"]
            reference_model = model_info["filename"]
            reference_elo = model_info["approx_elo"]
            model_path = manifest_path.parent / reference_model
            
            print(f"\n{'='*60}")
            print(f"Level {level}: {reference_model} (~{reference_elo} Elo)")
            print(f"{'='*60}")
            
            assert model_path.exists(), f"Reference model not found: {model_path}"
            
            # Create level output directory
            level_dir = games_dir / f"level_{level:02d}"
            level_dir.mkdir(exist_ok=True)
            
            # Set log path for candidate if it supports logging
            # llm_log.jsonl will be saved in games/level_XX/llm_log.jsonl
            if candidate.requires_logging:
                log_path = level_dir / "llm_log.jsonl"
                candidate.set_log_path(log_path)
                print(f"  LLM log will be saved to: {log_path}")
            
            # Restart reference server with level-specific model
            ref_endpoint = restart_reference_server(str(model_path))
            
            # Create reference player connected to the restarted server
            reference = KataGoPlayer(
                endpoint=ref_endpoint,
                name=f"reference_level_{level}"
            )
            
            await reference.start()
            
            try:
                # Generate game variations
                variations = generate_game_variations(games_per_level)
                
                # Run all games in parallel
                game_results = await run_level_games(
                    candidate=candidate,
                    reference=reference,
                    variations=variations,
                    max_parallel=max_parallel,
                    max_moves=max_moves_per_game,
                    verbose=verbose
                )
                
                # Tally results
                wins = sum(1 for r in game_results if r.candidate_won)
                losses = sum(1 for r in game_results if not r.candidate_won and r.winner != "draw")
                draws = sum(1 for r in game_results if r.winner == "draw")
                
                # Update Elo for each game and collect game details
                level_games = []
                for result in game_results:
                    if result.winner != "draw":
                        candidate_elo = compute_elo_update(
                            candidate_elo, reference_elo, result.candidate_won
                        )
                    
                    # Save SGF
                    sgf_path = level_dir / f"game_{result.game_id:03d}.sgf"
                    with open(sgf_path, "w") as f:
                        f.write(result.sgf)
                    
                    # Collect game details for summary
                    level_games.append({
                        "game_id": result.game_id,
                        "game_history": result.moves,  # Entire move history
                        "result": {
                            "winner": result.winner,
                            "win_reason": result.win_reason,
                            "move_count": result.move_count,
                            "candidate_won": result.candidate_won
                        },
                        "game_config": {
                            "rule_name": result.rule_name,
                            "rule_string": result.rule_string,
                            "komi": result.komi,
                            "candidate_color": result.candidate_color
                        }
                    })
            
            finally:
                await reference.stop()
            
            # Compute level results
            games_played = wins + losses + draws
            win_rate = wins / games_played if games_played > 0 else 0.0
            promoted = win_rate >= promotion_threshold
            
            results["levels"].append({
                "level": level,
                "candidate_model": candidate.name,
                "reference_model": reference_model,
                "reference_elo": reference_elo,
                "games_played": games_played,
                "wins": wins,
                "losses": losses,
                "draws": draws,
                "win_rate": win_rate,
                "promoted": promoted,
                "candidate_elo_after": candidate_elo,
                "games": level_games  # Per-game details with full history
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
            
            # Save intermediate results (same format as final)
            results_json = {
                "candidate_model": candidate.name,
                "promotion_result": {
                    "final_elo": candidate_elo,
                    "highest_level": level,
                    "total_games": results["total_games"],
                    "stopped_reason": None
                },
                "levels": []
            }
            
            for lev_data in results["levels"]:
                level_summary = {
                    "level": lev_data["level"],
                    "candidate_model": lev_data["candidate_model"],
                    "reference_model": lev_data["reference_model"],
                    "wins": lev_data["wins"],
                    "losses": lev_data["losses"],
                    "draws": lev_data["draws"],
                    "promoted": lev_data["promoted"],
                    "games": []
                }
                
                for game in lev_data.get("games", []):
                    level_summary["games"].append({
                        "game_id": game["game_id"],
                        "game_history": game["game_history"],
                        "result": game["result"]
                    })
                
                results_json["levels"].append(level_summary)
            
            with open(run_dir / "results.json", "w") as f:
                json.dump(results_json, f, indent=2)
    
    finally:
        await candidate.stop()
    
    # Final status
    if results["stopped_reason"] is None:
        if results["highest_level"] == models[-1]["level"]:
            results["stopped_reason"] = "completed_all_levels"
        else:
            results["stopped_reason"] = "unknown"
    
    # Save results.json with requested format (simplified summary)
    results_json = {
        "candidate_model": candidate.name,
        "promotion_result": {
            "final_elo": results["final_elo"],
            "highest_level": results["highest_level"],
            "total_games": results["total_games"],
            "stopped_reason": results["stopped_reason"]
        },
        "levels": []
    }
    
    for level_data in results["levels"]:
        level_summary = {
            "level": level_data["level"],
            "candidate_model": level_data["candidate_model"],
            "reference_model": level_data["reference_model"],
            "wins": level_data["wins"],
            "losses": level_data["losses"],
            "draws": level_data["draws"],
            "promoted": level_data["promoted"],
            "games": []
        }
        
        # Add game history and results for each game
        for game in level_data.get("games", []):
            level_summary["games"].append({
                "game_id": game["game_id"],
                "game_history": game["game_history"],  # Entire move history
                "result": game["result"]  # Winner, win_reason, move_count, candidate_won
            })
        
        results_json["levels"].append(level_summary)
    
    # Save results.json (ensure directory exists)
    run_dir.mkdir(parents=True, exist_ok=True)
    results_path = run_dir / "results.json"
    try:
        with open(results_path, "w") as f:
            json.dump(results_json, f, indent=2)
        print(f"\nResults saved to: {run_dir}")
        print(f"  - results.json: Evaluation summary with game history and results")
    except Exception as e:
        print(f"\nERROR: Failed to save results.json: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"Model: {model_name}")
    print(f"Candidate: {candidate.name}")
    print(f"Final Elo: {results['final_elo']:.0f}")
    print(f"Highest Level: {results['highest_level']}")
    print(f"Total Games: {results['total_games']}")
    print(f"Stopped: {results['stopped_reason']}")
    print(f"\nResults saved to: {run_dir}")
    print(f"  - results.json: Evaluation summary with game history and results")
    if candidate.requires_logging:
        print(f"  - games/level_XX/llm_log.jsonl: LLM interaction logs (one per level)")
    print(f"  - games/level_XX/game_XXX.sgf: Game records (one per game)")
    
    return results


def run_ladder_evaluation(
    candidate: Player,
    manifest_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    model_name: Optional[str] = None,
    games_per_level: Optional[int] = None,
    promotion_threshold: Optional[float] = None,
    starting_elo: Optional[float] = None,
    max_parallel: Optional[int] = None,
    max_moves_per_game: Optional[int] = None,
    verbose: bool = False,
    max_levels: Optional[int] = None,
    reference_endpoint: Optional[str] = None
) -> dict:
    """Synchronous wrapper for run_ladder_evaluation_async."""
    return asyncio.run(run_ladder_evaluation_async(
        candidate=candidate,
        manifest_path=manifest_path,
        output_dir=output_dir,
        model_name=model_name,
        games_per_level=games_per_level,
        promotion_threshold=promotion_threshold,
        starting_elo=starting_elo,
        max_parallel=max_parallel,
        max_moves_per_game=max_moves_per_game,
        verbose=verbose,
        max_levels=max_levels,
        reference_endpoint=reference_endpoint
    ))
