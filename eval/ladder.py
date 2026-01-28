"""Ladder evaluation logic for Go.

Implements the ladder-style competition pipeline with Elo rating.
Games within a level run in parallel using async I/O.
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from eval.players.base import Player
from eval.players.katago_player import KataGoPlayer
from eval.game import play_single_game, generate_game_variations, GameResult


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


async def play_game_with_id(
    game_id: int,
    candidate: Player,
    reference: Player,
    rule_name: str,
    rule_string: str,
    komi: float,
    candidate_color: str,
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
        verbose=verbose
    )


async def run_level_games(
    candidate: Player,
    reference: Player,
    variations: List,
    max_parallel: int = 4,
    verbose: bool = False
) -> List[GameResult]:
    """Run all games for a level in parallel.
    
    Args:
        candidate: Candidate player
        reference: Reference player
        variations: List of (rule_name, rule_string, komi, candidate_color)
        max_parallel: Max concurrent games
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
    manifest_path: Path,
    output_dir: Path,
    model_name: str,
    games_per_level: int = 48,
    promotion_threshold: float = 0.55,
    starting_elo: float = 1000.0,
    candidate_gpu: int = 0,
    reference_gpu: int = 1,
    max_parallel: int = 4,
    verbose: bool = False,
    max_levels: Optional[int] = None
) -> dict:
    """Run the full ladder evaluation asynchronously.
    
    Args:
        candidate: The candidate player to evaluate
        manifest_path: Path to reference models manifest
        output_dir: Base output directory
        model_name: Name for this evaluation run
        games_per_level: Games per level
        promotion_threshold: Win rate needed for promotion
        starting_elo: Starting Elo estimate
        candidate_gpu: GPU for candidate (if KataGo)
        reference_gpu: GPU for reference KataGo
        max_parallel: Max parallel games per level
        verbose: Print detailed output
        max_levels: Maximum levels to evaluate (None = all)
    
    Returns:
        Results dictionary
    """
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
    
    # Save config
    config = {
        "model_name": model_name,
        "candidate_name": candidate.name,
        "games_per_level": games_per_level,
        "promotion_threshold": promotion_threshold,
        "starting_elo": starting_elo,
        "timestamp": datetime.now().isoformat()
    }
    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    results = {
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
            if candidate.requires_logging:
                candidate.set_log_path(level_dir / "llm_log.jsonl")
            
            # Create reference KataGo player
            reference = KataGoPlayer(
                model_path=str(model_path),
                gpu_id=reference_gpu,
                port=8200 + level,  # Unique port per level
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
                    verbose=verbose
                )
                
                # Tally results
                wins = sum(1 for r in game_results if r.candidate_won)
                losses = sum(1 for r in game_results if not r.candidate_won and r.winner != "draw")
                draws = sum(1 for r in game_results if r.winner == "draw")
                
                # Update Elo for each game
                for result in game_results:
                    if result.winner != "draw":
                        candidate_elo = compute_elo_update(
                            candidate_elo, reference_elo, result.candidate_won
                        )
                    
                    # Save SGF
                    sgf_path = level_dir / f"game_{result.game_id:03d}.sgf"
                    with open(sgf_path, "w") as f:
                        f.write(result.sgf)
            
            finally:
                await reference.stop()
            
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
    
    finally:
        await candidate.stop()
    
    # Final status
    if results["stopped_reason"] is None:
        if results["highest_level"] == models[-1]["level"]:
            results["stopped_reason"] = "completed_all_levels"
        else:
            results["stopped_reason"] = "unknown"
    
    # Save final results
    with open(run_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    with open(run_dir / "summary.json", "w") as f:
        json.dump({
            "model_name": model_name,
            "candidate_name": candidate.name,
            "final_elo": results["final_elo"],
            "highest_level": results["highest_level"],
            "total_games": results["total_games"],
            "stopped_reason": results["stopped_reason"],
            "timestamp": datetime.now().isoformat()
        }, f, indent=2)
    
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
    
    return results


def run_ladder_evaluation(
    candidate: Player,
    manifest_path: Path,
    output_dir: Path,
    model_name: str,
    games_per_level: int = 48,
    promotion_threshold: float = 0.55,
    starting_elo: float = 1000.0,
    candidate_gpu: int = 0,
    reference_gpu: int = 1,
    max_parallel: int = 4,
    verbose: bool = False,
    max_levels: Optional[int] = None
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
        candidate_gpu=candidate_gpu,
        reference_gpu=reference_gpu,
        max_parallel=max_parallel,
        verbose=verbose,
        max_levels=max_levels
    ))
