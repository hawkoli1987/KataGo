#!/usr/bin/env python3
"""Ladder evaluation pipeline for Go-playing models.

Evaluates a candidate model against KataGo reference models of increasing strength.
Both candidate and reference models are served as stateless HTTP endpoints,
enabling async parallel game execution.

Usage:
    # OpenAI/vLLM candidate (auto-detects model)
    python3 eval/eval.py \\
        --candidate-type openai \\
        --candidate-endpoint http://localhost:8002/v1 \\
        --model-name "qwen3-test"

    # KataGo candidate (connect to running server)
    python3 eval/eval.py \\
        --candidate-type katago \\
        --candidate-endpoint http://hopper-34:9200 \\
        --model-name "katago-level3"

    # With custom settings
    python3 eval/eval.py \\
        --candidate-type openai \\
        --candidate-endpoint http://localhost:8002/v1 \\
        --model-name "my-eval" \\
        --games-per-level 48 \\
        --max-parallel 4
"""

import argparse
from pathlib import Path
from typing import Optional

from eval.players import OpenAIPlayer, KataGoPlayer
from eval.ladder import run_ladder_evaluation


# Default paths (relative to repo root)
DEFAULT_MANIFEST_PATH = Path("assets/models/manifest.json")
DEFAULT_OUTPUT_DIR = Path("data/eval")


def create_openai_player(
    endpoint: str,
    model: Optional[str] = None,
    temperature: float = 0.1,
    max_tokens: int = 1024
) -> OpenAIPlayer:
    """Create an OpenAI-compatible player."""
    return OpenAIPlayer(
        api_base=endpoint,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens
    )


def create_katago_player(endpoint: str, name: Optional[str] = None) -> KataGoPlayer:
    """Create a KataGo player connecting to a remote server."""
    return KataGoPlayer(endpoint=endpoint, name=name)


def main():
    parser = argparse.ArgumentParser(
        description="Ladder evaluation for Go-playing models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Required
    parser.add_argument("--model-name", required=True,
                        help="Name for this evaluation run (creates output folder)")
    parser.add_argument("--candidate-type", required=True, choices=["openai", "katago"],
                        help="Type of candidate model")
    
    # Candidate config
    parser.add_argument("--candidate-endpoint", type=str, required=True,
                        help="API endpoint for candidate (OpenAI: http://localhost:8002/v1, KataGo: http://hopper-34:9200)")
    parser.add_argument("--candidate-model", type=str,
                        help="Model name for OpenAI (optional, auto-detected from endpoint)")
    
    # Paths
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH,
                        help="Path to reference models manifest")
    
    # Evaluation params
    parser.add_argument("--games-per-level", type=int, default=48,
                        help="Number of games per level (default: 48)")
    parser.add_argument("--max-parallel", type=int, default=4,
                        help="Max parallel games (default: 4)")
    parser.add_argument("--promotion-threshold", type=float, default=0.55,
                        help="Win rate for promotion (default: 0.55)")
    parser.add_argument("--starting-elo", type=float, default=1000.0,
                        help="Starting Elo rating (default: 1000)")
    parser.add_argument("--max-levels", type=int, default=None,
                        help="Max levels to evaluate (default: all)")
    
    # Output
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help="Output directory")
    parser.add_argument("--verbose", action="store_true",
                        help="Print move-by-move output")
    
    # LLM params
    parser.add_argument("--temperature", type=float, default=0.1,
                        help="LLM temperature (default: 0.1)")
    parser.add_argument("--max-tokens", type=int, default=1024,
                        help="LLM max tokens (default: 1024)")
    
    args = parser.parse_args()
    
    # Create candidate player
    if args.candidate_type == "openai":
        candidate = create_openai_player(
            endpoint=args.candidate_endpoint,
            model=args.candidate_model,
            temperature=args.temperature,
            max_tokens=args.max_tokens
        )
    else:  # katago
        candidate = create_katago_player(
            endpoint=args.candidate_endpoint,
            name=args.candidate_model
        )
    
    # Run evaluation
    results = run_ladder_evaluation(
        candidate=candidate,
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        model_name=args.model_name,
        games_per_level=args.games_per_level,
        promotion_threshold=args.promotion_threshold,
        starting_elo=args.starting_elo,
        max_parallel=args.max_parallel,
        verbose=args.verbose,
        max_levels=args.max_levels
    )
    
    print(f"\nEvaluation complete!")
    print(f"Final Elo: {results['final_elo']:.0f}")
    print(f"Highest Level: {results['highest_level']}")


if __name__ == "__main__":
    main()
