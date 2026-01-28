#!/usr/bin/env python3
"""Main evaluation script: ladder-style competition pipeline.

This script evaluates a candidate's Go-playing strength by having it play against
KataGo reference models in a ladder competition with Elo rating.

Supported candidate types:
- openai: LLM via OpenAI-compatible API (vLLM, etc.)
- huggingface: LLM via direct HuggingFace model loading
- katago: KataGo neural network model

Usage Examples:
    # OpenAI-compatible API (vLLM)
    python3 eval/eval.py \\
      --candidate-type openai \\
      --candidate-endpoint http://localhost:8002/v1 \\
      --model-name "qwen3-test"
    
    # HuggingFace model
    python3 eval/eval.py \\
      --candidate-type huggingface \\
      --candidate-model /path/to/checkpoint \\
      --model-name "my-hf-model"
    
    # KataGo model
    python3 eval/eval.py \\
      --candidate-type katago \\
      --candidate-model assets/models/level_02_kata1-b10c128-s197428736-d67404019.txt.gz \\
      --model-name "katago-level2"

Notes:
    - Use http:// (not https://) for local vLLM endpoints
    - For openai type, --candidate-model is auto-detected if not provided
    - See eval/config.json for configuration examples
"""

import argparse
import json
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from eval.players import OpenAIPlayer, HuggingFacePlayer, KataGoPlayer, Player
from eval.ladder import run_ladder_evaluation


def create_openai_player(args) -> OpenAIPlayer:
    """Create an OpenAI-compatible player."""
    return OpenAIPlayer(
        api_base=args.candidate_endpoint,
        model=args.candidate_model,
        api_key=args.candidate_api_key,
        temperature=args.temperature,
        max_tokens=args.max_tokens
    )


def create_huggingface_player(args) -> HuggingFacePlayer:
    """Create a HuggingFace player."""
    assert args.candidate_model is not None, \
        "HuggingFace player requires --candidate-model"
    
    return HuggingFacePlayer(
        model_name_or_path=args.candidate_model,
        device=args.device,
        temperature=args.temperature,
        max_new_tokens=args.max_tokens
    )


def create_katago_player(args) -> KataGoPlayer:
    """Create a KataGo player as candidate."""
    assert args.candidate_model is not None, \
        "KataGo player requires --candidate-model"
    
    model_path = Path(args.candidate_model)
    assert model_path.exists(), f"KataGo model not found: {model_path}"
    
    return KataGoPlayer(
        katago_path=args.katago_path,
        model_path=str(model_path),
        config_path=args.katago_config
    )


def create_player(args) -> Player:
    """Factory function to create the appropriate player."""
    if args.candidate_type == "openai":
        return create_openai_player(args)
    elif args.candidate_type == "huggingface":
        return create_huggingface_player(args)
    elif args.candidate_type == "katago":
        return create_katago_player(args)
    else:
        raise ValueError(f"Unknown candidate type: {args.candidate_type}")


def test_player(player: Player, num_tests: int = 3) -> bool:
    """Test player can generate valid moves."""
    from eval.players.base import validate_gtp_move
    
    print("\n" + "=" * 60)
    print("PLAYER TEST MODE")
    print(f"Testing player: {player.name}")
    print("=" * 60)
    
    test_cases = [
        {"move_history": [], "rules": "koSIMPLEscoreTERRITORYtaxSEKIsui0", "komi": 6.5, "color": "B"},
        {"move_history": [["B", "D4"], ["W", "Q16"]], "rules": "koSIMPLEscoreAREAtaxNONEsui0whbN", "komi": 7.5, "color": "B"},
        {"move_history": [["B", "D4"], ["W", "Q16"], ["B", "D16"]], "rules": "koPOSITIONALscoreAREAtaxNONEsui1", "komi": 7.5, "color": "W"},
    ]
    
    player.start()
    valid_count = 0
    
    try:
        for i, tc in enumerate(test_cases[:num_tests]):
            print(f"\nTest {i+1}/{num_tests}:")
            print(f"  Rules: {tc['rules']}")
            print(f"  Color: {tc['color']}")
            print(f"  History: {tc['move_history']}")
            
            move = player.get_move(**tc)
            valid = validate_gtp_move(move) if move else False
            print(f"  Move: {move}")
            print(f"  Valid: {valid}")
            
            if valid:
                valid_count += 1
    finally:
        player.stop()
    
    print(f"\n{'='*60}")
    print(f"RESULT: {valid_count}/{num_tests} valid moves")
    print(f"{'='*60}")
    
    return valid_count == num_tests


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Go-playing strength via ladder competition",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # OpenAI/vLLM
  python eval/eval.py --candidate-type openai --candidate-endpoint http://localhost:8002/v1 --model-name test

  # HuggingFace
  python eval/eval.py --candidate-type huggingface --candidate-model /path/to/model --model-name test

  # KataGo
  python eval/eval.py --candidate-type katago --candidate-model assets/models/level_02_*.txt.gz --model-name test
"""
    )
    
    # Required arguments
    parser.add_argument("--model-name", type=str, required=True,
                        help="Name for this evaluation run (used as output folder)")
    parser.add_argument("--candidate-type", 
                        choices=["openai", "huggingface", "katago"], 
                        required=True,
                        help="Candidate player type")
    
    # Candidate model arguments
    parser.add_argument("--candidate-model", type=str, default=None,
                        help="Model name/path (auto-detected for openai type)")
    parser.add_argument("--candidate-endpoint", type=str, 
                        default="http://localhost:8001/v1",
                        help="API endpoint for openai type")
    parser.add_argument("--candidate-api-key", type=str, default=None,
                        help="API key for openai type")
    
    # Model parameters
    parser.add_argument("--temperature", type=float, default=0.1,
                        help="Sampling temperature (default: 0.1)")
    parser.add_argument("--max-tokens", type=int, default=1024,
                        help="Max tokens in response (default: 1024)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device for HuggingFace models (default: cuda)")
    
    # Evaluation parameters
    parser.add_argument("--games-per-level", type=int, default=48,
                        help="Games per level (default: 48)")
    parser.add_argument("--promotion-threshold", type=float, default=0.55,
                        help="Win rate for promotion (default: 0.55)")
    parser.add_argument("--starting-elo", type=float, default=1000.0,
                        help="Starting Elo estimate (default: 1000)")
    
    # KataGo paths
    parser.add_argument("--katago-path", type=str,
                        default="/scratch/Projects/SPEC-SF-AISG/katago/bin/katago-cuda/katago",
                        help="Path to KataGo executable")
    parser.add_argument("--katago-config", type=str,
                        default="/scratch/Projects/SPEC-SF-AISG/katago/bin/katago-cuda/default_gtp.cfg",
                        help="Path to KataGo config file")
    
    # I/O paths
    parser.add_argument("--manifest-path", type=Path,
                        default=Path("assets/models/manifest.json"),
                        help="Path to reference models manifest")
    parser.add_argument("--output-dir", type=Path, 
                        default=Path("data/eval"),
                        help="Output directory")
    
    # Options
    parser.add_argument("--verbose", action="store_true",
                        help="Print detailed output")
    parser.add_argument("--test-only", action="store_true",
                        help="Test player only, don't run evaluation")
    
    args = parser.parse_args()
    
    # Create player
    print(f"Creating {args.candidate_type} player...")
    player = create_player(args)
    print(f"Player: {player.name}")
    
    # Test-only mode
    if args.test_only:
        success = test_player(player)
        sys.exit(0 if success else 1)
    
    # Validate manifest exists
    assert args.manifest_path.exists(), \
        f"Manifest not found: {args.manifest_path}\n" \
        "Run 'python eval/download_reference_models.py' first."
    
    # Run evaluation
    results = run_ladder_evaluation(
        candidate=player,
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
