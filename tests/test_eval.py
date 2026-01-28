#!/usr/bin/env python3
"""Tests for the async evaluation pipeline using pytest.

Tests that each model type can play games via HTTP endpoints.
Both candidate and reference use FastAPI servers for async parallel execution.

Usage:
    pytest tests/test_eval.py -v
    pytest tests/test_eval.py -v -k "openai"
    pytest tests/test_eval.py -v -k "promotion"
"""

import asyncio
import json
import shutil
import sys
from pathlib import Path

import pytest

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from eval.players import OpenAIPlayer, KataGoPlayer, validate_gtp_move
from eval.game import play_single_game, GO_RULES, KOMI_VALUES
from eval.ladder import run_ladder_evaluation


# Test configuration
VLLM_ENDPOINT = "http://localhost:8002/v1"
MANIFEST_PATH = Path("assets/models/manifest.json")
TEST_OUTPUT_DIR = Path("data/eval/_test_runs")

# Model paths for KataGo tests (relative to repo root)
KATAGO_LEVEL1_MODEL = "assets/models/level_01_kata1-b6c96-s175395328-d26788732.txt.gz"
KATAGO_LEVEL2_MODEL = "assets/models/level_02_kata1-b10c128-s197428736-d67404019.txt.gz"
KATAGO_LEVEL3_MODEL = "assets/models/level_03_kata1-b15c192-s497233664-d149638345.txt.gz"

# Test parameters
GAMES_PER_LEVEL_TEST = 5
MAX_LEVELS_TEST = 3
MIN_VALID_MOVES = 10


@pytest.fixture(scope="module")
def cleanup_test_dir():
    """Clean up test output directory before tests."""
    if TEST_OUTPUT_DIR.exists():
        shutil.rmtree(TEST_OUTPUT_DIR)
    TEST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    yield


@pytest.fixture(scope="module")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# ============================================================================
# OpenAI Player Tests
# ============================================================================

class TestOpenAIPlayer:
    """Test OpenAI-compatible player (vLLM)."""
    
    @pytest.mark.asyncio
    async def test_can_generate_valid_moves(self):
        """Test player can generate valid GTP moves with real game rules."""
        player = OpenAIPlayer(api_base=VLLM_ENDPOINT)
        await player.start()
        
        try:
            # Use actual Go rules and positions
            test_cases = [
                # Empty board, Japanese rules
                {
                    "move_history": [],
                    "rules": GO_RULES[0][1],  # Japanese
                    "komi": KOMI_VALUES[1],  # 6.5
                    "color": "B"
                },
                # After a few opening moves, Chinese rules
                {
                    "move_history": [["B", "Q16"], ["W", "D4"], ["B", "D16"], ["W", "Q4"]],
                    "rules": GO_RULES[1][1],  # Chinese
                    "komi": KOMI_VALUES[2],  # 7.5
                    "color": "B"
                },
            ]
            
            for tc in test_cases:
                move = await player.get_move(**tc)
                assert move, f"Player returned empty move for {tc}"
                assert validate_gtp_move(move), f"Invalid move '{move}' for {tc}"
        
        finally:
            await player.stop()
    
    @pytest.mark.asyncio
    async def test_can_play_multiple_moves(self):
        """Test player can play a realistic opening sequence."""
        player = OpenAIPlayer(api_base=VLLM_ENDPOINT)
        await player.start()
        
        try:
            rules = GO_RULES[0][1]  # Japanese
            komi = KOMI_VALUES[1]  # 6.5
            
            # Build a realistic opening sequence
            move_history = []
            valid_count = 0
            
            # Play 10 moves alternating colors
            for i in range(MIN_VALID_MOVES):
                color = "B" if i % 2 == 0 else "W"
                move = await player.get_move(move_history, rules, komi, color)
                
                if move and validate_gtp_move(move):
                    valid_count += 1
                    if move.upper() not in ("PASS", "RESIGN"):
                        move_history.append([color, move])
            
            print(f"\n  OpenAI player made {valid_count}/{MIN_VALID_MOVES} valid moves")
            assert valid_count >= MIN_VALID_MOVES, \
                f"Player only made {valid_count} valid moves, expected >= {MIN_VALID_MOVES}"
        
        finally:
            await player.stop()
    
    @pytest.mark.asyncio
    async def test_llm_log_generated(self, cleanup_test_dir):
        """Test that llm_log.jsonl is generated with valid content."""
        log_path = TEST_OUTPUT_DIR / "openai_test" / "llm_log.jsonl"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        if log_path.exists():
            log_path.unlink()
        
        player = OpenAIPlayer(api_base=VLLM_ENDPOINT)
        player.set_log_path(log_path)
        
        await player.start()
        
        try:
            player.reset_game(game_id=1)
            
            # Play realistic opening moves
            await player.get_move([], GO_RULES[0][1], 6.5, "B")
            await player.get_move(
                [["B", "Q16"], ["W", "D4"]],
                GO_RULES[0][1], 6.5, "B"
            )
        
        finally:
            await player.stop()
        
        assert log_path.exists(), f"Log file not created: {log_path}"
        
        with open(log_path) as f:
            lines = f.readlines()
        
        assert len(lines) >= 2, "Expected at least 2 log entries"
        
        for line in lines:
            entry = json.loads(line)
            assert "game_id" in entry
            assert "ply" in entry
            assert "prompt" in entry
            assert "parsed_move" in entry
        
        print(f"\n  LLM log generated with {len(lines)} entries")


# ============================================================================
# KataGo Player Tests
# ============================================================================

class TestKataGoPlayer:
    """Test KataGo player via FastAPI server."""
    
    @pytest.mark.asyncio
    async def test_can_generate_valid_moves(self):
        """Test player can generate valid GTP moves with real game rules."""
        player = KataGoPlayer(
            model_path=KATAGO_LEVEL3_MODEL,
            gpu_id=0,
            port=8150
        )
        
        await player.start()
        
        try:
            # Use actual Go rules and positions
            test_cases = [
                # Empty board, Japanese rules
                {
                    "move_history": [],
                    "rules": GO_RULES[0][1],  # Japanese
                    "komi": KOMI_VALUES[1],  # 6.5
                    "color": "B"
                },
                # Mid-game position, Chinese rules
                {
                    "move_history": [["B", "Q16"], ["W", "D4"], ["B", "D16"], ["W", "Q4"]],
                    "rules": GO_RULES[1][1],  # Chinese
                    "komi": KOMI_VALUES[2],  # 7.5
                    "color": "B"
                },
            ]
            
            for tc in test_cases:
                move = await player.get_move(**tc)
                assert move, f"Player returned empty move for {tc}"
                assert validate_gtp_move(move), f"Invalid move '{move}' for {tc}"
        
        finally:
            await player.stop()
    
    @pytest.mark.asyncio
    async def test_can_play_multiple_moves(self):
        """Test KataGo can play a realistic game sequence."""
        player = KataGoPlayer(
            model_path=KATAGO_LEVEL3_MODEL,
            gpu_id=0,
            port=8151
        )
        
        await player.start()
        
        try:
            rules = GO_RULES[0][1]  # Japanese
            komi = KOMI_VALUES[1]  # 6.5
            
            # Build a realistic game sequence
            move_history = []
            valid_count = 0
            
            # Play 10 moves
            for i in range(MIN_VALID_MOVES):
                color = "B" if i % 2 == 0 else "W"
                move = await player.get_move(move_history, rules, komi, color)
                
                if move and validate_gtp_move(move):
                    valid_count += 1
                    if move.upper() not in ("PASS", "RESIGN"):
                        move_history.append([color, move])
            
            print(f"\n  KataGo player made {valid_count}/{MIN_VALID_MOVES} valid moves")
            assert valid_count >= MIN_VALID_MOVES, \
                f"KataGo only made {valid_count} valid moves, expected >= {MIN_VALID_MOVES}"
        
        finally:
            await player.stop()
    
    def test_no_llm_log_for_katago(self):
        """Test that KataGo players don't require LLM logging."""
        player = KataGoPlayer(
            model_path=KATAGO_LEVEL3_MODEL,
            gpu_id=0,
            port=8152
        )
        assert not player.requires_logging


# ============================================================================
# Game Flow Tests
# ============================================================================

class TestGameFlow:
    """Test the async game flow logic."""
    
    @pytest.mark.asyncio
    async def test_single_game_completes(self):
        """Test that a single game between KataGo players completes normally."""
        candidate = KataGoPlayer(
            model_path=KATAGO_LEVEL2_MODEL,
            gpu_id=0,
            port=8160,
            name="candidate_level2"
        )
        reference = KataGoPlayer(
            model_path=KATAGO_LEVEL1_MODEL,
            gpu_id=0,
            port=8161,
            name="reference_level1"
        )
        
        await candidate.start()
        await reference.start()
        
        try:
            result = await play_single_game(
                game_id=1,
                candidate=candidate,
                reference=reference,
                rule_name=GO_RULES[0][0],  # Japanese
                rule_string=GO_RULES[0][1],
                komi=KOMI_VALUES[1],  # 6.5
                candidate_color="B",
                max_moves=500,
                verbose=False
            )
            
            assert result is not None
            assert result.winner in ["B", "W", "draw"]
            assert result.move_count > 0
            assert result.sgf.startswith("(;GM[1]")
            
            print(f"\n  Game completed: {result.winner}+{result.win_reason}, {result.move_count} moves")
        
        finally:
            await candidate.stop()
            await reference.stop()


# ============================================================================
# Promotion Tests
# ============================================================================

class TestKataGoPromotion:
    """Test KataGo candidate promotion through levels."""
    
    def test_promotion(self, cleanup_test_dir):
        """Test that KataGo level 3 can beat lower levels and promote."""
        candidate = KataGoPlayer(
            model_path=KATAGO_LEVEL3_MODEL,
            gpu_id=0,
            port=8170
        )
        
        print(f"\nRunning promotion test: {GAMES_PER_LEVEL_TEST} games/level, max {MAX_LEVELS_TEST} levels")
        
        results = run_ladder_evaluation(
            candidate=candidate,
            manifest_path=MANIFEST_PATH,
            output_dir=TEST_OUTPUT_DIR,
            model_name="katago_level3_promotion_test",
            games_per_level=GAMES_PER_LEVEL_TEST,
            promotion_threshold=0.5,  # Lower threshold for small sample
            starting_elo=1000.0,
            candidate_gpu=0,
            reference_gpu=0,  # Same GPU for test (fewer resources needed)
            max_parallel=2,
            verbose=False,
            max_levels=MAX_LEVELS_TEST
        )
        
        # Verify results
        assert results["total_games"] > 0, "Should have played at least some games"
        assert results["highest_level"] >= 1, "Should have completed at least level 1"
        
        print(f"\n  Final Elo: {results['final_elo']:.0f}")
        print(f"  Highest Level: {results['highest_level']}")
        print(f"  Total Games: {results['total_games']}")


# ============================================================================
# Main entry point
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
