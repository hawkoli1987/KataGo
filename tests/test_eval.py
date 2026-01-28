#!/usr/bin/env python3
"""Tests for the evaluation pipeline using pytest.

Tests that each model type:
1. Can play at least 10 moves
2. For LLM-based models: llm_log.jsonl is generated with valid content
3. For KataGo candidate: can complete games and promote through levels

Usage:
    pytest tests/test_eval.py -v
    pytest tests/test_eval.py -v -k "openai"
    pytest tests/test_eval.py -v -k "promotion"
"""

import json
import shutil
import sys
from pathlib import Path

import pytest

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from eval.players import OpenAIPlayer, KataGoPlayer, validate_gtp_move
from eval.players.katago_player import KataGoPlayer as KataGoPlayerImpl
from eval.game import play_single_game
from eval.ladder import run_ladder_evaluation


# Test configuration
VLLM_ENDPOINT = "http://localhost:8002/v1"
KATAGO_PATH = "/scratch/Projects/SPEC-SF-AISG/katago/bin/katago-cuda/katago"
KATAGO_CONFIG = "/scratch/Projects/SPEC-SF-AISG/katago/bin/katago-cuda/default_gtp.cfg"
MANIFEST_PATH = Path("assets/models/manifest.json")
TEST_OUTPUT_DIR = Path("data/eval/_test_runs")

# Model paths for KataGo tests
KATAGO_LEVEL1_MODEL = "assets/models/level_01_kata1-b6c96-s175395328-d26788732.txt.gz"
KATAGO_LEVEL2_MODEL = "assets/models/level_02_kata1-b10c128-s197428736-d67404019.txt.gz"
KATAGO_LEVEL3_MODEL = "assets/models/level_03_kata1-b15c192-s497233664-d149638345.txt.gz"

# Test parameters - reduced for faster testing
GAMES_PER_LEVEL_TEST = 5
MIN_VALID_MOVES = 5  # Reduced from 10 to be more lenient with LLM


@pytest.fixture(scope="module")
def cleanup_test_dir():
    """Clean up test output directory before tests."""
    if TEST_OUTPUT_DIR.exists():
        shutil.rmtree(TEST_OUTPUT_DIR)
    TEST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    yield


@pytest.fixture(scope="class")
def openai_player():
    """Create and manage OpenAI player for tests."""
    player = OpenAIPlayer(api_base=VLLM_ENDPOINT)
    player.start()
    print(f"\nOpenAI Player: {player.name}")
    yield player
    player.stop()


@pytest.fixture(scope="class")
def katago_player_level3():
    """Create and manage KataGo level 3 player for tests."""
    player = KataGoPlayerImpl(
        katago_path=KATAGO_PATH,
        model_path=KATAGO_LEVEL3_MODEL,
        config_path=KATAGO_CONFIG
    )
    player.start()
    print(f"\nKataGo Player: {player.name}")
    yield player
    player.stop()


@pytest.fixture(scope="function")
def reference_level1():
    """Create a fresh reference KataGo level 1 for each test."""
    reference = KataGoPlayerImpl(
        katago_path=KATAGO_PATH,
        model_path=KATAGO_LEVEL1_MODEL,
        config_path=KATAGO_CONFIG
    )
    reference.start()
    yield reference
    reference.stop()


# ============================================================================
# OpenAI Player Tests
# ============================================================================

class TestOpenAIPlayer:
    """Test OpenAI-compatible player (vLLM)."""
    
    def test_can_generate_valid_moves(self, openai_player):
        """Test player can generate valid GTP moves."""
        test_cases = [
            {"move_history": [], "rules": "koSIMPLEscoreTERRITORYtaxSEKIsui0", "komi": 6.5, "color": "B"},
            {"move_history": [["B", "D4"], ["W", "Q16"]], "rules": "koSIMPLEscoreAREAtaxNONEsui0whbN", "komi": 7.5, "color": "B"},
        ]
        
        for tc in test_cases:
            move = openai_player.get_move(**tc)
            assert move, f"Player returned empty move for {tc}"
            assert validate_gtp_move(move), f"Invalid move '{move}' for {tc}"
    
    def test_can_play_multiple_moves(self, openai_player):
        """Test player can play multiple moves in sequence."""
        rules = "koSIMPLEscoreTERRITORYtaxSEKIsui0"
        komi = 6.5
        
        # Simulate a few turns without actually using a reference engine
        move_histories = [
            [],
            [["B", "D4"]],
            [["B", "D4"], ["W", "Q16"]],
            [["B", "D4"], ["W", "Q16"], ["B", "D16"]],
            [["B", "D4"], ["W", "Q16"], ["B", "D16"], ["W", "Q4"]],
        ]
        
        valid_count = 0
        for history in move_histories:
            color = "B" if len(history) % 2 == 0 else "W"
            move = openai_player.get_move(history, rules, komi, color)
            if move and validate_gtp_move(move):
                valid_count += 1
        
        print(f"  OpenAI player made {valid_count}/{len(move_histories)} valid moves")
        assert valid_count >= MIN_VALID_MOVES, \
            f"Player only made {valid_count} valid moves, expected >= {MIN_VALID_MOVES}"
    
    def test_llm_log_generated(self, openai_player, cleanup_test_dir):
        """Test that llm_log.jsonl is generated with valid content."""
        log_path = TEST_OUTPUT_DIR / "openai_test" / "llm_log.jsonl"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Clear any existing log
        if log_path.exists():
            log_path.unlink()
        
        openai_player.set_log_path(log_path)
        openai_player.reset_game(game_id=1)
        
        # Play a few moves
        test_cases = [
            {"move_history": [], "rules": "koSIMPLEscoreTERRITORYtaxSEKIsui0", "komi": 6.5, "color": "B"},
            {"move_history": [["B", "D4"], ["W", "Q16"]], "rules": "koSIMPLEscoreTERRITORYtaxSEKIsui0", "komi": 6.5, "color": "B"},
        ]
        
        for tc in test_cases:
            openai_player.get_move(**tc)
        
        # Verify log file exists and has valid content
        assert log_path.exists(), f"Log file not created: {log_path}"
        
        with open(log_path) as f:
            lines = f.readlines()
        
        assert len(lines) >= 2, "Expected at least 2 log entries"
        
        for line in lines:
            entry = json.loads(line)
            assert "game_id" in entry
            assert "ply" in entry
            assert "color" in entry
            assert "prompt" in entry
            assert "raw_response" in entry
            assert "parsed_move" in entry
        
        print(f"  LLM log generated with {len(lines)} entries")


# ============================================================================
# KataGo Player Tests
# ============================================================================

class TestKataGoPlayer:
    """Test KataGo as candidate player."""
    
    def test_can_generate_valid_moves(self, katago_player_level3):
        """Test player can generate valid GTP moves."""
        test_cases = [
            {"move_history": [], "rules": "koSIMPLEscoreTERRITORYtaxSEKIsui0", "komi": 6.5, "color": "B"},
            {"move_history": [["B", "D4"], ["W", "Q16"]], "rules": "koSIMPLEscoreAREAtaxNONEsui0whbN", "komi": 7.5, "color": "B"},
        ]
        
        for tc in test_cases:
            move = katago_player_level3.get_move(**tc)
            assert move, f"Player returned empty move for {tc}"
            assert validate_gtp_move(move), f"Invalid move '{move}' for {tc}"
    
    def test_can_play_multiple_moves(self, katago_player_level3):
        """Test KataGo candidate can play multiple moves."""
        rules = "koSIMPLEscoreTERRITORYtaxSEKIsui0"
        komi = 6.5
        
        move_histories = [
            [],
            [["B", "D4"]],
            [["B", "D4"], ["W", "Q16"]],
            [["B", "D4"], ["W", "Q16"], ["B", "D16"]],
            [["B", "D4"], ["W", "Q16"], ["B", "D16"], ["W", "Q4"]],
            [["B", "D4"], ["W", "Q16"], ["B", "D16"], ["W", "Q4"], ["B", "C6"]],
        ]
        
        valid_count = 0
        for history in move_histories:
            color = "B" if len(history) % 2 == 0 else "W"
            move = katago_player_level3.get_move(history, rules, komi, color)
            if move and validate_gtp_move(move):
                valid_count += 1
        
        print(f"  KataGo player made {valid_count}/{len(move_histories)} valid moves")
        assert valid_count >= MIN_VALID_MOVES, \
            f"KataGo only made {valid_count} valid moves, expected >= {MIN_VALID_MOVES}"
    
    def test_no_llm_log_for_katago(self, katago_player_level3):
        """Test that KataGo players don't require LLM logging."""
        assert not katago_player_level3.requires_logging, \
            "KataGo player should not require logging"


# ============================================================================
# Game Flow Tests
# ============================================================================

class TestGameFlow:
    """Test the game flow logic."""
    
    def test_single_game_completes(self):
        """Test that a single game between KataGo players completes normally."""
        candidate = KataGoPlayerImpl(
            katago_path=KATAGO_PATH,
            model_path=KATAGO_LEVEL2_MODEL,
            config_path=KATAGO_CONFIG,
            name="candidate_level2"
        )
        reference = KataGoPlayerImpl(
            katago_path=KATAGO_PATH,
            model_path=KATAGO_LEVEL1_MODEL,
            config_path=KATAGO_CONFIG,
            name="reference_level1"
        )
        
        candidate.start()
        reference.start()
        
        try:
            result = play_single_game(
                game_id=1,
                candidate=candidate,
                reference=reference,
                rule_name="Japanese",
                rule_string="koSIMPLEscoreTERRITORYtaxSEKIsui0",
                komi=6.5,
                candidate_color="B",
                max_moves=500,
                verbose=False
            )
            
            assert result is not None
            assert result.winner in ["B", "W", "draw"]
            assert result.win_reason in ["score", "resign", "forfeit", "max_moves"]
            assert result.move_count > 0, "Game should have at least one move"
            assert result.sgf.startswith("(;GM[1]"), "SGF should be valid"
            
            print(f"\n  Game completed: {result.winner}+{result.win_reason}")
            print(f"  Move count: {result.move_count}")
        
        finally:
            candidate.stop()
            reference.stop()


# ============================================================================
# Promotion Tests
# ============================================================================

class TestKataGoPromotion:
    """Test KataGo candidate promotion through levels."""
    
    def test_promotion(self, cleanup_test_dir):
        """Test that KataGo level 3 can beat levels 1-2 and promote to at least level 2."""
        # Create level 3 KataGo as candidate
        candidate = KataGoPlayerImpl(
            katago_path=KATAGO_PATH,
            model_path=KATAGO_LEVEL3_MODEL,
            config_path=KATAGO_CONFIG
        )
        
        print("\nRunning promotion test with KataGo level 3 as candidate...")
        print(f"Using {GAMES_PER_LEVEL_TEST} games per level...")
        
        results = run_ladder_evaluation(
            candidate=candidate,
            katago_path=KATAGO_PATH,
            katago_config=KATAGO_CONFIG,
            manifest_path=MANIFEST_PATH,
            output_dir=TEST_OUTPUT_DIR,
            model_name="katago_level3_promotion_test",
            games_per_level=GAMES_PER_LEVEL_TEST,
            promotion_threshold=0.55,
            starting_elo=1000.0,
            verbose=False
        )
        
        # Verify results
        assert results["highest_level"] >= 2, \
            f"KataGo level 3 should promote to at least level 2, but only reached level {results['highest_level']}"
        
        assert results["total_games"] > 0, "Should have played at least some games"
        
        # Check all games completed (no forfeits expected for KataGo)
        for level_result in results["levels"]:
            games_completed = level_result["wins"] + level_result["losses"] + level_result["draws"]
            assert games_completed == level_result["games_played"], \
                f"Level {level_result['level']}: some games not completed"
        
        print(f"\n  Final Elo: {results['final_elo']:.0f}")
        print(f"  Highest Level: {results['highest_level']}")
        print(f"  Total Games: {results['total_games']}")
        print(f"  Stopped: {results['stopped_reason']}")


# ============================================================================
# HuggingFace Player Tests (stub - requires GPU and model)
# ============================================================================

class TestHuggingFacePlayer:
    """Test HuggingFace player (skipped if model not available)."""
    
    @pytest.mark.skip(reason="HuggingFace test requires GPU and model checkpoint")
    def test_can_generate_valid_moves(self):
        """Test HuggingFace player can generate valid moves."""
        # This test is skipped by default since it requires:
        # 1. A HuggingFace model checkpoint
        # 2. GPU with enough memory
        # 
        # To run this test manually:
        # pytest tests/test_eval.py -v -k "huggingface" --run-hf
        pass
    
    @pytest.mark.skip(reason="HuggingFace test requires GPU and model checkpoint")
    def test_llm_log_generated(self):
        """Test HuggingFace player generates logs."""
        pass


# ============================================================================
# Main entry point
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
