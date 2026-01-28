#!/usr/bin/env python3
"""Tests for the async evaluation pipeline using pytest.

Tests that each model type can play games via HTTP endpoints.
Both candidate and reference use FastAPI servers for async parallel execution.

Configuration is loaded from configs/config.yaml.
Environment variables can override config values:
    OPENAI_API_BASE:     Override OpenAI endpoint
    CANDIDATE_ENDPOINT:  Override candidate KataGo endpoint  
    REFERENCE_ENDPOINT:  Override reference KataGo endpoint

Usage:
    # Run all tests (requires servers via `make run_all`)
    pytest tests/test_eval.py -v
    
    # Run only OpenAI tests
    pytest tests/test_eval.py -v -k "openai"
    
    # Run only KataGo tests  
    pytest tests/test_eval.py -v -k "katago"
"""

import asyncio
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from eval.players import OpenAIPlayer, KataGoPlayer, validate_gtp_move
from eval.game import play_single_game, GO_RULES, KOMI_VALUES
from eval.ladder import run_ladder_evaluation


# =============================================================================
# Configuration Loading
# =============================================================================

REPO_ROOT = Path(__file__).parent.parent.resolve()
CONFIG_PATH = REPO_ROOT / "configs" / "config.yaml"
MANAGE_SCRIPT = REPO_ROOT / "runtime" / "manage_servers.sh"


def load_config() -> dict:
    """Load configuration from config.yaml."""
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


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


# Load configuration
_config = load_config()
_test_cfg = _config.get("test", {})
_servers_cfg = _config.get("servers", {})

# Server endpoints (can be overridden via environment variables)
OPENAI_ENDPOINT = os.environ.get(
    "OPENAI_API_BASE", 
    _servers_cfg.get("openai", {}).get("primary", "http://localhost:8002/v1")
)
OPENAI_ENDPOINT_2 = os.environ.get(
    "OPENAI_API_BASE2",
    _servers_cfg.get("openai", {}).get("secondary", "http://localhost:8002/v1")
)

# KataGo endpoints - try to get from manage_servers.sh first, then config, then env
_dynamic_endpoints = get_server_endpoints()
CANDIDATE_ENDPOINT = os.environ.get(
    "CANDIDATE_ENDPOINT",
    _dynamic_endpoints.get("candidate_endpoint") or 
    f"http://localhost:{_servers_cfg.get('candidate', {}).get('port', 9200)}"
)
REFERENCE_ENDPOINT = os.environ.get(
    "REFERENCE_ENDPOINT", 
    _dynamic_endpoints.get("reference_endpoint") or
    f"http://localhost:{_servers_cfg.get('reference', {}).get('port', 9201)}"
)

# Test parameters from config
GAMES_PER_LEVEL_TEST = _test_cfg.get("games_per_level", 5)
MAX_LEVELS_TEST = _test_cfg.get("max_levels", 3)
MIN_VALID_MOVES = _test_cfg.get("min_valid_moves", 10)
MAX_MOVES_PER_GAME = _test_cfg.get("max_moves_per_game", 100)
TEST_OUTPUT_DIR = Path(_test_cfg.get("output_dir", "/tmp/katago_test_runs"))

# Paths from config
MANIFEST_PATH = REPO_ROOT / _config.get("eval", {}).get("manifest_path", "assets/models/manifest.json")


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
        player = OpenAIPlayer(api_base=OPENAI_ENDPOINT)
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
        player = OpenAIPlayer(api_base=OPENAI_ENDPOINT)
        await player.start()
        
        try:
            rules = GO_RULES[0][1]  # Japanese
            komi = KOMI_VALUES[1]  # 6.5
            
            # Build a realistic opening sequence
            move_history = []
            valid_count = 0
            
            # Play moves alternating colors
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
        
        player = OpenAIPlayer(api_base=OPENAI_ENDPOINT)
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
# KataGo Player Tests (requires running KataGo servers via `make run_all`)
# ============================================================================

class TestKataGoPlayer:
    """Test KataGo player via remote FastAPI server.
    
    Requires: KataGo servers running at CANDIDATE_ENDPOINT and REFERENCE_ENDPOINT
    Start servers with: `make run_all`
    """
    
    @pytest.mark.asyncio
    async def test_can_generate_valid_moves(self):
        """Test player can generate valid GTP moves with real game rules."""
        player = KataGoPlayer(endpoint=CANDIDATE_ENDPOINT, name="katago_test")
        
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
        player = KataGoPlayer(endpoint=CANDIDATE_ENDPOINT, name="katago_multi")
        
        await player.start()
        
        try:
            rules = GO_RULES[0][1]  # Japanese
            komi = KOMI_VALUES[1]  # 6.5
            
            # Build a realistic game sequence
            move_history = []
            valid_count = 0
            
            # Play moves
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
        player = KataGoPlayer(endpoint=CANDIDATE_ENDPOINT)
        assert not player.requires_logging


# ============================================================================
# Game Flow Tests (requires running KataGo servers)
# ============================================================================

class TestGameFlow:
    """Test the async game flow logic using remote KataGo servers."""
    
    @pytest.mark.asyncio
    async def test_single_game_completes(self):
        """Test that a single game between KataGo players completes normally.
        
        Uses candidate and reference endpoints for the two players.
        """
        candidate = KataGoPlayer(endpoint=CANDIDATE_ENDPOINT, name="candidate")
        reference = KataGoPlayer(endpoint=REFERENCE_ENDPOINT, name="reference")
        
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
                max_moves=MAX_MOVES_PER_GAME,
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
# Promotion Tests (requires running KataGo servers + manifest)
# ============================================================================

class TestKataGoPromotion:
    """Test KataGo candidate promotion through levels.
    
    Note: This test requires:
    1. KataGo servers running via `make run_all`
    2. assets/models/manifest.json with level models
    
    Skip if manifest doesn't exist.
    """
    
    @pytest.mark.skipif(not MANIFEST_PATH.exists(), reason="Manifest not found")
    def test_promotion(self, cleanup_test_dir):
        """Test that a candidate can progress through ladder levels."""
        # Use remote endpoint for candidate
        candidate = KataGoPlayer(endpoint=CANDIDATE_ENDPOINT, name="candidate_test")
        
        print(f"\nRunning promotion test: {GAMES_PER_LEVEL_TEST} games/level, max {MAX_LEVELS_TEST} levels")
        print(f"  Candidate endpoint: {CANDIDATE_ENDPOINT}")
        print(f"  Reference endpoint: {REFERENCE_ENDPOINT}")
        
        results = run_ladder_evaluation(
            candidate=candidate,
            manifest_path=MANIFEST_PATH,
            output_dir=TEST_OUTPUT_DIR,
            model_name="promotion_test",
            games_per_level=GAMES_PER_LEVEL_TEST,
            promotion_threshold=0.5,  # Lower threshold for small sample
            starting_elo=1000.0,
            max_parallel=2,
            max_moves_per_game=MAX_MOVES_PER_GAME,
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
