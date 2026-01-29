#!/usr/bin/env python3
"""Simplified evaluation tests - only promotion tests for KataGo and OpenAI.

Tests ladder evaluation (promotion through levels) for both KataGo and OpenAI candidates.
Configuration is loaded from configs/config.yaml.

Usage:
    # Run all promotion tests (requires servers via `make run_all`)
    pytest tests/test_eval.py -v
    
    # Run only KataGo promotion test
    pytest tests/test_eval.py -v -k "katago"
    
    # Run only OpenAI promotion test
    pytest tests/test_eval.py -v -k "openai"
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

from eval.players import OpenAIPlayer, KataGoPlayer
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
MAX_MOVES_PER_GAME = _test_cfg.get("max_moves_per_game", 100)
TEST_OUTPUT_DIR = REPO_ROOT / "data" / "eval" / "_test_runs"

# Paths from config
MANIFEST_PATH = REPO_ROOT / _config.get("eval", {}).get("manifest_path", "assets/models/manifest.json")


@pytest.fixture(scope="module")
def cleanup_test_dir():
    """Clean up test output directory before tests.
    
    Note: This fixture cleans up the directory before tests run,
    but does NOT clean up after tests complete, so results persist.
    """
    if TEST_OUTPUT_DIR.exists():
        shutil.rmtree(TEST_OUTPUT_DIR)
    TEST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    yield
    # Don't clean up after - keep results for inspection


@pytest.fixture(scope="module")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# ============================================================================
# Promotion Tests
# ============================================================================

class TestKataGoPromotion:
    """Test KataGo candidate promotion through levels.
    
    Requires:
    1. KataGo servers running via `make run_all`
    2. assets/models/manifest.json with level models
    
    Skip if manifest doesn't exist.
    """
    
    @pytest.mark.skipif(not MANIFEST_PATH.exists(), reason="Manifest not found")
    def test_promotion(self, cleanup_test_dir):
        """Test that a KataGo candidate can progress through ladder levels."""
        candidate = KataGoPlayer(endpoint=CANDIDATE_ENDPOINT, name="katago_candidate")
        
        print(f"\nRunning KataGo promotion test: {GAMES_PER_LEVEL_TEST} games/level, max {MAX_LEVELS_TEST} levels")
        print(f"  Candidate endpoint: {CANDIDATE_ENDPOINT}")
        print(f"  Reference endpoint: {REFERENCE_ENDPOINT}")
        
        results = run_ladder_evaluation(
            candidate=candidate,
            manifest_path=MANIFEST_PATH,
            output_dir=TEST_OUTPUT_DIR,
            model_name="katago_promotion_test",
            games_per_level=GAMES_PER_LEVEL_TEST,
            promotion_threshold=0.5,
            starting_elo=1000.0,
            max_parallel=2,
            max_moves_per_game=MAX_MOVES_PER_GAME,
            verbose=False,
            max_levels=MAX_LEVELS_TEST
        )
        
        # Verify results
        assert results["total_games"] > 0, "Should have played at least some games"
        assert results["highest_level"] >= 1, "Should have completed at least level 1"
        
        # Verify game files were saved
        run_dir = TEST_OUTPUT_DIR / "katago_promotion_test"
        games_dir = run_dir / "games"
        assert games_dir.exists(), f"Games directory not created: {games_dir}"
        
        # Check that SGF files exist
        sgf_files = list(games_dir.glob("level_*/game_*.sgf"))
        assert len(sgf_files) > 0, f"No SGF files found in {games_dir}"
        
        # Verify at least one SGF has content
        for sgf_file in sgf_files[:3]:
            with open(sgf_file) as f:
                content = f.read()
                assert content.startswith("(;GM[1]"), f"SGF file {sgf_file} has invalid content"
                assert len(content) > 50, f"SGF file {sgf_file} seems too short"
        
        print(f"\n  Final Elo: {results['final_elo']:.0f}")
        print(f"  Highest Level: {results['highest_level']}")
        print(f"  Total Games: {results['total_games']}")
        print(f"  SGF files saved: {len(sgf_files)}")


class TestOpenAIPromotion:
    """Test OpenAI candidate promotion through levels.
    
    Requires:
    1. KataGo reference servers running via `make run_all`
    2. OpenAI/vLLM endpoint accessible
    3. assets/models/manifest.json with level models
    
    Skip if manifest doesn't exist.
    """
    
    @pytest.mark.skipif(not MANIFEST_PATH.exists(), reason="Manifest not found")
    def test_promotion(self, cleanup_test_dir):
        """Test that an OpenAI candidate can progress through ladder levels."""
        candidate = OpenAIPlayer(api_base=OPENAI_ENDPOINT)
        
        print(f"\nRunning OpenAI promotion test: {GAMES_PER_LEVEL_TEST} games/level, max {MAX_LEVELS_TEST} levels")
        print(f"  Candidate endpoint: {OPENAI_ENDPOINT}")
        print(f"  Reference endpoint: {REFERENCE_ENDPOINT}")
        
        results = run_ladder_evaluation(
            candidate=candidate,
            manifest_path=MANIFEST_PATH,
            output_dir=TEST_OUTPUT_DIR,
            model_name="openai_promotion_test",
            games_per_level=GAMES_PER_LEVEL_TEST,
            promotion_threshold=0.5,
            starting_elo=1000.0,
            max_parallel=2,
            max_moves_per_game=MAX_MOVES_PER_GAME,
            verbose=False,
            max_levels=MAX_LEVELS_TEST
        )
        
        # Verify results
        assert results["total_games"] > 0, "Should have played at least some games"
        assert results["highest_level"] >= 1, "Should have completed at least level 1"
        
        # Verify game files were saved
        run_dir = TEST_OUTPUT_DIR / "openai_promotion_test"
        games_dir = run_dir / "games"
        assert games_dir.exists(), f"Games directory not created: {games_dir}"
        
        # Check that SGF files exist
        sgf_files = list(games_dir.glob("level_*/game_*.sgf"))
        assert len(sgf_files) > 0, f"No SGF files found in {games_dir}"
        
        # Verify at least one SGF has content
        for sgf_file in sgf_files[:3]:
            with open(sgf_file) as f:
                content = f.read()
                assert content.startswith("(;GM[1]"), f"SGF file {sgf_file} has invalid content"
                assert len(content) > 50, f"SGF file {sgf_file} seems too short"
        
        print(f"\n  Final Elo: {results['final_elo']:.0f}")
        print(f"  Highest Level: {results['highest_level']}")
        print(f"  Total Games: {results['total_games']}")
        print(f"  SGF files saved: {len(sgf_files)}")


# ============================================================================
# Main entry point
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
