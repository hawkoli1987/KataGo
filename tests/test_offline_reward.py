#!/usr/bin/env python3
"""
Tests for offline training reward function.

Tests the three-component reward system: r_wr, r_move, r_legal.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from train.offline_env.reward import (
    compute_score,
    extract_root_winrate_and_move,
    normalize_move,
    load_reward_weights
)


class TestNormalizeMove:
    """Test move normalization."""
    
    def test_normalize_uppercase(self):
        assert normalize_move("D4") == "d4"
        assert normalize_move("Q16") == "q16"
        assert normalize_move("PASS") == "pass"
    
    def test_normalize_lowercase(self):
        assert normalize_move("d4") == "d4"
        assert normalize_move("pass") == "pass"
    
    def test_normalize_with_spaces(self):
        assert normalize_move("  D4  ") == "d4"
        assert normalize_move("  pass  ") == "pass"
    
    def test_normalize_empty(self):
        assert normalize_move("") == ""
        assert normalize_move(None) == ""


class TestExtractRootWinrateAndMove:
    """Test extraction of root_winrate and top_move from LLM response."""
    
    def test_clean_json(self):
        response = '{"root_winrate": 0.65, "top_move": "D4"}'
        winrate, move = extract_root_winrate_and_move(response)
        assert winrate == 0.65
        assert move == "d4"
    
    def test_json_with_reasoning(self):
        response = 'The position looks balanced. {"root_winrate": 0.55, "top_move": "Q16"}'
        winrate, move = extract_root_winrate_and_move(response)
        assert winrate == 0.55
        assert move == "q16"
    
    def test_thinking_format(self):
        response = '<think>Let me analyze...</think>{"root_winrate": 0.60, "top_move": "E4"}'
        winrate, move = extract_root_winrate_and_move(response)
        assert winrate == 0.60
        assert move == "e4"
    
    def test_case_insensitive_fields(self):
        response = '{"rootWinrate": 0.65, "topMove": "D4"}'
        winrate, move = extract_root_winrate_and_move(response)
        assert winrate == 0.65
        assert move == "d4"
    
    def test_clamp_winrate(self):
        response = '{"root_winrate": 1.5, "top_move": "D4"}'  # > 1.0
        winrate, move = extract_root_winrate_and_move(response)
        assert winrate == 1.0
        
        response = '{"root_winrate": -0.5, "top_move": "D4"}'  # < 0.0
        winrate, move = extract_root_winrate_and_move(response)
        assert winrate == 0.0
    
    def test_invalid_json(self):
        response = "Invalid response"
        winrate, move = extract_root_winrate_and_move(response)
        assert winrate is None
        assert move is None
    
    def test_missing_fields(self):
        response = '{"root_winrate": 0.65}'  # Missing top_move
        winrate, move = extract_root_winrate_and_move(response)
        assert winrate == 0.65
        assert move is None


class TestComputeScore:
    """Test reward computation."""
    
    def test_perfect_match(self):
        """Perfect prediction: r_wr=0, r_move=1, r_legal=0 -> total=1.0"""
        response = '{"root_winrate": 0.65, "top_move": "D4"}'
        extra_info = {
            "root_winrate": 0.65,
            "top_move": "D4",
            "move_error": None
        }
        score = compute_score("katago/offline", response, "0.65", extra_info)
        assert score == pytest.approx(1.0, abs=0.01)  # r_move=1, others=0
    
    def test_winrate_error(self):
        """Winrate off by 0.1: r_wr=-0.01, r_move=1, r_legal=0"""
        response = '{"root_winrate": 0.60, "top_move": "D4"}'
        extra_info = {
            "root_winrate": 0.65,
            "top_move": "D4",
            "move_error": None
        }
        score = compute_score("katago/offline", response, "0.65", extra_info)
        # r_wr = -(0.60 - 0.65)^2 = -0.0025
        # r_move = 1
        # r_legal = 0
        # total = 1.0 * (-0.0025) + 1.0 * 1.0 + 1.0 * 0.0 = 0.9975
        assert score == pytest.approx(0.9975, abs=0.01)
    
    def test_move_mismatch(self):
        """Wrong move: r_wr=0, r_move=0, r_legal=0 -> total=0.0"""
        response = '{"root_winrate": 0.65, "top_move": "E4"}'
        extra_info = {
            "root_winrate": 0.65,
            "top_move": "D4",
            "move_error": None
        }
        score = compute_score("katago/offline", response, "0.65", extra_info)
        assert score == pytest.approx(0.0, abs=0.01)
    
    def test_illegal_move(self):
        """Illegal move: r_wr=-1, r_move=0, r_legal=-1 -> total=-2.0"""
        response = '{"root_winrate": 0.65, "top_move": "XX"}'
        extra_info = {
            "root_winrate": 0.65,
            "top_move": "D4",
            "move_error": "Illegal move"
        }
        score = compute_score("katago/offline", response, "0.65", extra_info)
        # r_wr = -1, r_move = 0, r_legal = -1
        # total = 1.0 * (-1) + 1.0 * 0 + 1.0 * (-1) = -2.0
        assert score == pytest.approx(-2.0, abs=0.01)
    
    def test_invalid_format(self):
        """Invalid format: r_wr=-1, r_move=0, r_legal=0 -> total=-1.0"""
        response = "Invalid response"
        extra_info = {
            "root_winrate": 0.65,
            "top_move": "D4",
            "move_error": None
        }
        score = compute_score("katago/offline", response, "0.65", extra_info)
        assert score == pytest.approx(-1.0, abs=0.01)
    
    def test_move_case_insensitive(self):
        """Move comparison should be case-insensitive."""
        response = '{"root_winrate": 0.65, "top_move": "d4"}'  # lowercase
        extra_info = {
            "root_winrate": 0.65,
            "top_move": "D4",  # uppercase
            "move_error": None
        }
        score = compute_score("katago/offline", response, "0.65", extra_info)
        # Should match (case-insensitive)
        assert score == pytest.approx(1.0, abs=0.01)
    
    def test_thinking_mode_extraction(self):
        """Test extraction from thinking mode response."""
        response = '<think>Let me analyze the position...</think>{"root_winrate": 0.55, "top_move": "Q16"}'
        extra_info = {
            "root_winrate": 0.55,
            "top_move": "Q16",
            "move_error": None
        }
        score = compute_score("katago/offline", response, "0.55", extra_info)
        assert score == pytest.approx(1.0, abs=0.01)
    
    def test_reward_components_stored(self):
        """Test that individual reward components are stored in extra_info."""
        response = '{"root_winrate": 0.60, "top_move": "D4"}'
        extra_info = {
            "root_winrate": 0.65,
            "top_move": "D4",
            "move_error": None
        }
        score = compute_score("katago/offline", response, "0.65", extra_info)
        
        # Check that components are stored
        assert "_reward_components" in extra_info
        components = extra_info["_reward_components"]
        assert "r_wr" in components
        assert "r_move" in components
        assert "r_legal" in components
        assert "reward_total" in components
        assert components["reward_total"] == score


class TestLoadRewardWeights:
    """Test loading reward weights from config."""
    
    def test_load_weights(self):
        """Test that weights can be loaded."""
        weights = load_reward_weights()
        assert "r_wr_weight" in weights
        assert "r_move_weight" in weights
        assert "r_legal_weight" in weights
        assert isinstance(weights["r_wr_weight"], float)
        assert isinstance(weights["r_move_weight"], float)
        assert isinstance(weights["r_legal_weight"], float)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
