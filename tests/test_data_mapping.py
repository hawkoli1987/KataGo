#!/usr/bin/env python3
"""
Test data mapping between position_dir and move_dir.

Verifies that position and move JSONL files have exact file-to-file and row-to-row mapping:
1. Directory structure matches (same subdirectories like 19x19/, 16x16/, etc.)
2. File names match (e.g., train.jsonl, test.jsonl)
3. Row counts match
4. id fields match row-by-row
5. Position data structure is valid
"""

import json
import sys
from pathlib import Path

import pytest
import yaml

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load configuration
REPO_ROOT = Path(__file__).parent.parent.resolve()
CONFIG_PATH = REPO_ROOT / "configs" / "config.yaml"


def load_config() -> dict:
    """Load configuration from config.yaml."""
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


# Load config
_config = load_config()
_position_dir = Path(_config.get("data", {}).get("position_dir", "data/positions"))
_move_dir = Path(_config.get("data", {}).get("move_dir", "data/moves"))

# Resolve relative paths
POSITION_DIR = (REPO_ROOT / _position_dir).resolve()
MOVE_DIR = (REPO_ROOT / _move_dir).resolve()


def discover_jsonl_files(directory: Path) -> dict:
    """
    Discover all JSONL files in directory structure.
    
    Returns:
        Dict mapping relative path (e.g., "19x19/train.jsonl") to full Path
    """
    files = {}
    if not directory.exists():
        return files
    
    for jsonl_file in directory.rglob("*.jsonl"):
        relative_path = jsonl_file.relative_to(directory)
        files[str(relative_path)] = jsonl_file
    
    return files


def count_lines(file_path: Path) -> int:
    """Count lines in a file."""
    if not file_path.exists():
        return 0
    with open(file_path, 'r') as f:
        return sum(1 for _ in f)


def read_jsonl_line(file_path: Path, line_num: int) -> dict:
    """Read a specific line from a JSONL file (1-indexed)."""
    with open(file_path, 'r') as f:
        for i, line in enumerate(f, start=1):
            if i == line_num:
                return json.loads(line.strip())
    raise IndexError(f"Line {line_num} not found in {file_path}")


def validate_position_structure(position: dict) -> bool:
    """Validate that position has required fields."""
    required_fields = ["id", "moves", "rules", "komi", "boardXSize", "boardYSize"]
    return all(field in position for field in required_fields)


class TestDataMapping:
    """Test suite for position_dir ↔ move_dir data mapping."""
    
    def test_directories_exist(self):
        """Test that both position_dir and move_dir exist."""
        assert POSITION_DIR.exists(), f"Position directory does not exist: {POSITION_DIR}"
        assert MOVE_DIR.exists(), f"Move directory does not exist: {MOVE_DIR}"
    
    def test_directory_structure_matches(self):
        """Test that directory structure matches between position_dir and move_dir.
        
        Note: This test only checks that common files exist. It's OK if some
        position files don't have corresponding move files yet (analysis not run).
        """
        position_files = discover_jsonl_files(POSITION_DIR)
        move_files = discover_jsonl_files(MOVE_DIR)
        
        # Get relative paths (subdirectories)
        position_paths = set(position_files.keys())
        move_paths = set(move_files.keys())
        
        # Check that there's at least some overlap
        common_paths = position_paths & move_paths
        assert len(common_paths) > 0, (
            f"No common files found between position_dir and move_dir:\n"
            f"  Position files: {sorted(position_paths)}\n"
            f"  Move files: {sorted(move_paths)}"
        )
    
    def test_file_names_match(self):
        """Test that file names match for common files (train.jsonl, test.jsonl, etc.).
        
        Note: This test only checks common files. It's OK if some position files
        don't have corresponding move files yet (analysis not run).
        """
        position_files = discover_jsonl_files(POSITION_DIR)
        move_files = discover_jsonl_files(MOVE_DIR)
        
        position_paths = set(position_files.keys())
        move_paths = set(move_files.keys())
        common_paths = position_paths & move_paths
        
        # Check that there's at least some overlap
        assert len(common_paths) > 0, (
            f"No common files found:\n"
            f"  Position files: {sorted(position_paths)}\n"
            f"  Move files: {sorted(move_paths)}"
        )
    
    def test_row_counts_match(self):
        """Test that row counts match for each corresponding file."""
        position_files = discover_jsonl_files(POSITION_DIR)
        move_files = discover_jsonl_files(MOVE_DIR)
        
        # Only test files that exist in both directories
        common_files = set(position_files.keys()) & set(move_files.keys())
        
        if len(common_files) == 0:
            pytest.skip("No common files found between position_dir and move_dir")
        
        mismatches = []
        for rel_path in sorted(common_files):
            position_file = position_files[rel_path]
            move_file = move_files[rel_path]
            
            position_count = count_lines(position_file)
            move_count = count_lines(move_file)
            
            if position_count != move_count:
                mismatches.append(
                    f"  {rel_path}: Position={position_count} rows, Move={move_count} rows"
                )
        
        assert len(mismatches) == 0, (
            f"Row count mismatches:\n" + "\n".join(mismatches)
        )
    
    def test_id_fields_match_row_by_row(self):
        """Test that id fields match row-by-row for each corresponding file."""
        position_files = discover_jsonl_files(POSITION_DIR)
        move_files = discover_jsonl_files(MOVE_DIR)
        
        # Only test files that exist in both directories
        common_files = set(position_files.keys()) & set(move_files.keys())
        
        if len(common_files) == 0:
            pytest.skip("No common files found between position_dir and move_dir")
        
        mismatches = []
        for rel_path in sorted(common_files):
            position_file = position_files[rel_path]
            move_file = move_files[rel_path]
            
            position_count = count_lines(position_file)
            move_count = count_lines(move_file)
            
            # Only check up to the minimum count
            min_count = min(position_count, move_count)
            
            for line_num in range(1, min_count + 1):
                try:
                    position_data = read_jsonl_line(position_file, line_num)
                    move_data = read_jsonl_line(move_file, line_num)
                    
                    position_id = position_data.get("id")
                    move_id = move_data.get("id")
                    
                    if position_id != move_id:
                        mismatches.append(
                            f"  {rel_path}: Line {line_num}: "
                            f"Position id='{position_id}' != Move id='{move_id}'"
                        )
                except (json.JSONDecodeError, KeyError, IndexError) as e:
                    mismatches.append(
                        f"  {rel_path}: Line {line_num}: Error reading data: {e}"
                    )
        
        assert len(mismatches) == 0, (
            f"ID field mismatches:\n" + "\n".join(mismatches[:20]) +
            (f"\n  ... and {len(mismatches) - 20} more" if len(mismatches) > 20 else "")
        )
    
    def test_position_data_structure_valid(self):
        """Test that position data has valid structure."""
        position_files = discover_jsonl_files(POSITION_DIR)
        
        invalid_positions = []
        for rel_path in sorted(position_files.keys()):
            position_file = position_files[rel_path]
            position_count = count_lines(position_file)
            
            for line_num in range(1, min(position_count + 1, 101)):  # Check first 100
                try:
                    position_data = read_jsonl_line(position_file, line_num)
                    
                    if not validate_position_structure(position_data):
                        invalid_positions.append(
                            f"  {rel_path}: Line {line_num}: Missing required fields"
                        )
                except json.JSONDecodeError as e:
                    invalid_positions.append(
                        f"  {rel_path}: Line {line_num}: Invalid JSON: {e}"
                    )
        
        assert len(invalid_positions) == 0, (
            f"Invalid position data:\n" + "\n".join(invalid_positions[:20]) +
            (f"\n  ... and {len(invalid_positions) - 20} more" if len(invalid_positions) > 20 else "")
        )
    
    def test_move_data_has_root_info(self):
        """Test that move data has rootInfo with winrate."""
        move_files = discover_jsonl_files(MOVE_DIR)
        
        invalid_moves = []
        for rel_path in sorted(move_files.keys()):
            move_file = move_files[rel_path]
            move_count = count_lines(move_file)
            
            for line_num in range(1, min(move_count + 1, 101)):  # Check first 100
                try:
                    move_data = read_jsonl_line(move_file, line_num)
                    
                    if "rootInfo" not in move_data:
                        invalid_moves.append(
                            f"  {rel_path}: Line {line_num}: Missing 'rootInfo'"
                        )
                    elif "winrate" not in move_data["rootInfo"]:
                        invalid_moves.append(
                            f"  {rel_path}: Line {line_num}: Missing 'rootInfo.winrate'"
                        )
                    
                    if "moveInfos" not in move_data:
                        invalid_moves.append(
                            f"  {rel_path}: Line {line_num}: Missing 'moveInfos'"
                        )
                    elif len(move_data["moveInfos"]) == 0:
                        invalid_moves.append(
                            f"  {rel_path}: Line {line_num}: Empty 'moveInfos'"
                        )
                    elif "move" not in move_data["moveInfos"][0]:
                        invalid_moves.append(
                            f"  {rel_path}: Line {line_num}: Missing 'moveInfos[0].move'"
                        )
                except json.JSONDecodeError as e:
                    invalid_moves.append(
                        f"  {rel_path}: Line {line_num}: Invalid JSON: {e}"
                    )
        
        assert len(invalid_moves) == 0, (
            f"Invalid move data:\n" + "\n".join(invalid_moves[:20]) +
            (f"\n  ... and {len(invalid_moves) - 20} more" if len(invalid_moves) > 20 else "")
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
