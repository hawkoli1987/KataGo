#!/usr/bin/env python3
"""
Offline analysis script for KataGo positions.

Takes extracted parquet files, runs analysis via KataGo engine endpoint,
and appends analysis results (top moves, child winrates, PVs, etc.) to the parquet.

Usage:
    python analyze_offline.py --input-dir /path/to/parquet_dir --endpoint http://hopper-34:9100
"""

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests
import yaml


@dataclass
class AnalysisConfig:
    """Configuration for analysis."""
    endpoint: str
    timeout: float = 30.0
    max_workers: int = 8
    batch_size: int = 100
    retry_count: int = 3
    retry_delay: float = 1.0


@dataclass 
class MoveInfo:
    """Analysis result for a single move."""
    move: str
    order: int
    visits: int
    winrate: float
    scoreLead: float
    prior: float
    pv: List[str] = field(default_factory=list)
    utility: Optional[float] = None
    lcb: Optional[float] = None


@dataclass
class AnalysisResult:
    """Complete analysis result for a position."""
    position_id: str
    success: bool
    error_message: Optional[str] = None
    current_player: Optional[str] = None
    root_visits: Optional[int] = None
    root_winrate: Optional[float] = None  # From rootInfo (before making any move)
    root_scoreLead: Optional[float] = None
    root_utility: Optional[float] = None
    move_infos: List[MoveInfo] = field(default_factory=list)
    raw_response: Optional[str] = None


def load_config(config_path: Path) -> dict:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def call_katago_api(
    endpoint: str,
    query: Dict[str, Any],
    timeout: float = 30.0
) -> Dict[str, Any]:
    """
    Call KataGo analysis API.
    
    Args:
        endpoint: Full URL to the analysis endpoint (e.g., http://hopper-34:9100/analysis)
        query: KataGo query dict
        timeout: Request timeout in seconds
        
    Returns:
        API response dict
        
    Raises:
        requests.RequestException: On network/HTTP errors
    """
    # KataGo analysis engine expects JSONL (one JSON per line)
    response = requests.post(
        endpoint,
        data=json.dumps(query),
        headers={'Content-Type': 'application/json'},
        timeout=timeout
    )
    response.raise_for_status()
    return response.json()


def parse_analysis_response(response: Dict[str, Any], position_id: str) -> AnalysisResult:
    """Parse KataGo analysis response into structured result."""
    
    # Check for error
    if 'error' in response:
        return AnalysisResult(
            position_id=position_id,
            success=False,
            error_message=response['error'],
            raw_response=json.dumps(response)
        )
    
    # Extract rootInfo
    root_info = response.get('rootInfo', {})
    
    # Extract moveInfos
    move_infos = []
    for m in response.get('moveInfos', []):
        move_infos.append(MoveInfo(
            move=m.get('move', ''),
            order=m.get('order', 0),
            visits=m.get('visits', 0),
            winrate=m.get('winrate', 0.0),
            scoreLead=m.get('scoreLead', 0.0),
            prior=m.get('prior', 0.0),
            pv=m.get('pv', []),
            utility=m.get('utility'),
            lcb=m.get('lcb')
        ))
    
    return AnalysisResult(
        position_id=position_id,
        success=True,
        current_player=root_info.get('currentPlayer'),
        root_visits=root_info.get('visits'),
        root_winrate=root_info.get('winrate'),
        root_scoreLead=root_info.get('scoreLead'),
        root_utility=root_info.get('utility'),
        move_infos=move_infos,
        raw_response=json.dumps(response)
    )


def analyze_position(
    row: pd.Series,
    config: AnalysisConfig
) -> AnalysisResult:
    """
    Analyze a single position from the parquet data.
    
    Args:
        row: DataFrame row containing position data
        config: Analysis configuration
        
    Returns:
        AnalysisResult with analysis data
    """
    # Extract katago_query from extra_info
    extra_info = row.get('extra_info', {})
    if isinstance(extra_info, str):
        extra_info = json.loads(extra_info)
    
    katago_query_str = extra_info.get('katago_query', '')
    position_id = extra_info.get('id', 'unknown')
    
    if not katago_query_str:
        return AnalysisResult(
            position_id=position_id,
            success=False,
            error_message="No katago_query in extra_info"
        )
    
    query = json.loads(katago_query_str)
    
    # Retry logic
    last_error = None
    for attempt in range(config.retry_count):
        try:
            response = call_katago_api(config.endpoint, query, config.timeout)
            return parse_analysis_response(response, position_id)
        except requests.RequestException as e:
            last_error = str(e)
            if attempt < config.retry_count - 1:
                time.sleep(config.retry_delay)
    
    return AnalysisResult(
        position_id=position_id,
        success=False,
        error_message=f"API call failed after {config.retry_count} attempts: {last_error}"
    )


def result_to_columns(result: AnalysisResult) -> Dict[str, Any]:
    """Convert AnalysisResult to DataFrame columns."""
    
    # Base columns
    cols = {
        'analysis_success': result.success,
        'analysis_error': result.error_message,
        'api_root_winrate': result.root_winrate,
        'api_root_scoreLead': result.root_scoreLead,
        'api_root_visits': result.root_visits,
        'api_root_utility': result.root_utility,
    }
    
    # Top moves (all returned moves)
    if result.move_infos:
        cols['api_top_moves'] = json.dumps([m.move for m in result.move_infos])
        cols['api_child_winrates'] = json.dumps([m.winrate for m in result.move_infos])
        cols['api_child_scoreLeads'] = json.dumps([m.scoreLead for m in result.move_infos])
        cols['api_child_visits'] = json.dumps([m.visits for m in result.move_infos])
        cols['api_child_priors'] = json.dumps([m.prior for m in result.move_infos])
        cols['api_pvs'] = json.dumps([m.pv for m in result.move_infos])
        
        # Also store first (best) move separately for convenience
        best = result.move_infos[0]
        cols['api_best_move'] = best.move
        cols['api_best_winrate'] = best.winrate
        cols['api_best_scoreLead'] = best.scoreLead
        cols['api_best_pv'] = json.dumps(best.pv)
    else:
        cols['api_top_moves'] = None
        cols['api_child_winrates'] = None
        cols['api_child_scoreLeads'] = None
        cols['api_child_visits'] = None
        cols['api_child_priors'] = None
        cols['api_pvs'] = None
        cols['api_best_move'] = None
        cols['api_best_winrate'] = None
        cols['api_best_scoreLead'] = None
        cols['api_best_pv'] = None
    
    # Store raw response for debugging
    cols['api_raw_response'] = result.raw_response
    
    return cols


def analyze_parquet_file(
    parquet_path: Path,
    config: AnalysisConfig,
    skip_analyzed: bool = True
) -> pd.DataFrame:
    """
    Analyze all positions in a parquet file.
    
    Args:
        parquet_path: Path to parquet file
        config: Analysis configuration
        skip_analyzed: Skip rows that already have analysis results
        
    Returns:
        DataFrame with analysis columns added
    """
    print(f"Loading {parquet_path}...")
    df = pd.read_parquet(parquet_path)
    original_count = len(df)
    
    # Check if already analyzed
    if skip_analyzed and 'analysis_success' in df.columns:
        to_analyze = df[df['analysis_success'].isna() | (df['analysis_success'] == False)]
        print(f"  {len(to_analyze)} positions need analysis (skipping {original_count - len(to_analyze)} already analyzed)")
    else:
        to_analyze = df
        print(f"  {len(to_analyze)} positions to analyze")
    
    if len(to_analyze) == 0:
        return df
    
    # Analyze positions with thread pool
    results = []
    with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
        futures = {
            executor.submit(analyze_position, row, config): idx 
            for idx, row in to_analyze.iterrows()
        }
        
        completed = 0
        for future in as_completed(futures):
            idx = futures[future]
            try:
                result = future.result()
                results.append((idx, result))
            except Exception as e:
                results.append((idx, AnalysisResult(
                    position_id=str(idx),
                    success=False,
                    error_message=str(e)
                )))
            
            completed += 1
            if completed % 100 == 0:
                print(f"  Analyzed {completed}/{len(to_analyze)} positions...")
    
    # Add results to DataFrame
    for idx, result in results:
        cols = result_to_columns(result)
        for col, val in cols.items():
            df.loc[idx, col] = val
    
    success_count = sum(1 for _, r in results if r.success)
    print(f"  Completed: {success_count}/{len(results)} successful")
    
    return df


def main():
    parser = argparse.ArgumentParser(
        description='Analyze KataGo positions offline and append results to parquet files.'
    )
    parser.add_argument(
        '--input-dir', '-i',
        type=Path,
        required=True,
        help='Directory containing parquet files (searches recursively)'
    )
    parser.add_argument(
        '--endpoint', '-e',
        type=str,
        help='KataGo analysis endpoint URL (e.g., http://hopper-34:9100/analysis)'
    )
    parser.add_argument(
        '--config', '-c',
        type=Path,
        default=Path(__file__).parent / 'config.yaml',
        help='Path to config.yaml (default: runtime/config.yaml)'
    )
    parser.add_argument(
        '--max-workers', '-w',
        type=int,
        default=8,
        help='Maximum concurrent requests (default: 8)'
    )
    parser.add_argument(
        '--timeout',
        type=float,
        default=30.0,
        help='Request timeout in seconds (default: 30)'
    )
    parser.add_argument(
        '--skip-analyzed',
        action='store_true',
        default=True,
        help='Skip positions already analyzed (default: True)'
    )
    parser.add_argument(
        '--no-skip-analyzed',
        action='store_false',
        dest='skip_analyzed',
        help='Re-analyze all positions'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be analyzed without making changes'
    )
    
    args = parser.parse_args()
    
    # Determine endpoint
    endpoint = args.endpoint
    if not endpoint:
        # Try to load from config
        if args.config.exists():
            cfg = load_config(args.config)
            port = cfg.get('port', 9100)
            endpoint = f"http://hopper-34:{port}/analysis"
            print(f"Using endpoint from config: {endpoint}")
        else:
            endpoint = "http://hopper-34:9100/analysis"
            print(f"Using default endpoint: {endpoint}")
    
    config = AnalysisConfig(
        endpoint=endpoint,
        timeout=args.timeout,
        max_workers=args.max_workers
    )
    
    # Find parquet files
    parquet_files = list(args.input_dir.rglob('*.parquet'))
    if not parquet_files:
        print(f"No parquet files found in {args.input_dir}")
        return
    
    print(f"Found {len(parquet_files)} parquet files")
    
    if args.dry_run:
        for pf in parquet_files:
            df = pd.read_parquet(pf)
            print(f"  {pf}: {len(df)} rows")
        return
    
    # Process each file
    total_analyzed = 0
    total_success = 0
    
    for pf in parquet_files:
        print(f"\nProcessing {pf}...")
        df = analyze_parquet_file(pf, config, args.skip_analyzed)
        
        # Save back to parquet (in-place modification)
        df.to_parquet(pf, index=False)
        print(f"  Saved to {pf}")
        
        if 'analysis_success' in df.columns:
            analyzed = df['analysis_success'].notna().sum()
            success = df['analysis_success'].sum()
            total_analyzed += analyzed
            total_success += success
    
    print(f"\nTotal: {total_success}/{total_analyzed} positions analyzed successfully")


if __name__ == '__main__':
    main()
