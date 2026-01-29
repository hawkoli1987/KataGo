#!/usr/bin/env python3
"""
Offline analysis script for KataGo positions using optimized asyncio concurrency.

Reads JSONL input (one KataGo query per line), sends to analysis engine concurrently,
writes JSONL output (one response per line, 1-to-1 row matching).

If analysis fails for a row, the output row will be an empty object: {}
"""

import argparse
import asyncio
import json
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import aiohttp


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class AnalysisConfig:
    """Configuration for analysis."""
    endpoint: str
    timeout: float = 60.0
    max_concurrent: int = 512  # Increased for better throughput
    retry_count: int = 3
    retry_delay: float = 1.0


# =============================================================================
# Async API Client
# =============================================================================

async def call_katago_api(
    session: aiohttp.ClientSession,
    endpoint: str,
    query: dict,
    timeout: float
) -> dict:
    """
    Call KataGo analysis API asynchronously.
    
    Returns:
        API response dict
    """
    async with session.post(
        endpoint,
        json=query,
        timeout=aiohttp.ClientTimeout(total=timeout)
    ) as response:
        response.raise_for_status()
        # Server may return text/plain instead of application/json
        return await response.json(content_type=None)


async def analyze_single(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    line_idx: int,
    query: dict,
    config: AnalysisConfig,
    error_tracker: Optional[Dict[str, int]] = None
) -> Tuple[int, dict, float, Optional[str]]:
    """
    Analyze a single query with retry logic.
    
    Network errors are caught and result in empty response.
    This is intentional - failed rows produce {} in output.
    
    Returns:
        (line_idx, response_dict, request_time, error_type) - response_dict is {} on network failure
    """
    request_start = time.time()
    last_error = None
    async with semaphore:
        for attempt in range(config.retry_count):
            try:
                response = await call_katago_api(
                    session, config.endpoint, query, config.timeout
                )
                request_time = time.time() - request_start
                return line_idx, response, request_time, None
            except aiohttp.ClientResponseError as e:
                last_error = f"HTTP_{e.status}"
                if error_tracker is not None:
                    error_tracker[last_error] = error_tracker.get(last_error, 0) + 1
                if attempt < config.retry_count - 1:
                    await asyncio.sleep(config.retry_delay * (attempt + 1))
            except aiohttp.ClientError as e:
                error_type = type(e).__name__
                last_error = f"ClientError_{error_type}"
                if error_tracker is not None:
                    error_tracker[last_error] = error_tracker.get(last_error, 0) + 1
                if attempt < config.retry_count - 1:
                    await asyncio.sleep(config.retry_delay * (attempt + 1))
            except asyncio.TimeoutError:
                last_error = "TimeoutError"
                if error_tracker is not None:
                    error_tracker[last_error] = error_tracker.get(last_error, 0) + 1
                if attempt < config.retry_count - 1:
                    await asyncio.sleep(config.retry_delay * (attempt + 1))
        
        # All retries exhausted
        request_time = time.time() - request_start
        return line_idx, {}, request_time, last_error


# =============================================================================
# Main Analysis Pipeline
# =============================================================================

def load_queries(input_path: Path, limit: Optional[int] = None, board_size: Optional[int] = None) -> List[dict]:
    """
    Load and parse queries from JSONL file.
    
    Args:
        input_path: Path to input JSONL file
        limit: Maximum number of queries to load (for testing)
        board_size: If specified, only load queries with this board size (e.g., 19 for 19x19)
    
    Returns:
        List of query dictionaries
    """
    load_start = time.time()
    queries = []
    filtered_count = 0
    
    with open(input_path, 'r') as f:
        for i, line in enumerate(f):
            if limit and len(queries) >= limit:
                break
            line = line.strip()
            assert line, "Empty line in input file"
            query = json.loads(line)
            
            # Filter by board size if specified
            if board_size is not None:
                query_board_size = query.get('boardXSize') or query.get('boardYSize')
                if query_board_size != board_size:
                    filtered_count += 1
                    continue
            
            queries.append(query)
    
    load_time = time.time() - load_start
    if load_time > 1.0:
        print(f"Loading took {load_time:.2f}s ({len(queries)} queries)")
    
    if board_size is not None and filtered_count > 0:
        print(f"Filtered out {filtered_count} queries not matching board size {board_size}x{board_size}")
    
    return queries


def save_results(output_path: Path, results: Dict[int, dict], total: int) -> int:
    """
    Save results to JSONL file in original order.
    
    Returns:
        Number of successful results
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    success_count = 0
    
    with open(output_path, 'w') as f:
        for idx in range(total):
            response = results.get(idx, {})
            if response:
                success_count += 1
            f.write(json.dumps(response) + '\n')
    
    return success_count


async def run_analysis_async(
    queries: List[dict],
    config: AnalysisConfig,
    progress_interval: int = 500
) -> Dict[int, dict]:
    """
    Run analysis on all queries concurrently using optimized asyncio.
    
    Uses connection pooling and high concurrency for maximum throughput.
    
    Returns:
        Dict mapping line_idx to response
    """
    # Use connector with connection pooling
    connector = aiohttp.TCPConnector(
        limit=config.max_concurrent,  # Max total connections
        limit_per_host=config.max_concurrent,  # Max per host
        ttl_dns_cache=300,
        force_close=False  # Reuse connections
    )
    
    semaphore = asyncio.Semaphore(config.max_concurrent)
    results: Dict[int, dict] = {}
    total = len(queries)
    start_time = time.time()
    completed = 0
    active_requests = 0
    max_active_seen = 0
    
    async def track_active(coro, idx):
        nonlocal active_requests, max_active_seen
        active_requests += 1
        max_active_seen = max(max_active_seen, active_requests)
        try:
            return await coro
        finally:
            active_requests -= 1
    
    request_times = []
    error_tracker = defaultdict(int)
    
    async with aiohttp.ClientSession(connector=connector) as session:
        print(f"Creating {total} async tasks...")
        task_start = time.time()
        tasks = [
            track_active(analyze_single(session, semaphore, idx, query, config, error_tracker), idx)
            for idx, query in enumerate(queries)
        ]
        task_creation_time = time.time() - task_start
        print(f"Task creation took {task_creation_time:.3f}s")
        print(f"Starting async execution with max_concurrent={config.max_concurrent}...")
        
        first_response_time = None
        failed_count = 0
        for coro in asyncio.as_completed(tasks):
            idx, response, req_time, error_type = await coro
            request_times.append(req_time)
            
            if first_response_time is None:
                first_response_time = time.time() - start_time
                print(f"First response received after {first_response_time:.2f}s")
            
            results[idx] = response
            if not response:
                failed_count += 1
            
            completed += 1
            
            if completed % progress_interval == 0 or completed == total:
                elapsed = time.time() - start_time
                rate = completed / elapsed if elapsed > 0 else 0
                eta_sec = (total - completed) / rate if rate > 0 else 0
                eta_min = eta_sec / 60
                success_rate = ((completed - failed_count) / completed * 100) if completed > 0 else 0
                if request_times:
                    avg_req_time = sum(request_times[-100:]) / min(100, len(request_times))
                    print(f"  Analyzed {completed}/{total} ({rate:.1f} queries/sec, ETA: {eta_min:.1f}min, "
                          f"success: {success_rate:.1f}%, avg_req_time: {avg_req_time:.3f}s)")
                else:
                    print(f"  Analyzed {completed}/{total} ({rate:.1f} queries/sec, ETA: {eta_min:.1f}min, "
                          f"max_active: {max_active_seen})")
    
    if request_times:
        avg_time = sum(request_times) / len(request_times)
        min_time = min(request_times)
        max_time = max(request_times)
        print(f"\nRequest timing stats: avg={avg_time:.3f}s, min={min_time:.3f}s, max={max_time:.3f}s")
    print(f"Max concurrent requests observed: {max_active_seen}")
    
    if error_tracker:
        print(f"\nError summary:")
        for error_type, count in sorted(error_tracker.items(), key=lambda x: -x[1]):
            print(f"  {error_type}: {count}")
    else:
        print(f"\nNo errors encountered!")
    
    return results


def run_analysis(
    input_path: Path,
    output_path: Path,
    config: AnalysisConfig,
    limit: Optional[int] = None
) -> Tuple[int, int]:
    """
    Run analysis on JSONL file.
    
    Returns:
        (total_lines, success_count)
    """
    # Load queries
    queries = load_queries(input_path, limit=limit)
    total = len(queries)
    
    print(f"Loaded {total} queries from {input_path}")
    print(f"Endpoint: {config.endpoint}")
    print(f"Max concurrent requests: {config.max_concurrent}")
    
    start_time = time.time()
    
    # Run async analysis
    results = asyncio.run(run_analysis_async(queries, config))
    
    # Save results
    success_count = save_results(output_path, results, total)
    
    elapsed = time.time() - start_time
    rate = total / elapsed if elapsed > 0 else 0
    print(f"\nCompleted: {success_count}/{total} successful ({rate:.1f} queries/sec)")
    print(f"Output: {output_path}")
    
    return total, success_count


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Analyze KataGo positions from JSONL file using optimized asyncio.'
    )
    parser.add_argument(
        '--input', '-i',
        type=Path,
        required=True,
        help='Input JSONL file (one KataGo query per line)'
    )
    parser.add_argument(
        '--output', '-o',
        type=Path,
        required=True,
        help='Output JSONL file (one response per line, 1-to-1 with input)'
    )
    parser.add_argument(
        '--endpoint', '-e',
        type=str,
        required=True,
        help='KataGo analysis endpoint URL (e.g., http://hopper-34:9200/analysis)'
    )
    parser.add_argument(
        '--max-concurrent', '-c',
        type=int,
        default=512,
        help='Maximum concurrent requests (default: 512)'
    )
    parser.add_argument(
        '--timeout',
        type=float,
        default=60.0,
        help='Request timeout in seconds (default: 60)'
    )
    parser.add_argument(
        '--limit', '-n',
        type=int,
        default=None,
        help='Limit number of queries to process (for testing)'
    )
    
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    
    config = AnalysisConfig(
        endpoint=args.endpoint,
        timeout=args.timeout,
        max_concurrent=args.max_concurrent
    )
    
    total, success = run_analysis(args.input, args.output, config, limit=args.limit)
    
    return 0 if success == total else 1


if __name__ == '__main__':
    sys.exit(main())
