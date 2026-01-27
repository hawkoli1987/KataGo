#!/usr/bin/env python3
"""
Offline analysis script for KataGo positions using asyncio.

Reads JSONL input (one KataGo query per line), sends to analysis engine,
writes JSONL output (one response per line, 1-to-1 row matching).

If analysis fails for a row, the output row will be an empty object: {}
"""

import argparse
import asyncio
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import aiohttp


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class AnalysisConfig:
    """Configuration for analysis."""
    endpoint: str
    timeout: float = 60.0
    max_concurrent: int = 256
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
    config: AnalysisConfig
) -> tuple[int, dict]:
    """
    Analyze a single query with retry logic.
    
    Network errors are caught and result in empty response.
    This is intentional - failed rows produce {} in output.
    
    Returns:
        (line_idx, response_dict) - response_dict is {} on network failure
    """
    async with semaphore:
        for attempt in range(config.retry_count):
            try:
                response = await call_katago_api(
                    session, config.endpoint, query, config.timeout
                )
                return line_idx, response
            except (aiohttp.ClientError, asyncio.TimeoutError):
                if attempt < config.retry_count - 1:
                    await asyncio.sleep(config.retry_delay)
        
        # All retries exhausted
        return line_idx, {}


# =============================================================================
# Main Analysis Pipeline
# =============================================================================

def load_queries(input_path: Path) -> list[dict]:
    """Load and parse all queries from JSONL file."""
    queries = []
    with open(input_path, 'r') as f:
        for line in f:
            line = line.strip()
            assert line, "Empty line in input file"
            query = json.loads(line)
            queries.append(query)
    return queries


def save_results(output_path: Path, results: dict[int, dict], total: int) -> int:
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
    queries: list[dict],
    config: AnalysisConfig,
    progress_interval: int = 100
) -> dict[int, dict]:
    """
    Run analysis on all queries concurrently.
    
    Returns:
        Dict mapping line_idx to response
    """
    semaphore = asyncio.Semaphore(config.max_concurrent)
    results: dict[int, dict] = {}
    total = len(queries)
    start_time = time.time()
    completed = 0
    
    async with aiohttp.ClientSession() as session:
        tasks = [
            analyze_single(session, semaphore, idx, query, config)
            for idx, query in enumerate(queries)
        ]
        
        for coro in asyncio.as_completed(tasks):
            idx, response = await coro
            results[idx] = response
            completed += 1
            
            if completed % progress_interval == 0:
                elapsed = time.time() - start_time
                rate = completed / elapsed
                print(f"  Analyzed {completed}/{total} ({rate:.1f} queries/sec)")
    
    return results


def run_analysis(
    input_path: Path,
    output_path: Path,
    config: AnalysisConfig
) -> tuple[int, int]:
    """
    Run analysis on JSONL file.
    
    Returns:
        (total_lines, success_count)
    """
    # Load queries
    queries = load_queries(input_path)
    total = len(queries)
    
    print(f"Loaded {total} queries from {input_path}")
    print(f"Endpoint: {config.endpoint}")
    print(f"Max concurrent: {config.max_concurrent}")
    
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
        description='Analyze KataGo positions from JSONL file using asyncio.'
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
        default=256,
        help='Maximum concurrent requests (default: 256)'
    )
    parser.add_argument(
        '--timeout',
        type=float,
        default=60.0,
        help='Request timeout in seconds (default: 60)'
    )
    
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    
    config = AnalysisConfig(
        endpoint=args.endpoint,
        timeout=args.timeout,
        max_concurrent=args.max_concurrent
    )
    
    total, success = run_analysis(args.input, args.output, config)
    
    return 0 if success == total else 1


if __name__ == '__main__':
    sys.exit(main())

# =============================================================================
# Usage Examples
# =============================================================================
#
# # Basic analysis
# python runtime/analyze_offline.py \
#     --input ./data/inputs/19x19_koSIMPLEscoreTERRITORYtaxSEKIsui0_6.5/test.jsonl \
#     --output ./data/outputs/19x19_koSIMPLEscoreTERRITORYtaxSEKIsui0_6.5/test.jsonl \
#     --endpoint http://hopper-34:9200/analysis
#
# # With custom concurrency
# python runtime/analyze_offline.py \
#     --input positions.jsonl \
#     --output results.jsonl \
#     --endpoint http://localhost:9200/analysis \
#     --max-concurrent 64
