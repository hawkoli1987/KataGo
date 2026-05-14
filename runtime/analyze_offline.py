#!/usr/bin/env python3
"""
Offline analysis script for KataGo positions using optimized asyncio concurrency.

Reads JSONL input (one KataGo query per line), sends to analysis engine concurrently,
writes JSONL output (one response per line, 1-to-1 row matching).

Supports incremental/resumable writes: results are flushed to disk every --chunk-size
queries. If the process is interrupted, re-running with the same output file resumes
from where it left off (lines already written are skipped).

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
    timeout: float = 130.0
    max_concurrent: int = 512
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
    Failed rows (all retries exhausted) produce {} in output.
    """
    request_start = time.time()
    last_error = None
    async with semaphore:
        attempt = 0
        capacity_wait = 5.0  # initial backoff for 429 responses
        while attempt < config.retry_count:
            # Use a suffixed ID on retries to avoid 409 collision with the previous
            # attempt still held in the server's pending dict.
            retry_query = ({**query, 'id': f"{query['id']}_r{attempt}"}
                           if attempt > 0 else query)
            try:
                response = await call_katago_api(
                    session, config.endpoint, retry_query, config.timeout
                )
                # Strip the retry suffix so downstream consumers see the original id.
                if attempt > 0 and isinstance(response, dict) and 'id' in response:
                    response['id'] = query['id']
                request_time = time.time() - request_start
                return line_idx, response, request_time, None
            except aiohttp.ClientResponseError as e:
                if e.status == 429:
                    # Server at capacity: backoff without consuming retry budget.
                    await asyncio.sleep(capacity_wait)
                    capacity_wait = min(capacity_wait * 2, 60.0)
                    continue
                last_error = f"HTTP_{e.status}"
            except aiohttp.ClientError as e:
                last_error = f"ClientError_{type(e).__name__}"
            except asyncio.TimeoutError:
                last_error = "TimeoutError"
            except Exception as e:
                last_error = f"UnexpectedError_{type(e).__name__}"

            if error_tracker is not None:
                error_tracker[last_error] = error_tracker.get(last_error, 0) + 1
            if attempt < config.retry_count - 1:
                await asyncio.sleep(config.retry_delay * (attempt + 1))
            attempt += 1

        request_time = time.time() - request_start
        return line_idx, {}, request_time, last_error


# =============================================================================
# Chunk Analysis
# =============================================================================

async def run_chunk_async(
    queries: List[dict],
    config: AnalysisConfig,
    chunk_offset: int,
    total_queries: int,
    progress_interval: int = 500,
) -> Tuple[Dict[int, dict], int]:
    """
    Run analysis on one chunk of queries concurrently.

    Returns:
        (results dict keyed by chunk-local index, failed_count)
    """
    connector = aiohttp.TCPConnector(
        limit=config.max_concurrent,
        limit_per_host=config.max_concurrent,
        ttl_dns_cache=300,
        force_close=False,
    )
    semaphore = asyncio.Semaphore(config.max_concurrent)
    results: Dict[int, dict] = {}
    total = len(queries)
    start_time = time.time()
    completed = 0
    failed_count = 0
    request_times = []
    error_tracker: Dict[str, int] = defaultdict(int)

    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [
            analyze_single(session, semaphore, idx, query, config, error_tracker)
            for idx, query in enumerate(queries)
        ]
        for coro in asyncio.as_completed(tasks):
            idx, response, req_time, _ = await coro
            request_times.append(req_time)
            results[idx] = response
            if not response:
                failed_count += 1
            completed += 1

            if completed % progress_interval == 0 or completed == total:
                elapsed = time.time() - start_time
                rate = completed / elapsed if elapsed > 0 else 0
                eta_sec = (total_queries - chunk_offset - completed) / rate if rate > 0 else 0
                success_rate = (completed - failed_count) / completed * 100
                avg_req = sum(request_times[-100:]) / min(100, len(request_times))
                print(
                    f"  [{chunk_offset + completed}/{total_queries}] "
                    f"{rate:.1f} q/s  ETA {eta_sec/60:.1f}min  "
                    f"success {success_rate:.1f}%  avg_req {avg_req:.2f}s",
                    flush=True,
                )

    if error_tracker:
        print(f"  Errors in chunk: { dict(error_tracker) }", flush=True)

    return results, failed_count


# =============================================================================
# Resumable Pipeline
# =============================================================================

def count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with open(path) as f:
        return sum(1 for _ in f)


def run_analysis(
    input_path: Path,
    output_path: Path,
    config: AnalysisConfig,
    chunk_size: int = 50_000,
    limit: Optional[int] = None,
) -> Tuple[int, int]:
    """
    Analyze input JSONL in chunks, appending results to output after each chunk.

    Resumable: counts lines already in output_path and skips that many input lines.

    Returns:
        (total_processed, total_success)
    """
    # --- Resume state ---
    already_done = count_lines(output_path)
    total_input = count_lines(input_path)
    if limit is not None:
        total_input = min(total_input, limit)

    remaining = total_input - already_done
    if already_done > 0:
        print(f"Resuming: {already_done} lines already written, {remaining} remaining.", flush=True)
    else:
        print(f"Starting fresh: {total_input} queries to process.", flush=True)

    if remaining <= 0:
        print("Nothing to do.", flush=True)
        return already_done, already_done

    print(f"Endpoint:     {config.endpoint}", flush=True)
    print(f"Concurrency:  {config.max_concurrent}", flush=True)
    print(f"Chunk size:   {chunk_size}", flush=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_success = 0
    chunk_offset = already_done
    run_start = time.time()

    with open(input_path) as in_f, open(output_path, 'a') as out_f:
        # Skip already-processed lines
        for _ in range(already_done):
            next(in_f)

        while chunk_offset < total_input:
            # Load next chunk, capped by chunk_size and total_input
            target = min(chunk_size, total_input - chunk_offset)
            chunk: List[dict] = []
            for line in in_f:
                line = line.strip()
                if not line:
                    continue
                chunk.append(json.loads(line))
                if len(chunk) >= target:
                    break

            if not chunk:
                break

            chunk_num = chunk_offset // chunk_size + 1
            print(
                f"\n=== Chunk {chunk_num}: lines {chunk_offset}–{chunk_offset + len(chunk) - 1} ===",
                flush=True,
            )
            chunk_start = time.time()

            results, failed = asyncio.run(
                run_chunk_async(chunk, config, chunk_offset, total_input)
            )

            # Write in original order and flush immediately
            chunk_success = 0
            for idx in range(len(chunk)):
                response = results.get(idx, {})
                out_f.write(json.dumps(response) + '\n')
                if response:
                    chunk_success += 1
            out_f.flush()

            chunk_elapsed = time.time() - chunk_start
            total_success += chunk_success
            chunk_offset += len(chunk)

            print(
                f"  Chunk done: {chunk_success}/{len(chunk)} succeeded "
                f"in {chunk_elapsed:.1f}s — "
                f"total written: {chunk_offset}/{total_input}",
                flush=True,
            )

    elapsed = time.time() - run_start
    rate = (chunk_offset - already_done) / elapsed if elapsed > 0 else 0
    print(f"\nDone: {total_success}/{chunk_offset - already_done} succeeded ({rate:.1f} q/s)", flush=True)
    print(f"Output: {output_path}", flush=True)

    return chunk_offset, total_success


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Analyze KataGo positions from JSONL, writing results incrementally.'
    )
    parser.add_argument('--input',  '-i', type=Path, required=True,
                        help='Input JSONL file (one KataGo query per line)')
    parser.add_argument('--output', '-o', type=Path, required=True,
                        help='Output JSONL file (appended to; existing lines are skipped)')
    parser.add_argument('--endpoint', '-e', type=str, required=True,
                        help='KataGo analysis endpoint URL')
    parser.add_argument('--max-concurrent', '-c', type=int, default=512,
                        help='Max concurrent requests (default: 512)')
    parser.add_argument('--timeout', type=float, default=130.0,
                        help='Request timeout in seconds (default: 130; must exceed server timeout of 120s)')
    parser.add_argument('--chunk-size', type=int, default=50_000,
                        help='Queries per checkpoint (default: 50000)')
    parser.add_argument('--limit', '-n', type=int, default=None,
                        help='Max total queries to process (for testing)')
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = AnalysisConfig(
        endpoint=args.endpoint,
        timeout=args.timeout,
        max_concurrent=args.max_concurrent,
    )
    total, success = run_analysis(
        args.input, args.output, config,
        chunk_size=args.chunk_size,
        limit=args.limit,
    )
    return 0 if success > 0 else 1


if __name__ == '__main__':
    sys.exit(main())
