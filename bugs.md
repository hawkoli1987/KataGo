# Bug Log

## [2026-05-14] KataGo workers deadlock at high concurrency (asyncio StreamReader flow control)

**File:** `runtime/analysis_server.py`
**Symptom queue:** GPU 0% utilization on multiple workers, `in_flight` pinned at the cap, no progress

### Symptom

Running the offline analysis pipeline with the 8-GPU server (`katago_server_multi.pbs`) and a 96-concurrent client, individual worker GPUs would drop to 0% utilization and stay there indefinitely. The least-loaded router would shift traffic to healthy workers, but they would cascade into the same state over hours. Throughput collapsed from ~21 q/s (some workers stalled) to ~6 q/s (all 8 stalled and the client only got through via 429-backoff retries).

The wchan monitoring task (added during investigation, see `AnalysisEnginePool.monitor_loop`) captured each KataGo subprocess's main thread in `futex_wait_queue` and the analysis threads (65 of them) also in `futex_wait_queue` — a stable internal deadlock. The deadlock was unrecoverable; only restarting the server cleared it.

### Root cause

The asyncio `StreamReader` attached to a subprocess's stdout has a per-instance internal buffer limit (`_limit`, default **64 KB**). When the buffered byte count exceeds `2 × _limit = 128 KB`, asyncio calls `transport.pause_reading()` on the underlying socket transport. That tells uvloop to stop polling the socket fd for readability — uvloop stops draining data from the kernel.

uvloop creates **Unix socketpairs** (not pipes) for subprocess I/O. The kernel socket buffer is ~256 KB on this system. Once flow control kicks in:

1. asyncio buffer fills past 128 KB → `pause_reading()` called
2. uvloop stops draining the socket → kernel socket buffer fills (256 KB cap)
3. KataGo's `write()` for the next response blocks in `unix_stream_sendmsg`
4. That KataGo analysis thread is blocked while holding KataGo's stdout mutex
5. Other analysis threads pile up on the stdout mutex → `futex_wait_queue`
6. KataGo's internal query queue stays full (no thread is finishing) → main thread blocks on the queue's condition variable → `futex_wait_queue`
7. Deadlock: nothing in the KataGo process can move; nothing on our side can recover it

A burst of ~16 KataGo responses (each ~8 KB) is enough to cross the 128 KB threshold. With `MAX_IN_FLIGHT_PER_WORKER = 40` and KataGo's `nnMaxBatchSize = 64`, every neural-net batch evaluation completed many MCTS threads simultaneously, producing exactly this kind of burst.

### Investigation notes (what we ruled out)

- **`F_SETPIPE_SZ` on the subprocess stdout**: doesn't apply — uvloop uses socketpairs, not pipes.
- **`SO_RCVBUF` on the socket fd**: couldn't get the fd cleanly. uvloop's `ReadUnixTransport` doesn't expose `_pipe.fileno()` like CPython asyncio's `_UnixReadPipeTransport` does. Searching `/proc/self/fd` for the socket inode also fails because socketpairs have a different inode on each end.
- **Lowering `MAX_IN_FLIGHT_PER_WORKER` from 40 → 16**: works as a mitigation (16 × 8 KB = 128 KB ≤ flow-control threshold), but caps throughput at ~21 q/s on 8 GPUs.
- **Reducing KataGo's `nnMaxBatchSize`**: would reduce burst size at the source, but unnecessary once flow control is moved out of the way.

### Fix

Raise the StreamReader buffer limit on subprocess creation from the default 64 KB to **4 MB**:

```python
self._proc = await asyncio.create_subprocess_exec(
    *self._cmd,
    stdin=asyncio.subprocess.PIPE,
    stdout=asyncio.subprocess.PIPE,
    stderr=asyncio.subprocess.PIPE,
    limit=4 * 1024 * 1024,   # ← was: implicit 65536
    env=env,
)
```

The flow-control threshold becomes `2 × 4 MB = 8 MB`, well above any realistic burst (40 × 8 KB = 320 KB). `pause_reading()` is never called, uvloop drains the kernel socket continuously, and KataGo's writes never block. With this fix, `MAX_IN_FLIGHT_PER_WORKER` can safely return to 40+.

### Verification

A controlled 1-GPU stress test (`runtime/katago_server_test.pbs`) restored the original buggy conditions: `MAX_IN_FLIGHT_PER_WORKER=40`, client concurrency 40, single worker pinned at `in_flight=40` for the duration of the test.

| Metric | Before fix | After fix |
|---|---|---|
| GPU utilization | 0% (deadlocked within minutes) | 83–91% sustained |
| Power draw | ~122 W (idle) | 390–414 W |
| Throughput | dropped to 0 q/s | 6.1 q/s sustained on 1 GPU |
| Thread wchans | all 65 analysis threads in `futex_wait_queue` (deadlocked) | 65 in `futex_wait_queue` (normal idle between batches), main thread cycling through `unix_stream_data_wait` |

The 65 analysis threads sitting in `futex_wait_queue` looked alarming during investigation but is actually KataGo's normal state between NN batch evaluations — they're parked on a condition variable waiting for the next batch. The deadlock signature was the *combination* of all threads in `futex_wait_queue` **plus** the main thread in `futex_wait_queue` (blocked on queue insert) **plus** no forward progress in voluntary context switches. The post-fix state has the main thread in `unix_stream_data_wait` (reading next stdin query) with context switches climbing steadily.

### Lesson

When tunnelling a high-throughput subprocess through asyncio, there are **three** buffers in series, each with its own cap:
1. KataGo's internal stdio buffer (small, mutex-protected)
2. Kernel socket buffer (~256 KB, sysctl-controlled)
3. asyncio StreamReader buffer (**64 KB by default — easiest to overlook**)

Backpressure at any one of them propagates upstream and can deadlock the writer (which has no visibility into our flow-control state). When KataGo and our server are both running on the same node with plenty of RAM, the safe answer is to make the asyncio buffer big enough that flow control never engages — `limit=4 MB` on `create_subprocess_exec` does it for one line of code and ~32 MB of memory across 8 workers.

The monitor's "STALLED" heuristic (`in_flight ≥ 20` for many consecutive polls) is a false positive at high steady-state load. A real stall is detected by tracking whether the *set of pending future IDs* changes between polls — if the same IDs remain for >30 s, it's a deadlock.

## [2026-05-12] analyze_offline.py crashes entire run on empty server response

**File:** `runtime/analyze_offline.py`  
**Fixed in:** commit `1e5a08c5`

### Symptom

Running `analyze_offline.py` against the full 19×19 dataset (237k test / 2.1M train) crashed with:

```
json.decoder.JSONDecodeError: Expecting value: line 1 column 1 (char 0)
```

Result: 0 successful queries written, entire output file filled with `{}`.

### Root cause

The KataGo analysis server occasionally returns HTTP 200 with an **empty response body** for certain positions (e.g. positions it cannot evaluate). `aiohttp`'s `response.json(content_type=None)` raises `json.decoder.JSONDecodeError` on an empty body.

`analyze_single` only caught `aiohttp.ClientResponseError`, `aiohttp.ClientError`, and `asyncio.TimeoutError`. `JSONDecodeError` is not a subclass of any of these, so it propagated uncaught through:

```
analyze_single → call_katago_api → response.json()
    ↓ JSONDecodeError (uncaught)
track_active → re-raises
    ↓
asyncio.as_completed loop → await coro raises
    ↓
run_analysis_async crashes, aborting all remaining queries
```

The entire run aborted on the first bad response, not just the offending query.

### Fix

Added a broad `except Exception` catch at the end of the retry loop in `analyze_single`, consistent with how other errors are handled — logs the error type and returns `{}` for that query after retries are exhausted.

### Lesson

Any exception raised inside an `asyncio.as_completed` coroutine that isn't caught will propagate to the caller and abort the entire iteration. Always ensure task coroutines catch `Exception` broadly as a last resort when partial failure is acceptable.
