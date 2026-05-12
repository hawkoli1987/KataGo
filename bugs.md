# Bug Log

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
