# ISSUE-3: fastest_response Streaming Race Fix

## Problem

litellm's `abatch_completion_fastest_response` has two bugs in streaming mode:

1. **Winner determined at HTTP connection time, not first token**: The original code checks `isinstance(result, CustomStreamWrapper)` which resolves when HTTP 200 headers arrive, before any tokens are generated. This picks the provider with the fastest CDN, not the fastest inference.

2. **First model always wins due to asyncio scheduling**: `asyncio.create_task` schedules tasks cooperatively. Each task's `acompletion` has sync setup (model lookup, routing, parameter validation) that blocks the event loop. The first task sends its HTTP request before others start, gaining a systematic advantage.

## Root Cause Analysis

### Non-streaming (correct behavior, no fix needed)

`acompletion(stream=False)` returns `ModelResponse` only after the full response is generated. All tasks' HTTP requests are sent during the first round of async scheduling. The winner is the provider that completes inference fastest.

### Streaming (bug)

Original code:
```python
result = await self.acompletion(model=model, stream=True, **kwargs)
if isinstance(result, CustomStreamWrapper):  # Wins at HTTP connection, not first token
    for t in pending_tasks:
        t.cancel()
    return result
```

## Fix

Split `abatch_completion_fastest_response` into two methods:

### `_abatch_completion_fastest_response_non_streaming`

No change from original logic. Races on full `ModelResponse` completion.

### `_abatch_completion_fastest_response_streaming`

Each task independently:
1. Calls `acompletion(stream=True)` to get `CustomStreamWrapper`
2. Reads the first chunk via `async for chunk in stream_obj`
3. Returns `(stream_obj, first_chunk)` tuple

The first task to produce a `(stream, chunk)` tuple wins. Other tasks are cancelled. A wrapper is returned that yields the already-consumed first chunk followed by the remaining stream.

```python
async def _get_first_chunk(model, **kw):
    stream_obj = await self.acompletion(model=model, messages=messages, stream=True, **kw)
    async for chunk in stream_obj:
        return (stream_obj, chunk)
```

### Rejected approaches

| Approach | Problem |
|----------|---------|
| `asyncio.gather` on all `acompletion` calls (Phase 1) + race on tokens (Phase 2) | `gather` waits for the slowest model's HTTP response. If mock-slow takes 3s, total latency is 3s+ regardless of mock-fast being 2.4s |
| `asyncio.Event` barrier between connection and token reading | Same problem as gather - barrier waits for all connections |

The simple single-phase approach works because asyncio's scheduling overhead (~20ms per task) is negligible compared to real LLM inference latency (300ms+).

## Verification

### Mock LLM Server

Local mock server (`mock_llm_server.py`) with 3 models:
- `mock-slow`: 360ms delay
- `mock-medium`: 330ms delay
- `mock-fast`: 300ms delay

### Integration Test (`test_race.py`)

- 100 rounds per mode (non-streaming + streaming run in parallel)
- Each round: random model order
- Warmup round before test to eliminate cold start
- Validates: mock-fast wins 100%, latency overhead < 40ms

### Test Results

| Metric | Non-streaming | Streaming |
|--------|--------------|-----------|
| Winner correct | 100/100 | 100/100 |
| p50 latency | 320ms (20ms overhead) | 320ms (20ms overhead) |
| p99 latency | 330ms | 333ms |
| max latency | 338ms | 339ms |

Winner correctness: 100%. Latency overhead p50=20ms, well within 800ms P99.99 target.

## Files Changed

- `litellm/router.py`: Split `abatch_completion_fastest_response` into streaming/non-streaming methods
- `requirements/3/mock_llm_server.py`: Mock LLM server for integration testing
- `requirements/3/test_race.py`: 100-round integration test
- `requirements/3/user_requirements.md`: User requirements
- `requirements/3/design.md`: This document
