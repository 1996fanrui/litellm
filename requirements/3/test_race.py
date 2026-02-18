#!/usr/bin/env python3
"""Test fastest_response racing with mock models.

Expects mock-slow(3.0s), mock-medium(2.7s), mock-fast(2.4s).
Validates: mock-fast always wins, latency ~2.4s, never reaches 2.7s.
"""

import asyncio
import time
from openai import AsyncOpenAI

LITELLM_URL = "http://localhost:31750/v1"
LITELLM_KEY = "sk-123456"
# slow is first in list - should NOT win
MODEL_LIST = "mock-slow, mock-medium, mock-fast"

async def test_non_streaming():
    client = AsyncOpenAI(base_url=LITELLM_URL, api_key=LITELLM_KEY)
    start = time.time()
    resp = await client.chat.completions.create(
        model=MODEL_LIST,
        messages=[{"role": "user", "content": "test"}],
        max_tokens=10,
        stream=False,
        extra_body={"fastest_response": True},
    )
    elapsed = time.time() - start
    content = resp.choices[0].message.content
    model = resp.model
    return elapsed, content, model

async def test_streaming():
    client = AsyncOpenAI(base_url=LITELLM_URL, api_key=LITELLM_KEY)
    start = time.time()
    stream = await client.chat.completions.create(
        model=MODEL_LIST,
        messages=[{"role": "user", "content": "test"}],
        max_tokens=10,
        stream=True,
        extra_body={"fastest_response": True},
    )
    tokens = []
    model = None
    async for chunk in stream:
        if not model:
            model = chunk.model
        if chunk.choices and chunk.choices[0].delta.content:
            tokens.append(chunk.choices[0].delta.content)
    elapsed = time.time() - start
    content = "".join(tokens)
    return elapsed, content, model

async def main():
    print("=" * 60)
    print("FASTEST_RESPONSE RACE TEST (mock models)")
    print(f"Model order: {MODEL_LIST}")
    print(f"Expected winner: mock-fast (2.4s)")
    print("=" * 60)

    all_pass = True

    # Non-streaming tests
    print("\n--- Non-Streaming Tests ---")
    for i in range(3):
        elapsed, content, model = await test_non_streaming()
        winner_ok = "mock-fast" in content
        time_ok = elapsed < 2.7
        status = "PASS" if (winner_ok and time_ok) else "FAIL"
        if status == "FAIL":
            all_pass = False
        print(f"  Run {i+1}: {status} | time={elapsed:.2f}s | content={content!r} | model={model}")

    # Streaming tests
    print("\n--- Streaming Tests ---")
    for i in range(3):
        elapsed, content, model = await test_streaming()
        winner_ok = "mock-fast" in content
        time_ok = elapsed < 2.7
        status = "PASS" if (winner_ok and time_ok) else "FAIL"
        if status == "FAIL":
            all_pass = False
        print(f"  Run {i+1}: {status} | time={elapsed:.2f}s | content={content!r} | model={model}")

    print("\n" + "=" * 60)
    if all_pass:
        print("RESULT: ALL TESTS PASSED")
    else:
        print("RESULT: SOME TESTS FAILED")
    print("=" * 60)

asyncio.run(main())
