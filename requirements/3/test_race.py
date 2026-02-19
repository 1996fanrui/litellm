#!/usr/bin/env python3
"""Test fastest_response racing - 100 rounds, random order, strict timing.

mock-slow=360ms, mock-medium=330ms, mock-fast=300ms.
Requirement: mock-fast always wins, total time < 329ms.
Non-streaming and streaming tests run in parallel.
"""

import asyncio
import random
import time
from openai import AsyncOpenAI

LITELLM_URL = "http://localhost:31750/v1"
LITELLM_KEY = "sk-123456"
MODELS = ["mock-slow", "mock-medium", "mock-fast"]
MAX_TIME_MS = 329
ROUNDS = 100


async def race(client, models_str, stream):
    start = time.time()
    if stream:
        s = await client.chat.completions.create(
            model=models_str,
            messages=[{"role": "user", "content": "t"}],
            max_tokens=5, stream=True,
            extra_body={"fastest_response": True},
        )
        content = ""
        async for chunk in s:
            if chunk.choices and chunk.choices[0].delta.content:
                content += chunk.choices[0].delta.content
    else:
        resp = await client.chat.completions.create(
            model=models_str,
            messages=[{"role": "user", "content": "t"}],
            max_tokens=5, stream=False,
            extra_body={"fastest_response": True},
        )
        content = resp.choices[0].message.content
    elapsed_ms = (time.time() - start) * 1000
    return elapsed_ms, content


async def run_mode(mode):
    """Run 100 sequential rounds for one mode. Returns result dict."""
    stream = mode == "streaming"
    client = AsyncOpenAI(base_url=LITELLM_URL, api_key=LITELLM_KEY)

    # Warmup: 1 request to eliminate cold start
    await race(client, ", ".join(MODELS), stream)

    wins = {"mock-slow": 0, "mock-medium": 0, "mock-fast": 0, "unknown": 0}
    times = []
    failures = []

    for i in range(ROUNDS):
        order = MODELS[:]
        random.shuffle(order)
        models_str = ", ".join(order)
        elapsed_ms, content = await race(client, models_str, stream)
        times.append(elapsed_ms)

        winner = "unknown"
        for m in MODELS:
            if m in content:
                winner = m
                break
        wins[winner] += 1

        ok_winner = winner == "mock-fast"
        ok_time = elapsed_ms < MAX_TIME_MS
        if not ok_winner or not ok_time:
            failures.append(
                f"  Run {i+1}: order={order}, winner={winner}, time={elapsed_ms:.0f}ms"
            )

    times.sort()
    return {
        "mode": mode, "wins": wins, "times": times,
        "failures": failures,
        "p50": times[49], "p99": times[98], "max": times[99],
    }


async def main():
    print("=" * 60)
    print(f"  FASTEST_RESPONSE RACE TEST")
    print(f"  {ROUNDS} rounds per mode, random order, limit {MAX_TIME_MS}ms")
    print(f"  Models: slow=360ms, medium=330ms, fast=300ms")
    print(f"  Running non-streaming and streaming in parallel...")
    print("=" * 60)

    # Run both modes in parallel (each internally sequential)
    results = await asyncio.gather(
        run_mode("non-streaming"),
        run_mode("streaming"),
    )

    all_pass = True
    for r in results:
        print(f"\n--- {r['mode'].upper()} ---")
        print(f"  Winners: slow={r['wins']['mock-slow']}, medium={r['wins']['mock-medium']}, fast={r['wins']['mock-fast']}")
        print(f"  Latency: p50={r['p50']:.0f}ms, p99={r['p99']:.0f}ms, max={r['max']:.0f}ms")
        print(f"  Failures: {len(r['failures'])}/{ROUNDS}")
        if r['failures']:
            all_pass = False
            for f in r['failures'][:10]:
                print(f)
            if len(r['failures']) > 10:
                print(f"  ... and {len(r['failures'])-10} more")
        else:
            print(f"  ALL {ROUNDS} ROUNDS PASSED")

    print("\n" + "=" * 60)
    print(f"  RESULT: {'ALL TESTS PASSED' if all_pass else 'SOME TESTS FAILED'}")
    print("=" * 60)

asyncio.run(main())
