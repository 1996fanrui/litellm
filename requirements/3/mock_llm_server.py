#!/usr/bin/env python3
"""Mock LLM server with configurable delay per model for testing fastest_response racing.

Usage: python3 mock_llm_server.py
Serves on port 29990 with 3 models:
  - mock-slow:   3.0s delay
  - mock-medium: 2.7s delay
  - mock-fast:   2.4s delay
"""

import asyncio
import json
import time
from aiohttp import web

MODEL_DELAYS = {
    "mock-slow": 3.0,
    "mock-medium": 2.7,
    "mock-fast": 2.4,
}

async def chat_completions(request):
    data = await request.json()
    model = data.get("model", "unknown")
    stream = data.get("stream", False)
    delay = MODEL_DELAYS.get(model, 1.0)

    await asyncio.sleep(delay)

    if stream:
        response = web.StreamResponse()
        response.content_type = "text/event-stream"
        await response.prepare(request)

        # First chunk with role
        chunk1 = {
            "id": f"mock-{model}-{int(time.time())}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}]
        }
        await response.write(f"data: {json.dumps(chunk1)}\n\n".encode())

        # Content chunks
        for word in ["Hello", " from", f" {model}!"]:
            chunk = {
                "id": f"mock-{model}-{int(time.time())}",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model,
                "choices": [{"index": 0, "delta": {"content": word}, "finish_reason": None}]
            }
            await response.write(f"data: {json.dumps(chunk)}\n\n".encode())
            await asyncio.sleep(0.05)

        # Final chunk
        final = {
            "id": f"mock-{model}-{int(time.time())}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
        }
        await response.write(f"data: {json.dumps(final)}\n\n".encode())
        await response.write(b"data: [DONE]\n\n")
        return response
    else:
        result = {
            "id": f"mock-{model}-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": f"Hello from {model}!"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        }
        return web.json_response(result)

async def models_list(request):
    return web.json_response({
        "object": "list",
        "data": [{"id": m, "object": "model"} for m in MODEL_DELAYS]
    })

app = web.Application()
app.router.add_post("/v1/chat/completions", chat_completions)
app.router.add_get("/v1/models", models_list)

if __name__ == "__main__":
    print(f"Mock LLM server starting on :29990 with models: {MODEL_DELAYS}")
    web.run_app(app, host="0.0.0.0", port=29990)
