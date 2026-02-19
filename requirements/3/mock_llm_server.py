#!/usr/bin/env python3
"""Mock LLM server with configurable delay per model for testing fastest_response racing.

Usage: python3 mock_llm_server.py
Serves on port 29990 with 3 models:
  - mock-slow:   360ms delay
  - mock-medium: 330ms delay
  - mock-fast:   300ms delay
"""

import asyncio
import json
import time
from aiohttp import web

MODEL_DELAYS = {
    "mock-slow": 0.36,
    "mock-medium": 0.33,
    "mock-fast": 0.30,
}

async def chat_completions(request):
    data = await request.json()
    model = data.get("model", "unknown")
    stream = data.get("stream", False)
    delay = MODEL_DELAYS.get(model, 0.3)

    await asyncio.sleep(delay)

    if stream:
        response = web.StreamResponse()
        response.content_type = "text/event-stream"
        await response.prepare(request)

        chunk1 = {
            "id": f"mock-{model}-{int(time.time()*1000)}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [{"index": 0, "delta": {"role": "assistant", "content": f"Hello from {model}!"}, "finish_reason": None}]
        }
        await response.write(f"data: {json.dumps(chunk1)}\n\n".encode())

        final = {
            "id": f"mock-{model}-{int(time.time()*1000)}",
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
            "id": f"mock-{model}-{int(time.time()*1000)}",
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
