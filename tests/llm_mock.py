from __future__ import annotations

import json
from collections.abc import Callable

import httpx

from app.config import Settings
from app.llm_client import LlamaClient


def sse(*chunks: dict, done: bool = True) -> bytes:
    body = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks)
    return (body + ("data: [DONE]\n\n" if done else "")).encode()


def content_chunks(text: str, size: int = 7) -> list[dict]:
    pieces = [text[i : i + size] for i in range(0, len(text), size)]
    chunks = [{"choices": [{"delta": {"content": p}, "finish_reason": None}]} for p in pieces]
    chunks.append({"choices": [{"delta": {}, "finish_reason": "stop"}]})
    return chunks


def tool_call_chunks(name: str, arguments: dict | str, call_id: str = "call_1") -> list[dict]:
    raw = arguments if isinstance(arguments, str) else json.dumps(arguments)
    half = len(raw) // 2
    first = {
        "index": 0,
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": raw[:half]},
    }
    second = {"index": 0, "function": {"arguments": raw[half:]}}
    return [
        {"choices": [{"delta": {"tool_calls": [first]}, "finish_reason": None}]},
        {"choices": [{"delta": {"tool_calls": [second]}, "finish_reason": None}]},
        {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
    ]


class MockLlmServer:
    def __init__(self, responder: Callable[[dict, int], bytes | httpx.Response]) -> None:
        self.responder = responder
        self.requests: list[dict] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": "mock"}]})
        payload = json.loads(request.content)
        self.requests.append(payload)
        result = self.responder(payload, len(self.requests))
        if isinstance(result, httpx.Response):
            return result
        return httpx.Response(200, content=result, headers={"content-type": "text/event-stream"})

    def client(self, **settings) -> LlamaClient:
        return LlamaClient(
            Settings(base_url="http://mock/v1", **settings),
            transport=httpx.MockTransport(self.handler),
        )


def unreachable_client() -> LlamaClient:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    return LlamaClient(Settings(base_url="http://mock/v1"), transport=httpx.MockTransport(refuse))
