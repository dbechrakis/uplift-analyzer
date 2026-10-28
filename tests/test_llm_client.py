import httpx
import pytest

from app.llm_client import LlmResponseError, LlmUnavailableError
from tests.llm_mock import (
    MockLlmServer,
    content_chunks,
    sse,
    tool_call_chunks,
    unreachable_client,
)

MESSAGES = [{"role": "user", "content": "hi"}]


def collect(client, **kwargs):
    return list(client.stream_chat(MESSAGES, **kwargs))


def test_streams_content_deltas_in_order():
    server = MockLlmServer(lambda p, n: sse(*content_chunks("Hello from the model", size=5)))
    events = collect(server.client())
    assert [e.kind for e in events].count("content") == 4
    assert "".join(e.text for e in events if e.kind == "content") == "Hello from the model"
    assert events[-1].kind == "finish" and events[-1].finish_reason == "stop"


def test_assembles_tool_call_from_split_argument_deltas():
    args = {"outcome": "conversion"}
    server = MockLlmServer(lambda p, n: sse(*tool_call_chunks("relative_uplift", args)))
    events = collect(server.client(), tools=[{"type": "function"}])
    calls = next(e for e in events if e.kind == "tool_calls").tool_calls
    assert len(calls) == 1
    assert calls[0].name == "relative_uplift" and calls[0].id == "call_1"
    assert calls[0].arguments_json == '{"outcome": "conversion"}'
    assert events[-1].finish_reason == "tool_calls"


def test_request_payload_carries_tools_and_streaming():
    server = MockLlmServer(lambda p, n: sse(*content_chunks("ok")))
    collect(server.client(), tools=[{"type": "function", "function": {"name": "x"}}])
    payload = server.requests[0]
    assert payload["stream"] is True and payload["tool_choice"] == "auto"
    assert payload["tools"][0]["function"]["name"] == "x"
    assert payload["messages"] == MESSAGES


def test_unreachable_server_raises_actionable_error():
    with pytest.raises(LlmUnavailableError, match="Cannot reach the LLM server"):
        collect(unreachable_client())


def test_timeout_is_reported_as_unavailable():
    def slow(request):
        raise httpx.ReadTimeout("slow", request=request)

    from app.config import Settings
    from app.llm_client import LlamaClient

    client = LlamaClient(Settings(), transport=httpx.MockTransport(slow))
    with pytest.raises(LlmUnavailableError, match="timed out"):
        collect(client)


def test_http_error_status_raises_response_error():
    server = MockLlmServer(lambda p, n: httpx.Response(500, text="model not loaded"))
    with pytest.raises(LlmResponseError, match="500.*model not loaded"):
        collect(server.client())


def test_malformed_stream_chunk_raises_response_error():
    server = MockLlmServer(lambda p, n: b"data: {not json}\n\ndata: [DONE]\n\n")
    with pytest.raises(LlmResponseError, match="Malformed"):
        collect(server.client())


def test_is_available_reflects_server_state():
    assert MockLlmServer(lambda p, n: b"").client().is_available() is True
    assert unreachable_client().is_available() is False
