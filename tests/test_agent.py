import json

import pytest

from app.agent import SYSTEM_PROMPT, clean_text, run_agent
from app.tools import Toolbox
from core.metrics import average_treatment_effect
from core.synthetic import generate_synthetic
from tests.llm_mock import (
    MockLlmServer,
    content_chunks,
    sse,
    tool_call_chunks,
    unreachable_client,
)


@pytest.fixture(scope="module")
def toolbox():
    return Toolbox(generate_synthetic(n=12_000, seed=8))


def kinds(events):
    return [e.kind for e in events]


def responder_two_step(tool_name, arguments, answer_for):
    def respond(payload, n):
        tool_messages = [m for m in payload["messages"] if m["role"] == "tool"]
        if not tool_messages:
            return sse(*tool_call_chunks(tool_name, arguments))
        return sse(*content_chunks(answer_for(json.loads(tool_messages[-1]["content"]))))

    return respond


def test_full_loop_calls_core_then_explains_the_returned_result(toolbox):
    direct = average_treatment_effect(toolbox.df, "conversion", data_source="synthetic")
    server = MockLlmServer(
        responder_two_step(
            "average_treatment_effect",
            {"outcome": "conversion"},
            lambda result: f"The effect is {result['ate'] * 100:.3f} percentage points.",
        )
    )
    events = list(run_agent("Did it work?", toolbox, server.client()))
    assert kinds(events)[0] == "tool_call"
    assert "tool_result" in kinds(events) and kinds(events)[-1] == "final"
    final = events[-1]
    assert len(final.traces) == 1
    trace = final.traces[0]
    assert trace.name == "average_treatment_effect" and trace.result == direct.to_dict()
    assert f"{direct.ate * 100:.3f}" in final.text
    assert final.unverified_numbers == []
    tool_msg = next(m for m in server.requests[1]["messages"] if m["role"] == "tool")
    assert json.loads(tool_msg["content"])["ate"] == pytest.approx(direct.ate)
    assert server.requests[0]["messages"][0]["content"] == SYSTEM_PROMPT


def test_final_answer_streams_text_deltas(toolbox):
    server = MockLlmServer(lambda p, n: sse(*content_chunks("No tool needed here.", size=4)))
    events = list(run_agent("hello", toolbox, server.client()))
    assert kinds(events).count("text") == 5 and kinds(events)[-1] == "final"
    assert events[-1].text == "No tool needed here." and events[-1].traces == []


def test_hallucinated_numbers_are_flagged(toolbox):
    server = MockLlmServer(
        responder_two_step(
            "average_treatment_effect",
            {"outcome": "visit"},
            lambda result: "Uplift is 777.7 percentage points.",
        )
    )
    final = list(run_agent("q", toolbox, server.client()))[-1]
    assert "777.7" in final.unverified_numbers


def test_iteration_cap_stops_a_model_that_never_answers(toolbox):
    server = MockLlmServer(
        lambda p, n: sse(*tool_call_chunks("dataset_summary", {}, call_id=f"c{n}"))
    )
    events = list(run_agent("loop forever", toolbox, server.client(), max_iterations=3))
    assert len(server.requests) == 3
    assert kinds(events)[-1] == "limit" and "3" in events[-1].text
    assert "final" not in kinds(events)


def test_default_cap_comes_from_settings(toolbox):
    server = MockLlmServer(lambda p, n: sse(*tool_call_chunks("dataset_summary", {})))
    list(run_agent("x", toolbox, server.client(max_iterations=2)))
    assert len(server.requests) == 2


def test_unknown_tool_error_is_fed_back_and_model_recovers(toolbox):
    def respond(payload, n):
        if n == 1:
            return sse(*tool_call_chunks("delete_everything", {}))
        tool = next(m for m in payload["messages"] if m["role"] == "tool")
        assert "Unknown tool" in json.loads(tool["content"])["error"]
        return sse(*content_chunks("Sorry, I cannot do that."))

    events = list(run_agent("q", toolbox, MockLlmServer(respond).client()))
    final = events[-1]
    assert final.kind == "final" and final.traces[0].error is not None


def test_invalid_json_arguments_are_reported_to_the_model(toolbox):
    def respond(payload, n):
        if n == 1:
            return sse(*tool_call_chunks("relative_uplift", '{"outcome": '))
        tool = next(m for m in payload["messages"] if m["role"] == "tool")
        assert "not valid JSON" in json.loads(tool["content"])["error"]
        return sse(*content_chunks("done"))

    assert list(run_agent("q", toolbox, MockLlmServer(respond).client()))[-1].kind == "final"


def test_at_most_four_tool_calls_run_per_round(toolbox):
    many = {
        "choices": [
            {
                "delta": {
                    "tool_calls": [
                        {
                            "index": i,
                            "id": f"c{i}",
                            "function": {"name": "dataset_summary", "arguments": "{}"},
                        }
                        for i in range(7)
                    ]
                },
                "finish_reason": "tool_calls",
            }
        ]
    }
    server = MockLlmServer(lambda p, n: sse(many) if n == 1 else sse(*content_chunks("ok")))
    final = list(run_agent("q", toolbox, server.client()))[-1]
    assert len(final.traces) == 4


def test_unreachable_server_yields_clear_error_event(toolbox):
    events = list(run_agent("hi", toolbox, unreachable_client()))
    assert kinds(events) == ["error"]
    assert "Cannot reach the LLM server" in events[0].text


def test_server_error_status_yields_error_event(toolbox):
    import httpx

    server = MockLlmServer(lambda p, n: httpx.Response(503, text="loading model"))
    events = list(run_agent("hi", toolbox, server.client()))
    assert kinds(events) == ["error"] and "503" in events[0].text


def test_history_is_sent_without_tool_messages(toolbox):
    server = MockLlmServer(lambda p, n: sse(*content_chunks("ok")))
    history = [
        {"role": "user", "content": "earlier question"},
        {"role": "assistant", "content": "earlier answer", "traces": []},
        {"role": "tool", "content": "{}"},
    ]
    list(run_agent("new", toolbox, server.client(), history=history))
    roles = [m["role"] for m in server.requests[0]["messages"]]
    assert roles == ["system", "user", "assistant", "user"]


def test_think_blocks_are_removed_from_final_text():
    assert clean_text("<think>secret plan</think>\nAnswer.") == "Answer."
