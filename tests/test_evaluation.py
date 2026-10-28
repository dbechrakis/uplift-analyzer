import json
from pathlib import Path

import pytest

from app.agent import AgentEvent
from app.evaluation import run_case, score_case, summarize
from app.tools import TOOL_DEFINITIONS, Toolbox, ToolTrace
from core.synthetic import generate_synthetic
from tests.llm_mock import MockLlmServer, content_chunks, sse, tool_call_chunks

CASES = json.loads(Path("evals/tool_selection.json").read_text())


def final(text="ok", unverified=None):
    return AgentEvent("final", text=text, unverified_numbers=unverified or [])


def trace(name, **arguments):
    return ToolTrace(name=name, arguments=arguments, result={})


def test_case_file_is_well_formed_and_covers_every_tool():
    assert len({c["id"] for c in CASES}) == len(CASES) >= 15
    expected = {c["expected_tool"] for c in CASES} - {None, "*"}
    assert expected == set(TOOL_DEFINITIONS)
    for case in CASES:
        if case["expected_tool"] not in (None, "*"):
            assert set(case["expected_args"]) <= set(
                TOOL_DEFINITIONS[case["expected_tool"]]["properties"]
            )


def test_scoring_correct_tool_and_arguments():
    case = {
        "id": "x",
        "question": "q",
        "expected_tool": "relative_uplift",
        "expected_args": {"outcome": "visit"},
    }
    ok = score_case(case, [trace("relative_uplift", outcome="visit")], final(), 1.0)
    assert ok.tool_correct and ok.args_correct and ok.answered and ok.grounded
    wrong_arg = score_case(case, [trace("relative_uplift", outcome="conversion")], final(), 1.0)
    assert wrong_arg.tool_correct and not wrong_arg.args_correct
    wrong_tool = score_case(case, [trace("dataset_summary")], final(), 1.0)
    assert not wrong_tool.tool_correct and not wrong_tool.args_correct


def test_scoring_no_tool_and_any_tool_expectations():
    none_case = {"id": "n", "question": "q", "expected_tool": None, "expected_args": {}}
    assert score_case(none_case, [], final(), 1).tool_correct
    assert not score_case(none_case, [trace("dataset_summary")], final(), 1).tool_correct
    any_case = {"id": "a", "question": "q", "expected_tool": "*", "expected_args": {}}
    assert score_case(any_case, [trace("dataset_summary")], final(), 1).tool_correct


def test_scoring_flags_ungrounded_and_missing_answers():
    case = {"id": "x", "question": "q", "expected_tool": "*", "expected_args": {}}
    assert not score_case(case, [], final(unverified=["42"]), 1).grounded
    missing = score_case(case, [], AgentEvent("limit", text="stopped"), 1)
    assert not missing.answered
    assert not score_case(case, [], None, 1).answered


def test_summary_rates():
    case = {"id": "x", "question": "q", "expected_tool": "dataset_summary", "expected_args": {}}
    good = score_case(case, [trace("dataset_summary")], final(), 2.0)
    bad = score_case(case, [], final(unverified=["9"]), 4.0)
    summary = summarize([good, bad])
    assert summary["cases"] == 2 and summary["tool_selection_accuracy"] == 0.5
    assert summary["grounded_rate"] == 0.5 and summary["median_seconds"] == 3.0
    assert summarize([])["cases"] == 0


def test_run_case_against_mock_server():
    toolbox = Toolbox(generate_synthetic(n=6000, seed=1))

    def respond(payload, n):
        if n == 1:
            return sse(*tool_call_chunks("sample_ratio_check", {}))
        return sse(*content_chunks("The split looks fine."))

    case = {
        "id": "srm",
        "question": "Is the split fine?",
        "expected_tool": "sample_ratio_check",
        "expected_args": {},
    }
    result = run_case(case, toolbox, MockLlmServer(respond).client())
    assert result.tool_correct and result.answered and result.grounded
    assert result.called_tools == ["sample_ratio_check"]
    assert pytest.approx(result.seconds, abs=5) == result.seconds
