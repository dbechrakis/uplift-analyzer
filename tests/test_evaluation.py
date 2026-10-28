import json
from pathlib import Path

import pytest

from app.agent import AgentEvent
from app.evaluation import run_case, score_case, summarize
from app.tools import TOOL_DEFINITIONS, Toolbox, ToolTrace, tool_specs
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
    assert expected == {s["function"]["name"] for s in tool_specs()}
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


def test_also_accept_counts_defensible_alternatives_without_checking_their_args():
    case = {
        "id": "x",
        "question": "q",
        "expected_tool": "average_treatment_effect",
        "expected_args": {"outcome": "conversion"},
        "also_accept": ["relative_uplift"],
    }
    alt = score_case(case, [trace("relative_uplift", outcome="visit")], final(), 1.0)
    assert alt.tool_correct and alt.args_correct
    primary_wrong_arg = score_case(
        case, [trace("average_treatment_effect", outcome="visit")], final(), 1.0
    )
    assert primary_wrong_arg.tool_correct and not primary_wrong_arg.args_correct
    assert not score_case(case, [trace("dataset_summary")], final(), 1.0).tool_correct


def test_aggregate_reports_mean_spread_and_stability():
    from app.evaluation import aggregate

    case = {"id": "x", "question": "q", "expected_tool": "dataset_summary", "expected_args": {}}
    good = score_case(case, [trace("dataset_summary")], final(), 2.0)
    bad = score_case(case, [], final(), 4.0)
    summary = aggregate([[good, good], [good, bad], [good, good]])
    assert summary["runs"] == 3 and summary["cases"] == 2
    assert summary["tool_selection_accuracy"]["mean"] == pytest.approx(0.833, abs=0.001)
    assert summary["tool_selection_accuracy"]["min"] == 0.5
    assert summary["tool_selection_accuracy"]["std"] > 0
    assert summary["stable_cases"] == 1


def test_traces_are_stored_and_grounding_can_be_rescored_offline(tmp_path):
    from app.evaluation import rescore

    case = {
        "id": "x",
        "question": "q",
        "expected_tool": "average_treatment_effect",
        "expected_args": {"outcome": "visit"},
    }
    good_trace = ToolTrace(
        name="average_treatment_effect",
        arguments={"outcome": "visit"},
        result={"ate": 0.0123, "p_value": 1e-9},
    )
    ok = score_case(case, [good_trace], final("The effect is 1.23 points."), 1.0)
    assert ok.traces[0]["result"]["ate"] == 0.0123 and ok.traces[0]["arguments"] == {
        "outcome": "visit"
    }
    bad = score_case(case, [good_trace], final("The effect is 9.99 points.", ["9.99"]), 1.0)
    stale = {**bad.__dict__, "grounded": False}
    report = {"runs": [[ok.__dict__, stale]]}
    path = tmp_path / "r.json"
    path.write_text(json.dumps(report))
    redone = rescore(path)
    flags = [r["grounded"] for r in redone["runs"][0]]
    assert flags == [True, False]
    assert redone["aggregate"]["grounded_rate"]["mean"] == 0.5
