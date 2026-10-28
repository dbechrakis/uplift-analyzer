import json

import pytest

from app import tools
from app.tools import TOOL_DEFINITIONS, Toolbox, ToolError, tool_specs, validate_arguments
from core.metrics import average_treatment_effect
from core.synthetic import generate_synthetic


@pytest.fixture(scope="module")
def toolbox():
    return Toolbox(generate_synthetic(n=12_000, seed=5))


def test_specs_are_valid_openai_tool_definitions():
    specs = tool_specs()
    hidden = {n for n, d in TOOL_DEFINITIONS.items() if d.get("expose_to_llm") is False}
    assert hidden == {"compare_uplift_learners"}
    assert {s["function"]["name"] for s in specs} == set(TOOL_DEFINITIONS) - hidden
    for spec in specs:
        assert spec["type"] == "function"
        params = spec["function"]["parameters"]
        assert params["type"] == "object" and params["additionalProperties"] is False
        assert set(params["required"]) <= set(params["properties"])
    json.dumps(specs)


def test_tool_matches_direct_core_call(toolbox):
    trace = toolbox.call("average_treatment_effect", {"outcome": "visit"})
    direct = average_treatment_effect(toolbox.df, "visit", data_source="synthetic").to_dict()
    assert trace.error is None and trace.result == direct


@pytest.mark.parametrize(
    ("name", "args"),
    [
        ("dataset_summary", {}),
        ("sample_ratio_check", {"expected_share_treated": 0.5}),
        ("relative_uplift", {"outcome": "conversion"}),
        ("uplift_by_segment", {"feature": "f1", "outcome": "visit", "n_bins": 3}),
        ("covariate_balance", {}),
        ("regression_adjusted_effect", {"outcome": "visit"}),
        ("complier_effect", {"outcome": "conversion"}),
        ("required_sample_size", {"outcome": "conversion", "relative_lift": 0.2}),
        (
            "targeting_policy",
            {"outcome": "conversion", "audience_size": 50_000, "cost_per_contact": 0.01},
        ),
        ("compare_uplift_learners", {"outcome": "visit"}),
    ],
)
def test_every_tool_runs_and_is_json_serializable(toolbox, name, args, monkeypatch):
    if name in ("targeting_policy", "compare_uplift_learners"):
        monkeypatch.setattr(tools, "QINI_BOOTSTRAP", 8)
    trace = toolbox.call(name, args)
    assert trace.error is None
    json.dumps(trace.result)
    assert trace.result["data_source"] == "synthetic"


def test_qini_tool_strips_curves_for_the_llm_but_keeps_them_raw(toolbox, monkeypatch):
    monkeypatch.setattr(tools, "QINI_BOOTSTRAP", 10)
    trace = toolbox.call("qini_analysis", {"outcome": "conversion"})
    assert "curve_x" in trace.result
    payload = json.loads(trace.llm_payload())
    assert "curve_x" not in payload and payload["curve_points"] == len(trace.result["curve_x"])


@pytest.mark.parametrize(
    ("name", "args", "message"),
    [
        ("nope", {}, "Unknown tool"),
        ("relative_uplift", {}, "Missing required"),
        ("relative_uplift", {"outcome": "clicks"}, "must be one of"),
        ("relative_uplift", {"outcome": "visit", "extra": 1}, "Unexpected arguments"),
        ("uplift_by_segment", {"feature": "f1", "outcome": "visit", "n_bins": 99}, "at most"),
        ("uplift_by_segment", {"feature": "f1", "outcome": "visit", "n_bins": "3"}, "integer"),
        ("uplift_by_segment", {"feature": "f1", "outcome": "visit", "n_bins": True}, "integer"),
        ("sample_ratio_check", {"expected_share_treated": 2}, "at most"),
        ("complier_effect", {"outcome": "exposure"}, "must be one of"),
        ("required_sample_size", {"outcome": "visit"}, "Missing required"),
        ("required_sample_size", {"outcome": "visit", "relative_lift": 0}, "at least"),
        ("targeting_policy", {"outcome": "visit", "cost_per_contact": -1}, "at least"),
        ("targeting_policy", {"outcome": "visit", "audience_size": 5}, "at least"),
    ],
)
def test_invalid_calls_become_errors_not_exceptions(toolbox, name, args, message):
    trace = toolbox.call(name, args)
    assert trace.result is None and message in trace.error
    assert json.loads(trace.llm_payload())["error"] == trace.error


def test_non_object_arguments_rejected():
    with pytest.raises(ToolError):
        validate_arguments("dataset_summary", ["a"])


def test_results_are_cached_per_arguments(toolbox):
    first = toolbox.call("relative_uplift", {"outcome": "exposure"})
    second = toolbox.call("relative_uplift", {"outcome": "exposure"})
    other = toolbox.call("relative_uplift", {"outcome": "visit"})
    assert not first.cached and second.cached and not other.cached
    assert second.result is first.result


def test_signature_names_the_function_and_arguments(toolbox):
    trace = toolbox.call("relative_uplift", {"outcome": "visit"})
    assert trace.signature() == "relative_uplift(outcome='visit')"


def test_policy_and_balance_payloads_are_compact_for_the_llm(toolbox, monkeypatch):
    monkeypatch.setattr(tools, "QINI_BOOTSTRAP", 8)
    policy = toolbox.call("targeting_policy", {"outcome": "conversion"})
    assert "points" in policy.result
    assert "points" not in json.loads(policy.llm_payload())
    balance = toolbox.call("covariate_balance", {})
    assert len(balance.llm_payload()) < 4000


def test_required_sample_size_uses_the_control_rate_as_baseline(toolbox):
    trace = toolbox.call("required_sample_size", {"outcome": "visit", "relative_lift": 0.1})
    control = toolbox.df.loc[toolbox.df["treatment"] == 0, "visit"].mean()
    assert trace.result["baseline_rate"] == pytest.approx(control)
    assert trace.result["data_source"] == "synthetic"
