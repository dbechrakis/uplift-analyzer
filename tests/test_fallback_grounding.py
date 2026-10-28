import pytest

from app.fallback import explain
from app.grounding import ungrounded_numbers
from app.tools import TOOL_DEFINITIONS, Toolbox
from core.synthetic import generate_synthetic


@pytest.fixture(scope="module")
def toolbox():
    return Toolbox(generate_synthetic(n=12_000, seed=2))


def run(toolbox, name, **args):
    trace = toolbox.call(name, args)
    assert trace.error is None
    return trace.result


def test_every_tool_has_a_rule_based_explanation(toolbox, monkeypatch):
    from app import tools

    monkeypatch.setattr(tools, "QINI_BOOTSTRAP", 10)
    for name, definition in TOOL_DEFINITIONS.items():
        args = {}
        if "outcome" in definition["properties"]:
            args["outcome"] = "conversion"
        if "feature" in definition["properties"]:
            args["feature"] = "f0"
        if "relative_lift" in definition["properties"]:
            args["relative_lift"] = 0.2
        text = explain(name, run(toolbox, name, **args))
        assert isinstance(text, str) and len(text) > 40


def test_explanations_state_real_numbers_and_flag_synthetic(toolbox):
    result = run(toolbox, "average_treatment_effect", outcome="conversion")
    text = explain("average_treatment_effect", result)
    assert f"{result['n_treated']:,}" in text and f"{result['ate'] * 100:+.3f}" in text
    assert "SYNTHETIC" in text
    assert ungrounded_numbers(text, result) == []


def test_causal_note_only_for_real_randomized_data(toolbox):
    result = run(toolbox, "average_treatment_effect", outcome="visit")
    assert "causal" not in explain("average_treatment_effect", result)
    assert "not forecasts" in explain(
        "average_treatment_effect", {**result, "data_source": "criteo"}
    )


def test_srm_explanation_warns_on_mismatch(toolbox):
    bad = run(toolbox, "sample_ratio_check", expected_share_treated=0.3)
    assert "WARNING" in explain("sample_ratio_check", bad)
    assert "No sign" in explain("sample_ratio_check", {**bad, "mismatch": False})


def test_relative_uplift_explanation_handles_undefined_case():
    undefined = {
        "outcome": "conversion",
        "data_source": "constructed",
        "relative_uplift": None,
        "rate_control": 0.0,
        "rate_treated": 0.1,
        "ci_low": None,
        "ci_high": None,
    }
    assert "undefined" in explain("relative_uplift", undefined)


def test_grounding_accepts_rounded_percent_and_formatted_numbers():
    result = {"ate": 0.011923, "p_value": 7.47e-08, "n": 425183, "rel": 0.5914}
    text = "Effect +1.19 pp (about 1.2), p = 7.5e-08, 425,183 users, lift 59%, 59.1%."
    assert ungrounded_numbers(text, result) == []


def test_grounding_flags_invented_numbers_and_ignores_names_and_thresholds():
    result = {"ate": 0.0119}
    flagged = ungrounded_numbers("Feature f0 shows 42.5 pp at p < 0.05, ate 1.19.", result)
    assert flagged == ["42.5"]


def test_grounding_uses_tool_arguments_too():
    assert ungrounded_numbers("Using 7 bins.", {}, {"n_bins": 7}) == []


def test_grounding_accepts_true_p_value_bounds_only():
    result = {"p_value": 1e-46}
    assert ungrounded_numbers("Highly significant (p < 0.0001).", result) == []
    assert ungrounded_numbers("Highly significant (p < 0.0001).", {"p_value": 0.5}) == ["0.0001"]
    assert ungrounded_numbers("About 0.0003 of users.", result) == ["0.0003"]


def test_grounding_reads_numbers_embedded_in_string_labels():
    result = {"segments": [{"label": "= 4.68 (82% of users)", "n": 409162}]}
    assert ungrounded_numbers("The default group, 82% of users, sits at 4.68.", result) == []
    assert ungrounded_numbers("It covers 61% of users.", result) == ["61"]


def test_grounding_tolerates_scientific_notation_rounded_to_one_digit():
    assert ungrounded_numbers("p was about 4e-6.", {"p": 3.97e-6}) == []
    assert ungrounded_numbers("p was about 9e-6.", {"p": 3.97e-6}) == ["9e-6"]
