from __future__ import annotations

from typing import Any

SYNTHETIC_NOTE = "Note: this is SYNTHETIC demo data, not a real experiment."
CAUSAL_NOTE = (
    "Treatment was randomly assigned in this experiment, so this difference is a causal effect "
    "of the treatment, for this population and period only."
)


def _pct(value: float, digits: int = 2) -> str:
    return f"{value * 100:.{digits}f}%"


def _pp(value: float) -> str:
    return f"{value * 100:+.3f} percentage points"


def _significance(p_value: float) -> str:
    if p_value < 0.05:
        return "statistically significant at the 5% level"
    return "not statistically distinguishable from zero at the 5% level"


def _explain_ate(r: dict[str, Any]) -> str:
    return (
        f"Among {r['n_treated']:,} treated and {r['n_control']:,} control users, the "
        f"{r['outcome']} rate was {_pct(r['rate_treated'], 3)} with treatment versus "
        f"{_pct(r['rate_control'], 3)} without. The treatment effect is {_pp(r['ate'])} "
        f"(95% CI {r['ci_low'] * 100:.3f} to {r['ci_high'] * 100:.3f}, p = {r['p_value']:.3g}), "
        f"which is {_significance(r['p_value'])}."
    )


def _explain_relative(r: dict[str, Any]) -> str:
    if r["relative_uplift"] is None:
        return "Relative uplift is undefined because the control group has a zero rate."
    text = (
        f"Treatment changed the {r['outcome']} rate by {r['relative_uplift'] * 100:+.1f}% "
        f"relative to control ({_pct(r['rate_control'], 3)} to {_pct(r['rate_treated'], 3)})."
    )
    if r["ci_low"] is not None:
        text += f" 95% CI: {r['ci_low'] * 100:.1f}% to {r['ci_high'] * 100:.1f}%."
    return text


def _explain_segments(r: dict[str, Any]) -> str:
    rows = [s for s in r["segments"] if s["ate"] is not None]
    if not rows:
        return "No segment had both treated and control users, so no effect could be estimated."
    significant = [s for s in rows if s["significant"]]
    best = max(rows, key=lambda s: s["ate"])
    worst = min(rows, key=lambda s: s["ate"])
    text = (
        f"Splitting by {r['feature']} into {len(rows)} groups, {len(significant)} show a "
        f"significant {r['outcome']} effect after Benjamini-Hochberg correction. "
        f"The largest effect is {_pp(best['ate'])} for {best['label']}; the smallest is "
        f"{_pp(worst['ate'])} for {worst['label']}."
    )
    return text + " Segment differences are descriptive and not proof of why groups respond."


def _explain_qini(r: dict[str, Any]) -> str:
    beats = r["qini_ci_low"] > 0
    verdict = (
        "the interval excludes zero, so the model ranks users better than random targeting"
        if beats
        else "the interval includes zero, so there is no clear gain over random targeting"
    )
    return (
        f"A two-model uplift learner trained on {r['n_train']:,} users and evaluated on "
        f"{r['n_test']:,} held-out users gives a Qini coefficient of {r['qini_coefficient']:.1f} "
        f"(95% bootstrap CI {r['qini_ci_low']:.1f} to {r['qini_ci_high']:.1f}); {verdict}. "
        f"The coefficient is in incremental {r['outcome']} counts on the test set."
    )


def _explain_srm(r: dict[str, Any]) -> str:
    verdict = (
        "WARNING: the split differs from the design more than chance allows; "
        "results may be unreliable."
        if r["mismatch"]
        else "No sign of a sample ratio problem."
    )
    return (
        f"{_pct(r['observed_share_treated'], 2)} of users were treated versus "
        f"{_pct(r['expected_share_treated'], 2)} expected (p = {r['p_value']:.3g}). {verdict}"
    )


def _explain_summary(r: dict[str, Any]) -> str:
    return (
        f"The sample has {r['rows']:,} users, {_pct(r['treated_share'], 1)} treated. Base rates: "
        f"conversion {_pct(r['conversion_rate'], 3)}, visit {_pct(r['visit_rate'], 2)}, "
        f"exposure {_pct(r['exposure_rate'], 2)}."
    )


EXPLAINERS = {
    "average_treatment_effect": _explain_ate,
    "relative_uplift": _explain_relative,
    "uplift_by_segment": _explain_segments,
    "qini_analysis": _explain_qini,
    "sample_ratio_check": _explain_srm,
    "dataset_summary": _explain_summary,
}


def explain(name: str, result: dict[str, Any]) -> str:
    text = EXPLAINERS[name](result)
    if (
        name in ("average_treatment_effect", "relative_uplift")
        and result["data_source"] == "criteo"
    ):
        text += " " + CAUSAL_NOTE
    if result.get("data_source") == "synthetic":
        text += " " + SYNTHETIC_NOTE
    return text
