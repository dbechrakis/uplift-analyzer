from __future__ import annotations

from typing import Any

SYNTHETIC_NOTE = "Note: this is SYNTHETIC demo data, not a real experiment."
CAUSAL_NOTE = (
    "Users were randomised inside the original tests, but this public file pools several tests "
    "and the features are not perfectly balanced between groups, so read the covariate-adjusted "
    "effect next to the raw one. Effect sizes are benchmark values, not forecasts for a real "
    "campaign."
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


def _explain_balance(r: dict[str, Any]) -> str:
    worst = max(r["rows"], key=lambda row: abs(row["smd"]))
    verdict = (
        "treatment is slightly predictable from the features, so assignment is not perfectly "
        "random with respect to them; read the covariate-adjusted effect next to the raw one"
        if r["joint_imbalance_detected"]
        else "treatment is not predictable from the features, so there is no sign of imbalance"
    )
    return (
        f"The largest standardized mean difference between treated and control is "
        f"{abs(worst['smd']):.3f} ({worst['feature']}); {r['features_over_threshold']} of "
        f"{len(r['rows'])} features exceed the usual 0.1 threshold. A classifier predicts "
        f"treatment from the features with AUC {r['auc']:.4f} (95% CI {r['auc_ci_low']:.4f} to "
        f"{r['auc_ci_high']:.4f}; 0.5 means perfectly random): {verdict}."
    )


def _explain_adjusted(r: dict[str, Any]) -> str:
    return (
        f"After adjusting for {r['n_covariates']} features, the effect on {r['outcome']} is "
        f"{_pp(r['adjusted_ate'])} (95% CI {r['ci_low'] * 100:.3f} to {r['ci_high'] * 100:.3f}), "
        f"versus {_pp(r['unadjusted_ate'])} without adjustment. Adjustment changes the "
        f"variance of the estimate by {-r['variance_reduction'] * 100:+.0f}%."
    )


def _explain_cace(r: dict[str, Any]) -> str:
    text = (
        f"Only {_pct(r['exposure_rate_treated'], 1)} of treated users were actually exposed. "
        f"The intent-to-treat effect on {r['outcome']} (effect of being assigned) is "
        f"{_pp(r['itt'])}; for users who actually saw the ad it is {_pp(r['cace'])} "
        f"(95% CI {r['cace_ci_low'] * 100:.2f} to {r['cace_ci_high'] * 100:.2f})."
    )
    text += (
        f" Comparing exposed treated users with all control users would give "
        f"{_pp(r['naive_exposed_vs_control'])}, which mixes the ad effect with differences "
        "between people who get exposed and people who do not."
    )
    return text + " This assumes the ad affects users only through being exposed."


def _explain_policy(r: dict[str, Any]) -> str:
    return (
        f"Scenario: audience {r['audience_size']:,}, value {r['value_per_conversion']:g} per "
        f"conversion, cost {r['cost_per_contact']:g} per contact. Targeting the model's top "
        f"{r['best_fraction'] * 100:.1f}% ({r['best_contacts']:,} contacts) yields about "
        f"{r['best_incremental_conversions']:,.0f} incremental conversions and a profit of "
        f"{r['best_profit']:,.0f} (95% CI {r['best_profit_ci_low']:,.0f} to "
        f"{r['best_profit_ci_high']:,.0f}). Targeting everyone gives "
        f"{r['profit_target_all']:,.0f}; random targeting of the same share gives "
        f"{r['profit_random_same_fraction']:,.0f}. Targeting everyone only pays when the cost "
        f"per contact is below {r['breakeven_cost_target_all']:.4f}. Value and cost are "
        "assumptions, not measured."
    )


def _explain_sample_size(r: dict[str, Any]) -> str:
    return (
        f"To detect a {r['relative_effect'] * 100:+.0f}% relative change in {r['outcome']} from a "
        f"baseline of {_pct(r['baseline_rate'], 3)} with {r['power'] * 100:.0f}% power at the "
        f"5% level you need {r['n_total']:,} users in total ({r['n_treated']:,} treated and "
        f"{r['n_control']:,} control with a {r['treated_share'] * 100:.0f}% treated share)."
    )


def _explain_learners(r: dict[str, Any]) -> str:
    different = [row["name"] for row in r["learners"] if row["differs_from_t_learner"]]
    rows = "; ".join(f"{row['name']} {row['qini_coefficient']:.1f}" for row in r["learners"])
    verdict = (
        f"Significantly different from the T-learner: {', '.join(different)}."
        if different
        else "None is significantly different from the T-learner."
    )
    return (
        f"Qini coefficients on the same {r['n_test']:,} held-out users: {rows}. The best point "
        f"estimate is the {r['best_by_point_estimate']}. {verdict}"
    )


EXPLAINERS = {
    "covariate_balance": _explain_balance,
    "regression_adjusted_effect": _explain_adjusted,
    "complier_effect": _explain_cace,
    "targeting_policy": _explain_policy,
    "required_sample_size": _explain_sample_size,
    "compare_uplift_learners": _explain_learners,
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
