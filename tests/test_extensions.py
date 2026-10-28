import json

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from scipy import stats

from core.adjusted import regression_adjusted_effect
from core.cace import complier_effect
from core.data import FEATURES
from core.learners import LEARNER_NAMES, compare_uplift_learners
from core.metrics import average_treatment_effect
from core.policy import policy_profit, targeting_policy
from core.power import minimum_detectable_effect, required_sample_size
from core.synthetic import generate_synthetic


def frame(treatment, exposure, conversion, visit=None):
    n = len(treatment)
    df = pd.DataFrame(
        {
            "treatment": treatment,
            "exposure": exposure,
            "conversion": conversion,
            "visit": visit if visit is not None else np.zeros(n),
        }
    )
    for f in FEATURES:
        df[f] = 0.0
    return df


def constructed_cace_frame():
    treatment = [1] * 1000 + [0] * 1000
    exposure = [1] * 400 + [0] * 600 + [0] * 1000
    conversion = [1] * 100 + [0] * 300 + [1] * 60 + [0] * 540 + [1] * 130 + [0] * 870
    return frame(treatment, exposure, conversion)


def test_cace_known_answer():
    r = complier_effect(constructed_cace_frame())
    assert r.exposure_rate_treated == pytest.approx(0.4) and r.exposure_rate_control == 0
    assert r.itt == pytest.approx(0.03)
    assert r.cace == pytest.approx(0.075)
    assert r.one_sided is True
    assert r.never_taker_rate == pytest.approx(0.10)
    assert r.complier_rate_treated == pytest.approx(0.25)
    assert r.complier_rate_control == pytest.approx(0.175)
    assert r.complier_rate_treated - r.complier_rate_control == pytest.approx(r.cace)
    assert r.naive_exposed_vs_control == pytest.approx(0.12)
    assert r.cace_ci_low < r.cace < r.cace_ci_high
    json.loads(r.to_json())


def test_cace_itt_matches_average_treatment_effect():
    df = constructed_cace_frame()
    assert complier_effect(df).itt == pytest.approx(average_treatment_effect(df).ate)


def test_cace_ci_has_nominal_coverage_in_simulation():
    rng = np.random.default_rng(11)
    true_effect, hits, naive_gaps = 0.15, 0, []
    runs = 300
    for _ in range(runs):
        n = 6000
        z = (rng.random(n) < 0.5).astype(int)
        complier = rng.random(n) < 0.4
        exposed = z * complier
        base = np.where(complier, 0.20, 0.05)
        y = (rng.random(n) < base + true_effect * exposed).astype(int)
        r = complier_effect(frame(z, exposed, y))
        hits += r.cace_ci_low <= true_effect <= r.cace_ci_high
        naive_gaps.append(r.naive_exposed_vs_control - true_effect)
    assert 0.91 <= hits / runs <= 0.99
    assert np.mean(naive_gaps) > 0.05


def test_cace_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="first stage"):
        complier_effect(constructed_cace_frame(), outcome="exposure")
    no_stage = frame([1, 1, 0, 0], [0, 0, 0, 0], [1, 0, 1, 0])
    with pytest.raises(ValueError, match="weak first stage"):
        complier_effect(no_stage)


def test_cace_flags_two_sided_noncompliance():
    df = constructed_cace_frame()
    df.loc[df.index[-1], "exposure"] = 1
    r = complier_effect(df)
    assert r.one_sided is False and r.complier_rate_control is None


def adjustment_frame(n=20_000, seed=3):
    rng = np.random.default_rng(seed)
    df = frame((rng.random(n) < 0.5).astype(int), np.zeros(n), np.zeros(n))
    x = rng.normal(size=(n, len(FEATURES)))
    df[FEATURES] = x
    df["conversion"] = 0.1 * df["treatment"] + 0.8 * x[:, 0] + rng.normal(scale=0.5, size=n)
    return df


def test_regression_adjustment_matches_statsmodels_exactly():
    df = adjustment_frame(5000)
    r = regression_adjusted_effect(df)
    x = df[FEATURES].to_numpy()
    x = x - x.mean(axis=0)
    t = df["treatment"].to_numpy(float)
    design = np.column_stack([np.ones(len(df)), t, x, x * t[:, None]])
    fit = sm.OLS(df["conversion"].to_numpy(), design).fit(cov_type="HC1")
    assert r.adjusted_ate == pytest.approx(fit.params[1], abs=1e-9)
    assert r.std_error == pytest.approx(fit.bse[1], abs=1e-9)
    assert r.p_value == pytest.approx(fit.pvalues[1], abs=1e-9)


def test_regression_adjustment_shrinks_the_interval_when_covariates_predict():
    r = regression_adjusted_effect(adjustment_frame())
    assert r.adjusted_ate == pytest.approx(0.1, abs=0.03)
    assert r.std_error < r.unadjusted_std_error
    assert r.variance_reduction > 0.5
    assert r.ci_low < 0.1 < r.ci_high


def test_regression_adjustment_is_harmless_when_covariates_are_noise():
    df = adjustment_frame()
    df["conversion"] = 0.1 * df["treatment"] + np.random.default_rng(1).normal(size=len(df))
    r = regression_adjusted_effect(df)
    assert abs(r.variance_reduction) < 0.01


def test_regression_adjustment_validates_inputs(synthetic_df):
    with pytest.raises(ValueError):
        regression_adjusted_effect(synthetic_df, features=["nope"])
    with pytest.raises(ValueError):
        regression_adjusted_effect(synthetic_df[synthetic_df["treatment"] == 1])


def test_sample_size_reproduces_textbook_value():
    r = required_sample_size(0.5, absolute_lift=-0.1)
    assert r.n_treated == 388 and r.n_control == 388
    assert r.relative_effect == pytest.approx(-0.2)


def test_sample_size_relative_and_absolute_agree_and_unequal_split():
    a = required_sample_size(0.02, relative_lift=0.25)
    b = required_sample_size(0.02, absolute_lift=0.005)
    assert a.n_total == b.n_total
    skewed = required_sample_size(0.02, relative_lift=0.25, treated_share=0.85)
    assert skewed.n_total > a.n_total
    assert skewed.n_treated / skewed.n_total == pytest.approx(0.85, abs=0.001)


def test_sample_size_matches_simulated_power():
    r = required_sample_size(0.10, absolute_lift=0.03, power=0.8)
    rng = np.random.default_rng(5)
    rejections = 0
    for _ in range(400):
        x1 = rng.binomial(r.n_treated, 0.13)
        x0 = rng.binomial(r.n_control, 0.10)
        pooled = (x1 + x0) / (r.n_treated + r.n_control)
        se = np.sqrt(pooled * (1 - pooled) * (1 / r.n_treated + 1 / r.n_control))
        z = (x1 / r.n_treated - x0 / r.n_control) / se
        rejections += abs(z) > stats.norm.ppf(0.975)
    assert 0.74 <= rejections / 400 <= 0.86


def test_minimum_detectable_effect_inverts_sample_size():
    r = required_sample_size(0.05, relative_lift=0.2, treated_share=0.7)
    m = minimum_detectable_effect(0.05, r.n_total, treated_share=0.7)
    assert m.absolute_effect == pytest.approx(0.01, rel=0.01)
    assert minimum_detectable_effect(0.05, 4 * r.n_total, treated_share=0.7).absolute_effect < (
        m.absolute_effect
    )


def test_power_validates_inputs():
    with pytest.raises(ValueError):
        required_sample_size(0.5)
    with pytest.raises(ValueError):
        required_sample_size(0.5, relative_lift=0.1, absolute_lift=0.05)
    with pytest.raises(ValueError):
        required_sample_size(0.95, absolute_lift=0.1)
    with pytest.raises(ValueError):
        required_sample_size(0.0, relative_lift=0.1)
    with pytest.raises(ValueError):
        minimum_detectable_effect(0.5, 5)


def test_policy_profit_known_answer():
    fraction = np.array([0, 0.25, 0.5, 0.75, 1.0])
    qini = np.array([0, 1, 1, 1, 0.0])
    inc, model, random = policy_profit(fraction, qini, 1.0, 4, value=10.0, cost=1.0)
    assert inc.tolist() == [0, 1, 1, 1, 0]
    assert model.tolist() == [0, 9, 8, 7, -4]
    assert random.tolist() == [0, -1, -2, -3, -4]
    assert int(np.argmax(model)) == 1


@pytest.fixture(scope="module")
def policy_df():
    return generate_synthetic(n=30_000, seed=21)


def test_policy_breakeven_equals_value_times_ate(policy_df):
    r = targeting_policy(policy_df, max_iter=15, n_bootstrap=20, value_per_conversion=30.0)
    test_ate = average_treatment_effect(policy_df).ate
    assert r.breakeven_cost_target_all == pytest.approx(30.0 * test_ate, rel=0.15)
    assert r.points[0].fraction == 0 and r.points[-1].fraction == 1
    assert r.best_profit >= r.profit_target_all
    assert r.best_profit >= r.profit_random_same_fraction - 1e-6
    assert r.best_profit_ci_low <= r.best_profit_ci_high
    json.loads(r.to_json())


def test_policy_targets_nobody_when_contact_cost_exceeds_any_gain(policy_df):
    r = targeting_policy(policy_df, max_iter=10, n_bootstrap=10, cost_per_contact=1000.0)
    assert r.best_fraction == 0 and r.best_profit == 0


def test_policy_targets_a_subset_when_cost_sits_between_segments(policy_df):
    base = targeting_policy(policy_df, max_iter=15, n_bootstrap=10)
    cost = base.breakeven_cost_target_all * 1.3
    r = targeting_policy(policy_df, max_iter=15, n_bootstrap=10, cost_per_contact=cost)
    assert r.profit_target_all < 0 < r.best_profit
    assert 0 < r.best_fraction < 1


def test_policy_validates_inputs(policy_df):
    with pytest.raises(ValueError):
        targeting_policy(policy_df, value_per_conversion=0)
    with pytest.raises(ValueError):
        targeting_policy(policy_df, cost_per_contact=-1)


def test_learner_comparison_structure_and_determinism():
    df = generate_synthetic(n=12_000, seed=17)
    a = compare_uplift_learners(df, max_iter=10, n_bootstrap=12, seed=4)
    b = compare_uplift_learners(df, max_iter=10, n_bootstrap=12, seed=4)
    assert a.to_dict() == b.to_dict()
    assert [r.name for r in a.learners] == list(LEARNER_NAMES)
    t_row = a.learners[0]
    assert t_row.diff_vs_t_learner == 0 and not t_row.differs_from_t_learner
    assert a.best_by_point_estimate in LEARNER_NAMES
    for row in a.learners:
        assert row.ci_low <= row.ci_high
    json.loads(a.to_json())


def test_every_learner_finds_the_planted_heterogeneity():
    df = generate_synthetic(n=40_000, seed=8)
    r = compare_uplift_learners(df, max_iter=30, n_bootstrap=20, seed=2)
    assert all(row.qini_coefficient > 0 for row in r.learners)


def test_balance_is_clean_on_a_properly_randomized_sample():
    from core.balance import covariate_balance

    r = covariate_balance(generate_synthetic(n=30_000, seed=40), max_iter=20, n_bootstrap=40)
    assert r.max_abs_smd < 0.05 and r.features_over_threshold == 0
    assert r.auc_ci_low <= 0.5 + 0.02 and r.joint_imbalance_detected in (True, False)
    assert len(r.rows) == len(FEATURES)
    json.loads(r.to_json())


def test_balance_detects_a_planted_assignment_leak():
    from core.balance import covariate_balance

    df = generate_synthetic(n=30_000, seed=41)
    rng = np.random.default_rng(0)
    df["treatment"] = (rng.random(len(df)) < 1 / (1 + np.exp(-1.2 * df["f2"]))).astype("int8")
    r = covariate_balance(df, max_iter=20, n_bootstrap=40)
    assert r.joint_imbalance_detected is True and r.auc > 0.6
    leaky = max(r.rows, key=lambda row: abs(row.smd))
    assert leaky.feature == "f2" and leaky.p_value_adjusted < 1e-6
    assert r.features_over_threshold >= 1


def test_balance_validates_inputs(synthetic_df):
    from core.balance import covariate_balance

    with pytest.raises(ValueError):
        covariate_balance(synthetic_df, features=["nope"])
    with pytest.raises(ValueError):
        covariate_balance(synthetic_df[synthetic_df["treatment"] == 1])
