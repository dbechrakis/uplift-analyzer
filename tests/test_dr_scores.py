import numpy as np
import pandas as pd
import pytest

from core import dr_scores
from core.data import FEATURES
from core.synthetic import generate_synthetic


def imbalanced_frame(n: int = 40_000, seed: int = 0) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.random((n, len(FEATURES))).astype("float32")
    propensity = 0.2 + 0.6 * x[:, 0]
    treatment = (rng.random(n) < propensity).astype("int8")
    base = 0.05 + 0.35 * (1 - x[:, 0])
    effect = 0.30 * x[:, 0]
    p = np.clip(base + treatment * effect, 0, 1)
    outcome = (rng.random(n) < p).astype("int8")
    df = pd.DataFrame(x, columns=FEATURES)
    df["treatment"] = treatment
    df["conversion"] = outcome
    df["visit"] = outcome
    df["exposure"] = treatment
    df.attrs["data_source"] = "synthetic"
    return df, effect


def test_combine_scores_matches_a_hand_computed_example():
    mu1 = np.array([0.5, 0.5])
    mu0 = np.array([0.25, 0.25])
    propensity = np.array([0.5, 0.5])
    y = np.array([1.0, 0.0])
    t = np.array([1.0, 0.0])
    result = dr_scores.combine_scores(mu1, mu0, propensity, y, t)
    assert result[0] == pytest.approx(0.25 + 1.0 * (1 - 0.5) / 0.5)
    assert result[1] == pytest.approx(0.25 - 1.0 * (0 - 0.25) / 0.5)


def test_the_propensity_is_clipped_so_one_user_cannot_dominate():
    result = dr_scores.combine_scores(
        np.array([0.1]), np.array([0.1]), np.array([0.0]), np.array([1.0]), np.array([1.0])
    )
    assert result[0] == pytest.approx(0.9 / 0.01)


def test_the_scores_recover_the_average_effect_under_imbalanced_assignment():
    df, effect = imbalanced_frame()
    scores = dr_scores.cross_fitted_scores(df, "conversion", FEATURES, seed=0, max_iter=30)
    assert scores.mean() == pytest.approx(effect.mean(), abs=0.01)
    raw_gap = (
        df.loc[df["treatment"] == 1, "conversion"].mean()
        - df.loc[df["treatment"] == 0, "conversion"].mean()
    )
    assert abs(scores.mean() - effect.mean()) < abs(raw_gap - effect.mean())


def test_the_corrected_qini_is_closer_to_the_truth_when_assignment_is_imbalanced():
    df, effect = imbalanced_frame(seed=1)
    uplift_score = df["f0"].to_numpy(float)
    y = df["conversion"].to_numpy()
    t = df["treatment"].to_numpy()
    share = float(t.mean())
    oracle = dr_scores.corrected_qini_coefficient(effect, uplift_score, share)
    raw = dr_scores.qini_coefficient(y, t, uplift_score)
    scores = dr_scores.cross_fitted_scores(df, "conversion", FEATURES, seed=0, max_iter=30)
    corrected = dr_scores.corrected_qini_coefficient(scores, uplift_score, share)
    assert abs(corrected - oracle) < abs(raw - oracle)
    low, high = dr_scores.bootstrap_corrected_ci(scores, uplift_score, share, 100, 0.05, 0)
    assert low <= oracle <= high


def test_a_users_own_outcome_enters_its_score_only_through_the_residual_term():
    df, _ = imbalanced_frame(n=6_000, seed=2)
    first = dr_scores.cross_fitted_scores(df, "conversion", FEATURES, seed=0, max_iter=20)
    row = 17
    flipped = df.copy()
    flipped.loc[flipped.index[row], "conversion"] = (
        1 - flipped.loc[flipped.index[row], "conversion"]
    )
    second = dr_scores.cross_fitted_scores(flipped, "conversion", FEATURES, seed=0, max_iter=20)
    assert second[row] != first[row]
    own_change = abs(second[row] - first[row])
    assert own_change >= 1.0 - 1e-9


def test_the_corrected_curve_has_the_units_of_the_raw_curve_when_assignment_is_random():
    df = generate_synthetic(n=60_000, seed=3, treatment_share=0.5)
    uplift_score = df["f0"].to_numpy(float)
    y = df["conversion"].to_numpy()
    t = df["treatment"].to_numpy()
    scores = dr_scores.cross_fitted_scores(df, "conversion", FEATURES, seed=0, max_iter=30)
    corrected = dr_scores.corrected_qini_coefficient(scores, uplift_score, float(t.mean()))
    raw = dr_scores.qini_coefficient(y, t, uplift_score)
    assert corrected == pytest.approx(raw, rel=0.35, abs=15)


def test_the_analysis_reports_both_coefficients_and_where_each_lies(synthetic_df):
    result = dr_scores.doubly_robust_qini_analysis(
        synthetic_df, outcome="conversion", max_iter=20, n_bootstrap=30, folds=3
    )
    assert result.data_source == "synthetic"
    assert result.n_train + result.n_test == len(synthetic_df)
    assert result.difference == pytest.approx(result.qini_corrected - result.qini_raw)
    assert result.qini_raw_ci_low <= result.qini_raw <= result.qini_raw_ci_high
    assert result.raw_inside_corrected_interval == (
        result.qini_corrected_ci_low <= result.qini_raw <= result.qini_corrected_ci_high
    )
