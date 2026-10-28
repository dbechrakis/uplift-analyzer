import json

import numpy as np
import pytest

from core.synthetic import generate_synthetic
from core.uplift_model import qini_analysis, qini_arrays, qini_coefficient

Y = np.array([1, 0, 0, 1])
T = np.array([1, 0, 1, 0])


def test_qini_curve_known_answer():
    x, q = qini_arrays(Y, T, np.array([4.0, 3.0, 2.0, 1.0]))
    assert x.tolist() == [0, 0.25, 0.5, 0.75, 1.0]
    assert q.tolist() == [0, 1, 1, 1, 0]
    assert qini_coefficient(Y, T, np.array([4.0, 3.0, 2.0, 1.0])) == pytest.approx(0.75)


def test_reversed_ranking_scores_lower():
    good = qini_coefficient(Y, T, np.array([4.0, 3.0, 2.0, 1.0]))
    bad = qini_coefficient(Y, T, np.array([1.0, 2.0, 3.0, 4.0]))
    assert bad < good


def test_qini_curve_ends_at_total_incremental_responses():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 1000)
    t = rng.integers(0, 2, 1000)
    _, q = qini_arrays(y, t, rng.random(1000))
    expected = y[t == 1].sum() - y[t == 0].sum() * (t == 1).sum() / (t == 0).sum()
    assert q[-1] == pytest.approx(expected)


def test_random_scores_give_coefficient_near_zero():
    df = generate_synthetic(n=60_000, seed=3, heterogeneous=False)
    y, t = df["conversion"].to_numpy(), df["treatment"].to_numpy()
    scores = np.random.default_rng(5).random(len(df))
    _, q = qini_arrays(y, t, scores)
    assert abs(qini_coefficient(y, t, scores)) < 0.1 * abs(q[-1])


def test_qini_analysis_detects_planted_heterogeneity(synthetic_df):
    r = qini_analysis(synthetic_df, max_iter=30, n_bootstrap=60, seed=7)
    assert r.qini_coefficient > 0
    assert r.qini_ci_low < r.qini_coefficient < r.qini_ci_high
    assert r.data_source == "synthetic"
    assert r.n_train + r.n_test == len(synthetic_df)
    assert r.curve_x[0] == 0 and r.curve_x[-1] == 1
    assert r.curve_qini[0] == 0
    assert r.curve_random[-1] == pytest.approx(r.curve_qini[-1])
    assert len(r.curve_x) <= 100
    json.loads(r.to_json())


def test_qini_analysis_is_deterministic():
    df = generate_synthetic(n=8000, seed=11)
    a = qini_analysis(df, max_iter=10, n_bootstrap=10, seed=2)
    b = qini_analysis(df, max_iter=10, n_bootstrap=10, seed=2)
    assert a.to_dict() == b.to_dict()


def test_qini_analysis_validates_inputs(synthetic_df):
    with pytest.raises(ValueError):
        qini_analysis(synthetic_df, features=["nope"])
    with pytest.raises(ValueError):
        qini_analysis(synthetic_df, outcome="clicks")
