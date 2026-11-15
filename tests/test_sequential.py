from __future__ import annotations

import numpy as np
import pytest

from core.sequential import (
    RULES,
    confidence_sequence,
    cuped_adjust,
    decision_matrix,
    difference_and_se,
    first_true,
    msprt_p_values,
    spending_boundaries,
    stopping_look,
    z_statistics,
)


def simulate(n_experiments: int, n_arm: int, n_looks: int, seed: int, effect: float = 0.0):
    rng = np.random.default_rng(seed)
    block = n_arm // n_looks
    a = rng.normal(0.0, 1.0, size=(n_experiments, n_looks, block))
    b = rng.normal(effect, 1.0, size=(n_experiments, n_looks, block))
    sum_a = np.cumsum(a.sum(axis=2), axis=1)
    sum_b = np.cumsum(b.sum(axis=2), axis=1)
    sq_a = np.cumsum((a**2).sum(axis=2), axis=1)
    sq_b = np.cumsum((b**2).sum(axis=2), axis=1)
    n = np.arange(1, n_looks + 1) * block
    return sum_a, sum_b, sq_a, sq_b, n.astype(float)


def rejection_rate(rule: str, n_looks: int, seed: int, effect: float = 0.0, tau2: float = 0.0004):
    sum_a, sum_b, sq_a, sq_b, n = simulate(20_000, 400, n_looks, seed, effect)
    diff, se = difference_and_se(sum_a, sum_b, sq_a, sq_b, n)
    info = n / n[-1]
    exceeds = decision_matrix(rule, diff, se, info, 0.05, tau2)
    return float((first_true(exceeds) >= 0).mean())


def test_pocock_constant_for_five_looks():
    bounds = spending_boundaries(np.arange(1, 6) / 5, 0.05, "pocock")
    assert bounds == pytest.approx(2.413, abs=0.01)
    assert np.ptp(bounds) == 0.0


def test_obf_spending_final_value_for_five_looks_and_classical_constant():
    info = np.arange(1, 6) / 5
    bounds = spending_boundaries(info, 0.05, "obf")
    assert bounds[-1] == pytest.approx(2.063, abs=0.01)
    assert np.all(np.diff(bounds) < 0)
    sum_a, sum_b, sq_a, sq_b, n = simulate(40_000, 500, 5, 11)
    z = z_statistics(sum_a, sum_b, sq_a, sq_b, n)
    classical = 2.040 / np.sqrt(info)
    assert float((stopping_look(z, classical) >= 0).mean()) == pytest.approx(0.05, abs=0.006)


def test_group_sequential_rules_hold_level_and_peeking_does_not():
    assert 0.044 <= rejection_rate("fixed", 5, 1) <= 0.056
    assert 0.044 <= rejection_rate("pocock", 5, 2) <= 0.056
    assert 0.044 <= rejection_rate("obf", 5, 3) <= 0.056
    assert rejection_rate("peek", 5, 4) > 0.12
    assert rejection_rate("bonferroni", 5, 5) <= 0.056
    assert rejection_rate("msprt", 5, 6) <= 0.056


def test_msprt_p_values_never_increase():
    sum_a, sum_b, sq_a, sq_b, n = simulate(500, 400, 40, 7, effect=0.05)
    diff, se = difference_and_se(sum_a, sum_b, sq_a, sq_b, n)
    p = msprt_p_values(diff, se, 0.01)
    assert np.all(np.diff(p, axis=1) <= 1e-15)
    assert p.min() >= 0 and p.max() <= 1


def test_confidence_sequence_covers_the_true_effect_at_every_look():
    effect = 0.1
    sum_a, sum_b, sq_a, sq_b, n = simulate(5_000, 400, 20, 8, effect=effect)
    diff, se = difference_and_se(sum_a, sum_b, sq_a, sq_b, n)
    low, high = confidence_sequence(diff, se, 0.01, 0.05)
    covered = ((low <= effect) & (effect <= high)).all(axis=1)
    assert covered.mean() >= 0.95


def test_cuped_keeps_the_mean_difference_and_cuts_variance_by_squared_correlation():
    rng = np.random.default_rng(9)
    n = 400_000
    x = rng.normal(size=n)
    y = 0.6 * x + 0.8 * rng.normal(size=n)
    adjusted = cuped_adjust(y, x)
    assert np.var(adjusted) / np.var(y) == pytest.approx(1 - 0.36, abs=0.01)
    assert adjusted.mean() == pytest.approx(y.mean(), abs=1e-9)
    arm = rng.random(n) < 0.5
    raw_gap = y[arm].mean() - y[~arm].mean()
    adjusted_gap = adjusted[arm].mean() - adjusted[~arm].mean()
    assert adjusted_gap == pytest.approx(raw_gap, abs=0.01)


def test_cuped_with_fixed_theta_and_center_is_row_local():
    y = np.array([1.0, 0.0, 1.0])
    x = np.array([0.2, 0.1, 0.4])
    assert cuped_adjust(y, x, theta=1.0, center=0.2) == pytest.approx([1.0, 0.1, 0.8])


def test_one_look_equals_fixed_except_msprt_which_is_conservative():
    sum_a, sum_b, sq_a, sq_b, n = simulate(4_000, 400, 1, 10, effect=0.1)
    diff, se = difference_and_se(sum_a, sum_b, sq_a, sq_b, n)
    info = np.array([1.0])
    fixed = decision_matrix("fixed", diff, se, info)
    for rule in ("peek", "bonferroni", "pocock", "obf"):
        assert np.array_equal(decision_matrix(rule, diff, se, info), fixed)
    assert decision_matrix("msprt", diff, se, info, mixing_variance=0.0004).sum() <= fixed.sum()


def test_rules_and_inputs_are_validated():
    assert set(RULES) == {"fixed", "peek", "bonferroni", "pocock", "obf", "msprt"}
    with pytest.raises(ValueError):
        spending_boundaries([0.5, 0.4, 1.0])
    with pytest.raises(ValueError):
        spending_boundaries([0.5, 0.9])
    with pytest.raises(ValueError):
        spending_boundaries([0.5, 1.0], kind="other")
    with pytest.raises(ValueError):
        decision_matrix("other", np.zeros((1, 2)), np.ones((1, 2)), np.array([0.5, 1.0]))
    with pytest.raises(ValueError):
        decision_matrix("msprt", np.zeros((1, 2)), np.ones((1, 2)), np.array([0.5, 1.0]))
    with pytest.raises(ValueError):
        msprt_p_values(np.zeros(2), np.ones(2), 0.0)


def test_unequal_arms_use_their_own_counts():
    diff, se = difference_and_se(
        np.array([10.0]),
        np.array([60.0]),
        np.array([10.0]),
        np.array([60.0]),
        np.array([100.0]),
        np.array([200.0]),
    )
    assert diff[0] == pytest.approx(0.3 - 0.1)
    var_a = (10 - 100 * 0.1**2) / 99
    var_b = (60 - 200 * 0.3**2) / 199
    assert se[0] == pytest.approx(np.sqrt(var_a / 100 + var_b / 200))
    z = z_statistics(
        np.array([10.0]),
        np.array([60.0]),
        np.array([10.0]),
        np.array([60.0]),
        np.array([100.0]),
        np.array([200.0]),
    )
    assert z[0] == pytest.approx(diff[0] / se[0])


def test_msprt_treats_a_zero_standard_error_as_no_evidence():
    diff = np.array([[0.0, 0.0, 0.2, 0.2]])
    se = np.array([[0.0, 0.0, 0.05, 0.04]])
    p = msprt_p_values(diff, se, 0.01)
    assert np.all(np.isfinite(p))
    assert p[0, 0] == 1.0 and p[0, 1] == 1.0
    assert p[0, 3] < 0.05
    low, high = confidence_sequence(diff, se, 0.01)
    assert np.all(np.isfinite(low)) and np.all(np.isfinite(high))
