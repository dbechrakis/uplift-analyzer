from __future__ import annotations

import numpy as np

HOLDOUT_MODULUS = 5


def holdout_rows(n_pool: int) -> np.ndarray:
    rows = np.arange(n_pool)
    return rows[rows % HOLDOUT_MODULUS == 0]


def eligible_rows(n_pool: int) -> np.ndarray:
    rows = np.arange(n_pool)
    return rows[rows % HOLDOUT_MODULUS != 0]


def draw_experiment(
    eligible: np.ndarray, n_arm: int, seed_sequence: np.random.SeedSequence
) -> tuple[np.ndarray, np.ndarray]:
    if 2 * n_arm > len(eligible):
        raise ValueError("The pool is too small for two arms of this size")
    rng = np.random.default_rng(seed_sequence)
    picked = eligible[rng.choice(len(eligible), size=2 * n_arm, replace=False)]
    return picked[:n_arm], picked[n_arm:]


def inject_lift(
    outcome_b: np.ndarray, relative_lift: float, baseline_rate: float, rng: np.random.Generator
) -> np.ndarray:
    if not 0 < baseline_rate < 1:
        raise ValueError("baseline_rate must be strictly between 0 and 1")
    flip_probability = baseline_rate * relative_lift / (1 - baseline_rate)
    if not 0 <= flip_probability <= 1:
        raise ValueError("The relative lift is not reachable from this baseline rate")
    flips = (outcome_b == 0) & (rng.random(len(outcome_b)) < flip_probability)
    return np.where(flips, 1, outcome_b).astype(outcome_b.dtype)


def look_sizes(n_arm: int, n_looks: int) -> np.ndarray:
    if n_looks < 1 or n_arm % n_looks:
        raise ValueError("n_looks must divide n_arm")
    return (np.arange(1, n_looks + 1) * (n_arm // n_looks)).astype(np.int64)


def cumulative_sums(outcome: np.ndarray, looks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(outcome, dtype=np.float64)
    sums = np.cumsum(values)[looks - 1]
    squares = np.cumsum(values * values)[looks - 1]
    return sums, squares
