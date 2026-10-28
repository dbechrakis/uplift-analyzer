from __future__ import annotations

import numpy as np
import pandas as pd

from core._common import check_alpha, check_outcome, resolve_source
from core.data import FEATURES
from core.results import PolicyPoint, PolicyResult
from core.uplift_model import qini_arrays, split_holdout, t_learner_scores


def policy_profit(
    fraction: np.ndarray,
    qini: np.ndarray,
    scale: float,
    audience_size: int,
    value: float,
    cost: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    incremental = qini * scale
    contacts = fraction * audience_size
    profit_model = value * incremental - cost * contacts
    profit_random = value * qini[-1] * scale * fraction - cost * contacts
    return incremental, profit_model, profit_random


def targeting_policy(
    df: pd.DataFrame,
    outcome: str = "conversion",
    audience_size: int = 1_000_000,
    value_per_conversion: float = 30.0,
    cost_per_contact: float = 0.05,
    features: list[str] | None = None,
    test_size: float = 0.3,
    seed: int = 42,
    max_iter: int = 100,
    n_bootstrap: int = 200,
    n_points: int = 101,
    alpha: float = 0.05,
    data_source: str | None = None,
) -> PolicyResult:
    check_alpha(alpha)
    check_outcome(outcome)
    if audience_size < 1 or value_per_conversion <= 0 or cost_per_contact < 0:
        raise ValueError("audience_size and value must be positive and cost non-negative")
    features = list(features or FEATURES)
    train, test = split_holdout(df, outcome, test_size, seed)
    score = t_learner_scores(train, test, outcome, features, seed, max_iter)
    y, t = test[outcome].to_numpy(), test["treatment"].to_numpy()
    fraction, qini = qini_arrays(y, t, score)
    treated_share = float(test["treatment"].mean())
    scale = audience_size / (len(test) * treated_share)
    _, profit_model, profit_random = policy_profit(
        fraction, qini, scale, audience_size, value_per_conversion, cost_per_contact
    )
    best = int(np.argmax(profit_model))
    n = len(y)
    rng = np.random.default_rng(seed)
    boot = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        fb, qb = qini_arrays(y[idx], t[idx], score[idx])
        position = min(int(round(fraction[best] * (len(fb) - 1))), len(fb) - 1)
        profit = value_per_conversion * qb[position] * scale - (
            cost_per_contact * fb[position] * audience_size
        )
        boot[b] = profit
    low, high = np.quantile(boot, [alpha / 2, 1 - alpha / 2])
    grid = np.unique(np.linspace(0, len(fraction) - 1, n_points).round().astype(int))
    points = [
        PolicyPoint(
            fraction=float(fraction[i]),
            contacts=int(round(fraction[i] * audience_size)),
            incremental_conversions=float(qini[i] * scale),
            profit_model=float(profit_model[i]),
            profit_random=float(profit_random[i]),
        )
        for i in grid
    ]
    breakeven = value_per_conversion * float(qini[-1]) * scale / audience_size
    return PolicyResult(
        outcome=outcome,
        data_source=resolve_source(df, data_source),
        audience_size=audience_size,
        value_per_conversion=value_per_conversion,
        cost_per_contact=cost_per_contact,
        best_fraction=float(fraction[best]),
        best_contacts=int(round(fraction[best] * audience_size)),
        best_incremental_conversions=float(qini[best] * scale),
        best_profit=float(profit_model[best]),
        best_profit_ci_low=float(low),
        best_profit_ci_high=float(high),
        profit_target_all=float(profit_model[-1]),
        profit_random_same_fraction=float(profit_random[best]),
        breakeven_cost_target_all=breakeven,
        n_bootstrap=n_bootstrap,
        confidence_level=1 - alpha,
        points=points,
    )
