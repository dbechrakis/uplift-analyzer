from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from core._common import check_alpha, check_outcome, resolve_source
from core.data import FEATURES
from core.results import AdjustedEffectResult


def regression_adjusted_effect(
    df: pd.DataFrame,
    outcome: str = "conversion",
    features: list[str] | None = None,
    alpha: float = 0.05,
    data_source: str | None = None,
) -> AdjustedEffectResult:
    check_alpha(alpha)
    check_outcome(outcome)
    features = list(features or FEATURES)
    unknown = [f for f in features if f not in FEATURES]
    if unknown:
        raise ValueError(f"Unknown features: {unknown}")
    t = df["treatment"].to_numpy(dtype=float)
    if t.min() == t.max():
        raise ValueError("Both treatment and control rows are required")
    y = df[outcome].to_numpy(dtype=float)
    x = df[features].to_numpy(dtype=float)
    x = x - x.mean(axis=0)
    n = len(y)
    design = np.column_stack([np.ones(n), t, x, x * t[:, None]])
    gram_inv = np.linalg.inv(design.T @ design)
    beta = gram_inv @ (design.T @ y)
    resid = y - design @ beta
    weighted = design * resid[:, None]
    meat = weighted.T @ weighted
    k = design.shape[1]
    cov = gram_inv @ meat @ gram_inv * (n / (n - k))
    effect = float(beta[1])
    se = float(np.sqrt(cov[1, 1]))
    z_crit = stats.norm.ppf(1 - alpha / 2)
    z = effect / se if se > 0 else 0.0
    treated = y[t == 1]
    control = y[t == 0]
    raw_ate = float(treated.mean() - control.mean())
    raw_se = float(np.sqrt(treated.var(ddof=1) / len(treated) + control.var(ddof=1) / len(control)))
    return AdjustedEffectResult(
        outcome=outcome,
        data_source=resolve_source(df, data_source),
        n=n,
        n_covariates=len(features),
        adjusted_ate=effect,
        std_error=se,
        ci_low=effect - z_crit * se,
        ci_high=effect + z_crit * se,
        confidence_level=1 - alpha,
        p_value=float(2 * stats.norm.sf(abs(z))),
        unadjusted_ate=raw_ate,
        unadjusted_std_error=raw_se,
        variance_reduction=1 - (se / raw_se) ** 2 if raw_se > 0 else 0.0,
    )
