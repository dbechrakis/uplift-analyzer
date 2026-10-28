from __future__ import annotations

import math

import pandas as pd
from scipy import stats

from core._common import check_alpha, resolve_source, split_arms
from core.results import AteResult, RelativeUpliftResult


def ate_from_counts(
    n_t: int, x_t: int, n_c: int, x_c: int, alpha: float = 0.05
) -> tuple[float, float, float, float, float, float]:
    rate_t = x_t / n_t
    rate_c = x_c / n_c
    ate = rate_t - rate_c
    se = math.sqrt(rate_t * (1 - rate_t) / n_t + rate_c * (1 - rate_c) / n_c)
    z_crit = stats.norm.ppf(1 - alpha / 2)
    pooled = (x_t + x_c) / (n_t + n_c)
    se_pooled = math.sqrt(pooled * (1 - pooled) * (1 / n_t + 1 / n_c))
    if se_pooled == 0:
        z, p_value = 0.0, 1.0
    else:
        z = ate / se_pooled
        p_value = float(2 * stats.norm.sf(abs(z)))
    return ate, se, ate - z_crit * se, ate + z_crit * se, z, p_value


def average_treatment_effect(
    df: pd.DataFrame,
    outcome: str = "conversion",
    alpha: float = 0.05,
    data_source: str | None = None,
) -> AteResult:
    check_alpha(alpha)
    treated, control = split_arms(df, outcome)
    n_t, n_c = len(treated), len(control)
    x_t, x_c = int(treated.sum()), int(control.sum())
    ate, se, low, high, z, p_value = ate_from_counts(n_t, x_t, n_c, x_c, alpha)
    return AteResult(
        outcome=outcome,
        data_source=resolve_source(df, data_source),
        n_treated=n_t,
        n_control=n_c,
        rate_treated=x_t / n_t,
        rate_control=x_c / n_c,
        ate=ate,
        std_error=se,
        ci_low=low,
        ci_high=high,
        confidence_level=1 - alpha,
        z_statistic=z,
        p_value=p_value,
    )


def relative_uplift(
    df: pd.DataFrame,
    outcome: str = "conversion",
    alpha: float = 0.05,
    data_source: str | None = None,
) -> RelativeUpliftResult:
    check_alpha(alpha)
    treated, control = split_arms(df, outcome)
    n_t, n_c = len(treated), len(control)
    x_t, x_c = int(treated.sum()), int(control.sum())
    rate_t, rate_c = x_t / n_t, x_c / n_c
    source = resolve_source(df, data_source)
    if rate_c == 0:
        return RelativeUpliftResult(outcome, source, rate_t, rate_c, None, None, None, 1 - alpha)
    lift = rate_t / rate_c - 1
    if rate_t == 0:
        return RelativeUpliftResult(outcome, source, rate_t, rate_c, lift, None, None, 1 - alpha)
    se_log = math.sqrt((1 - rate_t) / (n_t * rate_t) + (1 - rate_c) / (n_c * rate_c))
    z_crit = stats.norm.ppf(1 - alpha / 2)
    ratio = rate_t / rate_c
    low = ratio * math.exp(-z_crit * se_log) - 1
    high = ratio * math.exp(z_crit * se_log) - 1
    return RelativeUpliftResult(outcome, source, rate_t, rate_c, lift, low, high, 1 - alpha)
