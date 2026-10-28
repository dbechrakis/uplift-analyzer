from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats

from core._common import check_alpha, check_outcome, resolve_source
from core.metrics import ate_from_counts
from core.results import CaceResult


def complier_effect(
    df: pd.DataFrame,
    outcome: str = "conversion",
    alpha: float = 0.05,
    data_source: str | None = None,
) -> CaceResult:
    check_alpha(alpha)
    check_outcome(outcome)
    if outcome == "exposure":
        raise ValueError("exposure is the first stage; choose conversion or visit as the outcome")
    treated = df[df["treatment"] == 1]
    control = df[df["treatment"] == 0]
    if treated.empty or control.empty:
        raise ValueError("Both treatment and control rows are required")
    n1, n0 = len(treated), len(control)
    y1, y0 = treated[outcome].to_numpy(float), control[outcome].to_numpy(float)
    d1, d0 = treated["exposure"].to_numpy(float), control["exposure"].to_numpy(float)
    numerator = y1.mean() - y0.mean()
    denominator = d1.mean() - d0.mean()
    if denominator <= 0:
        raise ValueError("Exposure must be higher in the treated group (weak first stage)")
    var_n = y1.var(ddof=1) / n1 + y0.var(ddof=1) / n0
    var_d = d1.var(ddof=1) / n1 + d0.var(ddof=1) / n0
    cov_nd = np.cov(y1, d1, ddof=1)[0, 1] / n1 + np.cov(y0, d0, ddof=1)[0, 1] / n0
    ratio = numerator / denominator
    variance = (var_n - 2 * ratio * cov_nd + ratio**2 * var_d) / denominator**2
    se = math.sqrt(max(variance, 0.0))
    z_crit = stats.norm.ppf(1 - alpha / 2)
    _, _, itt_low, itt_high, _, _ = ate_from_counts(n1, int(y1.sum()), n0, int(y0.sum()), alpha)
    one_sided = bool(d0.sum() == 0)
    complier_control = complier_treated = never_taker = None
    if one_sided:
        pi = d1.mean()
        exposed = treated[treated["exposure"] == 1][outcome]
        unexposed = treated[treated["exposure"] == 0][outcome]
        complier_treated = float(exposed.mean())
        if len(unexposed):
            never_taker = float(unexposed.mean())
            complier_control = float((y0.mean() - (1 - pi) * never_taker) / pi)
    naive = float(treated.loc[treated["exposure"] == 1, outcome].mean() - y0.mean())
    return CaceResult(
        outcome=outcome,
        data_source=resolve_source(df, data_source),
        n_treated=n1,
        n_control=n0,
        exposure_rate_treated=float(d1.mean()),
        exposure_rate_control=float(d0.mean()),
        itt=float(numerator),
        itt_ci_low=itt_low,
        itt_ci_high=itt_high,
        cace=float(ratio),
        cace_std_error=se,
        cace_ci_low=float(ratio - z_crit * se),
        cace_ci_high=float(ratio + z_crit * se),
        confidence_level=1 - alpha,
        one_sided=one_sided,
        complier_rate_control=complier_control,
        complier_rate_treated=complier_treated,
        never_taker_rate=never_taker,
        naive_exposed_vs_control=naive,
    )
