from __future__ import annotations

import pandas as pd
from scipy import stats

from core._common import check_alpha, resolve_source
from core.results import SrmResult

CRITEO_DESIGN_TREATMENT_SHARE = 0.85


def sample_ratio_check(
    df: pd.DataFrame,
    expected_share_treated: float = CRITEO_DESIGN_TREATMENT_SHARE,
    alpha: float = 0.001,
    data_source: str | None = None,
) -> SrmResult:
    check_alpha(alpha)
    if not 0 < expected_share_treated < 1:
        raise ValueError("expected_share_treated must be between 0 and 1")
    n_t = int((df["treatment"] == 1).sum())
    n_c = int((df["treatment"] == 0).sum())
    total = n_t + n_c
    if total == 0:
        raise ValueError("No rows to check")
    expected = [total * expected_share_treated, total * (1 - expected_share_treated)]
    chi2, p_value = stats.chisquare([n_t, n_c], f_exp=expected)
    return SrmResult(
        data_source=resolve_source(df, data_source),
        n_treated=n_t,
        n_control=n_c,
        observed_share_treated=n_t / total,
        expected_share_treated=expected_share_treated,
        chi2_statistic=float(chi2),
        p_value=float(p_value),
        alpha=alpha,
        mismatch=bool(p_value < alpha),
    )
