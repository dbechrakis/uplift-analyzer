from __future__ import annotations

import pandas as pd
from statsmodels.stats.multitest import multipletests

from core._common import check_alpha, check_outcome, resolve_source
from core.data import FEATURES
from core.metrics import ate_from_counts
from core.results import SegmentResult, SegmentRow


def uplift_by_segment(
    df: pd.DataFrame,
    feature: str,
    n_bins: int = 5,
    outcome: str = "conversion",
    alpha: float = 0.05,
    data_source: str | None = None,
) -> SegmentResult:
    check_alpha(alpha)
    check_outcome(outcome)
    if feature not in FEATURES:
        raise ValueError(f"feature must be one of {FEATURES}")
    if n_bins < 2:
        raise ValueError("n_bins must be at least 2")
    bins = pd.qcut(df[feature], q=n_bins, duplicates="drop")
    rows: list[dict] = []
    for interval, group in df.groupby(bins, observed=True):
        treated = group.loc[group["treatment"] == 1, outcome]
        control = group.loc[group["treatment"] == 0, outcome]
        row = {
            "label": f"({interval.left:.4g}, {interval.right:.4g}]",
            "lower": float(interval.left),
            "upper": float(interval.right),
            "n": len(group),
            "n_treated": len(treated),
            "n_control": len(control),
            "rate_treated": None,
            "rate_control": None,
            "ate": None,
            "ci_low": None,
            "ci_high": None,
            "p_value": None,
        }
        if len(treated) and len(control):
            ate, _, low, high, _, p_value = ate_from_counts(
                len(treated), int(treated.sum()), len(control), int(control.sum()), alpha
            )
            row.update(
                rate_treated=float(treated.mean()),
                rate_control=float(control.mean()),
                ate=ate,
                ci_low=low,
                ci_high=high,
                p_value=p_value,
            )
        rows.append(row)
    testable = [i for i, r in enumerate(rows) if r["p_value"] is not None]
    adjusted: dict[int, float] = {}
    if testable:
        _, p_adj, _, _ = multipletests([rows[i]["p_value"] for i in testable], method="fdr_bh")
        adjusted = dict(zip(testable, (float(p) for p in p_adj), strict=True))
    segments = [
        SegmentRow(
            **r,
            p_value_adjusted=adjusted.get(i),
            significant=bool(i in adjusted and adjusted[i] < alpha),
        )
        for i, r in enumerate(rows)
    ]
    return SegmentResult(
        feature=feature,
        outcome=outcome,
        data_source=resolve_source(df, data_source),
        n_bins_requested=n_bins,
        confidence_level=1 - alpha,
        adjustment="Benjamini-Hochberg (FDR)",
        segments=segments,
    )
