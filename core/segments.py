from __future__ import annotations

import pandas as pd
from statsmodels.stats.multitest import multipletests

from core._common import check_alpha, check_outcome, resolve_source
from core.data import FEATURES
from core.metrics import ate_from_counts
from core.results import SegmentResult, SegmentRow


def _partition(series: pd.Series, n_bins: int) -> list[tuple[str, float, float, bool, pd.Index]]:
    shares = series.value_counts(normalize=True)
    atoms = [v for v, share in shares.items() if share >= 1.0 / n_bins][: n_bins - 1]
    parts: list[tuple[str, float, float, bool, pd.Index]] = []
    for value in atoms:
        mask = series == value
        label = f"= {float(value):.4g} ({shares[value]:.0%} of users)"
        parts.append((label, float(value), float(value), True, series.index[mask]))
    rest = series[~series.isin(atoms)]
    if len(rest):
        bins = pd.qcut(rest, q=max(n_bins - len(atoms), 1), duplicates="drop")
        for interval, chunk in rest.groupby(bins, observed=True):
            label = f"({interval.left:.4g}, {interval.right:.4g}]"
            parts.append((label, float(interval.left), float(interval.right), False, chunk.index))
    return sorted(parts, key=lambda part: (part[1] + part[2]) / 2)


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
    parts = _partition(df[feature], n_bins)
    rows: list[dict] = []
    for label, lower, upper, is_atom, index in parts:
        group = df.loc[index]
        treated = group.loc[group["treatment"] == 1, outcome]
        control = group.loc[group["treatment"] == 0, outcome]
        row = {
            "label": label,
            "lower": lower,
            "upper": upper,
            "n": len(group),
            "n_treated": len(treated),
            "n_control": len(control),
            "rate_treated": None,
            "rate_control": None,
            "ate": None,
            "ci_low": None,
            "ci_high": None,
            "p_value": None,
            "is_atom": is_atom,
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
        n_bins_realized=len(segments),
    )
