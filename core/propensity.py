from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import KFold

from core._common import check_alpha, check_outcome, resolve_source
from core.data import FEATURES
from core.results import PropensityResult


def cross_fitted_propensity(
    df: pd.DataFrame, features: list[str], n_folds: int, seed: int, max_iter: int
) -> np.ndarray:
    scores = np.empty(len(df))
    x = df[features]
    t = df["treatment"].to_numpy()
    folds = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for train_idx, test_idx in folds.split(x):
        model = HistGradientBoostingClassifier(max_iter=max_iter, random_state=seed)
        model.fit(x.iloc[train_idx], t[train_idx])
        scores[test_idx] = model.predict_proba(x.iloc[test_idx])[:, 1]
    return scores


def propensity_weighted_effect(
    df: pd.DataFrame,
    outcome: str = "conversion",
    features: list[str] | None = None,
    n_folds: int = 2,
    clip: tuple[float, float] = (0.01, 0.99),
    alpha: float = 0.05,
    seed: int = 42,
    max_iter: int = 100,
    data_source: str | None = None,
) -> PropensityResult:
    check_alpha(alpha)
    check_outcome(outcome)
    features = list(features or FEATURES)
    unknown = [f for f in features if f not in FEATURES]
    if unknown:
        raise ValueError(f"Unknown features: {unknown}")
    if n_folds < 2 or not 0 < clip[0] < clip[1] < 1:
        raise ValueError("n_folds must be at least 2 and clip must satisfy 0 < low < high < 1")
    t = df["treatment"].to_numpy(dtype=float)
    if t.min() == t.max():
        raise ValueError("Both treatment and control rows are required")
    y = df[outcome].to_numpy(dtype=float)
    raw_score = cross_fitted_propensity(df, features, n_folds, seed, max_iter)
    e = np.clip(raw_score, clip[0], clip[1])
    w1 = t / e
    w0 = (1 - t) / (1 - e)
    mu1 = (w1 * y).sum() / w1.sum()
    mu0 = (w0 * y).sum() / w0.sum()
    psi = w1 * (y - mu1) / w1.mean() - w0 * (y - mu0) / w0.mean()
    se = float(psi.std(ddof=1) / np.sqrt(len(y)))
    effect = float(mu1 - mu0)
    z_crit = stats.norm.ppf(1 - alpha / 2)
    return PropensityResult(
        outcome=outcome,
        data_source=resolve_source(df, data_source),
        n=len(y),
        n_folds=n_folds,
        clip_low=clip[0],
        clip_high=clip[1],
        share_clipped=float(((raw_score < clip[0]) | (raw_score > clip[1])).mean()),
        propensity_min=float(raw_score.min()),
        propensity_max=float(raw_score.max()),
        ipw_ate=effect,
        std_error=se,
        ci_low=effect - z_crit * se,
        ci_high=effect + z_crit * se,
        confidence_level=1 - alpha,
        unadjusted_ate=float(y[t == 1].mean() - y[t == 0].mean()),
        effective_n_treated=float(w1.sum() ** 2 / (w1**2).sum()),
        effective_n_control=float(w0.sum() ** 2 / (w0**2).sum()),
    )
