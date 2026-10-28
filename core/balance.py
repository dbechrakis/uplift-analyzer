from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from statsmodels.stats.multitest import multipletests

from core._common import check_alpha, resolve_source
from core.data import FEATURES
from core.results import BalanceResult, BalanceRow

SMD_THRESHOLD = 0.1


def covariate_balance(
    df: pd.DataFrame,
    features: list[str] | None = None,
    alpha: float = 0.05,
    seed: int = 42,
    max_iter: int = 100,
    n_bootstrap: int = 100,
    data_source: str | None = None,
) -> BalanceResult:
    check_alpha(alpha)
    features = list(features or FEATURES)
    unknown = [f for f in features if f not in FEATURES]
    if unknown:
        raise ValueError(f"Unknown features: {unknown}")
    treated = df[df["treatment"] == 1]
    control = df[df["treatment"] == 0]
    if treated.empty or control.empty:
        raise ValueError("Both treatment and control rows are required")
    raw: list[tuple[str, float, float, float, float]] = []
    for f in features:
        a, b = treated[f].to_numpy(float), control[f].to_numpy(float)
        pooled_sd = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
        smd = float((a.mean() - b.mean()) / pooled_sd) if pooled_sd > 0 else 0.0
        p_value = float(stats.ttest_ind(a, b, equal_var=False).pvalue)
        raw.append((f, float(a.mean()), float(b.mean()), smd, p_value))
    adjusted = multipletests([r[4] for r in raw], method="fdr_bh")[1]
    rows = [
        BalanceRow(f, mt, mc, smd, p, float(pa))
        for (f, mt, mc, smd, p), pa in zip(raw, adjusted, strict=True)
    ]
    train, test = train_test_split(df, test_size=0.3, random_state=seed, stratify=df["treatment"])
    model = HistGradientBoostingClassifier(max_iter=max_iter, random_state=seed)
    model.fit(train[features], train["treatment"])
    prob = model.predict_proba(test[features])[:, 1]
    label = test["treatment"].to_numpy()
    auc = float(roc_auc_score(label, prob))
    rng = np.random.default_rng(seed)
    boots = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        idx = rng.integers(0, len(label), len(label))
        boots[b] = roc_auc_score(label[idx], prob[idx])
    low, high = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return BalanceResult(
        data_source=resolve_source(df, data_source),
        n_treated=len(treated),
        n_control=len(control),
        max_abs_smd=max(abs(r.smd) for r in rows),
        smd_threshold=SMD_THRESHOLD,
        features_over_threshold=sum(abs(r.smd) > SMD_THRESHOLD for r in rows),
        auc=auc,
        auc_ci_low=float(low),
        auc_ci_high=float(high),
        n_bootstrap=n_bootstrap,
        confidence_level=1 - alpha,
        joint_imbalance_detected=bool(low > 0.5),
        rows=rows,
    )
