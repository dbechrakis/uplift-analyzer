from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split

from core._common import check_alpha, check_outcome, resolve_source
from core.data import FEATURES
from core.results import QiniResult

MODEL_NAME = "T-learner (two HistGradientBoostingClassifier models)"


def qini_arrays(y: np.ndarray, t: np.ndarray, score: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(-score, kind="stable")
    y_s = y[order].astype(float)
    t_s = t[order].astype(float)
    cum_t = np.cumsum(t_s)
    cum_c = np.cumsum(1 - t_s)
    cum_yt = np.cumsum(y_s * t_s)
    cum_yc = np.cumsum(y_s * (1 - t_s))
    ratio = np.divide(cum_t, cum_c, out=np.zeros_like(cum_t), where=cum_c > 0)
    qini = np.concatenate([[0.0], cum_yt - cum_yc * ratio])
    x = np.arange(len(qini)) / (len(qini) - 1)
    return x, qini


def auuc_from_curve(x: np.ndarray, qini: np.ndarray) -> tuple[float, float, float]:
    area = float(np.trapezoid(qini, x))
    random_area = float(0.5 * qini[-1])
    return area, random_area, area - random_area


def qini_coefficient(y: np.ndarray, t: np.ndarray, score: np.ndarray) -> float:
    x, qini = qini_arrays(y, t, score)
    return auuc_from_curve(x, qini)[2]


def bootstrap_qini_ci(
    y: np.ndarray,
    t: np.ndarray,
    score: np.ndarray,
    n_bootstrap: int,
    alpha: float,
    seed: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(y)
    values = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        values[b] = qini_coefficient(y[idx], t[idx], score[idx])
    low, high = np.quantile(values, [alpha / 2, 1 - alpha / 2])
    return float(low), float(high)


def downsample_curve(
    x: np.ndarray, qini: np.ndarray, n_points: int
) -> tuple[list[float], list[float], list[float]]:
    idx = np.unique(np.linspace(0, len(x) - 1, n_points).round().astype(int))
    xs = x[idx]
    return xs.tolist(), qini[idx].tolist(), (qini[-1] * xs).tolist()


def fit_t_learner(
    train: pd.DataFrame,
    outcome: str,
    features: list[str],
    seed: int,
    max_iter: int,
) -> tuple[HistGradientBoostingClassifier, HistGradientBoostingClassifier]:
    def make() -> HistGradientBoostingClassifier:
        return HistGradientBoostingClassifier(max_iter=max_iter, random_state=seed)

    treated = train[train["treatment"] == 1]
    control = train[train["treatment"] == 0]
    for name, arm in (("treated", treated), ("control", control)):
        if arm[outcome].nunique() < 2:
            raise ValueError(f"The {name} training arm needs both outcome classes")
    model_t = make().fit(treated[features], treated[outcome])
    model_c = make().fit(control[features], control[outcome])
    return model_t, model_c


def split_holdout(
    df: pd.DataFrame, outcome: str, test_size: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    strata = df["treatment"].astype(int) * 2 + df[outcome].astype(int)
    return train_test_split(df, test_size=test_size, random_state=seed, stratify=strata)


def t_learner_scores(
    train: pd.DataFrame,
    test: pd.DataFrame,
    outcome: str,
    features: list[str],
    seed: int,
    max_iter: int,
) -> np.ndarray:
    model_t, model_c = fit_t_learner(train, outcome, features, seed, max_iter)
    return model_t.predict_proba(test[features])[:, 1] - model_c.predict_proba(test[features])[:, 1]


def qini_analysis(
    df: pd.DataFrame,
    outcome: str = "conversion",
    features: list[str] | None = None,
    test_size: float = 0.3,
    seed: int = 42,
    max_iter: int = 100,
    n_bootstrap: int = 200,
    n_curve_points: int = 100,
    alpha: float = 0.05,
    data_source: str | None = None,
) -> QiniResult:
    check_alpha(alpha)
    check_outcome(outcome)
    features = list(features or FEATURES)
    unknown = [f for f in features if f not in FEATURES]
    if unknown:
        raise ValueError(f"Unknown features: {unknown}")
    train, test = split_holdout(df, outcome, test_size, seed)
    score = t_learner_scores(train, test, outcome, features, seed, max_iter)
    y = test[outcome].to_numpy()
    t = test["treatment"].to_numpy()
    x, qini = qini_arrays(y, t, score)
    auuc, auuc_random, coefficient = auuc_from_curve(x, qini)
    low, high = bootstrap_qini_ci(y, t, score, n_bootstrap, alpha, seed)
    cx, cq, cr = downsample_curve(x, qini, n_curve_points)
    return QiniResult(
        outcome=outcome,
        data_source=resolve_source(df, data_source),
        model=MODEL_NAME,
        features=features,
        n_train=len(train),
        n_test=len(test),
        auuc=auuc,
        auuc_random=auuc_random,
        qini_coefficient=coefficient,
        qini_ci_low=low,
        qini_ci_high=high,
        n_bootstrap=n_bootstrap,
        confidence_level=1 - alpha,
        curve_x=cx,
        curve_qini=cq,
        curve_random=cr,
    )
