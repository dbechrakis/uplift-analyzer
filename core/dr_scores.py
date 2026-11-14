from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold

from core._common import check_alpha, check_outcome, resolve_source
from core.data import FEATURES, load_criteo_sample
from core.results import DoublyRobustQiniResult
from core.uplift_model import (
    auuc_from_curve,
    bootstrap_qini_ci,
    qini_coefficient,
    split_holdout,
    t_learner_scores,
)

PROPENSITY_CLIP = (0.01, 0.99)
OUTPUT_PATH = Path("outputs") / "dr_qini.json"


def _classifier(seed: int, max_iter: int) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(max_iter=max_iter, random_state=seed)


def combine_scores(
    mu1: np.ndarray, mu0: np.ndarray, propensity: np.ndarray, y: np.ndarray, t: np.ndarray
) -> np.ndarray:
    e = np.clip(propensity, *PROPENSITY_CLIP)
    treated_residual = t * (y - mu1) / e
    control_residual = (1 - t) * (y - mu0) / (1 - e)
    return mu1 - mu0 + treated_residual - control_residual


def cross_fitted_scores(
    df: pd.DataFrame,
    outcome: str,
    features: list[str],
    seed: int,
    max_iter: int,
    folds: int = 5,
) -> np.ndarray:
    check_outcome(outcome)
    y = df[outcome].to_numpy(float)
    t = df["treatment"].to_numpy(float)
    x = df[features]
    strata = df["treatment"].astype(int) * 2 + df[outcome].astype(int)
    scores = np.empty(len(df))
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    for fit_index, score_index in splitter.split(x, strata):
        fit_t = t[fit_index] == 1
        model_1 = _classifier(seed, max_iter).fit(x.iloc[fit_index[fit_t]], y[fit_index[fit_t]])
        model_0 = _classifier(seed, max_iter).fit(x.iloc[fit_index[~fit_t]], y[fit_index[~fit_t]])
        model_e = _classifier(seed, max_iter).fit(x.iloc[fit_index], t[fit_index])
        block = x.iloc[score_index]
        scores[score_index] = combine_scores(
            model_1.predict_proba(block)[:, 1],
            model_0.predict_proba(block)[:, 1],
            model_e.predict_proba(block)[:, 1],
            y[score_index],
            t[score_index],
        )
    return scores


def corrected_qini_arrays(
    scores: np.ndarray, uplift_score: np.ndarray, treated_share: float
) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(-uplift_score, kind="stable")
    qini = np.concatenate([[0.0], np.cumsum(scores[order]) * treated_share])
    x = np.arange(len(qini)) / (len(qini) - 1)
    return x, qini


def corrected_qini_coefficient(
    scores: np.ndarray, uplift_score: np.ndarray, treated_share: float
) -> float:
    x, qini = corrected_qini_arrays(scores, uplift_score, treated_share)
    return auuc_from_curve(x, qini)[2]


def bootstrap_corrected_ci(
    scores: np.ndarray,
    uplift_score: np.ndarray,
    treated_share: float,
    n_bootstrap: int,
    alpha: float,
    seed: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(scores)
    values = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        values[b] = corrected_qini_coefficient(scores[idx], uplift_score[idx], treated_share)
    low, high = np.quantile(values, [alpha / 2, 1 - alpha / 2])
    return float(low), float(high)


def doubly_robust_qini_analysis(
    df: pd.DataFrame,
    outcome: str = "conversion",
    features: list[str] | None = None,
    test_size: float = 0.3,
    seed: int = 42,
    max_iter: int = 100,
    folds: int = 5,
    n_bootstrap: int = 200,
    alpha: float = 0.05,
    data_source: str | None = None,
) -> DoublyRobustQiniResult:
    check_alpha(alpha)
    check_outcome(outcome)
    features = list(features or FEATURES)
    train, test = split_holdout(df, outcome, test_size, seed)
    uplift_score = t_learner_scores(train, test, outcome, features, seed, max_iter)
    y = test[outcome].to_numpy()
    t = test["treatment"].to_numpy()
    treated_share = float(t.mean())
    raw = qini_coefficient(y, t, uplift_score)
    raw_low, raw_high = bootstrap_qini_ci(y, t, uplift_score, n_bootstrap, alpha, seed)
    scores = cross_fitted_scores(test, outcome, features, seed, max_iter, folds)
    corrected = corrected_qini_coefficient(scores, uplift_score, treated_share)
    low, high = bootstrap_corrected_ci(
        scores, uplift_score, treated_share, n_bootstrap, alpha, seed
    )
    return DoublyRobustQiniResult(
        outcome=outcome,
        data_source=resolve_source(df, data_source),
        n_train=len(train),
        n_test=len(test),
        folds=folds,
        qini_raw=raw,
        qini_raw_ci_low=raw_low,
        qini_raw_ci_high=raw_high,
        qini_corrected=corrected,
        qini_corrected_ci_low=low,
        qini_corrected_ci_high=high,
        difference=corrected - raw,
        raw_inside_corrected_interval=bool(low <= raw <= high),
        corrected_inside_raw_interval=bool(raw_low <= corrected <= raw_high),
        n_bootstrap=n_bootstrap,
        confidence_level=1 - alpha,
    )


def main() -> None:
    df = load_criteo_sample()
    results = {
        o: doubly_robust_qini_analysis(df, outcome=o).to_dict() for o in ("conversion", "visit")
    }
    OUTPUT_PATH.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8", newline="\n")
    for outcome, r in results.items():
        print(outcome, round(r["qini_raw"], 2), round(r["qini_corrected"], 2))


if __name__ == "__main__":
    main()
