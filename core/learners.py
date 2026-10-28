from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from core._common import check_alpha, check_outcome, resolve_source
from core.data import FEATURES
from core.results import LearnerComparisonResult, LearnerRow
from core.uplift_model import qini_coefficient, split_holdout, t_learner_scores

LEARNER_NAMES = ("T-learner", "S-learner", "X-learner", "Transformed outcome")


def _classifier(seed: int, max_iter: int) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(max_iter=max_iter, random_state=seed)


def _regressor(seed: int, max_iter: int) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(max_iter=max_iter, random_state=seed)


def s_learner_scores(train, test, outcome, features, seed, max_iter) -> np.ndarray:
    columns = [*features, "treatment"]
    model = _classifier(seed, max_iter).fit(train[columns], train[outcome])
    on = test[columns].assign(treatment=1)
    off = test[columns].assign(treatment=0)
    return model.predict_proba(on)[:, 1] - model.predict_proba(off)[:, 1]


def x_learner_scores(train, test, outcome, features, seed, max_iter) -> np.ndarray:
    treated = train[train["treatment"] == 1]
    control = train[train["treatment"] == 0]
    mu1 = _classifier(seed, max_iter).fit(treated[features], treated[outcome])
    mu0 = _classifier(seed, max_iter).fit(control[features], control[outcome])
    effect_treated = treated[outcome].to_numpy() - mu0.predict_proba(treated[features])[:, 1]
    effect_control = mu1.predict_proba(control[features])[:, 1] - control[outcome].to_numpy()
    tau1 = _regressor(seed, max_iter).fit(treated[features], effect_treated)
    tau0 = _regressor(seed, max_iter).fit(control[features], effect_control)
    propensity = float(train["treatment"].mean())
    return propensity * tau0.predict(test[features]) + (1 - propensity) * tau1.predict(
        test[features]
    )


def transformed_outcome_scores(train, test, outcome, features, seed, max_iter) -> np.ndarray:
    propensity = float(train["treatment"].mean())
    t = train["treatment"].to_numpy(float)
    target = train[outcome].to_numpy(float) * (t - propensity) / (propensity * (1 - propensity))
    return _regressor(seed, max_iter).fit(train[features], target).predict(test[features])


def compare_uplift_learners(
    df: pd.DataFrame,
    outcome: str = "conversion",
    features: list[str] | None = None,
    test_size: float = 0.3,
    seed: int = 42,
    max_iter: int = 100,
    n_bootstrap: int = 100,
    alpha: float = 0.05,
    data_source: str | None = None,
) -> LearnerComparisonResult:
    check_alpha(alpha)
    check_outcome(outcome)
    features = list(features or FEATURES)
    train, test = split_holdout(df, outcome, test_size, seed)
    scores = {
        "T-learner": t_learner_scores(train, test, outcome, features, seed, max_iter),
        "S-learner": s_learner_scores(train, test, outcome, features, seed, max_iter),
        "X-learner": x_learner_scores(train, test, outcome, features, seed, max_iter),
        "Transformed outcome": transformed_outcome_scores(
            train, test, outcome, features, seed, max_iter
        ),
    }
    y, t = test[outcome].to_numpy(), test["treatment"].to_numpy()
    point = {name: qini_coefficient(y, t, s) for name, s in scores.items()}
    rng = np.random.default_rng(seed)
    n = len(y)
    boot = {name: np.empty(n_bootstrap) for name in scores}
    for b in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        for name, s in scores.items():
            boot[name][b] = qini_coefficient(y[idx], t[idx], s[idx])
    bounds = [alpha / 2, 1 - alpha / 2]
    rows = []
    for name in scores:
        low, high = np.quantile(boot[name], bounds)
        diff = boot[name] - boot["T-learner"]
        d_low, d_high = np.quantile(diff, bounds)
        rows.append(
            LearnerRow(
                name=name,
                qini_coefficient=float(point[name]),
                ci_low=float(low),
                ci_high=float(high),
                diff_vs_t_learner=float(point[name] - point["T-learner"]),
                diff_ci_low=float(d_low),
                diff_ci_high=float(d_high),
                differs_from_t_learner=bool(name != "T-learner" and (d_low > 0 or d_high < 0)),
            )
        )
    return LearnerComparisonResult(
        outcome=outcome,
        data_source=resolve_source(df, data_source),
        n_train=len(train),
        n_test=len(test),
        n_bootstrap=n_bootstrap,
        confidence_level=1 - alpha,
        best_by_point_estimate=max(point, key=point.get),
        learners=rows,
    )
