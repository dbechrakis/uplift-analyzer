from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core import sequential_study as study
from core.data import FEATURES


@pytest.fixture
def small_study(monkeypatch):
    monkeypatch.setattr(study, "N_ARM", 400)
    monkeypatch.setattr(study, "N_BLOCKS", 20)
    monkeypatch.setattr(study, "LOOK_COUNTS", (5, 20))
    rng = np.random.default_rng(0)
    n = 6_000
    frame = pd.DataFrame(rng.normal(size=(n, 12)), columns=FEATURES)
    logit = -2.5 + 0.8 * frame["f0"] - 0.5 * frame["f1"]
    frame["visit"] = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(np.int8)
    frame["conversion"] = (rng.random(n) < 0.02).astype(np.int8)
    frame["treatment"] = 0
    return frame


def test_wilson_interval_contains_the_estimate():
    low, high = study.wilson_interval(50, 1_000)
    assert low < 0.05 < high
    assert low >= 0 and high <= 1


def test_scenario_rows_are_complete_and_labelled(small_study):
    covariates = study.fit_covariates(small_study)
    seeds = np.random.SeedSequence(42).spawn(60)
    scenario = study.run_scenario(small_study, covariates, seeds, 0.0, "test")
    assert set(scenario) == {(o, a) for o in study.OUTCOMES for a in study.ADJUSTMENTS}
    diff, se = scenario[("visit", "none")]
    assert diff.shape == (60, 20)
    assert np.all(se > 0)
    tau2 = study.mixing_variance_for(0.1, 800, 0.5)
    row = study.summarise(diff, se, "peek", 5, "visit", "none", 0.0, 0.1, tau2)
    assert row.data_source == "criteo"
    assert row.n_experiments == 60
    assert 0 <= row.rejection_ci_low <= row.rejection_rate <= row.rejection_ci_high <= 1
    lifted = study.summarise(diff, se, "obf", 20, "visit", "none", 0.05, 0.1, tau2)
    assert lifted.data_source == "constructed"
    assert study.variance_saving(scenario, "visit") > 0


def test_cuped_saving_is_positive_when_features_predict_the_outcome(small_study):
    covariates = study.fit_covariates(small_study)
    seeds = np.random.SeedSequence(1).spawn(40)
    scenario = study.run_scenario(small_study, covariates, seeds, 0.0, "test")
    assert study.variance_saving(scenario, "visit") > 0.02


def fabricated_rows(overrides: dict | None = None) -> list[dict]:
    overrides = overrides or {}
    rows = []
    for rule in study.RULES:
        for k in study.LOOK_COUNTS:
            for adjustment in study.ADJUSTMENTS:
                for lift in (0.0, *study.LIFTS):
                    row = {
                        "rule": rule,
                        "n_looks": k,
                        "adjustment": adjustment,
                        "injected_lift": lift,
                        "outcome": "visit",
                        "rejection_rate": 0.05 if lift == 0 else 0.5,
                        "share_stopped_early": 0.6,
                        "mean_stop_n_per_arm": 30_000.0,
                        "mean_lift_early_winners": 0.08,
                    }
                    row.update(overrides.get((rule, k, adjustment, lift), {}))
                    rows.append(row)
    return rows


def test_grading_passes_and_fails_by_its_own_rule():
    graded = study.grade_expectations(fabricated_rows(), 0.2, 0.195)
    assert graded["Q1"]["passed"] and graded["Q3"]["passed"] and graded["Q8"]["passed"]
    assert graded["Q2"]["passed"] is False
    assert graded["Q5"]["passed"] is False
    assert graded["Q9"]["passed"] is None
    broken = fabricated_rows({("fixed", 5, "none", 0.0): {"rejection_rate": 0.07}})
    assert study.grade_expectations(broken, 0.3, 0.195)["Q1"]["passed"] is False
    assert study.grade_expectations(broken, 0.3, 0.195)["Q8"]["passed"] is False
