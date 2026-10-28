from __future__ import annotations

import argparse
import gzip
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from core import data as data_module
from core.cace import complier_effect
from core.data import DATASET_ROWS, DATASET_SHA256, DTYPES, FEATURES, raw_path, sha256_of
from core.metrics import ate_from_counts, average_treatment_effect, relative_uplift
from core.propensity import propensity_weighted_effect
from core.segments import uplift_by_segment
from core.srm import sample_ratio_check

ADJUST_CHUNK = 1_000_000
CARD_FIGURES = {"treatment_ratio": 0.85, "visit_rate": 0.046992, "conversion_rate": 0.00292}


def load_full(directory: Path | None = None) -> pd.DataFrame:
    path = raw_path(directory)
    parts = []
    with gzip.open(path, "rt") as handle:
        for chunk in pd.read_csv(handle, dtype=DTYPES, chunksize=data_module.CHUNK_ROWS):
            parts.append(chunk)
    full = pd.concat(parts, ignore_index=True)
    full.attrs["data_source"] = "criteo"
    return full


def full_adjusted_effects(full: pd.DataFrame, outcomes: tuple[str, ...]) -> dict[str, dict]:
    means = full[FEATURES].mean().to_numpy(dtype=np.float64)
    k = 2 + 2 * len(FEATURES)
    n = len(full)
    gram = np.zeros((k, k))
    xty = {o: np.zeros(k) for o in outcomes}
    chunk = ADJUST_CHUNK

    def design(block: pd.DataFrame) -> np.ndarray:
        x = block[FEATURES].to_numpy(dtype=np.float64) - means
        t = block["treatment"].to_numpy(dtype=np.float64)
        return np.column_stack([np.ones(len(block)), t, x, x * t[:, None]])

    for start in range(0, n, chunk):
        block = full.iloc[start : start + chunk]
        d = design(block)
        gram += d.T @ d
        for o in outcomes:
            xty[o] += d.T @ block[o].to_numpy(dtype=np.float64)
    gram_inv = np.linalg.inv(gram)
    beta = {o: gram_inv @ xty[o] for o in outcomes}
    meat = {o: np.zeros((k, k)) for o in outcomes}
    for start in range(0, n, chunk):
        block = full.iloc[start : start + chunk]
        d = design(block)
        for o in outcomes:
            resid = block[o].to_numpy(dtype=np.float64) - d @ beta[o]
            weighted = d * resid[:, None]
            meat[o] += weighted.T @ weighted
    result = {}
    for o in outcomes:
        cov = gram_inv @ meat[o] @ gram_inv * (n / (n - k))
        result[o] = {"adjusted_ate": float(beta[o][1]), "std_error": float(np.sqrt(cov[1, 1]))}
    return result


PROPENSITY_ROWS = 4_000_000


def full_data_propensity(full: pd.DataFrame, outcomes: tuple[str, ...], seed: int = 7) -> dict:
    rng = np.random.default_rng(seed)
    rows = min(PROPENSITY_ROWS, len(full))
    subset = full.iloc[rng.permutation(len(full))[:rows]].reset_index(drop=True)
    subset.attrs["data_source"] = "criteo"
    return {o: propensity_weighted_effect(subset, o, seed=seed).to_dict() for o in outcomes}


def full_data_checks(full: pd.DataFrame) -> dict:
    treated = full[full["treatment"] == 1]
    control = full[full["treatment"] == 0]
    card = {
        "treatment_ratio": float(full["treatment"].mean()),
        "visit_rate": float(full["visit"].mean()),
        "conversion_rate": float(full["conversion"].mean()),
    }
    smd = {}
    for f in FEATURES:
        a, b = treated[f].to_numpy(np.float64), control[f].to_numpy(np.float64)
        smd[f] = float((a.mean() - b.mean()) / np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2))
    unadjusted = {
        o: average_treatment_effect(full, o).to_dict() for o in ("conversion", "visit", "exposure")
    }
    adjusted = full_adjusted_effects(full, ("conversion", "visit"))
    cace = {o: complier_effect(full, o).to_dict() for o in ("conversion", "visit")}
    propensity = full_data_propensity(full, ("conversion", "visit"))
    return {
        "rows": len(full),
        "card_figures": CARD_FIGURES,
        "computed_figures": card,
        "control_exposure_rate": float(control["exposure"].mean()),
        "smd": smd,
        "max_abs_smd": max(abs(v) for v in smd.values()),
        "unadjusted": unadjusted,
        "adjusted": adjusted,
        "propensity": propensity,
        "cace": cace,
        "relative_uplift": {o: relative_uplift(full, o).to_dict() for o in ("conversion", "visit")},
        "sample_ratio": sample_ratio_check(full).to_dict(),
    }


def sample_coverage(full: pd.DataFrame, truth: dict, draws: int, size: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    arrays = {c: full[c].to_numpy() for c in ("treatment", "conversion", "visit")}
    hits = {"conversion": 0, "visit": 0}
    srm_flags = 0
    estimates = {"conversion": [], "visit": []}
    for _ in range(draws):
        idx = rng.choice(len(full), size=size, replace=False)
        t = arrays["treatment"][idx]
        for o in hits:
            y = arrays[o][idx]
            n_t, n_c = int(t.sum()), int((1 - t).sum())
            ate, _, low, high, _, _ = ate_from_counts(
                n_t, int(y[t == 1].sum()), n_c, int(y[t == 0].sum())
            )
            estimates[o].append(ate)
            hits[o] += low <= truth[o] <= high
        n_t = int(t.sum())
        srm = sample_ratio_check(pd.DataFrame({"treatment": t}))
        srm_flags += srm.mismatch
    return {
        "draws": draws,
        "sample_size": size,
        "ci_coverage_of_full_data_ate": {o: hits[o] / draws for o in hits},
        "srm_false_alarm_rate": srm_flags / draws,
        "estimate_sd": {o: float(np.std(estimates[o], ddof=1)) for o in estimates},
    }


def simulate_code_correctness(seed: int = 7, runs: int = 2000, fdr_runs: int = 300) -> dict:
    rng = np.random.default_rng(seed)
    n_t, n_c, p_c, effect = 4000, 1000, 0.05, 0.02
    ate_hits = rel_hits = type1 = 0
    for _ in range(runs):
        x_t = rng.binomial(n_t, p_c + effect)
        x_c = rng.binomial(n_c, p_c)
        ate, _, low, high, _, _ = ate_from_counts(n_t, x_t, n_c, x_c)
        ate_hits += low <= effect <= high
        frame = pd.DataFrame(
            {
                "treatment": [1] * n_t + [0] * n_c,
                "conversion": [1] * x_t + [0] * (n_t - x_t) + [1] * x_c + [0] * (n_c - x_c),
            }
        )
        for col in ("visit", "exposure", *FEATURES):
            frame[col] = 0
        rel = relative_uplift(frame)
        rel_true = effect / p_c
        rel_hits += rel.ci_low <= rel_true <= rel.ci_high
        null_t = rng.binomial(n_t, p_c)
        null_c = rng.binomial(n_c, p_c)
        type1 += ate_from_counts(n_t, null_t, n_c, null_c)[5] < 0.05
    false_discoveries, discoveries = 0, 0
    for _ in range(fdr_runs):
        n = 20_000
        frame = pd.DataFrame(
            {
                "f0": rng.normal(size=n),
                "treatment": (rng.random(n) < 0.5).astype(int),
                "conversion": (rng.random(n) < 0.1).astype(int),
            }
        )
        for col in ("visit", "exposure", *FEATURES[1:]):
            frame[col] = 0
        segs = uplift_by_segment(frame, "f0", n_bins=10)
        found = sum(s.significant for s in segs.segments)
        discoveries += found
        false_discoveries += found > 0
    return {
        "runs": runs,
        "ate_ci_coverage": ate_hits / runs,
        "relative_uplift_ci_coverage": rel_hits / runs,
        "type_i_error_at_5_percent": type1 / runs,
        "fdr_null_runs": fdr_runs,
        "familywise_false_positive_rate_bh_10_null_segments": false_discoveries / fdr_runs,
        "mean_false_discoveries_per_run": discoveries / fdr_runs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Full-data validation record")
    parser.add_argument("--output", default="outputs/validation.json")
    parser.add_argument("--draws", type=int, default=100)
    args = parser.parse_args()
    started = time.time()
    path = raw_path()
    record: dict = {
        "sha256": sha256_of(path),
        "sha256_expected": DATASET_SHA256,
        "rows_expected": DATASET_ROWS,
    }
    full = load_full()
    record["full_data"] = full_data_checks(full)
    truth = {o: record["full_data"]["unadjusted"][o]["ate"] for o in ("conversion", "visit")}
    record["sampling"] = sample_coverage(full, truth, args.draws, 500_000, seed=2026)
    record["code_simulations"] = simulate_code_correctness()
    record["runtime_seconds"] = round(time.time() - started)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(record, indent=2))
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
