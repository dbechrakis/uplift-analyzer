from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from core.data import DEFAULT_SEED, FEATURES, load_control_pool, load_criteo_sample
from core.metrics import average_treatment_effect
from core.power import minimum_detectable_effect
from core.replay import (
    cumulative_sums,
    draw_experiment,
    eligible_rows,
    holdout_rows,
    inject_lift,
    look_sizes,
)
from core.results import SequentialRuleResult
from core.sequential import RULES, decision_matrix, difference_and_se, first_true

N_ARM = 50_000
N_BLOCKS = 200
LOOK_COUNTS = (5, 20, 200)
NULL_REPLICATIONS = 5_000
LIFT_REPLICATIONS = 2_000
LIFTS = (0.02, 0.05, 0.10)
OUTCOMES = ("visit", "conversion")
ADJUSTMENTS = ("none", "cuped")
BAND_LOW = 0.044
BAND_HIGH = 0.056
OUTPUT = Path("outputs/sequential.json")


def wilson_interval(successes: int, total: int, z: float = 1.959964) -> tuple[float, float]:
    p = successes / total
    denom = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / denom
    half = z * np.sqrt(p * (1 - p) / total + z**2 / (4 * total**2)) / denom
    return float(center - half), float(center + half)


def fit_covariates(pool: pd.DataFrame) -> dict[str, dict[str, float | np.ndarray]]:
    x = pool[FEATURES].to_numpy(dtype=np.float64)
    holdout = holdout_rows(len(pool))
    fitted: dict[str, dict[str, float | np.ndarray]] = {}
    for outcome in OUTCOMES:
        y = pool[outcome].to_numpy(dtype=np.float64)
        model = LogisticRegression().fit(x[holdout], y[holdout])
        score = model.predict_proba(x)[:, 1]
        held = score[holdout]
        theta = float(np.cov(y[holdout], held)[0, 1] / np.var(held, ddof=1))
        center = float(held.mean())
        fitted[outcome] = {
            "theta": theta,
            "center": center,
            "term": theta * (score - center),
            "baseline": float(y.mean()),
        }
    return fitted


def mixing_variance_for(baseline: float, n_total: int, treated_share: float) -> float:
    effect = minimum_detectable_effect(baseline, n_total, 0.05, 0.8, treated_share)
    return float(effect.absolute_effect**2)


def run_scenario(
    pool: pd.DataFrame,
    covariates: dict[str, dict[str, float | np.ndarray]],
    seeds: list[np.random.SeedSequence],
    lift: float,
    label: str,
) -> dict[tuple[str, str], tuple[np.ndarray, np.ndarray]]:
    eligible = eligible_rows(len(pool))
    looks = look_sizes(N_ARM, N_BLOCKS)
    n = looks.astype(np.float64)
    outcomes = {o: pool[o].to_numpy() for o in OUTCOMES}
    diffs = {(o, a): np.empty((len(seeds), N_BLOCKS)) for o in OUTCOMES for a in ADJUSTMENTS}
    ses = {key: np.empty((len(seeds), N_BLOCKS)) for key in diffs}
    started = time.time()
    for i, seed in enumerate(seeds):
        a_rows, b_rows = draw_experiment(eligible, N_ARM, seed)
        rng = np.random.default_rng(seed.spawn(1)[0])
        for outcome in OUTCOMES:
            ya = outcomes[outcome][a_rows]
            yb = outcomes[outcome][b_rows]
            if lift:
                yb = inject_lift(yb, lift, float(covariates[outcome]["baseline"]), rng)
            term = covariates[outcome]["term"]
            variants = {
                "none": (ya.astype(np.float64), yb.astype(np.float64)),
                "cuped": (ya - term[a_rows], yb - term[b_rows]),
            }
            for adjustment, (va, vb) in variants.items():
                sa, qa = cumulative_sums(va, looks)
                sb, qb = cumulative_sums(vb, looks)
                d, s = difference_and_se(sa, sb, qa, qb, n)
                diffs[(outcome, adjustment)][i] = d
                ses[(outcome, adjustment)][i] = s
        if (i + 1) % 500 == 0:
            print(f"{label}: {i + 1}/{len(seeds)} in {time.time() - started:.0f}s", flush=True)
    return {key: (diffs[key], ses[key]) for key in diffs}


def summarise(
    diff: np.ndarray,
    se: np.ndarray,
    rule: str,
    n_looks: int,
    outcome: str,
    adjustment: str,
    lift: float,
    baseline: float,
    tau2: float,
) -> SequentialRuleResult:
    step = N_BLOCKS // n_looks
    index = np.arange(step - 1, N_BLOCKS, step)
    sub_diff = diff[:, index]
    sub_se = se[:, index]
    sizes = look_sizes(N_ARM, N_BLOCKS)[index]
    information = sizes / N_ARM
    exceeds = decision_matrix(rule, sub_diff, sub_se, information, 0.05, tau2)
    stop = first_true(exceeds)
    rejected = stop >= 0
    safe = np.where(rejected, stop, n_looks - 1)
    stop_n = np.where(rejected, sizes[safe], N_ARM)
    early = rejected & (stop < n_looks - 1)
    at_stop = sub_diff[np.arange(len(stop)), safe]
    winners = early & (at_stop > 0)
    low, high = wilson_interval(int(rejected.sum()), len(stop))
    return SequentialRuleResult(
        rule=rule,
        adjustment=adjustment,
        outcome=outcome,
        data_source="criteo" if lift == 0 else "constructed",
        n_looks=n_looks,
        injected_lift=lift,
        n_experiments=len(stop),
        n_arm=N_ARM,
        rejection_rate=float(rejected.mean()),
        rejection_ci_low=low,
        rejection_ci_high=high,
        mean_stop_n_per_arm=float(stop_n.mean()),
        median_stop_n_per_arm=float(np.median(stop_n)),
        share_stopped_early=float(early.mean()),
        mean_lift_early_winners=float(at_stop[winners].mean() / baseline)
        if winners.any()
        else None,
    )


def variance_saving(
    scenario: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]], outcome: str
) -> float:
    raw = scenario[(outcome, "none")][1][:, -1]
    adjusted = scenario[(outcome, "cuped")][1][:, -1]
    return float(np.mean(1 - (adjusted / raw) ** 2))


def find(rows: list[dict], **keys: object) -> dict:
    for row in rows:
        if all(row[k] == v for k, v in keys.items()):
            return row
    raise KeyError(keys)


def grade_expectations(rows: list[dict], saving: float, reference_saving: float) -> dict:
    def pick(rule: str, k: int, lift: float, adjustment: str = "none") -> dict:
        return find(
            rows, rule=rule, n_looks=k, injected_lift=lift, adjustment=adjustment, outcome="visit"
        )

    def in_band(rate: float) -> bool:
        return BAND_LOW <= rate <= BAND_HIGH

    graded: dict[str, dict] = {}
    fixed = {k: pick("fixed", k, 0.0)["rejection_rate"] for k in LOOK_COUNTS}
    graded["Q1"] = {"passed": all(in_band(v) for v in fixed.values()), "values": fixed}
    peek = {k: pick("peek", k, 0.0)["rejection_rate"] for k in LOOK_COUNTS}
    floors = {5: 0.12, 20: 0.20, 200: 0.30}
    graded["Q2"] = {"passed": all(peek[k] > floors[k] for k in LOOK_COUNTS), "values": peek}
    q3 = {
        f"{rule}_{k}": pick(rule, k, 0.0)["rejection_rate"]
        for rule in ("pocock", "obf")
        for k in (5, 20)
    }
    graded["Q3"] = {"passed": all(in_band(v) for v in q3.values()), "values": q3}
    q4 = {k: pick("msprt", k, 0.0)["rejection_rate"] for k in LOOK_COUNTS}
    graded["Q4"] = {
        "passed": all(v <= BAND_HIGH for v in q4.values()) and q4[200] < 0.05,
        "values": q4,
    }
    bonf_null = {k: pick("bonferroni", k, 0.0)["rejection_rate"] for k in LOOK_COUNTS}
    power_gap = {
        f"{k}_{lift}": (
            pick("bonferroni", k, lift)["rejection_rate"] - pick("obf", k, lift)["rejection_rate"]
        )
        for k in LOOK_COUNTS
        for lift in LIFTS
    }
    graded["Q5"] = {
        "passed": all(v <= BAND_HIGH for v in bonf_null.values())
        and all(v < 0 for v in power_gap.values()),
        "values": {"bonferroni_null": bonf_null, "bonferroni_minus_obf_power": power_gap},
    }
    q6 = {
        k: {
            "share_stopped_early": pick("obf", k, 0.05)["share_stopped_early"],
            "mean_stop_n_per_arm": pick("obf", k, 0.05)["mean_stop_n_per_arm"],
        }
        for k in LOOK_COUNTS
    }
    graded["Q6"] = {
        "passed": all(
            v["share_stopped_early"] > 0.5 and v["mean_stop_n_per_arm"] < N_ARM for v in q6.values()
        ),
        "values": q6,
    }
    q7: dict[str, float | None] = {}
    for rule in ("peek", "pocock", "obf"):
        for k in LOOK_COUNTS:
            q7[f"{rule}_{k}"] = pick(rule, k, 0.05)["mean_lift_early_winners"]
    graded["Q7"] = {
        "passed": all(v is not None and v > 0.05 for v in q7.values()),
        "values": q7,
    }
    graded["Q8"] = {
        "passed": abs(saving - reference_saving) <= 0.03,
        "values": {"saving": saving, "reference": reference_saving},
    }
    q9 = {
        f"{rule}_{k}": {
            "none": pick(rule, k, 0.0, "none")["rejection_rate"],
            "cuped": pick(rule, k, 0.0, "cuped")["rejection_rate"],
        }
        for rule in RULES
        for k in LOOK_COUNTS
    }
    graded["Q9"] = {"passed": None, "values": q9}
    return graded


def real_replay(n_looks: int = 20) -> dict:
    df = load_criteo_sample()
    order = np.random.default_rng(DEFAULT_SEED).permutation(len(df))
    shuffled = df.iloc[order]
    treatment = shuffled["treatment"].to_numpy()
    y = shuffled["visit"].to_numpy(dtype=np.float64)
    marks = np.arange(1, n_looks + 1) * (len(df) // n_looks)
    cum_b = np.cumsum(treatment == 1)[marks - 1].astype(np.float64)
    cum_a = marks - cum_b
    sum_b = np.cumsum(y * (treatment == 1))[marks - 1]
    sum_a = np.cumsum(y * (treatment == 0))[marks - 1]
    sq_b, sq_a = sum_b, sum_a
    diff, se = difference_and_se(sum_a, sum_b, sq_a, sq_b, cum_a, cum_b)
    information = (1 / se**2) / (1 / se[-1] ** 2)
    control_rate = float(y[treatment == 0].mean())
    tau2 = mixing_variance_for(control_rate, len(df), float(treatment.mean()))
    rows = []
    for rule in RULES:
        exceeds = decision_matrix(rule, diff, se, information, 0.05, tau2)
        stop = int(first_true(exceeds))
        look = stop if stop >= 0 else n_looks - 1
        rows.append(
            {
                "rule": rule,
                "stopped": stop >= 0,
                "look": look + 1,
                "share_of_sample": float(marks[look] / len(df)),
                "estimate": float(diff[look]),
                "std_error": float(se[look]),
            }
        )
    final = average_treatment_effect(df, "visit")
    return {
        "data_source": "criteo",
        "sample_size": len(df),
        "seed": DEFAULT_SEED,
        "n_looks": n_looks,
        "outcome": "visit",
        "mixing_variance": tau2,
        "final_estimate": final.ate,
        "final_ci_low": final.ci_low,
        "final_ci_high": final.ci_high,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sequential testing replay on the Criteo control pool"
    )
    parser.add_argument("--output", default=str(OUTPUT))
    parser.add_argument("--null-replications", type=int, default=NULL_REPLICATIONS)
    parser.add_argument("--lift-replications", type=int, default=LIFT_REPLICATIONS)
    args = parser.parse_args()
    started = time.time()
    pool = load_control_pool()
    covariates = fit_covariates(pool)
    baselines = {o: float(covariates[o]["baseline"]) for o in OUTCOMES}
    tau2 = {o: mixing_variance_for(baselines[o], 2 * N_ARM, 0.5) for o in OUTCOMES}
    scenarios = {0.0: np.random.SeedSequence(DEFAULT_SEED).spawn(args.null_replications)}
    for index, lift in enumerate(LIFTS, start=1):
        key = np.random.SeedSequence(DEFAULT_SEED, spawn_key=(index,))
        scenarios[lift] = key.spawn(args.lift_replications)
    rows: list[dict] = []
    saving: dict[str, float] = {}
    for lift, seeds in scenarios.items():
        result = run_scenario(pool, covariates, seeds, lift, f"lift {lift}")
        if lift == 0.0:
            saving = {o: variance_saving(result, o) for o in OUTCOMES}
        for (outcome, adjustment), (diff, se) in result.items():
            for n_looks in LOOK_COUNTS:
                for rule in RULES:
                    row = summarise(
                        diff,
                        se,
                        rule,
                        n_looks,
                        outcome,
                        adjustment,
                        lift,
                        baselines[outcome],
                        tau2[outcome],
                    )
                    rows.append(row.to_dict())
    reference = 1 - (0.00013125541434961204 / 0.00014631642098864926) ** 2
    record = {
        "data_source": "criteo",
        "pool_rows": len(pool),
        "n_arm": N_ARM,
        "look_counts": list(LOOK_COUNTS),
        "null_replications": args.null_replications,
        "lift_replications": args.lift_replications,
        "lifts": list(LIFTS),
        "seed": DEFAULT_SEED,
        "mixing_variance": tau2,
        "baseline_rates": baselines,
        "cuped_theta": {o: covariates[o]["theta"] for o in OUTCOMES},
        "variance_saving_null": saving,
        "reference_variance_saving": reference,
        "expectations": grade_expectations(rows, saving["visit"], reference),
        "rows": rows,
        "real_replay": real_replay(),
        "runtime_seconds": time.time() - started,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(record, indent=2))
    print(json.dumps(record["expectations"], indent=1)[:3000])


if __name__ == "__main__":
    main()
