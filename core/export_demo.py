from __future__ import annotations

import argparse
import json
import time
from datetime import date
from pathlib import Path

from core.adjusted import regression_adjusted_effect
from core.balance import covariate_balance
from core.cace import complier_effect
from core.data import DATASET_ROWS, FEATURES, default_seed, load_criteo_sample
from core.learners import compare_uplift_learners
from core.metrics import average_treatment_effect, relative_uplift
from core.segments import uplift_by_segment
from core.srm import sample_ratio_check
from core.uplift_model import qini_analysis, split_holdout

OUTCOMES = ("conversion", "visit")


def build_demo_data(
    sample_size: int,
    seed: int,
    validation_path: Path,
    qini_bootstrap: int = 200,
    learner_bootstrap: int = 100,
    max_iter: int = 100,
) -> dict:
    df = load_criteo_sample(sample_size, seed)
    started = time.time()
    validation = json.loads(validation_path.read_text()) if validation_path.exists() else None
    out: dict = {
        "generated": date.today().isoformat(),
        "data_source": "criteo",
        "sample_size": sample_size,
        "seed": seed,
        "full_rows": DATASET_ROWS,
        "sample_ratio": sample_ratio_check(df).to_dict(),
        "balance": covariate_balance(df).to_dict(),
        "outcomes": {},
    }
    if validation:
        full = validation["full_data"]
        out["full_data"] = {
            "unadjusted": {o: full["unadjusted"][o] for o in OUTCOMES},
            "adjusted": full["adjusted"],
            "cace": full["cace"],
            "max_abs_smd": full["max_abs_smd"],
        }
        out["sampling_validation"] = validation["sampling"]
        out["code_simulations"] = validation["code_simulations"]
    for outcome in OUTCOMES:
        _, test = split_holdout(df, outcome, 0.3, 42)
        qini = qini_analysis(
            df,
            outcome=outcome,
            n_bootstrap=qini_bootstrap,
            n_curve_points=201,
            max_iter=max_iter,
        ).to_dict()
        out["outcomes"][outcome] = {
            "control_rate": float(df.loc[df["treatment"] == 0, outcome].mean()),
            "ate": average_treatment_effect(df, outcome).to_dict(),
            "adjusted": regression_adjusted_effect(df, outcome).to_dict(),
            "relative": relative_uplift(df, outcome).to_dict(),
            "cace": complier_effect(df, outcome).to_dict(),
            "segments": {
                f: uplift_by_segment(df, f, n_bins=5, outcome=outcome).to_dict() for f in FEATURES
            },
            "qini": qini,
            "policy_scale": {
                "n_test": len(test),
                "treated_share_test": float(test["treatment"].mean()),
            },
            "learners": compare_uplift_learners(
                df, outcome, n_bootstrap=learner_bootstrap, max_iter=max_iter
            ).to_dict(),
        }
    out["runtime_seconds"] = round(time.time() - started)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Export aggregate results for the static demo")
    parser.add_argument("--output", default="demo/data.json")
    parser.add_argument("--sample-size", type=int, default=500_000)
    parser.add_argument("--validation", default="outputs/validation.json")
    args = parser.parse_args()
    data = build_demo_data(args.sample_size, default_seed(), Path(args.validation))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(data, separators=(",", ":")))
    print(f"wrote {args.output} ({Path(args.output).stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
