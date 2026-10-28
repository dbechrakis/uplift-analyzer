from __future__ import annotations

import argparse
import json
import time

from core.data import default_sample_size, default_seed, describe, load_criteo_sample
from core.metrics import average_treatment_effect, relative_uplift
from core.segments import uplift_by_segment
from core.srm import sample_ratio_check
from core.synthetic import SYNTHETIC_BANNER, generate_synthetic
from core.uplift_model import qini_analysis


def build_report(df, outcome: str, feature: str, n_bootstrap: int) -> dict:
    return {
        "dataset": describe(df).__dict__,
        "sample_ratio": sample_ratio_check(df).to_dict(),
        "ate": average_treatment_effect(df, outcome).to_dict(),
        "relative_uplift": relative_uplift(df, outcome).to_dict(),
        "segments": uplift_by_segment(df, feature, outcome=outcome).to_dict(),
        "qini": qini_analysis(df, outcome=outcome, n_bootstrap=n_bootstrap).to_dict(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all core analyses and print JSON")
    parser.add_argument("--sample-size", type=int, default=default_sample_size())
    parser.add_argument("--seed", type=int, default=default_seed())
    parser.add_argument("--outcome", default="conversion")
    parser.add_argument("--feature", default="f0")
    parser.add_argument("--bootstrap", type=int, default=200)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--compact", action="store_true")
    args = parser.parse_args()
    started = time.time()
    if args.synthetic:
        print(SYNTHETIC_BANNER)
        df = generate_synthetic(n=min(args.sample_size, 100_000), seed=args.seed)
    else:
        df = load_criteo_sample(args.sample_size, args.seed, download=args.download)
    report = build_report(df, args.outcome, args.feature, args.bootstrap)
    report["runtime_seconds"] = round(time.time() - started, 1)
    if args.compact:
        for key in ("curve_x", "curve_qini", "curve_random"):
            report["qini"].pop(key)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
