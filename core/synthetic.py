from __future__ import annotations

import numpy as np
import pandas as pd

from core.data import FEATURES

SYNTHETIC_BANNER = "SYNTHETIC DATA - for tests and demos only, never real results"


def generate_synthetic(
    n: int = 20_000,
    seed: int = 0,
    treatment_share: float = 0.85,
    base_rate: float = 0.05,
    heterogeneous: bool = True,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, len(FEATURES))).astype("float32")
    treatment = (rng.random(n) < treatment_share).astype("int8")
    logit_base = np.log(base_rate / (1 - base_rate)) + 0.3 * x[:, 1]
    lift = 0.8 * (x[:, 0] > 0) if heterogeneous else np.full(n, 0.4)
    p = 1 / (1 + np.exp(-(logit_base + treatment * lift)))
    conversion = (rng.random(n) < p).astype("int8")
    p_visit = np.clip(p * 4, 0, 0.95)
    visit = np.maximum(conversion, (rng.random(n) < p_visit).astype("int8")).astype("int8")
    exposure = (treatment * (rng.random(n) < 0.7)).astype("int8")
    df = pd.DataFrame(x, columns=FEATURES)
    df["treatment"] = treatment
    df["conversion"] = conversion
    df["visit"] = visit
    df["exposure"] = exposure
    df.attrs["data_source"] = "synthetic"
    return df
