from __future__ import annotations

import pandas as pd

from core.data import OUTCOMES
from core.results import DATA_SOURCES


def resolve_source(df: pd.DataFrame, data_source: str | None) -> str:
    source = data_source or df.attrs.get("data_source", "unspecified")
    if source not in DATA_SOURCES:
        raise ValueError(f"data_source must be one of {DATA_SOURCES}")
    return source


def check_outcome(outcome: str) -> None:
    if outcome not in OUTCOMES:
        raise ValueError(f"outcome must be one of {OUTCOMES}")


def check_alpha(alpha: float) -> None:
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")


def split_arms(df: pd.DataFrame, outcome: str) -> tuple[pd.Series, pd.Series]:
    check_outcome(outcome)
    treated = df.loc[df["treatment"] == 1, outcome]
    control = df.loc[df["treatment"] == 0, outcome]
    if treated.empty or control.empty:
        raise ValueError("Both treatment and control rows are required")
    return treated, control
