from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

DATA_SOURCES = ("criteo", "synthetic", "constructed", "unspecified")


class Serializable:
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)  # type: ignore[call-overload]

    def to_json(self, **kwargs: Any) -> str:
        return json.dumps(self.to_dict(), **kwargs)


@dataclass(frozen=True)
class AteResult(Serializable):
    outcome: str
    data_source: str
    n_treated: int
    n_control: int
    rate_treated: float
    rate_control: float
    ate: float
    std_error: float
    ci_low: float
    ci_high: float
    confidence_level: float
    z_statistic: float
    p_value: float


@dataclass(frozen=True)
class RelativeUpliftResult(Serializable):
    outcome: str
    data_source: str
    rate_treated: float
    rate_control: float
    relative_uplift: float | None
    ci_low: float | None
    ci_high: float | None
    confidence_level: float
    method: str = "delta method on log risk ratio"


@dataclass(frozen=True)
class SegmentRow(Serializable):
    label: str
    lower: float
    upper: float
    n: int
    n_treated: int
    n_control: int
    rate_treated: float | None
    rate_control: float | None
    ate: float | None
    ci_low: float | None
    ci_high: float | None
    p_value: float | None
    p_value_adjusted: float | None
    significant: bool


@dataclass(frozen=True)
class SegmentResult(Serializable):
    feature: str
    outcome: str
    data_source: str
    n_bins_requested: int
    confidence_level: float
    adjustment: str
    segments: list[SegmentRow] = field(default_factory=list)


@dataclass(frozen=True)
class QiniResult(Serializable):
    outcome: str
    data_source: str
    model: str
    features: list[str]
    n_train: int
    n_test: int
    auuc: float
    auuc_random: float
    qini_coefficient: float
    qini_ci_low: float
    qini_ci_high: float
    n_bootstrap: int
    confidence_level: float
    curve_x: list[float]
    curve_qini: list[float]
    curve_random: list[float]


@dataclass(frozen=True)
class SrmResult(Serializable):
    data_source: str
    n_treated: int
    n_control: int
    observed_share_treated: float
    expected_share_treated: float
    chi2_statistic: float
    p_value: float
    alpha: float
    mismatch: bool
