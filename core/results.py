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
    is_atom: bool = False


@dataclass(frozen=True)
class SegmentResult(Serializable):
    feature: str
    outcome: str
    data_source: str
    n_bins_requested: int
    confidence_level: float
    adjustment: str
    segments: list[SegmentRow] = field(default_factory=list)
    n_bins_realized: int = 0


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


@dataclass(frozen=True)
class AdjustedEffectResult(Serializable):
    outcome: str
    data_source: str
    n: int
    n_covariates: int
    adjusted_ate: float
    std_error: float
    ci_low: float
    ci_high: float
    confidence_level: float
    p_value: float
    unadjusted_ate: float
    unadjusted_std_error: float
    variance_reduction: float
    estimator: str = "Lin (2013) interacted OLS, HC1 robust standard errors"


@dataclass(frozen=True)
class CaceResult(Serializable):
    outcome: str
    data_source: str
    n_treated: int
    n_control: int
    exposure_rate_treated: float
    exposure_rate_control: float
    itt: float
    itt_ci_low: float
    itt_ci_high: float
    cace: float
    cace_std_error: float
    cace_ci_low: float
    cace_ci_high: float
    confidence_level: float
    one_sided: bool
    complier_rate_control: float | None
    complier_rate_treated: float | None
    never_taker_rate: float | None
    naive_exposed_vs_control: float
    method: str = "Wald instrumental-variable estimator, delta-method CI"


@dataclass(frozen=True)
class PowerResult(Serializable):
    mode: str
    baseline_rate: float
    treated_rate: float
    absolute_effect: float
    relative_effect: float
    alpha: float
    power: float
    treated_share: float
    n_total: int
    n_treated: int
    n_control: int
    method: str = "two-sided two-proportion z-test, pooled variance under the null"


@dataclass(frozen=True)
class PolicyPoint(Serializable):
    fraction: float
    contacts: int
    incremental_conversions: float
    profit_model: float
    profit_random: float


@dataclass(frozen=True)
class PolicyResult(Serializable):
    outcome: str
    data_source: str
    audience_size: int
    value_per_conversion: float
    cost_per_contact: float
    best_fraction: float
    best_contacts: int
    best_incremental_conversions: float
    best_profit: float
    best_profit_ci_low: float
    best_profit_ci_high: float
    profit_target_all: float
    profit_random_same_fraction: float
    breakeven_cost_target_all: float
    n_bootstrap: int
    confidence_level: float
    points: list[PolicyPoint] = field(default_factory=list)
    scenario_note: str = (
        "Scenario: value and cost are assumptions, not measured. The profit interval reflects "
        "sampling noise at the chosen fraction; the fraction itself is chosen on the same "
        "held-out users, so the optimum is slightly optimistic."
    )


@dataclass(frozen=True)
class LearnerRow(Serializable):
    name: str
    qini_coefficient: float
    ci_low: float
    ci_high: float
    diff_vs_t_learner: float
    diff_ci_low: float
    diff_ci_high: float
    differs_from_t_learner: bool


@dataclass(frozen=True)
class LearnerComparisonResult(Serializable):
    outcome: str
    data_source: str
    n_train: int
    n_test: int
    n_bootstrap: int
    confidence_level: float
    best_by_point_estimate: str
    learners: list[LearnerRow] = field(default_factory=list)


@dataclass(frozen=True)
class BalanceRow(Serializable):
    feature: str
    mean_treated: float
    mean_control: float
    smd: float
    p_value: float
    p_value_adjusted: float


@dataclass(frozen=True)
class BalanceResult(Serializable):
    data_source: str
    n_treated: int
    n_control: int
    max_abs_smd: float
    smd_threshold: float
    features_over_threshold: int
    auc: float
    auc_ci_low: float
    auc_ci_high: float
    n_bootstrap: int
    confidence_level: float
    joint_imbalance_detected: bool
    rows: list[BalanceRow] = field(default_factory=list)


@dataclass(frozen=True)
class PropensityResult(Serializable):
    outcome: str
    data_source: str
    n: int
    n_folds: int
    clip_low: float
    clip_high: float
    share_clipped: float
    propensity_min: float
    propensity_max: float
    ipw_ate: float
    std_error: float
    ci_low: float
    ci_high: float
    confidence_level: float
    unadjusted_ate: float
    effective_n_treated: float
    effective_n_control: float
    method: str = (
        "Hajek inverse-propensity weighting with cross-fitted gradient-boosting propensities; "
        "linearized standard error that treats the propensity as known (conservative)"
    )
