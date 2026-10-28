from __future__ import annotations

import math

from scipy import optimize, stats

from core.results import PowerResult


def _check_rate(name: str, value: float) -> None:
    if not 0 < value < 1:
        raise ValueError(f"{name} must be strictly between 0 and 1")


def _check_design(alpha: float, power: float, treated_share: float) -> None:
    _check_rate("alpha", alpha)
    _check_rate("power", power)
    _check_rate("treated_share", treated_share)


def _total_n(p0: float, p1: float, alpha: float, power: float, share: float) -> float:
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_power = stats.norm.ppf(power)
    pooled = share * p1 + (1 - share) * p0
    null_sd = math.sqrt(pooled * (1 - pooled) * (1 / share + 1 / (1 - share)))
    alt_sd = math.sqrt(p1 * (1 - p1) / share + p0 * (1 - p0) / (1 - share))
    return (z_alpha * null_sd + z_power * alt_sd) ** 2 / (p1 - p0) ** 2


def _result(
    mode: str,
    p0: float,
    p1: float,
    alpha: float,
    power: float,
    share: float,
    n_treated: int,
    n_control: int,
) -> PowerResult:
    return PowerResult(
        mode=mode,
        baseline_rate=p0,
        treated_rate=p1,
        absolute_effect=p1 - p0,
        relative_effect=(p1 - p0) / p0,
        alpha=alpha,
        power=power,
        treated_share=share,
        n_total=n_treated + n_control,
        n_treated=n_treated,
        n_control=n_control,
    )


def required_sample_size(
    baseline_rate: float,
    relative_lift: float | None = None,
    absolute_lift: float | None = None,
    alpha: float = 0.05,
    power: float = 0.8,
    treated_share: float = 0.5,
) -> PowerResult:
    _check_design(alpha, power, treated_share)
    _check_rate("baseline_rate", baseline_rate)
    if (relative_lift is None) == (absolute_lift is None):
        raise ValueError("Provide exactly one of relative_lift or absolute_lift")
    delta = absolute_lift if absolute_lift is not None else baseline_rate * relative_lift
    if delta == 0:
        raise ValueError("The effect must be non-zero")
    treated_rate = baseline_rate + delta
    _check_rate("treated rate", treated_rate)
    n_float = _total_n(baseline_rate, treated_rate, alpha, power, treated_share)
    n_treated = math.ceil(n_float * treated_share)
    n_control = math.ceil(n_float * (1 - treated_share))
    return _result(
        "required_sample_size",
        baseline_rate,
        treated_rate,
        alpha,
        power,
        treated_share,
        n_treated,
        n_control,
    )


def minimum_detectable_effect(
    baseline_rate: float,
    n_total: int,
    alpha: float = 0.05,
    power: float = 0.8,
    treated_share: float = 0.5,
) -> PowerResult:
    _check_design(alpha, power, treated_share)
    _check_rate("baseline_rate", baseline_rate)
    if n_total < 10:
        raise ValueError("n_total must be at least 10")

    def gap(delta: float) -> float:
        return _total_n(baseline_rate, baseline_rate + delta, alpha, power, treated_share) - n_total

    upper = (1 - baseline_rate) * 0.999999
    if gap(upper) > 0:
        raise ValueError("The sample is too small to detect any effect up to 100%")
    delta = optimize.brentq(gap, 1e-12, upper, xtol=1e-14)
    n_treated = math.ceil(n_total * treated_share)
    return _result(
        "minimum_detectable_effect",
        baseline_rate,
        baseline_rate + delta,
        alpha,
        power,
        treated_share,
        n_treated,
        int(n_total) - n_treated,
    )
