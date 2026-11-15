from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy import optimize, special, stats

RULES = ("fixed", "peek", "bonferroni", "pocock", "obf", "msprt")
GRID_POINTS = 600
TAIL_CAP = 8.0


def difference_and_se(
    sum_a: np.ndarray,
    sum_b: np.ndarray,
    sumsq_a: np.ndarray,
    sumsq_b: np.ndarray,
    n_a: np.ndarray,
    n_b: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    n_b = n_a if n_b is None else n_b
    mean_a = sum_a / n_a
    mean_b = sum_b / n_b
    var_a = np.maximum(sumsq_a - n_a * mean_a**2, 0.0) / (n_a - 1)
    var_b = np.maximum(sumsq_b - n_b * mean_b**2, 0.0) / (n_b - 1)
    return mean_b - mean_a, np.sqrt(var_a / n_a + var_b / n_b)


def z_statistics(
    sum_a: np.ndarray,
    sum_b: np.ndarray,
    sumsq_a: np.ndarray,
    sumsq_b: np.ndarray,
    n_a: np.ndarray,
    n_b: np.ndarray | None = None,
) -> np.ndarray:
    diff, se = difference_and_se(sum_a, sum_b, sumsq_a, sumsq_b, n_a, n_b)
    return np.divide(diff, se, out=np.zeros_like(diff), where=se > 0)


def _midpoint_grid(half_width: float) -> tuple[np.ndarray, float]:
    edges = np.linspace(-half_width, half_width, GRID_POINTS + 1)
    return (edges[:-1] + edges[1:]) / 2, float(edges[1] - edges[0])


def _first_density(t_first: float, half_width: float) -> tuple[np.ndarray, np.ndarray, float]:
    nodes, step = _midpoint_grid(half_width)
    sd = np.sqrt(t_first)
    return nodes, stats.norm.pdf(nodes, scale=sd), step


def _next_density(
    nodes: np.ndarray,
    density: np.ndarray,
    step: float,
    time_step: float,
    half_width: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    new_nodes, new_step = _midpoint_grid(half_width)
    sd = np.sqrt(time_step)
    kernel = stats.norm.pdf((new_nodes[None, :] - nodes[:, None]) / sd) / sd
    return new_nodes, (density * step) @ kernel, new_step


def _continue_mass(
    nodes: np.ndarray, density: np.ndarray, step: float, time_step: float, half_width: float
) -> float:
    sd = np.sqrt(time_step)
    inside = special.ndtr((half_width - nodes) / sd) - special.ndtr((-half_width - nodes) / sd)
    return float(np.sum(density * step * inside))


def _surviving_mass(
    c: float,
    first: bool,
    nodes: np.ndarray | None,
    density: np.ndarray | None,
    step: float,
    time_step: float,
    root_t: float,
) -> float:
    if first:
        return float(2 * special.ndtr(c) - 1)
    return _continue_mass(nodes, density, step, time_step, c * root_t)


def _boundaries_for_spending(information: np.ndarray, cumulative_spend: np.ndarray) -> np.ndarray:
    bounds = np.empty(len(information))
    nodes = density = None
    step = 0.0
    surviving = 1.0
    for k, t in enumerate(information):
        spent = cumulative_spend[k] - (cumulative_spend[k - 1] if k else 0.0)
        target = surviving - spent
        root_t = np.sqrt(t)
        time_step = t - information[k - 1] if k else t
        args = (k == 0, nodes, density, step, time_step, root_t)
        if _surviving_mass(TAIL_CAP, *args) <= target:
            bound = TAIL_CAP
        else:
            bound = float(
                optimize.brentq(
                    lambda c, args=args, target=target: _surviving_mass(c, *args) - target,
                    1e-9,
                    TAIL_CAP,
                    xtol=1e-10,
                )
            )
        bounds[k] = bound
        half_width = bound * root_t
        if k == 0:
            nodes, density, step = _first_density(t, half_width)
        else:
            nodes, density, step = _next_density(nodes, density, step, time_step, half_width)
        surviving = float(np.sum(density * step))
    return bounds


def _constant_survival(c: float, information: np.ndarray) -> float:
    nodes, density, step = _first_density(information[0], c * np.sqrt(information[0]))
    for k in range(1, len(information)):
        nodes, density, step = _next_density(
            nodes,
            density,
            step,
            information[k] - information[k - 1],
            c * np.sqrt(information[k]),
        )
    return float(np.sum(density * step))


@lru_cache(maxsize=64)
def _cached_boundaries(
    information: tuple[float, ...], alpha: float, kind: str
) -> tuple[float, ...]:
    t = np.asarray(information)
    if kind == "pocock":
        if len(t) == 1:
            return (float(stats.norm.ppf(1 - alpha / 2)),)
        root = optimize.brentq(
            lambda c: _constant_survival(c, t) - (1 - alpha), 1.5, 6.0, xtol=1e-6
        )
        return tuple([float(root)] * len(t))
    if kind == "obf":
        z_half = stats.norm.ppf(1 - alpha / 2)
        spend = 2 - 2 * stats.norm.cdf(z_half / np.sqrt(t))
        return tuple(float(b) for b in _boundaries_for_spending(t, spend))
    raise ValueError("kind must be 'pocock' or 'obf'")


def spending_boundaries(
    information_fractions: np.ndarray | list[float], alpha: float = 0.05, kind: str = "obf"
) -> np.ndarray:
    t = np.asarray(information_fractions, dtype=float)
    if t.ndim != 1 or len(t) == 0:
        raise ValueError("information_fractions must be a non-empty vector")
    if np.any(np.diff(t) <= 0) or t[0] <= 0 or abs(t[-1] - 1.0) > 1e-9:
        raise ValueError("information_fractions must increase and end at 1")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")
    return np.asarray(_cached_boundaries(tuple(np.round(t, 12)), float(alpha), kind))


def msprt_log_likelihood_ratio(
    diff: np.ndarray, se: np.ndarray, mixing_variance: float
) -> np.ndarray:
    v = se**2
    total = v + mixing_variance
    return 0.5 * np.log(v / total) + mixing_variance * diff**2 / (2 * v * total)


def msprt_p_values(diff: np.ndarray, se: np.ndarray, mixing_variance: float) -> np.ndarray:
    if mixing_variance <= 0:
        raise ValueError("mixing_variance must be positive")
    log_lr = msprt_log_likelihood_ratio(diff, se, mixing_variance)
    p = np.exp(-np.clip(log_lr, 0.0, None))
    return np.minimum.accumulate(p, axis=-1)


def confidence_sequence(
    diff: np.ndarray, se: np.ndarray, mixing_variance: float, alpha: float = 0.05
) -> tuple[np.ndarray, np.ndarray]:
    if mixing_variance <= 0:
        raise ValueError("mixing_variance must be positive")
    v = se**2
    total = v + mixing_variance
    radius = np.sqrt(v * total / mixing_variance * (2 * np.log(1 / alpha) + np.log(total / v)))
    return diff - radius, diff + radius


def first_true(exceeds: np.ndarray) -> np.ndarray:
    first = np.argmax(exceeds, axis=-1)
    return np.where(exceeds.any(axis=-1), first, -1)


def stopping_look(statistics: np.ndarray, boundaries: np.ndarray) -> np.ndarray:
    return first_true(np.abs(statistics) >= boundaries)


def decision_matrix(
    rule: str,
    diff: np.ndarray,
    se: np.ndarray,
    information: np.ndarray,
    alpha: float = 0.05,
    mixing_variance: float | None = None,
) -> np.ndarray:
    if rule not in RULES:
        raise ValueError(f"rule must be one of {RULES}")
    n_looks = diff.shape[-1]
    z = np.divide(diff, se, out=np.zeros_like(diff), where=se > 0)
    if rule == "msprt":
        if mixing_variance is None:
            raise ValueError("msprt needs a mixing_variance")
        return msprt_p_values(diff, se, mixing_variance) <= alpha
    if rule == "fixed":
        exceeds = np.zeros(z.shape, dtype=bool)
        exceeds[..., -1] = np.abs(z[..., -1]) >= stats.norm.ppf(1 - alpha / 2)
        return exceeds
    if rule == "peek":
        return np.abs(z) >= stats.norm.ppf(1 - alpha / 2)
    if rule == "bonferroni":
        return np.abs(z) >= stats.norm.ppf(1 - alpha / (2 * n_looks))
    return np.abs(z) >= spending_boundaries(information, alpha, rule)


def cuped_adjust(
    y: np.ndarray,
    x: np.ndarray,
    theta: float | None = None,
    center: float | None = None,
) -> np.ndarray:
    if theta is None:
        var_x = np.var(x, ddof=1)
        theta = float(np.cov(y, x)[0, 1] / var_x) if var_x > 0 else 0.0
    center = float(np.mean(x)) if center is None else center
    return y - theta * (x - center)
