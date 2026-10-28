from __future__ import annotations

from typing import Any

import plotly.graph_objects as go

TREATED = "#2a6fdb"
CONTROL = "#8a8f98"
SIGNIFICANT = "#1b9e77"
NEUTRAL = "#b0b5bd"


def rates_figure(ate: dict[str, Any]) -> go.Figure:
    fig = go.Figure(
        go.Bar(
            x=["Control", "Treated"],
            y=[ate["rate_control"] * 100, ate["rate_treated"] * 100],
            marker_color=[CONTROL, TREATED],
            text=[f"{ate['rate_control'] * 100:.3f}%", f"{ate['rate_treated'] * 100:.3f}%"],
            textposition="outside",
        )
    )
    fig.update_layout(
        title=f"{ate['outcome'].capitalize()} rate by group",
        yaxis_title="Rate (%)",
        showlegend=False,
        margin={"t": 50, "b": 30},
    )
    return fig


def effect_figure(ate: dict[str, Any]) -> go.Figure:
    effect, low, high = (ate[k] * 100 for k in ("ate", "ci_low", "ci_high"))
    fig = go.Figure(
        go.Scatter(
            x=[effect],
            y=["Treatment effect"],
            mode="markers",
            marker={"size": 12, "color": TREATED},
            error_x={
                "type": "data",
                "symmetric": False,
                "array": [high - effect],
                "arrayminus": [effect - low],
                "color": TREATED,
            },
        )
    )
    fig.add_vline(x=0, line_dash="dash", line_color=CONTROL)
    fig.update_layout(
        title=f"Effect on {ate['outcome']} with {ate['confidence_level']:.0%} CI",
        xaxis_title="Percentage points",
        margin={"t": 50, "b": 30},
        height=220,
    )
    return fig


def segments_figure(result: dict[str, Any]) -> go.Figure:
    rows = [s for s in result["segments"] if s["ate"] is not None]
    effects = [s["ate"] * 100 for s in rows]
    fig = go.Figure(
        go.Scatter(
            x=[s["label"] for s in rows],
            y=effects,
            mode="markers",
            marker={
                "size": 11,
                "color": [SIGNIFICANT if s["significant"] else NEUTRAL for s in rows],
            },
            error_y={
                "type": "data",
                "color": NEUTRAL,
                "symmetric": False,
                "array": [(s["ci_high"] - s["ate"]) * 100 for s in rows],
                "arrayminus": [(s["ate"] - s["ci_low"]) * 100 for s in rows],
            },
        )
    )
    fig.add_hline(y=0, line_dash="dash", line_color=CONTROL)
    fig.update_layout(
        title=f"Effect on {result['outcome']} by {result['feature']} bin "
        "(green = significant after FDR)",
        xaxis_title=f"{result['feature']} bin",
        yaxis_title="Percentage points",
        showlegend=False,
        margin={"t": 60, "b": 30},
    )
    return fig


def qini_figure(result: dict[str, Any]) -> go.Figure:
    fig = go.Figure()
    fig.add_scatter(
        x=[x * 100 for x in result["curve_x"]],
        y=result["curve_qini"],
        name="Uplift model",
        line={"color": TREATED, "width": 3},
    )
    fig.add_scatter(
        x=[x * 100 for x in result["curve_x"]],
        y=result["curve_random"],
        name="Random targeting",
        line={"color": CONTROL, "dash": "dash"},
    )
    fig.update_layout(
        title=f"Qini curve ({result['outcome']}), AUUC {result['auuc']:.1f}",
        xaxis_title="Share of users targeted (%)",
        yaxis_title="Cumulative incremental responses",
        margin={"t": 50, "b": 30},
        legend={"orientation": "h", "y": -0.2},
    )
    return fig


def balance_figure(result: dict[str, Any]) -> go.Figure:
    rows = sorted(result["rows"], key=lambda r: abs(r["smd"]))
    fig = go.Figure(
        go.Bar(
            x=[r["smd"] for r in rows],
            y=[r["feature"] for r in rows],
            orientation="h",
            marker_color=[SIGNIFICANT if abs(r["smd"]) < 0.1 else TREATED for r in rows],
        )
    )
    for bound in (-result["smd_threshold"], result["smd_threshold"]):
        fig.add_vline(x=bound, line_dash="dash", line_color=CONTROL)
    fig.update_layout(
        title="Standardized mean difference, treated minus control (dashed = 0.1 rule of thumb)",
        xaxis_title="SMD",
        xaxis_range=[-0.12, 0.12],
        showlegend=False,
        margin={"t": 60, "b": 30},
        height=380,
    )
    return fig


def policy_figure(result: dict[str, Any]) -> go.Figure:
    points = result["points"]
    x = [p["fraction"] * 100 for p in points]
    fig = go.Figure()
    fig.add_scatter(
        x=x,
        y=[p["profit_model"] for p in points],
        name="Uplift model",
        line={"color": TREATED, "width": 3},
    )
    fig.add_scatter(
        x=x,
        y=[p["profit_random"] for p in points],
        name="Random targeting",
        line={"color": CONTROL, "dash": "dash"},
    )
    fig.add_hline(y=0, line_color=NEUTRAL)
    fig.add_scatter(
        x=[result["best_fraction"] * 100],
        y=[result["best_profit"]],
        mode="markers",
        name="Best share",
        marker={"size": 13, "color": SIGNIFICANT},
    )
    fig.update_layout(
        title=f"Scenario profit by share of audience targeted ({result['outcome']})",
        xaxis_title="Share of audience targeted (%)",
        yaxis_title="Profit",
        margin={"t": 50, "b": 30},
        legend={"orientation": "h", "y": -0.2},
    )
    return fig


def learners_figure(result: dict[str, Any]) -> go.Figure:
    rows = result["learners"]
    q = [r["qini_coefficient"] for r in rows]
    fig = go.Figure(
        go.Scatter(
            x=[r["name"] for r in rows],
            y=q,
            mode="markers",
            marker={"size": 12, "color": TREATED},
            error_y={
                "type": "data",
                "symmetric": False,
                "array": [r["ci_high"] - v for r, v in zip(rows, q, strict=True)],
                "arrayminus": [v - r["ci_low"] for r, v in zip(rows, q, strict=True)],
                "color": TREATED,
            },
        )
    )
    fig.add_hline(y=0, line_dash="dash", line_color=CONTROL)
    fig.update_layout(
        title=f"Qini coefficient by uplift learner ({result['outcome']}, 95% bootstrap CI)",
        yaxis_title="Qini coefficient (incremental responses)",
        showlegend=False,
        margin={"t": 60, "b": 30},
    )
    return fig
