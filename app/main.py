from __future__ import annotations

import os

import streamlit as st

from app import charts
from app.agent import run_agent
from app.config import Settings
from app.fallback import explain
from app.llm_client import LlamaClient
from app.tools import TOOL_DEFINITIONS, Toolbox, ToolTrace
from core.data import (
    DEFAULT_SEED,
    FEATURES,
    OUTCOMES,
    DatasetNotFoundError,
    cache_path,
    load_criteo_sample,
    manual_download_instructions,
    raw_path,
)
from core.power import minimum_detectable_effect
from core.synthetic import SYNTHETIC_BANNER, generate_synthetic

SAMPLE_SIZES = [50_000, 100_000, 250_000, 500_000, 1_000_000, 2_000_000]
SYNTHETIC_MAX = 200_000
SOURCES = ["Criteo (real)", "Synthetic demo"]
TAB_LABELS = [
    "Overview",
    "Exposure",
    "Segments",
    "Qini",
    "Targeting policy",
    "Experiment design",
    "Ask the data",
]
EXAMPLE_QUESTIONS = [
    "Did the treatment increase conversions, and by how much?",
    "Is the experiment randomization trustworthy?",
    "What share of users should we target to maximize profit?",
    "How many users do we need to detect a 10% conversion lift?",
]


def default_source_index() -> int:
    forced = os.environ.get("UPLIFT_APP_SOURCE")
    if forced:
        return 1 if forced == "synthetic" else 0
    return 0 if raw_path().exists() or cache_path(0, 0).parent.exists() else 1


@st.cache_resource(show_spinner="Loading sample...")
def get_toolbox(source: str, sample_size: int, seed: int) -> Toolbox:
    if source == "synthetic":
        return Toolbox(generate_synthetic(n=min(sample_size, SYNTHETIC_MAX), seed=seed))
    return Toolbox(load_criteo_sample(sample_size, seed))


@st.cache_data(ttl=10, show_spinner=False)
def llm_online(base_url: str) -> bool:
    return LlamaClient(Settings.from_env()).is_available()


def show_trace(trace: ToolTrace, expanded: bool = False) -> None:
    label = f"Function called: {trace.signature()}"
    if trace.cached:
        label += " (cached)"
    with st.expander(label, expanded=expanded):
        if trace.error:
            st.error(trace.error)
        else:
            st.json(trace.result, expanded=False)


def run_tool(toolbox: Toolbox, name: str, **arguments) -> ToolTrace:
    trace = toolbox.call(name, arguments)
    if trace.error:
        st.error(trace.error)
    return trace


def render_overview(toolbox: Toolbox, outcome: str) -> None:
    summary = run_tool(toolbox, "dataset_summary")
    srm = run_tool(toolbox, "sample_ratio_check")
    balance = run_tool(toolbox, "covariate_balance")
    ate = run_tool(toolbox, "average_treatment_effect", outcome=outcome)
    adjusted = run_tool(toolbox, "regression_adjusted_effect", outcome=outcome)
    rel = run_tool(toolbox, "relative_uplift", outcome=outcome)
    if not all(t.result for t in (summary, srm, balance, ate, adjusted, rel)):
        return
    cols = st.columns(5)
    cols[0].metric("Users in sample", f"{summary.result['rows']:,}")
    cols[1].metric("Treated share", f"{summary.result['treated_share']:.1%}")
    cols[2].metric(f"Effect on {outcome}", f"{ate.result['ate'] * 100:+.3f} pp")
    cols[3].metric("Covariate-adjusted effect", f"{adjusted.result['adjusted_ate'] * 100:+.3f} pp")
    lift = rel.result["relative_uplift"]
    cols[4].metric("Relative uplift", "n/a" if lift is None else f"{lift:+.1%}")
    st.subheader("Is the experiment trustworthy?")
    if srm.result["mismatch"]:
        st.error(explain("sample_ratio_check", srm.result))
    else:
        st.success(explain("sample_ratio_check", srm.result))
    if balance.result["joint_imbalance_detected"]:
        st.warning(explain("covariate_balance", balance.result))
    else:
        st.success(explain("covariate_balance", balance.result))
    st.plotly_chart(charts.balance_figure(balance.result), use_container_width=True)
    left, right = st.columns(2)
    left.plotly_chart(charts.rates_figure(ate.result), use_container_width=True)
    right.plotly_chart(charts.effect_figure(ate.result), use_container_width=True)
    st.markdown("**Plain-language summary (rule-based, no LLM)**")
    st.write(explain("average_treatment_effect", ate.result))
    st.write(explain("regression_adjusted_effect", adjusted.result))
    st.write(explain("relative_uplift", rel.result))
    for trace in (summary, srm, balance, ate, adjusted, rel):
        show_trace(trace)


def render_exposure(toolbox: Toolbox, outcome: str) -> None:
    if outcome == "exposure":
        st.info("Exposure is the first stage here. Pick conversion or visit as the outcome.")
        return
    trace = run_tool(toolbox, "complier_effect", outcome=outcome)
    if not trace.result:
        return
    r = trace.result
    cols = st.columns(4)
    cols[0].metric("Treated users exposed", f"{r['exposure_rate_treated']:.1%}")
    cols[1].metric("Intent-to-treat effect", f"{r['itt'] * 100:+.3f} pp")
    cols[2].metric("Effect on exposed (CACE)", f"{r['cace'] * 100:+.2f} pp")
    cols[3].metric("Naive exposed vs control", f"{r['naive_exposed_vs_control'] * 100:+.2f} pp")
    st.write(explain("complier_effect", r))
    if r["one_sided"] and r["complier_rate_control"] is not None:
        st.dataframe(
            [
                {
                    "group": "Exposed users if they had not been treated",
                    "rate": r["complier_rate_control"],
                },
                {"group": "Exposed users, treated", "rate": r["complier_rate_treated"]},
                {"group": "Treated users who were never exposed", "rate": r["never_taker_rate"]},
            ],
            hide_index=True,
            use_container_width=True,
        )
    show_trace(trace)


def render_segments(toolbox: Toolbox, outcome: str, feature: str, n_bins: int) -> None:
    trace = run_tool(toolbox, "uplift_by_segment", feature=feature, outcome=outcome, n_bins=n_bins)
    if not trace.result:
        return
    realized = len(trace.result["segments"])
    if realized < n_bins:
        st.info(
            f"{feature} has repeated values, so {realized} bins were formed instead of {n_bins}."
        )
    st.plotly_chart(charts.segments_figure(trace.result), use_container_width=True)
    st.dataframe(
        [
            {
                "bin": s["label"],
                "users": s["n"],
                "treated": s["n_treated"],
                "control": s["n_control"],
                "effect (pp)": None if s["ate"] is None else round(s["ate"] * 100, 4),
                "p (FDR-adjusted)": s["p_value_adjusted"],
                "significant": s["significant"],
            }
            for s in trace.result["segments"]
        ],
        use_container_width=True,
        hide_index=True,
    )
    st.write(explain("uplift_by_segment", trace.result))
    show_trace(trace)


def render_qini(toolbox: Toolbox, outcome: str) -> None:
    st.caption(
        "Fits two gradient-boosting models on a train split and scores a held-out test split. "
        "Large samples take a while on the first run. Results are cached."
    )
    key = f"qini_{outcome}"
    if st.button("Run Qini analysis", key=f"run_{key}"):
        with st.spinner("Fitting uplift model..."):
            st.session_state[key] = toolbox.call("qini_analysis", {"outcome": outcome})
    trace = st.session_state.get(key)
    if trace is not None:
        if trace.error:
            st.error(trace.error)
        else:
            st.plotly_chart(charts.qini_figure(trace.result), use_container_width=True)
            st.write(explain("qini_analysis", trace.result))
            show_trace(trace)
    st.subheader("Which uplift learner?")
    st.caption("T-, S-, X-learner and transformed outcome on the same split, paired bootstrap.")
    lkey = f"learners_{outcome}"
    if st.button("Compare uplift learners", key=f"run_{lkey}"):
        with st.spinner("Fitting four learners..."):
            st.session_state[lkey] = toolbox.call("compare_uplift_learners", {"outcome": outcome})
    ltrace = st.session_state.get(lkey)
    if ltrace is not None:
        if ltrace.error:
            st.error(ltrace.error)
        else:
            st.plotly_chart(charts.learners_figure(ltrace.result), use_container_width=True)
            st.write(explain("compare_uplift_learners", ltrace.result))
            show_trace(ltrace)


def render_policy(toolbox: Toolbox, outcome: str) -> None:
    st.caption(
        "Scenario planner. Value and cost are assumptions you set, not measurements. "
        "The model ranks users. The Qini curve says how many extra conversions each share buys."
    )
    cols = st.columns(3)
    audience = cols[0].number_input("Audience size", 1_000, 1_000_000_000, 1_000_000, step=100_000)
    value = cols[1].number_input("Value per conversion", 0.01, 1_000_000.0, 30.0, step=5.0)
    cost = cols[2].number_input("Cost per contact", 0.0, 10_000.0, 0.05, step=0.01, format="%.3f")
    key = f"policy_{outcome}_{audience}_{value}_{cost}"
    if st.button("Find the best targeting share", key="run_policy"):
        with st.spinner("Fitting uplift model and bootstrapping..."):
            st.session_state["policy_last"] = (
                key,
                toolbox.call(
                    "targeting_policy",
                    {
                        "outcome": outcome,
                        "audience_size": int(audience),
                        "value_per_conversion": float(value),
                        "cost_per_contact": float(cost),
                    },
                ),
            )
    last = st.session_state.get("policy_last")
    if last is None:
        return
    trace = last[1]
    if trace.error:
        st.error(trace.error)
        return
    r = trace.result
    metrics = st.columns(4)
    metrics[0].metric("Best share to target", f"{r['best_fraction']:.1%}")
    metrics[1].metric("Incremental conversions", f"{r['best_incremental_conversions']:,.0f}")
    metrics[2].metric("Profit at best share", f"{r['best_profit']:,.0f}")
    metrics[3].metric("Profit if everyone is targeted", f"{r['profit_target_all']:,.0f}")
    st.plotly_chart(charts.policy_figure(r), use_container_width=True)
    st.write(explain("targeting_policy", r))
    show_trace(trace)


def render_design(toolbox: Toolbox, outcome: str) -> None:
    st.caption("Plan a future test: how many users, or what effect your sample can detect.")
    control_rate = float(toolbox.df.loc[toolbox.df["treatment"] == 0, outcome].mean())
    left, right = st.columns(2)
    with left:
        st.subheader("How many users do I need?")
        lift = st.number_input("Relative lift to detect (%)", 1.0, 1000.0, 10.0, step=1.0)
        power = st.slider("Power", 0.5, 0.99, 0.8, 0.01)
        share = st.slider("Treated share", 0.05, 0.95, 0.5, 0.05)
        trace = toolbox.call(
            "required_sample_size",
            {
                "outcome": outcome,
                "relative_lift": lift / 100,
                "power": float(power),
                "treated_share": float(share),
            },
        )
        if trace.error:
            st.error(trace.error)
        else:
            st.metric("Users needed in total", f"{trace.result['n_total']:,}")
            st.write(explain("required_sample_size", trace.result))
            show_trace(trace)
    with right:
        st.subheader("What can my sample detect?")
        n_total = st.number_input("Total users", 100, 1_000_000_000, 100_000, step=10_000)
        mde_power = st.slider("Power ", 0.5, 0.99, 0.8, 0.01)
        mde_share = st.slider("Treated share ", 0.05, 0.95, 0.5, 0.05)
        try:
            mde = minimum_detectable_effect(
                control_rate, int(n_total), power=float(mde_power), treated_share=float(mde_share)
            )
            st.metric("Smallest detectable relative lift", f"{mde.relative_effect:+.1%}")
            st.caption(
                f"Baseline {control_rate:.3%} (control group rate). Absolute effect "
                f"{mde.absolute_effect * 100:.3f} pp at {mde_power:.0%} power and 5% significance."
            )
        except ValueError as exc:
            st.error(str(exc))


def render_chat_history() -> None:
    for message in st.session_state.setdefault("chat", []):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("warning"):
                st.warning(message["warning"])
            for trace in message.get("traces", []):
                show_trace(trace)


def unverified_warning(numbers: list[str]) -> str | None:
    if not numbers:
        return None
    return (
        "These numbers in the explanation were not found in the tool results, verify them "
        f"against the raw JSON: {', '.join(numbers)}"
    )


def ask_llm(question: str, toolbox: Toolbox, client: LlamaClient) -> None:
    history = list(st.session_state["chat"])
    st.session_state["chat"].append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        placeholder = st.empty()
        streamed = ""
        shown: list[ToolTrace] = []
        outcome = {"content": "", "warning": None}
        for event in run_agent(question, toolbox, client, history):
            if event.kind == "text":
                streamed += event.text
                placeholder.markdown(streamed + "▌")
            elif event.kind == "tool_call":
                streamed = ""
                placeholder.markdown(f"Running `{event.call.name}`...")
            elif event.kind == "tool_result":
                shown.append(event.trace)
                show_trace(event.trace)
            elif event.kind == "final":
                outcome["content"] = event.text or "(no answer)"
                outcome["warning"] = unverified_warning(event.unverified_numbers)
            elif event.kind in ("error", "limit"):
                outcome["content"] = event.text
                st.error(event.text)
        placeholder.markdown(outcome["content"])
        if outcome["warning"]:
            st.warning(outcome["warning"])
    st.session_state["chat"].append(
        {**outcome, "role": "assistant", "traces": shown, "warning": outcome["warning"]}
    )


def render_fallback(toolbox: Toolbox, outcome: str) -> None:
    st.info(
        "LLM mode is off or the server is unreachable. Pick an analysis below: the same core "
        "functions run and a rule-based explanation is shown next to the raw result."
    )
    name = st.selectbox("Analysis", list(TOOL_DEFINITIONS), key="fallback_tool")
    arguments: dict = {}
    properties = TOOL_DEFINITIONS[name]["properties"]
    if "outcome" in properties:
        arguments["outcome"] = st.selectbox("Outcome", OUTCOMES, index=OUTCOMES.index(outcome))
    if "feature" in properties:
        arguments["feature"] = st.selectbox("Feature", FEATURES, key="fallback_feature")
        arguments["n_bins"] = st.slider("Bins", 2, 10, 5, key="fallback_bins")
    if "expected_share_treated" in properties:
        arguments["expected_share_treated"] = st.number_input(
            "Expected treated share", 0.01, 0.99, 0.85, key="fallback_share"
        )
    if st.button("Run analysis", key="fallback_run"):
        with st.spinner("Computing..."):
            trace = toolbox.call(name, arguments)
        if trace.error:
            st.error(trace.error)
        else:
            st.write(explain(name, trace.result))
        show_trace(trace, expanded=True)


def render_ask(toolbox: Toolbox, outcome: str, use_llm: bool, settings: Settings) -> None:
    online = use_llm and llm_online(settings.base_url)
    if not online:
        render_fallback(toolbox, outcome)
        return
    st.caption(
        "The model only chooses which function to run and explains its result. "
        "Every number comes from the core functions. Raw output is shown for each call."
    )
    render_chat_history()
    cols = st.columns(len(EXAMPLE_QUESTIONS))
    picked = None
    for col, example in zip(cols, EXAMPLE_QUESTIONS, strict=True):
        if col.button(example, use_container_width=True):
            picked = example
    question = st.chat_input("Ask about the experiment") or picked
    if question:
        ask_llm(question, toolbox, LlamaClient(settings))


def main() -> None:
    st.set_page_config(page_title="Uplift Analyzer", page_icon=None, layout="wide")
    settings = Settings.from_env()
    st.title("Uplift Analyzer")
    st.caption("Incrementality of a marketing treatment on the Criteo Uplift dataset")
    with st.sidebar:
        st.header("Data")
        source_label = st.radio("Source", SOURCES, index=default_source_index())
        source = "synthetic" if source_label == SOURCES[1] else "criteo"
        sample_size = st.select_slider(
            "Sample size", SAMPLE_SIZES, value=500_000, format_func=lambda v: f"{v:,}"
        )
        seed = st.number_input("Random seed", value=DEFAULT_SEED, step=1)
        st.header("Analysis")
        outcome = st.selectbox("Outcome", OUTCOMES)
        feature = st.selectbox("Segment feature", FEATURES)
        n_bins = st.slider("Segment bins", 2, 10, 5)
        st.header("LLM")
        use_llm = st.toggle("Use local LLM when available", value=True)
        if use_llm:
            if llm_online(settings.base_url):
                st.success("LLM server connected")
            else:
                st.warning("LLM server offline: no-LLM mode")
            st.caption(f"{settings.base_url}")
    if source == "synthetic":
        st.warning(f"{SYNTHETIC_BANNER}. Capped at {SYNTHETIC_MAX:,} rows.")
    try:
        toolbox = get_toolbox(source, sample_size, int(seed))
    except DatasetNotFoundError:
        st.error("Criteo dataset not found")
        st.code(manual_download_instructions())
        st.info("Switch the sidebar source to 'Synthetic demo' to explore the app meanwhile.")
        st.stop()
    tabs = st.tabs(TAB_LABELS)
    with tabs[0]:
        render_overview(toolbox, outcome)
    with tabs[1]:
        render_exposure(toolbox, outcome)
    with tabs[2]:
        render_segments(toolbox, outcome, feature, n_bins)
    with tabs[3]:
        render_qini(toolbox, outcome)
    with tabs[4]:
        render_policy(toolbox, outcome)
    with tabs[5]:
        render_design(toolbox, outcome)
    with tabs[6]:
        render_ask(toolbox, outcome, use_llm, settings)


main()
