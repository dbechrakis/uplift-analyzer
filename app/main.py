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
from core.synthetic import SYNTHETIC_BANNER, generate_synthetic

SAMPLE_SIZES = [50_000, 100_000, 250_000, 500_000, 1_000_000, 2_000_000]
SYNTHETIC_MAX = 200_000
SOURCES = ["Criteo (real)", "Synthetic demo"]
EXAMPLE_QUESTIONS = [
    "Did the treatment increase conversions, and by how much?",
    "Is the treatment/control split trustworthy?",
    "Which f0 segments respond most to the treatment?",
    "Can an uplift model target users better than random?",
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
    ate = run_tool(toolbox, "average_treatment_effect", outcome=outcome)
    rel = run_tool(toolbox, "relative_uplift", outcome=outcome)
    if not (summary.result and srm.result and ate.result and rel.result):
        return
    cols = st.columns(4)
    cols[0].metric("Users in sample", f"{summary.result['rows']:,}")
    cols[1].metric("Treated share", f"{summary.result['treated_share']:.1%}")
    cols[2].metric(f"Effect on {outcome}", f"{ate.result['ate'] * 100:+.3f} pp")
    lift = rel.result["relative_uplift"]
    cols[3].metric("Relative uplift", "n/a" if lift is None else f"{lift:+.1%}")
    if srm.result["mismatch"]:
        st.error(explain("sample_ratio_check", srm.result))
    else:
        st.success(explain("sample_ratio_check", srm.result))
    left, right = st.columns(2)
    left.plotly_chart(charts.rates_figure(ate.result), use_container_width=True)
    right.plotly_chart(charts.effect_figure(ate.result), use_container_width=True)
    st.markdown("**Plain-language summary (rule-based, no LLM)**")
    st.write(explain("average_treatment_effect", ate.result))
    st.write(explain("relative_uplift", rel.result))
    for trace in (summary, srm, ate, rel):
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
        "Large samples take a while on the first run; results are cached."
    )
    key = f"qini_{outcome}"
    if st.button("Run Qini analysis", key=f"run_{key}"):
        with st.spinner("Fitting uplift model..."):
            st.session_state[key] = toolbox.call("qini_analysis", {"outcome": outcome})
    trace = st.session_state.get(key)
    if trace is None:
        return
    if trace.error:
        st.error(trace.error)
        return
    st.plotly_chart(charts.qini_figure(trace.result), use_container_width=True)
    st.write(explain("qini_analysis", trace.result))
    show_trace(trace)


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
        "Every number comes from the core functions; raw output is shown for each call."
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
    tabs = st.tabs(["Overview", "Segments", "Qini", "Ask the data"])
    with tabs[0]:
        render_overview(toolbox, outcome)
    with tabs[1]:
        render_segments(toolbox, outcome, feature, n_bins)
    with tabs[2]:
        render_qini(toolbox, outcome)
    with tabs[3]:
        render_ask(toolbox, outcome, use_llm, settings)


main()
