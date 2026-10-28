from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from app.agent import AgentEvent, run_agent
from app.config import Settings
from app.llm_client import LlamaClient
from app.tools import Toolbox, ToolTrace
from core.data import load_criteo_sample
from core.synthetic import generate_synthetic

ANY_TOOL = "*"


@dataclass
class CaseResult:
    id: str
    question: str
    expected_tool: str | None
    called_tools: list[str]
    tool_correct: bool
    args_correct: bool
    tool_errors: int
    answered: bool
    grounded: bool
    seconds: float
    answer: str


def score_case(
    case: dict[str, Any], traces: list[ToolTrace], final: AgentEvent | None, seconds: float
) -> CaseResult:
    expected = case["expected_tool"]
    accepted = {expected, *case.get("also_accept", [])}
    called = [t.name for t in traces]
    if expected == ANY_TOOL:
        tool_correct = True
    elif expected is None:
        tool_correct = not called
    else:
        tool_correct = bool(called) and called[0] in accepted
    args_correct = True
    wanted = case.get("expected_args") or {}
    if expected not in (None, ANY_TOOL) and wanted:
        if not tool_correct:
            args_correct = False
        elif called[0] == expected:
            args_correct = all(traces[0].arguments.get(k) == v for k, v in wanted.items())
    answered = final is not None and final.kind == "final" and bool(final.text.strip())
    return CaseResult(
        id=case["id"],
        question=case["question"],
        expected_tool=expected,
        called_tools=called,
        tool_correct=tool_correct,
        args_correct=args_correct,
        tool_errors=sum(1 for t in traces if t.error),
        answered=answered,
        grounded=final is not None and not final.unverified_numbers,
        seconds=round(seconds, 1),
        answer=final.text if final is not None else "",
    )


def summarize(results: list[CaseResult]) -> dict[str, Any]:
    n = len(results)

    def rate(attr: str) -> float:
        return round(sum(getattr(r, attr) for r in results) / n, 3) if n else 0.0

    return {
        "cases": n,
        "tool_selection_accuracy": rate("tool_correct"),
        "argument_accuracy": rate("args_correct"),
        "answered_rate": rate("answered"),
        "grounded_rate": rate("grounded"),
        "tool_error_cases": sum(1 for r in results if r.tool_errors),
        "median_seconds": round(statistics.median(r.seconds for r in results), 1) if n else 0.0,
    }


def run_case(case: dict[str, Any], toolbox: Toolbox, client: LlamaClient) -> CaseResult:
    started = time.perf_counter()
    final: AgentEvent | None = None
    traces: list[ToolTrace] = []
    for event in run_agent(case["question"], toolbox, client):
        if event.kind in ("final", "limit", "error"):
            final, traces = event, event.traces
    return score_case(case, traces, final, time.perf_counter() - started)


def aggregate(runs: list[list[CaseResult]]) -> dict[str, Any]:
    summaries = [summarize(r) for r in runs]
    metrics = (
        "tool_selection_accuracy",
        "argument_accuracy",
        "answered_rate",
        "grounded_rate",
        "median_seconds",
    )
    out: dict[str, Any] = {"runs": len(runs), "cases": summaries[0]["cases"]}
    for m in metrics:
        values = [s[m] for s in summaries]
        out[m] = {
            "mean": round(statistics.fmean(values), 3),
            "std": round(statistics.stdev(values), 3) if len(values) > 1 else 0.0,
            "min": min(values),
            "max": max(values),
        }
    stable = 0
    for case_runs in zip(*runs, strict=True):
        flags = {r.tool_correct and r.args_correct for r in case_runs}
        stable += len(flags) == 1
    out["stable_cases"] = stable
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate tool selection against a live LLM")
    parser.add_argument("--cases", default="evals/tool_selection.json")
    parser.add_argument("--output", default=None)
    parser.add_argument("--sample-size", type=int, default=100_000)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--min-tool-accuracy", type=float, default=None)
    parser.add_argument("--min-answered", type=float, default=None)
    args = parser.parse_args()
    settings = Settings.from_env()
    client = LlamaClient(settings)
    if not client.is_available():
        raise SystemExit(f"LLM server not reachable at {settings.base_url}")
    if args.synthetic:
        df = generate_synthetic(args.sample_size)
    else:
        df = load_criteo_sample(args.sample_size)
    toolbox = Toolbox(df)
    cases = json.loads(Path(args.cases).read_text())
    runs: list[list[CaseResult]] = []
    for number in range(1, args.runs + 1):
        results: list[CaseResult] = []
        for case in cases:
            result = run_case(case, toolbox, client)
            results.append(result)
            mark = "ok " if result.tool_correct and result.args_correct else "BAD"
            print(
                f"run {number} {mark} {result.id:<10} {result.called_tools} {result.seconds}s",
                flush=True,
            )
        runs.append(results)
        print(json.dumps(summarize(results)), flush=True)
    report = {
        "model": Path(settings.model_path).name,
        "data_source": df.attrs.get("data_source"),
        "sample_size": len(df),
        "aggregate": aggregate(runs),
        "summaries": [summarize(r) for r in runs],
        "runs": [[asdict(r) for r in results] for results in runs],
    }
    print(json.dumps(report["aggregate"], indent=2))
    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2))
    aggregate_report = report["aggregate"]
    failed = []
    tool_mean = aggregate_report["tool_selection_accuracy"]["mean"]
    if args.min_tool_accuracy is not None and tool_mean < args.min_tool_accuracy:
        failed.append("tool selection accuracy")
    answered_mean = aggregate_report["answered_rate"]["mean"]
    if args.min_answered is not None and answered_mean < args.min_answered:
        failed.append("answered rate")
    if failed:
        raise SystemExit(f"Evaluation gate failed: {', '.join(failed)}")


if __name__ == "__main__":
    main()
