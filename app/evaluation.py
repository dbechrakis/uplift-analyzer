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
    called = [t.name for t in traces]
    if expected == ANY_TOOL:
        tool_correct = True
    elif expected is None:
        tool_correct = not called
    else:
        tool_correct = bool(called) and called[0] == expected
    args_correct = True
    wanted = case.get("expected_args") or {}
    if expected not in (None, ANY_TOOL) and wanted:
        args_correct = tool_correct and all(
            traces[0].arguments.get(k) == v for k, v in wanted.items()
        )
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate tool selection against a live LLM")
    parser.add_argument("--cases", default="evals/tool_selection.json")
    parser.add_argument("--output", default=None)
    parser.add_argument("--sample-size", type=int, default=100_000)
    parser.add_argument("--synthetic", action="store_true")
    args = parser.parse_args()
    settings = Settings.from_env()
    client = LlamaClient(settings)
    if not client.is_available():
        raise SystemExit(f"LLM server not reachable at {settings.base_url}")
    df = (
        generate_synthetic(args.sample_size)
        if args.synthetic
        else load_criteo_sample(args.sample_size)
    )
    toolbox = Toolbox(df)
    cases = json.loads(Path(args.cases).read_text())
    results: list[CaseResult] = []
    for case in cases:
        result = run_case(case, toolbox, client)
        results.append(result)
        mark = "ok " if result.tool_correct and result.args_correct else "BAD"
        print(f"{mark} {result.id:<10} {result.called_tools} {result.seconds}s", flush=True)
    report = {
        "model": Path(settings.model_path).name,
        "data_source": df.attrs.get("data_source"),
        "sample_size": len(df),
        "summary": summarize(results),
        "results": [asdict(r) for r in results],
    }
    print(json.dumps(report["summary"], indent=2))
    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
