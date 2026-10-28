from __future__ import annotations

import json
import re
from collections.abc import Iterator
from dataclasses import dataclass, field

from app.grounding import ungrounded_numbers
from app.llm_client import LlamaClient, LlmError, ToolCall
from app.tools import Toolbox, ToolTrace

MAX_TOOL_CALLS_PER_TURN = 4
THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)

SYSTEM_PROMPT = (
    "You are an analytics assistant for a marketing incrementality study on the Criteo uplift "
    "experiment. You NEVER compute, estimate or recall statistics yourself, and you hold no data: "
    "never ask the user for data or context, the tools already have it. To answer any question "
    "that needs a number (including row counts and base rates), call exactly the tool that "
    "computes it, wait for its result, then explain that result in plain business language. "
    "Pick the most specific tool: complier_effect for the effect on users who actually saw the "
    "ad; regression_adjusted_effect when asked to adjust or control for features; "
    "covariate_balance when asked whether groups are comparable or assignment was biased; "
    "sample_ratio_check for the treatment share; targeting_policy for which share of the "
    "audience to target or profit; required_sample_size for planning a new test; "
    "average_treatment_effect for the plain effect. "
    "Copy numbers exactly as they appear in tool results, keep the outcome and group names "
    "straight, and never combine or recompute them. Values in results are fractions: write 0.01 "
    "as 1% and a relative uplift of 0.07 as +7% (never as 0.07 times). Treatment was randomised "
    "inside the original tests, but groups are not perfectly balanced, so mention covariate "
    "adjustment when it matters and never claim more than the randomized comparison supports. "
    "If a result has data_source 'synthetic', say it is synthetic demo data. Profit results are "
    "scenarios built on assumed value and cost. If no tool can answer the question, say so "
    "briefly. Keep answers under 150 words."
)


@dataclass
class AgentEvent:
    kind: str
    text: str = ""
    call: ToolCall | None = None
    trace: ToolTrace | None = None
    traces: list[ToolTrace] = field(default_factory=list)
    unverified_numbers: list[str] = field(default_factory=list)


def clean_text(text: str) -> str:
    return THINK_BLOCK.sub("", text).strip()


def _parse_arguments(raw: str) -> dict | str:
    if not raw.strip():
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        return f"Arguments are not valid JSON: {exc.msg}"
    return parsed


def run_agent(
    question: str,
    toolbox: Toolbox,
    client: LlamaClient,
    history: list[dict] | None = None,
    max_iterations: int | None = None,
) -> Iterator[AgentEvent]:
    limit = max_iterations or client.settings.max_iterations
    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(m for m in (history or [])[-6:] if m["role"] in ("user", "assistant"))
    messages.append({"role": "user", "content": question})
    traces: list[ToolTrace] = []
    for _ in range(limit):
        text = ""
        calls: list[ToolCall] = []
        try:
            for event in client.stream_chat(messages, tools=toolbox.specs):
                if event.kind == "content":
                    text += event.text
                    yield AgentEvent("text", text=event.text)
                elif event.kind == "tool_calls":
                    calls = event.tool_calls[:MAX_TOOL_CALLS_PER_TURN]
        except LlmError as exc:
            yield AgentEvent("error", text=str(exc), traces=traces)
            return
        if not calls:
            final = clean_text(text)
            flagged = ungrounded_numbers(
                final, [t.result for t in traces if t.result], [t.arguments for t in traces]
            )
            yield AgentEvent("final", text=final, traces=traces, unverified_numbers=flagged)
            return
        messages.append(
            {
                "role": "assistant",
                "content": text or None,
                "tool_calls": [
                    {
                        "id": c.id,
                        "type": "function",
                        "function": {"name": c.name, "arguments": c.arguments_json or "{}"},
                    }
                    for c in calls
                ],
            }
        )
        for call in calls:
            yield AgentEvent("tool_call", call=call)
            arguments = _parse_arguments(call.arguments_json)
            if isinstance(arguments, str):
                trace = ToolTrace(name=call.name, arguments={}, error=arguments)
            else:
                trace = toolbox.call(call.name, arguments)
            traces.append(trace)
            yield AgentEvent("tool_result", call=call, trace=trace)
            messages.append(
                {"role": "tool", "tool_call_id": call.id, "content": trace.llm_payload()}
            )
    yield AgentEvent(
        "limit",
        text=f"Stopped after {limit} tool-calling rounds without a final answer.",
        traces=traces,
    )
