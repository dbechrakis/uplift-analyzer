from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

import pandas as pd

from core.data import FEATURES, OUTCOMES, describe
from core.metrics import average_treatment_effect, relative_uplift
from core.segments import uplift_by_segment
from core.srm import CRITEO_DESIGN_TREATMENT_SHARE, sample_ratio_check
from core.uplift_model import qini_analysis

CURVE_KEYS = ("curve_x", "curve_qini", "curve_random")
QINI_BOOTSTRAP = 100

OUTCOME_PARAM = {
    "type": "string",
    "enum": list(OUTCOMES),
    "description": "Outcome column: conversion (purchase), visit (site visit) or exposure.",
}

TOOL_DEFINITIONS: dict[str, dict[str, Any]] = {
    "dataset_summary": {
        "description": "Row count, treated share and base rates of the loaded sample.",
        "properties": {},
        "required": [],
    },
    "sample_ratio_check": {
        "description": "Check the treatment/control split against the design (sample ratio).",
        "properties": {
            "expected_share_treated": {
                "type": "number",
                "minimum": 0.01,
                "maximum": 0.99,
                "description": f"Designed treated share, default {CRITEO_DESIGN_TREATMENT_SHARE}.",
            }
        },
        "required": [],
    },
    "average_treatment_effect": {
        "description": "Average treatment effect (rate difference) with CI and p-value.",
        "properties": {"outcome": OUTCOME_PARAM},
        "required": ["outcome"],
    },
    "relative_uplift": {
        "description": "Relative uplift versus control with a confidence interval.",
        "properties": {"outcome": OUTCOME_PARAM},
        "required": ["outcome"],
    },
    "uplift_by_segment": {
        "description": "Treatment effect per quantile bin of one feature, FDR-adjusted.",
        "properties": {
            "feature": {"type": "string", "enum": FEATURES},
            "outcome": OUTCOME_PARAM,
            "n_bins": {"type": "integer", "minimum": 2, "maximum": 10},
        },
        "required": ["feature", "outcome"],
    },
    "qini_analysis": {
        "description": "Fit an uplift model and report Qini curve, AUUC and a bootstrap interval.",
        "properties": {"outcome": OUTCOME_PARAM},
        "required": ["outcome"],
    },
}


class ToolError(ValueError):
    pass


@dataclass
class ToolTrace:
    name: str
    arguments: dict[str, Any]
    result: dict[str, Any] | None = None
    error: str | None = None
    cached: bool = False
    duration_seconds: float = 0.0

    def llm_payload(self) -> str:
        if self.error is not None:
            return json.dumps({"error": self.error})
        assert self.result is not None
        compact = {k: v for k, v in self.result.items() if k not in CURVE_KEYS}
        if "curve_x" in self.result:
            compact["curve_points"] = len(self.result["curve_x"])
        return json.dumps(compact)

    def signature(self) -> str:
        args = ", ".join(f"{k}={v!r}" for k, v in self.arguments.items())
        return f"{self.name}({args})"


def tool_specs() -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": definition["description"],
                "parameters": {
                    "type": "object",
                    "properties": definition["properties"],
                    "required": definition["required"],
                    "additionalProperties": False,
                },
            },
        }
        for name, definition in TOOL_DEFINITIONS.items()
    ]


def validate_arguments(name: str, arguments: Any) -> dict[str, Any]:
    if name not in TOOL_DEFINITIONS:
        raise ToolError(f"Unknown tool '{name}'. Available: {sorted(TOOL_DEFINITIONS)}")
    if not isinstance(arguments, dict):
        raise ToolError("Arguments must be a JSON object")
    definition = TOOL_DEFINITIONS[name]
    properties = definition["properties"]
    extra = sorted(set(arguments) - set(properties))
    if extra:
        raise ToolError(f"Unexpected arguments: {extra}")
    missing = [k for k in definition["required"] if k not in arguments]
    if missing:
        raise ToolError(f"Missing required arguments: {missing}")
    for key, value in arguments.items():
        schema = properties[key]
        if "enum" in schema and value not in schema["enum"]:
            raise ToolError(f"'{key}' must be one of {schema['enum']}")
        if schema["type"] == "integer" and (isinstance(value, bool) or not isinstance(value, int)):
            raise ToolError(f"'{key}' must be an integer")
        if schema["type"] == "number" and (
            isinstance(value, bool) or not isinstance(value, int | float)
        ):
            raise ToolError(f"'{key}' must be a number")
        if "minimum" in schema and value < schema["minimum"]:
            raise ToolError(f"'{key}' must be at least {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            raise ToolError(f"'{key}' must be at most {schema['maximum']}")
    return dict(arguments)


class Toolbox:
    def __init__(self, df: pd.DataFrame, data_source: str | None = None) -> None:
        self.df = df
        self.data_source = data_source or df.attrs.get("data_source", "unspecified")
        self._cache: dict[str, dict[str, Any]] = {}

    @property
    def specs(self) -> list[dict]:
        return tool_specs()

    def _run(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        df, source = self.df, self.data_source
        if name == "dataset_summary":
            return {**describe(df, source).__dict__, "data_source": source}
        if name == "sample_ratio_check":
            share = args.get("expected_share_treated", CRITEO_DESIGN_TREATMENT_SHARE)
            return sample_ratio_check(df, share, data_source=source).to_dict()
        if name == "average_treatment_effect":
            return average_treatment_effect(df, args["outcome"], data_source=source).to_dict()
        if name == "relative_uplift":
            return relative_uplift(df, args["outcome"], data_source=source).to_dict()
        if name == "uplift_by_segment":
            return uplift_by_segment(
                df,
                args["feature"],
                n_bins=args.get("n_bins", 5),
                outcome=args["outcome"],
                data_source=source,
            ).to_dict()
        return qini_analysis(
            df, outcome=args["outcome"], n_bootstrap=QINI_BOOTSTRAP, data_source=source
        ).to_dict()

    def call(self, name: str, arguments: Any) -> ToolTrace:
        started = time.perf_counter()
        trace = ToolTrace(name=name, arguments=arguments if isinstance(arguments, dict) else {})
        try:
            args = validate_arguments(name, arguments)
            key = f"{name}:{json.dumps(args, sort_keys=True)}"
            if key in self._cache:
                trace.result, trace.cached = self._cache[key], True
            else:
                trace.result = self._run(name, args)
                self._cache[key] = trace.result
        except (ToolError, ValueError) as exc:
            trace.error = str(exc)
        trace.duration_seconds = time.perf_counter() - started
        return trace
