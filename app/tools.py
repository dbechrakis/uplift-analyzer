from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

import pandas as pd

from core.adjusted import regression_adjusted_effect
from core.balance import covariate_balance
from core.cace import complier_effect
from core.data import FEATURES, OUTCOMES, describe
from core.learners import compare_uplift_learners
from core.metrics import average_treatment_effect, relative_uplift
from core.policy import targeting_policy
from core.power import required_sample_size
from core.segments import uplift_by_segment
from core.srm import CRITEO_DESIGN_TREATMENT_SHARE, sample_ratio_check
from core.uplift_model import qini_analysis

CURVE_KEYS = ("curve_x", "curve_qini", "curve_random", "points")
QINI_BOOTSTRAP = 100
BALANCE_BOOTSTRAP = 50

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
        "description": "Absolute difference in rates (treated minus control) with CI and p-value.",
        "properties": {"outcome": OUTCOME_PARAM},
        "required": ["outcome"],
    },
    "relative_uplift": {
        "description": "Percentage change versus the control rate, with a confidence interval.",
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
    "covariate_balance": {
        "description": "Check that treated and control users look alike on the features.",
        "properties": {},
        "required": [],
    },
    "regression_adjusted_effect": {
        "description": "Treatment effect adjusted for the features (CUPED-style), tighter CI.",
        "properties": {"outcome": OUTCOME_PARAM},
        "required": ["outcome"],
    },
    "complier_effect": {
        "description": "Effect on users who actually saw the ad (CACE) versus the intent-to-treat.",
        "properties": {"outcome": {**OUTCOME_PARAM, "enum": ["conversion", "visit"]}},
        "required": ["outcome"],
    },
    "qini_analysis": {
        "description": "Fit an uplift model and report Qini curve, AUUC and a bootstrap interval.",
        "properties": {"outcome": OUTCOME_PARAM},
        "required": ["outcome"],
    },
    "targeting_policy": {
        "description": "Best share of the audience to target and its profit, given a value per "
        "conversion and a cost per contact (scenario inputs).",
        "properties": {
            "outcome": OUTCOME_PARAM,
            "value_per_conversion": {"type": "number", "minimum": 0.01, "maximum": 1000000},
            "cost_per_contact": {"type": "number", "minimum": 0, "maximum": 10000},
            "audience_size": {"type": "integer", "minimum": 1000, "maximum": 1000000000},
        },
        "required": ["outcome"],
    },
    "required_sample_size": {
        "description": "Users needed to detect a relative lift, with the control rate as baseline.",
        "properties": {
            "outcome": OUTCOME_PARAM,
            "relative_lift": {"type": "number", "minimum": 0.001, "maximum": 10},
            "power": {"type": "number", "minimum": 0.5, "maximum": 0.99},
            "treated_share": {"type": "number", "minimum": 0.05, "maximum": 0.95},
        },
        "required": ["outcome", "relative_lift"],
    },
    "compare_uplift_learners": {
        "description": "Compare four uplift learners on the same holdout (UI only).",
        "properties": {"outcome": OUTCOME_PARAM},
        "required": ["outcome"],
        "expose_to_llm": False,
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
        if definition.get("expose_to_llm", True)
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
        if name == "covariate_balance":
            return covariate_balance(
                df, n_bootstrap=BALANCE_BOOTSTRAP, data_source=source
            ).to_dict()
        if name == "regression_adjusted_effect":
            return regression_adjusted_effect(df, args["outcome"], data_source=source).to_dict()
        if name == "complier_effect":
            return complier_effect(df, args["outcome"], data_source=source).to_dict()
        if name == "targeting_policy":
            return targeting_policy(
                df,
                args["outcome"],
                audience_size=args.get("audience_size", 1_000_000),
                value_per_conversion=args.get("value_per_conversion", 30.0),
                cost_per_contact=args.get("cost_per_contact", 0.05),
                n_bootstrap=QINI_BOOTSTRAP,
                data_source=source,
            ).to_dict()
        if name == "required_sample_size":
            control = df.loc[df["treatment"] == 0, args["outcome"]]
            baseline = float(control.mean())
            result = required_sample_size(
                baseline,
                relative_lift=args["relative_lift"],
                power=args.get("power", 0.8),
                treated_share=args.get("treated_share", 0.5),
            )
            return {
                **result.to_dict(),
                "outcome": args["outcome"],
                "data_source": source,
                "baseline_source": "control-group rate in the loaded sample",
            }
        if name == "compare_uplift_learners":
            return compare_uplift_learners(
                df, args["outcome"], n_bootstrap=QINI_BOOTSTRAP, data_source=source
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
