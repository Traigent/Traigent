#!/usr/bin/env python3
"""Structured JSON extraction with custom metric (json_score)."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

MOCK = str(os.getenv("TRAIGENT_MOCK_LLM", "")).lower() in {"1", "true", "yes", "y"}
BASE = Path(__file__).parent
if MOCK:
    os.environ["HOME"] = str(BASE)
    results_dir = BASE / ".traigent_local"
    results_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TRAIGENT_RESULTS_FOLDER"] = str(results_dir)
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage

try:
    import traigent
except ImportError:  # pragma: no cover - support IDE execution paths
    import importlib

    module_path = Path(__file__).resolve()
    for depth in (2, 3):
        try:
            sys.path.append(str(module_path.parents[depth]))
        except IndexError:
            continue
    traigent = importlib.import_module("traigent")

from traigent.api.types import OptimizationResult

DATA_ROOT = Path(__file__).resolve().parents[2] / "datasets" / "structured-output-json"
if MOCK:
    try:
        traigent.initialize(execution_mode="edge_analytics")
    except Exception:
        pass
DATASET = str(DATA_ROOT / "evaluation_set.jsonl")
PROMPT_PATH = BASE / "prompt.txt"


def _prompt() -> str:
    return PROMPT_PATH.read_text().strip()


_PROMPT = _prompt()


def _print_results(result: OptimizationResult) -> None:
    """Pretty-print aggregated and raw optimization data."""

    primary = result.objectives[0] if result.objectives else None

    def _mean_response_time(meta: Any) -> float | None:
        if not isinstance(meta, dict):
            return None
        entries: list[dict[str, Any]] = []
        measures = meta.get("measures")
        if isinstance(measures, list):
            entries = [entry for entry in measures if isinstance(entry, dict)]
        elif isinstance(measures, dict):
            entries = [measures]
        if not entries:
            eval_result = meta.get("evaluation_result")
            example_results = getattr(eval_result, "example_results", None)
            if isinstance(example_results, list):
                entries = [
                    entry for entry in example_results if isinstance(entry, dict)
                ]
        times = [
            float(entry["response_time"])
            for entry in entries
            if entry.get("response_time") is not None
        ]
        if times:
            return sum(times) / len(times)
        return None

    df_raw = result.to_dataframe()
    if "metadata" in df_raw.columns:
        df_raw["avg_response_time"] = df_raw["metadata"].apply(_mean_response_time)
    else:
        df_raw["avg_response_time"] = None

    config_cols = ["temperature", "format_hint", "schema_rigidity"]

    df = result.to_aggregated_dataframe(primary_objective=primary)
    preferred_cols = [
        "temperature",
        "format_hint",
        "schema_rigidity",
        "samples_count",
        "json_score",
        "accuracy",
        "cost",
        "duration",
        "avg_response_time",
    ]
    cols = [c for c in preferred_cols if c in df.columns]
    if cols:
        df = df[cols]

    if df_raw["avg_response_time"].notna().any():
        response_avg = (
            df_raw.groupby(config_cols, dropna=False)["avg_response_time"]
            .mean()
            .reset_index()
        )
        df = df.merge(response_avg, on=config_cols, how="left")

    if "avg_response_time" in df.columns:
        df["avg_response_time"] = df["avg_response_time"].astype(float).round(3)

    if primary and primary in df.columns:
        minimize_patterns = ["cost", "latency", "error", "loss", "time", "duration"]
        ascending = any(p in primary.lower() for p in minimize_patterns)
        df = df.sort_values(by=primary, ascending=ascending, na_position="last")

    if not df.empty:
        print("\nAggregated configurations and performance:")
        print(df.to_string(index=False))

    preferred_raw = [
        "trial_id",
        "status",
        "temperature",
        "format_hint",
        "schema_rigidity",
        "json_score",
        "accuracy",
        "cost",
        "duration",
        "avg_response_time",
    ]
    cols_raw = [c for c in preferred_raw if c in df_raw.columns]
    if cols_raw:
        df_raw = df_raw[cols_raw]

    if "avg_response_time" in df_raw.columns:
        df_raw["avg_response_time"] = df_raw["avg_response_time"].astype(float).round(3)

    if primary and primary in df_raw.columns:
        minimize_patterns = ["cost", "latency", "error", "loss", "time", "duration"]
        ascending = any(p in primary.lower() for p in minimize_patterns)
        df_raw = df_raw.sort_values(by=primary, ascending=ascending, na_position="last")

    if not df_raw.empty:
        print("\nRaw (per-sample) trials:")
        print(df_raw.to_string(index=False))


def json_score_metric(output: str, expected: dict, _llm_metrics: dict | None) -> float:
    try:
        obj = json.loads(output)
    except Exception:
        return 0.0
    score = 0.0
    if isinstance(expected, dict):
        total = len(expected) or 1
        matched = 0
        for k, v in expected.items():
            if k in obj and str(obj[k]).strip().lower() == str(v).strip().lower():
                matched += 1
        score = matched / total
    return float(score)


@traigent.optimize(
    eval_dataset=DATASET,
    objectives=["json_score"],
    configuration_space={
        "temperature": [0.0, 0.2],
        "format_hint": ["strict_json", "relaxed_json"],
        "schema_rigidity": ["strict", "lenient"],
    },
    metric_functions={"json_score": json_score_metric},
    execution_mode="edge_analytics",
    injection_mode="seamless",
)
def extract_fields(
    text: str,
    temperature: float = 0.0,
    format_hint: str = "strict_json",
    schema_rigidity: str = "strict",
) -> str:
    if MOCK:
        import re

        vendor = None
        amount = None
        # Simple heuristics
        if "Acme" in text:
            vendor = "Acme Corp"
        elif "Globex" in text:
            vendor = "Globex"
        elif "MegaCo" in text:
            vendor = "MegaCo"

        amount_match = re.search(
            r"\$\s*([0-9]+(?:\.[0-9]+)?)|USD\s*([0-9]+(?:\.[0-9]+)?)", text
        )
        if amount_match:
            amount = float(amount_match.group(1) or amount_match.group(2))
        obj = {"vendor": vendor or "Unknown", "amount": amount or 0}
        return json.dumps(obj)
    assert os.getenv("ANTHROPIC_API_KEY"), "Missing ANTHROPIC_API_KEY"
    hints = f"Use {format_hint}. Schema is {schema_rigidity}."
    prompt = f"Text:\n{text}\n\n{hints}\n\n{_PROMPT}"
    response = ChatAnthropic(
        model_name="claude-3-5-sonnet-20241022",
        temperature=temperature,
        timeout=None,
        stop=None,
    ).invoke([HumanMessage(content=prompt)])
    return str(response.content).strip()


if __name__ == "__main__":
    print("Struggling to get clean, valid JSON back from messy text?")

    async def main() -> None:
        trials = 9 if not MOCK else 4
        r = await extract_fields.optimize(algorithm="grid", max_trials=trials)
        print({"best_config": r.best_config, "best_score": r.best_score})
        _print_results(r)

    asyncio.run(main())
