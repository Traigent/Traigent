#!/usr/bin/env python3
"""Email drafting style/tone optimization (style, tone, temperature)."""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from typing import Any

MOCK = str(os.getenv("TRAIGENT_MOCK_MODE", "")).lower() in {"1", "true", "yes", "y"}
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

DATA_ROOT = (
    Path(__file__).resolve().parents[2] / "datasets" / "prompt-style-optimization"
)
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

    config_cols = ["style", "tone", "temperature"]

    df = result.to_aggregated_dataframe(primary_objective=primary)
    preferred_cols = [
        "style",
        "tone",
        "temperature",
        "samples_count",
        "style_accuracy",
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
        "style",
        "tone",
        "temperature",
        "style_accuracy",
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


def style_accuracy_metric(
    output: str, expected: str, _llm_metrics: dict | None
) -> float:
    exp_style, exp_tone = expected.split(",")
    ok_style = ("- " in output) if exp_style == "bulleted" else ("- " not in output)
    ok_tone = (
        ("Dear" in output)
        if exp_tone == "formal"
        else ("Hi" in output or "Hello" in output)
    )
    return 1.0 if (ok_style and ok_tone) else 0.0


@traigent.optimize(
    eval_dataset=DATASET,
    objectives=["style_accuracy"],
    configuration_space={
        "style": ["bulleted", "paragraph"],
        "tone": ["formal", "friendly"],
        "temperature": [0.0, 0.2],
    },
    metric_functions={"style_accuracy": style_accuracy_metric},
    execution_mode="edge_analytics",
    injection_mode="seamless",
    algorithm="bayesian",
)
def draft_email(brief: str) -> str:
    if MOCK:
        cfg = traigent.get_trial_config()
        style = cfg.get("style", "paragraph")
        tone = cfg.get("tone", "formal")
        header = "Dear team," if tone == "formal" else "Hi team,"
        if style == "bulleted":
            body = "\n- Thanks for your work\n- Next steps confirmed\n- We'll follow up soon"
        else:
            body = "\nThank you for your work and we confirm next steps. We'll follow up soon."
        return f"{header}{body}"
    assert os.getenv("ANTHROPIC_API_KEY"), "Missing ANTHROPIC_API_KEY"
    cfg = traigent.get_trial_config()
    style = cfg.get("style", "paragraph")
    tone = cfg.get("tone", "formal")
    temperature = float(cfg.get("temperature", 0.0))
    tone_prompt = (
        "Use a formal tone (start with 'Dear')."
        if tone == "formal"
        else "Use a friendly tone (start with 'Hi')."
    )
    style_prompt = (
        "Use bullet points." if style == "bulleted" else "Use a single paragraph."
    )
    prompt = f"Brief: {brief}\n\n{tone_prompt}\n{style_prompt}\n\n{_PROMPT}"
    response = ChatAnthropic(
        model_name="claude-3-5-sonnet-20241022",
        temperature=temperature,
        timeout=None,
        stop=None,
    ).invoke([HumanMessage(content=prompt)])
    return str(response.content).strip()


if __name__ == "__main__":
    print("Battling tone and style? Bullet points or narrative—let the data decide.")

    async def main() -> None:
        trials = 9 if not MOCK else 4
        r = await draft_email.optimize(max_trials=trials)
        print({"best_config": r.best_config, "best_score": r.best_score})
        _print_results(r)

    asyncio.run(main())
