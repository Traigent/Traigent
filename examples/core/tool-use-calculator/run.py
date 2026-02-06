#!/usr/bin/env python3
"""Math QA with tool-use toggle via parameter injection and parallel config."""

from __future__ import annotations

import asyncio
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

from traigent.api.types import OptimizationResult  # noqa: E402

os.environ.setdefault("TRAIGENT_COST_APPROVED", "true")


DATA_ROOT = Path(__file__).resolve().parents[2] / "datasets" / "tool-use-calculator"
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

    config_cols = ["use_tool", "max_tool_calls", "temperature"]

    df = result.to_aggregated_dataframe(primary_objective=primary)
    preferred_cols = [
        "use_tool",
        "max_tool_calls",
        "temperature",
        "samples_count",
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
        "use_tool",
        "max_tool_calls",
        "temperature",
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


def _calc(expr: str, max_calls: int) -> str:
    try:
        # very restricted eval context
        safe = {"__builtins__": {}}
        val = eval(expr, safe, {})  # noqa: S307 (controlled input in examples)
        return str(val)
    except Exception:
        return ""


@traigent.optimize(
    eval_dataset=DATASET,
    objectives=["accuracy"],
    configuration_space={
        "use_tool": [True, False],
        "max_tool_calls": [1, 2],
        "temperature": [0.0, 0.2],
    },
    injection_mode="parameter",
    config_param="config",
    execution_mode="edge_analytics",
)
def solve_math(expr: str, config: dict | None = None) -> str:
    if MOCK:
        cfg = config or {}
        if cfg.get("use_tool", False):
            return _calc(expr, int(cfg.get("max_tool_calls", 1))) or "0"
        # no-tool path: still evaluate simply for demo determinism
        return _calc(expr, 0) or "0"
    assert os.getenv("ANTHROPIC_API_KEY"), "Missing ANTHROPIC_API_KEY"
    cfg = config or {}
    tool_hint = ""
    if cfg.get("use_tool", False):
        tool = _calc(expr, int(cfg.get("max_tool_calls", 1)))
        if tool:
            tool_hint = f"Calculator result: {tool}. Prefer this if consistent.\n\n"
    prompt = f"Expression: {expr}\n\n{tool_hint}{_PROMPT}"
    response = ChatAnthropic(
        model_name="claude-3-5-sonnet-20241022",
        temperature=float(cfg.get("temperature", 0.0)),
        timeout=None,
        stop=None,
    ).invoke([HumanMessage(content=prompt)])
    raw = str(response.content).strip()
    # normalize to leading number
    num = "".join(ch for ch in raw if ch.isdigit())
    return num or raw.split()[0][:16]


if __name__ == "__main__":
    try:
        print(
            "Tired of arithmetic slips? Toggle tool use and watch math answers stabilize."
        )

        async def main() -> None:
            workers = 2 if not MOCK else 1
            trials = 8 if not MOCK else 4
            parallel_config = {
                "trial_concurrency": 2 if workers > 1 else 1,
                "thread_workers": workers,
            }
            r = await solve_math.optimize(
                max_trials=trials,
                algorithm="random",
                parallel_config=parallel_config,
            )
            print({"best_config": r.best_config, "best_score": r.best_score})
            _print_results(r)

        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nCancelled by user.")
        raise SystemExit(130) from None
