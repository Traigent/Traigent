#!/usr/bin/env python3
"""Demonstration script that exercises weighted-score concurrency in TraiGent."""

from __future__ import annotations

import argparse
import asyncio
import os
import re
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
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

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
from traigent.cloud.backend_client import BackendIntegratedClient
from traigent.config.parallel import ParallelConfig
from traigent.core.objectives import ObjectiveDefinition, ObjectiveSchema
from traigent.optimizers.pruners import CeilingPruner

DATA_ROOT = (
    Path(__file__).resolve().parents[2] / "datasets" / "multi-objective-tradeoff"
)

# Setup logging to file - do this AFTER traigent import to override their config
import logging

LOG_LEVEL = os.getenv("TRAIGENT_LOG_LEVEL", "INFO").upper()
log_file = BASE / "debug_concurrency.log"

# Get root logger and add file handler
root_logger = logging.getLogger()
root_logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))

# Add file handler to root logger
file_handler = logging.FileHandler(log_file, mode="w")
file_handler.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
file_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)
root_logger.addHandler(file_handler)
logger = logging.getLogger(__name__)

print(f"Logging to {log_file} at level {LOG_LEVEL}")

DEFAULT_WEIGHTED_CONCURRENCY = int(
    os.getenv("TRAIGENT_WEIGHTED_UPDATE_CONCURRENCY", "3")
)
BackendIntegratedClient.weighted_update_concurrency = DEFAULT_WEIGHTED_CONCURRENCY
print(
    f"Weighted update concurrency set to {DEFAULT_WEIGHTED_CONCURRENCY} "
    "(override via TRAIGENT_WEIGHTED_UPDATE_CONCURRENCY)."
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


def _parse_float_env(name: str, default: float | None) -> float | None:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError:
        logger.warning("Invalid %s value %s; using %s", name, raw, default)
        return default
    return value


def _parse_int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw)
    except ValueError:
        logger.warning("Invalid %s value %s; using %s", name, raw, default)
        return default
    return value


OBJECTIVE_MODE = os.getenv("TRAIGENT_OBJECTIVE_MODE", "multi").strip().lower()

if OBJECTIVE_MODE in {"multi", "multi_objective", "both"}:
    OPTIMIZATION_OBJECTIVES = ObjectiveSchema.from_objectives(
        [
            ObjectiveDefinition(name="accuracy", orientation="maximize", weight=0.7),
            ObjectiveDefinition(name="cost", orientation="minimize", weight=0.3),
        ]
    )
elif OBJECTIVE_MODE == "cost":
    OPTIMIZATION_OBJECTIVES = ["cost"]
else:
    OPTIMIZATION_OBJECTIVES = ["accuracy"]

model_env = os.getenv("TRAIGENT_MODEL_SET")
if model_env:
    MODEL_CHOICES = [m.strip() for m in model_env.split(",") if m.strip()]
else:
    MODEL_CHOICES = [
        "openai/gpt-4o-mini",
        "openai/gpt-4.1-nano",  # Very cheap baseline
        "openai/gpt-4.1",  # Noticeably more expensive, good pruning signal
    ]

MAX_TRIALS = int(os.getenv("TRAIGENT_MAX_TRIALS", "24"))
BUDGET_LIMIT = _parse_float_env("TRAIGENT_BUDGET_LIMIT", 0.012)
PRUNER_COST_THRESHOLD = _parse_float_env("TRAIGENT_COST_THRESHOLD", 0.0015)
PRUNER_EPSILON = _parse_float_env("TRAIGENT_PRUNER_EPSILON", 1e-4) or 1e-4
PRUNER_WARMUP_STEPS = int(os.getenv("TRAIGENT_PRUNER_WARMUP_STEPS", "2"))
PRUNER_MIN_COMPLETED = int(os.getenv("TRAIGENT_PRUNER_MIN_COMPLETED", "2"))

DEFAULT_WORKERS = max(
    1, _parse_int_env("TRAIGENT_PARALLEL_WORKERS", os.cpu_count() or 4)
)
CONCURRENCY_PROFILE = os.getenv("TRAIGENT_CONCURRENCY_PROFILE", "auto").strip().lower()
TEMPERATURE_CHOICES = [0.0, 0.3, 0.6]
TOTAL_CONFIGS = len(MODEL_CHOICES) * len(TEMPERATURE_CHOICES)

if CONCURRENCY_PROFILE == "sequential":
    example_concurrency = 1
    trial_concurrency = 1
    resolved_mode = "sequential"
elif CONCURRENCY_PROFILE == "parallel":
    example_concurrency = max(
        1, _parse_int_env("TRAIGENT_EXAMPLE_CONCURRENCY", DEFAULT_WORKERS)
    )
    trial_concurrency = max(
        2,
        _parse_int_env(
            "TRAIGENT_TRIAL_CONCURRENCY", min(DEFAULT_WORKERS * 2, TOTAL_CONFIGS)
        ),
    )
    resolved_mode = "parallel"
else:
    example_concurrency = max(
        1, _parse_int_env("TRAIGENT_EXAMPLE_CONCURRENCY", DEFAULT_WORKERS)
    )
    trial_concurrency = max(
        1,
        _parse_int_env(
            "TRAIGENT_TRIAL_CONCURRENCY", min(DEFAULT_WORKERS, TOTAL_CONFIGS)
        ),
    )
    resolved_mode = "auto"

PARALLEL_CONFIG = ParallelConfig(
    mode=resolved_mode,
    trial_concurrency=trial_concurrency,
    example_concurrency=example_concurrency,
    thread_workers=DEFAULT_WORKERS,
)

traigent.configure(parallel_config=PARALLEL_CONFIG, parallel_workers=DEFAULT_WORKERS)

print(
    "Configured optimization with objectives=",
    (
        OPTIMIZATION_OBJECTIVES
        if isinstance(OPTIMIZATION_OBJECTIVES, list)
        else "accuracy+cost"
    ),
)
print(f"Model candidates: {MODEL_CHOICES}")
if BUDGET_LIMIT is not None:
    print(f"Budget limit: ${BUDGET_LIMIT:.6f}")
print(
    f"Pruner: min_completed={PRUNER_MIN_COMPLETED}, warmup_steps={PRUNER_WARMUP_STEPS}, "
    f"epsilon={PRUNER_EPSILON}, cost_threshold={PRUNER_COST_THRESHOLD}"
)
print(
    "Resolved concurrency: "
    f"profile={CONCURRENCY_PROFILE or 'auto'}, "
    f"mode={PARALLEL_CONFIG.mode or 'auto'}, "
    f"trial_concurrency={PARALLEL_CONFIG.trial_concurrency}, "
    f"example_concurrency={PARALLEL_CONFIG.example_concurrency}, "
    f"thread_workers={PARALLEL_CONFIG.thread_workers}"
)
if (
    PARALLEL_CONFIG.mode == "parallel"
    and (PARALLEL_CONFIG.trial_concurrency or 0)
    * (PARALLEL_CONFIG.example_concurrency or 0)
    > 8
):
    print(
        "⚠️  High concurrency requested — provider throttling may increase per-example latency."
    )


def _truncate(text: str | None, width: int = 48) -> str:
    """Trim long strings for debug dumps."""

    if not text:
        return ""
    shortened = text.strip().replace("\n", " ")
    if len(shortened) <= width:
        return shortened
    return f"{shortened[: width - 3]}..."


def _extract_expression(question: str) -> str:
    """Remove leading instruction text and trailing punctuation to get the math expression."""

    stripped = question.strip()
    stripped = stripped.rstrip(".?!")

    prefixes = [
        "compute ",
        "what is ",
        "evaluate ",
        "simplify ",
        "calculate ",
        "find the value of ",
    ]

    lowered = stripped.lower()
    for prefix in prefixes:
        if lowered.startswith(prefix):
            return stripped[len(prefix) :].strip()
    return stripped


def _evaluate_expression(question: str) -> str:
    """Evaluate the mathematical expression embedded in the question."""

    expr = _extract_expression(question)
    expr = expr.replace("^", "**")

    safe_globals = {"__builtins__": {}, "pow": pow, "bin": bin, "int": int}

    try:
        value = eval(
            expr, safe_globals, {}
        )  # noqa: S307 - controlled input for examples
    except Exception as exc:  # pragma: no cover - defensive fallback
        raise ValueError(f"Failed to evaluate expression '{expr}': {exc}") from exc

    if isinstance(value, bool):
        value = int(value)
    if isinstance(value, float):
        value = int(round(value))
    elif not isinstance(value, int):
        value = int(value)

    return str(int(value))


def _normalize_model_output(raw: str) -> str:
    """Best-effort extraction of the first integer from the model response."""

    text = raw.strip()

    if not text:
        return ""

    # Prefer explicit "Answer is/Answer:" patterns anywhere in the response
    match = re.search(r"answer[^\d-]*(?:is|:)?\s*(-?\d+)", text, re.IGNORECASE)
    if match:
        return match.group(1)

    equals_hits = re.findall(r"=\s*(-?\d+)", text)
    if equals_hits:
        return equals_hits[-1]

    # Strip enumerated list prefixes so we do not capture step numbers like "5.".
    text_without_lists = re.sub(r"(?m)^\s*\(?\d+[\).]\s+", "", text)

    all_ints = re.findall(r"-?\d+", text_without_lists)
    if all_ints:
        return all_ints[-1]

    # Fall back to the first non-empty line (truncated) if no numbers were found
    first_line = next((line.strip() for line in text.splitlines() if line.strip()), "")
    return first_line[:64]


def _print_results(result: OptimizationResult) -> None:
    """Pretty-print aggregated and raw optimization data."""
    raw = result.to_dataframe()

    if raw is not None and not raw.empty:
        print("\n=== Raw trial results ===")
        print(raw.to_string(index=False))

    aggregated = None
    if raw is not None and not raw.empty:
        raw = raw.copy()
        config_cols = [
            col for col in ["model", "temperature", "max_tokens"] if col in raw.columns
        ]
        if config_cols:
            agg_spec: dict[str, tuple[str, str]] = {
                "samples_count": ("trial_id", "count")
            }
            for col in ["accuracy", "cost", "duration", "avg_response_time"]:
                if col in raw.columns:
                    agg_spec[col] = (col, "mean")
            if "examples_attempted" in raw.columns:
                agg_spec["examples_attempted"] = ("examples_attempted", "sum")
            if "total_cost" in raw.columns:
                agg_spec["total_cost"] = ("total_cost", "sum")

            aggregated = (
                raw.groupby(config_cols, dropna=False).agg(**agg_spec).reset_index()
            )

    if aggregated is not None and not aggregated.empty:
        display_cols = [
            "model",
            "temperature",
            "max_tokens",
            "samples_count",
            "examples_attempted",
            "accuracy",
            "cost",
            "total_cost",
            "duration",
            "avg_response_time",
        ]
        cols = [c for c in display_cols if c in aggregated.columns]
        table = aggregated[cols] if cols else aggregated
        if "avg_response_time" in table.columns:
            table.loc[:, "avg_response_time"] = (
                table["avg_response_time"].astype(float).round(3)
            )
        if "duration" in table.columns:
            table.loc[:, "duration"] = table["duration"].astype(float).round(3)
        if "total_cost" in table.columns:
            table.loc[:, "total_cost"] = table["total_cost"].astype(float).round(6)
        print("\nAggregated configurations and performance:")
        print(table.to_string(index=False))

    if raw is not None and not raw.empty:
        preferred_raw = [
            "trial_id",
            "status",
            "model",
            "temperature",
            "max_tokens",
            "accuracy",
            "cost",
            "total_cost",
            "examples_attempted",
            "duration",
            "avg_response_time",
        ]
        cols_raw = [c for c in preferred_raw if c in raw.columns]
        table_raw = raw[cols_raw] if cols_raw else raw
        if "avg_response_time" in table_raw.columns:
            table_raw.loc[:, "avg_response_time"] = (
                table_raw["avg_response_time"].astype(float).round(3)
            )
        if "duration" in table_raw.columns:
            table_raw.loc[:, "duration"] = table_raw["duration"].astype(float).round(3)
        if "total_cost" in table_raw.columns:
            table_raw.loc[:, "total_cost"] = (
                table_raw["total_cost"].astype(float).round(6)
            )
        print("\nRaw (per-sample) trials:")
        print(table_raw.to_string(index=False))

    summary = result.get_summary()
    print("\nOptimization summary:")
    total_cost = float(summary.get("total_cost", 0.0) or 0.0)
    total_duration = float(summary.get("total_duration", 0.0) or 0.0)
    total_examples = int(summary.get("total_examples_attempted", 0) or 0)
    print(f"  total_trials: {summary.get('total_trials', 0)}")
    print(f"    completed: {summary.get('completed_trials', 0)}")
    print(f"    pruned: {summary.get('pruned_trials', 0)}")
    print(f"    failed: {summary.get('failed_trials', 0)}")
    print(f"  total_examples_attempted: {total_examples}")
    print(f"  total_cost: ${total_cost:.6f}")
    print(f"  total_duration: {total_duration:.3f}s")

    per_model = summary.get("trials_per_model") or {}
    if per_model:
        print("  trials_per_model:")
        for model, count in per_model.items():
            print(f"    {model}: {count}")


def _dump_example_results(
    result: OptimizationResult, show_full_output: bool = False
) -> None:
    """Dump per-example metrics so latency and cost can be audited."""

    if not result.trials:
        print("\nNo trials captured.")
        return

    print("\nDetailed example metrics:")

    for trial in result.trials:
        config = trial.config or {}
        model = config.get("model", "<unknown-model>")
        print(
            f"\nTrial {trial.trial_id}: model={model} temperature={config.get('temperature')} "
            f"max_tokens={config.get('max_tokens')} status={trial.status}"
        )

        example_results = getattr(trial, "example_results", None)
        if not example_results:
            metadata = getattr(trial, "metadata", {}) or {}
            example_results = metadata.get("example_results")
            if not example_results:
                evaluation_result = metadata.get("evaluation_result")
                if evaluation_result is not None and hasattr(
                    evaluation_result, "example_results"
                ):
                    example_results = evaluation_result.example_results

        if not example_results:
            print("  (no example_results captured)")
            meta_examples = (trial.metadata or {}).get("examples_attempted")
            meta_cost = (trial.metadata or {}).get("total_example_cost")
            if meta_examples is not None or meta_cost is not None:
                total_msg = "  -> totals:"
                if meta_examples is not None:
                    total_msg += f" examples_attempted={meta_examples}"
                if meta_cost is not None:
                    total_msg += f" total_cost=${float(meta_cost):.6f}"
                print(total_msg)
            continue

        for idx, example in enumerate(example_results, start=1):
            if hasattr(example, "metrics"):
                metrics = example.metrics or {}
                execution_time = getattr(example, "execution_time", None)
                success = getattr(example, "success", None)
                actual_output = getattr(example, "actual_output", None)
            else:
                metrics = example.get("metrics", {})
                execution_time = example.get("execution_time")
                success = example.get("success")
                actual_output = example.get("actual_output")

            accuracy = metrics.get("accuracy") if isinstance(metrics, dict) else None
            total_cost = (
                metrics.get("total_cost") if isinstance(metrics, dict) else None
            )

            function_time = None
            model_response_time = None
            if isinstance(metrics, dict):
                function_time = metrics.get("function_duration")
                model_response_time = metrics.get("model_response_time")
                if model_response_time is None:
                    maybe_resp = metrics.get("response_time")
                    if maybe_resp is not None:
                        model_response_time = maybe_resp
                    else:
                        resp_ms = metrics.get("response_time_ms")
                        if resp_ms is not None:
                            model_response_time = float(resp_ms) / 1000.0
            if function_time is None:
                function_time = execution_time
            if model_response_time is None:
                model_response_time = function_time

            input_tokens = (
                metrics.get("input_tokens") if isinstance(metrics, dict) else None
            )
            output_tokens = (
                metrics.get("output_tokens") if isinstance(metrics, dict) else None
            )

            function_display = (
                f"{function_time:.3f}s"
                if isinstance(function_time, (int, float))
                else "?"
            )
            model_display = (
                f"{model_response_time:.3f}s"
                if isinstance(model_response_time, (int, float))
                else "?"
            )
            accuracy_display = (
                f"{accuracy:.3f}" if isinstance(accuracy, (int, float)) else "?"
            )
            cost_display = (
                f"${total_cost:.6f}" if isinstance(total_cost, (int, float)) else "?"
            )

            sample = str(actual_output) if actual_output is not None else ""
            if not show_full_output:
                sample = _truncate(sample)

            print(
                f"  Example {idx:02d}: success={success} accuracy={accuracy_display} "
                f"function_time={function_display} model_time={model_display} "
                f"input_tokens={input_tokens} output_tokens={output_tokens} cost={cost_display} output='{sample}'"
            )

        meta_examples = (trial.metadata or {}).get("examples_attempted")
        meta_cost = (trial.metadata or {}).get("total_example_cost")
        if meta_examples is not None or meta_cost is not None:
            total_msg = "  -> totals:"
            if meta_examples is not None:
                total_msg += f" examples_attempted={meta_examples}"
            if meta_cost is not None:
                total_msg += f" total_cost=${float(meta_cost):.6f}"
            print(total_msg)


def _validate_concurrency_log(expected_limit: int) -> None:
    """Surface the weighted score concurrency log entry for quick validation."""

    try:
        contents = log_file.read_text().splitlines()
    except FileNotFoundError:
        print(
            "\nWeighted update concurrency validation: log file missing. "
            f"Expected to find {log_file}."
        )
        return

    matches = [
        line
        for line in contents
        if "Updated" in line and "weighted scores" in line and "concurrency=" in line
    ]

    if matches:
        print("\nWeighted update concurrency validation:")
        print(matches[-1])
    else:
        print(
            "\nWeighted update concurrency validation: no concurrency log entry found. "
            f"Check {log_file} for full logs (expected concurrency={expected_limit})."
        )


_OPTIMIZE_KWARGS: dict[str, Any] = {
    "eval_dataset": DATASET,
    "objectives": OPTIMIZATION_OBJECTIVES,
    "configuration_space": {
        "model": MODEL_CHOICES,
        "temperature": TEMPERATURE_CHOICES,
    },
    "execution_mode": "edge_analytics",
    "injection_mode": "seamless",
    "parallel_config": PARALLEL_CONFIG,
    "algorithm": "bayesian",
    "max_trials": MAX_TRIALS,
    "pruner": CeilingPruner(
        min_completed_trials=PRUNER_MIN_COMPLETED,
        warmup_steps=PRUNER_WARMUP_STEPS,
        epsilon=PRUNER_EPSILON,
        cost_threshold=PRUNER_COST_THRESHOLD,
    ),
}

if BUDGET_LIMIT is not None and BUDGET_LIMIT > 0:
    _OPTIMIZE_KWARGS["budget_limit"] = BUDGET_LIMIT


@traigent.optimize(**_OPTIMIZE_KWARGS)
def answer(
    question: str,
    model: str = "openai/gpt-4o-mini",
    temperature: float = 0.1,
    # max_tokens: int = 128,
) -> str:
    """Simple function that answers math questions using LLM."""

    if MOCK:
        return _evaluate_expression(question)

    assert os.getenv("OPENAI_API_KEY"), "Missing OPENAI_API_KEY"

    model_name = model.replace("openai/", "") if model.startswith("openai/") else model
    prompt = f"Expression: {question}\n\n{_PROMPT}"

    response = ChatOpenAI(
        model_name=model_name,
        temperature=temperature,
        # max_tokens=max_tokens,
        timeout=None,
    ).invoke([HumanMessage(content=prompt)])

    raw = str(response.content).strip()
    return _normalize_model_output(raw)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Optuna optimization demo")
    parser.add_argument(
        "--verbose-results",
        action="store_true",
        help="Print detailed per-example outputs for each trial",
    )
    parser.add_argument(
        "--max-trials",
        type=int,
        default=None,
        help="Override the number of optimization trials to run",
    )
    parser.add_argument(
        "--budget-limit",
        type=float,
        default=None,
        help="Override the cost budget used for early stopping",
    )
    args = parser.parse_args()

    print("Crunching math expressions with Optuna…")

    async def main() -> None:
        trials = args.max_trials if args.max_trials is not None else MAX_TRIALS
        runtime_budget = (
            args.budget_limit if args.budget_limit is not None else BUDGET_LIMIT
        )

        optimize_kwargs: dict[str, Any] = {"max_trials": trials}
        if runtime_budget is not None and runtime_budget > 0:
            optimize_kwargs.update(
                {
                    "budget_limit": runtime_budget,
                    "budget_metric": "total_cost",
                }
            )

        r = await answer.optimize(**optimize_kwargs)
        print({"best_config": r.best_config, "best_score": r.best_score})
        _print_results(r)
        if args.verbose_results:
            _dump_example_results(r, show_full_output=True)
        _validate_concurrency_log(DEFAULT_WEIGHTED_CONCURRENCY)

    asyncio.run(main())
