#!/usr/bin/env python3
"""Math expression evaluation with Anthropic Claude models (accuracy vs cost).

This example demonstrates:
- Multi-objective optimization balancing accuracy and cost
- Grid search algorithm
- Parallel configuration options
- Model comparison across Anthropic Claude models
"""

from __future__ import annotations

import argparse
import asyncio
import os
import re
import sys
from pathlib import Path

# Environment setup must happen before imports
MOCK = str(os.getenv("TRAIGENT_MOCK_LLM", "")).lower() in {"1", "true", "yes", "y"}
BASE = Path(__file__).parent
if MOCK:
    os.environ["HOME"] = str(BASE)
    results_dir = BASE / ".traigent_local"
    results_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TRAIGENT_RESULTS_FOLDER"] = str(results_dir)

# External dependencies
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage

# Local imports with fallback for IDE execution
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
from traigent.config.parallel import ParallelConfig
from traigent.core.objectives import ObjectiveDefinition, ObjectiveSchema

# ==============================================================================
# Configuration and Constants
# ==============================================================================

DATA_ROOT = (
    Path(__file__).resolve().parents[2] / "datasets" / "multi-objective-tradeoff"
)
DATASET = str(DATA_ROOT / "evaluation_set.jsonl")
PROMPT_PATH = BASE / "prompt.txt"

# Mock mode initialization
if MOCK:
    try:
        traigent.initialize(execution_mode="edge_analytics")
    except Exception:
        pass


# ==============================================================================
# Helper Functions
# ==============================================================================


def _prompt() -> str:
    """Load the prompt template from file."""
    return PROMPT_PATH.read_text().strip()


_PROMPT = _prompt()


def _truncate(text: str | None, width: int = 48) -> str:
    """Trim long strings for debug dumps."""
    if not text:
        return ""
    shortened = text.strip().replace("\n", " ")
    if len(shortened) <= width:
        return shortened
    return f"{shortened[: width - 3]}..."


def _parse_int_env(name: str, default: int) -> int:
    """Parse integer environment variable with fallback."""
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _extract_expression(question: str) -> str:
    """Remove leading instruction text and trailing punctuation to get the math expression."""
    stripped = question.strip().rstrip(".?!")

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

    # Convert to integer
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

    # Look for equals signs
    equals_hits = re.findall(r"=\s*(-?\d+)", text)
    if equals_hits:
        return equals_hits[-1]

    # Strip enumerated list prefixes so we do not capture step numbers like "5."
    text_without_lists = re.sub(r"(?m)^\s*\(?\d+[\).]\s+", "", text)

    # Find all integers
    all_ints = re.findall(r"-?\d+", text_without_lists)
    if all_ints:
        return all_ints[-1]

    # Fall back to the first non-empty line (truncated) if no numbers were found
    first_line = next((line.strip() for line in text.splitlines() if line.strip()), "")
    return first_line[:64]


# ==============================================================================
# Optimization Configuration
# ==============================================================================

# Objective configuration
OBJECTIVE_SCHEMA = ObjectiveSchema.from_objectives(
    [
        ObjectiveDefinition(name="accuracy", orientation="maximize", weight=0.7),
        ObjectiveDefinition(name="cost", orientation="minimize", weight=0.3),
    ]
)

# Parallel configuration
DEFAULT_WORKERS = max(1, _parse_int_env("TRAIGENT_PARALLEL_WORKERS", 1))
PARALLEL_MODE_ENV = os.getenv("TRAIGENT_PARALLEL_MODE", "sequential").strip().lower()

if PARALLEL_MODE_ENV == "parallel":
    TRIAL_CONCURRENCY = max(
        2, _parse_int_env("TRAIGENT_TRIAL_CONCURRENCY", DEFAULT_WORKERS)
    )
    EXAMPLE_CONCURRENCY = max(
        1, _parse_int_env("TRAIGENT_EXAMPLE_CONCURRENCY", DEFAULT_WORKERS)
    )
    EFFECTIVE_MODE = "parallel"
else:
    TRIAL_CONCURRENCY = 1
    EXAMPLE_CONCURRENCY = 1
    EFFECTIVE_MODE = "sequential"

GLOBAL_PARALLEL_CONFIG = ParallelConfig(
    mode=EFFECTIVE_MODE,
    trial_concurrency=TRIAL_CONCURRENCY,
    example_concurrency=EXAMPLE_CONCURRENCY,
    thread_workers=DEFAULT_WORKERS,
)

# Configure Traigent
traigent.configure(parallel_config=GLOBAL_PARALLEL_CONFIG)


# ==============================================================================
# Result Display Functions
# ==============================================================================


def _print_results(result: OptimizationResult) -> None:
    """Pretty-print aggregated and raw optimization data."""
    primary = result.objectives[0] if result.objectives else None

    # Display aggregated results
    aggregated = result.to_aggregated_dataframe(primary_objective=primary)
    if not aggregated.empty:
        display_cols = [
            "model",
            "temperature",
            "max_tokens",
            "samples_count",
            "accuracy",
            "cost",
            "duration",
            "avg_response_time",
        ]
        cols = [c for c in display_cols if c in aggregated.columns]
        table = aggregated[cols] if cols else aggregated

        # Format numeric columns
        if "avg_response_time" in table.columns:
            table.loc[:, "avg_response_time"] = (
                table["avg_response_time"].astype(float).round(3)
            )
        if "duration" in table.columns:
            table.loc[:, "duration"] = table["duration"].astype(float).round(3)

        print("\nAggregated configurations and performance:")
        print(table.to_string(index=False))

    # Display raw per-trial results
    raw = result.to_dataframe()
    if not raw.empty:
        preferred_raw = [
            "trial_id",
            "status",
            "model",
            "temperature",
            "max_tokens",
            "accuracy",
            "cost",
            "duration",
            "avg_response_time",
        ]
        cols_raw = [c for c in preferred_raw if c in raw.columns]
        table_raw = raw[cols_raw] if cols_raw else raw

        # Format numeric columns
        if "avg_response_time" in table_raw.columns:
            table_raw.loc[:, "avg_response_time"] = (
                table_raw["avg_response_time"].astype(float).round(3)
            )
        if "duration" in table_raw.columns:
            table_raw.loc[:, "duration"] = table_raw["duration"].astype(float).round(3)

        print("\nRaw (per-sample) trials:")
        print(table_raw.to_string(index=False))


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
            f"\nTrial {trial.trial_id} – model={model}, "
            f"temperature={config.get('temperature')}, "
            f"max_tokens={config.get('max_tokens')}"
        )

        # Extract example results
        example_results = (trial.metadata or {}).get("example_results") or []
        if not example_results:
            print("  (no example_results captured)")
            continue

        # Display individual example results
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

            # Extract timing metrics
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

            # Format display values
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
                f"input_tokens={input_tokens} output_tokens={output_tokens} "
                f"cost={cost_display} output='{sample}'"
            )


# ==============================================================================
# Main Function
# ==============================================================================


@traigent.optimize(
    eval_dataset=DATASET,
    objectives=OBJECTIVE_SCHEMA,
    configuration_space={
        "model": [
            # "claude-3-5-haiku-latest",
            "claude-3-7-sonnet-latest",
            "claude-3-haiku-20240307",
            # "claude-opus-4-1-20250805",
            # "claude-sonnet-4-5-20250929"
        ],
        "temperature": [0.1],
        "max_tokens": [128],
    },
    execution_mode="edge_analytics",
    injection_mode="seamless",
    parallel_config=GLOBAL_PARALLEL_CONFIG,
    algorithm="grid",
    max_trials=3,
)
def answer(
    question: str,
    model: str = "claude-3-7-sonnet-latest",
    temperature: float = 0.1,
    max_tokens: int = 128,
) -> str:
    """Simple function that answers math questions using Anthropic Claude.

    The SDK handles all error logging and metrics collection.
    """
    if MOCK:
        return _evaluate_expression(question)

    assert os.getenv("ANTHROPIC_API_KEY"), "Missing ANTHROPIC_API_KEY"

    prompt = f"Expression: {question}\n\n{_PROMPT}"

    # Simple LLM call - let the SDK handle any errors
    response = ChatAnthropic(
        model_name=model,
        temperature=temperature,
        timeout=None,
        stop=None,
    ).invoke([HumanMessage(content=prompt)])

    # Extract and normalize the output
    raw = str(response.content).strip()
    return _normalize_model_output(raw)


# ==============================================================================
# Entry Point
# ==============================================================================


async def main() -> None:
    """Run the optimization and display results."""
    parser = argparse.ArgumentParser(
        description="Run the multi-objective tradeoff demo"
    )
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
    args = parser.parse_args()

    print("Crunching math expressions while balancing accuracy and cost…")

    trials = args.max_trials if args.max_trials is not None else (10 if not MOCK else 4)
    r = await answer.optimize(max_trials=trials)

    print({"best_config": r.best_config, "best_score": r.best_score})
    _print_results(r)

    if args.verbose_results:
        _dump_example_results(r, show_full_output=True)


if __name__ == "__main__":
    asyncio.run(main())
