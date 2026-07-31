#!/usr/bin/env python3
"""Manual check: per-example trial metrics reach the backend payload as ``measures``.

Run it directly, from anywhere::

    python manual_validation/backend_measures_passthrough_check.py

**What this checks.** It runs a real local optimization with a custom evaluator,
then feeds each completed ``TrialResult`` to
``traigent.core.metadata_helpers.build_backend_metadata`` — the exact producer
``BackendSessionManager._submit_trial_to_backend`` calls to build the payload it
persists locally and submits remotely. It then inspects ``metadata["measures"]``
for the per-example metrics the evaluator produced.

**What this does NOT check.** It never contacts a backend, and it does not prove
anything about the wire. An earlier revision claimed to: it patched
``traigent.cloud.backend_client.BackendIntegratedClient`` and printed whatever
that captured. Under ``execution_mode="local"`` the orchestrator never
constructs that client, so the patch target was dead, ``submitted_results`` was
always empty, and the harness printed "No data was sent to backend" and exited
0 — it could not fail. Checking the producer is what is verifiable without a
live backend, so that is what it now does, and it exits non-zero when the
producer yields no measures.

It previously sat in ``tests/integration/`` named ``test_metrics_fix.py``, where
it collected zero tests (``@traigent.optimize`` returns a non-function, so pytest
skipped it with a ``PytestCollectionWarning``) while its stale
``execution_mode="edge_analytics"`` raised at import and broke suite collection.
"""

import asyncio
import json
import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent

# Run from any working directory: make the checkout importable when Traigent is
# not pip-installed, and declare this directory as the dataset sandbox root. The
# SDK rejects datasets outside that root, so the previous `/tmp` dataset made the
# documented invocation fail with a ConfigurationError before anything ran.
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
os.environ.setdefault("TRAIGENT_DATASET_ROOT", str(_HERE))

import traigent  # noqa: E402
from traigent.api.types import ExampleResult, TrialResult  # noqa: E402
from traigent.config.types import TraigentConfig  # noqa: E402
from traigent.core.metadata_helpers import build_backend_metadata  # noqa: E402
from traigent.evaluators.base import EvaluationExample  # noqa: E402

# Written under the (gitignored) artifacts directory inside the dataset root.
_DATASET_PATH = _HERE / "_run_artifacts" / "backend_measures_check.jsonl"

_OBJECTIVE = "accuracy"

# Fields the evaluator's metrics must survive into. ``response_time`` and
# ``score`` are added by the pipeline, not by the evaluator.
_EXPECTED_MEASURE_FIELDS = ["accuracy", "score", "response_time"]

# Only present when a real LLM ran; reported, never required.
_LLM_MEASURE_FIELDS = [
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "input_cost",
    "output_cost",
    "total_cost",
]


def create_test_dataset() -> str:
    """Write the evaluation dataset inside the dataset root and return its path."""
    data = [
        {"input": {"x": 2, "y": 3}, "output": 5},
    ]

    _DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_DATASET_PATH, "w", encoding="utf-8") as handle:
        for item in data:
            json.dump(item, handle)
            handle.write("\n")
    return str(_DATASET_PATH)


def custom_evaluator(
    func: Callable, config: dict[str, Any], example: EvaluationExample
) -> ExampleResult:
    """Custom evaluator that simulates LLM metrics."""
    x = example.input_data["x"]
    y = example.input_data["y"]
    expected = example.expected_output

    # Simulate function call
    result = func(x, y, **config)

    # Calculate accuracy
    if result == expected:
        accuracy = 1.0
    else:
        diff = abs(result - expected)
        max_val = max(abs(expected), 1)
        accuracy = max(0.0, 1.0 - (diff / max_val))

    # Return result with just accuracy metric
    # The CustomEvaluatorWrapper should add the LLM metrics
    return ExampleResult(
        example_id=f"example_{x}_{y}",
        input_data=example.input_data,
        expected_output=expected,
        actual_output=result,
        metrics={"accuracy": accuracy},
        execution_time=0.1,  # This should be overridden by actual metrics
        success=True,
        error_message=None,
    )


@traigent.optimize(
    eval_dataset=create_test_dataset(),
    objectives=[_OBJECTIVE],
    configuration_space={"multiplier": [1.0]},
    execution_mode="local",
)
def add_and_scale(x: int, y: int, multiplier: float = 1.0) -> float:
    """Subject under check: adds two numbers and multiplies by a factor."""
    return (x + y) * multiplier


def _report_trial(trial: TrialResult, index: int, config: TraigentConfig) -> list[str]:
    """Print one trial's backend payload; return the problems found in it."""
    problems: list[str] = []

    metadata = build_backend_metadata(trial, _OBJECTIVE, config, "manual_validation")
    measures = metadata.get("measures")

    print(f"\n  Trial {index}:")
    print(f"    Config: {trial.config}")
    print(f"    Score: {trial.get_metric(_OBJECTIVE)}")

    if not measures:
        problems.append(
            f"trial {index}: backend metadata carries no 'measures' "
            f"(keys present: {sorted(metadata)})"
        )
        print("    ❌ No measures in the backend metadata")
        return problems

    print(f"    Measures ({len(measures)} examples):")
    for example_index, measure in enumerate(measures, start=1):
        metrics = measure.get("metrics", {})
        print(
            f"      Example {example_index} (example_id={measure.get('example_id')}):"
        )
        for key, value in metrics.items():
            print(f"        {key}: {value}")

        missing = [field for field in _EXPECTED_MEASURE_FIELDS if field not in metrics]
        if missing:
            problems.append(
                f"trial {index}, example {example_index}: missing {missing}"
            )
            print(f"        ❌ Missing fields: {missing}")
        else:
            print("        ✅ All basic fields present")

        present_llm = [field for field in _LLM_MEASURE_FIELDS if field in metrics]
        if present_llm:
            print(f"        ✅ LLM metrics present: {present_llm}")
        else:
            print("        ℹ️ No LLM metrics (expected without real LLM calls)")

    return problems


async def main() -> int:
    """Run the check. Returns the process exit code."""
    print("🧪 Checking Backend Measures Passthrough")
    print("=" * 60)
    print(f"Dataset root: {os.environ['TRAIGENT_DATASET_ROOT']}")

    print("\n📊 Running optimization with custom evaluator...")
    result = await add_and_scale.optimize(
        algorithm="grid", max_trials=1, custom_evaluator=custom_evaluator
    )

    print("\n✅ Optimization completed")
    print(f"Best config: {result.best_config}")
    print(f"Best score: {result.best_score:.3f}")

    if not result.trials:
        print("\n❌ The run produced no trials — nothing to check.")
        return 1

    # The same call BackendSessionManager makes before persisting/submitting.
    config = TraigentConfig(execution_mode="local")

    print("\n📤 Backend submission payload (built by build_backend_metadata):")
    problems: list[str] = []
    for index, trial in enumerate(result.trials, start=1):
        problems.extend(_report_trial(trial, index, config))

    print("\n" + "=" * 60)
    if problems:
        print("❌ Measures passthrough is BROKEN:")
        for problem in problems:
            print(f"  - {problem}")
        return 1

    print("💡 Summary:")
    print("- The custom evaluator returned metrics with an 'accuracy' field")
    print("- Those metrics reached metadata['measures'] for every trial/example")
    print("- LLM token/cost fields appear here only when a real LLM ran")
    print(
        "- Backend expects: input_tokens, output_tokens (not prompt_tokens, completion_tokens)"
    )
    print("\n✅ Measures passthrough OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
