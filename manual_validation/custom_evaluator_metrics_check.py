#!/usr/bin/env python3
"""Manual check: custom-evaluator metrics survive the optimization pipeline.

Run it directly, from anywhere::

    python manual_validation/custom_evaluator_metrics_check.py

**What this checks.** A custom evaluator's per-example ``accuracy`` reaches
``TrialResult.metrics`` for every trial, and reaches the per-example
``measures`` array of the payload the backend receives. It exits non-zero when a
trial loses them — the earlier revision printed and exited 0 unconditionally, so
it could not fail.

**What this does NOT check.** The run is ``execution_mode="local"``: it creates
no backend session and writes no ``configuration_runs`` row. The ``psql`` recipe
printed at the end is therefore a *follow-up for a backend-tracked run*, not a
verification of this one — an earlier revision presented it as though this run
had populated that row.

It previously sat in ``tests/integration/`` named ``test_custom_evaluator_fix.py``,
where it collected zero tests (``@traigent.optimize`` returns a non-function, so
pytest skipped it with a ``PytestCollectionWarning``) while its stale
``execution_mode="edge_analytics"`` raised at import and broke suite collection.
"""

import asyncio
import json
import logging
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
from traigent.api.types import ExampleResult  # noqa: E402
from traigent.config.types import TraigentConfig  # noqa: E402
from traigent.core.metadata_helpers import build_backend_metadata  # noqa: E402
from traigent.evaluators.base import EvaluationExample  # noqa: E402

# Enable debug logging to see the flow
logging.basicConfig(
    level=logging.DEBUG, format="%(name)s - %(levelname)s - %(message)s"
)

# Written under the (gitignored) artifacts directory inside the dataset root.
_DATASET_PATH = _HERE / "_run_artifacts" / "custom_evaluator_check.jsonl"

_OBJECTIVE = "accuracy"


def create_test_dataset() -> str:
    """Write the evaluation dataset inside the dataset root and return its path."""
    data = [
        {"input": {"x": 2, "y": 3}, "output": 5},
        {"input": {"x": 5, "y": 7}, "output": 12},
        {"input": {"x": 10, "y": 20}, "output": 30},
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
    """Custom evaluator that calculates accuracy based on how close the result is."""
    x = example.input_data["x"]
    y = example.input_data["y"]
    expected = example.expected_output

    # Call the function
    result = func(x, y, **config)

    # Calculate accuracy based on how close we are to expected
    if result == expected:
        accuracy = 1.0
    else:
        # Give partial credit based on how close we are
        diff = abs(result - expected)
        max_val = max(abs(expected), 1)
        accuracy = max(0.0, 1.0 - (diff / max_val))

    print(
        f"  Evaluator: x={x}, y={y}, expected={expected}, got={result}, accuracy={accuracy:.2f}"
    )

    return ExampleResult(
        example_id=f"example_{x}_{y}",
        input_data=example.input_data,
        expected_output=expected,
        actual_output=result,
        metrics={"accuracy": accuracy},
        execution_time=0.1,
        success=True,
        error_message=None,
    )


@traigent.optimize(
    eval_dataset=create_test_dataset(),
    objectives=[_OBJECTIVE],
    configuration_space={"multiplier": [0.5, 1.0, 1.5, 2.0]},
    execution_mode="local",
)
def add_and_scale(x: int, y: int, multiplier: float = 1.0) -> float:
    """Subject under check: adds two numbers and multiplies by a factor."""
    return (x + y) * multiplier


async def main() -> int:
    """Run the check. Returns the process exit code."""
    print("🧪 Checking Custom Evaluator Metrics Storage")
    print("=" * 60)
    print(f"Dataset root: {os.environ['TRAIGENT_DATASET_ROOT']}")

    print("\n📊 Running optimization with custom evaluator...")
    print("-" * 50)

    result = await add_and_scale.optimize(
        algorithm="grid", max_trials=4, custom_evaluator=custom_evaluator
    )

    print("\n✅ Optimization completed")
    print(f"Best config: {result.best_config}")
    print(f"Best score: {result.best_score:.3f}")
    print(f"Best metrics: {result.best_metrics}")

    problems: list[str] = []
    if not result.trials:
        problems.append("the run produced no trials")

    # The same call BackendSessionManager makes before persisting/submitting a
    # trial — this is where the evaluator's per-example metrics have to land.
    config = TraigentConfig(execution_mode="local")

    print("\n📊 All trial metrics:")
    for index, trial in enumerate(result.trials):
        print(f"  Trial {index}: config={trial.config}, metrics={trial.metrics}")

        if _OBJECTIVE not in (trial.metrics or {}):
            problems.append(f"trial {index}: '{_OBJECTIVE}' missing from trial.metrics")

        metadata = build_backend_metadata(
            trial, _OBJECTIVE, config, "manual_validation"
        )
        measures = metadata.get("measures") or []
        if not measures:
            problems.append(f"trial {index}: no per-example measures in the payload")
            continue

        without_objective = [
            measure.get("example_id")
            for measure in measures
            if _OBJECTIVE not in (measure.get("metrics") or {})
        ]
        if without_objective:
            problems.append(
                f"trial {index}: examples {without_objective} carry no "
                f"'{_OBJECTIVE}' measure"
            )
        print(f"    measures: {len(measures)} example(s), all with '{_OBJECTIVE}'")

    print("\n" + "=" * 60)
    if problems:
        print("❌ Custom-evaluator metrics did NOT survive the pipeline:")
        for problem in problems:
            print(f"  - {problem}")
        return 1

    print("✅ Custom-evaluator metrics survived into every trial and measure")
    print("\n💡 For a backend-tracked run (this local one writes no row), check:")
    print("psql $DB_URL \\")
    print(
        '  -c "SELECT id, measures FROM configuration_runs ORDER BY created_at DESC LIMIT 1;"'
    )
    print("\nThe measures should contain non-zero accuracy values!")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
