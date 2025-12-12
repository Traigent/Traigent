#!/usr/bin/env python3
"""Test script to verify custom evaluator metrics are properly stored in database."""

import asyncio
import json
import logging
import tempfile
from collections.abc import Callable
from typing import Any

import traigent
from traigent.api.types import ExampleResult
from traigent.evaluators.base import EvaluationExample

# Enable debug logging to see the flow
logging.basicConfig(
    level=logging.DEBUG, format="%(name)s - %(levelname)s - %(message)s"
)


def create_test_dataset() -> str:
    """Create a test dataset file."""
    data = [
        {"input": {"x": 2, "y": 3}, "output": 5},
        {"input": {"x": 5, "y": 7}, "output": 12},
        {"input": {"x": 10, "y": 20}, "output": 30},
    ]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for item in data:
            json.dump(item, f)
            f.write("\n")
        return f.name


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
    objectives=["accuracy"],
    configuration_space={"multiplier": [0.5, 1.0, 1.5, 2.0]},
    execution_mode="edge_analytics",
)
def test_function(x: int, y: int, multiplier: float = 1.0) -> float:
    """Test function that adds two numbers and multiplies by a factor."""
    return (x + y) * multiplier


async def main():
    """Run the test."""
    print("🧪 Testing Custom Evaluator Metrics Storage Fix")
    print("=" * 60)

    print("\n📊 Running optimization with custom evaluator...")
    print("-" * 50)

    result = await test_function.optimize(
        algorithm="grid", max_trials=4, custom_evaluator=custom_evaluator
    )

    print("\n✅ Optimization completed")
    print(f"Best config: {result.best_config}")
    print(f"Best score: {result.best_score:.3f}")
    print(f"Best metrics: {result.best_metrics}")

    print("\n📊 All trial metrics:")
    for i, trial in enumerate(result.trials):
        print(f"  Trial {i}: config={trial.config}, metrics={trial.metrics}")

    print("\n" + "=" * 60)
    print("💡 Now check the database with:")
    print("psql postgresql://optigen:optigen_local@localhost:5432/optigen \\")
    print(
        '  -c "SELECT id, measures FROM configuration_runs ORDER BY created_at DESC LIMIT 1;"'
    )
    print("\nThe measures should contain non-zero accuracy values!")


if __name__ == "__main__":
    asyncio.run(main())
