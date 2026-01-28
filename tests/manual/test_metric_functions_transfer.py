"""Targeted test to verify metric_functions are transferred to example_results.

This test isolates the bug where custom metric functions are not being
transferred from example_metric.custom_metrics to example_results[i].metrics.
"""

import asyncio

from traigent.evaluators.base import Dataset, EvaluationExample
from traigent.evaluators.local import LocalEvaluator


def accuracy_metric(output: str, expected: str) -> float:
    """Simple accuracy metric."""
    return 1.0 if str(output).lower() == str(expected).lower() else 0.0


def cost_metric(output: str, expected: str) -> float:
    """Simulated cost metric."""
    return len(output) * 0.001  # Cost based on output length


async def main():
    """Run the test."""
    # Create a simple dataset
    examples = [
        EvaluationExample(
            input_data={"question": "What is 2+2?"},
            expected_output="4",
        ),
        EvaluationExample(
            input_data={"question": "What is the capital of France?"},
            expected_output="Paris",
        ),
    ]
    dataset = Dataset(examples=examples)

    # Simple function that returns the expected output
    def simple_func(question: str) -> str:
        answers = {
            "What is 2+2?": "4",
            "What is the capital of France?": "Paris",
        }
        return answers.get(question, "unknown")

    # Create evaluator with metric_functions
    metric_functions = {
        "accuracy": accuracy_metric,
        "total_cost": cost_metric,
    }

    evaluator = LocalEvaluator(
        metrics=["accuracy"],
        timeout=60.0,
        max_workers=1,
        detailed=True,  # CRITICAL: Must be True for example_results
        execution_mode="edge_analytics",
        privacy_enabled=False,
        mock_mode_config={"enabled": False},
        metric_functions=metric_functions,
    )

    print(f"Evaluator detailed mode: {evaluator.detailed}")
    print(f"Evaluator metric_functions: {evaluator.metric_functions}")

    # Run evaluation
    result = await evaluator.evaluate(
        func=simple_func,
        config={"model": "test"},
        dataset=dataset,
    )

    print("\n=== Evaluation Result ===")
    print(f"Aggregated metrics: {result.aggregated_metrics}")
    print(f"Number of example_results: {len(result.example_results)}")

    # Check example_results
    for i, ex_result in enumerate(result.example_results):
        print(f"\n--- Example {i} ---")
        print(f"  Input: {ex_result.input_data}")
        print(f"  Expected: {ex_result.expected_output}")
        print(f"  Actual: {ex_result.actual_output}")
        print(f"  Success: {ex_result.success}")
        print(f"  Metrics: {ex_result.metrics}")

        # Check if custom metrics are present
        if "accuracy" in ex_result.metrics:
            print(f"  ✓ 'accuracy' metric present: {ex_result.metrics['accuracy']}")
        else:
            print("  ✗ 'accuracy' metric MISSING!")

        if "total_cost" in ex_result.metrics:
            print(f"  ✓ 'total_cost' metric present: {ex_result.metrics['total_cost']}")
        else:
            print("  ✗ 'total_cost' metric MISSING!")

    # Summary
    print("\n=== Summary ===")
    all_metrics_present = all(
        "accuracy" in ex.metrics and "total_cost" in ex.metrics
        for ex in result.example_results
    )
    if all_metrics_present:
        print("✓ All custom metrics successfully transferred to example_results!")
    else:
        print("✗ BUG CONFIRMED: Custom metrics NOT transferred to example_results!")
        print("\nThis confirms the bug is in the SDK collection modules, not mocking.")

    # Now test the metadata builder path
    print("\n=== Testing Metadata Builder Path ===")
    from datetime import UTC, datetime

    from traigent.api.types import TrialResult, TrialStatus
    from traigent.config.types import TraigentConfig
    from traigent.core.metadata_helpers import build_backend_metadata

    # Create a mock trial result with example_results
    trial_result = TrialResult(
        trial_id="test-trial-001",
        config={"model": "test"},
        metrics=result.aggregated_metrics,
        status=TrialStatus.COMPLETED,
        duration=1.0,
        timestamp=datetime.now(UTC),
        metadata={"example_results": result.example_results},
    )

    # Build metadata like the backend submission code does
    traigent_config = TraigentConfig(execution_mode="edge_analytics")
    trial_metadata = build_backend_metadata(
        trial_result=trial_result,
        primary_objective="accuracy",
        traigent_config=traigent_config,
        dataset_name="test_dataset",
    )

    print(f"Trial metadata keys: {trial_metadata.keys()}")
    if "measures" in trial_metadata:
        print(f"Number of measures: {len(trial_metadata['measures'])}")
        for i, measure in enumerate(trial_metadata["measures"]):
            print(f"\n  Measure {i}: {measure}")
    else:
        print("✗ 'measures' key MISSING from trial_metadata!")
        print("  This is where the bug is - measures not being built.")


if __name__ == "__main__":
    asyncio.run(main())
