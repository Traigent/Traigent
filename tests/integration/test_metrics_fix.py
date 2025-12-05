#!/usr/bin/env python3
"""Test script to verify that metrics are properly passed through to backend."""

import asyncio
import json
import tempfile
from collections.abc import Callable
from typing import Any
from unittest.mock import MagicMock, patch

import traigent
from traigent.api.types import ExampleResult
from traigent.evaluators.base import EvaluationExample


def create_test_dataset() -> str:
    """Create a test dataset file."""
    data = [
        {"input": {"x": 2, "y": 3}, "output": 5},
    ]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for item in data:
            json.dump(item, f)
            f.write("\n")
        return f.name


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
    objectives=["accuracy"],
    configuration_space={"multiplier": [1.0]},
    execution_mode="edge_analytics",
)
def test_function(x: int, y: int, multiplier: float = 1.0) -> float:
    """Test function that adds two numbers and multiplies by a factor."""
    return (x + y) * multiplier


async def main():
    """Run the test and verify metrics are properly captured."""
    print("🧪 Testing Metrics Fix")
    print("=" * 60)

    # Mock the backend client to capture what's being sent
    with patch(
        "traigent.cloud.backend_client.BackendIntegratedClient"
    ) as MockBackendClient:
        mock_client = MagicMock()
        MockBackendClient.return_value = mock_client

        # Mock the submit_result method to capture calls
        submitted_results = []

        def capture_submit(session_id, config, score, metadata):
            submitted_results.append(
                {
                    "session_id": session_id,
                    "config": config,
                    "score": score,
                    "metadata": metadata,
                }
            )
            return {"status": "success"}

        mock_client.submit_result = capture_submit
        mock_client.is_enabled = lambda: True
        mock_client.create_session = lambda **kwargs: "test_session_123"

        # Run optimization
        print("\n📊 Running optimization with custom evaluator...")
        result = await test_function.optimize(
            algorithm="grid", max_trials=1, custom_evaluator=custom_evaluator
        )

        print("\n✅ Optimization completed")
        print(f"Best config: {result.best_config}")
        print(f"Best score: {result.best_score:.3f}")

        # Check what was sent to the backend
        if submitted_results:
            print("\n📤 Data sent to backend:")
            for i, submission in enumerate(submitted_results):
                print(f"\n  Submission {i+1}:")
                print(f"    Config: {submission['config']}")
                print(f"    Score: {submission['score']}")

                # Check measures
                if "measures" in submission["metadata"]:
                    measures = submission["metadata"]["measures"]
                    print(f"    Measures ({len(measures)} examples):")
                    for j, measure in enumerate(measures):
                        print(f"      Example {j+1}:")
                        for key, value in measure.items():
                            print(f"        {key}: {value}")

                        # Verify expected fields are present
                        expected_fields = ["accuracy", "score", "response_time"]
                        # Note: We won't have token/cost fields without real LLM calls
                        # But they should be added when using real LLMs

                        missing_fields = [
                            f for f in expected_fields if f not in measure
                        ]
                        if missing_fields:
                            print(f"        ⚠️ Missing fields: {missing_fields}")
                        else:
                            print("        ✅ All basic fields present")

                        # Check for LLM metrics (won't be present without real LLM)
                        llm_fields = [
                            "input_tokens",
                            "output_tokens",
                            "total_tokens",
                            "input_cost",
                            "output_cost",
                            "total_cost",
                        ]
                        present_llm_fields = [f for f in llm_fields if f in measure]
                        if present_llm_fields:
                            print(
                                f"        ✅ LLM metrics present: {present_llm_fields}"
                            )
                        else:
                            print(
                                "        ℹ️ No LLM metrics (expected without real LLM calls)"
                            )
                else:
                    print("    ⚠️ No measures field in metadata")
        else:
            print("\n⚠️ No data was sent to backend")

    print("\n" + "=" * 60)
    print("💡 Summary:")
    print("- Custom evaluator returns metrics with 'accuracy' field")
    print("- CustomEvaluatorWrapper should add LLM metrics when available")
    print("- Orchestrator should include all metrics in 'measures' field")
    print(
        "- Backend expects: input_tokens, output_tokens (not prompt_tokens, completion_tokens)"
    )


if __name__ == "__main__":
    asyncio.run(main())
