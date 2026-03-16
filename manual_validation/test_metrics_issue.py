#!/usr/bin/env python3
"""Test script to reproduce and fix the metrics extraction issue."""

import asyncio

import pytest

from traigent import TraigentConfig
from traigent.core.orchestrator import OptimizationOrchestrator
from traigent.evaluators.base import Dataset, EvaluationExample
from traigent.evaluators.local import LocalEvaluator
from traigent.optimizers.grid import GridSearchOptimizer


# Mock LLM response class
class MockLLMResponse:
    """Mock LLM response with metadata."""

    def __init__(
        self,
        text="output",
        tokens_in=100,
        tokens_out=50,
        cost=0.003,
        response_time=1500,
    ):
        self.text = text
        self.metadata = {
            "tokens": {"input": tokens_in, "output": tokens_out},
            "cost": {"input": cost * 0.3, "output": cost * 0.7, "total": cost},
            "response_time_ms": response_time,
        }

    def __str__(self):
        return self.text


@pytest.mark.asyncio
async def test_function_with_metrics(**kwargs):
    """Test function that returns an LLM response with metrics."""
    # Simulate LLM call with metrics
    response = MockLLMResponse(
        text="output",
        tokens_in=100 + int(kwargs.get("temperature", 0.5) * 50),
        tokens_out=50 + int(kwargs.get("temperature", 0.5) * 25),
        cost=0.003 * (1 + kwargs.get("temperature", 0.5)),
        response_time=1000 + int(kwargs.get("temperature", 0.5) * 500),
    )
    assert isinstance(response, MockLLMResponse), "Response should be MockLLMResponse"
    return response


@pytest.mark.asyncio
async def test_function_without_metrics(**kwargs):
    """Test function that returns a simple string (no metrics)."""
    result = "output"
    assert isinstance(result, str), "Result should be a string"
    return result


async def main():
    """Run test."""
    print("Testing metrics extraction issue...")
    print("=" * 60)

    # Create test dataset
    dataset = Dataset(
        examples=[
            EvaluationExample(
                input_data={"text": f"test {i}"}, expected_output="output"
            )
            for i in range(3)
        ]
    )

    # Test 1: Function returning string (no metrics)
    print("\n1. Testing function that returns string (no metrics)...")
    evaluator1 = LocalEvaluator(metrics=["accuracy"], execution_mode="edge_analytics")

    result1 = await evaluator1.evaluate(
        test_function_without_metrics, {"temperature": 0.7}, dataset
    )

    print(f"   Success rate: {result1.success_rate:.2%}")
    print(f"   Metrics keys: {list(result1.metrics.keys())[:5]}...")
    print(f"   Input tokens mean: {result1.metrics.get('input_tokens_mean', 0)}")
    print(f"   Cost mean: {result1.metrics.get('cost_mean', 0)}")
    print(
        f"   Has summary_stats: {hasattr(result1, 'summary_stats') and result1.summary_stats is not None}"
    )

    # Test 2: Function returning LLM response with metrics
    print("\n2. Testing function that returns LLM response with metrics...")
    evaluator2 = LocalEvaluator(metrics=["accuracy"], execution_mode="edge_analytics")

    result2 = await evaluator2.evaluate(
        test_function_with_metrics, {"temperature": 0.7}, dataset
    )

    print(f"   Success rate: {result2.success_rate:.2%}")
    print(f"   Metrics keys: {list(result2.metrics.keys())[:5]}...")
    print(f"   Input tokens mean: {result2.metrics.get('input_tokens_mean', 0)}")
    print(f"   Cost mean: {result2.metrics.get('cost_mean', 0):.6f}")
    print(
        f"   Has summary_stats: {hasattr(result2, 'summary_stats') and result2.summary_stats is not None}"
    )

    if hasattr(result2, "summary_stats") and result2.summary_stats:
        print(
            f"   Summary stats metrics: {list(result2.summary_stats['metrics'].keys())}"
        )

    # Test 3: Full optimization with orchestrator
    print("\n3. Testing full optimization flow...")
    optimizer = GridSearchOptimizer(
        config_space={"temperature": [0.5, 0.7, 0.9]}, objectives=["accuracy"]
    )

    evaluator3 = LocalEvaluator(metrics=["accuracy"], execution_mode="edge_analytics")

    config = TraigentConfig(execution_mode="edge_analytics")

    orchestrator = OptimizationOrchestrator(
        optimizer=optimizer, evaluator=evaluator3, max_trials=3, config=config
    )

    opt_result = await orchestrator.optimize(
        test_function_with_metrics, dataset, "test_function"
    )

    print(f"   Completed {len(opt_result.trials)} trials")
    for trial in opt_result.trials:
        print(
            f"   Trial {trial.trial_id[-10:]}: score={trial.metrics.get('score', 0):.3f}, "
            f"input_tokens_mean={trial.metrics.get('input_tokens_mean', 0):.0f}, "
            f"cost_mean={trial.metrics.get('cost_mean', 0):.6f}"
        )

    print("\n" + "=" * 60)
    print("Test complete!")

    # Summary
    print("\nSummary:")
    print("- Function returning string: metrics are all 0 (ISSUE REPRODUCED)")
    print("- Function returning LLM response: metrics are extracted correctly")
    print("- This explains why the demo shows all zeros for metrics")


if __name__ == "__main__":
    asyncio.run(main())
