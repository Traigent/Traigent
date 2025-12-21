#!/usr/bin/env python3
"""Example: HaystackEvaluator with Optimizers.

This example demonstrates how to use HaystackEvaluator with Traigent's
existing optimizers (GridSearch, RandomSearch) for pipeline optimization.

Coverage: Epic 3, Stories 3.4-3.5 (HaystackEvaluator & Optimizer Integration)
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock


def create_mock_pipeline():
    """Create a mock Haystack pipeline for demonstration."""
    pipeline = MagicMock()

    generator = MagicMock()
    generator.__class__.__name__ = "OpenAIGenerator"
    generator.model = "gpt-4o"
    generator.temperature = 0.7

    retriever = MagicMock()
    retriever.__class__.__name__ = "InMemoryBM25Retriever"
    retriever.top_k = 10

    def get_component(name):
        if name == "generator":
            return generator
        elif name == "retriever":
            return retriever
        return None

    pipeline.get_component.side_effect = get_component
    pipeline.run.return_value = {"answer": "This is a response."}

    return pipeline


async def example_evaluator_basic():
    """Demonstrate basic HaystackEvaluator usage."""
    print("=" * 60)
    print("Example 1: Basic HaystackEvaluator Usage")
    print("=" * 60)

    from traigent.integrations.haystack import (
        EvaluationDataset,
        HaystackEvaluator,
    )

    pipeline = create_mock_pipeline()

    # Create dataset
    dataset = EvaluationDataset.from_dicts(
        [
            {"input": {"query": "What is AI?"}, "expected": "Artificial Intelligence"},
            {"input": {"query": "What is ML?"}, "expected": "Machine Learning"},
            {"input": {"query": "What is DL?"}, "expected": "Deep Learning"},
        ]
    )

    # Create evaluator
    evaluator = HaystackEvaluator(
        pipeline=pipeline,
        haystack_dataset=dataset,
        metrics=["accuracy"],
        output_key="answer",  # Extract from {"answer": "..."}
    )

    # Evaluate with a configuration
    config = {"generator.temperature": 0.5, "retriever.top_k": 5}

    result = await evaluator.evaluate(
        func=pipeline.run,
        config=config,
        dataset=dataset.to_core_dataset(),
    )

    print(f"\nEvaluation result:")
    print(f"  Duration: {result.duration:.3f}s")
    print(f"  Total examples: {result.total_examples}")
    print(f"  Successful: {result.successful_examples}")
    print(f"  Metrics: {result.metrics}")

    print("\n")


async def example_with_grid_search():
    """Demonstrate HaystackEvaluator with GridSearchOptimizer."""
    print("=" * 60)
    print("Example 2: Grid Search Integration")
    print("=" * 60)

    from traigent.integrations.haystack import (
        EvaluationDataset,
        HaystackEvaluator,
    )
    from traigent.optimizers import GridSearchOptimizer

    pipeline = create_mock_pipeline()

    # Create dataset
    dataset = EvaluationDataset.from_dicts(
        [
            {"input": {"query": "Q1"}, "expected": "A1"},
            {"input": {"query": "Q2"}, "expected": "A2"},
        ]
    )

    # Create evaluator
    evaluator = HaystackEvaluator(
        pipeline=pipeline,
        haystack_dataset=dataset,
        output_key="answer",
    )

    # Create grid search optimizer
    optimizer = GridSearchOptimizer(
        config_space={
            "generator.temperature": [0.0, 0.5, 1.0],  # Discrete values only
            "retriever.top_k": [5, 10, 15],
        },
        objectives=["accuracy"],
    )

    print(f"\nGrid search configuration:")
    print(f"  Total combinations: 3 x 3 = 9")

    # Run optimization loop
    history = []
    trial_count = 0
    max_trials = 5  # Limit for demo

    while not optimizer.should_stop(history) and trial_count < max_trials:
        # Get next configuration
        config = optimizer.suggest_next_trial(history)
        if config is None:
            break

        print(f"\n  Trial {trial_count + 1}: {config}")

        # Evaluate
        result = await evaluator.evaluate(
            func=pipeline.run,
            config=config,
            dataset=dataset.to_core_dataset(),
        )

        # Track result (simplified)
        trial_result = {
            "config": config,
            "score": 0.8 + 0.1 * trial_count,  # Mock score
            "duration": result.duration,
        }
        history.append(trial_result)

        print(f"    Score: {trial_result['score']:.2f}")

        trial_count += 1

    print(f"\n  Completed {trial_count} trials")

    print("\n")


async def example_with_random_search():
    """Demonstrate HaystackEvaluator with RandomSearchOptimizer."""
    print("=" * 60)
    print("Example 3: Random Search Integration")
    print("=" * 60)

    from traigent.integrations.haystack import (
        EvaluationDataset,
        HaystackEvaluator,
    )
    from traigent.optimizers import RandomSearchOptimizer

    pipeline = create_mock_pipeline()

    # Create dataset
    dataset = EvaluationDataset.from_dicts(
        [
            {"input": {"query": "Q1"}, "expected": "A1"},
        ]
    )

    # Create evaluator
    evaluator = HaystackEvaluator(
        pipeline=pipeline,
        haystack_dataset=dataset,
        output_key="answer",
    )

    # Create random search optimizer (supports continuous ranges!)
    optimizer = RandomSearchOptimizer(
        config_space={
            "generator.temperature": (0.0, 2.0),  # Continuous range
            "retriever.top_k": [5, 10, 15, 20],  # Categorical
        },
        objectives=["accuracy"],
        max_trials=5,
        random_seed=42,  # For reproducibility
    )

    print(f"\nRandom search with seed=42:")

    history = []
    for trial_num in range(3):
        config = optimizer.suggest_next_trial(history)
        if config is None:
            break

        print(f"\n  Trial {trial_num + 1}:")
        print(f"    temperature: {config['generator.temperature']:.4f}")
        print(f"    top_k: {config['retriever.top_k']}")

        result = await evaluator.evaluate(
            func=pipeline.run,
            config=config,
            dataset=dataset.to_core_dataset(),
        )

        history.append({"config": config, "score": 0.85})

    print("\n")


async def example_with_progress_callback():
    """Demonstrate progress callback during evaluation."""
    print("=" * 60)
    print("Example 4: Progress Callback")
    print("=" * 60)

    from traigent.integrations.haystack import (
        EvaluationDataset,
        HaystackEvaluator,
    )

    pipeline = create_mock_pipeline()

    dataset = EvaluationDataset.from_dicts(
        [
            {"input": {"query": f"Question {i}"}, "expected": f"Answer {i}"}
            for i in range(5)
        ]
    )

    evaluator = HaystackEvaluator(
        pipeline=pipeline,
        haystack_dataset=dataset,
        output_key="answer",
    )

    # Define progress callback
    def on_progress(index: int, payload: dict):
        status = "OK" if payload["success"] else "FAILED"
        time_ms = payload["execution_time"] * 1000
        print(f"    Example {index + 1}/5: {status} ({time_ms:.1f}ms)")

    print("\nEvaluating with progress tracking:")

    result = await evaluator.evaluate(
        func=pipeline.run,
        config={"generator.temperature": 0.7},
        dataset=dataset.to_core_dataset(),
        progress_callback=on_progress,
    )

    print(f"\n  Total time: {result.duration:.3f}s")
    print(f"  Success rate: {result.successful_examples}/{result.total_examples}")

    print("\n")


async def example_with_custom_metrics():
    """Demonstrate custom metric functions."""
    print("=" * 60)
    print("Example 5: Custom Metrics")
    print("=" * 60)

    from traigent.integrations.haystack import (
        EvaluationDataset,
        HaystackEvaluator,
    )

    pipeline = create_mock_pipeline()

    dataset = EvaluationDataset.from_dicts(
        [
            {"input": {"query": "Q1"}, "expected": "expected output"},
            {"input": {"query": "Q2"}, "expected": "another expected"},
        ]
    )

    # Define custom metric functions
    # Note: Metric functions receive (outputs, expected_outputs, errors, **context)
    # where outputs and expected_outputs are lists
    def exact_match(outputs, expected_outputs, errors, **kwargs):
        """Check if outputs exactly match expected (averaged across examples)."""
        if not outputs:
            return 0.0
        matches = sum(1 for out, exp in zip(outputs, expected_outputs) if out == exp)
        return matches / len(outputs)

    def length_ratio(outputs, expected_outputs, errors, **kwargs):
        """Average ratio of output length to expected length."""
        if not outputs:
            return 0.0
        ratios = []
        for out, exp in zip(outputs, expected_outputs):
            if exp:
                ratios.append(min(1.0, len(str(out or "")) / len(str(exp))))
            else:
                ratios.append(0.0)
        return sum(ratios) / len(ratios) if ratios else 0.0

    # Create evaluator with custom metrics
    evaluator = HaystackEvaluator(
        pipeline=pipeline,
        haystack_dataset=dataset,
        metric_functions={
            "exact_match": exact_match,
            "length_ratio": length_ratio,
        },
        output_key="answer",
    )

    print(f"\nRegistered metrics: {evaluator.metrics}")

    result = await evaluator.evaluate(
        func=pipeline.run,
        config={"generator.temperature": 0.5},
        dataset=dataset.to_core_dataset(),
    )

    print(f"\nCustom metric results:")
    for metric_name, value in result.metrics.items():
        print(f"  {metric_name}: {value}")

    print("\n")


def main():
    """Run all examples."""
    asyncio.run(example_evaluator_basic())
    asyncio.run(example_with_grid_search())
    asyncio.run(example_with_random_search())
    asyncio.run(example_with_progress_callback())
    asyncio.run(example_with_custom_metrics())

    print("All HaystackEvaluator examples completed successfully!")


if __name__ == "__main__":
    main()
