#!/usr/bin/env python3
"""Example: Execute Pipeline with Configuration.

This example demonstrates how to execute a Haystack pipeline with
different configurations for optimization evaluation.

Coverage: Epic 3, Stories 3.2-3.3 (Execute Pipeline with Configuration)
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock

# Suppress tracebacks in example output for cleaner demo
# (errors are still captured and reported in the results)
logging.getLogger("traigent.integrations.haystack.execution").setLevel(logging.CRITICAL)


def create_mock_pipeline():
    """Create a mock Haystack pipeline for demonstration."""
    pipeline = MagicMock()

    # Mock generator component
    generator = MagicMock()
    generator.__class__.__name__ = "OpenAIGenerator"
    generator.model = "gpt-4o"
    generator.temperature = 0.7
    generator.max_tokens = 1024

    # Mock retriever component
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
    pipeline.run.return_value = {"llm": {"replies": ["This is a test response."]}}

    return pipeline


def example_apply_config():
    """Demonstrate applying configuration to pipeline."""
    print("=" * 60)
    print("Example 1: Apply Configuration to Pipeline")
    print("=" * 60)

    from traigent.integrations.haystack import apply_config

    pipeline = create_mock_pipeline()

    # Check original values
    generator = pipeline.get_component("generator")
    print(f"\nBefore config:")
    print(f"  generator.temperature = {generator.temperature}")
    print(f"  generator.max_tokens = {generator.max_tokens}")

    # Apply new configuration
    config = {
        "generator.temperature": 0.3,
        "generator.max_tokens": 2048,
    }

    apply_config(pipeline, config)

    print(f"\nAfter config:")
    print(f"  generator.temperature = {generator.temperature}")
    print(f"  generator.max_tokens = {generator.max_tokens}")

    print("\n")


def example_execute_with_config():
    """Demonstrate executing pipeline with configuration."""
    print("=" * 60)
    print("Example 2: Execute Pipeline with Configuration")
    print("=" * 60)

    from traigent.integrations.haystack import (
        EvaluationDataset,
        execute_with_config,
    )

    pipeline = create_mock_pipeline()

    # Create evaluation dataset
    dataset = EvaluationDataset.from_dicts(
        [
            {"input": {"query": "What is AI?"}, "expected": "Artificial Intelligence"},
            {"input": {"query": "What is ML?"}, "expected": "Machine Learning"},
        ]
    )

    # Execute with configuration
    config = {
        "generator.temperature": 0.5,
        "retriever.top_k": 5,
    }

    result = execute_with_config(
        pipeline=pipeline,
        config=config,
        dataset=dataset,
        copy_pipeline=True,  # Don't mutate original
    )

    print(f"\nExecution result:")
    print(f"  Success: {result.success}")
    print(f"  Total time: {result.total_execution_time:.3f}s")
    print(f"  Examples run: {len(result.example_results)}")

    for i, example_result in enumerate(result.example_results):
        print(f"\n  Example {i + 1}:")
        print(f"    Success: {example_result.success}")
        print(f"    Time: {example_result.execution_time:.3f}s")
        if example_result.output:
            print(f"    Output: {example_result.output}")

    print("\n")


def example_error_handling():
    """Demonstrate error handling during execution."""
    print("=" * 60)
    print("Example 3: Error Handling")
    print("=" * 60)

    from traigent.integrations.haystack import (
        EvaluationDataset,
        execute_with_config,
    )

    pipeline = create_mock_pipeline()

    # Make the pipeline fail on some inputs
    call_count = [0]

    def run_with_errors(**kwargs):
        call_count[0] += 1
        if call_count[0] == 2:
            raise RuntimeError("API rate limit exceeded")
        return {"llm": {"replies": ["Response"]}}

    pipeline.run.side_effect = run_with_errors

    # Create dataset
    dataset = EvaluationDataset.from_dicts(
        [
            {"input": {"query": "Q1"}, "expected": "A1"},
            {"input": {"query": "Q2"}, "expected": "A2"},  # This will fail
            {"input": {"query": "Q3"}, "expected": "A3"},
        ]
    )

    # Execute with error handling
    result = execute_with_config(
        pipeline=pipeline,
        config={"generator.temperature": 0.5},
        dataset=dataset,
        abort_on_error=False,  # Continue on individual failures
    )

    print(f"\nExecution with errors:")
    print(f"  Overall success: {result.success}")
    print(f"  Success count: {result.success_count}")
    print(f"  Failed count: {result.failed_count}")

    for i, example_result in enumerate(result.example_results):
        status = "OK" if example_result.success else f"FAILED: {example_result.error}"
        print(f"  Example {i + 1}: {status}")

    print("\n")


def example_multiple_configs():
    """Demonstrate running with multiple configurations."""
    print("=" * 60)
    print("Example 4: Multiple Configuration Runs")
    print("=" * 60)

    from traigent.integrations.haystack import (
        EvaluationDataset,
        execute_with_config,
    )

    pipeline = create_mock_pipeline()

    # Create dataset
    dataset = EvaluationDataset.from_dicts(
        [
            {"input": {"query": "Q1"}, "expected": "A1"},
        ]
    )

    # Test multiple configurations
    configs = [
        {"generator.temperature": 0.0, "retriever.top_k": 5},
        {"generator.temperature": 0.5, "retriever.top_k": 10},
        {"generator.temperature": 1.0, "retriever.top_k": 15},
    ]

    print("\nRunning with different configurations:")
    for i, config in enumerate(configs):
        result = execute_with_config(
            pipeline=pipeline,
            config=config,
            dataset=dataset,
            copy_pipeline=True,
        )

        print(f"\n  Config {i + 1}: {config}")
        print(f"    Success: {result.success}")
        print(f"    Time: {result.total_execution_time:.3f}s")

    print("\n")


if __name__ == "__main__":
    example_apply_config()
    example_execute_with_config()
    example_error_handling()
    example_multiple_configs()

    print("All execute_with_config examples completed successfully!")
