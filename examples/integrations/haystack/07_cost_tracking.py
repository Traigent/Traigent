#!/usr/bin/env python3
"""Example: Cost Tracking for Haystack Pipelines.

This example demonstrates how to track token usage and compute costs
for Haystack pipeline evaluations.

Coverage: Epic 4, Story 4.1 (Track Token Usage and Cost Per Run)
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock


def create_mock_pipeline_with_usage():
    """Create a mock Haystack pipeline that returns token usage."""
    pipeline = MagicMock()

    generator = MagicMock()
    generator.__class__.__name__ = "OpenAIGenerator"
    generator.model = "gpt-4o"
    generator.temperature = 0.7

    def get_component(name):
        if name == "generator":
            return generator
        return None

    pipeline.get_component.side_effect = get_component

    # Mock run that returns token usage in meta (like real Haystack)
    call_count = [0]

    def run_with_usage(**kwargs):
        call_count[0] += 1
        # Simulate varying token counts
        input_tokens = 50 + call_count[0] * 10
        output_tokens = 30 + call_count[0] * 5
        return {
            "llm": {
                "replies": [f"Response {call_count[0]}"],
                "meta": [
                    {
                        "model": "gpt-4o",
                        "usage": {
                            "prompt_tokens": input_tokens,
                            "completion_tokens": output_tokens,
                            "total_tokens": input_tokens + output_tokens,
                        },
                    }
                ],
            }
        }

    pipeline.run.side_effect = run_with_usage

    return pipeline


def example_token_extraction():
    """Demonstrate extracting token usage from pipeline outputs."""
    print("=" * 60)
    print("Example 1: Token Usage Extraction")
    print("=" * 60)

    from traigent.integrations.haystack import TokenUsage, extract_token_usage

    # OpenAI-style output (Haystack OpenAIGenerator)
    openai_output = {
        "llm": {
            "replies": ["Hello, I'm an AI assistant."],
            "meta": [
                {
                    "model": "gpt-4o",
                    "usage": {
                        "prompt_tokens": 100,
                        "completion_tokens": 50,
                        "total_tokens": 150,
                    },
                }
            ],
        }
    }

    tokens = extract_token_usage(openai_output)
    print(f"\nOpenAI-style output:")
    print(f"  Input tokens: {tokens.input_tokens}")
    print(f"  Output tokens: {tokens.output_tokens}")
    print(f"  Total tokens: {tokens.total_tokens}")
    print(f"  Model: {tokens.model}")

    # Anthropic-style output (Haystack AnthropicGenerator)
    anthropic_output = {
        "llm": {
            "replies": ["Hello from Claude!"],
            "meta": [
                {
                    "model": "claude-3-5-sonnet-20241022",
                    "usage": {
                        "input_tokens": 80,
                        "output_tokens": 40,
                    },
                }
            ],
        }
    }

    tokens = extract_token_usage(anthropic_output)
    print(f"\nAnthropic-style output:")
    print(f"  Input tokens: {tokens.input_tokens}")
    print(f"  Output tokens: {tokens.output_tokens}")
    print(f"  Total tokens: {tokens.total_tokens}")
    print(f"  Model: {tokens.model}")

    # Output without usage (custom component)
    no_usage_output = {"custom": {"result": "Some value"}}
    tokens = extract_token_usage(no_usage_output)
    print(f"\nOutput without usage:")
    print(f"  Input tokens: {tokens.input_tokens}")
    print(f"  Output tokens: {tokens.output_tokens}")
    print(f"  (Zeros are returned for missing data)")

    print("\n")


def example_cost_calculation():
    """Demonstrate calculating costs from token usage."""
    print("=" * 60)
    print("Example 2: Cost Calculation")
    print("=" * 60)

    from traigent.integrations.haystack import (
        HaystackCostTracker,
        TokenUsage,
        get_cost_metrics,
    )

    # Create cost tracker
    tracker = HaystackCostTracker()

    # Calculate cost for GPT-4o usage
    tokens = TokenUsage(
        input_tokens=1000,
        output_tokens=500,
        model="gpt-4o",
    )

    cost = tracker.calculate_cost(tokens)
    print(f"\nGPT-4o (1000 in, 500 out):")
    print(f"  Input cost: ${cost.input_cost:.6f}")
    print(f"  Output cost: ${cost.output_cost:.6f}")
    print(f"  Total cost: ${cost.total_cost:.6f}")

    # Calculate for Claude
    tokens = TokenUsage(
        input_tokens=1000,
        output_tokens=500,
        model="claude-3-5-sonnet-20241022",
    )

    cost = tracker.calculate_cost(tokens)
    print(f"\nClaude 3.5 Sonnet (1000 in, 500 out):")
    print(f"  Input cost: ${cost.input_cost:.6f}")
    print(f"  Output cost: ${cost.output_cost:.6f}")
    print(f"  Total cost: ${cost.total_cost:.6f}")

    # Convert to metrics dict (for EvaluationResult)
    metrics = get_cost_metrics(cost)
    print(f"\nMetrics dict (for aggregated_metrics):")
    for key, value in metrics.items():
        print(f"  {key}: {value}")

    print("\n")


def example_aggregate_costs():
    """Demonstrate aggregating costs across multiple examples."""
    print("=" * 60)
    print("Example 3: Aggregate Costs")
    print("=" * 60)

    from traigent.integrations.haystack import HaystackCostTracker, extract_token_usage

    # Simulate multiple pipeline outputs
    outputs = [
        {
            "llm": {
                "replies": ["Response 1"],
                "meta": [
                    {
                        "model": "gpt-4o",
                        "usage": {
                            "prompt_tokens": 100,
                            "completion_tokens": 50,
                        },
                    }
                ],
            }
        },
        {
            "llm": {
                "replies": ["Response 2"],
                "meta": [
                    {
                        "model": "gpt-4o",
                        "usage": {
                            "prompt_tokens": 120,
                            "completion_tokens": 60,
                        },
                    }
                ],
            }
        },
        {
            "llm": {
                "replies": ["Response 3"],
                "meta": [
                    {
                        "model": "gpt-4o",
                        "usage": {
                            "prompt_tokens": 80,
                            "completion_tokens": 40,
                        },
                    }
                ],
            }
        },
    ]

    tracker = HaystackCostTracker()

    # Extract and calculate in one step
    total_cost = tracker.extract_and_calculate(outputs)

    print(f"\nAggregated across {len(outputs)} examples:")
    print(f"  Total input tokens: {total_cost.tokens.input_tokens}")
    print(f"  Total output tokens: {total_cost.tokens.output_tokens}")
    print(f"  Total tokens: {total_cost.tokens.total_tokens}")
    print(f"  Total cost: ${total_cost.total_cost:.6f}")

    # Calculate per-example average
    avg_cost = total_cost.total_cost / len(outputs)
    avg_tokens = total_cost.tokens.total_tokens / len(outputs)
    print(f"\nPer-example average:")
    print(f"  Average tokens: {avg_tokens:.1f}")
    print(f"  Average cost: ${avg_cost:.6f}")

    print("\n")


async def example_evaluator_with_costs():
    """Demonstrate HaystackEvaluator with cost tracking."""
    print("=" * 60)
    print("Example 4: HaystackEvaluator with Cost Tracking")
    print("=" * 60)

    from traigent.integrations.haystack import EvaluationDataset, HaystackEvaluator

    pipeline = create_mock_pipeline_with_usage()

    dataset = EvaluationDataset.from_dicts(
        [
            {"input": {"query": "What is AI?"}, "expected": "Artificial Intelligence"},
            {"input": {"query": "What is ML?"}, "expected": "Machine Learning"},
            {"input": {"query": "What is DL?"}, "expected": "Deep Learning"},
        ]
    )

    # Create evaluator with cost tracking enabled (default)
    evaluator = HaystackEvaluator(
        pipeline=pipeline,
        haystack_dataset=dataset,
        output_key="llm.replies",
        track_costs=True,  # This is the default
    )

    # Run evaluation
    result = await evaluator.evaluate(
        func=pipeline.run,
        config={"generator.temperature": 0.7},
        dataset=dataset.to_core_dataset(),
    )

    print(f"\nEvaluation completed:")
    print(f"  Examples: {result.total_examples}")
    print(f"  Duration: {result.duration:.3f}s")

    print(f"\nCost metrics included in result:")
    cost_keys = [
        "total_cost",
        "input_cost",
        "output_cost",
        "input_tokens",
        "output_tokens",
        "total_tokens",
    ]
    for key in cost_keys:
        value = result.aggregated_metrics.get(key, "N/A")
        if isinstance(value, float) and key.endswith("cost"):
            print(f"  {key}: ${value:.6f}")
        else:
            print(f"  {key}: {value}")

    print("\n")


async def example_cost_tracking_disabled():
    """Demonstrate disabling cost tracking."""
    print("=" * 60)
    print("Example 5: Cost Tracking Disabled")
    print("=" * 60)

    from traigent.integrations.haystack import EvaluationDataset, HaystackEvaluator

    pipeline = create_mock_pipeline_with_usage()

    dataset = EvaluationDataset.from_dicts(
        [{"input": {"query": "Test"}, "expected": "Result"}]
    )

    # Create evaluator with cost tracking disabled
    evaluator = HaystackEvaluator(
        pipeline=pipeline,
        haystack_dataset=dataset,
        output_key="llm.replies",
        track_costs=False,  # Disable cost tracking
    )

    result = await evaluator.evaluate(
        func=pipeline.run,
        config={"generator.temperature": 0.5},
        dataset=dataset.to_core_dataset(),
    )

    print(f"\nEvaluation without cost tracking:")
    print(f"  Examples: {result.total_examples}")

    # Check if cost metrics are present
    has_cost = "total_cost" in result.aggregated_metrics
    print(f"  Cost metrics present: {has_cost}")

    if not has_cost:
        print("  (Cost tracking was disabled)")

    print("\n")


def main():
    """Run all examples."""
    example_token_extraction()
    example_cost_calculation()
    example_aggregate_costs()
    asyncio.run(example_evaluator_with_costs())
    asyncio.run(example_cost_tracking_disabled())

    print("All cost tracking examples completed successfully!")


if __name__ == "__main__":
    main()
