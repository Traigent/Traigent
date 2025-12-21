"""Tests for Haystack cost tracking module.

Coverage: Epic 4, Story 4.1 (Track Token Usage and Cost Per Run)
"""

from __future__ import annotations

import pytest

from traigent.integrations.haystack.cost_tracking import (
    CostResult,
    HaystackCostTracker,
    TokenUsage,
    extract_token_usage,
    get_cost_metrics,
)


class TestTokenUsage:
    """Tests for TokenUsage dataclass."""

    def test_default_values(self):
        """Test default values are zeros."""
        tokens = TokenUsage()
        assert tokens.input_tokens == 0
        assert tokens.output_tokens == 0
        assert tokens.total_tokens == 0
        assert tokens.model is None
        assert tokens.component is None

    def test_total_computed_from_input_output(self):
        """Test total is computed if not provided."""
        tokens = TokenUsage(input_tokens=100, output_tokens=50)
        assert tokens.total_tokens == 150

    def test_explicit_total_preserved(self):
        """Test explicit total is preserved."""
        tokens = TokenUsage(input_tokens=100, output_tokens=50, total_tokens=200)
        assert tokens.total_tokens == 200

    def test_addition(self):
        """Test adding two TokenUsage instances."""
        t1 = TokenUsage(input_tokens=100, output_tokens=50, model="gpt-4o")
        t2 = TokenUsage(input_tokens=80, output_tokens=40, model="gpt-4o-mini")

        result = t1 + t2
        assert result.input_tokens == 180
        assert result.output_tokens == 90
        assert result.total_tokens == 270
        # First model is preserved
        assert result.model == "gpt-4o"


class TestCostResult:
    """Tests for CostResult dataclass."""

    def test_default_values(self):
        """Test default values are zeros."""
        cost = CostResult()
        assert cost.input_cost == 0.0
        assert cost.output_cost == 0.0
        assert cost.total_cost == 0.0
        assert cost.model_used is None

    def test_addition(self):
        """Test adding two CostResult instances."""
        c1 = CostResult(input_cost=0.01, output_cost=0.02, total_cost=0.03)
        c2 = CostResult(input_cost=0.005, output_cost=0.01, total_cost=0.015)

        result = c1 + c2
        assert result.input_cost == 0.015
        assert result.output_cost == 0.03
        assert result.total_cost == 0.045


class TestExtractTokenUsage:
    """Tests for extract_token_usage function."""

    def test_openai_style_output(self):
        """Test extraction from OpenAI-style output."""
        output = {
            "llm": {
                "replies": ["Hello"],
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

        tokens = extract_token_usage(output)
        assert tokens.input_tokens == 100
        assert tokens.output_tokens == 50
        assert tokens.total_tokens == 150
        assert tokens.model == "gpt-4o"

    def test_anthropic_style_output(self):
        """Test extraction from Anthropic-style output."""
        output = {
            "llm": {
                "replies": ["Hello"],
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

        tokens = extract_token_usage(output)
        assert tokens.input_tokens == 80
        assert tokens.output_tokens == 40
        assert tokens.model == "claude-3-5-sonnet-20241022"

    def test_direct_usage_field(self):
        """Test extraction from direct usage field."""
        output = {
            "usage": {
                "prompt_tokens": 50,
                "completion_tokens": 25,
            }
        }

        tokens = extract_token_usage(output)
        assert tokens.input_tokens == 50
        assert tokens.output_tokens == 25

    def test_missing_usage(self):
        """Test graceful handling of missing usage data."""
        output = {"result": "Some value"}

        tokens = extract_token_usage(output)
        assert tokens.input_tokens == 0
        assert tokens.output_tokens == 0
        assert tokens.total_tokens == 0

    def test_none_output(self):
        """Test handling of None output."""
        tokens = extract_token_usage(None)
        assert tokens.input_tokens == 0
        assert tokens.output_tokens == 0

    def test_empty_meta_list(self):
        """Test handling of empty meta list."""
        output = {"llm": {"replies": ["Hello"], "meta": []}}

        tokens = extract_token_usage(output)
        assert tokens.input_tokens == 0

    def test_component_attribution(self):
        """Test component attribution is preserved."""
        output = {"usage": {"prompt_tokens": 10, "completion_tokens": 5}}

        tokens = extract_token_usage(output, component="generator")
        assert tokens.component == "generator"


class TestHaystackCostTracker:
    """Tests for HaystackCostTracker class."""

    def test_calculate_cost_with_model(self):
        """Test cost calculation with known model."""
        tracker = HaystackCostTracker()
        tokens = TokenUsage(input_tokens=1000, output_tokens=500, model="gpt-4o")

        cost = tracker.calculate_cost(tokens)
        # Should have non-zero costs for known model
        assert cost.total_cost > 0
        assert cost.model_used == "gpt-4o"

    def test_calculate_cost_no_model(self):
        """Test cost calculation without model returns zero."""
        tracker = HaystackCostTracker()
        tokens = TokenUsage(input_tokens=100, output_tokens=50)

        cost = tracker.calculate_cost(tokens)
        assert cost.total_cost == 0.0

    def test_calculate_cost_zero_tokens(self):
        """Test cost calculation with zero tokens."""
        tracker = HaystackCostTracker()
        tokens = TokenUsage(model="gpt-4o")

        cost = tracker.calculate_cost(tokens)
        assert cost.total_cost == 0.0
        assert cost.model_used == "gpt-4o"

    def test_default_model(self):
        """Test default model is used when token doesn't have one."""
        tracker = HaystackCostTracker(model="gpt-4o")
        tokens = TokenUsage(input_tokens=100, output_tokens=50)

        cost = tracker.calculate_cost(tokens)
        assert cost.model_used == "gpt-4o"
        assert cost.total_cost > 0

    def test_calculate_total_cost(self):
        """Test aggregating costs across multiple usages."""
        tracker = HaystackCostTracker()
        usages = [
            TokenUsage(input_tokens=100, output_tokens=50, model="gpt-4o"),
            TokenUsage(input_tokens=100, output_tokens=50, model="gpt-4o"),
        ]

        total = tracker.calculate_total_cost(usages)
        assert total.tokens.input_tokens == 200
        assert total.tokens.output_tokens == 100
        assert total.total_cost > 0

    def test_extract_and_calculate(self):
        """Test extraction and calculation in one step."""
        tracker = HaystackCostTracker()
        outputs = [
            {
                "llm": {
                    "meta": [
                        {
                            "model": "gpt-4o",
                            "usage": {"prompt_tokens": 100, "completion_tokens": 50},
                        }
                    ]
                }
            }
        ]

        cost = tracker.extract_and_calculate(outputs)
        assert cost.tokens.input_tokens == 100
        assert cost.tokens.output_tokens == 50
        assert cost.total_cost > 0


class TestGetCostMetrics:
    """Tests for get_cost_metrics function."""

    def test_all_fields_included(self):
        """Test all required fields are in metrics dict."""
        tokens = TokenUsage(input_tokens=100, output_tokens=50)
        cost = CostResult(
            input_cost=0.01,
            output_cost=0.02,
            total_cost=0.03,
            tokens=tokens,
            model_used="gpt-4o",
        )

        metrics = get_cost_metrics(cost)

        assert "total_cost" in metrics
        assert "input_cost" in metrics
        assert "output_cost" in metrics
        assert "input_tokens" in metrics
        assert "output_tokens" in metrics
        assert "total_tokens" in metrics

        assert metrics["total_cost"] == 0.03
        assert metrics["input_cost"] == 0.01
        assert metrics["output_cost"] == 0.02
        assert metrics["input_tokens"] == 100
        assert metrics["output_tokens"] == 50

    def test_compatible_with_extract_cost_from_results(self):
        """Test metrics format is compatible with orchestrator extraction."""
        # The orchestrator looks for 'total_cost' key
        cost = CostResult(total_cost=0.05)
        metrics = get_cost_metrics(cost)

        assert metrics.get("total_cost") == 0.05


class TestIntegration:
    """Integration tests for cost tracking."""

    @pytest.mark.asyncio
    async def test_evaluator_includes_cost_metrics(self):
        """Test HaystackEvaluator includes cost metrics in results."""
        from unittest.mock import MagicMock

        from traigent.integrations.haystack import EvaluationDataset, HaystackEvaluator

        # Create mock pipeline with token usage
        pipeline = MagicMock()
        pipeline.run.return_value = {
            "llm": {
                "replies": ["Response"],
                "meta": [
                    {
                        "model": "gpt-4o",
                        "usage": {"prompt_tokens": 100, "completion_tokens": 50},
                    }
                ],
            }
        }

        dataset = EvaluationDataset.from_dicts(
            [{"input": {"query": "test"}, "expected": "result"}]
        )

        evaluator = HaystackEvaluator(
            pipeline=pipeline,
            haystack_dataset=dataset,
            output_key="llm.replies",
            track_costs=True,
        )

        result = await evaluator.evaluate(
            func=pipeline.run,
            config={},
            dataset=dataset.to_core_dataset(),
        )

        # Check cost metrics are included
        assert "total_cost" in result.aggregated_metrics
        assert "input_tokens" in result.aggregated_metrics
        assert "output_tokens" in result.aggregated_metrics
        assert result.aggregated_metrics["input_tokens"] == 100
        assert result.aggregated_metrics["output_tokens"] == 50

    @pytest.mark.asyncio
    async def test_evaluator_cost_tracking_disabled(self):
        """Test cost metrics are not included when tracking disabled."""
        from unittest.mock import MagicMock

        from traigent.integrations.haystack import EvaluationDataset, HaystackEvaluator

        pipeline = MagicMock()
        pipeline.run.return_value = {"llm": {"replies": ["Response"]}}

        dataset = EvaluationDataset.from_dicts(
            [{"input": {"query": "test"}, "expected": "result"}]
        )

        evaluator = HaystackEvaluator(
            pipeline=pipeline,
            haystack_dataset=dataset,
            output_key="llm.replies",
            track_costs=False,  # Disabled
        )

        result = await evaluator.evaluate(
            func=pipeline.run,
            config={},
            dataset=dataset.to_core_dataset(),
        )

        # Check cost metrics are NOT included
        assert "total_cost" not in result.aggregated_metrics
