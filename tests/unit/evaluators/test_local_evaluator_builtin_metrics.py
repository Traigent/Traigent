"""Tests for LocalEvaluator built-in metrics computation with metric_functions.

Covers lines 1179-1208 in traigent/evaluators/local.py: the branching logic
that computes built-in metrics even when custom metric_functions are provided.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from traigent.evaluators.base import Dataset, EvaluationExample
from traigent.evaluators.local import LocalEvaluator


def _make_dataset() -> Dataset:
    """Create a minimal dataset for testing."""
    return Dataset(
        examples=[
            EvaluationExample(
                input_data={"text": "hello world"},
                expected_output="positive",
            ),
            EvaluationExample(
                input_data={"text": "goodbye world"},
                expected_output="negative",
            ),
        ],
        name="test_builtin_metrics",
    )


async def _simple_func(text: str) -> str:
    return "positive"


class TestBuiltinMetricsWithMetricFunctions:
    """Test the branching logic at lines 1179-1208 of local.py.

    When metric_functions is set (from scoring_function), the evaluator should
    still compute built-in metrics (from _metric_registry) that are not
    overridden by custom metric_functions.
    """

    @pytest.mark.asyncio
    async def test_metric_functions_with_builtin_metrics_computes_both(self):
        """When metric_functions overrides 'accuracy' but metrics includes
        built-in 'latency' and 'cost', those built-ins should still be
        computed via compute_metrics with metrics_override=['latency', 'cost'].

        Covers the if-branch with base_metric_names populated.
        """

        def custom_accuracy(actual, expected, **kwargs):
            return 0.99

        evaluator = LocalEvaluator(
            metrics=["accuracy", "latency", "success_rate"],
            metric_functions={"accuracy": custom_accuracy},
            detailed=True,
            execution_mode="edge_analytics",
        )
        dataset = _make_dataset()

        mock_compute = MagicMock(return_value={"latency": 0.5, "success_rate": 0.9})
        with patch.object(evaluator, "compute_metrics", mock_compute):
            result = await evaluator.evaluate(_simple_func, {}, dataset)

        # compute_metrics should have been called exactly once
        mock_compute.assert_called_once()

        # Verify metrics_override was passed with only the built-in metrics
        # (not 'accuracy', which is overridden by metric_functions)
        call_kwargs = mock_compute.call_args[1]
        assert "metrics_override" in call_kwargs
        override = call_kwargs["metrics_override"]
        assert "accuracy" not in override
        assert "latency" in override
        assert "success_rate" in override

        # self.metrics should NOT have been mutated (thread-safe)
        assert evaluator.metrics == ["accuracy", "latency", "success_rate"]

        # The mocked built-in values should appear in aggregated_metrics
        assert result.aggregated_metrics.get("latency") == 0.5

    @pytest.mark.asyncio
    async def test_metric_functions_all_custom_no_builtins(self):
        """When all metrics in self.metrics are only in metric_functions
        (none in _metric_registry or _ragas_metric_names),
        aggregated_metrics should be {} from the else branch.

        Covers the else branch: aggregated_metrics = {}.
        """

        def custom_my_score(actual, expected, **kwargs):
            return 0.88

        # 'my_score' is NOT in _metric_registry or _ragas_metric_names
        evaluator = LocalEvaluator(
            metrics=["my_score"],
            metric_functions={"my_score": custom_my_score},
            detailed=True,
            execution_mode="edge_analytics",
        )
        dataset = _make_dataset()

        # Mock compute_metrics -- it should NOT be called in this branch
        mock_compute = MagicMock(return_value={})
        with patch.object(evaluator, "compute_metrics", mock_compute):
            result = await evaluator.evaluate(_simple_func, {}, dataset)

        # compute_metrics should NOT have been called since there are
        # no built-in metrics to compute (all are custom-only).
        mock_compute.assert_not_called()

    @pytest.mark.asyncio
    async def test_self_metrics_not_mutated_during_compute(self):
        """self.metrics should never be mutated during evaluate(),
        ensuring thread-safety for concurrent evaluations.

        Verifies that metrics_override is used instead of self.metrics mutation.
        """

        def custom_accuracy(actual, expected, **kwargs):
            return 0.77

        original_metrics = ["accuracy", "latency", "cost"]
        evaluator = LocalEvaluator(
            metrics=original_metrics.copy(),
            metric_functions={"accuracy": custom_accuracy},
            detailed=True,
            execution_mode="edge_analytics",
        )
        dataset = _make_dataset()

        # Track self.metrics during compute_metrics call
        captured_metrics: list[list[str]] = []

        def spy_compute_metrics(*args, **kwargs):
            captured_metrics.append(list(evaluator.metrics))
            return {"latency": 0.1, "cost": 0.001}

        mock_compute = MagicMock(side_effect=spy_compute_metrics)
        with patch.object(evaluator, "compute_metrics", mock_compute):
            await evaluator.evaluate(_simple_func, {}, dataset)

        # self.metrics should be unchanged even during the call
        assert len(captured_metrics) == 1
        assert captured_metrics[0] == original_metrics
        assert evaluator.metrics == original_metrics

    @pytest.mark.asyncio
    async def test_self_metrics_unchanged_on_exception(self):
        """self.metrics should remain unchanged even if compute_metrics raises."""

        def custom_accuracy(actual, expected, **kwargs):
            return 0.5

        original_metrics = ["accuracy", "latency"]
        evaluator = LocalEvaluator(
            metrics=original_metrics.copy(),
            metric_functions={"accuracy": custom_accuracy},
            detailed=True,
            execution_mode="edge_analytics",
        )
        dataset = _make_dataset()

        mock_compute = MagicMock(side_effect=RuntimeError("boom"))
        with patch.object(evaluator, "compute_metrics", mock_compute):
            with pytest.raises(RuntimeError, match="boom"):
                await evaluator.evaluate(_simple_func, {}, dataset)

        # self.metrics should be unchanged (no mutation to restore)
        assert evaluator.metrics == original_metrics

    @pytest.mark.asyncio
    async def test_no_metric_functions_uses_compute_metrics_directly(self):
        """When metric_functions is empty, the else branch is taken and
        compute_metrics is called normally without metrics_override.

        Covers else branch: no metric_functions.
        """
        evaluator = LocalEvaluator(
            metrics=["accuracy", "latency"],
            metric_functions=None,  # No custom metric functions
            detailed=True,
            execution_mode="edge_analytics",
        )
        dataset = _make_dataset()

        mock_compute = MagicMock(return_value={"accuracy": 0.9, "latency": 0.2})
        with patch.object(evaluator, "compute_metrics", mock_compute):
            result = await evaluator.evaluate(_simple_func, {}, dataset)

        # compute_metrics should be called once (the else branch)
        mock_compute.assert_called_once()

        # No metrics_override should be passed in the else branch
        call_kwargs = mock_compute.call_args[1]
        assert "metrics_override" not in call_kwargs

        # The result should include the mocked values
        assert result.aggregated_metrics.get("latency") == 0.2
