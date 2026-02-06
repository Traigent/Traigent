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
        computed via compute_metrics with only ['latency', 'cost'].

        Covers lines 1180-1206 (the if-branch with base_metric_names populated).
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

        # Capture what self.metrics is set to during the compute_metrics call.
        captured_metrics: list[list[str]] = []

        def spy_compute_metrics(*args, **kwargs):
            captured_metrics.append(list(evaluator.metrics))
            return {"latency": 0.5, "success_rate": 0.9}

        mock_compute = MagicMock(side_effect=spy_compute_metrics)
        with patch.object(evaluator, "compute_metrics", mock_compute):
            result = await evaluator.evaluate(_simple_func, {}, dataset)

        # compute_metrics should have been called exactly once
        mock_compute.assert_called_once()

        # During the call, self.metrics should have been temporarily set to
        # only the built-in metrics (latency, success_rate) -- not accuracy
        # (which is overridden by metric_functions).
        assert len(captured_metrics) == 1
        assert "accuracy" not in captured_metrics[0]
        assert "latency" in captured_metrics[0]
        assert "success_rate" in captured_metrics[0]

        # The mocked built-in values should appear in aggregated_metrics
        assert result.aggregated_metrics.get("latency") == 0.5

    @pytest.mark.asyncio
    async def test_metric_functions_all_custom_no_builtins(self):
        """When all metrics in self.metrics are only in metric_functions
        (none in _metric_registry or _ragas_metric_names),
        aggregated_metrics should be {} from the else branch.

        Covers lines 1207-1208 (the else branch: aggregated_metrics = {}).
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
    async def test_metric_functions_restores_metrics_after_compute(self):
        """After evaluate completes, self.metrics should be restored to its
        original value even though it was temporarily overwritten.

        Covers lines 1194-1206 (the try/finally block).
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

        mock_compute = MagicMock(return_value={"latency": 0.1, "cost": 0.001})
        with patch.object(evaluator, "compute_metrics", mock_compute):
            await evaluator.evaluate(_simple_func, {}, dataset)

        # self.metrics must be restored to the original list
        assert evaluator.metrics == original_metrics

    @pytest.mark.asyncio
    async def test_metric_functions_restores_metrics_on_exception(self):
        """self.metrics should be restored even if compute_metrics raises.

        Covers the finally block at lines 1205-1206.
        """

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

        # Even after an exception, self.metrics must be restored
        assert evaluator.metrics == original_metrics

    @pytest.mark.asyncio
    async def test_no_metric_functions_uses_compute_metrics_directly(self):
        """When metric_functions is empty, the else branch at line 1209-1217
        is taken and compute_metrics is called normally with full self.metrics.

        Covers lines 1209-1217 (else branch: no metric_functions).
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
        # The result should include the mocked values
        assert result.aggregated_metrics.get("latency") == 0.2
