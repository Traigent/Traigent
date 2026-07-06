"""Regression tests: detailed-mode transfer of mapping-returning metrics.

`LocalEvaluator._apply_custom_metric_functions` supports a custom metric
function that returns a Mapping of sub-metrics; the sub-keys (not the function
name) are written into ``custom_metrics``. In ``detailed=True`` mode the
per-example transfer loop in ``_process_single_output`` previously iterated the
*function names* (``self.metric_functions``) and indexed
``custom_metrics[<func name>]`` -- which a mapping-returning function never
populates -- raising ``KeyError``. The optimization pipeline runs in detailed
mode, so any mapping-returning metric function crashed the trial.

Fix: ``_apply_custom_metric_functions`` returns the keys it actually produced,
and the transfer loop iterates those. Scalar functions are unaffected (their
produced key is the function name); mapping functions now transfer their
sub-keys correctly.
"""

from __future__ import annotations

import pytest

from traigent.evaluators.base import Dataset, EvaluationExample
from traigent.evaluators.local import LocalEvaluator


def _dataset() -> Dataset:
    return Dataset(
        examples=[
            EvaluationExample(input_data={"text": "q1"}, expected_output="YES"),
            EvaluationExample(input_data={"text": "q2"}, expected_output="YES"),
        ],
        name="detailed_mode_mapping_metric_transfer",
    )


async def _func(text: str) -> str:
    return "YES"


class TestDetailedModeMappingMetricTransfer:
    @pytest.mark.asyncio
    async def test_mapping_metric_does_not_keyerror_in_detailed_mode(self) -> None:
        """A mapping-returning metric function must not crash the trial in
        detailed mode, and its sub-metrics must be recorded."""

        def combo(output, expected, **kwargs):
            return {"precision": 0.8, "recall": 0.6}

        evaluator = LocalEvaluator(
            metrics=["accuracy"],
            metric_functions={"combo": combo},
            detailed=True,
            execution_mode="local",
        )

        # Pre-fix this raised KeyError: 'combo'.
        result = await evaluator.evaluate(_func, {}, _dataset())

        assert result.metrics.get("precision") == 0.8
        assert result.metrics.get("recall") == 0.6

    @pytest.mark.asyncio
    async def test_mapping_subkeys_transferred_to_example_results(self) -> None:
        """The produced sub-keys are transferred into per-example results in
        detailed mode."""

        def combo(output, expected, **kwargs):
            return {"precision": 0.8, "recall": 0.6}

        evaluator = LocalEvaluator(
            metrics=["accuracy"],
            metric_functions={"combo": combo},
            detailed=True,
            execution_mode="local",
        )

        result = await evaluator.evaluate(_func, {}, _dataset())

        assert result.example_results, "expected detailed example_results"
        first = result.example_results[0]
        assert first.metrics.get("precision") == 0.8
        assert first.metrics.get("recall") == 0.6
        # The container function name is never a produced key.
        assert "combo" not in first.metrics

    @pytest.mark.asyncio
    async def test_scalar_metric_still_transferred_by_name(self) -> None:
        """Scalar-returning metric functions are unaffected: the function name
        is the produced key and is transferred in detailed mode."""

        def scalar_metric(output, expected, **kwargs):
            return 0.42

        evaluator = LocalEvaluator(
            metrics=["accuracy"],
            metric_functions={"scalar_metric": scalar_metric},
            detailed=True,
            execution_mode="local",
        )

        result = await evaluator.evaluate(_func, {}, _dataset())

        assert result.example_results
        assert result.example_results[0].metrics.get("scalar_metric") == 0.42

    @pytest.mark.asyncio
    async def test_no_bogus_aggregate_for_mapping_function_name(self) -> None:
        """The aggregate must not fabricate a ``0.0`` under the mapping
        function's own name (which is never a produced metric key)."""

        def combo(output, expected, **kwargs):
            return {"precision": 0.8, "recall": 0.6}

        evaluator = LocalEvaluator(
            metrics=["accuracy"],
            metric_functions={"combo": combo},
            detailed=True,
            execution_mode="local",
        )

        result = await evaluator.evaluate(_func, {}, _dataset())

        # The function name is not a metric -> no aggregate entry for it.
        assert "combo" not in result.metrics
        # The real produced sub-keys are aggregated.
        assert result.metrics.get("precision") == 0.8
        assert result.metrics.get("recall") == 0.6

    @pytest.mark.asyncio
    async def test_mapping_with_mixed_none_records_only_non_none(self) -> None:
        """A mapping with a non-objective ``None`` sub-key records the
        non-None siblings and skips the None one (no crash, no 0.0 fabricated
        for the skipped key)."""

        def combo(output, expected, **kwargs):
            return {"present": 0.5, "missing": None}

        evaluator = LocalEvaluator(
            metrics=["accuracy"],
            metric_functions={"combo": combo},
            detailed=True,
            execution_mode="local",
        )

        result = await evaluator.evaluate(_func, {}, _dataset())

        assert result.example_results[0].metrics.get("present") == 0.5
        assert "missing" not in result.example_results[0].metrics
        assert "missing" not in result.metrics

    @pytest.mark.asyncio
    async def test_non_detailed_scalar_zero_aggregate_preserved(self) -> None:
        """The detailed-mode aggregate fix must NOT regress non-detailed mode:
        a scalar metric function returning a legitimate 0.0 keeps its 0.0
        aggregate (no per-example rows exist to re-derive it)."""

        def zero_metric(output, expected, **kwargs):
            return 0.0

        evaluator = LocalEvaluator(
            metrics=["accuracy"],
            metric_functions={"zero_metric": zero_metric},
            detailed=False,
            execution_mode="local",
        )

        result = await evaluator.evaluate(_func, {}, _dataset())

        assert result.metrics.get("zero_metric") == 0.0
