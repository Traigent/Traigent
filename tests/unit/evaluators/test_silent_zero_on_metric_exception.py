"""Regression tests for Traigent#1722 (silent-failure audit, g6).

Pre-fix, any exception raised by a metric/scoring function was caught,
logged once, and silently replaced with 0.0 -- indistinguishable from a
legitimate 0.0 score. When the failing metric was an optimization
*objective*, the fabricated 0.0 pinned/corrupted the search.

Post-fix:
* An objective metric (a name in ``self.metrics``, which every real
  construction path populates directly from the optimization's
  ``objectives``) that raises fails the trial CLOSED: ``EvaluationError``
  propagates instead of a fabricated score.
* A non-objective / informational metric that raises still gets a
  structured degradation record ({metric_name, example_id, error_type,
  error_message, is_objective}) -- never a bare, unaccompanied 0.0.
* A metric that legitimately returns 0.0 (no exception) is unaffected: no
  degradation record, value unchanged.

Covers the three sites named in the bug report:
* traigent/evaluators/local.py -- ``_invoke_metric_function`` /
  ``EvaluationResult.metric_errors``
* traigent/evaluators/base.py -- ``BaseEvaluator.compute_metrics``
* traigent/evaluators/metrics.py -- ``MetricsComputer.compute_custom_metrics``
"""

from __future__ import annotations

import pytest

from traigent.evaluators.base import Dataset, EvaluationExample
from traigent.evaluators.local import LocalEvaluator
from traigent.evaluators.metrics import MetricsComputer
from traigent.invokers.base import InvocationResult
from traigent.utils.exceptions import EvaluationError


def _dataset() -> Dataset:
    return Dataset(
        examples=[
            EvaluationExample(input_data={"text": "q1"}, expected_output="YES"),
            EvaluationExample(input_data={"text": "q2"}, expected_output="YES"),
        ],
        name="silent_zero_on_metric_exception",
    )


async def _func(text: str) -> str:
    return "YES"


# ---------------------------------------------------------------------------
# Site 1: traigent/evaluators/local.py -- _invoke_metric_function (custom
# metric_functions path, via LocalEvaluator.evaluate()).
# ---------------------------------------------------------------------------


class TestLocalEvaluatorObjectiveMetricFailsClosed:
    @pytest.mark.asyncio
    async def test_raising_objective_metric_fails_trial_not_silent_zero(self) -> None:
        """A custom metric function that IS an objective must fail the trial
        closed instead of silently reporting a fabricated 0.0."""

        def failing_objective(output, expected, **kwargs):
            raise ValueError("boom")

        evaluator = LocalEvaluator(
            metrics=["custom_objective"],
            metric_functions={"custom_objective": failing_objective},
            detailed=True,
            execution_mode="local",
        )

        with pytest.raises(EvaluationError) as exc_info:
            await evaluator.evaluate(_func, {}, _dataset())

        assert exc_info.value.details is not None
        assert exc_info.value.details["metric_name"] == "custom_objective"
        assert exc_info.value.details["is_objective"] is True
        assert exc_info.value.details["error_type"] == "ValueError"
        assert "custom_objective" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_raising_informational_metric_records_degradation(self) -> None:
        """A custom metric that raises but is NOT one of the evaluator's
        objectives must still surface a structured degradation record,
        never a bare unaccompanied 0.0."""

        def failing_informational(output, expected, **kwargs):
            raise RuntimeError("informational metric blew up")

        evaluator = LocalEvaluator(
            metrics=["accuracy"],
            metric_functions={"debug_metric": failing_informational},
            detailed=True,
            execution_mode="local",
        )

        result = await evaluator.evaluate(_func, {}, _dataset())

        # Sentinel value still present (compat), but never bare.
        assert result.metrics.get("debug_metric") == 0.0

        # Structured degradation record must exist and be joinable back to
        # the failing metric.
        assert result.metric_errors, "expected a recorded metric_errors entry"
        records = [
            r for r in result.metric_errors if r["metric_name"] == "debug_metric"
        ]
        assert records, (
            f"no degradation record for debug_metric: {result.metric_errors}"
        )
        record = records[0]
        assert record["error_type"] == "RuntimeError"
        assert "blew up" in record["error_message"]
        assert record["is_objective"] is False
        assert "example_id" in record

    @pytest.mark.asyncio
    async def test_legitimate_zero_is_unchanged_and_unflagged(self) -> None:
        """A metric function that legitimately computes 0.0 (no exception)
        must be unaffected: the value is 0.0 and no degradation record is
        created for it."""

        def real_zero(output, expected, **kwargs):
            return 0.0

        evaluator = LocalEvaluator(
            metrics=["accuracy"],
            metric_functions={"real_zero_metric": real_zero},
            detailed=True,
            execution_mode="local",
        )

        result = await evaluator.evaluate(_func, {}, _dataset())

        assert result.metrics.get("real_zero_metric") == 0.0
        assert result.metric_errors == []


# ---------------------------------------------------------------------------
# Site 2: traigent/evaluators/base.py -- BaseEvaluator.compute_metrics
# (registry metric path, reached via LocalEvaluator's built-in/registered
# metrics lane).
# ---------------------------------------------------------------------------


class TestBaseEvaluatorRegistryMetricFailsClosed:
    @pytest.mark.asyncio
    async def test_raising_objective_registry_metric_fails_trial(self) -> None:
        """A registered metric (via register_metric/override_metric) that IS
        an objective must fail the trial closed."""

        evaluator = LocalEvaluator(
            metrics=["accuracy", "flaky_registry_metric"],
            detailed=True,
            execution_mode="local",
        )

        def flaky(outputs, expected_outputs, errors, **kwargs):
            raise KeyError("registry metric exploded")

        evaluator.register_metric("flaky_registry_metric", flaky)

        with pytest.raises(EvaluationError) as exc_info:
            await evaluator.evaluate(_func, {}, _dataset())

        assert exc_info.value.details["metric_name"] == "flaky_registry_metric"
        assert exc_info.value.details["is_objective"] is True


# ---------------------------------------------------------------------------
# Site 3: traigent/evaluators/metrics.py -- MetricsComputer.compute_custom_metrics
# (standalone utility, not wired into the live optimization pipeline -- see
# design note in metrics.py -- so it always records + keeps the 0.0 sentinel
# rather than failing closed).
# ---------------------------------------------------------------------------


class TestMetricsComputerRecordsDegradation:
    def test_failing_custom_metric_records_degradation_not_bare_zero(self) -> None:
        computer = MetricsComputer(metrics=["accuracy"])

        def failing_func(outputs: list, expected: list) -> float:
            raise ValueError("custom metric failed")

        computer.add_custom_metric("failing_metric", failing_func)

        invocation_results = [InvocationResult(result="output1", is_successful=True)]
        expected_outputs = ["output1"]

        result = computer.compute_metrics(invocation_results, expected_outputs)

        # Sentinel unchanged (back-compat), but paired with a structured record.
        assert result.metrics["failing_metric"] == 0.0
        metric_errors = result.metadata["metric_errors"]
        assert metric_errors
        record = next(r for r in metric_errors if r["metric_name"] == "failing_metric")
        assert record["error_type"] == "ValueError"
        assert "custom metric failed" in record["error_message"]
        assert record["is_objective"] is False

    def test_successful_custom_metric_has_no_degradation_record(self) -> None:
        computer = MetricsComputer(metrics=["accuracy"])

        def ok_func(outputs: list, expected: list) -> float:
            return 0.0

        computer.add_custom_metric("real_zero", ok_func)

        invocation_results = [InvocationResult(result="output1", is_successful=True)]
        expected_outputs = ["output1"]

        result = computer.compute_metrics(invocation_results, expected_outputs)

        assert result.metrics["real_zero"] == 0.0
        assert result.metadata["metric_errors"] == []
