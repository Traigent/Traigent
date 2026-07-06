"""Regression tests for the None-return sibling of Traigent#1722/#1691.

The #1722 fix guards the *exception* path: a metric that RAISES and is an
optimization objective fails the trial closed (``EvaluationError``) instead of
being silently coerced to a fabricated 0.0. But a metric that *returns* ``None``
raises no exception, so it bypassed that guard entirely:

* a scalar ``None`` from an objective metric was coerced to ``0.0``
  (``_apply_custom_metric_functions``), and
* a ``None`` value inside a returned mapping was silently dropped.

Either way an optimization objective could be pinned to a fabricated worst
score (or vanish), silently corrupting the search -- the exact harm #1722
prevents on the exception path. A ``None`` return is a common user bug (a
metric with a missing ``return`` branch, or ``results.get("score")`` when the
key is absent).

Post-fix:
* An objective metric (name in ``self.metrics``) that returns ``None`` -- as a
  scalar or as a mapping entry keyed by the objective name -- fails the trial
  CLOSED with ``EvaluationError`` (details: is_objective=True,
  failure_mode="returned_none").
* A non-objective / informational metric that returns ``None`` retains the
  legacy behaviour: ``0.0`` for a scalar, silently skipped for a mapping entry.
* A metric returning a legitimate value is unaffected.
"""

from __future__ import annotations

import pytest

from traigent.evaluators.base import Dataset, EvaluationExample
from traigent.evaluators.local import LocalEvaluator
from traigent.utils.exceptions import EvaluationError


def _dataset() -> Dataset:
    return Dataset(
        examples=[
            EvaluationExample(input_data={"text": "q1"}, expected_output="YES"),
            EvaluationExample(input_data={"text": "q2"}, expected_output="YES"),
        ],
        name="silent_zero_on_metric_none_return",
    )


async def _func(text: str) -> str:
    return "YES"


class TestScalarNoneReturn:
    @pytest.mark.asyncio
    async def test_objective_returning_none_fails_closed(self) -> None:
        """A scalar ``None`` from an objective metric must fail the trial
        closed, not become a fabricated 0.0."""

        def none_objective(output, expected, **kwargs):
            return None

        evaluator = LocalEvaluator(
            metrics=["custom_objective"],
            metric_functions={"custom_objective": none_objective},
            detailed=True,
            execution_mode="local",
        )

        with pytest.raises(EvaluationError) as exc_info:
            await evaluator.evaluate(_func, {}, _dataset())

        assert exc_info.value.details is not None
        assert exc_info.value.details["metric_name"] == "custom_objective"
        assert exc_info.value.details["is_objective"] is True
        assert exc_info.value.details["failure_mode"] == "returned_none"
        assert exc_info.value.details["error_type"] == "NoneReturn"
        assert "custom_objective" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_informational_returning_none_keeps_legacy_zero(self) -> None:
        """A non-objective scalar ``None`` retains the legacy 0.0 sentinel and
        does not crash the trial."""

        def none_informational(output, expected, **kwargs):
            return None

        evaluator = LocalEvaluator(
            metrics=["accuracy"],
            metric_functions={"debug_metric": none_informational},
            detailed=True,
            execution_mode="local",
        )

        result = await evaluator.evaluate(_func, {}, _dataset())

        assert result.metrics.get("debug_metric") == 0.0

    @pytest.mark.asyncio
    async def test_legitimate_value_is_unchanged(self) -> None:
        """A metric returning a real value is unaffected by the None guard."""

        def real_metric(output, expected, **kwargs):
            return 0.75

        evaluator = LocalEvaluator(
            metrics=["custom_objective"],
            metric_functions={"custom_objective": real_metric},
            detailed=True,
            execution_mode="local",
        )

        result = await evaluator.evaluate(_func, {}, _dataset())

        assert result.metrics.get("custom_objective") == 0.75


class TestMappingNoneReturn:
    @pytest.mark.asyncio
    async def test_objective_subkey_none_fails_closed(self) -> None:
        """A ``None`` mapping entry keyed by an objective name must fail the
        trial closed rather than being silently dropped."""

        def combo(output, expected, **kwargs):
            return {"custom_objective": None, "aux": 0.5}

        evaluator = LocalEvaluator(
            metrics=["custom_objective"],
            metric_functions={"combo": combo},
            detailed=True,
            execution_mode="local",
        )

        with pytest.raises(EvaluationError) as exc_info:
            await evaluator.evaluate(_func, {}, _dataset())

        assert exc_info.value.details["metric_name"] == "custom_objective"
        assert exc_info.value.details["is_objective"] is True
        assert exc_info.value.details["failure_mode"] == "returned_none"
        assert exc_info.value.details["error_type"] == "NoneReturn"

    @pytest.mark.asyncio
    async def test_non_objective_subkey_none_is_skipped(self) -> None:
        """A ``None`` mapping entry that is NOT an objective is silently
        skipped (legacy behaviour) -- no fail-closed -- and sibling entries are
        still recorded.

        Uses ``detailed=False`` to avoid an unrelated pre-existing constraint:
        the detailed-mode transfer loop (``_process_single_output``) reads
        ``custom_metrics[<func name>]`` for every ``metric_functions`` entry,
        which a mapping-returning function never populates. That KeyError is
        independent of the None-guard under test here (flagged separately).
        """

        def combo(output, expected, **kwargs):
            return {"aux_none": None, "aux_value": 0.5}

        evaluator = LocalEvaluator(
            metrics=["accuracy"],
            metric_functions={"combo": combo},
            detailed=False,
            execution_mode="local",
        )

        # Must complete without raising (None sub-key skipped, not fail-closed).
        result = await evaluator.evaluate(_func, {}, _dataset())

        # Skipped None entry absent; sibling value recorded.
        assert "aux_none" not in result.metrics
        assert result.metrics.get("aux_value") == 0.5
