"""Regression tests for #1692: ``plateau_window``/``plateau_epsilon``/
``semantic_saturation`` early-stop knobs silently no-op'd when configured on
the ``@traigent.optimize(...)`` DECORATOR (they only worked at
``.optimize()`` call time).

Root cause (pre-fix): ``plateau_window``, ``plateau_epsilon``, and
``semantic_saturation`` were listed in ``_OPTIMIZE_DEFAULTS``
(``traigent/api/decorators.py``), which put them in ``_DIRECT_OPTION_KEYS``.
``_process_runtime_overrides`` routes any decorator kwarg in
``_DIRECT_OPTION_KEYS`` to ``record_option`` (the dead ``combined_settings``
bucket) instead of ``combined_runtime_overrides`` (the bucket that flows into
``OptimizedFunction._decorator_runtime_overrides`` and, from there, into the
orchestrator's stop-condition config). Nothing ever read
``combined_settings["plateau_window"]`` back out, so a decorator-only
early-stop config silently ran the full trial budget instead of stopping at
convergence.

The fix removes these three keys from ``_OPTIMIZE_DEFAULTS`` so they fall
through to ``combined_runtime_overrides`` like their working sibling
``metric_limit`` — reaching ``_decorator_runtime_overrides`` and, at call
time, being merged with (and overridable by) explicit ``.optimize(...)``
kwargs via ``OptimizedFunction._prepare_algorithm_kwargs``.

These tests drive a real (mocked-evaluator, offline, local-grid) optimization
run end-to-end and assert on ``len(result.trials)`` / ``result.stop_reason``
-- the same observable behavior #1692 reported — rather than only inspecting
internal routing dicts.
"""

from __future__ import annotations

import pytest

import traigent
from traigent.api.types import ExampleResult, OptimizationResult, OptimizationStatus
from traigent.evaluators.base import Dataset, EvaluationExample

# 4 * 2 = 8 fully-enumerable configs; large enough that "stopped early"
# (2 trials) is unambiguously distinguishable from "ran to completion" (8).
_GRID_SPACE = {
    "temperature": [0.0, 0.25, 0.5, 0.75],
    "model": ["a", "b"],
}
_TOTAL_CONFIGS = 8


def _dataset() -> Dataset:
    return Dataset(
        [EvaluationExample({"text": "case-0"}, "ok")],
        name="decorator_early_stop_1692",
    )


def _constant_score_evaluator(func, config, example) -> ExampleResult:
    """Evaluator returning an identical score for every trial.

    A perfectly flat score sequence is the canonical plateau/saturation
    trigger: with no improvement across trials, both
    ``PlateauAfterNStopCondition`` and the semantic-saturation condition
    should fire as soon as their trial-window requirement is met.
    """
    return ExampleResult(
        example_id="e0",
        input_data=example.input_data,
        expected_output=example.expected_output,
        actual_output="ok",
        metrics={"accuracy": 1.0},
        execution_time=0.001,
        success=True,
    )


class TestPlateauConfiguredOnlyAtDecorator:
    """plateau_window/plateau_epsilon set ONLY on @traigent.optimize(...)."""

    @pytest.mark.asyncio
    async def test_decoration_time_plateau_actually_stops_early(self) -> None:
        """Bug #1692: this used to run the full 8-trial budget (max_trials_
        reached) because the decorator's plateau_window/plateau_epsilon were
        dropped by ``_DIRECT_OPTION_KEYS`` routing. Fixed: the run must stop
        after exactly ``plateau_window`` trials with ``stop_reason=plateau``.
        """

        @traigent.optimize(
            eval_dataset=_dataset(),
            objectives=["accuracy"],
            configuration_space=_GRID_SPACE,
            injection_mode="parameter",
            algorithm="grid",
            offline=True,
            cost_approved=True,
            custom_evaluator=_constant_score_evaluator,
            max_trials=_TOTAL_CONFIGS,
            plateau_window=2,
            plateau_epsilon=0.0,
        )
        def answer(text: str, config) -> str:
            return "ok"

        # Sanity: the decoration-time values must actually reach the
        # runtime-override bucket the orchestrator reads (not the dead
        # combined_settings bucket) -- this is the direct routing assertion.
        assert answer._decorator_runtime_overrides.get("plateau_window") == 2
        assert answer._decorator_runtime_overrides.get("plateau_epsilon") == 0.0

        result = await answer.optimize()

        assert isinstance(result, OptimizationResult)
        assert result.status is OptimizationStatus.COMPLETED
        assert result.stop_reason == "plateau", (
            f"expected early stop via plateau, got stop_reason="
            f"{result.stop_reason!r} with {len(result.trials)} trials -- "
            "decoration-time plateau_window/plateau_epsilon did not wire "
            "through (issue #1692)"
        )
        assert len(result.trials) == 2, (
            f"expected exactly plateau_window=2 trials before stopping, got "
            f"{len(result.trials)}"
        )


class TestSemanticSaturationConfiguredOnlyAtDecorator:
    """semantic_saturation set ONLY on @traigent.optimize(...)."""

    @pytest.mark.asyncio
    async def test_decoration_time_semantic_saturation_actually_stops_early(
        self,
    ) -> None:
        """Same bug class as plateau: semantic_saturation configured purely
        at the decorator must actually configure the stop condition."""

        @traigent.optimize(
            eval_dataset=_dataset(),
            objectives=["accuracy"],
            configuration_space=_GRID_SPACE,
            injection_mode="parameter",
            algorithm="grid",
            offline=True,
            cost_approved=True,
            custom_evaluator=_constant_score_evaluator,
            max_trials=_TOTAL_CONFIGS,
            semantic_saturation={
                "window": 2,
                "min_trials": 2,
                "continuous_objectives": [],
            },
        )
        def answer(text: str, config) -> str:
            return "ok"

        assert answer._decorator_runtime_overrides.get("semantic_saturation") == {
            "window": 2,
            "min_trials": 2,
            "continuous_objectives": [],
        }

        result = await answer.optimize()

        assert isinstance(result, OptimizationResult)
        assert result.status is OptimizationStatus.COMPLETED
        assert result.stop_reason == "semantic_saturation", (
            f"expected early stop via semantic_saturation, got stop_reason="
            f"{result.stop_reason!r} with {len(result.trials)} trials -- "
            "decoration-time semantic_saturation did not wire through "
            "(issue #1692)"
        )
        assert len(result.trials) == 2

    @pytest.mark.asyncio
    async def test_malformed_semantic_saturation_at_decorator_is_validated(
        self,
    ) -> None:
        """Extra symptom noted in #1692: a malformed decorator-level
        semantic_saturation dict must be rejected (validated) when the run
        executes, not silently swallowed by the dead combined_settings
        routing (which never validated it because it never forwarded it).

        Validation itself lives in the stop-condition manager built when the
        orchestrator starts a run, so the ValueError surfaces from
        ``.optimize()`` -- decoration alone cannot detect it either before or
        after this fix. What #1692 fixes is that it now surfaces AT ALL
        (previously: no error, ran the full trial budget)."""

        @traigent.optimize(
            eval_dataset=_dataset(),
            objectives=["accuracy"],
            configuration_space=_GRID_SPACE,
            injection_mode="parameter",
            algorithm="grid",
            offline=True,
            cost_approved=True,
            custom_evaluator=_constant_score_evaluator,
            max_trials=_TOTAL_CONFIGS,
            semantic_saturation={"metric": "accuracy", "epsilon": 0.0},
        )
        def bad(text: str, config) -> str:
            return "ok"

        with pytest.raises(ValueError, match="semantic_saturation"):
            await bad.optimize()


class TestCallTimeOverridesDecorationTimePlateau:
    """Precedence: an explicit call-time value must win over the decorator's."""

    @pytest.mark.asyncio
    async def test_call_time_plateau_window_overrides_decorator_value(self) -> None:
        """Decorator configures an aggressive plateau_window=2 (would stop
        after 2 trials, per the sibling test above). A call-time
        ``plateau_window=100`` (looser than the 8-config grid can ever
        trigger) must override it, so the run exhausts max_trials instead."""

        @traigent.optimize(
            eval_dataset=_dataset(),
            objectives=["accuracy"],
            configuration_space=_GRID_SPACE,
            injection_mode="parameter",
            algorithm="grid",
            offline=True,
            cost_approved=True,
            custom_evaluator=_constant_score_evaluator,
            max_trials=_TOTAL_CONFIGS,
            plateau_window=2,
            plateau_epsilon=0.0,
        )
        def answer(text: str, config) -> str:
            return "ok"

        result = await answer.optimize(plateau_window=100)

        assert isinstance(result, OptimizationResult)
        assert result.status is OptimizationStatus.COMPLETED
        assert result.stop_reason == "max_trials_reached", (
            "call-time plateau_window=100 should have overridden the "
            f"decorator's plateau_window=2, but got stop_reason="
            f"{result.stop_reason!r} with {len(result.trials)} trials"
        )
        assert len(result.trials) == _TOTAL_CONFIGS
