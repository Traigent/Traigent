"""Regression tests: declared objective orientation is authoritative.

Findings T1 + T3 (customer-facing selection correctness).

T1 — ``calculate_weighted_scores`` used to discard the declared
``ObjectiveSchema`` at its call sites (orchestrator → persisted
``weighted_results_v2.json``; backend_session_manager → portal) and re-guess
minimize/maximize from the metric NAME. The substring guess inverted the winner
both ways:

* false-negative — a custom minimize name (``spend``/``perplexity``/``toxicity``)
  is not matched → treated as maximize → the PRICIEST config is crowned;
* false-positive — ``uptime`` contains the substring ``time`` → treated as
  minimize → the WORST config is crowned.

The fix threads the declared schema through every call site (so declared
orientation wins), keeps the name guess only as a schema-less last resort, makes
that fallback match WHOLE TOKENS (killing ``uptime ⊃ time``), and makes it emit
a ``UserWarning`` naming each guessed orientation.

T3 — a plain-list ``objectives=["accuracy", "price"]`` silently defaulted an
unknown minimize name to ``maximize`` with no signal, so ``best_config`` itself
crowned the priciest. The fix emits a loud ``UserWarning`` on that fallthrough.
"""

from __future__ import annotations

from typing import Any

import pytest

from traigent.api.types import (
    OptimizationResult,
    OptimizationStatus,
    TrialResult,
    TrialStatus,
    _name_suggests_minimize,
    _tokenize_metric_name,
)
from traigent.core.objectives import (
    ObjectiveDefinition,
    ObjectiveSchema,
    create_default_objectives,
)

_CHEAP_CONFIG = {"model": "cheap"}
_PRICEY_CONFIG = {"model": "pricey"}


def _trial(
    trial_id: str, config: dict[str, Any], metrics: dict[str, float]
) -> TrialResult:
    return TrialResult(
        trial_id=trial_id,
        config=config,
        metrics=metrics,
        status=TrialStatus.COMPLETED,
        duration=1.0,
        timestamp=0.0,
    )


def _spend_result() -> OptimizationResult:
    """accuracy(max) + spend(min): the pricey config has marginally higher
    accuracy but ~100x the spend. Under the declared schema (spend weighted
    heavily) the CHEAP config must win."""
    return OptimizationResult(
        trials=[
            # Ordered pricey-first so a name-guess winner is visibly the priciest.
            _trial("t_pricey", _PRICEY_CONFIG, {"accuracy": 0.82, "spend": 0.100}),
            _trial("t_cheap", _CHEAP_CONFIG, {"accuracy": 0.80, "spend": 0.001}),
        ],
        best_config=_CHEAP_CONFIG,
        best_score=0.80,
        optimization_id="opt",
        duration=1.0,
        convergence_info={},
        status=OptimizationStatus.COMPLETED,
        objectives=["accuracy", "spend"],
        algorithm="grid",
        timestamp=0.0,
    )


def _spend_schema() -> ObjectiveSchema:
    return ObjectiveSchema.from_objectives(
        [
            ObjectiveDefinition(name="accuracy", orientation="maximize", weight=0.2),
            ObjectiveDefinition(name="spend", orientation="minimize", weight=0.8),
        ]
    )


class TestTokenHeuristic:
    """The schema-less fallback must match whole tokens, not substrings."""

    def test_uptime_is_not_minimize_false_positive(self) -> None:
        # Pre-fix: "uptime" contained "time" -> minimize. Post-fix: not.
        assert _tokenize_metric_name("uptime") == {"uptime"}
        assert _name_suggests_minimize("uptime") is False

    def test_real_minimize_names_still_detected(self) -> None:
        assert _name_suggests_minimize("cost") is True
        assert _name_suggests_minimize("total_cost") is True
        assert _name_suggests_minimize("response_time") is True
        assert _name_suggests_minimize("responseTime") is True
        assert _name_suggests_minimize("avg-latency-ms") is True

    def test_custom_minimize_name_is_not_guessed(self) -> None:
        # Exactly why the declared schema must be authoritative: these read as
        # maximize to the heuristic.
        assert _name_suggests_minimize("spend") is False
        assert _name_suggests_minimize("perplexity") is False


class TestDeclaredOrientationAuthoritative:
    """T1(a): a declared custom-minimize name crowns the CHEAP config."""

    def test_declared_schema_crowns_cheap_config(self) -> None:
        result = _spend_result()
        weighted = result.calculate_weighted_scores(objective_schema=_spend_schema())
        assert weighted["best_weighted_config"] == _CHEAP_CONFIG

    def test_name_guess_without_schema_crowns_priciest_and_warns(self) -> None:
        """Documents the pre-fix call-site bug: with the schema discarded, the
        name guess treats 'spend' as maximize and crowns the PRICIEST config,
        and now does so LOUDLY (fail-loud warning)."""
        result = _spend_result()
        with pytest.warns(UserWarning, match="guessed from metric names"):
            weighted = result.calculate_weighted_scores()
        assert weighted["best_weighted_config"] == _PRICEY_CONFIG


class TestUptimeFalsePositiveInWeightedScores:
    """T1(b): a maximize metric named 'uptime' is NOT treated as minimize by the
    schema-less weighted-scores fallback."""

    def test_uptime_not_in_resolved_minimize_objectives(self) -> None:
        result = OptimizationResult(
            trials=[
                _trial("t1", {"model": "a"}, {"accuracy": 0.9, "uptime": 0.99}),
                _trial("t2", {"model": "b"}, {"accuracy": 0.7, "uptime": 0.95}),
            ],
            best_config={"model": "a"},
            best_score=0.9,
            optimization_id="opt",
            duration=1.0,
            convergence_info={},
            status=OptimizationStatus.COMPLETED,
            objectives=["accuracy", "uptime"],
            algorithm="grid",
            timestamp=0.0,
        )
        with pytest.warns(UserWarning, match="guessed from metric names"):
            weighted = result.calculate_weighted_scores()
        # Pre-fix "uptime" ⊃ "time" put it here; post-fix it must be absent.
        assert "uptime" not in weighted["minimize_objectives"]


class TestPlainListUnknownNameWarns:
    """T3(c): a plain-list unknown name warns about the guessed orientation."""

    def test_price_defaults_to_maximize_with_loud_warning(self) -> None:
        with pytest.warns(UserWarning, match="price"):
            schema = create_default_objectives(["accuracy", "price"])
        price = next(o for o in schema.objectives if o.name == "price")
        # Still defaults to maximize (behavior preserved) — but no longer silent.
        assert price.orientation == "maximize"

    def test_known_names_do_not_warn(self) -> None:
        import warnings as _warnings

        with _warnings.catch_warnings():
            _warnings.simplefilter("error")
            schema = create_default_objectives(["accuracy", "cost"])
        cost = next(o for o in schema.objectives if o.name == "cost")
        assert cost.orientation == "minimize"


# --- T1 call-site threading (the persisted weighted_results_v2.json path) -----

from datetime import UTC, datetime  # noqa: E402
from unittest.mock import Mock  # noqa: E402

from traigent.evaluators.base import (  # noqa: E402
    BaseEvaluator,
    Dataset,
    EvaluationResult,
)
from traigent.optimizers.base import BaseOptimizer  # noqa: E402


class _StaticOptimizer(BaseOptimizer):
    def suggest_next_trial(self, history: list[TrialResult]) -> dict[str, Any]:
        return {"model": "mock"}

    def should_stop(self, history: list[TrialResult]) -> bool:
        return True


class _UnusedEvaluator(BaseEvaluator):
    async def evaluate(
        self,
        func: Any,
        config: dict[str, Any],
        dataset: Dataset,
        **kwargs: Any,
    ) -> EvaluationResult:
        raise AssertionError("test injects completed trials directly")


def _eligible_trial(
    trial_id: str, model: str, accuracy: float, cost: float
) -> TrialResult:
    return TrialResult(
        trial_id=trial_id,
        config={"model": model},
        metrics={"accuracy": accuracy, "cost": cost},
        status=TrialStatus.COMPLETED,
        duration=0.1,
        timestamp=datetime.now(UTC),
        metadata={
            "successful_examples": 2,
            "examples_attempted": 2,
            "comparability": {
                "schema_version": "1.0",
                "primary_objective": "accuracy",
                "evaluation_mode": "evaluated",
                "total_examples": 2,
                "examples_with_primary_metric": 2,
                "coverage_ratio": 1.0,
                "derivation_path": "explicit",
                "ranking_eligible": True,
                "warning_codes": [],
                "per_metric_coverage": {
                    "accuracy": {"present": 2, "total": 2, "ratio": 1.0},
                    "cost": {"present": 2, "total": 2, "ratio": 1.0},
                },
                "missing_example_ids": [],
            },
        },
    )


class TestOrchestratorThreadsSchemaIntoWeightedScores:
    """T1: the orchestrator persists weighted scores using the DECLARED schema,
    not a name guess — i.e. it passes ``objective_schema`` through to
    ``calculate_weighted_scores``. Pre-fix it called it with no arguments."""

    def test_persisted_weighted_scores_receive_declared_schema(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from traigent.core.orchestrator import OptimizationOrchestrator

        schema = ObjectiveSchema.from_objectives(
            [
                ObjectiveDefinition(
                    name="accuracy", orientation="maximize", weight=0.2
                ),
                ObjectiveDefinition(name="cost", orientation="minimize", weight=0.8),
            ]
        )
        optimizer = _StaticOptimizer(
            {"model": ["gpt-4o", "gpt-3.5-turbo"]},
            objectives=["accuracy", "cost"],
        )
        orchestrator = OptimizationOrchestrator(
            optimizer=optimizer,
            evaluator=_UnusedEvaluator(metrics=["accuracy", "cost"]),
            max_trials=2,
            objective_schema=schema,
        )
        orchestrator._trials = [
            _eligible_trial("t1", "gpt-4o", accuracy=0.9, cost=0.005),
            _eligible_trial("t2", "gpt-3.5-turbo", accuracy=0.7, cost=0.001),
        ]
        orchestrator._status = OptimizationStatus.COMPLETED
        # Enable the weighted-logging branch without touching real logging I/O.
        orchestrator._logger = Mock()
        orchestrator._logger_facade = Mock()

        captured: dict[str, Any] = {}

        def _spy(self: OptimizationResult, *args: Any, **kwargs: Any) -> dict[str, Any]:
            captured["objective_schema"] = kwargs.get("objective_schema", "MISSING")
            return {"weighted_scores": [], "best_weighted_score": 0.0}

        monkeypatch.setattr(
            OptimizationResult, "calculate_weighted_scores", _spy, raising=True
        )

        orchestrator._create_optimization_result()

        # Pre-fix: called with no args -> "MISSING". Post-fix: the declared schema.
        assert captured["objective_schema"] is schema
