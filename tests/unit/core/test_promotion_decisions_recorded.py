"""The optimizer records its closed promotion decision per candidate (additive, TRAIG-101)."""

from __future__ import annotations

from datetime import UTC, datetime

from traigent.api.types import TrialResult, TrialStatus
from traigent.core.orchestrator import OptimizationOrchestrator
from traigent.tvl.models import PromotionPolicy
from traigent.tvl.promotion_gate import ObjectiveSpec, PromotionGate
from traigent.utils.hashing import generate_config_hash

from .test_promotion_gate_integration import _MockEvaluator, _MockOptimizer


def _trial(x: int, accuracy: float) -> TrialResult:
    return TrialResult(
        trial_id=f"t{x}-{accuracy}",
        config={"x": x},
        metrics={"accuracy": accuracy},
        status=TrialStatus.COMPLETED,
        duration=0.1,
        timestamp=datetime.now(UTC),
    )


def _orchestrator(gate: bool) -> OptimizationOrchestrator:
    kwargs = {}
    if gate:
        policy = PromotionPolicy(
            dominance="epsilon_pareto", alpha=0.05, min_effect={"accuracy": 0.01}
        )
        kwargs["promotion_gate"] = PromotionGate(
            policy, [ObjectiveSpec("accuracy", "maximize")]
        )
    return OptimizationOrchestrator(
        optimizer=_MockOptimizer(), evaluator=_MockEvaluator(), max_trials=3, **kwargs
    )


def _h(x: int) -> str:
    return generate_config_hash({"x": x})


def test_first_candidate_is_an_explicit_no_incumbent_decision() -> None:
    o = _orchestrator(gate=True)
    o._update_best_trial_cache(_trial(1, 0.8))
    assert o._promotion_decisions == {_h(1): "no_incumbent"}


def test_no_gate_is_recorded_and_the_winner_is_unchanged() -> None:
    o = _orchestrator(gate=False)
    first, second = _trial(1, 0.5), _trial(2, 0.9)
    o._update_best_trial_cache(first)
    o._update_best_trial_cache(second)
    assert o._best_trial_cached is second  # behaviour as before
    assert o._promotion_decisions == {_h(1): "no_incumbent", _h(2): "no_gate"}


def test_insufficient_samples_then_gate_decisions() -> None:
    o = _orchestrator(gate=True)
    o._update_best_trial_cache(_trial(1, 0.75))
    o._update_best_trial_cache(_trial(2, 0.86))  # one sample each: gate cannot run
    assert o._promotion_decisions[_h(2)] == "insufficient_samples"
    o._incumbent_config_hash = _h(1)  # the simple comparison above moved it
    for acc in (0.76, 0.74, 0.75, 0.76):
        o._config_metrics_history[_h(1)].setdefault("accuracy", []).append(acc)
    for acc in (0.85, 0.86, 0.87, 0.85, 0.86):
        o._config_metrics_history[_h(3)] = {
            "accuracy": [
                *o._config_metrics_history.get(_h(3), {}).get("accuracy", []),
                acc,
            ]
        }
    assert o._evaluate_promotion(_h(3), _trial(3, 0.86)) is True
    assert o._promotion_decisions[_h(3)] == "promote"
    o._config_metrics_history[_h(4)] = {"accuracy": [0.40, 0.41, 0.39, 0.40, 0.41]}
    assert o._evaluate_promotion(_h(4), _trial(4, 0.4)) is False
    assert o._promotion_decisions[_h(4)] == "reject"


def test_gate_failure_is_recorded_as_gate_error_and_behaviour_is_unchanged() -> None:
    o = _orchestrator(gate=True)
    o._incumbent_config_hash = "inc"
    o._config_metrics_history["inc"] = {"accuracy": [0.7, 0.71, 0.72]}
    o._config_metrics_history["cand"] = {"accuracy": [0.9, 0.91, 0.92]}

    def boom(**_: object) -> None:
        raise RuntimeError("gate down")

    o._promotion_gate.evaluate = boom  # type: ignore[union-attr,method-assign]
    candidate = _trial(2, 0.91)
    assert (
        o._evaluate_promotion("cand", candidate) is True
    )  # simple comparison, as before
    assert o._promotion_decisions["cand"] == "gate_error"


def test_the_final_result_carries_the_decisions() -> None:
    o = _orchestrator(gate=False)
    for t in (_trial(1, 0.5), _trial(2, 0.9)):
        o._trials.append(t)
        o._update_best_trial_cache(t)
    result = o._create_optimization_result()
    assert result.promotion_decisions == {_h(1): "no_incumbent", _h(2): "no_gate"}
