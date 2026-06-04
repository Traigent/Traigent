"""Fail-closed strict-evidence promotion (FR-SDK-FAIL-CLOSED-PROMOTION-V1).

The four verified leak sites, closed strictly behind _is_strict_evidence_mode:
1. gate no_decision           -> never _simple_is_better (spy-verified)
2. insufficient samples       -> never _simple_is_better
3. gate exception             -> never _simple_is_better (Rule 1: deny)
4. terminal selector          -> NO_CERTIFIED_SELECTION, never re-derived
Non-strict behavior is pinned byte-identical (the legacy lane).
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from traigent.api.types import TrialResult, TrialStatus
from traigent.core.orchestrator import OptimizationOrchestrator
from traigent.core.result_selection import (
    NO_CERTIFIED_SELECTION,
    select_best_configuration,
)
from traigent.tvl.models import ChanceConstraint, PromotionPolicy, RequireCalibration


def _trial(score: float = 0.9, config=None) -> TrialResult:
    return TrialResult(
        trial_id="t1",
        config=config or {"model": "a"},
        metrics={"accuracy": score},
        status=TrialStatus.COMPLETED,
        duration=0.1,
        timestamp=0.0,
        metadata={"successful_examples": 1},
    )


def _orchestrator(policy: PromotionPolicy | None) -> OptimizationOrchestrator:
    orchestrator = OptimizationOrchestrator.__new__(OptimizationOrchestrator)
    orchestrator._strict_withheld_promotions = []
    orchestrator._promotion_gate = (
        None if policy is None else SimpleNamespace(policy=policy)
    )
    orchestrator._best_trial_cached = _trial(0.5)
    orchestrator.optimizer = SimpleNamespace(objectives=["accuracy"])
    orchestrator._config_metrics_history = {}
    orchestrator._incumbent_config_hash = "incumbent"
    return orchestrator


STRICT_POLICIES = {
    "require_calibration": PromotionPolicy(
        require_calibration=RequireCalibration(enabled=True)
    ),
    "chance_constraints": PromotionPolicy(
        chance_constraints=[
            ChanceConstraint(name="accuracy", threshold=0.8, confidence=0.95)
        ]
    ),
}

NON_STRICT_POLICY = PromotionPolicy()


class TestStrictModeDetection:
    @pytest.mark.parametrize("name,policy", sorted(STRICT_POLICIES.items()))
    def test_each_declared_trigger_is_strict(self, name, policy):
        assert _orchestrator(policy)._is_strict_evidence_mode(), name

    def test_plain_policy_is_not_strict(self):
        assert not _orchestrator(NON_STRICT_POLICY)._is_strict_evidence_mode()

    def test_no_gate_is_not_strict(self):
        assert not _orchestrator(None)._is_strict_evidence_mode()

    def test_disabled_require_calibration_is_not_strict(self):
        policy = PromotionPolicy(
            require_calibration=RequireCalibration(enabled=False)
        )
        assert not _orchestrator(policy)._is_strict_evidence_mode()


@pytest.mark.parametrize("name,policy", sorted(STRICT_POLICIES.items()))
class TestStrictFailClosed:
    def test_no_decision_withholds_without_simple_fallback(self, name, policy):
        orchestrator = _orchestrator(policy)
        orchestrator._simple_is_better = MagicMock(name="simple")  # spy
        decision = SimpleNamespace(decision="no_decision", reason="too few samples")
        assert orchestrator._handle_promotion_decision(decision, _trial()) is False
        orchestrator._simple_is_better.assert_not_called()
        assert orchestrator._strict_withheld_promotions

    def test_insufficient_samples_withholds(self, name, policy):
        orchestrator = _orchestrator(policy)
        orchestrator._simple_is_better = MagicMock(name="simple")
        orchestrator._has_sufficient_samples = MagicMock(return_value=False)
        orchestrator._config_metrics_history = {
            "candidate": {"accuracy": [0.9]},
            "incumbent": {"accuracy": [0.5]},
        }
        assert orchestrator._evaluate_promotion("candidate", _trial()) is False
        orchestrator._simple_is_better.assert_not_called()

    def test_gate_exception_fails_closed(self, name, policy):
        orchestrator = _orchestrator(policy)
        orchestrator._simple_is_better = MagicMock(name="simple")
        orchestrator._has_sufficient_samples = MagicMock(return_value=True)
        orchestrator._config_metrics_history = {
            "candidate": {"accuracy": [0.9, 0.91]},
            "incumbent": {"accuracy": [0.5, 0.52]},
        }
        orchestrator._promotion_gate.evaluate = MagicMock(
            side_effect=RuntimeError("gate blew up")
        )
        orchestrator._promotion_gate.policy = policy
        assert orchestrator._evaluate_promotion("candidate", _trial()) is False
        orchestrator._simple_is_better.assert_not_called()
        assert any(
            "gate exception" in reason
            for reason in orchestrator._strict_withheld_promotions
        )

    def test_promote_and_reject_still_flow_through(self, name, policy):
        orchestrator = _orchestrator(policy)
        promote = SimpleNamespace(decision="promote", reason="dominates")
        reject = SimpleNamespace(decision="reject", reason="dominated")
        assert orchestrator._handle_promotion_decision(promote, _trial()) is True
        assert orchestrator._handle_promotion_decision(reject, _trial()) is False


class TestNonStrictLaneUnchanged:
    """The legacy lane must stay byte-identical (no behavior change for
    modules without strict declarations)."""

    def test_no_decision_falls_back_to_simple(self):
        orchestrator = _orchestrator(NON_STRICT_POLICY)
        orchestrator._simple_is_better = MagicMock(return_value=True)
        decision = SimpleNamespace(decision="no_decision", reason="few samples")
        assert orchestrator._handle_promotion_decision(decision, _trial()) is True
        orchestrator._simple_is_better.assert_called_once()

    def test_gate_exception_falls_back_to_simple(self):
        orchestrator = _orchestrator(NON_STRICT_POLICY)
        orchestrator._simple_is_better = MagicMock(return_value=True)
        orchestrator._has_sufficient_samples = MagicMock(return_value=True)
        orchestrator._config_metrics_history = {
            "candidate": {"accuracy": [0.9]},
            "incumbent": {"accuracy": [0.5]},
        }
        orchestrator._promotion_gate.evaluate = MagicMock(
            side_effect=RuntimeError("gate blew up")
        )
        assert orchestrator._evaluate_promotion("candidate", _trial()) is True
        orchestrator._simple_is_better.assert_called_once()


class TestTerminalSelector:
    """Leak 4: select_best_configuration is gate-independent by default —
    require_certified makes it honor the certified incumbent or return the
    explicit no-winner shape."""

    def test_no_certified_winner_returns_explicit_empty(self):
        result = select_best_configuration(
            trials=[_trial(0.99)],  # high score MUST NOT win
            primary_objective="accuracy",
            config_space_keys={"model"},
            aggregate_configs=False,
            require_certified=True,
            certified_config=None,
        )
        assert result.best_config == {}
        assert result.best_score is None
        assert result.reason_code == NO_CERTIFIED_SELECTION

    def test_certified_incumbent_returned_verbatim(self):
        result = select_best_configuration(
            trials=[_trial(0.99, config={"model": "hot"})],
            primary_objective="accuracy",
            config_space_keys={"model"},
            aggregate_configs=False,
            require_certified=True,
            certified_config={"model": "certified"},
            certified_score=0.7,
        )
        # the certified incumbent wins even though a higher-scoring trial exists
        assert result.best_config == {"model": "certified"}
        assert result.best_score == 0.7
        assert result.reason_code is None

    def test_default_mode_unchanged(self):
        result = select_best_configuration(
            trials=[_trial(0.99, config={"model": "hot"})],
            primary_objective="accuracy",
            config_space_keys={"model"},
            aggregate_configs=False,
            comparability_mode="legacy",
        )
        assert result.best_config.get("model") == "hot"
