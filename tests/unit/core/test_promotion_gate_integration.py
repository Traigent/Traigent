"""Tests for PromotionGate integration with OptimizationOrchestrator."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import pytest

from traigent.api.types import TrialResult, TrialStatus
from traigent.core.orchestrator import OptimizationOrchestrator
from traigent.evaluators.base import BaseEvaluator
from traigent.optimizers.base import BaseOptimizer
from traigent.tvl.models import PromotionPolicy
from traigent.tvl.promotion_gate import (
    ObjectiveSpec,
    PromotionGate,
)


class _MockOptimizer(BaseOptimizer):
    """Mock optimizer for testing."""

    def __init__(self, config_space: dict | None = None) -> None:
        super().__init__(
            config_space=config_space or {"x": [1, 2, 3]},
            objectives=["accuracy"],
        )
        self._suggestions = iter([{"x": 1}, {"x": 2}, {"x": 3}])
        self._trial_count = 0

    def suggest_next_trial(self, history: list[TrialResult]):
        self._trial_count += 1
        try:
            return next(self._suggestions)
        except StopIteration:
            return None

    def should_stop(self, history: list[TrialResult]) -> bool:
        return self._trial_count >= 3


class _MockEvaluator(BaseEvaluator):
    """Mock evaluator for testing."""

    def __init__(self, results: list[dict] | None = None) -> None:
        self._results = iter(results or [{"accuracy": 0.8}])

    async def evaluate(self, func, config, dataset, **kwargs):
        try:
            metrics = next(self._results)
        except StopIteration:
            metrics = {"accuracy": 0.5}
        return TrialResult(
            trial_id=f"trial_{config.get('x', 0)}",
            config=config,
            metrics=metrics,
            status=TrialStatus.COMPLETED,
            duration=0.1,
            timestamp=datetime.now(UTC),
        )


class TestPromotionGateIntegration:
    """Tests for PromotionGate integration in orchestrator."""

    def test_orchestrator_accepts_promotion_gate(self) -> None:
        """Test that orchestrator accepts promotion_gate parameter."""
        policy = PromotionPolicy(
            dominance="epsilon_pareto",
            alpha=0.05,
            min_effect={"accuracy": 0.01},
        )
        objectives = [ObjectiveSpec("accuracy", "maximize")]
        gate = PromotionGate(policy, objectives)

        orchestrator = OptimizationOrchestrator(
            optimizer=_MockOptimizer(),
            evaluator=_MockEvaluator(),
            max_trials=3,
            promotion_gate=gate,
        )

        assert orchestrator._promotion_gate is gate
        assert orchestrator._config_metrics_history == {}
        assert orchestrator._incumbent_config_hash is None

    def test_orchestrator_without_promotion_gate(self) -> None:
        """Test that orchestrator works without promotion_gate."""
        orchestrator = OptimizationOrchestrator(
            optimizer=_MockOptimizer(),
            evaluator=_MockEvaluator(),
            max_trials=3,
        )

        assert orchestrator._promotion_gate is None
        assert orchestrator._config_metrics_history == {}

    def test_track_trial_metrics(self) -> None:
        """Test that trial metrics are tracked per config hash."""
        orchestrator = OptimizationOrchestrator(
            optimizer=_MockOptimizer(),
            evaluator=_MockEvaluator(),
            max_trials=3,
        )

        trial1 = TrialResult(
            trial_id="trial_1",
            config={"x": 1},
            metrics={"accuracy": 0.8, "latency": 100},
            status=TrialStatus.COMPLETED,
            duration=0.1,
            timestamp=datetime.now(UTC),
        )

        hash1 = orchestrator._track_trial_metrics(trial1)

        assert hash1 in orchestrator._config_metrics_history
        assert orchestrator._config_metrics_history[hash1]["accuracy"] == [0.8]
        assert orchestrator._config_metrics_history[hash1]["latency"] == [100.0]

        # Add another trial with same config
        trial2 = TrialResult(
            trial_id="trial_2",
            config={"x": 1},
            metrics={"accuracy": 0.85, "latency": 95},
            status=TrialStatus.COMPLETED,
            duration=0.1,
            timestamp=datetime.now(UTC),
        )

        hash2 = orchestrator._track_trial_metrics(trial2)

        assert hash1 == hash2  # Same config, same hash
        assert orchestrator._config_metrics_history[hash1]["accuracy"] == [0.8, 0.85]
        assert orchestrator._config_metrics_history[hash1]["latency"] == [100.0, 95.0]

    def test_simple_is_better_maximize(self) -> None:
        """Test simple comparison for maximize objectives."""
        orchestrator = OptimizationOrchestrator(
            optimizer=_MockOptimizer(),
            evaluator=_MockEvaluator(),
            max_trials=3,
        )

        # No best trial yet - always better
        trial1 = TrialResult(
            trial_id="trial_1",
            config={"x": 1},
            metrics={"accuracy": 0.8},
            status=TrialStatus.COMPLETED,
            duration=0.1,
            timestamp=datetime.now(UTC),
        )
        assert orchestrator._simple_is_better(trial1) is True

        # Set best trial
        orchestrator._best_trial_cached = trial1

        # Better trial
        trial2 = TrialResult(
            trial_id="trial_2",
            config={"x": 2},
            metrics={"accuracy": 0.9},
            status=TrialStatus.COMPLETED,
            duration=0.1,
            timestamp=datetime.now(UTC),
        )
        assert orchestrator._simple_is_better(trial2) is True

        # Worse trial
        trial3 = TrialResult(
            trial_id="trial_3",
            config={"x": 3},
            metrics={"accuracy": 0.7},
            status=TrialStatus.COMPLETED,
            duration=0.1,
            timestamp=datetime.now(UTC),
        )
        assert orchestrator._simple_is_better(trial3) is False

    def test_evaluate_promotion_without_gate(self) -> None:
        """Test promotion evaluation falls back to simple comparison without gate."""
        orchestrator = OptimizationOrchestrator(
            optimizer=_MockOptimizer(),
            evaluator=_MockEvaluator(),
            max_trials=3,
        )

        trial = TrialResult(
            trial_id="trial_1",
            config={"x": 1},
            metrics={"accuracy": 0.8},
            status=TrialStatus.COMPLETED,
            duration=0.1,
            timestamp=datetime.now(UTC),
        )

        # No incumbent - should promote
        assert orchestrator._evaluate_promotion("hash1", trial) is True

    def test_evaluate_promotion_with_gate_promotes(self) -> None:
        """Test promotion evaluation uses gate and promotes candidate."""
        policy = PromotionPolicy(
            dominance="epsilon_pareto",
            alpha=0.05,
            min_effect={"accuracy": 0.01},
        )
        objectives = [ObjectiveSpec("accuracy", "maximize")]
        gate = PromotionGate(policy, objectives)

        orchestrator = OptimizationOrchestrator(
            optimizer=_MockOptimizer(),
            evaluator=_MockEvaluator(),
            max_trials=3,
            promotion_gate=gate,
        )

        # Set up incumbent
        orchestrator._incumbent_config_hash = "incumbent_hash"
        orchestrator._config_metrics_history["incumbent_hash"] = {
            "accuracy": [0.75, 0.76, 0.74, 0.75, 0.76]
        }

        # Set up candidate with better metrics
        orchestrator._config_metrics_history["candidate_hash"] = {
            "accuracy": [0.85, 0.86, 0.87, 0.85, 0.86]
        }

        trial = TrialResult(
            trial_id="trial_1",
            config={"x": 2},
            metrics={"accuracy": 0.86},
            status=TrialStatus.COMPLETED,
            duration=0.1,
            timestamp=datetime.now(UTC),
        )

        # Should promote because candidate has significantly better metrics
        result = orchestrator._evaluate_promotion("candidate_hash", trial)
        assert result is True

    def test_evaluate_promotion_with_gate_insufficient_samples(self) -> None:
        """Test promotion falls back to simple comparison with insufficient samples."""
        policy = PromotionPolicy(
            dominance="epsilon_pareto",
            alpha=0.05,
            min_effect={"accuracy": 0.01},
        )
        objectives = [ObjectiveSpec("accuracy", "maximize")]
        gate = PromotionGate(policy, objectives)

        orchestrator = OptimizationOrchestrator(
            optimizer=_MockOptimizer(),
            evaluator=_MockEvaluator(),
            max_trials=3,
            promotion_gate=gate,
        )

        # Set up incumbent with only 1 sample (insufficient)
        orchestrator._incumbent_config_hash = "incumbent_hash"
        orchestrator._config_metrics_history["incumbent_hash"] = {"accuracy": [0.75]}

        # Candidate also has only 1 sample
        orchestrator._config_metrics_history["candidate_hash"] = {"accuracy": [0.85]}

        trial = TrialResult(
            trial_id="trial_1",
            config={"x": 2},
            metrics={"accuracy": 0.85},
            status=TrialStatus.COMPLETED,
            duration=0.1,
            timestamp=datetime.now(UTC),
        )

        # Set best trial for simple comparison
        orchestrator._best_trial_cached = TrialResult(
            trial_id="trial_0",
            config={"x": 1},
            metrics={"accuracy": 0.75},
            status=TrialStatus.COMPLETED,
            duration=0.1,
            timestamp=datetime.now(UTC),
        )

        # Should fall back to simple comparison and promote (0.85 > 0.75)
        result = orchestrator._evaluate_promotion("candidate_hash", trial)
        assert result is True

    def test_update_best_trial_cache_tracks_incumbent(self) -> None:
        """Test that update_best_trial_cache tracks incumbent config hash."""
        orchestrator = OptimizationOrchestrator(
            optimizer=_MockOptimizer(),
            evaluator=_MockEvaluator(),
            max_trials=3,
        )

        trial = TrialResult(
            trial_id="trial_1",
            config={"x": 1},
            metrics={"accuracy": 0.8},
            status=TrialStatus.COMPLETED,
            duration=0.1,
            timestamp=datetime.now(UTC),
        )

        orchestrator._update_best_trial_cache(trial)

        assert orchestrator._best_trial_cached is trial
        assert orchestrator._incumbent_config_hash is not None
        assert (
            "accuracy"
            in orchestrator._config_metrics_history[orchestrator._incumbent_config_hash]
        )


class TestPromotionGateDecisionLogging:
    """Tests for promotion decision logging."""

    def test_promotion_logs_on_promote(self) -> None:
        """Test that promotion decision is logged when promoting."""
        policy = PromotionPolicy(
            dominance="epsilon_pareto",
            alpha=0.05,
            min_effect={"accuracy": 0.01},
        )
        objectives = [ObjectiveSpec("accuracy", "maximize")]
        gate = PromotionGate(policy, objectives)

        orchestrator = OptimizationOrchestrator(
            optimizer=_MockOptimizer(),
            evaluator=_MockEvaluator(),
            max_trials=3,
            promotion_gate=gate,
        )

        # Set up metrics for promotion
        orchestrator._incumbent_config_hash = "incumbent_hash"
        orchestrator._config_metrics_history["incumbent_hash"] = {
            "accuracy": [0.70, 0.71, 0.72, 0.70, 0.71]
        }
        orchestrator._config_metrics_history["candidate_hash"] = {
            "accuracy": [0.90, 0.91, 0.92, 0.90, 0.91]
        }

        trial = TrialResult(
            trial_id="trial_1",
            config={"x": 2},
            metrics={"accuracy": 0.91},
            status=TrialStatus.COMPLETED,
            duration=0.1,
            timestamp=datetime.now(UTC),
        )

        with patch("traigent.core.orchestrator.logger") as mock_logger:
            result = orchestrator._evaluate_promotion("candidate_hash", trial)
            assert result is True
            # Check that info was logged for promotion
            mock_logger.info.assert_called()
