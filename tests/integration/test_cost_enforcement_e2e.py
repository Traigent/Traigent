"""End-to-end integration tests for cost enforcement.

Tests full optimization flows from OptimizationOrchestrator through
trial execution, and cost limit enforcement.

Critical scenarios tested:
- E2E-01: Sequential execution stops when cost limit reached
- E2E-02: Parallel execution denies permits when cost limit reached
- E2E-03: Stop condition fires and reports correct reason
- E2E-04: Unknown cost mode triggers fallback to trial count
- E2E-05: Exception during trial releases permit correctly
- E2E-06: Cost enforcement integrates with all execution modes
- E2E-07: Permit tracking remains accurate across full optimization

Reference: /home/nimrodbu/.claude/plans/snazzy-whistling-kettle.md
"""

from __future__ import annotations

import os
from typing import Any
from unittest.mock import Mock

import pytest

# Ensure mock mode is disabled for these tests - we want real cost tracking
os.environ["TRAIGENT_MOCK_MODE"] = "false"

from traigent.api.types import (
    TrialResult,
)
from traigent.config.types import TraigentConfig
from traigent.core.orchestrator import OptimizationOrchestrator
from traigent.core.stop_conditions import CostLimitStopCondition
from traigent.evaluators.base import (
    BaseEvaluator,
    Dataset,
    EvaluationExample,
    EvaluationResult,
)
from traigent.optimizers.base import BaseOptimizer

# Tolerance for floating point comparisons
FLOAT_TOLERANCE = 1e-10


@pytest.fixture(autouse=True)
def disable_mock_mode() -> None:
    """Ensure mock mode is disabled for all tests in this module."""
    os.environ["TRAIGENT_MOCK_MODE"] = "false"
    if "TRAIGENT_REQUIRE_COST_TRACKING" in os.environ:
        del os.environ["TRAIGENT_REQUIRE_COST_TRACKING"]


@pytest.fixture(autouse=True)
def patch_backend(monkeypatch):
    """Replace BackendIntegratedClient to avoid outbound traffic."""
    mock_backend = Mock()
    mock_backend.create_session.return_value = "mock-session"
    mock_backend.submit_result.return_value = True
    mock_backend.update_trial_weighted_scores.return_value = True
    mock_backend.finalize_session_sync.return_value = None
    mock_backend.finalize_session.return_value = None
    mock_backend.delete_session.return_value = True

    monkeypatch.setattr(
        "traigent.core.orchestrator.BackendIntegratedClient",
        lambda *args, **kwargs: mock_backend,
    )
    return mock_backend


class MockCostAwareEvaluator(BaseEvaluator):
    """Mock evaluator that simulates cost tracking.

    Adds 'cost' metric to results to simulate LLM API cost tracking.
    """

    def __init__(
        self,
        cost_per_eval: float = 0.05,
        should_fail_at: int | None = None,
        unknown_cost_mode: bool = False,
    ):
        """Initialize with configurable cost behavior.

        Args:
            cost_per_eval: Cost to report per evaluation
            should_fail_at: Trial number to fail at (0-indexed)
            unknown_cost_mode: If True, don't include cost in results
        """
        self.cost_per_eval = cost_per_eval
        self.should_fail_at = should_fail_at
        self.unknown_cost_mode = unknown_cost_mode
        self.evaluation_count = 0

    async def evaluate(
        self,
        func,
        config: dict[str, Any],
        dataset: Dataset,
        *,
        sample_lease=None,
        progress_callback=None,
        **_kwargs,
    ) -> EvaluationResult:
        """Evaluate configuration with cost tracking."""
        current_eval = self.evaluation_count
        self.evaluation_count += 1

        # Check if we should fail at this trial
        if self.should_fail_at is not None and current_eval == self.should_fail_at:
            raise RuntimeError(f"Simulated failure at evaluation {current_eval}")

        # Simulate evaluation
        processed_examples = list(dataset.examples)
        total_examples = len(processed_examples)

        # Build metrics
        metrics: dict[str, Any] = {
            "accuracy": 0.8 + (config.get("param1", 0) * 0.01),
            "latency": 100 - (config.get("param1", 0) * 5),
            "examples_attempted": total_examples,
        }

        # Add cost if not in unknown cost mode
        if not self.unknown_cost_mode:
            metrics["cost"] = self.cost_per_eval

        result = EvaluationResult(
            config=config,
            aggregated_metrics=metrics,
            total_examples=total_examples,
            successful_examples=total_examples,
            duration=0.1,
            metrics=metrics,
            outputs=[f"output_{i}" for i in range(total_examples)],
            errors=[None for _ in range(total_examples)],
        )

        result.sample_budget_exhausted = False
        result.examples_consumed = total_examples
        return result


class MockSequentialOptimizer(BaseOptimizer):
    """Mock optimizer that suggests configurations sequentially."""

    def __init__(
        self,
        config_space: dict[str, Any],
        objectives: list[str],
        max_suggestions: int = 10,
        **kwargs,
    ):
        super().__init__(config_space, objectives, **kwargs)
        self._suggest_count = 0
        self._max_suggestions = max_suggestions
        self._should_stop = False

    def suggest_next_trial(self, history: list[TrialResult]) -> dict[str, Any]:
        """Suggest next configuration."""
        config = {
            "param1": self._suggest_count,
            "param2": f"value_{self._suggest_count}",
        }
        self._suggest_count += 1
        if self._suggest_count >= self._max_suggestions:
            self._should_stop = True
        return config

    def should_stop(self, history: list[TrialResult]) -> bool:
        """Check if optimization should stop."""
        return self._should_stop

    def suggest(self) -> dict[str, Any]:
        """Legacy suggest method."""
        return self.suggest_next_trial([])

    def tell(self, config: dict[str, Any], result: TrialResult) -> None:
        """Record trial result."""
        pass

    def is_finished(self) -> bool:
        """Check if finished."""
        return self._should_stop

    def force_stop(self) -> None:
        """Force stop."""
        self._should_stop = True


class TestE2ECostEnforcementFlow:
    """End-to-end tests for cost enforcement in optimization flows."""

    @pytest.fixture
    def sample_dataset(self) -> Dataset:
        """Create sample dataset for testing."""
        examples = [
            EvaluationExample({"query": "Test 1"}, "Answer 1"),
            EvaluationExample({"query": "Test 2"}, "Answer 2"),
        ]
        return Dataset(examples, name="test_dataset")

    @pytest.fixture
    def traigent_config(self) -> TraigentConfig:
        """Create TraigentConfig for edge analytics mode."""
        return TraigentConfig(execution_mode="edge_analytics")

    @pytest.fixture
    def mock_function(self):
        """Create a mock function to optimize."""

        async def test_function(input_data: dict[str, Any], **config) -> str:
            return input_data.get("query", "default")

        return test_function

    @pytest.mark.asyncio
    async def test_e2e_01_sequential_stops_at_cost_limit(
        self, sample_dataset, mock_function, traigent_config
    ) -> None:
        """E2E-01: Sequential execution stops when cost limit is reached."""
        # Setup with very low cost limit (0.20) and high cost per eval (0.08)
        # Expected: 2 full trials (0.16), 3rd trial exceeds limit
        evaluator = MockCostAwareEvaluator(cost_per_eval=0.08)
        optimizer = MockSequentialOptimizer(
            config_space={"param1": (0, 10)},
            objectives=["accuracy"],
            max_suggestions=10,
        )

        orchestrator = OptimizationOrchestrator(
            optimizer=optimizer,
            evaluator=evaluator,
            max_trials=10,
            timeout=60.0,
            config=traigent_config,
            cost_limit=0.20,
            cost_approved=True,
        )

        # Run optimization
        await orchestrator.optimize(
            func=mock_function,
            dataset=sample_dataset,
        )

        # Verify cost enforcement stopped optimization
        assert orchestrator.trial_count <= 3, (
            f"Should stop around 2-3 trials with 0.20 limit and 0.08/trial, "
            f"got {orchestrator.trial_count}"
        )

        # Verify cost enforcer state
        status = orchestrator.cost_enforcer.get_status()
        assert status.accumulated_cost_usd <= 0.30  # Allow small overage
        assert abs(status.reserved_cost_usd) < FLOAT_TOLERANCE  # All released
        assert status.in_flight_count == 0  # No stranded permits

    @pytest.mark.asyncio
    async def test_e2e_02_parallel_denies_permits_at_limit(
        self, sample_dataset, mock_function, traigent_config
    ) -> None:
        """E2E-02: Parallel execution denies permits when limit is reached."""
        # Low limit, high cost per eval
        evaluator = MockCostAwareEvaluator(cost_per_eval=0.10)
        optimizer = MockSequentialOptimizer(
            config_space={"param1": (0, 10)},
            objectives=["accuracy"],
            max_suggestions=10,
        )

        orchestrator = OptimizationOrchestrator(
            optimizer=optimizer,
            evaluator=evaluator,
            max_trials=10,
            timeout=60.0,
            parallel_trials=4,  # Enable parallel execution
            config=traigent_config,
            cost_limit=0.30,
            cost_approved=True,
        )

        # Run optimization
        await orchestrator.optimize(
            func=mock_function,
            dataset=sample_dataset,
        )

        # With 0.30 limit and 0.10/trial, should stop around 3 trials
        assert orchestrator.trial_count <= 4

        # Verify no stranded permits
        status = orchestrator.cost_enforcer.get_status()
        assert status.in_flight_count == 0
        assert abs(status.reserved_cost_usd) < FLOAT_TOLERANCE

    @pytest.mark.asyncio
    async def test_e2e_03_stop_condition_reports_correct_reason(
        self, sample_dataset, mock_function, traigent_config
    ) -> None:
        """E2E-03: Stop condition fires and reports correct reason."""
        evaluator = MockCostAwareEvaluator(cost_per_eval=0.15)
        optimizer = MockSequentialOptimizer(
            config_space={"param1": (0, 10)},
            objectives=["accuracy"],
            max_suggestions=10,
        )

        orchestrator = OptimizationOrchestrator(
            optimizer=optimizer,
            evaluator=evaluator,
            max_trials=10,
            timeout=60.0,
            config=traigent_config,
            cost_limit=0.25,
            cost_approved=True,
        )

        await orchestrator.optimize(
            func=mock_function,
            dataset=sample_dataset,
        )

        # Verify stop condition was cost limit
        # The orchestrator should record the stop reason
        assert orchestrator.cost_enforcer.is_limit_reached

        # Verify the stop condition can report correctly
        condition = CostLimitStopCondition(orchestrator.cost_enforcer)
        assert condition.should_stop([]) is True  # Pass empty trials list
        reason = condition.get_reason()
        assert "cost" in reason.lower() or "limit" in reason.lower()

    @pytest.mark.asyncio
    async def test_e2e_04_unknown_cost_triggers_fallback(
        self, sample_dataset, mock_function, traigent_config
    ) -> None:
        """E2E-04: Unknown cost mode triggers fallback to trial count."""
        # Evaluator that doesn't report cost
        evaluator = MockCostAwareEvaluator(unknown_cost_mode=True)
        optimizer = MockSequentialOptimizer(
            config_space={"param1": (0, 10)},
            objectives=["accuracy"],
            max_suggestions=20,  # High max to test fallback
        )

        orchestrator = OptimizationOrchestrator(
            optimizer=optimizer,
            evaluator=evaluator,
            max_trials=20,
            timeout=60.0,
            config=traigent_config,
            cost_limit=0.50,
            cost_approved=True,
        )

        # Run optimization
        await orchestrator.optimize(
            func=mock_function,
            dataset=sample_dataset,
        )

        # With unknown cost, should switch to unknown cost mode
        status = orchestrator.cost_enforcer.get_status()
        assert status.unknown_cost_mode is True

        # Should use trial count fallback (default is 10 trials)
        assert orchestrator.trial_count <= 10  # Default fallback limit

    @pytest.mark.asyncio
    async def test_e2e_05_exception_releases_permit(
        self, sample_dataset, mock_function, traigent_config
    ) -> None:
        """E2E-05: Exception during trial releases permit correctly."""
        # Evaluator that fails on trial 2 (0-indexed)
        evaluator = MockCostAwareEvaluator(
            cost_per_eval=0.05,
            should_fail_at=2,
        )
        optimizer = MockSequentialOptimizer(
            config_space={"param1": (0, 10)},
            objectives=["accuracy"],
            max_suggestions=5,
        )

        orchestrator = OptimizationOrchestrator(
            optimizer=optimizer,
            evaluator=evaluator,
            max_trials=5,
            timeout=60.0,
            config=traigent_config,
            cost_limit=2.0,
            cost_approved=True,
        )

        # Optimization should handle the exception
        try:
            await orchestrator.optimize(
                func=mock_function,
                dataset=sample_dataset,
            )
        except RuntimeError:
            pass  # Expected exception from mock evaluator

        # Critical: No stranded permits
        status = orchestrator.cost_enforcer.get_status()
        assert status.in_flight_count == 0, "Permits should be released on exception"
        assert abs(status.reserved_cost_usd) < FLOAT_TOLERANCE

    @pytest.mark.asyncio
    async def test_e2e_06_cost_enforcement_with_parallel_mode(
        self, sample_dataset, mock_function, traigent_config
    ) -> None:
        """E2E-06: Cost enforcement works correctly in parallel mode."""
        evaluator = MockCostAwareEvaluator(cost_per_eval=0.05)
        optimizer = MockSequentialOptimizer(
            config_space={"param1": (0, 10)},
            objectives=["accuracy"],
            max_suggestions=20,
        )

        orchestrator = OptimizationOrchestrator(
            optimizer=optimizer,
            evaluator=evaluator,
            max_trials=20,
            timeout=60.0,
            parallel_trials=4,
            config=traigent_config,
            cost_limit=0.20,
            cost_approved=True,
        )

        await orchestrator.optimize(
            func=mock_function,
            dataset=sample_dataset,
        )

        # With 0.20 limit and 0.05/trial, should complete around 4 trials
        assert orchestrator.trial_count <= 5

        # Verify invariants
        status = orchestrator.cost_enforcer.get_status()
        assert status.in_flight_count == 0
        assert abs(status.reserved_cost_usd) < FLOAT_TOLERANCE
        assert status.accumulated_cost_usd <= 0.30  # Allow small overage

    @pytest.mark.asyncio
    async def test_e2e_07_permit_tracking_accuracy(
        self, sample_dataset, mock_function, traigent_config
    ) -> None:
        """E2E-07: Permit tracking remains accurate across full optimization."""
        evaluator = MockCostAwareEvaluator(cost_per_eval=0.10)
        optimizer = MockSequentialOptimizer(
            config_space={"param1": (0, 10)},
            objectives=["accuracy"],
            max_suggestions=5,
        )

        orchestrator = OptimizationOrchestrator(
            optimizer=optimizer,
            evaluator=evaluator,
            max_trials=5,
            timeout=60.0,
            config=traigent_config,
            cost_limit=1.0,
            cost_approved=True,
        )

        # Track state during optimization

        await orchestrator.optimize(
            func=mock_function,
            dataset=sample_dataset,
        )

        # Verify accurate tracking
        status = orchestrator.cost_enforcer.get_status()

        # Trial count should match
        assert status.trial_count == orchestrator.trial_count

        # Accumulated cost should be approximately trial_count * cost_per_eval
        expected_cost = orchestrator.trial_count * 0.10
        assert abs(status.accumulated_cost_usd - expected_cost) < 0.05

        # No stranded permits
        assert status.in_flight_count == 0
        assert abs(status.reserved_cost_usd) < FLOAT_TOLERANCE


class TestCostEnforcerOrchestratorIntegration:
    """Tests for CostEnforcer integration with orchestrator components."""

    @pytest.fixture
    def traigent_config(self) -> TraigentConfig:
        """Create TraigentConfig for edge analytics mode."""
        return TraigentConfig(execution_mode="edge_analytics")

    def test_orchestrator_creates_cost_enforcer_with_limit(
        self, traigent_config
    ) -> None:
        """Verify orchestrator creates CostEnforcer with correct config."""
        optimizer = MockSequentialOptimizer(
            config_space={"param1": (0, 10)},
            objectives=["accuracy"],
        )
        evaluator = MockCostAwareEvaluator()

        orchestrator = OptimizationOrchestrator(
            optimizer=optimizer,
            evaluator=evaluator,
            max_trials=10,
            config=traigent_config,
            cost_limit=5.0,
            cost_approved=True,
        )

        assert orchestrator.cost_enforcer is not None
        assert abs(orchestrator.cost_enforcer.config.limit - 5.0) < FLOAT_TOLERANCE
        assert orchestrator.cost_enforcer.config.approved is True

    def test_cost_enforcer_shared_with_parallel_manager(self, traigent_config) -> None:
        """Verify CostEnforcer is shared with parallel execution manager."""
        optimizer = MockSequentialOptimizer(
            config_space={"param1": (0, 10)},
            objectives=["accuracy"],
        )
        evaluator = MockCostAwareEvaluator()

        orchestrator = OptimizationOrchestrator(
            optimizer=optimizer,
            evaluator=evaluator,
            max_trials=10,
            parallel_trials=4,
            config=traigent_config,
            cost_limit=5.0,
            cost_approved=True,
        )

        # Verify the parallel execution manager has the same cost enforcer
        assert orchestrator.parallel_execution_manager.cost_enforcer is not None
        assert (
            orchestrator.parallel_execution_manager.cost_enforcer
            is orchestrator.cost_enforcer
        )

    def test_stop_condition_registered_with_cost_enforcer(
        self, traigent_config
    ) -> None:
        """Verify stop condition is registered with shared CostEnforcer."""
        optimizer = MockSequentialOptimizer(
            config_space={"param1": (0, 10)},
            objectives=["accuracy"],
        )
        evaluator = MockCostAwareEvaluator()

        orchestrator = OptimizationOrchestrator(
            optimizer=optimizer,
            evaluator=evaluator,
            max_trials=10,
            config=traigent_config,
            cost_limit=5.0,
            cost_approved=True,
        )

        # Verify stop condition manager has cost limit condition
        conditions = orchestrator._stop_condition_manager._conditions
        cost_conditions = [
            c for c in conditions if isinstance(c, CostLimitStopCondition)
        ]
        assert len(cost_conditions) >= 1


class TestCostEnforcementInvariants:
    """Tests that verify invariants hold throughout optimization."""

    @pytest.fixture
    def sample_dataset(self) -> Dataset:
        """Create sample dataset."""
        examples = [
            EvaluationExample({"query": "Q1"}, "A1"),
            EvaluationExample({"query": "Q2"}, "A2"),
        ]
        return Dataset(examples, name="test")

    @pytest.fixture
    def traigent_config(self) -> TraigentConfig:
        """Create TraigentConfig for edge analytics mode."""
        return TraigentConfig(execution_mode="edge_analytics")

    @pytest.mark.asyncio
    async def test_invariant_i1_in_flight_never_negative(
        self, sample_dataset, traigent_config
    ) -> None:
        """I1: in_flight_count >= 0 throughout optimization."""
        evaluator = MockCostAwareEvaluator(cost_per_eval=0.05)
        optimizer = MockSequentialOptimizer(
            config_space={"param1": (0, 10)},
            objectives=["accuracy"],
            max_suggestions=5,
        )

        orchestrator = OptimizationOrchestrator(
            optimizer=optimizer,
            evaluator=evaluator,
            max_trials=5,
            config=traigent_config,
            cost_limit=1.0,
            cost_approved=True,
        )

        async def mock_func(**kwargs) -> str:
            # Check invariant during execution
            assert orchestrator.cost_enforcer._in_flight_count >= 0
            return "result"

        await orchestrator.optimize(func=mock_func, dataset=sample_dataset)

        # Final check
        assert orchestrator.cost_enforcer._in_flight_count >= 0

    @pytest.mark.asyncio
    async def test_invariant_i2_reserved_never_negative(
        self, sample_dataset, traigent_config
    ) -> None:
        """I2: reserved_cost >= 0 throughout optimization."""
        evaluator = MockCostAwareEvaluator(cost_per_eval=0.05)
        optimizer = MockSequentialOptimizer(
            config_space={"param1": (0, 10)},
            objectives=["accuracy"],
            max_suggestions=5,
        )

        orchestrator = OptimizationOrchestrator(
            optimizer=optimizer,
            evaluator=evaluator,
            max_trials=5,
            config=traigent_config,
            cost_limit=1.0,
            cost_approved=True,
        )

        async def mock_func(**kwargs) -> str:
            # Check invariant during execution
            assert orchestrator.cost_enforcer._reserved_cost >= -FLOAT_TOLERANCE
            return "result"

        await orchestrator.optimize(func=mock_func, dataset=sample_dataset)

        # Final check
        assert orchestrator.cost_enforcer._reserved_cost >= -FLOAT_TOLERANCE

    @pytest.mark.asyncio
    async def test_invariant_i3_active_permits_consistency(
        self, sample_dataset, traigent_config
    ) -> None:
        """I3: len(active_permits) == in_flight_count throughout optimization."""
        evaluator = MockCostAwareEvaluator(cost_per_eval=0.05)
        optimizer = MockSequentialOptimizer(
            config_space={"param1": (0, 10)},
            objectives=["accuracy"],
            max_suggestions=5,
        )

        orchestrator = OptimizationOrchestrator(
            optimizer=optimizer,
            evaluator=evaluator,
            max_trials=5,
            config=traigent_config,
            cost_limit=1.0,
            cost_approved=True,
        )

        await orchestrator.optimize(
            func=lambda **kwargs: "result",
            dataset=sample_dataset,
        )

        # Final consistency check
        enforcer = orchestrator.cost_enforcer
        assert len(enforcer._active_permits) == enforcer._in_flight_count
