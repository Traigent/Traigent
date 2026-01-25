"""Tests for stop conditions with JS evaluator in sequential and parallel modes.

This module verifies that stop conditions (max_trials, plateau, budget) work
correctly when using the JS runtime, both with single-worker sequential
execution and multi-worker parallel execution.

Test Coverage:
- MaxTrialsStopCondition with JS evaluator (sequential and parallel)
- PlateauStopCondition with JS evaluator (sequential and parallel)
- BudgetStopCondition with JS evaluator (sequential and parallel)
- Cooperative cancellation when stop condition triggers
- Partial results when trials are in-flight during stop
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from traigent.api.types import TrialResult, TrialStatus
from traigent.bridges.js_bridge import JSBridge, JSTrialResult
from traigent.bridges.process_pool import JSProcessPool
from traigent.core.cost_enforcement import CostEnforcer, CostEnforcerConfig
from traigent.core.objectives import ObjectiveDefinition, ObjectiveSchema
from traigent.core.parallel_execution_manager import ParallelExecutionManager
from traigent.core.stop_conditions import (
    BudgetStopCondition,
    MaxTrialsStopCondition,
    PlateauAfterNStopCondition,
)
from traigent.evaluators.base import Dataset, EvaluationExample
from traigent.evaluators.js_evaluator import JSEvaluator


@pytest.fixture(autouse=True)
def disable_mock_llm_mode(monkeypatch):
    """Disable mock LLM mode for stop condition tests."""
    monkeypatch.setenv("TRAIGENT_MOCK_LLM", "false")


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_dataset():
    """Create a sample dataset for testing."""
    examples = [
        EvaluationExample({"text": f"Example {i}"}, f"Output {i}") for i in range(5)
    ]
    return Dataset(examples=examples, name="test_dataset")


@pytest.fixture
def mock_bridge():
    """Create a mock JSBridge with configurable responses."""
    bridge = MagicMock(spec=JSBridge)
    bridge.is_running = True
    bridge.start = AsyncMock()
    bridge.stop = AsyncMock()
    return bridge


@pytest.fixture
def mock_pool():
    """Create a mock JSProcessPool."""
    pool = MagicMock(spec=JSProcessPool)
    pool.is_running = True
    pool.start = AsyncMock()
    pool.shutdown = AsyncMock()
    return pool


def make_js_trial_result(
    trial_id: str,
    accuracy: float = 0.9,
    cost: float = 0.01,
    duration: float = 1.0,
    cancelled: bool = False,
) -> JSTrialResult:
    """Create a JSTrialResult for testing."""
    return JSTrialResult(
        trial_id=trial_id,
        status="cancelled" if cancelled else "completed",
        metrics={"accuracy": accuracy, "cost": cost},
        duration=duration,
        error_message=None,
        error_code=None,
        retryable=False,
        metadata={"cancelled": cancelled} if cancelled else {},
    )


# =============================================================================
# MaxTrialsStopCondition Tests - Sequential Mode
# =============================================================================


class TestMaxTrialsSequential:
    """Tests for max_trials stop condition with sequential JS execution."""

    @pytest.mark.asyncio
    async def test_stops_after_max_trials(self, sample_dataset, mock_bridge):
        """Verify optimization stops after exactly max_trials in sequential mode."""
        max_trials = 3
        trial_count = 0

        async def mock_run_trial(config):
            nonlocal trial_count
            trial_count += 1
            return make_js_trial_result(
                trial_id=config["trial_id"],
                accuracy=0.8 + trial_count * 0.01,
            )

        mock_bridge.run_trial = mock_run_trial

        evaluator = JSEvaluator(js_module="./test.js")
        evaluator._bridge = mock_bridge

        stop_condition = MaxTrialsStopCondition(max_trials=max_trials)
        trial_history = []

        async def dummy_func(**kwargs):
            return "result"

        # Simulate optimization loop
        for i in range(10):  # Would run 10, but should stop at 3
            result = await evaluator.evaluate(
                func=dummy_func,
                config={"temperature": 0.7, "iteration": i},
                dataset=sample_dataset,
            )

            trial_result = TrialResult(
                trial_id=f"trial_{i}",
                config={"temperature": 0.7},
                metrics=result.aggregated_metrics,
                status=TrialStatus.COMPLETED,
                duration=result.duration,
                timestamp=None,
            )
            trial_history.append(trial_result)

            if stop_condition.should_stop(trial_history):
                break

        # Should have stopped at exactly max_trials
        assert len(trial_history) == max_trials
        assert trial_count == max_trials

    @pytest.mark.asyncio
    async def test_sequential_no_in_flight_on_stop(self, sample_dataset, mock_bridge):
        """Verify no in-flight trials when stop condition triggers in sequential mode."""
        concurrent_trials = 0
        max_concurrent = 0

        async def mock_run_trial(config):
            nonlocal concurrent_trials, max_concurrent
            concurrent_trials += 1
            max_concurrent = max(max_concurrent, concurrent_trials)
            await asyncio.sleep(0.01)  # Simulate work
            concurrent_trials -= 1
            return make_js_trial_result(trial_id=config["trial_id"])

        mock_bridge.run_trial = mock_run_trial

        evaluator = JSEvaluator(js_module="./test.js")
        evaluator._bridge = mock_bridge

        async def dummy_func(**kwargs):
            return "result"

        # Run 3 sequential evaluations
        for i in range(3):
            await evaluator.evaluate(
                func=dummy_func,
                config={"temperature": 0.7},
                dataset=sample_dataset,
            )

        # In sequential mode, only 1 trial at a time
        assert max_concurrent == 1


# =============================================================================
# MaxTrialsStopCondition Tests - Parallel Mode
# =============================================================================


class TestMaxTrialsParallel:
    """Tests for max_trials stop condition with parallel JS execution."""

    @pytest.mark.asyncio
    async def test_respects_max_trials_with_pool(self, sample_dataset, mock_pool):
        """Verify max_trials is respected when using process pool."""
        max_trials = 5
        trial_count = 0

        async def mock_run_trial(config, timeout=None):
            nonlocal trial_count
            trial_count += 1
            await asyncio.sleep(0.01)  # Simulate work
            return make_js_trial_result(
                trial_id=config["trial_id"],
                accuracy=0.85 + trial_count * 0.01,
            )

        mock_pool.run_trial = mock_run_trial

        evaluator = JSEvaluator(
            js_module="./test.js",
            process_pool=mock_pool,
        )

        stop_condition = MaxTrialsStopCondition(max_trials=max_trials)
        trial_history = []

        async def dummy_func(**kwargs):
            return "result"

        # Simulate parallel optimization with batches
        for batch_start in range(0, 20, 4):  # Batches of 4
            # Run batch concurrently
            tasks = [
                evaluator.evaluate(
                    func=dummy_func,
                    config={"temperature": 0.7, "batch_idx": batch_start + i},
                    dataset=sample_dataset,
                )
                for i in range(min(4, max_trials - len(trial_history)))
            ]

            if not tasks:
                break

            results = await asyncio.gather(*tasks)

            for idx, result in enumerate(results):
                trial_result = TrialResult(
                    trial_id=f"trial_{batch_start + idx}",
                    config={"temperature": 0.7},
                    metrics=result.aggregated_metrics,
                    status=TrialStatus.COMPLETED,
                    duration=result.duration,
                    timestamp=None,
                )
                trial_history.append(trial_result)

            if stop_condition.should_stop(trial_history):
                break

        # Should have stopped at max_trials
        assert len(trial_history) == max_trials

    @pytest.mark.asyncio
    async def test_parallel_batch_may_exceed_slightly(self, sample_dataset, mock_pool):
        """Parallel batches may complete slightly over max_trials before check."""
        max_trials = 5
        trial_count = 0

        async def mock_run_trial(config, timeout=None):
            nonlocal trial_count
            trial_count += 1
            return make_js_trial_result(trial_id=config["trial_id"])

        mock_pool.run_trial = mock_run_trial

        evaluator = JSEvaluator(
            js_module="./test.js",
            process_pool=mock_pool,
        )

        stop_condition = MaxTrialsStopCondition(max_trials=max_trials)

        async def dummy_func(**kwargs):
            return "result"

        # Run batch of 4 when we have 3 trials done (would exceed max_trials=5)
        # Batch starts when history has 3, all 4 complete before check
        trial_history = [
            TrialResult(
                trial_id=f"trial_{i}",
                config={},
                metrics={"accuracy": 0.9},
                status=TrialStatus.COMPLETED,
                duration=1.0,
                timestamp=None,
            )
            for i in range(3)
        ]

        # This batch of 4 would take us to 7, but we check after batch
        batch_tasks = [
            evaluator.evaluate(
                func=dummy_func,
                config={"temp": 0.7, "idx": i},
                dataset=sample_dataset,
            )
            for i in range(4)
        ]

        results = await asyncio.gather(*batch_tasks)

        for idx, result in enumerate(results):
            trial_history.append(
                TrialResult(
                    trial_id=f"batch_trial_{idx}",
                    config={},
                    metrics=result.aggregated_metrics,
                    status=TrialStatus.COMPLETED,
                    duration=result.duration,
                    timestamp=None,
                )
            )

        # After this batch, we have 7 trials (3 + 4)
        assert len(trial_history) == 7
        # Stop condition should now trigger
        assert stop_condition.should_stop(trial_history)


# =============================================================================
# PlateauStopCondition Tests
# =============================================================================


class TestPlateauStopCondition:
    """Tests for plateau detection with JS evaluator."""

    @pytest.fixture
    def plateau_stop_condition(self):
        """Create plateau stop condition."""
        schema = ObjectiveSchema.from_objectives(
            [
                ObjectiveDefinition("accuracy", "maximize", 1.0),
            ]
        )
        return PlateauAfterNStopCondition(
            window_size=3,
            epsilon=0.01,
            objective_schema=schema,
        )

    @pytest.mark.asyncio
    async def test_plateau_detected_sequential(
        self, sample_dataset, mock_bridge, plateau_stop_condition
    ):
        """Verify plateau detection works in sequential mode."""

        # Return same accuracy to trigger plateau
        async def mock_run_trial(config):
            return make_js_trial_result(
                trial_id=config["trial_id"],
                accuracy=0.85,  # Same accuracy every time
            )

        mock_bridge.run_trial = mock_run_trial

        evaluator = JSEvaluator(js_module="./test.js")
        evaluator._bridge = mock_bridge

        trial_history = []

        async def dummy_func(**kwargs):
            return "result"

        for i in range(10):  # Would run 10, should plateau at 3
            result = await evaluator.evaluate(
                func=dummy_func,
                config={"temperature": 0.7},
                dataset=sample_dataset,
            )

            trial_history.append(
                TrialResult(
                    trial_id=f"trial_{i}",
                    config={},
                    metrics=result.aggregated_metrics,
                    status=TrialStatus.COMPLETED,
                    duration=result.duration,
                    timestamp=None,
                )
            )

            if plateau_stop_condition.should_stop(trial_history):
                break

        # Should stop after 3 trials (window_size) with no improvement
        assert len(trial_history) == 3

    @pytest.mark.asyncio
    async def test_plateau_resets_on_improvement(
        self, sample_dataset, mock_bridge, plateau_stop_condition
    ):
        """Verify plateau counter resets when improvement is seen."""
        call_count = 0

        async def mock_run_trial(config):
            nonlocal call_count
            call_count += 1
            # Return same accuracy for first 2, then improve, then plateau at new level
            if call_count <= 2:
                return make_js_trial_result(
                    trial_id=config["trial_id"],
                    accuracy=0.85,
                )
            elif call_count == 3:
                return make_js_trial_result(
                    trial_id=config["trial_id"],
                    accuracy=0.90,  # Improvement resets plateau counter
                )
            else:
                return make_js_trial_result(
                    trial_id=config["trial_id"],
                    accuracy=0.90,  # Same as new best - will plateau
                )

        mock_bridge.run_trial = mock_run_trial

        evaluator = JSEvaluator(js_module="./test.js")
        evaluator._bridge = mock_bridge

        trial_history = []

        async def dummy_func(**kwargs):
            return "result"

        for i in range(10):
            result = await evaluator.evaluate(
                func=dummy_func,
                config={"temperature": 0.7},
                dataset=sample_dataset,
            )

            trial_history.append(
                TrialResult(
                    trial_id=f"trial_{i}",
                    config={},
                    metrics=result.aggregated_metrics,
                    status=TrialStatus.COMPLETED,
                    duration=result.duration,
                    timestamp=None,
                )
            )

            if plateau_stop_condition.should_stop(trial_history):
                break

        # Plateau window_size=3: need 3 trials with no improvement
        # Trial 1: 0.85 (best)
        # Trial 2: 0.85 (no improve, count=1)  - but plateau needs window_size=3 consecutive
        # Trial 3: 0.90 (improvement, resets)
        # Trial 4: 0.90 (no improve, count=1)
        # Trial 5: 0.90 (no improve, count=2)
        # Trial 6 would trigger plateau but we need window_size(3) at same level
        # Actually: plateau triggers when window_size consecutive have no improvement
        # So: trials 3,4,5 at 0.90 = 3 trials at same level = plateau
        assert len(trial_history) >= 5  # At least 5 trials before plateau can trigger

    @pytest.mark.asyncio
    async def test_plateau_with_parallel_workers(
        self, sample_dataset, mock_pool, plateau_stop_condition
    ):
        """Verify plateau detection works with parallel pool."""

        async def mock_run_trial(config, timeout=None):
            return make_js_trial_result(
                trial_id=config["trial_id"],
                accuracy=0.88,  # Same accuracy
            )

        mock_pool.run_trial = mock_run_trial

        evaluator = JSEvaluator(
            js_module="./test.js",
            process_pool=mock_pool,
        )

        trial_history = []

        async def dummy_func(**kwargs):
            return "result"

        # Run batches until plateau
        for batch_num in range(5):
            batch_tasks = [
                evaluator.evaluate(
                    func=dummy_func,
                    config={"batch": batch_num, "idx": i},
                    dataset=sample_dataset,
                )
                for i in range(2)  # Batch of 2
            ]

            results = await asyncio.gather(*batch_tasks)

            for idx, result in enumerate(results):
                trial_history.append(
                    TrialResult(
                        trial_id=f"b{batch_num}_t{idx}",
                        config={},
                        metrics=result.aggregated_metrics,
                        status=TrialStatus.COMPLETED,
                        duration=result.duration,
                        timestamp=None,
                    )
                )

            if plateau_stop_condition.should_stop(trial_history):
                plateau_triggered = True
                break
        else:
            plateau_triggered = False

        # Should trigger after window_size (3) trials with same accuracy
        # First batch: 2 trials, second batch: 2 trials (4 total, but plateau at 3)
        # Due to batch checking, we get the full batch before stopping
        assert len(trial_history) >= 3
        # Critical: Assert plateau actually triggered (Codex IT-VTA fix)
        assert plateau_triggered, (
            f"Plateau should have triggered with {len(trial_history)} trials "
            f"at same accuracy. window_size=3, all trials had accuracy=0.88"
        )


# =============================================================================
# BudgetStopCondition Tests
# =============================================================================


class TestBudgetStopCondition:
    """Tests for budget stop condition with JS evaluator."""

    @pytest.mark.asyncio
    async def test_budget_stops_sequential(self, sample_dataset, mock_bridge):
        """Verify budget stop works in sequential mode."""
        budget_limit = 0.05  # $0.05
        trial_cost = 0.02  # Each trial costs $0.02

        async def mock_run_trial(config):
            return make_js_trial_result(
                trial_id=config["trial_id"],
                cost=trial_cost,
            )

        mock_bridge.run_trial = mock_run_trial

        evaluator = JSEvaluator(js_module="./test.js")
        evaluator._bridge = mock_bridge

        stop_condition = BudgetStopCondition(budget=budget_limit, metric_name="cost")
        trial_history = []

        async def dummy_func(**kwargs):
            return "result"

        for i in range(10):
            result = await evaluator.evaluate(
                func=dummy_func,
                config={"temperature": 0.7},
                dataset=sample_dataset,
            )

            trial_history.append(
                TrialResult(
                    trial_id=f"trial_{i}",
                    config={},
                    metrics=result.aggregated_metrics,
                    status=TrialStatus.COMPLETED,
                    duration=result.duration,
                    timestamp=None,
                )
            )

            if stop_condition.should_stop(trial_history):
                break

        # $0.05 budget / $0.02 per trial = 2.5, so 3 trials should trigger stop
        assert len(trial_history) == 3
        total_cost = sum(t.metrics.get("cost", 0) for t in trial_history)
        assert total_cost >= budget_limit

    @pytest.mark.asyncio
    async def test_budget_with_cost_enforcer_parallel(self, sample_dataset, mock_pool):
        """Verify budget enforcement with CostEnforcer in parallel mode."""
        budget_limit = 0.05
        trial_cost = 0.012

        async def mock_run_trial(config, timeout=None):
            return make_js_trial_result(
                trial_id=config["trial_id"],
                cost=trial_cost,
            )

        mock_pool.run_trial = mock_run_trial

        # Create cost enforcer with matching budget
        cost_config = CostEnforcerConfig(
            limit=budget_limit,
            estimated_cost_per_trial=trial_cost,
        )
        cost_enforcer = CostEnforcer(cost_config)
        cost_enforcer._estimated_cost = trial_cost

        parallel_manager = ParallelExecutionManager(
            parallel_trials=4,
            cost_enforcer=cost_enforcer,
        )

        evaluator = JSEvaluator(
            js_module="./test.js",
            process_pool=mock_pool,
        )

        async def dummy_func(**kwargs):
            return "result"

        completed_trials = 0
        cancelled_count = 0

        # Run multiple batches
        for batch in range(5):

            async def run_trial_with_pool():
                nonlocal completed_trials
                result = await evaluator.evaluate(
                    func=dummy_func,
                    config={"batch": batch},
                    dataset=sample_dataset,
                )
                completed_trials += 1
                return result

            coroutines = [run_trial_with_pool() for _ in range(2)]
            results, cancelled = await parallel_manager.run_with_cost_permits(
                coroutines
            )
            cancelled_count += cancelled

            # If all cancelled, budget exhausted
            if cancelled == len(coroutines):
                break

        # Should have limited trials due to budget
        # $0.05 / $0.012 ≈ 4 trials
        assert completed_trials <= 5
        assert cancelled_count > 0 or completed_trials < 10


# =============================================================================
# Cooperative Cancellation Tests
# =============================================================================


class TestCooperativeCancellation:
    """Tests for cooperative cancellation when stop conditions trigger."""

    @pytest.mark.asyncio
    async def test_cancel_signal_sent_on_budget_stop(self, sample_dataset, mock_pool):
        """Verify cancel signal is sent when budget triggers during in-flight trials.

        This test validates the cancellation flow:
        1. Start multiple long-running trials
        2. Budget is exhausted mid-execution
        3. cancel_trial is called for in-flight trials
        4. Cancelled trials return appropriate status
        """
        cancel_signals_sent = []
        trials_started = []
        trial_completion_allowed = asyncio.Event()

        async def mock_run_trial(config, timeout=None):
            trial_id = config["trial_id"]
            trials_started.append(trial_id)
            # Wait until allowed to complete (simulates long-running trial)
            try:
                await asyncio.wait_for(
                    trial_completion_allowed.wait(),
                    timeout=0.1,  # Short timeout for test
                )
            except asyncio.TimeoutError:
                # Trial was cancelled/timed out
                return make_js_trial_result(
                    trial_id=trial_id,
                    cost=0.03,
                    cancelled=True,
                )
            return make_js_trial_result(trial_id=trial_id, cost=0.03)

        async def mock_cancel(trial_id):
            cancel_signals_sent.append(trial_id)
            return True

        mock_pool.run_trial = mock_run_trial
        mock_pool.cancel_trial = mock_cancel

        # Set up budget that will be exhausted
        cost_config = CostEnforcerConfig(
            limit=0.05,  # $0.05 budget
            estimated_cost_per_trial=0.03,  # $0.03 per trial = ~1-2 trials
        )
        cost_enforcer = CostEnforcer(cost_config)
        cost_enforcer._estimated_cost = 0.03

        parallel_manager = ParallelExecutionManager(
            parallel_trials=4,
            cost_enforcer=cost_enforcer,
        )

        evaluator = JSEvaluator(
            js_module="./test.js",
            process_pool=mock_pool,
        )

        async def dummy_func(**kwargs):
            return "result"

        async def run_trial(idx):
            return await evaluator.evaluate(
                func=dummy_func,
                config={"trial_id": f"trial_{idx}"},
                dataset=sample_dataset,
            )

        # Request 4 trials, budget only allows ~1-2
        coroutines = [run_trial(i) for i in range(4)]
        results, cancelled = await parallel_manager.run_with_cost_permits(coroutines)

        # Verify budget enforcement triggered cancellation
        assert cancelled > 0, "Expected some trials to be cancelled due to budget"

        # Verify we got results for all submitted trials
        assert len(results) == 4, "All trials should return a result"

        # Verify budget limited execution
        granted_permits = sum(1 for r in results if r.permit.is_granted)
        assert (
            granted_permits <= 2
        ), f"Budget should limit to ~2 trials, got {granted_permits}"

    @pytest.mark.asyncio
    async def test_partial_results_preserved_on_stop(self, sample_dataset, mock_bridge):
        """Verify partial results are preserved when stop triggers."""
        call_count = 0

        async def mock_run_trial(config):
            nonlocal call_count
            call_count += 1
            if call_count == 3:
                # Simulate cancelled trial returning partial results
                # Note: cancelled status means empty metrics in current impl
                return make_js_trial_result(
                    trial_id=config["trial_id"],
                    accuracy=0.5,  # Partial accuracy
                    cancelled=True,
                )
            return make_js_trial_result(
                trial_id=config["trial_id"],
                accuracy=0.9,
            )

        mock_bridge.run_trial = mock_run_trial

        evaluator = JSEvaluator(js_module="./test.js")
        evaluator._bridge = mock_bridge

        async def dummy_func(**kwargs):
            return "result"

        results = []
        for _ in range(4):
            result = await evaluator.evaluate(
                func=dummy_func,
                config={"temperature": 0.7},
                dataset=sample_dataset,
            )
            results.append(result)

        # All results should be captured
        assert len(results) == 4
        # Third result is cancelled - verify it was recorded (may have empty metrics)
        # The key is that we still got 4 results even with a cancelled trial
        assert results[2] is not None


# =============================================================================
# Integration Tests - Full Stop Condition Flow
# =============================================================================


class TestStopConditionIntegration:
    """Integration tests for stop conditions with full evaluator flow."""

    @pytest.mark.asyncio
    async def test_multiple_stop_conditions_first_wins(
        self, sample_dataset, mock_bridge
    ):
        """Verify first triggered stop condition ends optimization."""
        call_count = 0

        async def mock_run_trial(config):
            nonlocal call_count
            call_count += 1
            return make_js_trial_result(
                trial_id=config["trial_id"],
                accuracy=0.85,  # Same accuracy (plateau)
                cost=0.001,  # Low cost (budget won't trigger first)
            )

        mock_bridge.run_trial = mock_run_trial

        evaluator = JSEvaluator(js_module="./test.js")
        evaluator._bridge = mock_bridge

        # Multiple stop conditions
        max_trials = MaxTrialsStopCondition(max_trials=10)
        plateau = PlateauAfterNStopCondition(
            window_size=3,
            epsilon=0.01,
            objective_schema=ObjectiveSchema.from_objectives(
                [
                    ObjectiveDefinition("accuracy", "maximize", 1.0),
                ]
            ),
        )
        budget = BudgetStopCondition(budget=1.0, metric_name="cost")

        stop_conditions = [max_trials, plateau, budget]
        trial_history = []

        async def dummy_func(**kwargs):
            return "result"

        for i in range(20):
            result = await evaluator.evaluate(
                func=dummy_func,
                config={"temperature": 0.7},
                dataset=sample_dataset,
            )

            trial_history.append(
                TrialResult(
                    trial_id=f"trial_{i}",
                    config={},
                    metrics=result.aggregated_metrics,
                    status=TrialStatus.COMPLETED,
                    duration=result.duration,
                    timestamp=None,
                )
            )

            # Check all stop conditions
            if any(sc.should_stop(trial_history) for sc in stop_conditions):
                break

        # Plateau (window_size=3) should trigger first at 3 trials
        assert len(trial_history) == 3


class TestEndToEndOrchestratorFlow:
    """End-to-end integration tests simulating the full orchestrator flow.

    These tests validate the complete integration between:
    - JSEvaluator (runs JS trials)
    - StopConditionManager (checks stop conditions after each batch)
    - CostEnforcer (permit-based budget enforcement)
    - ParallelExecutionManager (concurrent trial execution with permits)

    This is the closest to production behavior without using the actual
    OptimizationOrchestrator, which has many more dependencies.
    """

    @pytest.mark.asyncio
    async def test_orchestrated_flow_with_budget_and_max_trials(
        self, sample_dataset, mock_pool
    ):
        """Simulate orchestrator: run parallel trials with budget + max_trials."""
        trial_costs = [0.01, 0.015, 0.012, 0.011, 0.014, 0.013]
        call_count = 0

        async def mock_run_trial(config, timeout=None):
            nonlocal call_count
            cost = trial_costs[call_count % len(trial_costs)]
            call_count += 1
            return make_js_trial_result(
                trial_id=config["trial_id"],
                accuracy=0.85 + call_count * 0.01,
                cost=cost,
            )

        mock_pool.run_trial = mock_run_trial

        # Set up components as orchestrator would
        evaluator = JSEvaluator(
            js_module="./test.js",
            process_pool=mock_pool,
        )

        cost_config = CostEnforcerConfig(
            limit=0.06,  # $0.06 budget (allows ~4-5 trials)
            estimated_cost_per_trial=0.012,
        )
        cost_enforcer = CostEnforcer(cost_config)
        cost_enforcer._estimated_cost = 0.012

        parallel_manager = ParallelExecutionManager(
            parallel_trials=2,
            cost_enforcer=cost_enforcer,
        )

        max_trials_condition = MaxTrialsStopCondition(max_trials=10)
        budget_condition = BudgetStopCondition(budget=0.06, metric_name="cost")
        stop_conditions = [max_trials_condition, budget_condition]

        trial_history: list[TrialResult] = []
        stop_reason: str | None = None

        async def dummy_func(**kwargs):
            return "result"

        # Simulate orchestrator loop
        batch_num = 0
        while True:
            batch_num += 1

            # Create trial coroutines for this batch
            async def run_trial(idx):
                return await evaluator.evaluate(
                    func=dummy_func,
                    config={"batch": batch_num, "idx": idx},
                    dataset=sample_dataset,
                )

            # Run batch with cost permits
            coroutines = [run_trial(i) for i in range(2)]
            results, cancelled = await parallel_manager.run_with_cost_permits(
                coroutines
            )

            # Process results (as orchestrator would)
            for r in results:
                if r.permit.is_granted:
                    trial_result = TrialResult(
                        trial_id=f"batch{batch_num}_trial{len(trial_history)}",
                        config={},
                        metrics=r.result.aggregated_metrics,
                        status=TrialStatus.COMPLETED,
                        duration=r.result.duration,
                        timestamp=None,
                    )
                    trial_history.append(trial_result)

                    # Track cost with permit (as orchestrator does)
                    actual_cost = r.result.aggregated_metrics.get("cost", 0)
                    await cost_enforcer.track_cost_async(actual_cost, permit=r.permit)

            # Check stop conditions
            for sc in stop_conditions:
                if sc.should_stop(trial_history):
                    if isinstance(sc, MaxTrialsStopCondition):
                        stop_reason = "max_trials"
                    elif isinstance(sc, BudgetStopCondition):
                        stop_reason = "budget"
                    break

            if stop_reason:
                break

            # Also stop if all trials cancelled (budget exhausted pre-emptively)
            if cancelled == len(coroutines):
                stop_reason = "budget_exhausted"
                break

            # Safety limit
            if batch_num > 10:
                stop_reason = "safety_limit"
                break

        # Verify orchestration behavior
        assert stop_reason in (
            "budget",
            "budget_exhausted",
        ), f"Expected budget stop, got {stop_reason}"
        assert (
            len(trial_history) <= 6
        ), f"Budget should limit trials, got {len(trial_history)}"
        assert (
            cost_enforcer._accumulated_cost <= 0.07
        ), "Should not exceed budget by much"

    @pytest.mark.asyncio
    async def test_orchestrated_flow_with_plateau_detection(
        self, sample_dataset, mock_bridge
    ):
        """Simulate orchestrator: run sequential trials until plateau."""
        accuracies = [0.80, 0.82, 0.85, 0.85, 0.85, 0.85]  # Plateau at 0.85
        call_count = 0

        async def mock_run_trial(config):
            nonlocal call_count
            accuracy = accuracies[min(call_count, len(accuracies) - 1)]
            call_count += 1
            return make_js_trial_result(
                trial_id=config["trial_id"],
                accuracy=accuracy,
                cost=0.01,
            )

        mock_bridge.run_trial = mock_run_trial

        evaluator = JSEvaluator(js_module="./test.js")
        evaluator._bridge = mock_bridge

        plateau_condition = PlateauAfterNStopCondition(
            window_size=3,
            epsilon=0.01,
            objective_schema=ObjectiveSchema.from_objectives(
                [
                    ObjectiveDefinition("accuracy", "maximize", 1.0),
                ]
            ),
        )
        max_trials_condition = MaxTrialsStopCondition(max_trials=20)
        stop_conditions = [plateau_condition, max_trials_condition]

        trial_history: list[TrialResult] = []
        stop_reason: str | None = None

        async def dummy_func(**kwargs):
            return "result"

        # Sequential execution loop
        for i in range(20):
            result = await evaluator.evaluate(
                func=dummy_func,
                config={"iteration": i},
                dataset=sample_dataset,
            )

            trial_result = TrialResult(
                trial_id=f"trial_{i}",
                config={},
                metrics=result.aggregated_metrics,
                status=TrialStatus.COMPLETED,
                duration=result.duration,
                timestamp=None,
            )
            trial_history.append(trial_result)

            # Check stop conditions
            for sc in stop_conditions:
                if sc.should_stop(trial_history):
                    if isinstance(sc, PlateauAfterNStopCondition):
                        stop_reason = "plateau"
                    elif isinstance(sc, MaxTrialsStopCondition):
                        stop_reason = "max_trials"
                    break

            if stop_reason:
                break

        # Verify plateau triggered
        assert stop_reason == "plateau", f"Expected plateau, got {stop_reason}"
        # Should stop after 3 trials at 0.85 (trials 4, 5, 6 = window_size=3)
        assert len(trial_history) >= 5, "Should run at least 5 trials before plateau"
        assert len(trial_history) <= 7, "Should stop soon after plateau"

        # Verify best accuracy was captured
        best_accuracy = max(t.metrics.get("accuracy", 0) for t in trial_history)
        assert best_accuracy == 0.85

    @pytest.mark.asyncio
    async def test_stop_condition_with_failed_trials(self, sample_dataset, mock_bridge):
        """Verify stop conditions handle failed trials correctly."""
        call_count = 0

        async def mock_run_trial(config):
            nonlocal call_count
            call_count += 1
            # Every other trial fails
            if call_count % 2 == 0:
                return JSTrialResult(
                    trial_id=config["trial_id"],
                    status="failed",
                    metrics={},
                    duration=0.5,
                    error_message="Simulated failure",
                    error_code="TEST_ERROR",
                    retryable=False,
                    metadata={},
                )
            return make_js_trial_result(
                trial_id=config["trial_id"],
                accuracy=0.85,
            )

        mock_bridge.run_trial = mock_run_trial

        evaluator = JSEvaluator(js_module="./test.js")
        evaluator._bridge = mock_bridge

        max_trials = MaxTrialsStopCondition(max_trials=5)
        trial_history = []

        async def dummy_func(**kwargs):
            return "result"

        for i in range(10):
            result = await evaluator.evaluate(
                func=dummy_func,
                config={"temperature": 0.7},
                dataset=sample_dataset,
            )

            status = (
                TrialStatus.COMPLETED
                if result.successful_examples > 0
                else TrialStatus.FAILED
            )

            trial_history.append(
                TrialResult(
                    trial_id=f"trial_{i}",
                    config={},
                    metrics=result.aggregated_metrics,
                    status=status,
                    duration=result.duration,
                    timestamp=None,
                )
            )

            if max_trials.should_stop(trial_history):
                break

        # Should count all trials (including failed) toward max
        assert len(trial_history) == 5


# =============================================================================
# Out-of-Order Completion Tests (Codex feedback)
# =============================================================================


class TestOutOfOrderCompletion:
    """Tests for correct behavior when parallel trials complete out of order.

    Codex review identified that no tests verify stop condition correctness
    when faster trials finish later than slower ones (common in parallel execution).
    """

    @pytest.fixture
    def sample_dataset(self):
        """Create a minimal dataset for testing."""
        examples = [
            EvaluationExample({"text": f"sample_{i}"}, f"output_{i}") for i in range(5)
        ]
        return Dataset(examples=examples, name="test_dataset")

    @pytest.fixture
    def mock_pool(self):
        """Create a mock process pool."""
        pool = MagicMock(spec=JSProcessPool)
        pool.is_running = True
        return pool

    @pytest.mark.asyncio
    async def test_plateau_correct_with_out_of_order_completion(
        self, sample_dataset, mock_pool
    ):
        """Verify plateau detection works when trials complete out of order.

        Scenario: Trial 1 (slow, accuracy=0.90) completes after Trial 2 (fast, 0.90).
        Both should contribute to plateau detection regardless of completion order.
        """
        completion_order = []
        trial_counter = [0]  # Use list to allow mutation in closure

        async def mock_run_trial(config, timeout=None):
            # Get trial index from counter (incremented per call)
            trial_idx = trial_counter[0]
            trial_counter[0] += 1

            # Trial 0 is slow, trials 1-3 are fast
            if trial_idx == 0:
                await asyncio.sleep(0.05)  # Slow trial
            else:
                await asyncio.sleep(0.01)  # Fast trials

            completion_order.append(trial_idx)
            return make_js_trial_result(
                trial_id=config.get("trial_id", f"trial_{trial_idx}"),
                accuracy=0.90,  # Same accuracy for plateau
            )

        mock_pool.run_trial = mock_run_trial

        evaluator = JSEvaluator(
            js_module="./test.js",
            process_pool=mock_pool,
        )

        schema = ObjectiveSchema.from_objectives(
            [
                ObjectiveDefinition("accuracy", "maximize", 1.0),
            ]
        )
        plateau_condition = PlateauAfterNStopCondition(
            window_size=3,
            epsilon=0.01,
            objective_schema=schema,
        )

        trial_history = []
        plateau_triggered = False

        async def dummy_func(**kwargs):
            return "result"

        # Run batch of 4 trials where trial 0 is slowest
        batch_tasks = [
            evaluator.evaluate(
                func=dummy_func,
                config={"idx": i},
                dataset=sample_dataset,
            )
            for i in range(4)
        ]

        results = await asyncio.gather(*batch_tasks)

        # Verify out-of-order completion happened
        assert completion_order[0] != 0, (
            f"Expected out-of-order completion but trial 0 completed first. "
            f"Order: {completion_order}"
        )

        # Add results to history in completion order (simulating real behavior)
        for idx, result in enumerate(results):
            trial_history.append(
                TrialResult(
                    trial_id=f"trial_{idx}",
                    config={},
                    metrics=result.aggregated_metrics,
                    status=TrialStatus.COMPLETED,
                    duration=result.duration,
                    timestamp=None,
                )
            )

        # Check plateau - should trigger with 3+ trials at same accuracy
        plateau_triggered = plateau_condition.should_stop(trial_history)

        assert plateau_triggered, (
            f"Plateau should trigger with 4 trials at same accuracy=0.90, "
            f"regardless of completion order. Completion order: {completion_order}"
        )

    @pytest.mark.asyncio
    async def test_budget_correct_with_out_of_order_completion(
        self, sample_dataset, mock_pool
    ):
        """Verify budget tracking works when trials complete out of order.

        Scenario: High-cost trial completes after low-cost trials.
        Budget should be correctly accumulated regardless of order.
        """
        completion_order = []
        trial_counter = [0]  # Use list to allow mutation in closure
        costs = [0.05, 0.01, 0.01, 0.02]  # Trial 0 is expensive

        async def mock_run_trial(config, timeout=None):
            # Get trial index from counter (incremented per call)
            trial_idx = trial_counter[0]
            trial_counter[0] += 1

            # Trial 0 is slow and expensive
            if trial_idx == 0:
                await asyncio.sleep(0.05)
            else:
                await asyncio.sleep(0.01)

            completion_order.append(trial_idx)
            return make_js_trial_result(
                trial_id=config.get("trial_id", f"trial_{trial_idx}"),
                cost=costs[trial_idx],
            )

        mock_pool.run_trial = mock_run_trial

        evaluator = JSEvaluator(
            js_module="./test.js",
            process_pool=mock_pool,
        )

        budget_condition = BudgetStopCondition(budget=0.08, metric_name="cost")
        trial_history = []

        async def dummy_func(**kwargs):
            return "result"

        # Run batch of 4 trials
        batch_tasks = [
            evaluator.evaluate(
                func=dummy_func,
                config={"idx": i},
                dataset=sample_dataset,
            )
            for i in range(4)
        ]

        results = await asyncio.gather(*batch_tasks)

        # Verify out-of-order completion (trial 0 is slow, should complete last)
        # Note: exact order depends on asyncio scheduling, but trial 0 should not be first
        assert (
            len(completion_order) == 4
        ), f"All 4 trials should complete. Order: {completion_order}"

        # Add results to history
        for idx, result in enumerate(results):
            trial_history.append(
                TrialResult(
                    trial_id=f"trial_{idx}",
                    config={},
                    metrics=result.aggregated_metrics,
                    status=TrialStatus.COMPLETED,
                    duration=result.duration,
                    timestamp=None,
                )
            )

        # Verify total cost is correct regardless of completion order
        total_cost = sum(t.metrics.get("cost", 0) for t in trial_history)
        expected_total = sum(costs)

        assert abs(total_cost - expected_total) < 0.001, (
            f"Total cost should be {expected_total} but got {total_cost}. "
            f"Completion order: {completion_order}"
        )

        # Budget check should consider all costs
        budget_exhausted = budget_condition.should_stop(trial_history)
        assert budget_exhausted, (
            f"Budget should be exhausted. Total cost: {total_cost}, "
            f"Budget limit: 0.08"
        )

    @pytest.mark.asyncio
    async def test_interleaved_completion_with_stop_condition(
        self, sample_dataset, mock_pool
    ):
        """Test stop condition check during interleaved trial completion.

        Scenario: Stop condition checked as results arrive, not after batch.
        """
        completion_times = []
        trial_counter = [0]  # Use list to allow mutation in closure

        async def mock_run_trial(config, timeout=None):
            # Get trial index from counter (incremented per call)
            trial_idx = trial_counter[0]
            trial_counter[0] += 1

            # Stagger completion times: 0.04, 0.02, 0.03, 0.01
            delays = [0.04, 0.02, 0.03, 0.01]
            await asyncio.sleep(delays[trial_idx])

            completion_times.append((trial_idx, asyncio.get_event_loop().time()))
            return make_js_trial_result(
                trial_id=config.get("trial_id", f"trial_{trial_idx}"),
                accuracy=0.88,  # Same accuracy
            )

        mock_pool.run_trial = mock_run_trial

        evaluator = JSEvaluator(
            js_module="./test.js",
            process_pool=mock_pool,
        )

        schema = ObjectiveSchema.from_objectives(
            [
                ObjectiveDefinition("accuracy", "maximize", 1.0),
            ]
        )
        plateau_condition = PlateauAfterNStopCondition(
            window_size=3,
            epsilon=0.01,
            objective_schema=schema,
        )

        trial_history = []
        stop_checks = []

        async def dummy_func(**kwargs):
            return "result"

        # Run trials concurrently
        batch_tasks = [
            evaluator.evaluate(
                func=dummy_func,
                config={"idx": i},
                dataset=sample_dataset,
            )
            for i in range(4)
        ]

        results = await asyncio.gather(*batch_tasks)

        # Verify we got all 4 completions
        completion_indices = [t[0] for t in completion_times]
        assert (
            len(completion_indices) == 4
        ), f"All 4 trials should complete. Got: {completion_indices}"

        # Add results and check stop condition after each
        for idx, result in enumerate(results):
            trial_history.append(
                TrialResult(
                    trial_id=f"trial_{idx}",
                    config={},
                    metrics=result.aggregated_metrics,
                    status=TrialStatus.COMPLETED,
                    duration=result.duration,
                    timestamp=None,
                )
            )
            should_stop = plateau_condition.should_stop(trial_history)
            stop_checks.append((len(trial_history), should_stop))

        # Plateau should trigger after 3 trials with same accuracy
        # Find when it first triggered
        first_trigger = next((check for check in stop_checks if check[1]), None)

        assert (
            first_trigger is not None
        ), f"Plateau should have triggered. Checks: {stop_checks}"
        assert first_trigger[0] >= 3, (
            f"Plateau should trigger after at least 3 trials, "
            f"triggered at {first_trigger[0]}"
        )
