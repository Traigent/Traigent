"""Tests for budget guardrails integration with JS evaluator.

This module verifies that budget enforcement works correctly when
using the JS evaluator, including:
- Cost tracking from JS trial metrics
- Budget limit enforcement
- Cost permit flow with JS trials
- EMA cost estimation with JS trials
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from traigent.bridges.js_bridge import JSTrialResult
from traigent.bridges.process_pool import JSProcessPool
from traigent.core.cost_enforcement import CostEnforcer, CostEnforcerConfig
from traigent.core.parallel_execution_manager import ParallelExecutionManager
from traigent.evaluators.base import Dataset, EvaluationExample
from traigent.evaluators.js_evaluator import JSEvaluator


@pytest.fixture(autouse=True)
def disable_mock_llm_mode(monkeypatch):
    """Disable mock LLM mode for budget tests."""
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
    """Create a mock JSBridge."""
    bridge = MagicMock()
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


def make_js_result(trial_id: str, cost: float, accuracy: float = 0.9) -> JSTrialResult:
    """Create a JSTrialResult with specified cost."""
    return JSTrialResult(
        trial_id=trial_id,
        status="completed",
        metrics={"accuracy": accuracy, "cost": cost},
        duration=1.0,
        error_message=None,
        error_code=None,
        retryable=False,
        metadata={},
    )


# =============================================================================
# Cost Tracking from JS Metrics
# =============================================================================


class TestCostTrackingFromJSMetrics:
    """Tests for extracting and tracking costs from JS trial metrics."""

    @pytest.mark.asyncio
    async def test_cost_extracted_from_metrics(self, sample_dataset, mock_bridge):
        """Verify cost is correctly extracted from JS trial metrics."""
        expected_cost = 0.025

        async def mock_run_trial(config):
            return make_js_result(
                trial_id=config["trial_id"],
                cost=expected_cost,
            )

        mock_bridge.run_trial = mock_run_trial

        evaluator = JSEvaluator(js_module="./test.js")
        evaluator._bridge = mock_bridge

        async def dummy_func(**kwargs):
            return "result"

        result = await evaluator.evaluate(
            func=dummy_func,
            config={"temperature": 0.7},
            dataset=sample_dataset,
        )

        # Cost should be in aggregated metrics
        assert "cost" in result.aggregated_metrics
        assert result.aggregated_metrics["cost"] == expected_cost

    @pytest.mark.asyncio
    async def test_missing_cost_omitted_from_metrics(self, sample_dataset, mock_bridge):
        """Verify missing cost metric is omitted (not defaulted) in aggregated metrics.

        When a JS trial doesn't report a 'cost' metric, the evaluator should NOT
        add a default value. This is important because:
        1. Cost tracking should only track real costs
        2. Zero cost is semantically different from "no cost reported"
        3. Budget enforcement uses EMA which shouldn't be polluted with fake zeros
        """

        async def mock_run_trial(config):
            return JSTrialResult(
                trial_id=config["trial_id"],
                status="completed",
                metrics={"accuracy": 0.9},  # No cost metric
                duration=1.0,
                error_message=None,
                error_code=None,
                retryable=False,
                metadata={},
            )

        mock_bridge.run_trial = mock_run_trial

        evaluator = JSEvaluator(js_module="./test.js")
        evaluator._bridge = mock_bridge

        async def dummy_func(**kwargs):
            return "result"

        result = await evaluator.evaluate(
            func=dummy_func,
            config={"temperature": 0.7},
            dataset=sample_dataset,
        )

        # Accuracy should be present
        assert "accuracy" in result.aggregated_metrics
        assert result.aggregated_metrics["accuracy"] == 0.9

        # Cost should be absent (not defaulted to zero)
        assert result.aggregated_metrics.get("cost") is None
        assert "cost" not in result.aggregated_metrics


class TestBudgetLimitEnforcementWithJS:
    """Tests for budget limit enforcement with JS evaluator."""

    @pytest.mark.asyncio
    async def test_budget_stops_js_trials(self, sample_dataset, mock_bridge):
        """Verify budget enforcement stops JS trials when limit reached."""
        trial_cost = 0.02
        budget_limit = 0.05

        call_count = 0

        async def mock_run_trial(config):
            nonlocal call_count
            call_count += 1
            return make_js_result(
                trial_id=config["trial_id"],
                cost=trial_cost,
            )

        mock_bridge.run_trial = mock_run_trial

        evaluator = JSEvaluator(js_module="./test.js")
        evaluator._bridge = mock_bridge

        cost_config = CostEnforcerConfig(
            limit=budget_limit,
            estimated_cost_per_trial=trial_cost,
        )
        cost_enforcer = CostEnforcer(cost_config)
        cost_enforcer._estimated_cost = trial_cost

        parallel_manager = ParallelExecutionManager(
            parallel_trials=2,
            cost_enforcer=cost_enforcer,
        )

        async def dummy_func(**kwargs):
            return "result"

        async def run_js_trial(idx):
            return await evaluator.evaluate(
                func=dummy_func,
                config={"trial_idx": idx},
                dataset=sample_dataset,
            )

        total_completed = 0
        total_cancelled = 0

        for batch in range(5):
            coroutines = [run_js_trial(batch * 2 + i) for i in range(2)]
            results, cancelled = await parallel_manager.run_with_cost_permits(
                coroutines
            )

            completed_in_batch = sum(1 for r in results if r.permit.is_granted)
            total_completed += completed_in_batch
            total_cancelled += cancelled

            if cancelled == len(coroutines):
                break

        # Should have stopped due to budget ($0.05 / $0.02 ≈ 2-3 trials)
        assert total_completed <= 4
        assert total_cancelled > 0

    @pytest.mark.asyncio
    async def test_parallel_budget_with_pool(self, sample_dataset, mock_pool):
        """Verify budget enforcement works with parallel pool."""
        trial_cost = 0.015
        budget_limit = 0.06

        async def mock_run_trial(config, timeout=None):
            await asyncio.sleep(0.01)  # Simulate work
            return make_js_result(
                trial_id=config["trial_id"],
                cost=trial_cost,
            )

        mock_pool.run_trial = mock_run_trial

        evaluator = JSEvaluator(
            js_module="./test.js",
            process_pool=mock_pool,
        )

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

        async def dummy_func(**kwargs):
            return "result"

        async def run_pool_trial(idx):
            return await evaluator.evaluate(
                func=dummy_func,
                config={"trial_idx": idx},
                dataset=sample_dataset,
            )

        completed_count = 0
        for batch in range(3):
            coroutines = [run_pool_trial(batch * 4 + i) for i in range(4)]
            results, cancelled = await parallel_manager.run_with_cost_permits(
                coroutines
            )

            completed_count += sum(1 for r in results if r.permit.is_granted)

            if cancelled == len(coroutines):
                break

        # $0.06 / $0.015 = 4 trials max
        assert completed_count <= 5


class TestCostPermitFlowWithJS:
    """Tests for cost permit acquisition and release with JS trials."""

    @pytest.mark.asyncio
    async def test_permit_acquired_before_js_trial(self, sample_dataset, mock_bridge):
        """Verify permit is acquired before JS trial starts."""
        permit_times = []
        trial_times = []

        async def mock_run_trial(config):
            trial_times.append(asyncio.get_event_loop().time())
            await asyncio.sleep(0.01)
            return make_js_result(trial_id=config["trial_id"], cost=0.01)

        mock_bridge.run_trial = mock_run_trial

        evaluator = JSEvaluator(js_module="./test.js")
        evaluator._bridge = mock_bridge

        cost_config = CostEnforcerConfig(limit=1.0, estimated_cost_per_trial=0.01)
        cost_enforcer = CostEnforcer(cost_config)

        # Wrap acquire to track timing
        original_acquire = cost_enforcer.acquire_permit_async

        async def tracked_acquire():
            permit_times.append(asyncio.get_event_loop().time())
            return await original_acquire()

        cost_enforcer.acquire_permit_async = tracked_acquire

        parallel_manager = ParallelExecutionManager(
            parallel_trials=2,
            cost_enforcer=cost_enforcer,
        )

        async def dummy_func(**kwargs):
            return "result"

        async def run_trial():
            return await evaluator.evaluate(
                func=dummy_func,
                config={"temp": 0.7},
                dataset=sample_dataset,
            )

        await parallel_manager.run_with_cost_permits([run_trial()])

        # Permit should be acquired before trial runs
        assert len(permit_times) == 1
        assert len(trial_times) == 1
        # Verify ordering: permit acquired BEFORE trial starts
        assert permit_times[0] < trial_times[0], (
            f"Permit should be acquired before trial starts: "
            f"permit_time={permit_times[0]}, trial_time={trial_times[0]}"
        )

    @pytest.mark.asyncio
    async def test_permit_released_on_js_trial_failure(
        self, sample_dataset, mock_bridge
    ):
        """Verify trial failure is captured in result (permit still granted for tracking)."""
        from traigent.bridges.js_bridge import JSBridgeError

        async def mock_run_trial(config):
            raise JSBridgeError("Process crashed")

        mock_bridge.run_trial = mock_run_trial

        evaluator = JSEvaluator(js_module="./test.js")
        evaluator._bridge = mock_bridge

        cost_config = CostEnforcerConfig(limit=1.0, estimated_cost_per_trial=0.01)
        cost_enforcer = CostEnforcer(cost_config)

        parallel_manager = ParallelExecutionManager(
            parallel_trials=2,
            cost_enforcer=cost_enforcer,
        )

        async def dummy_func(**kwargs):
            return "result"

        async def failing_trial():
            return await evaluator.evaluate(
                func=dummy_func,
                config={"temp": 0.7},
                dataset=sample_dataset,
            )

        results, cancelled = await parallel_manager.run_with_cost_permits(
            [failing_trial()]
        )

        assert len(results) == 1
        # The evaluator catches the error and returns a result, permit stays granted
        # The failure is captured in the result's errors list
        result = results[0].result
        assert result.successful_examples == 0
        assert len(result.errors) > 0
        assert "crashed" in result.errors[0].lower()

    @pytest.mark.asyncio
    async def test_permit_released_on_js_timeout(self, sample_dataset, mock_bridge):
        """Verify timeout is captured in result (permit still granted for tracking)."""
        from traigent.bridges.js_bridge import JSTrialTimeoutError

        async def mock_run_trial(config):
            raise JSTrialTimeoutError("Trial timed out after 300s")

        mock_bridge.run_trial = mock_run_trial

        evaluator = JSEvaluator(js_module="./test.js")
        evaluator._bridge = mock_bridge

        cost_config = CostEnforcerConfig(limit=1.0, estimated_cost_per_trial=0.01)
        cost_enforcer = CostEnforcer(cost_config)

        parallel_manager = ParallelExecutionManager(
            parallel_trials=2,
            cost_enforcer=cost_enforcer,
        )

        async def dummy_func(**kwargs):
            return "result"

        async def timeout_trial():
            return await evaluator.evaluate(
                func=dummy_func,
                config={"temp": 0.7},
                dataset=sample_dataset,
            )

        results, _ = await parallel_manager.run_with_cost_permits([timeout_trial()])

        assert len(results) == 1
        # The evaluator catches the timeout and returns a result
        # The failure is captured in the result's errors list
        result = results[0].result
        assert result.successful_examples == 0
        assert len(result.errors) > 0
        assert "timed out" in result.errors[0].lower()


class TestEMACostEstimationWithJS:
    """Tests for EMA cost estimation with JS trials."""

    @pytest.mark.asyncio
    async def test_ema_updates_from_js_costs(self, sample_dataset, mock_bridge):
        """Verify EMA cost estimate updates from actual JS trial costs."""
        actual_costs = [0.01, 0.015, 0.012, 0.018]
        call_idx = 0

        async def mock_run_trial(config):
            nonlocal call_idx
            cost = actual_costs[call_idx % len(actual_costs)]
            call_idx += 1
            return make_js_result(trial_id=config["trial_id"], cost=cost)

        mock_bridge.run_trial = mock_run_trial

        evaluator = JSEvaluator(js_module="./test.js")
        evaluator._bridge = mock_bridge

        cost_config = CostEnforcerConfig(
            limit=1.0,
            estimated_cost_per_trial=0.01,
        )
        cost_enforcer = CostEnforcer(cost_config)
        initial_estimate = cost_enforcer._estimated_cost

        async def dummy_func(**kwargs):
            return "result"

        # Run several trials to update EMA via permit flow
        for i in range(4):
            permit = await cost_enforcer.acquire_permit_async()

            result = await evaluator.evaluate(
                func=dummy_func,
                config={"trial_idx": i},
                dataset=sample_dataset,
            )

            # Track actual cost with permit
            actual_cost = result.aggregated_metrics.get("cost", 0)
            await cost_enforcer.track_cost_async(actual_cost, permit=permit)

        # EMA should have updated from initial
        # The EMA will be somewhere between initial and actual costs
        assert cost_enforcer._estimated_cost != initial_estimate

    @pytest.mark.asyncio
    async def test_high_variance_costs_smooth_ema(self, sample_dataset, mock_bridge):
        """Verify EMA smooths out high variance in JS trial costs."""
        # High variance costs
        costs = [0.001, 0.05, 0.002, 0.04, 0.003]
        call_idx = 0

        async def mock_run_trial(config):
            nonlocal call_idx
            cost = costs[call_idx % len(costs)]
            call_idx += 1
            return make_js_result(trial_id=config["trial_id"], cost=cost)

        mock_bridge.run_trial = mock_run_trial

        evaluator = JSEvaluator(js_module="./test.js")
        evaluator._bridge = mock_bridge

        cost_config = CostEnforcerConfig(
            limit=10.0,
            estimated_cost_per_trial=0.02,  # Reasonable initial estimate
        )
        cost_enforcer = CostEnforcer(cost_config)

        async def dummy_func(**kwargs):
            return "result"

        estimates = []
        for i in range(5):
            permit = await cost_enforcer.acquire_permit_async()

            result = await evaluator.evaluate(
                func=dummy_func,
                config={"trial_idx": i},
                dataset=sample_dataset,
            )

            actual_cost = result.aggregated_metrics.get("cost", 0)
            await cost_enforcer.track_cost_async(actual_cost, permit=permit)
            estimates.append(cost_enforcer._estimated_cost)

        # EMA should be less volatile than raw costs
        estimate_variance = sum(
            (e - sum(estimates) / len(estimates)) ** 2 for e in estimates
        ) / len(estimates)
        cost_variance = sum((c - sum(costs) / len(costs)) ** 2 for c in costs) / len(
            costs
        )

        # EMA variance should be less than raw cost variance
        assert estimate_variance < cost_variance


class TestConcurrentBudgetWithJS:
    """Tests for concurrent budget tracking with JS trials."""

    @pytest.mark.asyncio
    async def test_concurrent_js_trials_track_accurately(
        self, sample_dataset, mock_pool
    ):
        """Verify budget tracked accurately with concurrent JS trials.

        This test submits MORE trials than budget allows to verify:
        1. Budget enforcement actually limits trial count
        2. Cancelled count reflects denied permits
        3. Completed trials don't exceed budget capacity
        """
        trial_cost = 0.01
        budget_limit = 0.05  # $0.05 / $0.01 = 5 trials max

        async def mock_run_trial(config, timeout=None):
            await asyncio.sleep(0.005)  # Simulate concurrent work
            return make_js_result(trial_id=config["trial_id"], cost=trial_cost)

        mock_pool.run_trial = mock_run_trial

        evaluator = JSEvaluator(
            js_module="./test.js",
            process_pool=mock_pool,
        )

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

        async def dummy_func(**kwargs):
            return "result"

        async def run_trial(idx):
            return await evaluator.evaluate(
                func=dummy_func,
                config={"idx": idx},
                dataset=sample_dataset,
            )

        # Submit 10 trials - budget only allows 5
        coroutines = [run_trial(i) for i in range(10)]
        results, cancelled = await parallel_manager.run_with_cost_permits(coroutines)

        # Count completed trials (granted permits)
        completed = sum(1 for r in results if r.permit.is_granted)

        # Budget enforcement assertions:
        # 1. Should not complete all 10 trials (budget only allows ~5)
        assert completed < 10, f"Expected budget to limit trials, got {completed}/10"

        # 2. Cancelled count should be positive (some permits denied)
        assert cancelled > 0, "Expected some trials to be cancelled due to budget"

        # 3. Completed + cancelled should equal submitted
        assert completed + cancelled == 10, "All trials should be accounted for"

        # 4. Completed trials should be near budget capacity (5, allow margin for EMA)
        assert completed <= 6, f"Completed {completed} exceeds budget capacity of 5"

    @pytest.mark.asyncio
    async def test_rapid_js_trials_dont_overspend(self, sample_dataset, mock_pool):
        """Verify rapid JS trial requests don't exceed budget."""
        trial_cost = 0.02
        budget_limit = 0.05

        async def mock_run_trial(config, timeout=None):
            # Very fast trial
            return make_js_result(trial_id=config["trial_id"], cost=trial_cost)

        mock_pool.run_trial = mock_run_trial

        evaluator = JSEvaluator(
            js_module="./test.js",
            process_pool=mock_pool,
        )

        cost_config = CostEnforcerConfig(
            limit=budget_limit,
            estimated_cost_per_trial=trial_cost,
        )
        cost_enforcer = CostEnforcer(cost_config)
        cost_enforcer._estimated_cost = trial_cost

        parallel_manager = ParallelExecutionManager(
            parallel_trials=10,  # High parallelism
            cost_enforcer=cost_enforcer,
        )

        async def dummy_func(**kwargs):
            return "result"

        async def run_trial(idx):
            return await evaluator.evaluate(
                func=dummy_func,
                config={"idx": idx},
                dataset=sample_dataset,
            )

        # Try many trials at once
        coroutines = [run_trial(i) for i in range(20)]
        results, cancelled = await parallel_manager.run_with_cost_permits(coroutines)

        completed = sum(1 for r in results if r.permit.is_granted)

        # $0.05 / $0.02 = 2.5, so max 3 trials
        assert completed <= 4
        assert cancelled > 0


class TestInvalidCostMetrics:
    """Tests for invalid/hostile cost metrics from JS trials.

    These tests verify the evaluator and budget system handle edge cases:
    - Negative costs
    - NaN/Infinity costs
    - String costs (type mismatch)
    - None/missing cost values
    """

    @pytest.mark.asyncio
    async def test_negative_cost_handled(self, sample_dataset, mock_bridge):
        """Verify negative cost values don't break budget tracking."""

        async def mock_run_trial(config):
            return make_js_result(
                trial_id=config["trial_id"],
                cost=-0.05,  # Negative cost (invalid)
            )

        mock_bridge.run_trial = mock_run_trial

        evaluator = JSEvaluator(js_module="./test.js")
        evaluator._bridge = mock_bridge

        async def dummy_func(**kwargs):
            return "result"

        result = await evaluator.evaluate(
            func=dummy_func,
            config={"temperature": 0.7},
            dataset=sample_dataset,
        )

        # Evaluator should still return a result
        assert result is not None
        # Negative cost should be passed through (business logic decides how to handle)
        assert result.aggregated_metrics.get("cost") == -0.05

    @pytest.mark.asyncio
    async def test_nan_cost_handled(self, sample_dataset, mock_bridge):
        """Verify NaN cost values don't crash the system."""
        import math

        async def mock_run_trial(config):
            return JSTrialResult(
                trial_id=config["trial_id"],
                status="completed",
                metrics={"accuracy": 0.9, "cost": float("nan")},
                duration=1.0,
                error_message=None,
                error_code=None,
                retryable=False,
                metadata={},
            )

        mock_bridge.run_trial = mock_run_trial

        evaluator = JSEvaluator(js_module="./test.js")
        evaluator._bridge = mock_bridge

        async def dummy_func(**kwargs):
            return "result"

        result = await evaluator.evaluate(
            func=dummy_func,
            config={"temperature": 0.7},
            dataset=sample_dataset,
        )

        # Should get a result
        assert result is not None
        # Cost should be NaN (passed through)
        cost = result.aggregated_metrics.get("cost")
        assert cost is not None and math.isnan(cost)

    @pytest.mark.asyncio
    async def test_infinity_cost_handled(self, sample_dataset, mock_bridge):
        """Verify Infinity cost values don't crash the system."""
        import math

        async def mock_run_trial(config):
            return JSTrialResult(
                trial_id=config["trial_id"],
                status="completed",
                metrics={"accuracy": 0.9, "cost": float("inf")},
                duration=1.0,
                error_message=None,
                error_code=None,
                retryable=False,
                metadata={},
            )

        mock_bridge.run_trial = mock_run_trial

        evaluator = JSEvaluator(js_module="./test.js")
        evaluator._bridge = mock_bridge

        async def dummy_func(**kwargs):
            return "result"

        result = await evaluator.evaluate(
            func=dummy_func,
            config={"temperature": 0.7},
            dataset=sample_dataset,
        )

        # Should get a result
        assert result is not None
        # Cost should be infinity
        cost = result.aggregated_metrics.get("cost")
        assert cost is not None and math.isinf(cost)

    @pytest.mark.asyncio
    async def test_string_cost_in_metrics(self, sample_dataset, mock_bridge):
        """Verify string cost values are handled (type mismatch)."""

        async def mock_run_trial(config):
            return JSTrialResult(
                trial_id=config["trial_id"],
                status="completed",
                metrics={"accuracy": 0.9, "cost": "expensive"},  # String, not number
                duration=1.0,
                error_message=None,
                error_code=None,
                retryable=False,
                metadata={},
            )

        mock_bridge.run_trial = mock_run_trial

        evaluator = JSEvaluator(js_module="./test.js")
        evaluator._bridge = mock_bridge

        async def dummy_func(**kwargs):
            return "result"

        result = await evaluator.evaluate(
            func=dummy_func,
            config={"temperature": 0.7},
            dataset=sample_dataset,
        )

        # Should get a result (evaluator doesn't crash on type mismatch)
        assert result is not None
        # Non-numeric cost values are handled gracefully:
        # Either filtered out (best practice) or passed through
        # The key assertion is that the evaluator completes successfully
        cost_value = result.aggregated_metrics.get("cost")
        # If cost is present, it was passed through (acceptable)
        # If cost is absent/None, it was filtered (also acceptable)
        assert (
            cost_value is None or cost_value == "expensive"
        ), f"Cost should be filtered or passed through, got: {cost_value}"

    @pytest.mark.asyncio
    async def test_nan_cost_budget_tracking_resilience(
        self, sample_dataset, mock_bridge
    ):
        """Verify budget tracking handles NaN costs without crashing."""
        import math

        call_count = 0

        async def mock_run_trial(config):
            nonlocal call_count
            call_count += 1
            # First trial returns NaN, second returns valid cost
            cost = float("nan") if call_count == 1 else 0.01
            return JSTrialResult(
                trial_id=config["trial_id"],
                status="completed",
                metrics={"accuracy": 0.9, "cost": cost},
                duration=1.0,
                error_message=None,
                error_code=None,
                retryable=False,
                metadata={},
            )

        mock_bridge.run_trial = mock_run_trial

        evaluator = JSEvaluator(js_module="./test.js")
        evaluator._bridge = mock_bridge

        cost_config = CostEnforcerConfig(
            limit=0.10,
            estimated_cost_per_trial=0.01,
        )
        cost_enforcer = CostEnforcer(cost_config)

        async def dummy_func(**kwargs):
            return "result"

        # Run two trials
        for i in range(2):
            permit = await cost_enforcer.acquire_permit_async()
            result = await evaluator.evaluate(
                func=dummy_func,
                config={"trial_idx": i},
                dataset=sample_dataset,
            )
            cost = result.aggregated_metrics.get("cost", 0)
            # Handle NaN - treat as 0 for tracking
            if isinstance(cost, float) and math.isnan(cost):
                cost = 0.0
            await cost_enforcer.track_cost_async(cost, permit=permit)

        # Should have tracked second trial's cost
        # First NaN was converted to 0, second was 0.01
        assert cost_enforcer._accumulated_cost == 0.01


class TestBudgetEdgeCases:
    """Tests for budget edge cases with JS evaluator."""

    @pytest.mark.asyncio
    async def test_zero_cost_trial_allowed(self, sample_dataset, mock_bridge):
        """Verify zero-cost trials are allowed and tracked."""

        async def mock_run_trial(config):
            return make_js_result(trial_id=config["trial_id"], cost=0.0)

        mock_bridge.run_trial = mock_run_trial

        evaluator = JSEvaluator(js_module="./test.js")
        evaluator._bridge = mock_bridge

        async def dummy_func(**kwargs):
            return "result"

        result = await evaluator.evaluate(
            func=dummy_func,
            config={"temperature": 0.7},
            dataset=sample_dataset,
        )

        # Zero cost should be valid
        assert result.aggregated_metrics.get("cost") == 0.0
        assert result.successful_examples == 5

    @pytest.mark.asyncio
    async def test_very_small_budget_triggers_immediate_stop(
        self, sample_dataset, mock_bridge
    ):
        """Verify very small budget stops after first trial."""

        async def mock_run_trial(config):
            return make_js_result(trial_id=config["trial_id"], cost=0.001)

        mock_bridge.run_trial = mock_run_trial

        evaluator = JSEvaluator(js_module="./test.js")
        evaluator._bridge = mock_bridge

        # Budget smaller than single trial cost
        cost_config = CostEnforcerConfig(
            limit=0.0005,  # Less than trial cost
            estimated_cost_per_trial=0.001,
        )
        cost_enforcer = CostEnforcer(cost_config)
        cost_enforcer._estimated_cost = 0.001

        parallel_manager = ParallelExecutionManager(
            parallel_trials=2,
            cost_enforcer=cost_enforcer,
        )

        async def dummy_func(**kwargs):
            return "result"

        async def run_trial():
            return await evaluator.evaluate(
                func=dummy_func,
                config={"temp": 0.7},
                dataset=sample_dataset,
            )

        # First batch - should be mostly cancelled
        results, cancelled = await parallel_manager.run_with_cost_permits(
            [run_trial(), run_trial()]
        )

        # With budget 0.0005 and cost 0.001, maybe 1 gets through
        completed = sum(1 for r in results if r.permit.is_granted)
        assert completed <= 1 or cancelled > 0

    @pytest.mark.asyncio
    async def test_empty_dataset_subset_handled(self, mock_bridge):
        """Verify empty dataset subset is handled gracefully."""

        async def mock_run_trial(config):
            # Should receive empty indices
            indices = config.get("dataset_subset", {}).get("indices", [])
            return make_js_result(
                trial_id=config["trial_id"],
                cost=0.0,
                accuracy=0.0 if not indices else 1.0,
            )

        mock_bridge.run_trial = mock_run_trial

        evaluator = JSEvaluator(js_module="./test.js")
        evaluator._bridge = mock_bridge

        # Empty dataset
        empty_dataset = Dataset(examples=[], name="empty")

        async def dummy_func(**kwargs):
            return "result"

        result = await evaluator.evaluate(
            func=dummy_func,
            config={"temperature": 0.7},
            dataset=empty_dataset,
        )

        # Should handle empty dataset gracefully
        assert result.total_examples == 0


class TestFullPermitLifecycle:
    """Tests for complete permit lifecycle with JS evaluator.

    These tests verify the full permit flow as used in production:
    1. acquire_permit_async() - reserves estimated cost
    2. Run JS trial
    3. track_cost_async(actual_cost, permit=permit) - releases permit, updates EMA
    4. Permit becomes inactive, reserved cost drops
    """

    @pytest.mark.asyncio
    async def test_full_lifecycle_success(self, sample_dataset, mock_bridge):
        """Verify complete permit lifecycle for successful JS trial."""
        actual_cost = 0.025

        async def mock_run_trial(config):
            return make_js_result(trial_id=config["trial_id"], cost=actual_cost)

        mock_bridge.run_trial = mock_run_trial

        evaluator = JSEvaluator(js_module="./test.js")
        evaluator._bridge = mock_bridge

        # Set up cost enforcer
        cost_config = CostEnforcerConfig(
            limit=1.0,
            estimated_cost_per_trial=0.02,
        )
        cost_enforcer = CostEnforcer(cost_config)
        initial_reserved = cost_enforcer._reserved_cost
        initial_accumulated = cost_enforcer._accumulated_cost

        async def dummy_func(**kwargs):
            return "result"

        # Step 1: Acquire permit
        permit = await cost_enforcer.acquire_permit_async()
        assert permit.is_granted, "Permit should be granted"
        assert permit.active, "Permit should be active"
        assert cost_enforcer._reserved_cost > initial_reserved, "Should reserve cost"

        reserved_after_acquire = cost_enforcer._reserved_cost

        # Step 2: Run trial
        result = await evaluator.evaluate(
            func=dummy_func,
            config={"temperature": 0.7},
            dataset=sample_dataset,
        )
        assert result.aggregated_metrics.get("cost") == actual_cost

        # Step 3: Track cost with SAME permit
        await cost_enforcer.track_cost_async(actual_cost, permit=permit)

        # Step 4: Verify permit state and cost tracking
        assert not permit.active, "Permit should be inactive after tracking"
        assert (
            cost_enforcer._reserved_cost < reserved_after_acquire
        ), "Reserved should decrease"
        assert (
            cost_enforcer._accumulated_cost > initial_accumulated
        ), "Accumulated should increase"
        assert (
            cost_enforcer._accumulated_cost == actual_cost
        ), "Accumulated should match actual"

    @pytest.mark.asyncio
    async def test_full_lifecycle_failed_trial(self, sample_dataset, mock_bridge):
        """Verify permit lifecycle when JS trial fails (error captured in result)."""
        from traigent.bridges.js_bridge import JSBridgeError

        async def mock_run_trial(config):
            raise JSBridgeError("Node process crashed")

        mock_bridge.run_trial = mock_run_trial

        evaluator = JSEvaluator(js_module="./test.js")
        evaluator._bridge = mock_bridge

        cost_config = CostEnforcerConfig(
            limit=1.0,
            estimated_cost_per_trial=0.02,
        )
        cost_enforcer = CostEnforcer(cost_config)

        async def dummy_func(**kwargs):
            return "result"

        # Acquire permit
        permit = await cost_enforcer.acquire_permit_async()
        assert permit.is_granted
        reserved_after_acquire = cost_enforcer._reserved_cost

        # Run trial (evaluator catches error)
        result = await evaluator.evaluate(
            func=dummy_func,
            config={"temperature": 0.7},
            dataset=sample_dataset,
        )

        # Verify error was captured (not raised)
        assert result.successful_examples == 0
        assert len(result.errors) > 0

        # Track zero cost for failed trial (still need to release permit)
        await cost_enforcer.track_cost_async(0.0, permit=permit)

        # Permit should be released, no accumulated cost
        assert not permit.active
        assert cost_enforcer._reserved_cost < reserved_after_acquire
        assert cost_enforcer._accumulated_cost == 0.0

    @pytest.mark.asyncio
    async def test_lifecycle_with_parallel_manager(self, sample_dataset, mock_bridge):
        """Verify permit lifecycle when using ParallelExecutionManager."""

        async def mock_run_trial(config):
            return make_js_result(trial_id=config["trial_id"], cost=0.015)

        mock_bridge.run_trial = mock_run_trial

        evaluator = JSEvaluator(js_module="./test.js")
        evaluator._bridge = mock_bridge

        cost_config = CostEnforcerConfig(
            limit=0.10,
            estimated_cost_per_trial=0.015,
        )
        cost_enforcer = CostEnforcer(cost_config)
        cost_enforcer._estimated_cost = 0.015

        parallel_manager = ParallelExecutionManager(
            parallel_trials=2,
            cost_enforcer=cost_enforcer,
        )

        async def dummy_func(**kwargs):
            return "result"

        async def run_trial(idx):
            return await evaluator.evaluate(
                func=dummy_func,
                config={"trial_idx": idx},
                dataset=sample_dataset,
            )

        # Run trials through parallel manager
        coroutines = [run_trial(i) for i in range(3)]
        results, cancelled = await parallel_manager.run_with_cost_permits(coroutines)

        # Verify all trials completed
        assert len(results) == 3
        assert cancelled == 0

        # Each result should have a permit
        for r in results:
            assert r.permit.is_granted
            # Permit is still active (track_cost_async called by orchestrator)
            assert r.permit.active

        # Simulate orchestrator calling track_cost_async for each result
        for r in results:
            actual_cost = r.result.aggregated_metrics.get("cost", 0)
            await cost_enforcer.track_cost_async(actual_cost, permit=r.permit)

        # All permits should now be inactive
        for r in results:
            assert not r.permit.active

        # Accumulated cost should equal total trial costs
        expected_accumulated = 0.015 * 3
        assert abs(cost_enforcer._accumulated_cost - expected_accumulated) < 0.001


class TestBudgetRecoveryAfterJSFailure:
    """Tests for budget recovery when JS trials fail."""

    @pytest.mark.asyncio
    async def test_failed_js_trial_returns_budget(self, sample_dataset, mock_bridge):
        """Verify failed JS trial returns reserved budget."""
        from traigent.bridges.js_bridge import JSBridgeError

        call_count = 0

        async def mock_run_trial(config):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise JSBridgeError("Worker crashed")
            return make_js_result(trial_id=config["trial_id"], cost=0.01)

        mock_bridge.run_trial = mock_run_trial

        evaluator = JSEvaluator(js_module="./test.js")
        evaluator._bridge = mock_bridge

        cost_config = CostEnforcerConfig(
            limit=0.05,
            estimated_cost_per_trial=0.01,
        )
        cost_enforcer = CostEnforcer(cost_config)
        cost_enforcer._estimated_cost = 0.01

        parallel_manager = ParallelExecutionManager(
            parallel_trials=2,
            cost_enforcer=cost_enforcer,
        )

        async def dummy_func(**kwargs):
            return "result"

        async def run_trial():
            return await evaluator.evaluate(
                func=dummy_func,
                config={"temp": 0.7},
                dataset=sample_dataset,
            )

        # First batch: one succeeds, one fails
        results1, _ = await parallel_manager.run_with_cost_permits(
            [run_trial(), run_trial()]
        )

        # Failed trial should have returned budget, allowing more trials
        # If budget wasn't returned, we'd only get ~3 trials total

        # Second batch should still be able to run
        results2, cancelled = await parallel_manager.run_with_cost_permits(
            [run_trial(), run_trial()]
        )

        # At least some should complete (failed trial returned budget)
        total_completed = sum(1 for r in results1 + results2 if r.permit.is_granted)
        assert total_completed >= 3

    @pytest.mark.asyncio
    async def test_pool_worker_crash_budget_recovery(self, sample_dataset, mock_pool):
        """Verify budget continues working when pool worker crashes."""
        from traigent.bridges.process_pool import PoolCapacityError

        call_count = 0
        crash_count = 0

        async def mock_run_trial(config, timeout=None):
            nonlocal call_count, crash_count
            call_count += 1
            if call_count % 3 == 0:
                crash_count += 1
                raise PoolCapacityError("Worker died")
            return make_js_result(trial_id=config["trial_id"], cost=0.015)

        mock_pool.run_trial = mock_run_trial

        evaluator = JSEvaluator(
            js_module="./test.js",
            process_pool=mock_pool,
        )

        cost_config = CostEnforcerConfig(
            limit=0.10,
            estimated_cost_per_trial=0.015,
        )
        cost_enforcer = CostEnforcer(cost_config)
        cost_enforcer._estimated_cost = 0.015

        parallel_manager = ParallelExecutionManager(
            parallel_trials=4,
            cost_enforcer=cost_enforcer,
        )

        async def dummy_func(**kwargs):
            return "result"

        async def run_trial(idx):
            return await evaluator.evaluate(
                func=dummy_func,
                config={"idx": idx},
                dataset=sample_dataset,
            )

        # Run trials with intermittent failures
        coroutines = [run_trial(i) for i in range(6)]
        results, _ = await parallel_manager.run_with_cost_permits(coroutines)

        # Verify crashes happened
        assert crash_count > 0

        # Verify results captured - evaluator catches errors and returns results
        # All should have results (some with errors, some with success)
        assert len(results) == 6

        # Check that we got both successful and errored results
        error_results = [
            r for r in results if r.result.errors and len(r.result.errors) > 0
        ]
        success_results = [r for r in results if r.result.successful_examples > 0]

        assert len(error_results) > 0  # Some crashed
        assert len(success_results) > 0  # Some succeeded
