"""Unit tests for JSEvaluator - JavaScript runtime evaluator for Node.js trials.

These tests use mocked JSBridge to avoid requiring Node.js installation.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from traigent.bridges.js_bridge import (
    JSBridge,
    JSBridgeError,
    JSTrialResult,
    JSTrialTimeoutError,
)
from traigent.evaluators.base import Dataset, EvaluationExample
from traigent.evaluators.js_evaluator import JSEvaluator, JSEvaluatorConfig

# =============================================================================
# JSEvaluatorConfig Tests
# =============================================================================


class TestJSEvaluatorConfig:
    """Tests for JSEvaluatorConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = JSEvaluatorConfig(js_module="./test.js")

        assert config.js_module == "./test.js"
        assert config.js_function == "runTrial"
        assert config.js_timeout == 300.0
        assert config.experiment_run_id is None

    def test_custom_values(self):
        """Test custom configuration values."""
        config = JSEvaluatorConfig(
            js_module="./dist/agent.js",
            js_function="customTrial",
            js_timeout=600.0,
            experiment_run_id="exp-123",
        )

        assert config.js_module == "./dist/agent.js"
        assert config.js_function == "customTrial"
        assert config.js_timeout == 600.0
        assert config.experiment_run_id == "exp-123"


# =============================================================================
# JSEvaluator Initialization Tests
# =============================================================================


class TestJSEvaluatorInit:
    """Tests for JSEvaluator initialization."""

    def test_basic_initialization(self):
        """Test basic evaluator initialization."""
        evaluator = JSEvaluator(js_module="./test.js")

        assert evaluator._js_config.js_module == "./test.js"
        assert evaluator._js_config.js_function == "runTrial"
        assert evaluator._bridge is None
        assert evaluator._trial_counter == 0

    def test_custom_initialization(self):
        """Test evaluator with custom parameters."""
        evaluator = JSEvaluator(
            js_module="./dist/agent.js",
            js_function="runOptimization",
            js_timeout=120.0,
            experiment_run_id="test-run",
        )

        assert evaluator._js_config.js_module == "./dist/agent.js"
        assert evaluator._js_config.js_function == "runOptimization"
        assert evaluator._js_config.js_timeout == 120.0
        assert evaluator._js_config.experiment_run_id == "test-run"


# =============================================================================
# JSEvaluator.evaluate() Tests
# =============================================================================


class TestJSEvaluatorEvaluate:
    """Tests for JSEvaluator.evaluate() method."""

    @pytest.fixture
    def sample_dataset(self):
        """Create a sample dataset for testing."""
        examples = [
            EvaluationExample({"text": "Hello"}, "Hi"),
            EvaluationExample({"text": "Goodbye"}, "Bye"),
            EvaluationExample({"text": "Thanks"}, "Welcome"),
        ]
        return Dataset(examples=examples, name="test_dataset")

    @pytest.fixture
    def mock_bridge(self):
        """Create a mock JSBridge."""
        bridge = MagicMock(spec=JSBridge)
        bridge.is_running = True
        bridge.start = AsyncMock()
        bridge.stop = AsyncMock()
        return bridge

    @pytest.mark.asyncio
    async def test_evaluate_success(self, sample_dataset, mock_bridge):
        """Test successful evaluation."""
        # Set up mock response
        mock_bridge.run_trial = AsyncMock(
            return_value=JSTrialResult(
                trial_id="test-trial",
                status="completed",
                metrics={"accuracy": 0.95, "latency_ms": 100},
                duration=2.5,
                metadata={"model": "gpt-4"},
            )
        )

        evaluator = JSEvaluator(js_module="./test.js")
        evaluator._bridge = mock_bridge

        async def dummy_func(**kwargs):
            return "result"

        result = await evaluator.evaluate(
            func=dummy_func,
            config={"temperature": 0.7},
            dataset=sample_dataset,
        )

        assert result.successful_examples == 3
        assert result.total_examples == 3
        assert result.aggregated_metrics["accuracy"] == 0.95
        assert result.aggregated_metrics["latency_ms"] == 100.0
        assert result.duration == 2.5
        assert not result.errors

    @pytest.mark.asyncio
    async def test_evaluate_failure(self, sample_dataset, mock_bridge):
        """Test evaluation with trial failure."""
        mock_bridge.run_trial = AsyncMock(
            return_value=JSTrialResult(
                trial_id="failed-trial",
                status="failed",
                duration=1.0,
                error_message="API rate limit exceeded",
                error_code="RATE_LIMIT",
                retryable=True,
            )
        )

        evaluator = JSEvaluator(js_module="./test.js")
        evaluator._bridge = mock_bridge

        async def dummy_func(**kwargs):
            return "result"

        result = await evaluator.evaluate(
            func=dummy_func,
            config={"temperature": 0.7},
            dataset=sample_dataset,
        )

        assert result.successful_examples == 0
        assert result.total_examples == 3
        assert result.aggregated_metrics == {}
        assert len(result.errors) == 1
        assert "rate limit" in result.errors[0].lower()
        assert result.summary_stats["error_code"] == "RATE_LIMIT"
        assert result.summary_stats["retryable"] is True

    @pytest.mark.asyncio
    async def test_evaluate_timeout(self, sample_dataset, mock_bridge):
        """Test evaluation with trial timeout."""
        mock_bridge.run_trial = AsyncMock(
            side_effect=JSTrialTimeoutError("Trial timed out after 300s")
        )

        evaluator = JSEvaluator(js_module="./test.js")
        evaluator._bridge = mock_bridge

        async def dummy_func(**kwargs):
            return "result"

        result = await evaluator.evaluate(
            func=dummy_func,
            config={"temperature": 0.7},
            dataset=sample_dataset,
        )

        assert result.successful_examples == 0
        assert len(result.errors) == 1
        assert "timed out" in result.errors[0].lower()

    @pytest.mark.asyncio
    async def test_evaluate_bridge_error(self, sample_dataset, mock_bridge):
        """Test evaluation with bridge error."""
        mock_bridge.run_trial = AsyncMock(
            side_effect=JSBridgeError("Process crashed")
        )

        evaluator = JSEvaluator(js_module="./test.js")
        evaluator._bridge = mock_bridge

        async def dummy_func(**kwargs):
            return "result"

        result = await evaluator.evaluate(
            func=dummy_func,
            config={"temperature": 0.7},
            dataset=sample_dataset,
        )

        assert result.successful_examples == 0
        assert len(result.errors) == 1
        assert "crashed" in result.errors[0].lower()

    @pytest.mark.asyncio
    async def test_trial_config_format(self, sample_dataset, mock_bridge):
        """Test that trial config sent to JS is properly formatted."""
        captured_config = None

        async def capture_config(trial_config: dict):
            nonlocal captured_config
            captured_config = trial_config
            return JSTrialResult(
                trial_id=trial_config["trial_id"],
                status="completed",
                metrics={"accuracy": 0.9},
                duration=1.0,
            )

        mock_bridge.run_trial = capture_config

        evaluator = JSEvaluator(
            js_module="./test.js",
            experiment_run_id="exp-456",
        )
        evaluator._bridge = mock_bridge

        async def dummy_func(**kwargs):
            return "result"

        await evaluator.evaluate(
            func=dummy_func,
            config={"model": "gpt-4", "temperature": 0.5},
            dataset=sample_dataset,
        )

        # Verify trial config format
        assert captured_config is not None
        assert "trial_id" in captured_config
        assert captured_config["trial_number"] == 1  # First trial
        assert captured_config["experiment_run_id"] == "exp-456"
        assert captured_config["config"] == {"model": "gpt-4", "temperature": 0.5}
        assert captured_config["dataset_subset"]["indices"] == [0, 1, 2]
        assert captured_config["dataset_subset"]["total"] == 3

    @pytest.mark.asyncio
    async def test_trial_counter_increments(self, sample_dataset, mock_bridge):
        """Test that trial counter increments with each evaluation."""
        trial_numbers = []

        async def capture_trial_number(trial_config: dict):
            trial_numbers.append(trial_config["trial_number"])
            return JSTrialResult(
                trial_id=trial_config["trial_id"],
                status="completed",
                metrics={"accuracy": 0.9},
                duration=1.0,
            )

        mock_bridge.run_trial = capture_trial_number

        evaluator = JSEvaluator(js_module="./test.js")
        evaluator._bridge = mock_bridge

        async def dummy_func(**kwargs):
            return "result"

        # Run 3 evaluations
        for _ in range(3):
            await evaluator.evaluate(
                func=dummy_func,
                config={"temperature": 0.7},
                dataset=sample_dataset,
            )

        assert trial_numbers == [1, 2, 3]


# =============================================================================
# JSEvaluator Bridge Lifecycle Tests
# =============================================================================


class TestJSEvaluatorBridgeLifecycle:
    """Tests for bridge lifecycle management."""

    @pytest.mark.asyncio
    async def test_ensure_bridge_creates_bridge(self):
        """Test that _ensure_bridge creates a bridge when needed."""
        evaluator = JSEvaluator(
            js_module="./test.js",
            js_function="runTrial",
            js_timeout=120.0,
        )

        # Mock the JSBridge class
        with patch("traigent.evaluators.js_evaluator.JSBridge") as MockBridge:
            mock_bridge_instance = MagicMock()
            mock_bridge_instance.is_running = True
            mock_bridge_instance.start = AsyncMock()
            MockBridge.return_value = mock_bridge_instance

            bridge = await evaluator._ensure_bridge()

            assert bridge == mock_bridge_instance
            mock_bridge_instance.start.assert_called_once()

    @pytest.mark.asyncio
    async def test_ensure_bridge_reuses_running_bridge(self):
        """Test that _ensure_bridge reuses an existing running bridge."""
        evaluator = JSEvaluator(js_module="./test.js")

        mock_bridge = MagicMock()
        mock_bridge.is_running = True
        mock_bridge.start = AsyncMock()
        evaluator._bridge = mock_bridge

        bridge = await evaluator._ensure_bridge()

        assert bridge == mock_bridge
        mock_bridge.start.assert_not_called()

    @pytest.mark.asyncio
    async def test_ensure_bridge_restarts_dead_bridge(self):
        """Test that _ensure_bridge restarts a dead bridge."""
        evaluator = JSEvaluator(js_module="./test.js")

        dead_bridge = MagicMock()
        dead_bridge.is_running = False
        evaluator._bridge = dead_bridge

        with patch("traigent.evaluators.js_evaluator.JSBridge") as MockBridge:
            new_bridge = MagicMock()
            new_bridge.is_running = True
            new_bridge.start = AsyncMock()
            MockBridge.return_value = new_bridge

            bridge = await evaluator._ensure_bridge()

            assert bridge == new_bridge
            new_bridge.start.assert_called_once()

    @pytest.mark.asyncio
    async def test_close_stops_bridge(self):
        """Test that close() stops the bridge."""
        evaluator = JSEvaluator(js_module="./test.js")

        mock_bridge = MagicMock()
        mock_bridge.stop = AsyncMock()
        evaluator._bridge = mock_bridge

        await evaluator.close()

        mock_bridge.stop.assert_called_once()
        assert evaluator._bridge is None

    @pytest.mark.asyncio
    async def test_close_handles_no_bridge(self):
        """Test that close() handles case when no bridge exists."""
        evaluator = JSEvaluator(js_module="./test.js")
        evaluator._bridge = None

        # Should not raise
        await evaluator.close()


# =============================================================================
# Sample Budget Integration Tests
# =============================================================================


class TestSampleBudgetIntegration:
    """Tests for sample budget lease integration."""

    @pytest.fixture
    def large_dataset(self):
        """Create a larger dataset for budget testing."""
        examples = [
            EvaluationExample({"text": f"Example {i}"}, f"Output {i}")
            for i in range(10)
        ]
        return Dataset(examples=examples, name="budget_test_dataset")

    @pytest.fixture
    def mock_sample_lease(self):
        """Create a mock sample budget lease."""
        lease = MagicMock()
        lease.remaining = 3
        lease.consume = MagicMock()
        return lease

    @pytest.mark.asyncio
    async def test_evaluate_respects_sample_lease(
        self, large_dataset, mock_sample_lease
    ):
        """Test that evaluate respects sample budget lease."""
        captured_config = None

        async def capture_config(trial_config: dict):
            nonlocal captured_config
            captured_config = trial_config
            return JSTrialResult(
                trial_id=trial_config["trial_id"],
                status="completed",
                metrics={"accuracy": 0.9},
                duration=1.0,
            )

        mock_bridge = MagicMock()
        mock_bridge.is_running = True
        mock_bridge.run_trial = capture_config

        evaluator = JSEvaluator(js_module="./test.js")
        evaluator._bridge = mock_bridge

        async def dummy_func(**kwargs):
            return "result"

        result = await evaluator.evaluate(
            func=dummy_func,
            config={"temperature": 0.7},
            dataset=large_dataset,
            sample_lease=mock_sample_lease,
        )

        # Should only use 3 examples (lease.remaining)
        assert captured_config["dataset_subset"]["indices"] == [0, 1, 2]
        assert captured_config["dataset_subset"]["total"] == 10
        assert result.total_examples == 3
        mock_sample_lease.consume.assert_called_once_with(3)

    @pytest.mark.asyncio
    async def test_failed_trial_does_not_consume_samples(
        self, large_dataset, mock_sample_lease
    ):
        """Test that failed trials do not consume sample budget."""
        mock_bridge = MagicMock()
        mock_bridge.is_running = True
        mock_bridge.run_trial = AsyncMock(
            return_value=JSTrialResult(
                trial_id="failed-trial",
                status="failed",
                error_message="Error",
            )
        )

        evaluator = JSEvaluator(js_module="./test.js")
        evaluator._bridge = mock_bridge

        async def dummy_func(**kwargs):
            return "result"

        await evaluator.evaluate(
            func=dummy_func,
            config={"temperature": 0.7},
            dataset=large_dataset,
            sample_lease=mock_sample_lease,
        )

        # Should not consume samples on failure
        mock_sample_lease.consume.assert_not_called()
