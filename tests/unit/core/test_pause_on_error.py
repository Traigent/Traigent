"""Tests for pause-on-error integration in orchestrator and trial lifecycle."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from traigent.core.exception_handler import VendorErrorCategory
from traigent.utils.exceptions import VendorPauseError

# ---------------------------------------------------------------------------
# Mock prompt adapter for testing
# ---------------------------------------------------------------------------


class MockPromptAdapter:
    """Controllable prompt adapter for testing."""

    def __init__(self, vendor_response: str = "stop", budget_response: str = "stop"):
        self.vendor_response = vendor_response
        self.budget_response = budget_response
        self.vendor_calls: list[tuple] = []
        self.budget_calls: list[tuple] = []

    def prompt_vendor_pause(self, error, category):
        self.vendor_calls.append((error, category))
        return self.vendor_response

    def prompt_budget_pause(self, accumulated, limit):
        self.budget_calls.append((accumulated, limit))
        return self.budget_response


# ---------------------------------------------------------------------------
# Orchestrator handler methods (unit tests)
# ---------------------------------------------------------------------------


class TestHandleVendorPause:
    """Test _handle_vendor_pause on the orchestrator."""

    @pytest.fixture
    def mock_orchestrator(self):
        """Create a minimal mock orchestrator with handler methods."""
        from traigent.core.orchestrator import OptimizationOrchestrator

        # We test the methods directly by calling them on a mock
        orch = MagicMock()
        # Bind the real method
        orch._handle_vendor_pause = (
            OptimizationOrchestrator._handle_vendor_pause.__get__(orch)
        )
        return orch

    @pytest.mark.asyncio
    async def test_resume_returns_continue(self, mock_orchestrator):
        adapter = MockPromptAdapter(vendor_response="resume")
        mock_orchestrator._prompt_adapter = adapter

        exc = VendorPauseError("rate limit", category=VendorErrorCategory.RATE_LIMIT)
        result = await mock_orchestrator._handle_vendor_pause(exc)

        assert result == "continue"
        assert len(adapter.vendor_calls) == 1

    @pytest.mark.asyncio
    async def test_stop_returns_break(self, mock_orchestrator):
        adapter = MockPromptAdapter(vendor_response="stop")
        mock_orchestrator._prompt_adapter = adapter

        exc = VendorPauseError("rate limit", category=VendorErrorCategory.RATE_LIMIT)
        result = await mock_orchestrator._handle_vendor_pause(exc)

        assert result == "break"

    @pytest.mark.asyncio
    async def test_no_adapter_returns_break(self, mock_orchestrator):
        mock_orchestrator._prompt_adapter = None

        exc = VendorPauseError("rate limit", category=VendorErrorCategory.RATE_LIMIT)
        result = await mock_orchestrator._handle_vendor_pause(exc)

        assert result == "break"


class TestHandleBudgetLimitPause:
    """Test _handle_budget_limit_pause on the orchestrator."""

    @pytest.fixture
    def mock_orchestrator(self):
        from traigent.core.orchestrator import OptimizationOrchestrator

        orch = MagicMock()
        orch._handle_budget_limit_pause = (
            OptimizationOrchestrator._handle_budget_limit_pause.__get__(orch)
        )

        # Mock cost enforcer
        status_mock = MagicMock()
        status_mock.accumulated_cost_usd = 1.50
        orch.cost_enforcer.get_status.return_value = status_mock
        orch.cost_enforcer.config.limit = 2.00

        return orch

    @pytest.mark.asyncio
    async def test_raise_limit_returns_continue(self, mock_orchestrator):
        adapter = MockPromptAdapter(budget_response="raise:5.0")
        mock_orchestrator._prompt_adapter = adapter

        result = await mock_orchestrator._handle_budget_limit_pause()

        assert result == "continue"
        mock_orchestrator.cost_enforcer.update_limit.assert_called_once_with(5.0)
        assert len(adapter.budget_calls) == 1

    @pytest.mark.asyncio
    async def test_stop_returns_break(self, mock_orchestrator):
        adapter = MockPromptAdapter(budget_response="stop")
        mock_orchestrator._prompt_adapter = adapter

        result = await mock_orchestrator._handle_budget_limit_pause()

        assert result == "break"
        mock_orchestrator.cost_enforcer.update_limit.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_adapter_returns_break(self, mock_orchestrator):
        mock_orchestrator._prompt_adapter = None

        result = await mock_orchestrator._handle_budget_limit_pause()

        assert result == "break"


# ---------------------------------------------------------------------------
# VendorPauseError propagation from trial_lifecycle
# ---------------------------------------------------------------------------


class TestVendorPauseErrorPropagation:
    """Test that VendorPauseError is raised correctly."""

    def test_vendor_pause_error_attributes(self):
        original = RuntimeError("429 Too Many Requests")
        exc = VendorPauseError(
            "rate limit hit",
            original_error=original,
            category=VendorErrorCategory.RATE_LIMIT,
        )
        assert exc.original_error is original
        assert exc.category == VendorErrorCategory.RATE_LIMIT
        assert "rate limit hit" in str(exc)

    def test_vendor_pause_error_default_category(self):
        exc = VendorPauseError("some error")
        assert exc.category is None
        assert exc.original_error is None


# ---------------------------------------------------------------------------
# Malformed budget response handling
# ---------------------------------------------------------------------------


class TestMalformedBudgetResponse:
    """Test _handle_budget_limit_pause with malformed adapter responses."""

    @pytest.fixture
    def mock_orchestrator(self):
        from traigent.core.orchestrator import OptimizationOrchestrator

        orch = MagicMock()
        orch._handle_budget_limit_pause = (
            OptimizationOrchestrator._handle_budget_limit_pause.__get__(orch)
        )
        status_mock = MagicMock()
        status_mock.accumulated_cost_usd = 1.50
        orch.cost_enforcer.get_status.return_value = status_mock
        orch.cost_enforcer.config.limit = 2.00
        return orch

    @pytest.mark.asyncio
    async def test_malformed_raise_value_returns_break(self, mock_orchestrator):
        """raise:abc triggers ValueError catch → break."""
        adapter = MockPromptAdapter(budget_response="raise:abc")
        mock_orchestrator._prompt_adapter = adapter

        result = await mock_orchestrator._handle_budget_limit_pause()

        assert result == "break"
        mock_orchestrator.cost_enforcer.update_limit.assert_not_called()

    @pytest.mark.asyncio
    async def test_empty_raise_returns_break(self, mock_orchestrator):
        """raise: (empty after colon) triggers IndexError/ValueError catch."""
        adapter = MockPromptAdapter(budget_response="raise:")
        mock_orchestrator._prompt_adapter = adapter

        result = await mock_orchestrator._handle_budget_limit_pause()

        assert result == "break"

    @pytest.mark.asyncio
    async def test_unexpected_response_returns_break(self, mock_orchestrator):
        """Random string that doesn't start with 'raise:' → break."""
        adapter = MockPromptAdapter(budget_response="unknown_response")
        mock_orchestrator._prompt_adapter = adapter

        result = await mock_orchestrator._handle_budget_limit_pause()

        assert result == "break"


# ---------------------------------------------------------------------------
# Trial lifecycle: vendor error re-raise
# ---------------------------------------------------------------------------


class TestTrialLifecycleVendorReraise:
    """Test that vendor errors in _execute_trial_with_tracing are re-raised."""

    def test_classify_and_reraise_rate_limit(self):
        """RateLimitError should be classified and wrapped in VendorPauseError."""
        from traigent.core.exception_handler import classify_vendor_error
        from traigent.utils.exceptions import RateLimitError

        exc = RateLimitError("429 Too Many Requests")
        category = classify_vendor_error(exc)
        assert category == VendorErrorCategory.RATE_LIMIT

        # Simulate what trial_lifecycle does
        pause_error = VendorPauseError(str(exc), original_error=exc, category=category)
        assert pause_error.original_error is exc
        assert pause_error.category == VendorErrorCategory.RATE_LIMIT

    def test_classify_and_reraise_quota(self):
        from traigent.core.exception_handler import classify_vendor_error
        from traigent.utils.exceptions import QuotaExceededError

        exc = QuotaExceededError("Quota exhausted")
        category = classify_vendor_error(exc)
        assert category == VendorErrorCategory.QUOTA_EXHAUSTED

        pause_error = VendorPauseError(str(exc), original_error=exc, category=category)
        assert pause_error.category == VendorErrorCategory.QUOTA_EXHAUSTED

    def test_classify_and_reraise_service_unavailable(self):
        from traigent.core.exception_handler import classify_vendor_error
        from traigent.utils.exceptions import ServiceUnavailableError

        exc = ServiceUnavailableError("Service down")
        category = classify_vendor_error(exc)
        assert category == VendorErrorCategory.SERVICE_UNAVAILABLE

        pause_error = VendorPauseError(str(exc), original_error=exc, category=category)
        assert pause_error.category == VendorErrorCategory.SERVICE_UNAVAILABLE

    def test_non_vendor_not_reclassified(self):
        """Non-vendor errors should not be classified."""
        from traigent.core.exception_handler import classify_vendor_error

        exc = ValueError("bad argument")
        assert classify_vendor_error(exc) is None


# ---------------------------------------------------------------------------
# Parallel batch vendor error detection
# ---------------------------------------------------------------------------


class TestParallelBatchVendorDetection:
    """Test the post-batch vendor error check in _run_parallel_batch."""

    def test_all_vendor_failures_detected(self):
        """When all results have vendor error messages, classify detects them."""
        from traigent.core.exception_handler import classify_vendor_error

        error_messages = [
            "429 Too Many Requests",
            "Rate limit exceeded",
            "quota exhausted for model",
        ]
        for msg in error_messages:
            category = classify_vendor_error(RuntimeError(msg))
            assert category is not None, f"Failed to classify: {msg}"

    def test_mixed_results_not_detected(self):
        """When some results are non-vendor, don't classify as vendor outage."""
        from traigent.core.exception_handler import classify_vendor_error

        vendor_msg = "429 Too Many Requests"
        normal_msg = "Division by zero in user function"

        assert classify_vendor_error(RuntimeError(vendor_msg)) is not None
        assert classify_vendor_error(RuntimeError(normal_msg)) is None
