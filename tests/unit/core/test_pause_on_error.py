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
