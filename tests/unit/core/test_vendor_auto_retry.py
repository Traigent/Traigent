"""Tests for bounded auto-retry/backoff on transient vendor errors (issue #1404).

A single transient ``429``/``503`` from the LLM provider must not auto-stop an
entire unattended/CI optimization run. The orchestrator should auto-retry
recoverable categories with bounded backoff before falling back to the
resume/stop prompt path (which auto-stops in non-interactive mode).
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from traigent.core.exception_handler import VendorErrorCategory
from traigent.core.orchestrator import OptimizationOrchestrator
from traigent.utils.exceptions import VendorPauseError


class _StopAdapter:
    def __init__(self):
        self.calls = 0

    def prompt_vendor_pause(self, error, category):
        self.calls += 1
        return "stop"


@pytest.fixture
def orch(monkeypatch):
    """Mock orchestrator with the real retry/pause handlers bound."""
    monkeypatch.setenv("TRAIGENT_VENDOR_RETRY_BACKOFF", "0")  # no real sleeping
    o = MagicMock()
    o._vendor_retry_counts = {}
    o._vendor_retry_settings = OptimizationOrchestrator._vendor_retry_settings
    o._maybe_auto_retry_vendor_error = (
        OptimizationOrchestrator._maybe_auto_retry_vendor_error.__get__(o)
    )
    o._handle_vendor_pause = OptimizationOrchestrator._handle_vendor_pause.__get__(o)
    return o


def _rate_limit_exc():
    return VendorPauseError("rate limit", category=VendorErrorCategory.RATE_LIMIT)


@pytest.mark.asyncio
async def test_transient_rate_limit_auto_retries_before_prompt(orch, monkeypatch):
    """First 429 auto-retries ("continue") instead of consulting the stop prompt."""
    monkeypatch.setenv("TRAIGENT_VENDOR_MAX_RETRIES", "2")
    adapter = _StopAdapter()
    orch._prompt_adapter = adapter

    result = await orch._handle_vendor_pause(_rate_limit_exc())

    assert result == "continue"
    assert adapter.calls == 0  # the stop prompt was NOT consulted


@pytest.mark.asyncio
async def test_retry_budget_is_bounded_then_stops(orch, monkeypatch):
    """After the retry budget is exhausted, it falls back to the stop prompt."""
    monkeypatch.setenv("TRAIGENT_VENDOR_MAX_RETRIES", "2")
    adapter = _StopAdapter()
    orch._prompt_adapter = adapter

    r1 = await orch._handle_vendor_pause(_rate_limit_exc())
    r2 = await orch._handle_vendor_pause(_rate_limit_exc())
    r3 = await orch._handle_vendor_pause(_rate_limit_exc())

    assert [r1, r2] == ["continue", "continue"]  # 2 bounded retries
    assert r3 == "break"  # budget exhausted -> prompt -> stop
    assert adapter.calls == 1  # prompt only consulted after budget exhausted


@pytest.mark.asyncio
async def test_disabled_when_max_retries_zero(orch, monkeypatch):
    """TRAIGENT_VENDOR_MAX_RETRIES=0 restores immediate stop (no auto-retry)."""
    monkeypatch.setenv("TRAIGENT_VENDOR_MAX_RETRIES", "0")
    adapter = _StopAdapter()
    orch._prompt_adapter = adapter

    result = await orch._handle_vendor_pause(_rate_limit_exc())

    assert result == "break"
    assert adapter.calls == 1


@pytest.mark.asyncio
async def test_non_recoverable_categories_not_retried(orch, monkeypatch):
    """Insufficient funds / quota exhausted are not transient -> no auto-retry."""
    monkeypatch.setenv("TRAIGENT_VENDOR_MAX_RETRIES", "2")
    adapter = _StopAdapter()
    orch._prompt_adapter = adapter

    for cat in (
        VendorErrorCategory.INSUFFICIENT_FUNDS,
        VendorErrorCategory.QUOTA_EXHAUSTED,
    ):
        orch._vendor_retry_counts.clear()
        exc = VendorPauseError("x", category=cat)
        result = await orch._handle_vendor_pause(exc)
        assert result == "break"


@pytest.mark.asyncio
async def test_service_unavailable_is_recoverable(orch, monkeypatch):
    """503 service-unavailable is a recoverable transient category."""
    monkeypatch.setenv("TRAIGENT_VENDOR_MAX_RETRIES", "1")
    adapter = _StopAdapter()
    orch._prompt_adapter = adapter

    exc = VendorPauseError("503", category=VendorErrorCategory.SERVICE_UNAVAILABLE)
    result = await orch._handle_vendor_pause(exc)

    assert result == "continue"
