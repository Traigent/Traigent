"""Tests for asyncio.CancelledError re-raise in IntegrationManager.

These tests verify that CancelledError is NOT swallowed by the broad
``except Exception`` handlers in each async method of IntegrationManager.
SonarQube S7497 requires CancelledError to always propagate.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from traigent.cloud.integration_manager import (
    IntegrationConfig,
    IntegrationManager,
    IntegrationMode,
    IntegrationResult,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_manager(initialized: bool = True) -> IntegrationManager:
    """Create an IntegrationManager with mocked internals."""
    cfg = IntegrationConfig(
        mode=IntegrationMode.EDGE_ANALYTICS,
        backend_base_url="http://localhost:5000",
    )
    mgr = IntegrationManager(config=cfg)
    if initialized:
        mgr._initialized = True
        mgr._mcp_client = MagicMock()
        mgr._backend_client = AsyncMock()
    return mgr


# ---------------------------------------------------------------------------
# initialize()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_initialize_propagates_cancelled_error():
    """CancelledError during initialize() must propagate."""
    mgr = _make_manager(initialized=False)
    with patch(
        "traigent.cloud.integration_manager.get_production_mcp_client",
        side_effect=asyncio.CancelledError,
    ):
        with pytest.raises(asyncio.CancelledError):
            await mgr.initialize()


# ---------------------------------------------------------------------------
# start_optimization_integration()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_start_optimization_integration_propagates_cancelled_error():
    """CancelledError during start_optimization_integration() must propagate."""
    mgr = _make_manager()
    request = MagicMock()
    request.metadata = {}
    request.dataset.examples = []

    # Force _start_edge_analytics_integration to raise CancelledError
    with patch.object(
        mgr,
        "_start_edge_analytics_integration",
        new_callable=AsyncMock,
        side_effect=asyncio.CancelledError,
    ):
        with pytest.raises(asyncio.CancelledError):
            await mgr.start_optimization_integration(
                request, mode=IntegrationMode.EDGE_ANALYTICS
            )


# ---------------------------------------------------------------------------
# _start_edge_analytics_integration()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_start_edge_analytics_integration_propagates_cancelled_error():
    """CancelledError in _start_edge_analytics_integration() must propagate."""
    mgr = _make_manager()
    request = MagicMock()

    # Patch logger.info to raise CancelledError on the first call inside the method
    with patch(
        "traigent.cloud.integration_manager.logger.info",
        side_effect=asyncio.CancelledError,
    ):
        with pytest.raises(asyncio.CancelledError):
            await mgr._start_edge_analytics_integration(request, "test_integration")


# ---------------------------------------------------------------------------
# _start_privacy_integration() — called as _start_privacy_first_integration
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_start_privacy_integration_propagates_cancelled_error():
    """CancelledError in _start_privacy_integration() must propagate."""
    mgr = _make_manager()
    request = MagicMock()
    request.agent_specification = "spec"
    request.dataset = MagicMock()
    request.function_name = "test_fn"
    request.configuration_space = {}
    request.objectives = []
    request.max_trials = 10

    mgr._mcp_client.create_optimization_workflow = AsyncMock(
        side_effect=asyncio.CancelledError
    )

    with pytest.raises(asyncio.CancelledError):
        await mgr._start_privacy_integration(request, "test_integration")


# ---------------------------------------------------------------------------
# _start_cloud_integration() — cloud SaaS
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_start_cloud_integration_propagates_cancelled_error():
    """CancelledError in _start_cloud_integration() must propagate."""
    mgr = _make_manager()
    request = MagicMock()
    request.agent_specification = "spec"
    request.function_name = "test_fn"
    request.configuration_space = {}
    request.objectives = []
    request.max_trials = 10
    request.dataset = MagicMock()

    mgr._mcp_client.create_optimization_workflow = AsyncMock(
        side_effect=asyncio.CancelledError
    )

    with pytest.raises(asyncio.CancelledError):
        await mgr._start_cloud_integration(request, "test_integration")


# ---------------------------------------------------------------------------
# get_next_trial()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_next_trial_propagates_cancelled_error():
    """CancelledError in get_next_trial() must propagate."""
    mgr = _make_manager()

    with patch(
        "traigent.cloud.integration_manager.lifecycle_manager"
    ) as mock_lifecycle:
        mock_lifecycle.get_session_state.side_effect = asyncio.CancelledError
        with pytest.raises(asyncio.CancelledError):
            await mgr.get_next_trial("session_123")


# ---------------------------------------------------------------------------
# submit_trial_results()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_submit_trial_results_propagates_cancelled_error():
    """CancelledError in submit_trial_results() must propagate."""
    mgr = _make_manager()

    # Patch _get_integration_for_session to raise CancelledError
    with patch.object(
        mgr,
        "_get_integration_for_session",
        side_effect=asyncio.CancelledError,
    ):
        with pytest.raises(asyncio.CancelledError):
            await mgr.submit_trial_results(
                session_id="session_123",
                trial_id="trial_1",
                config={"model": "gpt-4"},
                metrics={"accuracy": 0.9},
                duration=1.0,
            )


# ---------------------------------------------------------------------------
# finalize_session()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_finalize_session_propagates_cancelled_error():
    """CancelledError in finalize_session() must propagate."""
    mgr = _make_manager()

    with patch(
        "traigent.cloud.integration_manager.lifecycle_manager"
    ) as mock_lifecycle:
        mock_lifecycle.complete_session.side_effect = asyncio.CancelledError
        with pytest.raises(asyncio.CancelledError):
            await mgr.finalize_session("session_123")


# ---------------------------------------------------------------------------
# cancel_session()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cancel_session_propagates_cancelled_error():
    """CancelledError in cancel_session() must propagate."""
    mgr = _make_manager()

    with patch(
        "traigent.cloud.integration_manager.lifecycle_manager"
    ) as mock_lifecycle:
        mock_lifecycle.cancel_session.side_effect = asyncio.CancelledError
        with pytest.raises(asyncio.CancelledError):
            await mgr.cancel_session("session_123")


# ---------------------------------------------------------------------------
# health_check()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_health_check_propagates_cancelled_error():
    """CancelledError in health_check() must propagate."""
    mgr = _make_manager()
    mgr._mcp_client.health_check = AsyncMock(side_effect=asyncio.CancelledError)

    with pytest.raises(asyncio.CancelledError):
        await mgr.health_check()


# ---------------------------------------------------------------------------
# cleanup()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cleanup_propagates_cancelled_error():
    """CancelledError in cleanup() must propagate."""
    mgr = _make_manager()
    mgr._mcp_client.disconnect = AsyncMock(side_effect=asyncio.CancelledError)

    with pytest.raises(asyncio.CancelledError):
        await mgr.cleanup()
