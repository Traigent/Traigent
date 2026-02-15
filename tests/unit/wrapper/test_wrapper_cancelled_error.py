"""Tests for asyncio.CancelledError re-raise in TraigentService.

Verifies that CancelledError is NOT swallowed by ``except Exception``
handlers in handle_execute() and handle_evaluate().
SonarQube S7497 requires CancelledError to always propagate.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from traigent.wrapper.service import TraigentService

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_service_with_handlers() -> TraigentService:
    """Create a TraigentService with execute and evaluate handlers that raise CancelledError."""
    svc = TraigentService(capability_id="test_agent")

    @svc.tvars
    def config_space() -> dict[str, Any]:
        return {"model": {"type": "enum", "values": ["gpt-4"]}}

    return svc


# ---------------------------------------------------------------------------
# handle_execute — CancelledError from execute handler
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_handle_execute_propagates_cancelled_error():
    """CancelledError from the execute handler must propagate."""
    svc = _make_service_with_handlers()

    @svc.execute
    async def run_agent(
        input_id: str, data: Any, config: dict[str, Any]
    ) -> dict[str, Any]:
        raise asyncio.CancelledError

    request = {
        "request_id": "req_1",
        "capability_id": "test_agent",
        "config": {},
        "inputs": [{"input_id": "ex_0", "data": {"question": "hi"}}],
    }

    with pytest.raises(asyncio.CancelledError):
        await svc.handle_execute(request)


# ---------------------------------------------------------------------------
# handle_evaluate — CancelledError from evaluate handler
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_handle_evaluate_propagates_cancelled_error():
    """CancelledError from the evaluate handler must propagate."""
    svc = _make_service_with_handlers()

    @svc.evaluate
    async def score(
        output: Any, target: Any, config: dict[str, Any]
    ) -> dict[str, float]:
        raise asyncio.CancelledError

    request = {
        "request_id": "req_2",
        "capability_id": "test_agent",
        "config": {},
        "evaluations": [
            {
                "input_id": "ex_0",
                "output": "some output",
                "target": "expected",
            }
        ],
    }

    with pytest.raises(asyncio.CancelledError):
        await svc.handle_evaluate(request)
