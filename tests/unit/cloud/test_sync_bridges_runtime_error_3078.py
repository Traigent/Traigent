"""Sync bridges must not turn a worker RuntimeError into the wrong error (#3078, #3144)."""

from __future__ import annotations

import asyncio
import gc
import logging
import warnings
from typing import Any, cast
from unittest.mock import MagicMock, patch

import pytest

from traigent.cloud import session_operations as session_ops_module
from traigent.cloud.backend_client import BackendIntegratedClient
from traigent.cloud.session_operations import SessionOperations
from traigent.cloud.trial_operations import TrialOperations

# Opt into the connected code paths (see pyproject markers).
pytestmark = pytest.mark.backend_online

REAL_CAUSE = "REAL CAUSE: bookkeeping failed"
WRONG_CAUSE = "cannot be called from a running event loop"


async def _boom(*_args: Any, **_kwargs: Any) -> Any:
    raise RuntimeError(REAL_CAUSE)


def _create(ops: SessionOperations) -> Any:
    return ops.create_session("f", {"model": ["a", "b"]}, metadata={"max_trials": 2})


def test_create_session_inside_loop_reports_real_cause() -> None:
    ops = SessionOperations(cast(Any, MagicMock()))

    async def main() -> Any:
        return _create(ops)

    with (
        patch.object(session_ops_module, "_create_session_async", _boom),
        warnings.catch_warnings(record=True) as caught,
    ):
        warnings.simplefilter("always")
        result = asyncio.run(main())

    assert REAL_CAUSE in str(result.failure_detail)
    assert WRONG_CAUSE not in str(result.failure_detail)
    gc.collect()
    assert not [w for w in caught if "never awaited" in str(w.message)]


def test_register_trial_start_sync_inside_loop_runs_once_and_logs_real_cause(
    caplog: pytest.LogCaptureFixture,
) -> None:
    ops = TrialOperations(cast(Any, MagicMock()))
    calls: list[int] = []

    async def failing(*_a: Any, **_k: Any) -> Any:
        calls.append(1)
        raise RuntimeError(REAL_CAUSE)

    async def main() -> Any:
        return ops.register_trial_start_sync("s", "t", {})

    with (
        patch.object(ops, "register_trial_start", failing),
        caplog.at_level(logging.ERROR),
        warnings.catch_warnings(record=True) as caught,
    ):
        warnings.simplefilter("always")
        # register_trial_start swallows nothing here: the stub raises directly.
        result = asyncio.run(main())

    assert result is False
    assert len(calls) == 1
    assert REAL_CAUSE in caplog.text
    assert WRONG_CAUSE not in caplog.text
    gc.collect()
    assert not [w for w in caught if "never awaited" in str(w.message)]


def test_sync_auth_headers_inside_loop_surfaces_real_cause_once() -> None:
    client = object.__new__(BackendIntegratedClient)
    client.timeout = 5.0
    calls: list[int] = []

    async def failing_headers(target: Any = None) -> Any:
        calls.append(1)
        raise RuntimeError(REAL_CAUSE)

    auth = MagicMock()
    auth.get_headers = failing_headers
    client.auth_manager = MagicMock(auth=auth)

    async def main() -> Any:
        return client._get_sync_auth_headers()

    raised = False
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            asyncio.run(main())
        except Exception:  # noqa: BLE001 - only the bridge behaviour matters
            raised = True
        gc.collect()

    assert raised
    assert len(calls) == 1
    assert not [w for w in caught if "never awaited" in str(w.message)]
