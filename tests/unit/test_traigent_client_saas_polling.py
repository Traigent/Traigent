"""Regression tests for SaaS polling safeguards in TraigentClient."""

from __future__ import annotations

from unittest.mock import AsyncMock, Mock

import pytest

from traigent.traigent_client import TraigentClient
from traigent.utils.exceptions import OptimizationError


def _build_saas_client() -> TraigentClient:
    client = object.__new__(TraigentClient)
    backend = Mock()
    backend.__aenter__ = AsyncMock(return_value=backend)
    backend.__aexit__ = AsyncMock(return_value=None)
    client.backend_client = backend
    return client


@pytest.mark.asyncio
async def test_optimize_saas_times_out_when_status_never_completes() -> None:
    client = _build_saas_client()

    client.backend_client.upload_dataset = AsyncMock(return_value={"dataset_id": "d1"})
    client.backend_client.create_optimization_session = AsyncMock(
        return_value={"session_id": "s1"}
    )
    client.backend_client.get_session_status = AsyncMock(
        return_value={"status": "RUNNING", "completed_trials": 0}
    )

    with pytest.raises(OptimizationError, match="polling timed out"):
        await client._optimize_saas(
            function=lambda: "ok",
            dataset={"examples": []},
            configuration_space={"model": ["gpt-4o-mini"]},
            objectives=["accuracy"],
            max_trials=5,
            optimization_config={"poll_interval": 0.01, "max_poll_duration": 0.001},
        )


@pytest.mark.asyncio
async def test_optimize_saas_rejects_unexpected_status() -> None:
    client = _build_saas_client()

    client.backend_client.upload_dataset = AsyncMock(return_value={"dataset_id": "d1"})
    client.backend_client.create_optimization_session = AsyncMock(
        return_value={"session_id": "s1"}
    )
    client.backend_client.get_session_status = AsyncMock(
        return_value={"status": "PAUSED", "completed_trials": 2}
    )

    with pytest.raises(OptimizationError, match="Unexpected SaaS session status"):
        await client._optimize_saas(
            function=lambda: "ok",
            dataset={"examples": []},
            configuration_space={"model": ["gpt-4o-mini"]},
            objectives=["accuracy"],
            max_trials=5,
            optimization_config={"poll_interval": 0.01},
        )
