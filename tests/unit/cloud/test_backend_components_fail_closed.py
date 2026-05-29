"""Backend component managers must not return local placeholder results."""

from __future__ import annotations

import pytest

from traigent.cloud.backend_components import (
    BackendAuthManager,
    BackendClientConfig,
    BackendSessionManager,
    BackendTrialManager,
)


def _auth_manager() -> BackendAuthManager:
    return BackendAuthManager(  # pragma: allowlist secret
        api_key="test-key",  # pragma: allowlist secret
        rate_limit_calls=100,
        rate_limit_period=1,
    )


@pytest.mark.asyncio
async def test_backend_session_manager_fails_closed_until_wired() -> None:
    manager = BackendSessionManager(_auth_manager(), BackendClientConfig())

    with pytest.raises(NotImplementedError, match="not wired to a backend endpoint"):
        await manager.create_session("session-1", {"mode": "hybrid"})


@pytest.mark.asyncio
async def test_backend_trial_manager_fails_closed_until_wired() -> None:
    manager = BackendTrialManager(_auth_manager(), BackendClientConfig())

    with pytest.raises(NotImplementedError, match="backend optimizer endpoint"):
        await manager.get_next_privacy_trial("session-1", trial_count=2)
