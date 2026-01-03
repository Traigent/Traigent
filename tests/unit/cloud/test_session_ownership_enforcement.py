"""Tests covering ownership enforcement behaviour for hardened endpoints."""

import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest

from traigent.cloud.client import CloudServiceError
from traigent.cloud.optimizer_client import OptimizerDirectClient
from traigent.cloud.session_operations import SessionOperations
from traigent.cloud.trial_operations import TrialOperations


@pytest.mark.asyncio
async def test_trial_operations_register_forbidden_logs(caplog, monkeypatch):
    """403 responses should emit remediation guidance and return False."""
    # Disable offline mode so backend calls are actually made
    monkeypatch.setenv("TRAIGENT_OFFLINE_MODE", "false")

    backend_config = SimpleNamespace(
        backend_base_url="https://api.traigent.ai",
        api_base_url="https://api.traigent.ai/v1",
    )
    auth_core = SimpleNamespace(
        get_headers=AsyncMock(return_value={"Authorization": "Bearer tok"}),
        get_owner_fingerprint=lambda: {
            "owner_user_id": "owner-1",
            "owner_api_key_preview": "tg_preview",
            "credential_source": "unit-test",
        },
    )
    auth_manager = SimpleNamespace(
        augment_headers=AsyncMock(return_value={"Authorization": "Bearer tok"}),
        auth=auth_core,
    )
    client = SimpleNamespace(
        backend_config=backend_config,
        auth_manager=auth_manager,
        _map_to_backend_status=Mock(return_value="ACTIVE"),
    )

    trial_ops = TrialOperations(client)

    mock_response = Mock(status=403)
    mock_response.text = AsyncMock(return_value="Forbidden: session owner mismatch")

    response_cm = AsyncMock()
    response_cm.__aenter__.return_value = mock_response

    session = AsyncMock()
    session.post = Mock(return_value=response_cm)

    session_cm = AsyncMock()
    session_cm.__aenter__.return_value = session

    with patch(
        "traigent.cloud.trial_operations.aiohttp.ClientSession", return_value=session_cm
    ):
        with caplog.at_level(logging.ERROR):
            success = await trial_ops.register_trial_start(
                "session-1", "trial-99", {"temperature": 0.5}
            )

    assert success is False
    auth_manager.augment_headers.assert_awaited_once()
    assert any("Re-authenticate" in message for message in caplog.messages)


@pytest.mark.asyncio
async def test_session_operations_status_forbidden():
    """Hybrid session status should surface ownership remediation on 403."""

    backend_config = SimpleNamespace(
        backend_base_url="https://api.traigent.ai",
        api_base_url="https://api.traigent.ai/v1",
    )
    auth_core = SimpleNamespace(
        get_headers=AsyncMock(return_value={"Authorization": "Bearer tok"}),
        get_owner_fingerprint=lambda: {
            "owner_user_id": "owner-1",
            "owner_api_key_preview": "tg_preview",
            "credential_source": "unit-test",
        },
    )
    auth_manager = SimpleNamespace(auth=auth_core)

    mock_response = Mock(status=403)
    mock_response.text = AsyncMock(return_value="Forbidden: session owner mismatch")

    response_cm = AsyncMock()
    response_cm.__aenter__.return_value = mock_response

    session = AsyncMock()
    session.get = Mock(return_value=response_cm)

    client = SimpleNamespace(
        backend_config=backend_config,
        auth_manager=auth_manager,
        _ensure_session=AsyncMock(return_value=session),
        _active_sessions={},
        _revoke_security_session=Mock(),
        session_bridge=SimpleNamespace(get_session_mapping=Mock(return_value=None)),
    )

    session_ops = SessionOperations(client)

    with pytest.raises(CloudServiceError) as excinfo:
        await session_ops.get_hybrid_session_status("session-1")

    message = str(excinfo.value)
    assert "Forbidden" in message
    assert "Re-authenticate" in message


@pytest.mark.asyncio
async def test_optimizer_direct_client_forbidden_errors():
    """Optimizer direct client should raise clear errors on 403 responses."""

    client = OptimizerDirectClient("https://optimizer", "token")

    # Prepare shared mocks for forbidden responses
    forbidden_response = Mock(status=403)
    forbidden_response.text = AsyncMock(return_value="Forbidden: ownership mismatch")
    response_cm = AsyncMock()
    response_cm.__aenter__.return_value = forbidden_response

    # get_next_configuration
    client.session = AsyncMock()
    client.session.get = Mock(return_value=response_cm)

    with pytest.raises(ValueError) as excinfo:
        await client.get_next_configuration("session-1")
    assert "403" in str(excinfo.value)
    assert "owner or admin token" in str(excinfo.value)

    # get_session_status
    client.session.get.return_value = response_cm
    with pytest.raises(ValueError) as excinfo:
        await client.get_session_status("session-1")
    assert "403" in str(excinfo.value)

    # _submit_single and _submit_batch
    client.session.post = Mock(return_value=response_cm)
    with pytest.raises(ValueError) as excinfo:
        await client._submit_single("session-1", {"metric": 1.0})
    assert "403" in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        await client._submit_batch("session-1", [{"metric": 1.0}])
    assert "403" in str(excinfo.value)
