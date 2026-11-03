"""Characterize less common synchronous session creation paths."""

import asyncio
from threading import Lock
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from traigent.cloud.client import CloudServiceError
from traigent.cloud.session_operations import SessionOperations
from traigent.utils.exceptions import ValidationError

pytestmark = pytest.mark.backend_online


@pytest.fixture
def ops(monkeypatch):
    monkeypatch.setattr(
        "traigent.cloud.session_operations.is_backend_offline", lambda: False
    )
    client = SimpleNamespace(
        auth_manager=SimpleNamespace(has_api_key=Mock(return_value=True)),
        _create_traigent_session_via_api=AsyncMock(
            return_value=("session", "experiment", "run")
        ),
        _active_sessions_lock=Lock(),
        _active_sessions={},
        _max_active_sessions=5,
        session_bridge=SimpleNamespace(create_session_mapping=Mock()),
        _register_security_session=Mock(),
        local_storage=None,
        backend_config=SimpleNamespace(api_base_url=None, backend_base_url=None),
    )
    result = SessionOperations(client)
    result._create_local_fallback_session = Mock(return_value="local")
    return result


@pytest.mark.parametrize(
    ("metadata", "expected"), [(None, 10), ({"max_trials": "3"}, 3)]
)
def test_max_trials_default_and_valid(ops, metadata, expected):
    result = ops.create_session("fn", {"x": [1]}, metadata=metadata)
    assert result.backend_connected
    assert (
        ops.client._create_traigent_session_via_api.call_args.args[0].max_trials
        == expected
    )


@pytest.mark.parametrize("value", ["bad", 0, -1])
def test_invalid_max_trials_raises(ops, value):
    with pytest.raises(
        ValidationError, match="metadata.max_trials must be a positive integer"
    ):
        ops.create_session("fn", {"x": [1]}, metadata={"max_trials": value})
    ops.client._create_traigent_session_via_api.assert_not_called()


def test_typed_legacy_failure_is_preserved(ops):
    error = CloudServiceError("backend rejected request")
    error.typed_legacy_session_create_400 = True
    ops.client._create_traigent_session_via_api.side_effect = error
    result = ops.create_session("fn", {"x": [1]})
    assert result.session_id == "local"
    assert result.degraded is True
    assert result.typed_legacy_session_create_400 is True


@pytest.mark.asyncio
async def test_create_from_running_loop(ops):
    result = ops.create_session("fn", {"x": [1]})
    assert result.session_id == "session"
    assert result.backend_connected is True
    assert asyncio.get_running_loop().is_running()


@pytest.mark.parametrize("governed", [False, True])
def test_unexpected_preflight_failure(ops, governed, caplog):
    ops.client.auth_manager.has_api_key.side_effect = RuntimeError(
        "unexpected preflight"
    )
    kwargs = {"promotion_policy": {"strict": True}} if governed else {}
    if governed:
        with pytest.raises(RuntimeError, match="unexpected preflight"):
            ops.create_session("fn", {"x": [1]}, **kwargs)
        ops._create_local_fallback_session.assert_not_called()
    else:
        result = ops.create_session("fn", {"x": [1]}, **kwargs)
        assert result.session_id == "local" and result.degraded is True
        assert "Unexpected error in create_session" in caplog.text


def test_governed_without_key_stays_local_and_warns(ops, caplog):
    ops.client.auth_manager.has_api_key.return_value = False
    result = ops.create_session("fn", {"x": [1]}, promotion_policy={"strict": True})
    assert result.session_id == "local"
    assert not result.degraded
    assert "no API key is configured" in caplog.text
    ops.client._create_traigent_session_via_api.assert_not_called()


def test_success_records_mapping_security_active_and_local(ops):
    ops._persist_connected_session_locally = Mock()
    result = ops.create_session("fn", {"x": [1]}, metadata={"max_trials": 2})
    assert result.session_id == "session"
    assert ops.client.session_bridge.create_session_mapping.call_args.kwargs == {
        "session_id": "session",
        "experiment_id": "experiment",
        "experiment_run_id": "run",
        "function_name": "fn",
        "configuration_space": {"x": [1]},
        "objectives": ["maximize"],
    }
    assert ops.client._register_security_session.call_args.args[0] == "session"
    assert ops.client._active_sessions["session"].max_trials == 2
    assert (
        ops._persist_connected_session_locally.call_args.kwargs["session_id"]
        == "session"
    )


def test_eviction_prefers_completed_session(ops):
    from datetime import UTC, datetime, timedelta

    from traigent.cloud.models import OptimizationSessionStatus

    now = datetime.now(UTC)
    ops.client._max_active_sessions = 2
    ops.client._active_sessions.update(
        {
            "active": SimpleNamespace(
                status=OptimizationSessionStatus.ACTIVE,
                created_at=now - timedelta(days=2),
            ),
            "completed": SimpleNamespace(
                status=OptimizationSessionStatus.COMPLETED,
                created_at=now - timedelta(days=1),
            ),
        }
    )
    ops.create_session("fn", {"x": [1]})
    assert set(ops.client._active_sessions) == {"active", "session"}


def test_auth_fallback_resets_before_local_fallback(ops):
    from traigent.cloud.auth import AuthenticationError

    events = []

    async def reset(reason):
        events.append(reason)

    ops._reset_client_session = reset
    ops.client._create_traigent_session_via_api.side_effect = AuthenticationError(
        "transport error"
    )
    ops._create_local_fallback_session.side_effect = lambda *args: (
        events.append("local") or "local"
    )
    result = ops.create_session("fn", {"x": [1]})
    assert result.session_id == "local" and result.degraded
    assert events == ["create_session auth_failure", "local"]
