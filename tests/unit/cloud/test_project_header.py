"""Traigent-backend writes carry X-Project-Id when a project is configured."""

from __future__ import annotations

import logging
from unittest.mock import AsyncMock, Mock, patch

import pytest

from traigent.cloud.backend_client import BackendIntegratedClient
from traigent.cloud.client import CloudServiceError, TraigentCloudClient
from traigent.config.project import apply_project_header

pytestmark = pytest.mark.backend_online

API_KEY = "tg_" + "a" * 61  # pragma: allowlist secret
SID = "sess-proj"
BODY_422 = (
    '{"success":false,"error_code":"PROJECT_CONTEXT_REQUIRED",'
    '"message":"customer text must not appear"}'
)
BODY_OK = "{}"


@pytest.fixture
def no_project(monkeypatch):
    monkeypatch.delenv("TRAIGENT_PROJECT_ID", raising=False)


@pytest.fixture
def with_project(monkeypatch):
    monkeypatch.setenv("TRAIGENT_PROJECT_ID", "proj_env")


def _make_client(**kwargs) -> BackendIntegratedClient:
    with patch("traigent.cloud.backend_client.AIOHTTP_AVAILABLE", True):
        client = BackendIntegratedClient(
            api_key=API_KEY, base_url="http://localhost:5000", **kwargs
        )
    client.auth_manager.auth.get_headers = AsyncMock(
        return_value={"X-API-Key": "k"}  # pragma: allowlist secret
    )
    client.session_bridge.create_session_mapping(
        session_id=SID,
        experiment_id="exp-1",
        experiment_run_id="run-1",
        function_name="f",
        configuration_space={},
        objectives=["accuracy"],
    )
    client.local_storage = Mock()
    return client


def _fake_http(status: int, body: str):
    response = AsyncMock()
    response.status = status
    response.text = AsyncMock(return_value=body)
    response.json = AsyncMock(return_value={})
    response.headers = {}
    response.__aenter__ = AsyncMock(return_value=response)
    response.__aexit__ = AsyncMock(return_value=None)
    session = AsyncMock()
    session.post = Mock(return_value=response)
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=None)
    return session


async def _create(client, session):
    ops = client._api_ops
    headers = await client.auth_manager.augment_headers({"Content-Type": "x"})
    with patch("aiohttp.ClientSession", return_value=session):
        await ops._post_session_creation({"problem_statement": "p"}, headers, None)


async def _finalize(client, session):
    with patch("aiohttp.ClientSession", return_value=session):
        return await client._session_ops._finalize_session_via_api(SID, None)


async def _next_trial(client, session):
    client._ensure_session = AsyncMock(return_value=session)
    with patch(
        "traigent.cloud.trial_operations.is_backend_offline", return_value=False
    ):
        return await client._trial_ops.request_trial_slot(SID)


async def _results(client, session):
    client._ensure_session = AsyncMock(return_value=session)
    with patch(
        "traigent.cloud.trial_operations.is_backend_offline", return_value=False
    ):
        return await client._trial_ops.submit_trial_result_via_session(
            SID, "t-1", {"m": "a"}, {"accuracy": 1.0}, "completed"
        )


WRITES = {
    "finalize": _finalize,
    "next_trial": _next_trial,
    "results": _results,
}


def _sent_headers(session) -> dict[str, str]:
    return session.post.call_args.kwargs["headers"]


@pytest.mark.asyncio
@pytest.mark.parametrize("write", list(WRITES))
async def test_writes_carry_project_header_when_configured(with_project, write):
    client = _make_client()
    session = _fake_http(200, BODY_OK)
    await WRITES[write](client, session)
    assert _sent_headers(session)["X-Project-Id"] == "proj_env"


@pytest.mark.asyncio
async def test_session_create_carries_project_header(with_project):
    client = _make_client()
    session = _fake_http(201, '{"session_id": "s", "status": "active"}')
    try:
        await _create(client, session)
    except Exception:  # response parsing is irrelevant here
        pass
    assert _sent_headers(session)["X-Project-Id"] == "proj_env"


@pytest.mark.asyncio
@pytest.mark.parametrize("write", list(WRITES))
async def test_writes_send_no_project_header_when_unconfigured(no_project, write):
    client = _make_client()
    session = _fake_http(200, BODY_OK)
    await WRITES[write](client, session)
    assert "X-Project-Id" not in _sent_headers(session)


@pytest.mark.asyncio
@pytest.mark.parametrize("write", list(WRITES))
async def test_explicit_project_id_wins_over_env(with_project, write):
    client = _make_client(project_id="proj_explicit")
    session = _fake_http(200, BODY_OK)
    await WRITES[write](client, session)
    assert _sent_headers(session)["X-Project-Id"] == "proj_explicit"


@pytest.mark.asyncio
async def test_real_merge_keeps_caller_header_case_insensitively(with_project):
    client = _make_client()
    merged = await client.auth_manager.augment_headers({"x-project-id": "caller"})
    names = [n for n in merged if n.lower() == "x-project-id"]
    assert names == ["x-project-id"]
    assert merged["x-project-id"] == "caller"


def test_helper_keeps_caller_header(with_project):
    headers = {"x-project-id": "caller_proj"}
    apply_project_header(headers)
    assert headers == {"x-project-id": "caller_proj"}


@pytest.mark.asyncio
async def test_cloud_client_headers_use_explicit_project(with_project):
    client = TraigentCloudClient.__new__(TraigentCloudClient)
    client._project_id = "proj_explicit"
    client.auth = Mock()
    client.auth.get_headers = AsyncMock(return_value={"X-API-Key": "k"})
    headers = await client._get_headers()
    assert headers["X-Project-Id"] == "proj_explicit"


@pytest.mark.asyncio
async def test_generic_and_cloud_target_headers_never_carry_project(with_project):
    from traigent.cloud.auth import AuthManager

    manager = AuthManager(api_key=API_KEY)
    for target in ("cloud", "backend", "both"):
        headers: dict[str, str] = {}
        manager._add_common_headers(headers, target=target)
        assert "X-Project-Id" not in headers


@pytest.mark.asyncio
async def test_provider_client_default_headers_never_contain_project(with_project):
    from traigent.agents.platforms import OpenAIAgentExecutor
    from traigent.cloud.auth import AuthManager

    manager = AuthManager(api_key=API_KEY)

    async def _noop() -> None:
        return None

    async def _mode_headers(mode, target):
        return {"X-API-Key": "k"}  # pragma: allowlist secret

    manager._ensure_authenticated_for_headers = _noop  # type: ignore[method-assign]
    manager._headers_for_mode = _mode_headers  # type: ignore[method-assign]
    manager._latch_emitted_credential = lambda h: None  # type: ignore[method-assign]
    manager._credentials = Mock()

    executor = OpenAIAgentExecutor.__new__(OpenAIAgentExecutor)
    executor.auth_manager = manager
    headers = await executor._get_authenticated_headers()
    assert headers  # the provider client does get the auth headers
    assert not any(k.lower() == "x-project-id" for k in headers)


@pytest.mark.asyncio
async def test_create_422_maps_to_clear_error(with_project):
    client = _make_client()
    with pytest.raises(CloudServiceError) as exc:
        await _create(client, _fake_http(422, BODY_422))
    assert "TRAIGENT_PROJECT_ID" in str(exc.value)
    assert "customer text" not in str(exc.value)


@pytest.mark.asyncio
async def test_finalize_422_is_surfaced_not_none(with_project):
    client = _make_client()
    with pytest.raises(CloudServiceError) as exc:
        await _finalize(client, _fake_http(422, BODY_422))
    assert "TRAIGENT_PROJECT_ID" in str(exc.value)


@pytest.mark.asyncio
async def test_finalize_other_422_still_returns_none(with_project):
    client = _make_client()
    result = await _finalize(
        client, _fake_http(422, '{"error_code":"validation_error"}')
    )
    assert result is None


@pytest.mark.asyncio
@pytest.mark.parametrize("write", ["next_trial", "results"])
async def test_trial_write_422_logs_clear_message(with_project, write, caplog):
    client = _make_client()
    with caplog.at_level(logging.ERROR):
        await WRITES[write](client, _fake_http(422, BODY_422))
    assert any("TRAIGENT_PROJECT_ID" in r.getMessage() for r in caplog.records)


@pytest.mark.asyncio
async def test_hybrid_create_422_maps_to_clear_error(with_project):
    client = _make_client()
    session = _fake_http(422, BODY_422)
    client._ensure_session = AsyncMock(return_value=session)
    with pytest.raises(CloudServiceError) as exc:
        await client._session_ops.create_hybrid_session(
            problem_statement="p",
            search_space={},
            optimization_config={},
        )
    assert "TRAIGENT_PROJECT_ID" in str(exc.value)
