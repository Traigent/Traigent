"""Login and token refresh must not replay on an ambiguous failure (#1975)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from traigent.cloud.auth import AuthStatus, UnifiedAuthConfig
from traigent.cloud.password_auth_handler import PasswordAuthHandler
from traigent.cloud.token_manager import TokenManager

pytestmark = pytest.mark.backend_online

PINNED_BACKEND_URL = "https://backend.example.test"


@pytest.fixture(autouse=True)
def _offline_pinned_backend(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("TRAIGENT_BACKEND_URL", PINNED_BACKEND_URL)
    monkeypatch.delenv("TRAIGENT_API_URL", raising=False)
    monkeypatch.setattr(
        "traigent.cloud.url_security.socket.getaddrinfo",
        lambda *_a, **_k: [(0, 0, 0, "", ("93.184.216.34", 0))],
    )
    monkeypatch.setattr("asyncio.sleep", AsyncMock())


def _failing_session(exc: BaseException) -> tuple[MagicMock, MagicMock]:
    session = MagicMock()
    session.post = MagicMock(side_effect=exc)
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=session)
    cm.__aexit__ = AsyncMock(return_value=False)
    return MagicMock(return_value=cm), session


@pytest.mark.asyncio
@pytest.mark.parametrize("exc", [TimeoutError(), ConnectionResetError()])
async def test_login_ambiguous_failure_is_single_shot(exc):
    factory, session = _failing_session(exc)
    with patch("traigent.cloud.password_auth_handler.aiohttp.ClientSession", factory):
        result = await PasswordAuthHandler()._perform_authentication(
            {"email": "a@example.com", "password": "pw"}  # pragma: allowlist secret
        )
    assert result is None
    assert session.post.call_count == 1, "login POST must not be replayed"


@pytest.mark.asyncio
@pytest.mark.parametrize("exc", [TimeoutError(), ConnectionResetError()])
async def test_refresh_ambiguous_failure_is_single_shot(exc):

    factory, session = _failing_session(exc)
    manager = TokenManager(
        UnifiedAuthConfig(cloud_base_url="http://localhost:5000", auto_refresh=False)
    )
    with patch("traigent.cloud.token_manager.aiohttp.ClientSession", factory):
        result = await manager.refresh_jwt_secure("test_refresh_token_12345")
    assert result.success is False
    assert result.status != AuthStatus.AUTHENTICATED
    assert session.post.call_count == 1, "refresh POST must not be replayed"


@pytest.mark.asyncio
async def test_helper_raises_non_retryable_and_keeps_passthrough():
    from traigent.cloud.resilient_client import single_shot_unkeyed_auth_call
    from traigent.utils.exceptions import NonRetryableError

    async def timeout():
        raise TimeoutError

    async def bad_value():
        raise ValueError("x")

    async def rate_limited():
        raise RuntimeError("Backend authentication rate limited")

    with pytest.raises(NonRetryableError):
        await single_shot_unkeyed_auth_call(timeout)()
    with pytest.raises(ValueError):
        await single_shot_unkeyed_auth_call(bad_value, (ValueError,))()
    with pytest.raises(RuntimeError):
        await single_shot_unkeyed_auth_call(rate_limited)()
