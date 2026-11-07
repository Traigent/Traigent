"""Login and token refresh must not replay on an ambiguous failure (#1975)."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from traigent.cloud.auth import (
    AuthCredentials,
    AuthMode,
    AuthStatus,
    UnifiedAuthConfig,
)
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
    from traigent.utils.exceptions import NonRetryableError, RateLimitError

    async def timeout():
        raise TimeoutError

    async def bad_value():
        raise ValueError("x")

    async def rate_limited():
        raise RateLimitError("Backend authentication rate limited")

    with pytest.raises(NonRetryableError):
        await single_shot_unkeyed_auth_call(timeout)()
    with pytest.raises(ValueError):
        await single_shot_unkeyed_auth_call(bad_value, (ValueError,))()
    with pytest.raises(RateLimitError):
        await single_shot_unkeyed_auth_call(rate_limited)()


@pytest.mark.asyncio
async def test_text_containing_429_without_status_is_not_retried():
    """A dropped connection whose text mentions 429 is still ambiguous (no replay)."""
    from traigent.cloud.resilient_client import (
        ResilientClient,
        single_shot_unkeyed_auth_call,
    )
    from traigent.utils.exceptions import NonRetryableError

    async def dropped():
        raise ConnectionResetError("connection reset, request id 429 rate limited")

    with pytest.raises(NonRetryableError):
        await single_shot_unkeyed_auth_call(dropped)()

    calls = 0

    async def op():
        nonlocal calls
        calls += 1
        raise RuntimeError("Backend authentication rate limited HTTP 429")

    with pytest.raises(NonRetryableError):
        await ResilientClient(max_retries=3).execute_with_retry(
            single_shot_unkeyed_auth_call(op)
        )
    assert calls == 1


def _response(status: int) -> MagicMock:
    response = MagicMock()
    response.status = status
    response.text = AsyncMock(return_value="")
    return response


@pytest.mark.asyncio
async def test_real_429_response_follows_rate_limit_path():
    from traigent.cloud.resilient_client import (
        ResilientClient,
        single_shot_unkeyed_auth_call,
    )
    from traigent.utils.exceptions import RateLimitError

    with pytest.raises(RateLimitError):
        await TokenManager._parse_jwt_refresh_response(_response(429), "tok")
    with pytest.raises(RateLimitError):
        await PasswordAuthHandler._check_login_response(_response(429), ValueError)

    calls = 0

    async def op():
        nonlocal calls
        calls += 1
        if calls < 3:
            await PasswordAuthHandler._check_login_response(_response(429), ValueError)
        return "ok"

    result = await ResilientClient(max_retries=3).execute_with_retry(
        single_shot_unkeyed_auth_call(op)
    )
    assert result == "ok"
    assert calls == 3, "429 is refused before processing, so it is retried"


def _jwt_manager(
    holder: dict, lock: asyncio.Lock, reads: list[bool] | None = None
) -> TokenManager:
    manager = TokenManager(
        UnifiedAuthConfig(cloud_base_url="http://localhost:5000", auto_refresh=False)
    )

    def get_credentials():
        if reads is not None:
            reads.append(lock.locked())
        return holder["creds"]

    def set_credentials(creds, _status):
        holder["creds"] = creds

    manager.set_callbacks(
        get_credentials=get_credentials,
        set_credentials=set_credentials,
        cache_credentials=AsyncMock(),
        auth_lock=lock,
    )
    return manager


@pytest.mark.asyncio
async def test_ambiguous_refresh_failure_forces_reauthentication():
    holder = {
        "creds": AuthCredentials(
            mode=AuthMode.JWT_TOKEN,
            jwt_token="old_access",
            refresh_token="old_refresh_token",  # pragma: allowlist secret
        )
    }
    manager = _jwt_manager(holder, asyncio.Lock())
    manager.store_tokens(
        {"access_token": "old_access", "refresh_token": "old_refresh_token"}
    )
    factory, session = _failing_session(TimeoutError())
    with patch("traigent.cloud.token_manager.aiohttp.ClientSession", factory):
        first = await manager.refresh_access_token()
        second = await manager.refresh_access_token()

    assert first.success is False
    assert session.post.call_count == 1, "old refresh token must not be resent"
    assert second.success is False
    assert second.status == AuthStatus.INVALID
    assert second.error_message == "No credentials available"
    assert holder["creds"] is None
    assert manager.refresh_token_secure is None


@pytest.mark.asyncio
async def test_refresh_reads_credentials_under_the_lock():
    holder = {
        "creds": AuthCredentials(mode=AuthMode.JWT_TOKEN, refresh_token="rt")
    }  # pragma: allowlist secret
    reads: list[bool] = []
    lock = asyncio.Lock()
    manager = _jwt_manager(holder, lock, reads)

    async with lock:
        task = asyncio.create_task(manager.refresh_access_token())
        for _ in range(5):  # real loop turns (asyncio.sleep is mocked here)
            fut = asyncio.get_running_loop().create_future()
            asyncio.get_running_loop().call_soon(fut.set_result, None)
            await fut
        assert reads == [], "credentials must not be read before taking the lock"
        holder["creds"] = None
    result = await task
    assert reads == [True], "credentials are read while holding the lock"
    assert result.error_message == "No credentials available"


# ---- round 2 review findings (#1975) ----


def _seeded_manager(holder: dict | None = None) -> tuple[TokenManager, dict]:
    holder = holder or {
        "creds": AuthCredentials(
            mode=AuthMode.JWT_TOKEN,
            jwt_token="old_access",
            refresh_token="old_refresh_token",  # pragma: allowlist secret
        )
    }
    manager = _jwt_manager(holder, asyncio.Lock())
    manager.store_tokens(
        {"access_token": "old_access", "refresh_token": "old_refresh_token"}
    )
    return manager, holder


def _session_with_response(response: MagicMock) -> tuple[MagicMock, MagicMock]:
    post_cm = MagicMock()
    post_cm.__aenter__ = AsyncMock(return_value=response)
    post_cm.__aexit__ = AsyncMock(return_value=False)
    session = MagicMock()
    session.post = MagicMock(return_value=post_cm)
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=session)
    cm.__aexit__ = AsyncMock(return_value=False)
    return MagicMock(return_value=cm), session


@pytest.mark.asyncio
async def test_value_error_with_429_text_is_not_retried():
    import json

    from traigent.cloud.resilient_client import (
        ResilientClient,
        single_shot_unkeyed_auth_call,
    )
    from traigent.utils.exceptions import NonRetryableError

    calls = 0

    async def op():
        nonlocal calls
        calls += 1
        raise json.JSONDecodeError("Expecting value 429 rate limit", "doc", 0)

    with pytest.raises(NonRetryableError):
        await ResilientClient(max_retries=3).execute_with_retry(
            single_shot_unkeyed_auth_call(op)
        )
    assert calls == 1

    # End to end through the refresh path: a decode error is ambiguous.
    response = _response(200)
    response.json = AsyncMock(
        side_effect=json.JSONDecodeError("bad 429 rate limit", "doc", 0)
    )
    factory, session = _session_with_response(response)
    manager, holder = _seeded_manager()
    with patch("traigent.cloud.token_manager.aiohttp.ClientSession", factory):
        result = await manager.refresh_access_token()
    assert result.success is False
    assert session.post.call_count == 1
    assert holder["creds"] is None
    assert manager.refresh_token_secure is None


@pytest.mark.asyncio
async def test_cancellation_during_refresh_invalidates_token_and_propagates():
    factory, session = _failing_session(asyncio.CancelledError())
    manager, holder = _seeded_manager()
    with patch("traigent.cloud.token_manager.aiohttp.ClientSession", factory):
        with pytest.raises(asyncio.CancelledError):
            await manager.refresh_access_token()
    assert session.post.call_count == 1
    assert holder["creds"] is None
    assert manager.refresh_token_secure is None


@pytest.mark.asyncio
async def test_malformed_2xx_body_clears_old_token():
    response = _response(200)
    response.json = AsyncMock(
        return_value={
            "success": True,
            "data": {"access_token": "new_access", "expires_in": "not-a-number"},
        }
    )
    factory, session = _session_with_response(response)
    manager, holder = _seeded_manager()
    with patch("traigent.cloud.token_manager.aiohttp.ClientSession", factory):
        result = await manager.refresh_access_token()
    assert result.success is False
    assert session.post.call_count == 1
    assert holder["creds"] is None
    assert manager.refresh_token_secure is None


@pytest.mark.asyncio
async def test_auth_posts_disable_redirects_and_3xx_is_not_retried():
    manager, holder = _seeded_manager()
    factory, session = _session_with_response(_response(307))
    with patch("traigent.cloud.token_manager.aiohttp.ClientSession", factory):
        result = await manager.refresh_access_token()
    assert result.success is False
    assert session.post.call_count == 1
    assert session.post.call_args.kwargs["allow_redirects"] is False
    assert holder["creds"] is None
    assert manager.refresh_token_secure is None

    factory, session = _session_with_response(_response(308))
    with patch("traigent.cloud.password_auth_handler.aiohttp.ClientSession", factory):
        login = await PasswordAuthHandler()._perform_authentication(
            {"email": "a@example.com", "password": "pw"}  # pragma: allowlist secret
        )
    assert login is None
    assert session.post.call_count == 1
    assert session.post.call_args.kwargs["allow_redirects"] is False


@pytest.mark.asyncio
async def test_cache_callback_failure_after_commit_keeps_new_token():
    response = _response(200)
    response.json = AsyncMock(
        return_value={
            "success": True,
            "data": {
                "access_token": "new_access",
                "refresh_token": "new_refresh_token",
                "expires_in": 3600,
            },
        }
    )
    factory, session = _session_with_response(response)
    holder = {
        "creds": AuthCredentials(
            mode=AuthMode.JWT_TOKEN,
            jwt_token="old_access",
            refresh_token="old_refresh_token",  # pragma: allowlist secret
        )
    }
    manager = TokenManager(
        UnifiedAuthConfig(
            cloud_base_url="http://localhost:5000",
            auto_refresh=False,
            cache_credentials=True,
        )
    )
    manager.set_callbacks(
        get_credentials=lambda: holder["creds"],
        set_credentials=lambda creds, _s: holder.update(creds=creds),
        cache_credentials=AsyncMock(side_effect=RuntimeError("disk full")),
        auth_lock=asyncio.Lock(),
    )
    manager.store_tokens(
        {"access_token": "old_access", "refresh_token": "old_refresh_token"}
    )
    with patch("traigent.cloud.token_manager.aiohttp.ClientSession", factory):
        result = await manager.refresh_access_token()
    assert result.success is True
    assert session.post.call_count == 1
    assert holder["creds"].jwt_token == "new_access"
    assert holder["creds"].refresh_token == "new_refresh_token"
    # The old refresh token is not retained anywhere.
    secure = manager.refresh_token_secure
    assert secure is None or secure.get_value() != "old_refresh_token"


@pytest.mark.asyncio
async def test_login_decode_error_with_429_text_is_single_shot():
    import json

    response = _response(200)
    response.json = AsyncMock(
        side_effect=json.JSONDecodeError("bad 429 rate limit", "doc", 0)
    )
    factory, session = _session_with_response(response)
    with patch("traigent.cloud.password_auth_handler.aiohttp.ClientSession", factory):
        result = await PasswordAuthHandler()._perform_authentication(
            {"email": "a@example.com", "password": "pw"}  # pragma: allowlist secret
        )
    assert result is None
    assert session.post.call_count == 1


# ---- #1975 follow-ups: credentials cache + OAuth2 refresh ----


def _oauth_manager(holder: dict, clear_cache=None) -> TokenManager:
    manager = TokenManager(
        UnifiedAuthConfig(cloud_base_url="https://api.example.test", auto_refresh=False)
    )

    def set_credentials(creds, _status):
        holder["creds"] = creds

    manager.set_callbacks(
        get_credentials=lambda: holder["creds"],
        set_credentials=set_credentials,
        cache_credentials=AsyncMock(),
        auth_lock=asyncio.Lock(),
        clear_cached_credentials=clear_cache,
    )
    return manager


def _oauth_holder() -> dict:
    return {
        "creds": AuthCredentials(
            mode=AuthMode.OAUTH2,
            refresh_token="old_oauth_refresh",  # pragma: allowlist secret
            client_id="client-id",
            client_secret="client-secret",  # pragma: allowlist secret
        )
    }


@pytest.mark.asyncio
async def test_invalidation_removes_cache_file_so_old_refresh_token_is_not_reloaded(
    tmp_path,
):
    from traigent.cloud.credential_resolver import CredentialResolver

    cache_file = tmp_path / "credentials.enc"
    config = UnifiedAuthConfig(
        cloud_base_url="http://localhost:5000",
        auto_refresh=False,
        credentials_file=str(cache_file),
    )
    resolver = CredentialResolver(config)
    old = AuthCredentials(
        mode=AuthMode.JWT_TOKEN,
        jwt_token="old_access",
        refresh_token="old_refresh_token",  # pragma: allowlist secret
    )
    await resolver.cache(old)
    assert cache_file.exists()
    reloaded = await resolver.load_cached()
    assert reloaded is not None and reloaded.refresh_token == "old_refresh_token"

    holder = {"creds": old}
    manager = TokenManager(config)
    manager.set_callbacks(
        get_credentials=lambda: holder["creds"],
        set_credentials=lambda c, _s: holder.update(creds=c),
        cache_credentials=AsyncMock(),
        auth_lock=asyncio.Lock(),
        clear_cached_credentials=resolver.clear_cached,
    )
    factory, session = _failing_session(TimeoutError())
    with patch("traigent.cloud.token_manager.aiohttp.ClientSession", factory):
        await manager.refresh_access_token()

    assert not cache_file.exists()
    assert await resolver.load_cached() is None


def _resolver(path, **kw):
    from traigent.cloud.credential_resolver import CredentialResolver

    return CredentialResolver(
        UnifiedAuthConfig(credentials_file=str(path), auto_refresh=False, **kw)
    )


def test_clear_cached_absent_file_and_no_path_are_noops(tmp_path):
    _resolver(tmp_path / "missing.enc").clear_cached()
    from traigent.cloud.credential_resolver import CredentialResolver

    CredentialResolver(UnifiedAuthConfig(auto_refresh=False)).clear_cached()


def test_clear_cached_clears_resolved_symlink_target_when_owned_regular_file(tmp_path):
    target = tmp_path / "real.enc"
    target.write_text("secret")
    link = tmp_path / "credentials.enc"
    link.symlink_to(target)
    _resolver(link).clear_cached()
    assert not target.exists(), "resolved target (what cache() wrote) is cleared"


def test_clear_cached_unlinks_only_link_when_target_not_owned_regular_file(
    tmp_path, monkeypatch
):
    target = tmp_path / "real.enc"
    target.write_text("keep me")
    link = tmp_path / "credentials.enc"
    link.symlink_to(target)
    monkeypatch.setattr("traigent.cloud.credential_resolver.os.getuid", lambda: -1)
    _resolver(link).clear_cached()
    assert target.read_text() == "keep me"
    assert not link.is_symlink()


def test_clear_cached_skipped_when_caching_disabled(tmp_path):
    f = tmp_path / "credentials.enc"
    f.write_text("not ours")
    _resolver(f, cache_credentials=False).clear_cached()
    assert f.read_text() == "not ours"


def test_clear_cached_truncates_when_delete_fails(tmp_path, monkeypatch):
    f = tmp_path / "credentials.enc"
    f.write_text("old token blob")
    monkeypatch.setattr(
        "pathlib.Path.unlink", MagicMock(side_effect=PermissionError("no"))
    )
    resolver = _resolver(f)
    resolver.clear_cached()
    assert f.read_text() == ""
    assert resolver._cache_load_disabled is False


@pytest.mark.asyncio
async def test_clear_cached_total_failure_disables_cache_loading(tmp_path, monkeypatch):
    resolver = _resolver(tmp_path / "credentials.enc")
    await resolver.cache(
        AuthCredentials(
            mode=AuthMode.JWT_TOKEN,
            jwt_token="a",
            refresh_token="old_refresh_token",  # pragma: allowlist secret
        )
    )
    assert await resolver.load_cached() is not None
    monkeypatch.setattr(
        "pathlib.Path.unlink", MagicMock(side_effect=PermissionError("no"))
    )
    real_open = open

    def deny_open(file, mode="r", *a, **k):
        if "+" in mode:
            raise PermissionError("no")
        return real_open(file, mode, *a, **k)

    monkeypatch.setattr("builtins.open", deny_open)
    resolver.clear_cached()
    monkeypatch.undo()
    assert resolver._cache_load_disabled is True
    assert await resolver.load_cached() is None, "file still exists but is not loaded"


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [503, 500, 408])
async def test_oauth_ambiguous_status_clears_state_and_returns_authresult(status):
    holder = _oauth_holder()
    clear_cache = MagicMock()
    manager = _oauth_manager(holder, clear_cache)
    factory, session = _session_with_response(_response(status))
    with patch("traigent.cloud.token_manager.aiohttp.ClientSession", factory):
        result = await manager.refresh_oauth2()
        again = await manager.refresh_oauth2()
    assert result.success is False and result.status == AuthStatus.INVALID
    assert holder["creds"] is None
    clear_cache.assert_called_once()
    assert again.status == AuthStatus.UNAUTHENTICATED
    assert session.post.call_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("exc", [TimeoutError(), ConnectionResetError()])
async def test_oauth_ambiguous_failure_is_single_shot_and_clears_token(exc):
    holder = _oauth_holder()
    clear_cache = MagicMock()
    manager = _oauth_manager(holder, clear_cache)
    factory, session = _failing_session(exc)
    with patch("traigent.cloud.token_manager.aiohttp.ClientSession", factory):
        first = await manager.refresh_oauth2()
        second = await manager.refresh_oauth2()

    assert first.success is False
    assert session.post.call_count == 1, "OAuth refresh POST must not be replayed"
    assert session.post.call_args.kwargs["allow_redirects"] is False
    assert holder["creds"] is None
    clear_cache.assert_called_once()
    assert second.success is False
    assert second.status == AuthStatus.UNAUTHENTICATED
    assert session.post.call_count == 1, "the old token must not be resent"


@pytest.mark.asyncio
async def test_oauth_cancellation_clears_and_propagates():
    holder = _oauth_holder()
    clear_cache = MagicMock()
    manager = _oauth_manager(holder, clear_cache)
    factory, session = _failing_session(asyncio.CancelledError())
    with patch("traigent.cloud.token_manager.aiohttp.ClientSession", factory):
        with pytest.raises(asyncio.CancelledError):
            await manager.refresh_oauth2()
    assert session.post.call_count == 1
    assert holder["creds"] is None
    clear_cache.assert_called_once()


@pytest.mark.asyncio
async def test_oauth_definitive_rejection_keeps_existing_behaviour():
    holder = _oauth_holder()
    clear_cache = MagicMock()
    manager = _oauth_manager(holder, clear_cache)
    response = _response(400)
    factory, session = _session_with_response(response)
    with patch("traigent.cloud.token_manager.aiohttp.ClientSession", factory):
        result = await manager.refresh_oauth2()
    assert result.success is False
    assert session.post.call_count == 1
    assert holder["creds"] is not None
    clear_cache.assert_not_called()
