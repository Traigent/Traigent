"""optimization_only: a 200 ``/keys/validate`` answering ``optimization_only: true`` latches
the mode for that credential's scope before validation returns.

Recording transports only: no real network.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from traigent.utils import optimization_only_egress as oo

from .harness import capture_egress
from .scenario import BACKEND_URL, FAKE_API_KEY
from .test_optimization_only_egress import DIRTY_TYPED_SESSION, _env
from .test_optimization_only_latch_guards import REFUSING_SITES

pytestmark = pytest.mark.backend_online

OTHER_KEY = "uk_" + "z" * 43
SITES = ("hybrid_create", "dataset_upload", "workflow_traces", "hosted_optimization")


@pytest.fixture(autouse=True)
def _clean(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    _env(monkeypatch, tmp_path)
    monkeypatch.delenv(oo.ENV_VAR, raising=False)
    oo.reset_latch()
    yield
    oo.reset_latch()


def _scope(key: str = FAKE_API_KEY) -> str:
    scope = oo.credential_scope(BACKEND_URL, {"X-API-Key": key})
    assert scope is not None
    return scope


def _validate(response: dict[str, Any]) -> Any:
    from traigent.cloud.auth import AuthManager, UnifiedAuthConfig

    auth = AuthManager(
        UnifiedAuthConfig(backend_base_url=BACKEND_URL, cache_credentials=False)
    )
    with capture_egress() as cap:
        cap.backend.validate_response = response
        result = asyncio.run(auth._validate_api_key_with_backend(FAKE_API_KEY))
    assert [r for r in cap.requests if r.path.endswith("/keys/validate")]
    return result


def test_true_latches_scope_and_everything_after_is_gated() -> None:
    assert _validate({"valid": True, "optimization_only": True}) is None
    assert oo.is_latched(_scope())
    for name in SITES:
        with capture_egress() as cap:
            with pytest.raises(oo.OptimizationOnlyRefusal):
                asyncio.run(REFUSING_SITES[name]())
        assert cap.requests == [], (name, [r.route for r in cap.requests])


def test_session_create_body_is_projected_after_latch() -> None:
    from .test_optimization_only_egress import _create_session_raw, _session_posts

    _validate({"valid": True, "optimization_only": True})
    expected = oo.project_for_phase(oo.PHASE_SESSION_CREATE_AUTO, DIRTY_TYPED_SESSION)
    with capture_egress() as cap:
        asyncio.run(_create_session_raw(DIRTY_TYPED_SESSION))
    (post,) = _session_posts(cap)
    assert json.loads(post.body) == expected
    assert "Alice" not in post.body.decode()


@pytest.mark.parametrize(
    "response",
    [
        {"valid": True, "optimization_only": False},
        {"valid": True},
        {"valid": True, "optimization_only": "true"},
        {"valid": True, "optimization_only": 1},
        {"valid": True, "optimization_only": None},
        {"valid": True, "optimization_only": [True]},
    ],
)
def test_anything_but_json_true_does_not_latch(response: dict[str, Any]) -> None:
    assert _validate(response) is None
    assert not oo.is_latched(_scope())
    assert oo.latched_origins() == set()


def test_false_never_unlatches() -> None:
    oo.latch_on(_scope())
    _validate({"valid": True, "optimization_only": False})
    assert oo.is_latched(_scope())


def test_explicit_false_client_does_not_override_the_latch() -> None:
    from traigent.cloud.backend_client import (
        BackendClientConfig,
        BackendIntegratedClient,
    )

    client = BackendIntegratedClient(
        api_key=FAKE_API_KEY,
        backend_config=BackendClientConfig(backend_base_url=BACKEND_URL),
        optimization_only=False,
    )
    _validate({"valid": True, "optimization_only": True})
    assert oo.is_active(oo.policy_of(client))


def test_other_credential_is_unaffected() -> None:
    _validate({"valid": True, "optimization_only": True})
    assert oo.is_latched(_scope())
    assert not oo.is_latched(_scope(OTHER_KEY))
    assert not oo.is_active(oo.Policy(None, _scope(OTHER_KEY)))


def test_invalid_key_body_does_not_latch() -> None:
    oo.reset_latch()
    assert (
        oo.note_key_validation({"valid": False, "optimization_only": True}, _scope())
        is False
    )
    assert oo.note_key_validation({"optimization_only": True}, None) is False
    assert not oo.is_latched(_scope())


def test_requests_fallback_path_latches_too(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("traigent.cloud.auth.AIOHTTP_AVAILABLE", False)
    assert _validate({"valid": True, "optimization_only": True}) is None
    assert oo.is_latched(_scope())


# ---- every credential a flagged manager emits is latched --------------------------------


def _validate_on(auth: Any, response: dict[str, Any]) -> None:
    with capture_egress() as cap:
        cap.backend.validate_response = response
        asyncio.run(auth._validate_api_key_with_backend(FAKE_API_KEY))


def _manager() -> Any:
    from traigent.cloud.auth import AuthManager, UnifiedAuthConfig

    return AuthManager(
        UnifiedAuthConfig(backend_base_url=BACKEND_URL, cache_credentials=False)
    )


def _flagged_manager() -> Any:
    auth = _manager()
    _validate_on(auth, {"valid": True, "optimization_only": True})
    assert auth._server_optimization_only is True
    return auth


def _store(auth: Any, jwt: str, api_key: str | None = None) -> None:
    from traigent.cloud.auth import AuthCredentials, AuthMode, AuthStatus

    auth._set_credentials_and_status(
        AuthCredentials(mode=AuthMode.JWT_TOKEN, jwt_token=jwt, api_key=api_key),
        AuthStatus.AUTHENTICATED,
    )


def _jwt_scope(jwt: str) -> str:
    scope = oo.credential_scope(BACKEND_URL, {"Authorization": f"Bearer {jwt}"})
    assert scope is not None
    return scope


@pytest.mark.parametrize("both", [False, True])
def test_stored_jwt_first_send_is_projected(both: bool) -> None:
    from .test_optimization_only_egress import _create_session_raw, _session_posts

    _store(_flagged_manager(), "jwt.one.sig", FAKE_API_KEY)
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer jwt.one.sig",
    }
    if both:
        headers["X-API-Key"] = FAKE_API_KEY
    expected = oo.project_for_phase(oo.PHASE_SESSION_CREATE_AUTO, DIRTY_TYPED_SESSION)
    with capture_egress() as cap:
        asyncio.run(_create_session_raw(DIRTY_TYPED_SESSION, headers=headers))
    (post,) = _session_posts(cap)
    assert json.loads(post.body) == expected
    assert "Alice" not in post.body.decode()


def test_refresh_response_without_api_key_stays_latched() -> None:
    auth = _flagged_manager()
    _store(auth, "jwt.one.sig", FAKE_API_KEY)
    _store(auth, "jwt.two.sig", None)  # the refresh response carried no API key
    assert oo.is_latched(_jwt_scope("jwt.one.sig"))
    assert oo.is_latched(_jwt_scope("jwt.two.sig"))


def test_get_auth_headers_latches_whatever_it_emits() -> None:
    auth = _flagged_manager()
    _store(auth, "jwt.h.sig")
    oo.reset_latch()  # drop the store-time latch: the emit path must restore it
    headers = asyncio.run(auth.get_auth_headers())
    assert headers.get("Authorization") == "Bearer jwt.h.sig"
    assert oo.is_latched(_jwt_scope("jwt.h.sig"))


def test_user_jwt_on_unflagged_manager_is_not_latched() -> None:
    auth = _manager()
    _validate_on(auth, {"valid": True, "optimization_only": False})
    assert auth._server_optimization_only is False
    _store(auth, "jwt.free.sig", FAKE_API_KEY)
    assert not oo.is_latched(_jwt_scope("jwt.free.sig"))
    assert not oo.is_active(oo.Policy(None, _jwt_scope("jwt.free.sig")))


def test_second_manager_with_unrelated_jwt_is_unaffected() -> None:
    _flagged_manager()
    other = _manager()
    _store(other, "jwt.other.sig", OTHER_KEY)
    assert not oo.is_latched(_jwt_scope("jwt.other.sig"))


def test_flag_is_sticky_when_a_later_validation_says_false() -> None:
    auth = _flagged_manager()
    _validate_on(auth, {"valid": True, "optimization_only": False})
    assert auth._server_optimization_only is True


def test_explicit_false_still_loses_for_emitted_token() -> None:
    _store(_flagged_manager(), "jwt.exp.sig")
    assert oo.is_active(oo.Policy(False, _jwt_scope("jwt.exp.sig")))


def test_unflagged_hot_path_computes_no_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[Any] = []
    real = oo.credential_scope
    monkeypatch.setattr(
        oo, "credential_scope", lambda *a, **k: calls.append(a) or real(*a, **k)
    )
    auth = _manager()
    _store(auth, "jwt.n.sig")
    asyncio.run(auth.get_auth_headers())
    assert calls == []


def test_stored_credentials_are_only_assigned_through_the_latching_setter() -> None:
    import ast

    import traigent.cloud.auth as auth_module

    tree = ast.parse(Path(auth_module.__file__).read_text())
    offenders: list[int] = []
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for node in ast.walk(fn):
            if not isinstance(node, ast.Assign):
                continue
            for target in node.targets:
                if (
                    isinstance(target, ast.Attribute)
                    and target.attr == "_credentials"
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "self"
                    and fn.name != "_set_credentials_and_status"
                    and not (
                        isinstance(node.value, ast.Constant)
                        and node.value.value is None
                    )
                ):
                    offenders.append(node.lineno)
    assert offenders == [], offenders


def test_login_on_flagged_manager_is_latched_for_direct_header_senders() -> None:
    """After a successful login, a sender that reads the stored JWT directly (no
    get_auth_headers) is refused: observability-style."""
    from traigent.cloud.auth import AuthCredentials, AuthMode, AuthResult, AuthStatus

    auth = _flagged_manager()
    creds = AuthCredentials(mode=AuthMode.JWT_TOKEN, jwt_token="jwt.login.sig")
    asyncio.run(
        auth._handle_dict_success(
            AuthResult(success=True, status=AuthStatus.AUTHENTICATED, credentials=creds)
        )
    )
    asyncio.run(
        auth._handle_successful_auth(
            AuthCredentials(mode=AuthMode.JWT_TOKEN, jwt_token="jwt.mode.sig"),
            AuthResult(success=True, status=AuthStatus.AUTHENTICATED),
        )
    )
    for token in ("jwt.login.sig", "jwt.mode.sig"):
        headers = {"Authorization": f"Bearer {token}"}
        assert oo.is_latched(_jwt_scope(token))
        with pytest.raises(oo.OptimizationOnlyRefusal):
            oo.refuse_if_active(
                "direct_header_send",
                explicit=oo.policy_from_headers(BACKEND_URL, headers),
            )


# ---- latching that itself raises fails closed and loud ----------------------------------


def _boom(*_a: Any, **_k: Any) -> Any:
    raise RuntimeError("scope boom " + FAKE_API_KEY)


def test_validate_latch_failure_on_flagged_response_still_restricts_manager(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(oo, "credential_scope", _boom)
    auth = _manager()
    with caplog.at_level("ERROR", logger="traigent.cloud.auth"):
        _validate_on(auth, {"valid": True, "optimization_only": True})
    assert auth._server_optimization_only is True
    text = caplog.text
    assert "RuntimeError" in text
    assert FAKE_API_KEY not in text
    assert "scope boom" not in text


def test_validate_latch_failure_on_unflagged_response_is_harmless(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from traigent.cloud.auth import _latch_key_validation

    monkeypatch.setattr(oo, "credential_scope", _boom)
    auth = _manager()
    _validate_on(auth, {"valid": True, "optimization_only": False})
    assert auth._server_optimization_only is False
    assert _latch_key_validation(BACKEND_URL, {}, {"valid": True}) is False
    assert _latch_key_validation(BACKEND_URL, {}, "not a dict") is False


def test_flagged_manager_refuses_to_emit_credential_when_latch_raises(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    from traigent.cloud.auth import AuthCredentials, AuthMode, AuthStatus

    auth = _flagged_manager()
    _store(auth, "jwt.pre.sig")  # stored, so no network
    monkeypatch.setattr(oo, "latch_on", _boom)
    creds = AuthCredentials(mode=AuthMode.JWT_TOKEN, jwt_token="jwt.fail.sig")
    with caplog.at_level("ERROR", logger="traigent.cloud.auth"):
        with pytest.raises(oo.OptimizationOnlyRefusal) as info:
            auth._set_credentials_and_status(creds, AuthStatus.AUTHENTICATED)
        with pytest.raises(oo.OptimizationOnlyRefusal):
            asyncio.run(auth.get_auth_headers())
    assert auth._credentials is not creds
    assert "jwt.fail.sig" not in str(info.value)
    assert "RuntimeError" in caplog.text
    assert "jwt.fail.sig" not in caplog.text
    assert FAKE_API_KEY not in caplog.text


def test_unflagged_manager_ignores_latch_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from traigent.cloud.auth import AuthCredentials, AuthMode, AuthStatus

    auth = _manager()
    monkeypatch.setattr(oo, "latch_on", _boom)
    monkeypatch.setattr(oo, "credential_scope", _boom)
    creds = AuthCredentials(mode=AuthMode.JWT_TOKEN, jwt_token="jwt.ok.sig")
    auth._set_credentials_and_status(creds, AuthStatus.AUTHENTICATED)
    assert auth._credentials is creds


# ---- refreshed tokens are latched BEFORE they are committed -----------------------------

NEW_JWT = "eyJhbGciOiJIUzI1NiJ9." + "n" * 40 + "." + "s" * 20
OLD_JWT = "eyJhbGciOiJIUzI1NiJ9." + "o" * 40 + "." + "s" * 20
OLD_REFRESH = "refresh-old-0123456789"


def _refresh_ready(auth: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """Store an OLD token + refresh token and make the refresh transport return a NEW one."""
    from traigent.cloud.auth import AuthCredentials, AuthMode, AuthStatus

    tm = auth._token_manager
    tm.store_tokens({"access_token": OLD_JWT, "refresh_token": OLD_REFRESH})
    auth._set_credentials_and_status(
        AuthCredentials(
            mode=AuthMode.OAUTH2,
            jwt_token=OLD_JWT,
            refresh_token=OLD_REFRESH,
            client_id="cid",
            client_secret="csecret",  # pragma: allowlist secret
        ),
        AuthStatus.AUTHENTICATED,
    )

    async def _fake_refresh(*_a: Any, **_k: Any) -> dict[str, Any]:
        return {
            "access_token": NEW_JWT,
            "refresh_token": "refresh-new-0123456789",
            "expires_in": 3600,
        }

    monkeypatch.setattr(
        "traigent.cloud.resilient_client.ResilientClient.execute_with_retry",
        _fake_refresh,
    )
    monkeypatch.setattr(
        "traigent.cloud.client.cloud_backend_egress_disabled", lambda *_a, **_k: False
    )

    async def _ok_url(url: str, **_k: Any) -> str:
        return url

    monkeypatch.setattr(
        "traigent.cloud.token_manager.validate_cloud_base_url_async", _ok_url
    )


def _assert_old_state_only(auth: Any) -> None:
    tm = auth._token_manager
    assert auth._credentials.jwt_token == OLD_JWT
    assert auth._credentials.refresh_token == OLD_REFRESH
    assert tm.get_authorization_header() == {"Authorization": f"Bearer {OLD_JWT}"}
    assert tm._refresh_token_secure.get_value() == OLD_REFRESH


def test_jwt_refresh_on_flagged_manager_that_cannot_latch_keeps_old_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    auth = _flagged_manager()
    _refresh_ready(auth, monkeypatch)
    real_latch_on = oo.latch_on
    monkeypatch.setattr(oo, "latch_on", _boom)
    result = asyncio.run(auth._token_manager.refresh_jwt_secure(OLD_REFRESH))
    assert result.success is False
    monkeypatch.setattr(oo, "latch_on", real_latch_on)  # reads latch; inspect old state
    _assert_old_state_only(auth)
    monkeypatch.setattr(oo, "latch_on", _boom)
    with pytest.raises(oo.OptimizationOnlyRefusal):
        asyncio.run(auth.get_auth_headers())


def test_jwt_refresh_on_unflagged_manager_ignores_latch_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    auth = _manager()
    _refresh_ready(auth, monkeypatch)
    monkeypatch.setattr(oo, "latch_on", _boom)
    result = asyncio.run(auth._token_manager.refresh_jwt_secure(OLD_REFRESH))
    assert result.success is True
    assert auth._credentials.jwt_token == NEW_JWT
    assert auth._token_manager.get_authorization_header() == {
        "Authorization": f"Bearer {NEW_JWT}"
    }


def test_oauth_refresh_on_flagged_manager_that_cannot_latch_keeps_old_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    auth = _flagged_manager()
    _refresh_ready(auth, monkeypatch)
    old_object = auth._credentials
    monkeypatch.setattr(oo, "latch_on", _boom)
    with pytest.raises(oo.OptimizationOnlyRefusal):
        asyncio.run(
            auth._token_manager._apply_oauth2_token_response(
                old_object,
                {
                    "access_token": NEW_JWT,
                    "refresh_token": "refresh-new-0123456789",
                    "expires_in": 60,
                },
            )
        )
    assert auth._credentials is old_object
    assert old_object.jwt_token == OLD_JWT
    assert old_object.refresh_token == OLD_REFRESH
    with pytest.raises(oo.OptimizationOnlyRefusal):
        asyncio.run(auth.get_auth_headers())


def test_oauth_refresh_on_unflagged_manager_ignores_latch_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    auth = _manager()
    _refresh_ready(auth, monkeypatch)
    monkeypatch.setattr(oo, "latch_on", _boom)
    result = asyncio.run(
        auth._token_manager._apply_oauth2_token_response(
            auth._credentials, {"access_token": NEW_JWT, "expires_in": 60}
        )
    )
    assert result.success is True
    assert auth._credentials.jwt_token == NEW_JWT
    assert auth._credentials.refresh_token == OLD_REFRESH


# ---- each credential is latched on its own, and every read latches ----------------------


def test_setter_latches_each_credential_header_separately() -> None:
    from traigent.cloud.auth import AuthCredentials, AuthMode, AuthStatus

    auth = _flagged_manager()
    oo.reset_latch()
    auth._set_credentials_and_status(
        AuthCredentials(mode=AuthMode.JWT_TOKEN, jwt_token=NEW_JWT, api_key=OTHER_KEY),
        AuthStatus.AUTHENTICATED,
    )
    assert oo.is_latched(_jwt_scope(NEW_JWT))
    assert oo.is_latched(_scope(OTHER_KEY))


def _getters(auth: Any) -> list[Any]:
    tm = auth._token_manager
    return [
        tm.get_authorization_header,
        lambda: tm.current_token,
        auth._get_api_key_for_internal_use,
        lambda: auth._api_key_token,
        auth.configured_credential_headers,
    ]


def _stage_unlatched_login(auth: Any) -> dict[str, Any]:
    """What password login / JWT construction does before the setter: stores into secure
    storage (JWT) and the API-key manager."""
    data = {
        "access_token": NEW_JWT,
        "refresh_token": "refresh-new-0123456789",
        "expires_in": 3600,
        "api_key": OTHER_KEY,
    }
    auth._build_credentials_from_token_data(data)  # stores the API key
    auth._store_secure_tokens(data)  # stores the JWT
    return data


def test_flagged_manager_never_hands_out_a_credential_it_could_not_latch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    auth = _flagged_manager()
    monkeypatch.setattr(oo, "latch_on", _boom)
    _stage_unlatched_login(auth)
    for getter in _getters(auth):
        with pytest.raises(oo.OptimizationOnlyRefusal):
            getter()


def test_flagged_manager_reads_latch_what_they_return() -> None:
    auth = _flagged_manager()
    _stage_unlatched_login(auth)
    oo.reset_latch()
    assert auth._token_manager.get_authorization_header() == {
        "Authorization": f"Bearer {NEW_JWT}"
    }
    assert oo.is_latched(_jwt_scope(NEW_JWT))
    oo.reset_latch()
    assert auth._get_api_key_for_internal_use() == OTHER_KEY
    assert oo.is_latched(_scope(OTHER_KEY))


def test_unflagged_manager_getters_are_unchanged_when_latch_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    auth = _manager()
    monkeypatch.setattr(oo, "latch_on", _boom)
    monkeypatch.setattr(oo, "credential_scope", _boom)
    _stage_unlatched_login(auth)
    assert auth._token_manager.get_authorization_header() == {
        "Authorization": f"Bearer {NEW_JWT}"
    }
    assert auth._get_api_key_for_internal_use() == OTHER_KEY
    for getter in _getters(auth):
        getter()


def test_every_api_key_return_path_is_guarded() -> None:
    from traigent.cloud.auth import AuthCredentials, AuthMode, AuthResult, AuthStatus

    key = OTHER_KEY
    creds = AuthCredentials(mode=AuthMode.API_KEY, api_key=key)

    def refuse(_headers: Any) -> None:
        raise oo.OptimizationOnlyRefusal("test")

    # Each case leaves exactly one source able to answer.
    cases = ("secure_token", "credentials", "provided", "last_result")
    for case in cases:
        auth = _manager()
        mgr = auth._api_key_manager
        mgr._api_key_token = None
        auth._credentials = None
        auth._provided_credentials = None
        auth._last_auth_result = None
        if case == "secure_token":
            mgr.set_token(key, source="explicit")
        elif case == "credentials":
            auth._credentials = creds
        elif case == "provided":
            auth._provided_credentials = creds
        else:
            auth._last_auth_result = AuthResult(
                success=True, status=AuthStatus.AUTHENTICATED, credentials=creds
            )
        assert mgr.get_key_for_internal_use() == key, case  # unflagged: unchanged
        mgr.read_guard = refuse
        mgr.guard_active = lambda: True
        with pytest.raises(oo.OptimizationOnlyRefusal):
            mgr.get_key_for_internal_use()
        mgr.guard_active = lambda: False  # inactive guard: unchanged
        assert mgr.get_key_for_internal_use() == key, case


def test_refresh_token_getter_is_guarded_on_flagged_manager(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    auth = _flagged_manager()
    auth._token_manager.store_tokens(
        {"access_token": NEW_JWT, "refresh_token": "refresh-new-0123456789"}
    )
    monkeypatch.setattr(oo, "latch_on", _boom)
    with pytest.raises(oo.OptimizationOnlyRefusal):
        _ = auth._token_manager.refresh_token_secure
