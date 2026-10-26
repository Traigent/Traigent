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
