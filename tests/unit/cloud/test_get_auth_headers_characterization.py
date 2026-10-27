"""Characterization tests pinning AuthManager.get_auth_headers.

Every collaborator is replaced by a recorder writing into one shared event
trace, so the order, number and arguments of calls (and reads of the
credentials attribute) are pinned as well as the returned headers.
"""

from __future__ import annotations

from typing import Any

import pytest

from traigent.cloud.auth import (
    AuthCredentials,
    AuthManager,
    AuthMode,
    AuthResult,
    AuthStatus,
    InvalidCredentialsError,
    _InvalidCredentialsWithSessionFailure,
)
from traigent.cloud.session_types import SessionCreationFailureDetail

DEFAULT_MESSAGE = "Authentication failed; refusing to emit auth headers."
INVALID_STATE_MESSAGE = (
    "Authentication state is invalid; refusing to emit auth headers."
)


class _TracedManager(AuthManager):
    """AuthManager recording every read of the credentials attribute."""

    def __getattribute__(self, name: str) -> Any:
        value = super().__getattribute__(name)
        if name == "_credentials":
            trace = self.__dict__.get("_trace")
            if trace is not None:
                trace.append(("read_credentials",))
        return value


def _manager(trace: list[Any]) -> AuthManager:
    manager = _TracedManager(api_key="tg_" + "x" * 61)
    manager.__dict__["_trace"] = trace
    return manager


def _install(
    monkeypatch: pytest.MonkeyPatch,
    manager: AuthManager,
    trace: list[Any],
    *,
    auth_state: Any = True,
    auth_result: Any = None,
    credentials_after_auth: Any = None,
) -> None:
    """Replace the collaborators of ``get_auth_headers`` with recorders."""

    def is_authenticated() -> Any:
        trace.append(("is_authenticated",))
        return auth_state

    async def authenticate() -> Any:
        trace.append(("authenticate",))
        if credentials_after_auth is not None:
            manager._credentials = credentials_after_auth
        return auth_result

    def api_key_headers() -> dict[str, str]:
        trace.append(("api_key_headers",))
        return {"k": "api"}

    async def jwt_headers() -> dict[str, str]:
        trace.append(("jwt_headers",))
        return {"k": "jwt"}

    def oauth_headers() -> dict[str, str]:
        trace.append(("oauth_headers",))
        return {"k": "oauth"}

    def service_headers(target: str) -> dict[str, str]:
        trace.append(("service_headers", target))
        return {"k": "service"}

    def dev_headers() -> dict[str, str]:
        trace.append(("dev_headers",))
        return {"k": "dev"}

    def latch(headers: dict[str, str]) -> None:
        trace.append(("latch", dict(headers)))

    def common(headers: dict[str, str], target: str) -> None:
        trace.append(("common", dict(headers), target))
        headers["common"] = target

    monkeypatch.setattr(manager, "is_authenticated", is_authenticated)
    monkeypatch.setattr(manager, "authenticate", authenticate)
    monkeypatch.setattr(manager, "_get_api_key_headers", api_key_headers)
    monkeypatch.setattr(manager, "_get_jwt_headers", jwt_headers)
    monkeypatch.setattr(manager, "_get_oauth_headers", oauth_headers)
    monkeypatch.setattr(manager, "_get_service_headers", service_headers)
    monkeypatch.setattr(manager, "_get_dev_headers", dev_headers)
    monkeypatch.setattr(manager, "_latch_emitted_credential", latch)
    monkeypatch.setattr(manager, "_add_common_headers", common)


def _credentials(mode: Any = AuthMode.API_KEY) -> AuthCredentials:
    return AuthCredentials(mode=mode)


def _failed(
    message: str | None = None, detail: SessionCreationFailureDetail | None = None
) -> AuthResult:
    return AuthResult(
        success=False,
        status=AuthStatus.INVALID,
        error_message=message,
        session_creation_failure=detail,
    )


# ------------------------------------------------------------- header success


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mode", "builder", "builder_event"),
    [
        (AuthMode.API_KEY, "api", ("api_key_headers",)),
        (AuthMode.JWT_TOKEN, "jwt", ("jwt_headers",)),
        (AuthMode.OAUTH2, "oauth", ("oauth_headers",)),
        (AuthMode.SERVICE_TO_SERVICE, "service", ("service_headers", "backend")),
        (AuthMode.DEVELOPMENT, "dev", ("dev_headers",)),
    ],
)
async def test_each_mode_uses_its_builder_then_latches_then_adds_common(
    monkeypatch: pytest.MonkeyPatch,
    mode: AuthMode,
    builder: str,
    builder_event: tuple[str, ...],
) -> None:
    trace: list[Any] = []
    manager = _manager(trace)
    manager._credentials = _credentials(mode)
    trace.clear()
    _install(monkeypatch, manager, trace)
    headers = await manager.get_auth_headers("backend")
    assert headers == {"k": builder, "common": "backend"}
    assert trace == [
        ("is_authenticated",),
        ("read_credentials",),
        ("read_credentials",),
        builder_event,
        ("latch", {"k": builder}),
        ("common", {"k": builder}, "backend"),
    ]


@pytest.mark.asyncio
async def test_default_target_is_both(monkeypatch: pytest.MonkeyPatch) -> None:
    trace: list[Any] = []
    manager = _manager(trace)
    manager._credentials = _credentials(AuthMode.SERVICE_TO_SERVICE)
    trace.clear()
    _install(monkeypatch, manager, trace)
    headers = await manager.get_auth_headers()
    assert headers == {"k": "service", "common": "both"}
    assert ("service_headers", "both") in trace
    assert ("common", {"k": "service"}, "both") in trace


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mode", [AuthMode.CLOUD, AuthMode.EDGE_ANALYTICS, AuthMode.DEMO]
)
async def test_unlisted_modes_emit_only_common_headers(
    monkeypatch: pytest.MonkeyPatch, mode: AuthMode
) -> None:
    trace: list[Any] = []
    manager = _manager(trace)
    manager._credentials = _credentials(mode)
    trace.clear()
    _install(monkeypatch, manager, trace)
    headers = await manager.get_auth_headers("cloud")
    assert headers == {"common": "cloud"}
    assert trace == [
        ("is_authenticated",),
        ("read_credentials",),
        ("read_credentials",),
        ("latch", {}),
        ("common", {}, "cloud"),
    ]


@pytest.mark.asyncio
async def test_mode_comparison_order_and_form(monkeypatch: pytest.MonkeyPatch) -> None:
    trace: list[Any] = []
    manager = _manager(trace)

    class Mode:
        def __eq__(self, other: object) -> bool:
            trace.append(("eq", other))
            return False

        def __hash__(self) -> int:
            return 0

    creds = _credentials()
    creds.__dict__["mode"] = Mode()
    manager._credentials = creds
    trace.clear()
    _install(monkeypatch, manager, trace)
    headers = await manager.get_auth_headers()
    assert headers == {"common": "both"}
    assert [e for e in trace if e[0] == "eq"] == [
        ("eq", AuthMode.API_KEY),
        ("eq", AuthMode.JWT_TOKEN),
        ("eq", AuthMode.OAUTH2),
        ("eq", AuthMode.SERVICE_TO_SERVICE),
        ("eq", AuthMode.DEVELOPMENT),
    ]


@pytest.mark.asyncio
async def test_mode_comparison_stops_at_first_match(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace: list[Any] = []
    manager = _manager(trace)

    class Mode:
        def __eq__(self, other: object) -> bool:
            trace.append(("eq", other))
            return other is AuthMode.OAUTH2

        def __hash__(self) -> int:
            return 0

    creds = _credentials()
    creds.__dict__["mode"] = Mode()
    manager._credentials = creds
    trace.clear()
    _install(monkeypatch, manager, trace)
    assert await manager.get_auth_headers() == {"k": "oauth", "common": "both"}
    assert [e for e in trace if e[0] == "eq"] == [
        ("eq", AuthMode.API_KEY),
        ("eq", AuthMode.JWT_TOKEN),
        ("eq", AuthMode.OAUTH2),
    ]


@pytest.mark.asyncio
async def test_returned_headers_are_the_builder_object(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace: list[Any] = []
    manager = _manager(trace)
    manager._credentials = _credentials()
    _install(monkeypatch, manager, trace)
    built: dict[str, str] = {"seed": "1"}
    monkeypatch.setattr(manager, "_get_api_key_headers", lambda: built)
    result = await manager.get_auth_headers()
    assert result is built
    assert built == {"seed": "1", "common": "both"}


# ------------------------------------------------------ authenticated checks


class _CountingTruth:
    def __init__(self, trace: list[Any], value: bool) -> None:
        self.trace = trace
        self.value = value

    def __bool__(self) -> bool:
        self.trace.append(("auth_state_bool", self.value))
        return self.value


@pytest.mark.asyncio
async def test_auth_state_truth_is_evaluated_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace: list[Any] = []
    manager = _manager(trace)
    manager._credentials = _credentials()
    trace.clear()
    _install(monkeypatch, manager, trace, auth_state=_CountingTruth(trace, True))
    await manager.get_auth_headers()
    assert trace[:2] == [("is_authenticated",), ("auth_state_bool", True)]
    assert sum(1 for e in trace if e[0] == "auth_state_bool") == 1
    assert ("authenticate",) not in trace


@pytest.mark.asyncio
async def test_awaitable_auth_state_is_awaited_then_truth_tested(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace: list[Any] = []
    manager = _manager(trace)
    manager._credentials = _credentials()
    trace.clear()

    async def state() -> Any:
        trace.append(("state_awaited",))
        return _CountingTruth(trace, True)

    _install(monkeypatch, manager, trace, auth_state=state())
    headers = await manager.get_auth_headers()
    assert headers == {"k": "api", "common": "both"}
    assert trace[:3] == [
        ("is_authenticated",),
        ("state_awaited",),
        ("auth_state_bool", True),
    ]
    assert ("authenticate",) not in trace


@pytest.mark.asyncio
async def test_unauthenticated_state_authenticates_then_emits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace: list[Any] = []
    manager = _manager(trace)
    trace.clear()
    ok = AuthResult(success=True, status=AuthStatus.AUTHENTICATED)
    _install(
        monkeypatch,
        manager,
        trace,
        auth_state=False,
        auth_result=ok,
        credentials_after_auth=_credentials(AuthMode.DEVELOPMENT),
    )
    headers = await manager.get_auth_headers("cloud")
    assert headers == {"k": "dev", "common": "cloud"}
    assert trace == [
        ("is_authenticated",),
        ("authenticate",),
        ("read_credentials",),
        ("read_credentials",),
        ("read_credentials",),
        ("dev_headers",),
        ("latch", {"k": "dev"}),
        ("common", {"k": "dev"}, "cloud"),
    ]


@pytest.mark.asyncio
async def test_failed_auth_raises_plain_invalid_credentials_with_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace: list[Any] = []
    manager = _manager(trace)
    manager._authenticated = True
    trace.clear()
    _install(
        monkeypatch, manager, trace, auth_state=False, auth_result=_failed("rejected")
    )
    with pytest.raises(InvalidCredentialsError) as info:
        await manager.get_auth_headers()
    assert type(info.value) is InvalidCredentialsError
    assert str(info.value) == "rejected"
    assert manager._authenticated is False
    assert [e[0] for e in trace if e[0] not in ("read_credentials",)] == [
        "is_authenticated",
        "authenticate",
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("message", [None, ""])
async def test_failed_auth_without_message_uses_default_text(
    monkeypatch: pytest.MonkeyPatch, message: str | None
) -> None:
    trace: list[Any] = []
    manager = _manager(trace)
    _install(
        monkeypatch, manager, trace, auth_state=False, auth_result=_failed(message)
    )
    with pytest.raises(InvalidCredentialsError) as info:
        await manager.get_auth_headers()
    assert type(info.value) is InvalidCredentialsError
    assert str(info.value) == DEFAULT_MESSAGE


@pytest.mark.asyncio
async def test_failed_auth_with_session_detail_raises_the_carrying_subclass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace: list[Any] = []
    manager = _manager(trace)
    detail = SessionCreationFailureDetail(status_code=403, backend_url="http://b")
    _install(
        monkeypatch,
        manager,
        trace,
        auth_state=False,
        auth_result=_failed("denied", detail),
    )
    with pytest.raises(_InvalidCredentialsWithSessionFailure) as info:
        await manager.get_auth_headers()
    assert str(info.value) == "denied"
    assert info.value.session_creation_failure is detail
    assert manager._authenticated is False
    assert not any(e[0] in ("latch", "common") for e in trace)


@pytest.mark.asyncio
async def test_failed_auth_with_session_detail_and_no_message_uses_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace: list[Any] = []
    manager = _manager(trace)
    detail = SessionCreationFailureDetail(status_code=500)
    _install(
        monkeypatch, manager, trace, auth_state=False, auth_result=_failed("", detail)
    )
    with pytest.raises(_InvalidCredentialsWithSessionFailure) as info:
        await manager.get_auth_headers()
    assert str(info.value) == DEFAULT_MESSAGE
    assert info.value.session_creation_failure is detail


@pytest.mark.asyncio
async def test_successful_auth_without_credentials_still_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace: list[Any] = []
    manager = _manager(trace)
    manager._authenticated = True
    ok = AuthResult(success=True, status=AuthStatus.AUTHENTICATED)
    _install(monkeypatch, manager, trace, auth_state=False, auth_result=ok)
    with pytest.raises(InvalidCredentialsError) as info:
        await manager.get_auth_headers()
    assert type(info.value) is InvalidCredentialsError
    assert str(info.value) == DEFAULT_MESSAGE
    assert manager._authenticated is False
    assert not any(e[0] in ("latch", "common") for e in trace)


@pytest.mark.asyncio
async def test_failed_auth_does_not_read_credentials_when_success_is_false(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace: list[Any] = []
    manager = _manager(trace)
    trace.clear()
    _install(monkeypatch, manager, trace, auth_state=False, auth_result=_failed("x"))
    with pytest.raises(InvalidCredentialsError):
        await manager.get_auth_headers()
    assert ("read_credentials",) not in trace


@pytest.mark.asyncio
async def test_failed_auth_reads_result_fields_in_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace: list[Any] = []
    manager = _manager(trace)

    class Result:
        @property
        def success(self) -> bool:
            trace.append(("success",))
            return False

        @property
        def session_creation_failure(self) -> None:
            trace.append(("failure",))
            return None

        @property
        def error_message(self) -> str:
            trace.append(("message",))
            return "m"

    _install(monkeypatch, manager, trace, auth_state=False, auth_result=Result())
    trace.clear()
    with pytest.raises(InvalidCredentialsError):
        await manager.get_auth_headers()
    assert trace == [
        ("is_authenticated",),
        ("authenticate",),
        ("success",),
        ("failure",),
        ("message",),
    ]


@pytest.mark.asyncio
async def test_authenticated_state_without_credentials_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace: list[Any] = []
    manager = _manager(trace)
    manager._authenticated = True
    trace.clear()
    _install(monkeypatch, manager, trace, auth_state=True)
    with pytest.raises(InvalidCredentialsError) as info:
        await manager.get_auth_headers("backend")
    assert type(info.value) is InvalidCredentialsError
    assert str(info.value) == INVALID_STATE_MESSAGE
    assert manager._authenticated is False
    assert trace == [("is_authenticated",), ("read_credentials",)]


@pytest.mark.asyncio
async def test_falsy_credentials_object_fails_closed_after_one_truth_test(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace: list[Any] = []
    manager = _manager(trace)

    class Falsy:
        mode = AuthMode.API_KEY

        def __bool__(self) -> bool:
            trace.append(("creds_bool",))
            return False

    manager.__dict__["_credentials"] = Falsy()
    manager._authenticated = True
    trace.clear()
    _install(monkeypatch, manager, trace, auth_state=True)
    with pytest.raises(InvalidCredentialsError) as info:
        await manager.get_auth_headers()
    assert str(info.value) == INVALID_STATE_MESSAGE
    assert manager._authenticated is False
    assert trace == [("is_authenticated",), ("read_credentials",), ("creds_bool",)]


@pytest.mark.asyncio
async def test_credentials_truth_is_tested_once_when_truthy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace: list[Any] = []
    manager = _manager(trace)

    class Truthy:
        mode = AuthMode.API_KEY

        def __bool__(self) -> bool:
            trace.append(("creds_bool",))
            return True

    manager.__dict__["_credentials"] = Truthy()
    trace.clear()
    _install(monkeypatch, manager, trace, auth_state=True)
    await manager.get_auth_headers()
    assert trace.count(("creds_bool",)) == 1


# ------------------------------------------------------------ error boundaries


@pytest.mark.asyncio
async def test_is_authenticated_error_propagates_without_side_effects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace: list[Any] = []
    manager = _manager(trace)
    manager._authenticated = True
    _install(monkeypatch, manager, trace)

    def boom() -> Any:
        raise RuntimeError("state failed")

    monkeypatch.setattr(manager, "is_authenticated", boom)
    with pytest.raises(RuntimeError, match="state failed"):
        await manager.get_auth_headers()
    assert manager._authenticated is True


@pytest.mark.asyncio
async def test_awaitable_auth_state_error_propagates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace: list[Any] = []
    manager = _manager(trace)
    manager._authenticated = True

    async def state() -> Any:
        raise RuntimeError("awaited state failed")

    _install(monkeypatch, manager, trace, auth_state=state())
    with pytest.raises(RuntimeError, match="awaited state failed"):
        await manager.get_auth_headers()
    assert manager._authenticated is True


@pytest.mark.asyncio
async def test_auth_state_bool_error_propagates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace: list[Any] = []
    manager = _manager(trace)

    class Bad:
        def __bool__(self) -> bool:
            raise RuntimeError("bool failed")

    _install(monkeypatch, manager, trace, auth_state=Bad())
    with pytest.raises(RuntimeError, match="bool failed"):
        await manager.get_auth_headers()
    assert ("authenticate",) not in trace


@pytest.mark.asyncio
async def test_authenticate_error_propagates_and_leaves_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace: list[Any] = []
    manager = _manager(trace)
    manager._authenticated = True
    _install(monkeypatch, manager, trace, auth_state=False)

    async def boom() -> Any:
        raise RuntimeError("authenticate failed")

    monkeypatch.setattr(manager, "authenticate", boom)
    with pytest.raises(RuntimeError, match="authenticate failed"):
        await manager.get_auth_headers()
    assert manager._authenticated is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mode", "attr"),
    [
        (AuthMode.API_KEY, "_get_api_key_headers"),
        (AuthMode.OAUTH2, "_get_oauth_headers"),
        (AuthMode.DEVELOPMENT, "_get_dev_headers"),
    ],
)
async def test_sync_builder_error_propagates_before_latch(
    monkeypatch: pytest.MonkeyPatch, mode: AuthMode, attr: str
) -> None:
    trace: list[Any] = []
    manager = _manager(trace)
    manager._credentials = _credentials(mode)
    _install(monkeypatch, manager, trace)

    def boom() -> Any:
        raise RuntimeError("builder failed")

    monkeypatch.setattr(manager, attr, boom)
    with pytest.raises(RuntimeError, match="builder failed"):
        await manager.get_auth_headers()
    assert not any(e[0] in ("latch", "common") for e in trace)


@pytest.mark.asyncio
async def test_jwt_builder_error_propagates_before_latch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace: list[Any] = []
    manager = _manager(trace)
    manager._credentials = _credentials(AuthMode.JWT_TOKEN)
    _install(monkeypatch, manager, trace)

    async def boom() -> Any:
        raise RuntimeError("jwt failed")

    monkeypatch.setattr(manager, "_get_jwt_headers", boom)
    with pytest.raises(RuntimeError, match="jwt failed"):
        await manager.get_auth_headers()
    assert not any(e[0] in ("latch", "common") for e in trace)


@pytest.mark.asyncio
async def test_service_builder_error_propagates_before_latch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace: list[Any] = []
    manager = _manager(trace)
    manager._credentials = _credentials(AuthMode.SERVICE_TO_SERVICE)
    _install(monkeypatch, manager, trace)

    def boom(target: str) -> Any:
        raise RuntimeError("service failed")

    monkeypatch.setattr(manager, "_get_service_headers", boom)
    with pytest.raises(RuntimeError, match="service failed"):
        await manager.get_auth_headers()
    assert not any(e[0] in ("latch", "common") for e in trace)


@pytest.mark.asyncio
async def test_latch_error_propagates_before_common_headers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace: list[Any] = []
    manager = _manager(trace)
    manager._credentials = _credentials()
    _install(monkeypatch, manager, trace)

    def boom(headers: dict[str, str]) -> None:
        raise RuntimeError("latch failed")

    monkeypatch.setattr(manager, "_latch_emitted_credential", boom)
    with pytest.raises(RuntimeError, match="latch failed"):
        await manager.get_auth_headers()
    assert not any(e[0] == "common" for e in trace)


@pytest.mark.asyncio
async def test_common_headers_error_propagates(monkeypatch: pytest.MonkeyPatch) -> None:
    trace: list[Any] = []
    manager = _manager(trace)
    manager._credentials = _credentials()
    _install(monkeypatch, manager, trace)

    def boom(headers: dict[str, str], target: str) -> None:
        raise RuntimeError("common failed")

    monkeypatch.setattr(manager, "_add_common_headers", boom)
    with pytest.raises(RuntimeError, match="common failed"):
        await manager.get_auth_headers()


@pytest.mark.asyncio
async def test_credentials_mode_accessor_error_propagates_before_builders(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace: list[Any] = []
    manager = _manager(trace)

    class Creds:
        @property
        def mode(self) -> AuthMode:
            trace.append(("mode_read",))
            raise RuntimeError("mode failed")

    manager.__dict__["_credentials"] = Creds()
    trace.clear()
    _install(monkeypatch, manager, trace)
    with pytest.raises(RuntimeError, match="mode failed"):
        await manager.get_auth_headers()
    assert ("mode_read",) in trace
    assert trace.count(("mode_read",)) == 1
    assert not any(e[0] in ("api_key_headers", "latch", "common") for e in trace)
