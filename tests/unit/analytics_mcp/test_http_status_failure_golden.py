"""Golden characterization tests for ``_http_status_failure``."""

from __future__ import annotations

from typing import Any

import pytest

from traigent.analytics_mcp.tools import _http_status_failure


class _Resp:
    def __init__(self, status: Any, payload: Any = None, raises: bool = False) -> None:
        self.status_code = status
        self._payload = payload
        self._raises = raises
        self.json_calls = 0

    def json(self) -> Any:
        self.json_calls += 1
        if self._raises:
            raise ValueError("bad json")
        return self._payload


class _Exc(Exception):
    def __init__(self, response: Any = None) -> None:
        super().__init__("x")
        self.response = response


def _call(exc: Exception) -> dict[str, Any]:
    return _http_status_failure(exc, what="run", backend_url="https://b.example")


def _expected(message: str, code: str, status: Any) -> dict[str, Any]:
    return {
        "ok": False,
        "code": code,
        "message": message,
        "http_status": status,
        "backend_url": "https://b.example",
    }


def test_401_and_403_do_not_read_body() -> None:
    r401, r403 = _Resp(401, {"error": {}}), _Resp(403)
    assert _call(_Exc(r401)) == _expected(
        "Analytics authentication failed; configure valid Traigent credentials.",
        "authentication_failed",
        401,
    )
    assert _call(_Exc(r403)) == _expected(
        "Analytics request was forbidden by the backend; check project permissions "
        "or whether a WAF/Cloudflare rule is blocking the request.",
        "forbidden",
        403,
    )
    assert r401.json_calls == 0 and r403.json_calls == 0


@pytest.mark.parametrize(
    "payload",
    [
        {"detail": "Analytics not ready"},
        {"a": [{"b": "Not_Yet-Available"}]},
        ["compute to trigger"],
    ],
)
def test_404_not_ready(payload: Any) -> None:
    r = _Resp(404, payload)
    assert _call(_Exc(r)) == _expected(
        "The backend found the run, but run analytics are not ready yet.",
        "not_ready",
        404,
    )
    assert r.json_calls == 1


@pytest.mark.parametrize(
    "resp", [_Resp(404, {"detail": "nope"}), _Resp(404, raises=True)]
)
def test_404_not_found(resp: _Resp) -> None:
    assert _call(_Exc(resp)) == _expected(
        "The requested run was not found on the configured backend.",
        "not_found",
        404,
    )
    assert resp.json_calls == 1


def test_409_recognized_code_message_and_revision() -> None:
    r = _Resp(
        409,
        {"error": {"code": "stale_revision", "message": "old", "current_revision": 7}},
    )
    out = _call(_Exc(r))
    assert out == {**_expected("old", "stale_revision", 409), "current_revision": 7}
    assert r.json_calls == 1


def test_409_unrecognized_code_and_defaults() -> None:
    out = _call(_Exc(_Resp(409, {"error": {"code": "weird", "current_revision": "7"}})))
    assert out == _expected(
        "The run request conflicts with the session's current state.", "conflict", 409
    )
    assert "current_revision" not in out
    out = _call(_Exc(_Resp(409, None)))
    assert out["code"] == "conflict"
    out = _call(_Exc(_Resp(409, {"error": "str"})))
    assert out["code"] == "conflict" and "current_revision" not in out


def test_409_bool_revision_is_int_and_kept() -> None:
    out = _call(
        _Exc(
            _Resp(409, {"error": {"code": "turn_in_flight", "current_revision": True}})
        )
    )
    assert out["code"] == "turn_in_flight"
    assert out["current_revision"] is True


def test_400_message_or_default() -> None:
    assert _call(_Exc(_Resp(400, {"error": {"message": "bad field"}}))) == _expected(
        "bad field", "validation_error", 400
    )
    assert _call(_Exc(_Resp(400, {"error": {"message": ""}}))) == _expected(
        "The run request was rejected as invalid.", "validation_error", 400
    )
    assert _call(_Exc(_Resp(400, raises=True)))["message"] == (
        "The run request was rejected as invalid."
    )


@pytest.mark.parametrize(
    "exc",
    [_Exc(None), _Exc(_Resp(500)), _Exc(_Resp("404")), _Exc(_Resp(None))],
)
def test_other_statuses_are_backend_unavailable(exc: Exception) -> None:
    status = getattr(exc.response, "status_code", None)  # type: ignore[attr-defined]
    out = _call(exc)
    assert out == _expected(
        "Could not retrieve run from the backend.",
        "backend_unavailable",
        status if isinstance(status, int) else None,
    )
