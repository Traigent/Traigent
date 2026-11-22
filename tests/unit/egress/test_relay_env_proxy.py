"""An environment proxy must not be able to divert traffic meant for the relay.

``TRAIGENT_EGRESS_RELAY=1`` admits plain http to the pinned loopback origin. The SDK's
HTTP clients honour ``HTTP_PROXY``/``http_proxy``/``ALL_PROXY``, so unless ``NO_PROXY``
covers the relay host the request would go to the proxy instead. The rule refuses.
"""

from __future__ import annotations

import socket
import threading
from collections.abc import Iterator
from unittest.mock import patch

import pytest

from traigent.cloud.url_security import (
    validate_always_blocked_cloud_url,
    validate_cloud_base_url,
)

pytestmark = pytest.mark.backend_online

_PORT = 54321
_V4 = f"http://127.0.0.1:{_PORT}"
_V6 = f"http://[::1]:{_PORT}"
_PROXY = "http://proxy.invalid:3128"
_MESSAGE = (
    "TRAIGENT_EGRESS_RELAY refused: an environment proxy "
    "(HTTP_PROXY, http_proxy or ALL_PROXY) would carry traffic meant for the relay; "
    "add the relay host to NO_PROXY (for an IPv6 relay address use 127.0.0.1 for "
    "the relay, or NO_PROXY=*)"
)
_VALIDATORS = [
    pytest.param(validate_cloud_base_url, id="validate_cloud_base_url"),
    pytest.param(validate_always_blocked_cloud_url, id="always_blocked"),
]


def _env(backend: str = _V4, **extra: str):
    base = {"TRAIGENT_EGRESS_RELAY": "1", "TRAIGENT_BACKEND_URL": backend, **extra}
    return patch.dict("os.environ", base, clear=True)


@pytest.mark.parametrize("validator", _VALIDATORS)
@pytest.mark.parametrize(
    "proxy_env",
    [
        {"HTTP_PROXY": _PROXY},
        {"http_proxy": _PROXY},
        {"ALL_PROXY": _PROXY},
        {"all_proxy": _PROXY},
        {"HTTP_PROXY": _PROXY, "NO_PROXY": "example.com"},
        {"HTTP_PROXY": _PROXY, "NO_PROXY": "localhost,10.0.0.0/8"},
        {"HTTP_PROXY": _PROXY, "NO_PROXY": f"127.0.0.1:{_PORT}"},  # not honoured
        {"HTTP_PROXY": _PROXY, "NO_PROXY": "127.0.0.1", "no_proxy": "example.com"},
        {"HTTP_PROXY": _PROXY, "NO_PROXY": "127.0.0.0/8"},
        {"HTTP_PROXY": _PROXY, "NO_PROXY": "a,*"},
        {"HTTP_PROXY": _PROXY, "NO_PROXY": "127.0.0.1", "no_proxy": ""},
        {"HTTP_PROXY": _PROXY, "NO_PROXY": "", "no_proxy": "127.0.0.1"},
    ],
)
@pytest.mark.parametrize("url", [_V4, _V6], ids=["v4", "v6"])
def test_proxy_not_bypassed_is_refused(validator, proxy_env, url) -> None:
    with _env(url, **proxy_env), pytest.raises(ValueError) as excinfo:
        validator(url)
    assert str(excinfo.value) == _MESSAGE
    assert _PROXY not in str(excinfo.value)


@pytest.mark.parametrize("validator", _VALIDATORS)
@pytest.mark.parametrize(
    ("url", "no_proxy"),
    [
        (_V4, "127.0.0.1"),
        (_V4, " example.com , 127.0.0.1 "),
        (_V4, "*"),
        (_V6, "*"),
        ("http://127.9.9.9:54321", "127.9.9.9"),
    ],
)
@pytest.mark.parametrize("var", ["NO_PROXY", "no_proxy"])
@pytest.mark.parametrize("proxy_var", ["HTTP_PROXY", "http_proxy", "ALL_PROXY"])
def test_proxy_bypassed_is_accepted(validator, url, no_proxy, var, proxy_var) -> None:
    with _env(url, **{proxy_var: _PROXY, var: no_proxy}):
        validator(url)


@pytest.mark.parametrize("validator", _VALIDATORS)
@pytest.mark.parametrize("url", [_V4, _V6])
def test_https_only_proxy_is_accepted(validator, url) -> None:
    # http traffic is never sent to an https_proxy by requests, httpx, aiohttp or urllib.
    with _env(url, HTTPS_PROXY=_PROXY, https_proxy=_PROXY):
        validator(url)


@pytest.mark.parametrize("validator", _VALIDATORS)
@pytest.mark.parametrize("token", ["::1", "[::1]", "::1,127.0.0.1", "*,::1"])
def test_ipv6_relay_host_needs_the_whole_value_star(validator, token) -> None:
    # urllib compares the bracketed host, requests/aiohttp the bare one: no token
    # form is safe in every library, so only the whole value "*" is accepted.
    with _env(_V6, HTTP_PROXY=_PROXY, NO_PROXY=token) as _:
        with pytest.raises(ValueError) as excinfo:
            validator(_V6)
    assert str(excinfo.value) == _MESSAGE


@pytest.mark.parametrize("validator", _VALIDATORS)
def test_ipv6_relay_host_with_no_proxy_configured_is_accepted(validator) -> None:
    with _env(_V6):
        validator(_V6)


@pytest.mark.parametrize("validator", _VALIDATORS)
def test_ipv6_star_must_hold_in_both_spellings(validator) -> None:
    with _env(_V6, HTTP_PROXY=_PROXY, NO_PROXY="*", no_proxy="x"):
        with pytest.raises(ValueError):
            validator(_V6)
    with _env(_V6, HTTP_PROXY=_PROXY, NO_PROXY="*", no_proxy="*"):
        validator(_V6)


@pytest.mark.parametrize("validator", _VALIDATORS)
def test_both_spellings_set_and_covering_are_accepted(validator) -> None:
    with _env(HTTP_PROXY=_PROXY, NO_PROXY="127.0.0.1", no_proxy="x, 127.0.0.1"):
        validator(_V4)


@pytest.mark.parametrize("validator", _VALIDATORS)
def test_empty_proxy_values_are_ignored(validator) -> None:
    with _env(HTTP_PROXY="", http_proxy="", ALL_PROXY=""):
        validator(_V4)


@pytest.mark.parametrize("validator", _VALIDATORS)
@pytest.mark.parametrize(
    "proxy_env",
    [{}, {"HTTP_PROXY": _PROXY}, {"ALL_PROXY": _PROXY, "NO_PROXY": "x"}],
)
def test_switch_off_is_unchanged_by_proxy_variables(validator, proxy_env) -> None:
    env = {"TRAIGENT_BACKEND_URL": _V4, **proxy_env}
    with patch.dict("os.environ", env, clear=True), pytest.raises(ValueError) as exc:
        validator(_V4)
    assert "must use https in production" in str(exc.value)
    assert "environment proxy" not in str(exc.value)


# --- end to end: a fake proxy must receive zero bytes -------------------------------


class _FakeProxy:
    def __init__(self) -> None:
        self.received = 0
        self._sock = socket.socket()
        self._sock.bind(("127.0.0.1", 0))
        self._sock.listen(8)
        self._sock.settimeout(0.2)
        self.port = self._sock.getsockname()[1]
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                conn, _ = self._sock.accept()
            except (TimeoutError, OSError):
                continue
            conn.settimeout(1.0)
            try:
                self.received += len(conn.recv(65536))
                conn.sendall(b"HTTP/1.1 502 Bad Gateway\r\nContent-Length: 0\r\n\r\n")
            except OSError:
                pass  # silent-ok: test double, a dropped connection is fine
            finally:
                conn.close()

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5)
        self._sock.close()


@pytest.fixture
def fake_proxy() -> Iterator[_FakeProxy]:
    proxy = _FakeProxy()
    yield proxy
    proxy.close()


def _production_posture(monkeypatch, relay, proxy: _FakeProxy) -> None:
    for name in (
        "TRAIGENT_ENVIRONMENT",
        "ENVIRONMENT",
        "TRAIGENT_ENV",
        "NO_PROXY",
        "no_proxy",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("TRAIGENT_EGRESS_RELAY", "1")
    monkeypatch.setenv("TRAIGENT_OFFLINE_MODE", "false")
    monkeypatch.setenv("TRAIGENT_BACKEND_URL", relay.origin)
    monkeypatch.setenv("TRAIGENT_API_KEY", "tg_canary_key_value")
    monkeypatch.setenv("HTTP_PROXY", f"http://127.0.0.1:{proxy.port}")
    monkeypatch.setenv("http_proxy", f"http://127.0.0.1:{proxy.port}")


def test_sdk_client_refuses_and_proxy_gets_zero_bytes(
    relay, upstream, fake_proxy, monkeypatch
) -> None:
    _production_posture(monkeypatch, relay, fake_proxy)
    from traigent.projects.client import ProjectManagementClient
    from traigent.projects.config import ProjectManagementConfig

    with pytest.raises(Exception) as excinfo:  # noqa: PT011 - any refusal type
        ProjectManagementClient(ProjectManagementConfig()).list_projects(search="alice")
    assert "NO_PROXY" in str(excinfo.value)
    assert fake_proxy.received == 0
    assert upstream.seen == []


def test_sdk_client_works_when_no_proxy_covers_the_relay(
    relay, upstream, fake_proxy, monkeypatch
) -> None:
    import json

    _production_posture(monkeypatch, relay, fake_proxy)
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    upstream.response_body = json.dumps(
        {"data": {"items": [], "total": 0, "page": 1, "per_page": 20}}
    ).encode()
    upstream.extra_headers = [("Content-Type", "application/json")]
    from traigent.projects.client import ProjectManagementClient
    from traigent.projects.config import ProjectManagementConfig

    ProjectManagementClient(ProjectManagementConfig()).list_projects(search="alice")
    assert fake_proxy.received == 0
    assert len(upstream.seen) == 1
