"""``TRAIGENT_EGRESS_RELAY``: one narrow plain-http rule for the local egress relay.

All cases run in production posture (no development marker) unless a test says
otherwise. ``patch.dict(..., clear=True)`` removes the suite's autouse development
marker for the duration of each case.
"""

from __future__ import annotations

import socket
from collections.abc import Iterator
from contextlib import contextmanager
from unittest.mock import patch

import pytest

from traigent.cloud.url_security import (
    private_backend_opt_in_covers,
    validate_always_blocked_cloud_url,
    validate_cloud_base_url,
)

pytestmark = pytest.mark.backend_online

_PORT = 54321
_V4 = f"http://127.0.0.1:{_PORT}"
_V6 = f"http://[::1]:{_PORT}"
_GATE_OFF_MESSAGE = (
    "TRAIGENT_EGRESS_RELAY requires the transport gate; "
    "unset TRAIGENT_TRANSPORT_GATE=off"
)


@contextmanager
def _env(**extra: str) -> Iterator[None]:
    with patch.dict("os.environ", extra, clear=True):
        yield


def _relay_env(backend: str = _V4, **extra: str) -> dict[str, str]:
    return {"TRAIGENT_EGRESS_RELAY": "1", "TRAIGENT_BACKEND_URL": backend, **extra}


_POSTURES = [
    pytest.param({}, id="no-env-marker"),
    pytest.param({"ENVIRONMENT": "production"}, id="production"),
    pytest.param({"ENVIRONMENT": "staging"}, id="staging"),
]
_VALIDATORS = [
    pytest.param(validate_cloud_base_url, id="validate_cloud_base_url"),
    pytest.param(validate_always_blocked_cloud_url, id="always_blocked"),
]


def _accepts(validator, url: str) -> None:
    result = validator(url)
    if validator is validate_cloud_base_url:
        assert result == url.rstrip("/")


@pytest.mark.parametrize("validator", _VALIDATORS)
@pytest.mark.parametrize("posture", _POSTURES)
@pytest.mark.parametrize(
    "url",
    [_V4, _V6, f"{_V4}/", "http://127.9.9.9:54321"],
    ids=["v4", "v6", "slash", "v4-127.9"],
)
def test_relay_origin_accepted_in_production_posture(validator, posture, url) -> None:
    pinned = url.rstrip("/")
    with _env(**_relay_env(pinned, **posture)):
        _accepts(validator, url)  # the validator strips a trailing slash


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "True"])
def test_switch_values_that_turn_it_on(value: str) -> None:
    with _env(**_relay_env(TRAIGENT_EGRESS_RELAY=value)):
        assert validate_cloud_base_url(_V4) == _V4


@pytest.mark.parametrize("value", [None, "", "0", "false", "yes", "2", "on", " 1"])
def test_switch_off_or_other_values_refuse(value: str | None) -> None:
    env = _relay_env()
    if value is None:
        del env["TRAIGENT_EGRESS_RELAY"]
    else:
        env["TRAIGENT_EGRESS_RELAY"] = value
    with _env(**env):
        with pytest.raises(ValueError, match="must use https in production"):
            validate_cloud_base_url(_V4)
        with pytest.raises(ValueError, match="must use https in production"):
            validate_always_blocked_cloud_url(_V4)


def test_refusal_message_names_the_switch_only_for_loopback_http() -> None:
    with _env():
        with pytest.raises(ValueError) as loop:
            validate_cloud_base_url(_V4)
        with pytest.raises(ValueError) as public:
            validate_cloud_base_url("http://api.example.com")
        with pytest.raises(ValueError) as loop_name:
            validate_cloud_base_url(f"http://localhost:{_PORT}")
    assert "TRAIGENT_EGRESS_RELAY=1" in str(loop.value)
    assert "TRAIGENT_EGRESS_RELAY" not in str(public.value)
    assert "TRAIGENT_EGRESS_RELAY" not in str(loop_name.value)


def _dns_must_not_run(*_a, **_k):
    raise AssertionError("DNS must not be used for the relay rule")


# (url, pinned backend, extra env) refused even with the switch on.
_REFUSED = [
    pytest.param(f"http://127.0.0.1:{_PORT + 1}", _V4, {}, id="different-port"),
    pytest.param(_V4, "", {}, id="backend-url-unset"),
    pytest.param(_V4, f"http://127.0.0.1:{_PORT + 1}", {}, id="pin-other-port"),
    pytest.param(
        f"http://localhost:{_PORT}",
        f"http://localhost:{_PORT}",
        {},
        id="localhost-pinned",
    ),
    pytest.param("http://127.0.0.1", "http://127.0.0.1", {}, id="no-port"),
    pytest.param(
        "http://127.0.0.1", "http://127.0.0.1:80", {}, id="no-port-pin-names-80"
    ),
    pytest.param(
        f"http://0.0.0.0:{_PORT}", f"http://0.0.0.0:{_PORT}", {}, id="unspecified"
    ),
    pytest.param(
        f"http://10.0.0.5:{_PORT}", f"http://10.0.0.5:{_PORT}", {}, id="private-10"
    ),
    pytest.param(
        f"http://192.168.1.5:{_PORT}",
        f"http://192.168.1.5:{_PORT}",
        {},
        id="private-192",
    ),
    pytest.param(
        f"http://169.254.169.254:{_PORT}",
        f"http://169.254.169.254:{_PORT}",
        {},
        id="metadata",
    ),
    pytest.param(
        f"http://[::ffff:127.0.0.1]:{_PORT}",
        f"http://[::ffff:127.0.0.1]:{_PORT}",
        {},
        id="ipv4-mapped-loopback",
    ),
    pytest.param(
        f"http://[::ffff:169.254.169.254]:{_PORT}",
        f"http://[::ffff:169.254.169.254]:{_PORT}",
        {},
        id="ipv4-mapped-metadata",
    ),
    pytest.param(f"http://127.1:{_PORT}", f"http://127.1:{_PORT}", {}, id="short-form"),
    pytest.param(
        f"http://api.example.com:{_PORT}",
        f"http://api.example.com:{_PORT}",
        {},
        id="public-host-http",
    ),
    pytest.param(
        f"http://relay.test:{_PORT}",
        f"http://relay.test:{_PORT}",
        {},
        id="name-resolving-to-loopback",
    ),
    pytest.param(
        f"http://user:pw@127.0.0.1:{_PORT}",  # pragma: allowlist secret
        _V4,
        {},
        id="userinfo",
    ),
    pytest.param(_V4, _V4, {"TRAIGENT_TRANSPORT_GATE": "off"}, id="transport-gate-off"),
]


@pytest.mark.parametrize("validator", _VALIDATORS)
@pytest.mark.parametrize(("url", "pinned", "extra"), _REFUSED)
def test_refused_even_with_the_switch_on(validator, url, pinned, extra) -> None:
    env = _relay_env(pinned, **extra)
    if not pinned:
        del env["TRAIGENT_BACKEND_URL"]
    loopback_answer = [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 0))]
    with _env(**env), patch("socket.getaddrinfo", return_value=loopback_answer):
        with pytest.raises(ValueError):
            validator(url)


@pytest.mark.parametrize("suffix", ["/../x", "/a b", "?q=1"])
def test_path_whitespace_and_query_checks_still_apply(suffix: str) -> None:
    with _env(**_relay_env()):
        with pytest.raises(ValueError):
            validate_cloud_base_url(f"{_V4}{suffix}")


def test_gate_off_refusal_uses_the_fixed_message() -> None:
    with _env(**_relay_env(TRAIGENT_TRANSPORT_GATE="off")):
        for validator in (validate_cloud_base_url, validate_always_blocked_cloud_url):
            with pytest.raises(ValueError) as excinfo:
                validator(_V4)
            assert str(excinfo.value) == _GATE_OFF_MESSAGE


def test_name_resolving_to_loopback_is_refused_without_dns() -> None:
    with _env(**_relay_env("http://relay.test:54321")):
        with patch("socket.getaddrinfo", _dns_must_not_run):
            with pytest.raises(ValueError, match="must use https in production"):
                validate_cloud_base_url("http://relay.test:54321")
            with pytest.raises(ValueError, match="must use https in production"):
                validate_always_blocked_cloud_url("http://relay.test:54321")


def test_api_url_pointing_elsewhere_is_not_the_pin() -> None:
    other = "http://127.0.0.1:60000"
    with _env(**_relay_env(_V4, TRAIGENT_API_URL=other)):
        with pytest.raises(ValueError, match="must use https in production"):
            validate_cloud_base_url(other)
        assert validate_cloud_base_url(_V4) == _V4


def test_api_url_alone_does_not_pin() -> None:
    with _env(TRAIGENT_EGRESS_RELAY="1", TRAIGENT_API_URL=_V4):
        with pytest.raises(ValueError, match="must use https in production"):
            validate_cloud_base_url(_V4)


def test_other_loopback_origin_stays_refused_while_the_relay_one_passes() -> None:
    with _env(**_relay_env()):
        assert validate_cloud_base_url(_V4) == _V4
        with pytest.raises(ValueError):
            validate_cloud_base_url(f"http://127.0.0.1:{_PORT + 1}")
        with pytest.raises(ValueError):
            validate_cloud_base_url(f"https://127.0.0.1:{_PORT + 1}")


def test_relay_switch_does_not_open_private_destinations_over_https() -> None:
    with _env(**_relay_env()):
        with pytest.raises(ValueError, match="private or loopback"):
            validate_cloud_base_url(f"https://127.0.0.1:{_PORT}")


@pytest.mark.parametrize("switch", ["1", "0", None])
def test_https_public_url_unaffected(switch: str | None) -> None:
    env = {"TRAIGENT_BACKEND_URL": _V4}
    if switch is not None:
        env["TRAIGENT_EGRESS_RELAY"] = switch
    answer = [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))]
    with _env(**env), patch("socket.getaddrinfo", return_value=answer):
        assert validate_cloud_base_url("https://api.example.com/") == (
            "https://api.example.com"
        )
        validate_always_blocked_cloud_url("https://api.example.com")


@pytest.mark.parametrize("switch", ["1", None])
@pytest.mark.parametrize(
    "url",
    [f"http://127.0.0.1:{_PORT}", "http://localhost:9999", "http://10.1.2.3:80"],
)
def test_dev_environment_identical_with_switch_on_or_off(switch, url) -> None:
    base = {"ENVIRONMENT": "development", "TRAIGENT_BACKEND_URL": _V4}
    with _env(**base):
        expected = validate_cloud_base_url(url)
    env = dict(base)
    if switch:
        env["TRAIGENT_EGRESS_RELAY"] = switch
    with _env(**env):
        assert validate_cloud_base_url(url) == expected


def test_dev_environment_still_blocks_metadata_with_switch_on() -> None:
    with _env(**_relay_env(ENVIRONMENT="development")):
        with pytest.raises(ValueError, match="metadata|link-local"):
            validate_cloud_base_url("http://169.254.169.254:80")


def test_private_backend_opt_in_unchanged_when_relay_switch_is_off() -> None:
    env = {
        "TRAIGENT_BACKEND_URL": "https://10.20.30.40:8443",
        "TRAIGENT_ALLOW_PRIVATE_BACKEND": "1",
    }
    with _env(**env):
        assert validate_cloud_base_url("https://10.20.30.40:8443") == (
            "https://10.20.30.40:8443"
        )
        # The self-hosted opt-in never admitted plain http outside development.
        with pytest.raises(ValueError, match="must use https in production"):
            validate_cloud_base_url("http://10.20.30.40:8443")


def test_private_backend_opt_in_alone_does_not_admit_relay_http() -> None:
    env = {"TRAIGENT_BACKEND_URL": _V4, "TRAIGENT_ALLOW_PRIVATE_BACKEND": "1"}
    with _env(**env):
        with pytest.raises(ValueError, match="must use https in production"):
            validate_cloud_base_url(_V4)


# --- redirect refusal helper ---------------------------------------------------


def test_redirect_helper_true_for_the_pinned_relay_origin() -> None:
    with _env(**_relay_env()):
        assert private_backend_opt_in_covers(_V4) is True
        assert private_backend_opt_in_covers(f"{_V4}/api/v1beta/projects") is True
        assert private_backend_opt_in_covers(f"http://127.0.0.1:{_PORT + 1}") is False
        assert private_backend_opt_in_covers("https://api.example.com") is False
    with _env(**_relay_env(TRAIGENT_EGRESS_RELAY="0")):
        assert private_backend_opt_in_covers(_V4) is False
    with _env(**_relay_env(TRAIGENT_TRANSPORT_GATE="off")):
        assert private_backend_opt_in_covers(_V4) is False


def test_transport_gate_refuses_redirects_for_the_relay_origin() -> None:
    from traigent.labels.transport_gate import _refuses_redirects

    client = object()
    with _env(**_relay_env()):
        assert _refuses_redirects(client, _V4) is True
        assert _refuses_redirects(client, f"http://127.0.0.1:{_PORT + 1}") is False
