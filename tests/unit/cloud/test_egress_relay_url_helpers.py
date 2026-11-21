"""The relay rule's small predicates in ``traigent.cloud.url_security``.

``test_egress_relay_url_gate.py`` drives the public validators; these pin the
predicates they are built from, so a refusal reason cannot move unnoticed.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from unittest.mock import patch

import pytest

from traigent.cloud.url_security import (
    _egress_relay_decision,
    _egress_relay_origin_covers,
    _egress_relay_origin_matches,
    _is_loopback_ip_literal,
    private_backend_opt_in_covers,
)

_PORT = 54321
_V4 = f"http://127.0.0.1:{_PORT}"


@contextmanager
def _env(**extra: str) -> Iterator[None]:
    with patch.dict("os.environ", extra, clear=True):
        yield


def _relay_env(**extra: str) -> dict[str, str]:
    return {"TRAIGENT_EGRESS_RELAY": "1", "TRAIGENT_BACKEND_URL": _V4, **extra}


class TestLoopbackIpLiteral:
    @pytest.mark.parametrize("host", ["127.0.0.1", "127.9.9.9", "::1", "[::1]"])
    def test_loopback_literals_are_accepted(self, host: str) -> None:
        assert _is_loopback_ip_literal(host) is True

    @pytest.mark.parametrize(
        "host",
        [
            None,
            "",
            "localhost",  # a name is never resolved
            "128.0.0.1",
            "10.0.0.1",
            "::2",
            "::ffff:127.0.0.1",  # an embedded form is not unwrapped
            "127.0.0.1.evil.example",
        ],
    )
    def test_everything_else_is_refused(self, host: str | None) -> None:
        assert _is_loopback_ip_literal(host) is False


class TestRelayOriginMatches:
    @pytest.mark.parametrize(
        "candidate",
        [
            f"{_V4}?x=1",  # a query
            f"{_V4}#frag",  # a fragment
            f"{_V4};p=1",  # params
            f"http://user@127.0.0.1:{_PORT}",  # userinfo
            f"http://:pw@127.0.0.1:{_PORT}",  # empty user with a password
            f"http://@127.0.0.1:{_PORT}",  # empty userinfo is still userinfo
            "http://127.0.0.1:54321\\x",  # a backslash
            "http://127.0.0.1",  # no explicit port
            "http://127.0.0.1:80",  # the scheme default port
            "http://127.0.0.1:0",
            "http://127.0.0.1:99999",  # out of range
            f"https://127.0.0.1:{_PORT}",  # only plain http is the relay's
            f"http://localhost:{_PORT}",  # a name, not a literal
            f"http://127.0.0.1:{_PORT + 1}",  # not the pinned port
        ],
    )
    def test_refused_even_when_the_switch_is_on_and_the_origin_pinned(
        self, candidate: str
    ) -> None:
        with _env(**_relay_env()):
            assert _egress_relay_origin_matches(candidate) is False

    def test_the_pinned_loopback_origin_matches_with_the_switch_on(self) -> None:
        with _env(**_relay_env()):
            assert _egress_relay_origin_matches(_V4) is True

    def test_the_same_origin_does_not_match_with_the_switch_off(self) -> None:
        with _env(TRAIGENT_BACKEND_URL=_V4):
            assert _egress_relay_origin_matches(_V4) is False

    def test_covers_recognizes_the_pin_even_with_a_path(self) -> None:
        with _env(**_relay_env()):
            assert _egress_relay_origin_covers(f"{_V4}/api/v1/x") is True
            assert _egress_relay_origin_covers("https://127.0.0.1:54321") is False
            assert _egress_relay_origin_covers("http://example.com:54321") is False


class TestRelayDecision:
    def test_an_origin_the_rule_does_not_cover_is_not_admitted(self) -> None:
        with _env(**_relay_env()):
            assert _egress_relay_decision("http://127.0.0.1:1") is False

    def test_a_covered_origin_is_admitted(self) -> None:
        with _env(**_relay_env()):
            assert _egress_relay_decision(_V4) is True

    def test_a_covered_origin_with_the_gate_off_raises_the_fixed_message(self) -> None:
        with _env(**_relay_env(TRAIGENT_TRANSPORT_GATE="off")):
            with pytest.raises(ValueError) as err:
                _egress_relay_decision(_V4)
        assert str(err.value) == (
            "TRAIGENT_EGRESS_RELAY requires the transport gate; "
            "unset TRAIGENT_TRANSPORT_GATE=off"
        )

    def test_an_uncovered_origin_with_the_gate_off_is_just_not_admitted(self) -> None:
        with _env(**_relay_env(TRAIGENT_TRANSPORT_GATE="off")):
            assert _egress_relay_decision("http://127.0.0.1:1") is False


class TestPrivateBackendOptInCovers:
    def test_the_self_hosted_opt_in_covers_exactly_the_configured_origin(self) -> None:
        env = {
            "TRAIGENT_BACKEND_URL": "https://10.20.30.40:8443",
            "TRAIGENT_ALLOW_PRIVATE_BACKEND": "1",
        }
        with _env(**env):
            assert private_backend_opt_in_covers("https://10.20.30.40:8443") is True
            assert private_backend_opt_in_covers("https://10.20.30.41:8443") is False

    def test_the_relay_rule_covers_its_pinned_origin(self) -> None:
        with _env(**_relay_env()):
            assert private_backend_opt_in_covers(_V4) is True

    def test_the_relay_rule_covers_nothing_when_the_gate_is_off(self) -> None:
        with _env(**_relay_env(TRAIGENT_TRANSPORT_GATE="off")):
            assert private_backend_opt_in_covers(_V4) is False

    def test_nothing_is_covered_without_either_switch(self) -> None:
        with _env(TRAIGENT_BACKEND_URL=_V4):
            assert private_backend_opt_in_covers(_V4) is False
