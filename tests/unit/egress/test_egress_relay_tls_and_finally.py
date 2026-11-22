"""Relay upstream TLS verification is always on; the finally block never swallows."""

from __future__ import annotations

import ssl
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from traigent.egress import relay as relay_module
from traigent.egress.log import RelayLogError


def test_upstream_tls_context_verifies_certificate_and_hostname() -> None:
    ctx = relay_module._upstream_tls_context()
    assert ctx.verify_mode == ssl.CERT_REQUIRED
    assert ctx.check_hostname is True
    assert ctx.minimum_version >= ssl.TLSVersion.TLSv1_2


def test_https_connection_uses_verifying_context() -> None:
    server = SimpleNamespace(upstream=("https", "example.com", 443))
    conn = relay_module._Server.connect(server)  # type: ignore[arg-type]
    ctx = conn._context  # noqa: SLF001
    assert ctx.verify_mode == ssl.CERT_REQUIRED
    assert ctx.check_hostname is True


def _handler(*, cut: bool, log_error: bool):
    h = relay_module._Handler.__new__(relay_module._Handler)
    h.cut = cut
    h.command = "GET"
    h.headers = {}
    h.server = MagicMock()
    h._reply = MagicMock()  # noqa: SLF001
    h._forward_headers = MagicMock(return_value=[])  # noqa: SLF001
    if log_error:
        h.server.log.log_response.side_effect = RelayLogError("x")
    return h


@pytest.mark.parametrize(("cut", "reported"), [(False, 1), (True, 0)])
def test_log_failure_reported_unless_cut(cut: bool, reported: int) -> None:
    h = _handler(cut=cut, log_error=True)
    h.server.connect.return_value.connect.side_effect = OSError("down")
    h._forward(1, MagicMock(), 0)  # noqa: SLF001
    assert h.server.report_log_incomplete.call_count == reported


def test_exception_in_try_block_propagates_through_finally() -> None:
    h = _handler(cut=False, log_error=False)
    conn = h.server.connect.return_value
    conn.connect.side_effect = RuntimeError("boom")  # not an expected network error
    with pytest.raises(RuntimeError, match="boom"):
        h._forward(1, MagicMock(), 0)  # noqa: SLF001
    conn.close.assert_called_once()
    h.server.log.log_response.assert_called_once()
