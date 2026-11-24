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


def test_unexpected_error_propagates_even_when_cut_and_log_fails() -> None:
    """Regression for S1143: the old ``return`` in finally swallowed this."""
    h = _handler(cut=True, log_error=True)
    conn = h.server.connect.return_value
    conn.connect.side_effect = RuntimeError("boom")
    with pytest.raises(RuntimeError, match="boom"):
        h._forward(1, MagicMock(), 0)  # noqa: SLF001
    conn.close.assert_called_once()
    h.server.report_log_incomplete.assert_not_called()


# --- real handshake tests -------------------------------------------------

x509 = pytest.importorskip("cryptography.x509")


def _make_ca_and_leaf(tmp_path):
    import datetime
    import ipaddress

    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import NameOID

    now = datetime.datetime.now(datetime.UTC)

    def name(cn: str):
        return x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, cn)])

    ca_key = ec.generate_private_key(ec.SECP256R1())
    ca = (
        x509.CertificateBuilder()
        .subject_name(name("test-ca"))
        .issuer_name(name("test-ca"))
        .public_key(ca_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(minutes=1))
        .not_valid_after(now + datetime.timedelta(hours=1))
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), True)
        .add_extension(
            x509.KeyUsage(
                digital_signature=True,
                content_commitment=False,
                key_encipherment=False,
                data_encipherment=False,
                key_agreement=False,
                key_cert_sign=True,
                crl_sign=True,
                encipher_only=False,
                decipher_only=False,
            ),
            True,
        )
        .add_extension(
            x509.SubjectKeyIdentifier.from_public_key(ca_key.public_key()), False
        )
        .sign(ca_key, hashes.SHA256())
    )
    leaf_key = ec.generate_private_key(ec.SECP256R1())
    leaf = (
        x509.CertificateBuilder()
        .subject_name(name("localhost"))
        .issuer_name(ca.subject)
        .public_key(leaf_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(minutes=1))
        .not_valid_after(now + datetime.timedelta(hours=1))
        .add_extension(
            x509.AuthorityKeyIdentifier.from_issuer_public_key(ca_key.public_key()),
            False,
        )
        .add_extension(
            x509.SubjectAlternativeName(
                [
                    x509.DNSName("localhost"),
                    x509.IPAddress(ipaddress.ip_address("127.0.0.1")),
                ]
            ),
            False,
        )
        .sign(ca_key, hashes.SHA256())
    )
    pem = serialization.Encoding.PEM
    ca_path = tmp_path / "ca.pem"
    ca_path.write_bytes(ca.public_bytes(pem))
    cert_path = tmp_path / "leaf.pem"
    cert_path.write_bytes(leaf.public_bytes(pem))
    key_path = tmp_path / "leaf.key"
    key_path.write_bytes(
        leaf_key.private_bytes(
            pem,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    return ca_path, cert_path, key_path


@pytest.fixture
def tls_server(tmp_path):
    import socket
    import threading

    ca_path, cert_path, key_path = _make_ca_and_leaf(tmp_path)
    server_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    server_ctx.load_cert_chain(cert_path, key_path)
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    sock.listen(8)
    sock.settimeout(0.2)
    stop = threading.Event()

    def serve() -> None:
        while not stop.is_set():
            try:
                conn, _ = sock.accept()
            except (TimeoutError, OSError):
                continue
            try:
                with server_ctx.wrap_socket(conn, server_side=True):
                    pass
            except (ssl.SSLError, OSError):
                conn.close()

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    yield sock.getsockname()[1], ca_path
    stop.set()
    thread.join(timeout=2)
    sock.close()


def _handshake(ctx: ssl.SSLContext, port: int, hostname: str) -> None:
    import socket

    with socket.create_connection(("127.0.0.1", port), timeout=5) as raw:
        with ctx.wrap_socket(raw, server_hostname=hostname):
            pass


def test_handshake_rejects_untrusted_certificate(tls_server) -> None:
    port, _ = tls_server
    with pytest.raises(ssl.SSLCertVerificationError):
        _handshake(relay_module._upstream_tls_context(), port, "localhost")


def test_handshake_rejects_hostname_mismatch(tls_server) -> None:
    port, ca_path = tls_server
    ctx = relay_module._upstream_tls_context()
    ctx.load_verify_locations(cafile=str(ca_path))
    with pytest.raises(ssl.SSLCertVerificationError):
        _handshake(ctx, port, "other.example")


def test_handshake_accepts_trusted_certificate(tls_server) -> None:
    port, ca_path = tls_server
    ctx = relay_module._upstream_tls_context()
    ctx.load_verify_locations(cafile=str(ca_path))
    _handshake(ctx, port, "localhost")
