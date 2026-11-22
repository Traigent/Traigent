"""Loopback-only forwarding relay that logs what the SDK sends to the backend.

For each request: read the body (spooled, hashed as it is read), write the request
log line durably, and only then forward the identical bytes to the upstream origin.
The upstream response is returned unchanged (redirects are never followed) and a
response line is logged under the same ``seq``.

Fail closed: if the request line cannot be written the relay answers 502 and
forwards nothing. If the response line cannot be written after the exchange the
response is still delivered (the upstream side effect already happened), one fixed
error line goes to stderr, and the relay then answers 502 to every further request
because the log is no longer complete.

Standard library only. TLS to the upstream always verifies against the default
trust store; there is no option to turn that off. The relay itself speaks plain
HTTP and binds ``127.0.0.1`` only.
"""

# Traceability: CONC-Layer-Core CONC-Quality-Security FUNC-SECURITY

from __future__ import annotations

import hashlib
import http.client
import http.server
import io
import ipaddress
import socket
import socketserver
import ssl
import sys
import tempfile
import threading
from typing import IO, Any
from urllib.parse import urlsplit

from traigent.egress.log import RelayLog, RelayLogError

BIND_HOST = "127.0.0.1"
_CHUNK = 64 * 1024
_SPOOL_MEMORY = 1024 * 1024
_UPSTREAM_TIMEOUT_S = 60.0
#: stop() lets running exchanges finish for this long, then cuts their sockets.
_STOP_DRAIN_S = 3.0
_STOP_JOIN_S = 2.0
_MAX_INTERIM = 8

MSG_LOG_FAILED = b"egress relay: request not forwarded because it could not be logged\n"
MSG_UPSTREAM_FAILED = b"egress relay: upstream unreachable\n"
MSG_BAD_REQUEST = b"egress relay: malformed request\n"
MSG_UNSUPPORTED_ENCODING = (
    b"egress relay: only a single chunked Transfer-Encoding is supported\n"
)
MSG_UPGRADE_UNSUPPORTED = b"egress relay: protocol upgrades are not supported\n"
STDERR_LOG_INCOMPLETE = (
    "egress relay: the log can no longer be written; refusing further requests"
)

_HOP_BY_HOP = frozenset(
    {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "proxy-connection",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
    }
)


class RelayConfigError(ValueError):
    """The relay was asked to do something it refuses to do."""


def _is_loopback_literal(host: str) -> bool:
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def parse_upstream(url: str) -> tuple[str, str, int]:
    """Return ``(scheme, host, port)`` for an upstream origin, or raise.

    ``https`` is required. Plain ``http`` is accepted only for a loopback IP
    literal (for tests). Only an origin is accepted: no path, query or userinfo.
    """
    try:
        parts = urlsplit(url.strip())
        host = parts.hostname
        port = parts.port
    except ValueError:
        raise RelayConfigError("the upstream URL is not valid") from None
    if parts.scheme not in ("https", "http") or not host:
        raise RelayConfigError("the upstream must be an https:// origin")
    if parts.scheme == "http" and not _is_loopback_literal(host):
        raise RelayConfigError(
            "the upstream must be https:// (plain http is allowed only for a "
            "loopback IP literal)"
        )
    if parts.username or parts.password or parts.query or parts.fragment:
        raise RelayConfigError("the upstream must be a bare origin")
    if parts.path not in ("", "/"):
        raise RelayConfigError("the upstream must be a bare origin")
    return parts.scheme, host, port or (443 if parts.scheme == "https" else 80)


class _FinalResponse(http.client.HTTPResponse):
    """Skip every interim (1xx) response; a 101 is returned for the caller to refuse.

    ``http.client`` itself skips only ``100 Continue``; a ``103 Early Hints`` would
    otherwise be delivered as if it were the final answer.
    """

    def begin(self) -> None:
        for _ in range(_MAX_INTERIM):
            super().begin()
            if self.status == 101 or not 100 <= self.status < 200:
                return
            self.headers = None  # type: ignore[assignment]  # let begin() read on
        raise http.client.HTTPException("too many interim responses")


class _Stopped(Exception):
    """stop() began before this exchange could safely reach the upstream."""


class _BadBody(Exception):
    pass


def _read_chunk_size(rfile: io.BufferedIOBase) -> int:
    line = rfile.readline(1024)
    try:
        size = int(line.split(b";", 1)[0].strip(), 16)
    except ValueError:
        raise _BadBody from None
    if size < 0:
        raise _BadBody
    return size


def _read_chunked(rfile: io.BufferedIOBase, sink: Any, digest: Any) -> int:
    total = 0
    while True:
        size = _read_chunk_size(rfile)
        if size == 0:
            # trailers, then the blank line
            while rfile.readline(8192) not in (b"\r\n", b"\n", b""):
                pass
            return total
        total += _read_fixed(rfile, size, sink, digest)
        if rfile.readline(8) not in (b"\r\n", b"\n"):
            raise _BadBody


def _read_fixed(rfile: io.BufferedIOBase, length: int, sink: Any, digest: Any) -> int:
    remaining = length
    while remaining:
        data = rfile.read(min(_CHUNK, remaining))
        if not data:
            raise _BadBody
        sink.write(data)
        digest.update(data)
        remaining -= len(data)
    return length


class _Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    timeout = 30  # an idle or stalled client cannot pin a handler thread
    server: _Server

    cut = False  # set by stop() when this exchange is ended by force
    _upstream_sock: socket.socket | None = None

    def setup(self) -> None:
        super().setup()
        self.server.register(self)

    def finish(self) -> None:
        try:
            super().finish()
        finally:
            self.server.unregister(self)

    def cut_sockets(self) -> None:
        self.cut = True
        socks = [self.connection]
        socks.append(self._upstream_sock)
        for sock in socks:
            try:
                if sock is not None:
                    sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        return  # never write request lines (paths, queries) to stderr

    def _reply(self, status: int, message: bytes) -> None:
        self.close_connection = True
        try:
            self.send_response(status)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(message)))
            self.send_header("Connection", "close")
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(message)
        except OSError:
            pass  # the client is gone; there is nobody to tell

    def _forward_headers(self, length: int | None) -> list[tuple[str, str]]:
        named = {
            token.strip().lower()
            for value in self.headers.get_all("Connection", [])
            for token in value.split(",")
        }
        skip = _HOP_BY_HOP | named | {"host", "content-length", "expect"}
        headers = [(k, v) for k, v in self.headers.items() if k.lower() not in skip]
        if length is not None:
            headers.append(("Content-Length", str(length)))
        return headers

    def _handle(self) -> None:
        self.close_connection = True
        if not self.path.startswith("/"):
            self._reply(400, MSG_BAD_REQUEST)
            return
        encodings = self.headers.get_all("Transfer-Encoding", [])
        lengths = self.headers.get_all("Content-Length", [])
        if len(lengths) > 1 or (encodings and lengths):
            self._reply(400, MSG_BAD_REQUEST)  # smuggling hygiene: one framing only
            return
        if encodings and (
            len(encodings) != 1 or encodings[0].strip().lower() != "chunked"
        ):
            self._reply(501, MSG_UNSUPPORTED_ENCODING)  # a coding we would lose
            return
        digest = hashlib.sha256()
        with tempfile.SpooledTemporaryFile(max_size=_SPOOL_MEMORY) as body:
            try:
                if encodings:
                    size = _read_chunked(self.rfile, body, digest)
                else:
                    raw = self.headers.get("Content-Length")
                    length = int(raw) if raw is not None else 0
                    if length < 0:
                        raise _BadBody
                    size = _read_fixed(self.rfile, length, body, digest)
            except (_BadBody, ValueError, OSError):
                self._reply(400, MSG_BAD_REQUEST)
                return
            try:
                seq = self.server.log.log_request(
                    self.command, self.path, size, digest.hexdigest()
                )
            except RelayLogError:
                self._reply(502, MSG_LOG_FAILED)
                return
            body.seek(0)
            self._forward(seq, body, size)

    def _send_upstream_head(self, upstream: Any, status: int) -> None:
        """Send the status line and the non-hop-by-hop upstream headers."""
        self.send_response_only(status, upstream.reason)
        named = {
            token.strip().lower()
            for value in upstream.msg.get_all("Connection", [])
            for token in value.split(",")
        }
        skip = _HOP_BY_HOP | named
        for key, value in upstream.getheaders():
            if key.lower() not in skip:
                self.send_header(key, value)
        self.send_header("Connection", "close")
        self.end_headers()

    def _open_upstream(
        self,
        conn: http.client.HTTPConnection,
        headers: list[tuple[str, str]],
        body: IO[bytes],
        size: int,
    ) -> http.client.HTTPResponse:
        """Connect, send the request and return the upstream response."""
        if not self.server.may_connect(self):
            raise _Stopped
        conn.connect()
        # Registered before any header or body byte is sent, so stop() can
        # cut it even while a large upload is blocked. Kept apart from
        # conn.sock: http.client drops that once a close-delimited
        # response starts.
        if not self.server.attach_upstream(self, conn.sock):
            raise _Stopped
        conn.putrequest(self.command, self.path, skip_accept_encoding=True)
        for key, value in headers:
            conn.putheader(key, value)
        conn.endheaders(body if size > 0 else None)
        return conn.getresponse()

    def _forward(self, seq: int, body: IO[bytes], size: int) -> None:
        has_body = size > 0 or self.command in ("POST", "PUT", "PATCH")
        headers = self._forward_headers(size if has_body else None)
        status = 502
        count = 0
        outcome: str | None = "upstream_unreachable"
        conn = self.server.connect()
        try:
            try:
                upstream = self._open_upstream(conn, headers, body, size)
            except _Stopped:
                self.cut = True
                self._reply(502, MSG_UPSTREAM_FAILED)
                return
            except (OSError, http.client.HTTPException, ValueError):
                self._reply(502, MSG_UPSTREAM_FAILED)
                return
            status = upstream.status
            outcome = None
            if status == 101:
                outcome = "upstream_unsupported"
                status = 502
                self._reply(502, MSG_UPGRADE_UNSUPPORTED)
                return
            try:
                self._send_upstream_head(upstream, status)
                if _response_has_body(self.command, status):
                    while True:
                        data = upstream.read1(_CHUNK)
                        if not data:
                            break
                        self.wfile.write(data)
                        count += len(data)
                        self.wfile.flush()
            except (OSError, http.client.HTTPException):
                outcome = "upstream_error"
        finally:
            conn.close()
            if self.cut:
                outcome = "relay_stopped"
            self._log_response(seq, status, count, outcome)

    def _log_response(
        self, seq: int, status: int, count: int, outcome: str | None
    ) -> None:
        try:
            self.server.log.log_response(seq, status, count, outcome)
        except RelayLogError:
            # The request line is durable and the response is delivered; the log
            # is now incomplete, so the next request is refused (RelayLog.failed).
            # A cut exchange stays silent.
            if not self.cut:
                self.server.report_log_incomplete()


def _response_has_body(command: str, status: int) -> bool:
    return command != "HEAD" and status >= 200 and status not in (204, 304)


def _upstream_tls_context() -> ssl.SSLContext:
    """TLS context for the upstream hop: system CAs, certificate and hostname checks on.

    There is deliberately no insecure mode: verification cannot be turned off.
    """
    context = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)
    context.minimum_version = ssl.TLSVersion.TLSv1_2
    context.verify_mode = ssl.CERT_REQUIRED
    context.check_hostname = True
    return context


class _Server(socketserver.ThreadingMixIn, http.server.HTTPServer):
    # Handler threads are daemons and tracked by hand: stop() drains for a bounded
    # time, then cuts the sockets of what is left, so it can never hang on a
    # trickling client or an endless upstream stream.
    daemon_threads = True
    allow_reuse_address = False

    def __init__(self, port: int, log: RelayLog, upstream: tuple[str, str, int]):
        self.log = log
        self.upstream = upstream
        self._reported = False
        self._report_lock = threading.Lock()
        self._active: set[_Handler] = set()
        self._active_lock = threading.Lock()
        self._active_empty = threading.Condition(self._active_lock)
        self._stopping = False
        super().__init__((BIND_HOST, port), _Handler)

    def register(self, handler: _Handler) -> None:
        with self._active_lock:
            self._active.add(handler)
            late = self._stopping
        if late:  # accepted while stopping
            handler.cut_sockets()

    def may_connect(self, handler: _Handler) -> bool:
        with self._active_lock:
            return not (self._stopping or handler.cut)

    def attach_upstream(self, handler: _Handler, sock: socket.socket | None) -> bool:
        """Track the upstream socket, unless stop() has begun (then close it)."""
        with self._active_lock:
            if sock is not None and not (self._stopping or handler.cut):
                handler._upstream_sock = sock
                return True
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        return False

    def unregister(self, handler: _Handler) -> None:
        with self._active_empty:
            self._active.discard(handler)
            if not self._active:
                self._active_empty.notify_all()

    def drain(self, drain_s: float, join_s: float) -> None:
        """Wait for running exchanges; cut what is left; wait briefly again."""
        with self._active_empty:
            self._stopping = True
            self._active_empty.wait_for(lambda: not self._active, timeout=drain_s)
            left = list(self._active)
        for handler in left:
            handler.cut_sockets()
        if left:
            with self._active_empty:
                self._active_empty.wait_for(lambda: not self._active, timeout=join_s)

    def report_log_incomplete(self) -> None:
        """Print the fixed stderr line, once."""
        with self._report_lock:
            if self._reported:
                return
            self._reported = True
        print(STDERR_LOG_INCOMPLETE, file=sys.stderr, flush=True)

    def handle_error(self, request: Any, client_address: Any) -> None:
        return  # no tracebacks: they can carry request data

    def connect(self) -> http.client.HTTPConnection:
        scheme, host, port = self.upstream
        conn: http.client.HTTPConnection
        if scheme == "https":
            conn = http.client.HTTPSConnection(
                host,
                port,
                timeout=_UPSTREAM_TIMEOUT_S,
                context=_upstream_tls_context(),
            )
        else:
            conn = http.client.HTTPConnection(host, port, timeout=_UPSTREAM_TIMEOUT_S)
        conn.response_class = _FinalResponse
        return conn


def _do(self: _Handler) -> None:
    self._handle()


for _method in ("GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"):
    setattr(_Handler, f"do_{_method}", _do)


class EgressRelay:
    """A relay bound to ``127.0.0.1``. ``host`` exists only so a wider bind is refused."""

    def __init__(
        self,
        upstream: str,
        log: RelayLog,
        *,
        port: int = 0,
        host: str = BIND_HOST,
    ) -> None:
        if host != BIND_HOST:
            raise RelayConfigError("the relay binds 127.0.0.1 only")
        self._log = log
        self._server = _Server(port, log, parse_upstream(upstream))
        self._thread: threading.Thread | None = None

    @property
    def port(self) -> int:
        return int(self._server.server_address[1])

    @property
    def origin(self) -> str:
        return f"http://{BIND_HOST}:{self.port}"

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._server.serve_forever, name="egress-relay", daemon=True
        )
        self._thread.start()

    def serve_forever(self) -> None:
        self._server.serve_forever()

    def stop(self, drain_s: float | None = None) -> None:
        """Stop accepting, let running exchanges finish for ``drain_s`` seconds
        (default 3), then end the rest by force; their response lines carry
        ``outcome: relay_stopped``. Always returns within a few seconds."""
        if self._thread is not None:
            self._server.shutdown()
            self._thread.join(timeout=5)
        self._server.server_close()  # closes the listening socket only
        self._server.drain(_STOP_DRAIN_S if drain_s is None else drain_s, _STOP_JOIN_S)
        self._log.close()
