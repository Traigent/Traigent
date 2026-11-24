"""Egress relay hardening: torn log, interim responses, streaming, bounded stop,
and request framing the relay refuses rather than rewrites."""

from __future__ import annotations

import http.client
import json
import socket
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path

import pytest
from click.testing import CliRunner

import traigent.egress.relay as relay_module
from tests.unit.egress.egress_relay_helpers import send, wait_lines
from traigent.cli.egress_commands import egress
from traigent.egress.log import RelayLog, RelayLogError
from traigent.egress.relay import EgressRelay

Script = Callable[[socket.socket], None]


class RawUpstream:
    """A loopback upstream that runs ``script(conn)`` after the request head arrives."""

    def __init__(self, script: Script) -> None:
        self._script = script
        self._server = socket.socket()
        self._server.bind(("127.0.0.1", 0))
        self._server.listen(8)
        self.url = f"http://127.0.0.1:{self._server.getsockname()[1]}"
        self.stop = threading.Event()
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def _serve(self) -> None:
        while not self.stop.is_set():
            try:
                conn, _ = self._server.accept()
            except OSError:
                return
            threading.Thread(target=self._one, args=(conn,), daemon=True).start()

    def _one(self, conn: socket.socket) -> None:
        try:
            head = b""
            while b"\r\n\r\n" not in head:
                data = conn.recv(4096)
                if not data:
                    return
                head += data
            self._script(conn)
        except OSError:
            pass
        finally:
            conn.close()

    def close(self) -> None:
        self.stop.set()
        self._server.close()


@pytest.fixture
def raw_relay(log_path: Path):
    made: list[tuple[RawUpstream, EgressRelay]] = []

    def build(script: Script) -> tuple[EgressRelay, RawUpstream]:
        upstream = RawUpstream(script)
        instance = EgressRelay(upstream.url, RelayLog(log_path))
        instance.start()
        made.append((upstream, instance))
        return instance, upstream

    yield build
    for upstream, instance in made:
        upstream.stop.set()
        instance.stop(drain_s=0.2)
        upstream.close()


# 1. a torn last line is refused, never appended to ---------------------------------


def test_torn_log_is_refused_and_left_untouched(tmp_path):
    path = tmp_path / "relay.jsonl"
    torn = b'{"schema_version":1,"seq":1,"kind":"requ'
    path.write_bytes(torn)
    with pytest.raises(RelayLogError, match="incomplete line"):
        RelayLog(path)
    assert path.read_bytes() == torn


def test_log_ending_in_ctrl_z_after_a_newline_is_refused_and_untouched(tmp_path):
    path = tmp_path / "relay.jsonl"
    content = b'{"schema_version":1,"seq":1,"kind":"request"}\n\x1a'
    path.write_bytes(content)
    with pytest.raises(RelayLogError, match="incomplete line"):
        RelayLog(path)
    assert path.read_bytes() == content


def test_cli_refuses_a_torn_log_with_a_fixed_message(tmp_path):
    path = tmp_path / "relay.jsonl"
    path.write_bytes(b'{"seq":1')
    result = CliRunner().invoke(
        egress,
        ["relay", "--upstream", "https://example.com", "--log", str(path)],
    )
    assert result.exit_code != 0
    assert "incomplete line" in result.output
    assert '{"seq"' not in result.output


def test_log_with_final_newline_or_empty_file_opens(tmp_path):
    empty = tmp_path / "empty.jsonl"
    empty.write_bytes(b"")
    RelayLog(empty).close()
    ok = tmp_path / "ok.jsonl"
    ok.write_text(json.dumps({"schema_version": 1, "seq": 4, "kind": "request"}) + "\n")
    log = RelayLog(ok)
    assert log.log_request("GET", "/x", 0, "0" * 64) == 5
    log.close()


# 2. interim responses --------------------------------------------------------------


def test_103_then_200_delivers_the_200(raw_relay, log_path):
    def script(conn: socket.socket) -> None:
        conn.sendall(
            b"HTTP/1.1 103 Early Hints\r\nLink: </a>; rel=preload\r\n\r\n"
            b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\nok"
        )

    instance, _ = raw_relay(script)
    status, _, payload = send(instance, "GET", "/api/v1/x")
    assert (status, payload) == (200, b"ok")
    assert wait_lines(log_path, 2)[1]["status"] == 200


def test_101_switching_protocols_is_refused_and_logged(raw_relay, log_path):
    def script(conn: socket.socket) -> None:
        conn.sendall(b"HTTP/1.1 101 Switching Protocols\r\nUpgrade: x\r\n\r\n")

    instance, _ = raw_relay(script)
    status, _, _ = send(instance, "GET", "/x")
    assert status == 502
    lines = wait_lines(log_path, 2)
    assert lines[1]["outcome"] == "upstream_unsupported"


# 3. streaming ----------------------------------------------------------------------


def test_small_chunks_are_passed_on_as_they_arrive(raw_relay):
    second_sent = threading.Event()

    def script(conn: socket.socket) -> None:
        conn.sendall(b"HTTP/1.1 200 OK\r\nConnection: close\r\n\r\none")
        second_sent.wait(10)
        conn.sendall(b"two")

    instance, _ = raw_relay(script)
    client = http.client.HTTPConnection("127.0.0.1", instance.port, timeout=5)
    client.request("GET", "/events")
    response = client.getresponse()
    first = response.read1(3)  # a stalled relay times out here
    second_sent.set()
    assert first == b"one"
    assert response.read() == b"two"
    client.close()


# 4. bounded stop -------------------------------------------------------------------


def _endless(conn: socket.socket) -> None:
    conn.sendall(b"HTTP/1.1 200 OK\r\nConnection: close\r\n\r\n")
    for _ in range(2000):
        conn.sendall(b"x")
        time.sleep(0.02)


def test_stop_returns_promptly_with_an_endless_upstream_stream(raw_relay, log_path):
    instance, _ = raw_relay(_endless)
    client = http.client.HTTPConnection("127.0.0.1", instance.port, timeout=10)
    client.request("GET", "/stream")
    response = client.getresponse()
    assert response.read1(1) == b"x"
    started = time.monotonic()
    instance.stop(drain_s=0.3)
    assert time.monotonic() - started < 4
    client.close()
    lines = wait_lines(log_path, 2)
    assert lines[1]["outcome"] == "relay_stopped"


def test_stop_returns_promptly_with_a_trickling_client(raw_relay):
    instance, _ = raw_relay(_endless)
    raw = socket.create_connection(("127.0.0.1", instance.port), timeout=10)
    raw.sendall(b"POST /x HTTP/1.1\r\nHost: h\r\nContent-Length: 100\r\n\r\nab")
    time.sleep(0.2)  # the handler is now waiting for the rest of the body
    started = time.monotonic()
    instance.stop(drain_s=0.3)
    assert time.monotonic() - started < 4
    raw.close()


def test_exec_returns_promptly_after_the_child_exits_with_a_stream_open(
    tmp_path, log_path, monkeypatch
):
    monkeypatch.setattr(relay_module, "_STOP_DRAIN_S", 0.3)

    def silent_after_headers(conn: socket.socket) -> None:
        conn.sendall(b"HTTP/1.1 200 OK\r\nConnection: close\r\n\r\n")
        time.sleep(30)

    upstream = RawUpstream(silent_after_headers)
    child = tmp_path / "child.py"
    child.write_text(
        "import os,socket,sys,urllib.parse as u\n"
        "p=u.urlparse(os.environ['TRAIGENT_BACKEND_URL'])\n"
        "s=socket.create_connection((p.hostname,p.port))\n"
        "s.sendall(b'GET /x HTTP/1.1\\r\\nHost: h\\r\\n\\r\\n')\n"
        "s.recv(4096)\n"
        "os._exit(0)\n"
    )
    started = time.monotonic()
    try:
        result = CliRunner().invoke(
            egress,
            [
                "relay",
                "--upstream",
                upstream.url,
                "--log",
                str(log_path),
                "--exec",
                f"{sys.executable} {child}",
            ],
        )
    finally:
        upstream.close()
    assert result.exit_code == 0
    assert time.monotonic() - started < 15
    assert wait_lines(log_path, 2)[1].get("outcome") == "relay_stopped"


# 5. request framing the relay will not rewrite -------------------------------------


def _raw_status(port: int, head: bytes) -> int:
    raw = socket.create_connection(("127.0.0.1", port), timeout=10)
    try:
        raw.sendall(head)
        return int(raw.recv(4096).split(b" ", 2)[1])
    finally:
        raw.close()


def test_transfer_encoding_other_than_plain_chunked_is_refused(
    relay, upstream, log_path
):
    head = (
        b"POST /x HTTP/1.1\r\nHost: h\r\nTransfer-Encoding: gzip, chunked\r\n\r\n"
        b"1\r\na\r\n0\r\n\r\n"
    )
    assert _raw_status(relay.port, head) == 501
    assert upstream.seen == []
    assert log_path.read_text() == ""  # refused before forwarding: no log line


def test_transfer_encoding_with_content_length_is_refused(relay, upstream, log_path):
    head = (
        b"POST /x HTTP/1.1\r\nHost: h\r\nTransfer-Encoding: chunked\r\n"
        b"Content-Length: 1\r\n\r\n1\r\na\r\n0\r\n\r\n"
    )
    assert _raw_status(relay.port, head) == 400
    assert upstream.seen == []
    assert log_path.read_text() == ""


def test_chunked_in_any_case_is_still_accepted(relay, upstream):
    head = (
        b"POST /x HTTP/1.1\r\nHost: h\r\nTransfer-Encoding: Chunked\r\n\r\n"
        b"1\r\na\r\n0\r\n\r\n"
    )
    assert _raw_status(relay.port, head) == 200
    assert upstream.seen[0].body == b"a"


# 6. stop() during a blocked upload, and stop() before the upstream is reached --------


def _listener() -> socket.socket:
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(8)
    return server


def _wait_until(predicate: Callable[[], bool], timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return predicate()


def test_stop_cuts_an_upstream_that_stopped_reading_during_a_large_upload(log_path):
    upstream = _listener()
    upstream.settimeout(10)
    instance = EgressRelay(
        f"http://127.0.0.1:{upstream.getsockname()[1]}", RelayLog(log_path)
    )
    instance.start()
    total = 48 * 1024 * 1024  # generated, never stored; far above the socket buffers

    def upload() -> None:
        try:
            raw = socket.create_connection(("127.0.0.1", instance.port), timeout=30)
            raw.sendall(
                b"POST /x HTTP/1.1\r\nHost: h\r\nContent-Length: %d\r\n\r\n" % total
            )
            block = b"\0" * (1024 * 1024)
            for _ in range(total // len(block)):
                raw.sendall(block)
            raw.close()
        except OSError:
            pass

    threading.Thread(target=upload, daemon=True).start()
    peer, _ = upstream.accept()  # accepts, then never reads
    try:
        time.sleep(1.0)  # let the relay fill the socket buffers and block in send
        started = time.monotonic()
        instance.stop(drain_s=0.3)
        assert time.monotonic() - started < 4
        assert _wait_until(lambda: not instance._server._active, 3), "handler alive"
        peer.settimeout(5)
        closed = False
        try:
            while (
                True
            ):  # buffered bytes, then EOF or reset: the upstream side saw a close
                if not peer.recv(1 << 20):
                    closed = True
                    break
        except ConnectionResetError:
            closed = True
        assert closed
        peer.settimeout(0.5)  # and nothing more arrives afterwards
        try:
            assert peer.recv(1) == b""
        except (ConnectionResetError, TimeoutError):
            pass
    finally:
        peer.close()
        upstream.close()
    assert lines_are_whole(log_path)


def lines_are_whole(path: Path) -> bool:
    text = path.read_text()
    return all(json.loads(x) for x in text.splitlines())


def test_exchange_that_has_not_connected_when_stop_begins_never_connects(log_path):
    upstream = _listener()
    upstream.settimeout(1.0)
    instance = EgressRelay(
        f"http://127.0.0.1:{upstream.getsockname()[1]}", RelayLog(log_path)
    )
    instance.start()
    entered = threading.Event()
    release = threading.Event()
    original = instance._server.connect

    def gated():  # noqa: ANN202
        entered.set()
        release.wait(10)  # the handler is past the log line, not yet connected
        return original()

    instance._server.connect = gated  # type: ignore[method-assign]
    outcome: list[int] = []

    def client() -> None:
        try:
            outcome.append(send(instance, "POST", "/x", b"body")[0])
        except OSError:
            outcome.append(-1)

    thread = threading.Thread(target=client, daemon=True)
    thread.start()
    assert entered.wait(10)
    instance.stop(drain_s=0.2)
    release.set()
    assert _wait_until(lambda: not instance._server._active, 3)
    thread.join(10)
    with pytest.raises(TimeoutError):
        upstream.accept()  # the relay never connected
    upstream.close()
    assert outcome and outcome[0] != 200
