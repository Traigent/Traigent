"""Egress relay: forwarding fidelity, fail-closed behaviour and what the log holds."""

from __future__ import annotations

import hashlib
import http.client
import json
import os
import stat
import threading
import time
from pathlib import Path

import pytest

from tests.unit.egress.egress_relay_helpers import break_log, send, wait_lines
from traigent.egress.log import RelayLog
from traigent.egress.relay import EgressRelay, RelayConfigError, parse_upstream

CANARY = "CANARY-7f3a9c-do-not-log"


def _lines(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines()]


@pytest.mark.parametrize(
    "body",
    [
        json.dumps({"a": 1, "b": [1, 2, 3]}).encode(),
        bytes(range(256)) * 4,
        b"",
        os.urandom(5 * 1024 * 1024),
    ],
    ids=["json", "binary", "empty", "5MB"],
)
def test_body_forwarded_identically_and_hash_matches(relay, upstream, log_path, body):
    status, _, _ = send(relay, "POST", "/api/v1/projects", body)
    assert status == 200
    assert upstream.seen[0].body == body
    request = _lines(log_path)[0]
    assert request["bytes"] == len(body)
    assert request["sha256"] == hashlib.sha256(upstream.seen[0].body).hexdigest()


@pytest.mark.parametrize("method", ["GET", "POST", "PUT", "PATCH", "DELETE"])
def test_methods_forwarded_and_logged(relay, upstream, log_path, method):
    body = b"x" if method in ("POST", "PUT", "PATCH") else None
    status, _, payload = send(relay, method, "/api/v1/projects", body)
    assert (status, payload) == (200, upstream.response_body)
    assert upstream.seen[0].method == method
    lines = wait_lines(log_path, 2)
    assert [ln["kind"] for ln in lines] == ["request", "response"]
    assert lines[0]["method"] == method
    assert lines[1]["status"] == 200
    assert lines[1]["bytes"] == len(upstream.response_body)
    assert lines[0]["seq"] == lines[1]["seq"]


def test_query_forwarded_but_not_logged(relay, upstream, log_path):
    send(relay, "GET", f"/api/v1/projects?token={CANARY}&page=2")
    assert upstream.seen[0].path == f"/api/v1/projects?token={CANARY}&page=2"
    assert CANARY not in log_path.read_text()
    assert _lines(log_path)[0]["route"] == "/api/v1/projects"


def test_canary_in_header_query_path_and_body_never_logged(relay, upstream, log_path):
    send(
        relay,
        "POST",
        f"/api/v1/projects/{CANARY}/projects?q={CANARY}",
        json.dumps({CANARY: CANARY}).encode(),
        {"X-Custom": CANARY, "Authorization": f"Bearer {CANARY}"},
    )
    assert upstream.seen[0].headers["X-Custom"] == CANARY
    text = log_path.read_text()
    assert CANARY not in text
    assert CANARY.lower() not in text.lower()
    assert _lines(log_path)[0]["route"] == "/api/v1/projects/{}/projects"


def test_authorization_header_reaches_upstream_and_is_not_logged(
    relay, upstream, log_path
):
    send(
        relay, "GET", "/api/v1/projects", headers={"Authorization": "Bearer s3cr3t-tok"}
    )
    assert upstream.seen[0].headers["Authorization"] == "Bearer s3cr3t-tok"
    text = log_path.read_text()
    assert "s3cr3t-tok" not in text and "uthorization" not in text


def test_host_header_is_the_upstream_host(relay, upstream):
    send(relay, "GET", "/x", headers={"Host": "example.invalid"})
    assert upstream.seen[0].headers["Host"] == f"127.0.0.1:{upstream.port}"


def test_unwritable_log_returns_502_and_forwards_nothing(relay, upstream):
    break_log(relay._log)  # the log can no longer be written
    status, _, payload = send(relay, "POST", "/api/v1/projects", b"body")
    assert status == 502
    assert b"not forwarded" in payload
    assert upstream.seen == []


def test_upstream_down_returns_502_and_is_logged(log_path):
    log = RelayLog(log_path)
    instance = EgressRelay("http://127.0.0.1:9", log)  # nothing listens on 9
    instance.start()
    try:
        status, _, _ = send(instance, "GET", "/api/v1/projects")
    finally:
        instance.stop()
    assert status == 502
    lines = _lines(log_path)
    assert [ln["kind"] for ln in lines] == ["request", "response"]
    assert lines[1]["status"] == 502
    assert lines[1]["outcome"] == "upstream_unreachable"


def test_redirect_passed_through_not_followed(relay, upstream):
    upstream.status = 302
    upstream.extra_headers = [("Location", "http://127.0.0.1:1/elsewhere")]
    upstream.response_body = b""
    status, headers, _ = send(relay, "GET", "/api/v1/projects")
    assert status == 302
    assert headers["Location"] == "http://127.0.0.1:1/elsewhere"
    assert len(upstream.seen) == 1


def test_response_headers_pass_through_including_duplicates(relay, upstream):
    upstream.extra_headers = [("Set-Cookie", "a=1"), ("Set-Cookie", "b=2")]
    conn = http.client.HTTPConnection("127.0.0.1", relay.port, timeout=30)
    conn.request("GET", "/x")
    response = conn.getresponse()
    cookies = [v for k, v in response.getheaders() if k.lower() == "set-cookie"]
    response.read()
    conn.close()
    assert cookies == ["a=1", "b=2"]


def test_concurrent_requests_get_unique_increasing_seq(relay, upstream, log_path):
    results: list[int] = []

    def one(i: int) -> None:
        results.append(send(relay, "POST", "/api/v1/projects", f"b{i}".encode())[0])

    threads = [threading.Thread(target=one, args=(i,)) for i in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)
    assert results == [200] * 20
    deadline = time.monotonic() + 10  # the response line lands just after the reply
    while time.monotonic() < deadline and len(_lines(log_path)) < 40:
        time.sleep(0.02)
    lines = _lines(log_path)
    requests_ = [ln["seq"] for ln in lines if ln["kind"] == "request"]
    responses = sorted(ln["seq"] for ln in lines if ln["kind"] == "response")
    assert requests_ == sorted(requests_) == list(range(1, 21))
    assert responses == list(range(1, 21))


def test_request_line_is_fsynced_before_upstream_sees_a_byte(
    upstream, log_path, monkeypatch
):
    """fsync of the request line has completed when the relay first connects."""
    import traigent.egress.log as log_module

    completed: list[int] = []
    real_fsync = os.fsync

    def recording_fsync(fd: int) -> None:
        real_fsync(fd)
        completed.append(fd)

    monkeypatch.setattr(log_module.os, "fsync", recording_fsync)
    instance = EgressRelay(upstream.url, RelayLog(log_path))
    at_connect: list[int] = []
    original = instance._server.connect

    def spy():  # noqa: ANN202
        at_connect.append(len(completed))  # before any byte goes upstream
        return original()

    instance._server.connect = spy  # type: ignore[method-assign]
    instance.start()
    try:
        send(instance, "POST", "/x", b"1")
    finally:
        instance.stop()
    assert at_connect == [1]


def test_fsync_failure_returns_502_and_forwards_nothing(relay, upstream, monkeypatch):
    import traigent.egress.log as log_module

    def failing_fsync(fd: int) -> None:
        raise OSError("disk gone")

    monkeypatch.setattr(log_module.os, "fsync", failing_fsync)
    status, _, payload = send(relay, "POST", "/x", b"1")
    assert status == 502 and b"not forwarded" in payload
    assert upstream.seen == []


def test_bind_is_loopback_and_wider_bind_refused(upstream, log_path):
    log = RelayLog(log_path)
    instance = EgressRelay(upstream.url, log)
    try:
        assert instance._server.server_address[0] == "127.0.0.1"
    finally:
        instance.stop()
    for host in ("0.0.0.0", "::", "192.0.2.1", "localhost"):  # noqa: S104
        with pytest.raises(RelayConfigError):
            EgressRelay(
                upstream.url, RelayLog(log_path.with_name(host + ".jsonl")), host=host
            )


@pytest.mark.parametrize(
    "url",
    [
        "http://example.com",
        "http://localhost:80",
        "ftp://x",
        "example.com",
        "https://",
        "https://h/path",
        "https://u:p@h",
        "https://h?q=1",
    ],
)
def test_upstream_rules_refuse(url):
    with pytest.raises(RelayConfigError):
        parse_upstream(url)


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://api.example.com", ("https", "api.example.com", 443)),
        ("https://api.example.com:8443/", ("https", "api.example.com", 8443)),
        ("http://127.0.0.1:8080", ("http", "127.0.0.1", 8080)),
        ("http://[::1]:8080", ("http", "::1", 8080)),
    ],
)
def test_upstream_rules_accept(url, expected):
    assert parse_upstream(url) == expected


@pytest.mark.skipif(os.name == "nt", reason="POSIX permission bits")
def test_log_file_mode_is_0600(relay, log_path):
    send(relay, "GET", "/x")
    assert stat.S_IMODE(log_path.stat().st_mode) == 0o600


@pytest.mark.skipif(
    os.name == "nt", reason="the writer lock is POSIX-only (fcntl.flock)"
)
def test_second_relay_cannot_share_a_log(relay, log_path):
    with pytest.raises(Exception, match="already writing"):
        RelayLog(log_path)


def test_seq_continues_across_restarts(upstream, log_path):
    for _ in range(2):
        instance = EgressRelay(upstream.url, RelayLog(log_path))
        instance.start()
        send(instance, "GET", "/x")
        instance.stop()
    seqs = [ln["seq"] for ln in _lines(log_path) if ln["kind"] == "request"]
    assert seqs == [1, 2]


def test_chunked_request_body_is_forwarded_and_hashed(relay, upstream, log_path):
    conn = http.client.HTTPConnection("127.0.0.1", relay.port, timeout=30)
    conn.request(
        "POST",
        "/x",
        body=iter([b"hello ", b"world"]),
        encode_chunked=True,
        headers={"Transfer-Encoding": "chunked"},
    )
    assert conn.getresponse().status == 200
    conn.close()
    assert upstream.seen[0].body == b"hello world"
    assert _lines(log_path)[0]["sha256"] == hashlib.sha256(b"hello world").hexdigest()


def test_absolute_form_target_is_refused(relay, upstream):
    status, _, _ = send(relay, "GET", "http://evil.example/x")
    assert status == 400
    assert upstream.seen == []


def test_response_log_failure_still_delivers_then_refuses_everything(
    relay, upstream, log_path, capsys
):
    class Breaker(list):
        def append(self, item):  # noqa: ANN001
            super().append(item)
            break_log(relay._log)  # the request line is out; the response line fails

    upstream.seen = Breaker(upstream.seen)
    status, _, payload = send(relay, "POST", "/api/v1/projects", b"one")
    assert (status, payload) == (200, upstream.response_body)  # still delivered
    deadline = time.monotonic() + 10
    while not relay._log.failed and time.monotonic() < deadline:
        time.sleep(0.02)
    for _ in range(2):
        status, _, payload = send(relay, "GET", "/api/v1/projects")
        assert status == 502 and b"not forwarded" in payload
    assert len(upstream.seen) == 1  # nothing further was forwarded
    err = capsys.readouterr().err
    assert err.count("the log can no longer be written") == 1
    assert "/api/v1" not in err
    assert [ln["kind"] for ln in _lines(log_path)] == ["request"]


@pytest.mark.skipif(os.name == "nt", reason="POSIX permission bits")
def test_existing_loose_log_file_is_tightened_to_0600(tmp_path):
    path = tmp_path / "a" / "b" / "relay.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text("")
    path.chmod(0o644)
    log = RelayLog(path)
    log.close()
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    created = tmp_path / "new" / "deep" / "relay.jsonl"
    RelayLog(created).close()
    assert stat.S_IMODE(created.stat().st_mode) == 0o600
    assert stat.S_IMODE(created.parent.stat().st_mode) == 0o700


@pytest.mark.parametrize(
    ("method", "status"), [("HEAD", 200), ("GET", 204), ("GET", 304)]
)
def test_responses_without_a_body(relay, upstream, log_path, method, status):
    upstream.status = status
    upstream.response_body = b"" if status != 200 else b"ignored-for-head"
    code, _, payload = send(relay, method, "/api/v1/projects")
    assert code == status and payload == b""
    assert wait_lines(log_path, 2)[1]["bytes"] == 0


def test_hop_by_hop_headers_not_forwarded_either_way(relay, upstream):
    upstream.extra_headers = [("Keep-Alive", "timeout=5"), ("X-Keep", "yes")]
    _, headers, _ = send(
        relay,
        "GET",
        "/x",
        headers={
            "Keep-Alive": "timeout=9",
            "Proxy-Authorization": "Basic abc",
            "Connection": "X-Named, close",
            "X-Named": "drop-me",
            "X-Other": "keep-me",
        },
    )
    sent = {k.lower(): v for k, v in upstream.seen[0].headers.items()}
    for name in ("keep-alive", "proxy-authorization", "x-named"):
        assert name not in sent
    assert sent["x-other"] == "keep-me"
    assert "Keep-Alive" not in headers and headers["X-Keep"] == "yes"


def test_expect_100_continue_is_answered_and_not_forwarded(relay, upstream, log_path):
    conn = http.client.HTTPConnection("127.0.0.1", relay.port, timeout=30)
    conn.request("POST", "/x", body=b"payload", headers={"Expect": "100-continue"})
    assert conn.getresponse().status == 200
    conn.close()
    assert upstream.seen[0].body == b"payload"
    assert "expect" not in {k.lower() for k in upstream.seen[0].headers}


def test_upstream_timeout_is_502_and_logged(log_path, monkeypatch):
    import socket

    import traigent.egress.relay as relay_module

    monkeypatch.setattr(relay_module, "_UPSTREAM_TIMEOUT_S", 0.3)
    silent = socket.socket()
    silent.bind(("127.0.0.1", 0))
    silent.listen(1)  # accepts into the backlog, never answers
    instance = EgressRelay(
        f"http://127.0.0.1:{silent.getsockname()[1]}", RelayLog(log_path)
    )
    instance.start()
    try:
        status, _, _ = send(instance, "GET", "/x")
    finally:
        instance.stop()
        silent.close()
    assert status == 502
    lines = _lines(log_path)
    assert lines[1]["outcome"] == "upstream_unreachable"


def test_client_disconnect_mid_body_forwards_nothing_and_relay_survives(
    relay, upstream, log_path
):
    import socket

    raw = socket.create_connection(("127.0.0.1", relay.port), timeout=10)
    raw.sendall(b"POST /x HTTP/1.1\r\nHost: h\r\nContent-Length: 100\r\n\r\nabc")
    raw.close()
    time.sleep(0.2)
    assert upstream.seen == []
    assert send(relay, "GET", "/x")[0] == 200
    assert [ln["kind"] for ln in wait_lines(log_path, 2)] == ["request", "response"]
