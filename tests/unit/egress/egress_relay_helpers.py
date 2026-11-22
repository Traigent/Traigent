"""Fake upstream and relay fixtures for the egress relay tests (loopback only)."""

from __future__ import annotations

import http.client
import http.server
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from traigent.egress.log import RelayLog
from traigent.egress.relay import EgressRelay


@dataclass
class Seen:
    method: str
    path: str
    headers: dict[str, str]
    body: bytes


@dataclass
class FakeUpstream:
    seen: list[Seen] = field(default_factory=list)
    status: int = 200
    response_body: bytes = b'{"ok":true}'
    extra_headers: list[tuple[str, str]] = field(default_factory=list)
    port: int = 0

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"


def _make_handler(state: FakeUpstream) -> type[http.server.BaseHTTPRequestHandler]:
    class Handler(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args: Any) -> None:
            return

        def _serve(self) -> None:
            length = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(length) if length else b""
            state.seen.append(
                Seen(self.command, self.path, dict(self.headers.items()), body)
            )
            self.send_response(state.status)
            for key, value in state.extra_headers:
                self.send_header(key, value)
            self.send_header("Content-Length", str(len(state.response_body)))
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(state.response_body)

    for method in ("GET", "POST", "PUT", "PATCH", "DELETE", "HEAD"):
        setattr(Handler, f"do_{method}", Handler._serve)
    return Handler


def send(
    relay: EgressRelay,
    method: str,
    path: str,
    body: bytes | None = None,
    headers: dict[str, str] | None = None,
) -> tuple[int, dict[str, str], bytes]:
    conn = http.client.HTTPConnection("127.0.0.1", relay.port, timeout=30)
    try:
        conn.request(method, path, body=body, headers=headers or {})
        response = conn.getresponse()
        return response.status, dict(response.getheaders()), response.read()
    finally:
        conn.close()


def wait_lines(path: Path, count: int, timeout: float = 15.0) -> list[dict]:
    """Return the log lines once ``count`` exist.

    The response line is written just after the reply reaches the client, so a test
    that reads the log right after ``send`` must wait for it.
    """
    deadline = time.monotonic() + timeout
    lines: list[dict] = []
    while time.monotonic() < deadline:
        lines = [json.loads(x) for x in path.read_text().splitlines()]
        if len(lines) >= count:
            return lines
        time.sleep(0.02)
    return lines


def break_log(log: RelayLog) -> None:
    """Make every further write to ``log`` fail, without closing a reusable fd."""
    stale = log._fd
    log._fd = os.open(os.devnull, os.O_RDONLY)  # writes to it fail with EBADF
    os.close(stale)
