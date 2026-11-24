"""Fixtures for the egress relay tests: a fake upstream and a relay in front of it."""

from __future__ import annotations

import http.server
import threading
from pathlib import Path

import pytest

from tests.unit.egress.egress_relay_helpers import FakeUpstream, _make_handler
from traigent.egress.log import RelayLog
from traigent.egress.relay import EgressRelay


@pytest.fixture
def upstream():
    state = FakeUpstream()
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _make_handler(state))
    server.daemon_threads = True
    state.port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield state
    server.shutdown()
    server.server_close()
    thread.join(timeout=5)


@pytest.fixture
def log_path(tmp_path: Path) -> Path:
    return tmp_path / "egress" / "relay.jsonl"


@pytest.fixture
def relay(upstream: FakeUpstream, log_path: Path):
    log = RelayLog(log_path)
    instance = EgressRelay(upstream.url, log)
    instance.start()
    yield instance
    instance.stop()
