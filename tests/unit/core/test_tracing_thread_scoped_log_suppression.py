"""The OTel header-log suppression must only silence the exporting thread."""

from __future__ import annotations

import logging
import threading

import pytest

from traigent.core import tracing

SYNTHETIC_CREDENTIAL = "x-api-key=tg_SYNTHETIC_SECRET_123;"


class _Capture(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


@pytest.fixture
def capture():
    handler = _Capture()
    names = ["urllib3.connectionpool", "requests", "opentelemetry.util.re"]
    loggers = [logging.getLogger(n) for n in names]
    saved = [(lg.level, lg.propagate) for lg in loggers]
    for lg in loggers:
        lg.setLevel(logging.DEBUG)
        lg.addHandler(handler)
    yield handler
    for lg, (lvl, prop) in zip(loggers, saved, strict=True):
        lg.removeHandler(handler)
        lg.setLevel(lvl)
        lg.propagate = prop


def _hold_suppression():
    entered = threading.Event()
    release = threading.Event()
    log_done = threading.Event()

    def worker() -> None:
        with tracing._quiet_otel_header_logs():
            entered.set()
            logging.getLogger("opentelemetry.util.re").warning(SYNTHETIC_CREDENTIAL)
            logging.getLogger("urllib3.connectionpool").warning(SYNTHETIC_CREDENTIAL)
            log_done.set()
            release.wait(10)

    t = threading.Thread(target=worker)
    t.start()
    assert entered.wait(10) and log_done.wait(10)
    return t, release


def test_other_threads_urllib3_and_requests_warnings_survive_export(capture):
    t, release = _hold_suppression()
    try:
        logging.getLogger("urllib3.connectionpool").warning("host app urllib3 warn")
        logging.getLogger("requests").warning("host app requests warn")
    finally:
        release.set()
        t.join(10)
    assert "host app urllib3 warn" in capture.messages
    assert "host app requests warn" in capture.messages


def test_exporting_thread_credential_record_is_dropped(capture):
    t, release = _hold_suppression()
    release.set()
    t.join(10)
    assert not any("SYNTHETIC_SECRET" in m for m in capture.messages)


def test_nested_and_concurrent_contexts_remove_only_their_own_filter(capture):
    lg = logging.getLogger("urllib3.connectionpool")
    before = list(lg.filters)
    t, release = _hold_suppression()
    with tracing._quiet_otel_header_logs():
        with tracing._quiet_otel_header_logs():
            lg.warning(SYNTHETIC_CREDENTIAL)
        lg.warning(SYNTHETIC_CREDENTIAL)
    assert len(lg.filters) == len(before) + 1  # only the worker's remains
    release.set()
    t.join(10)
    assert lg.filters == before
    assert not any("SYNTHETIC_SECRET" in m for m in capture.messages)
