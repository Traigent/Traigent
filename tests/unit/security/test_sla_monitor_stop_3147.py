"""Issue #3147: SLAMonitor.stop_monitoring returns promptly and stops the
thread, and a restart never leaves two monitor loops running."""

from __future__ import annotations

import threading
import time
from unittest.mock import Mock

import pytest

from traigent.security.enterprise import (
    MetricsCollector,
    SLAConfiguration,
    SLAMonitor,
    SLATier,
)


def _monitor() -> tuple[SLAMonitor, Mock]:
    collector = MetricsCollector()
    collect = Mock(wraps=collector.collect_system_metrics)
    collector.collect_system_metrics = collect  # type: ignore[method-assign]
    sla = SLAConfiguration.from_tier(SLATier.STANDARD, "admin@test.com")
    return SLAMonitor(sla, collector), collect


def _monitor_threads(monitor: SLAMonitor) -> list[threading.Thread]:
    return [
        t
        for t in threading.enumerate()
        if getattr(t, "_target", None) is not None
        and getattr(t._target, "__self__", None) is monitor
    ]


def _wait_for(predicate, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


@pytest.mark.timeout(30)
def test_stop_returns_promptly_and_thread_exits() -> None:
    monitor, collect = _monitor()
    monitor.start_monitoring(interval_seconds=60)
    assert _wait_for(lambda: collect.call_count >= 1)
    thread = monitor.monitor_thread
    assert thread is not None

    started = time.monotonic()
    monitor.stop_monitoring()
    elapsed = time.monotonic() - started

    assert elapsed < 2.0
    assert not thread.is_alive()
    assert monitor.running is False


@pytest.mark.timeout(30)
def test_restart_runs_exactly_one_loop() -> None:
    monitor, collect = _monitor()
    monitor.start_monitoring(interval_seconds=60)
    assert _wait_for(lambda: collect.call_count >= 1)
    monitor.stop_monitoring()
    monitor.start_monitoring(interval_seconds=60)
    try:
        assert _wait_for(lambda: collect.call_count >= 2)
        assert len(_monitor_threads(monitor)) == 1
    finally:
        monitor.stop_monitoring()
    assert _monitor_threads(monitor) == []
