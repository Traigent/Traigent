"""Flush-on-exit in a real subprocess (with a control) and the hard deadline."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import time

SCRIPT = textwrap.dedent(
    """
    import sys
    import traigent.observability.otel as otel

    otel.init(
        api_key="k",
        endpoint=sys.argv[1],
        exit_flush=sys.argv[2] == "1",
        exit_flush_timeout_s=float(sys.argv[3]),
        schedule_delay_s=60,   # only the exit hook can deliver
    )
    with otel.observe("job", as_type="chain"):
        pass
    # no explicit flush and no shutdown: rely on the exit hook alone
    """
)


def _run(endpoint: str, exit_flush: bool, timeout_s: float = 5.0):
    env = dict(os.environ, TRAIGENT_ENV="development")
    for key in ("ENVIRONMENT", "TRAIGENT_OFFLINE_MODE", "TRAIGENT_OFFLINE"):
        env.pop(key, None)
    start = time.time()
    proc = subprocess.run(  # noqa: S603
        [
            sys.executable,
            "-c",
            SCRIPT,
            endpoint,
            "1" if exit_flush else "0",
            str(timeout_s),
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    return proc, time.time() - start


def test_exit_hook_delivers_spans_when_the_process_ends(collector):
    proc, _ = _run(collector.base_url, exit_flush=True)
    assert proc.returncode == 0, proc.stderr
    names = [s.name for _, _, s in collector.spans()]
    assert names == ["job"]


def test_control_without_exit_flush_nothing_is_delivered(collector):
    proc, _ = _run(collector.base_url, exit_flush=False)
    assert proc.returncode == 0, proc.stderr
    assert collector.spans() == []  # proves the test above depends on the hook


def test_exit_hook_is_bounded_when_the_collector_is_down():
    """A dead endpoint must not hold the process past the exit deadline."""
    dead = "http://127.0.0.1:9"  # discard port: connection refused
    base, base_elapsed = _run(dead, exit_flush=False)
    with_hook, elapsed = _run(dead, exit_flush=True, timeout_s=0.3)
    assert base.returncode == 0 and with_hook.returncode == 0
    # retry budget alone would be ~minutes (5 attempts, backoff to 30s);
    # the hard deadline keeps the extra time near exit_flush_timeout_s.
    assert elapsed - base_elapsed < 1.5
