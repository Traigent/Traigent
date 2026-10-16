"""Flush-on-exit in a real subprocess (with a control) and the hard deadline."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap

SCRIPT = textwrap.dedent("""
    import atexit
    import sys
    import time

    import traigent.observability.otel as otel

    # Time exactly the exit work otel.init() registers (#2479). atexit runs
    # handlers last-in-first-out, so ``_window_end`` (registered just before
    # init) runs right after init's hooks and ``_window_start`` (registered
    # just after) runs right before them. Interpreter start-up, imports and
    # teardown stay outside the window, so load on the machine cannot move it.
    _started = []

    def _window_end():
        print(f"EXIT_WINDOW={time.monotonic() - _started[0]!r}", flush=True)

    atexit.register(_window_end)
    otel.init(
        api_key="k",
        endpoint=sys.argv[1],
        exit_flush=sys.argv[2] == "1",
        exit_flush_timeout_s=float(sys.argv[3]),
        schedule_delay_s=60,   # only the exit hook can deliver
    )
    atexit.register(lambda: _started.append(time.monotonic()))
    with otel.observe("job", as_type="chain"):
        pass
    # no explicit flush and no shutdown: rely on the exit hook alone
    """)


def _run(endpoint: str, exit_flush: bool, timeout_s: float = 5.0):
    """Run SCRIPT; return the process and the seconds its init-registered
    exit hooks took (see the window comment in SCRIPT)."""
    env = dict(os.environ, TRAIGENT_ENV="development")
    for key in ("ENVIRONMENT", "TRAIGENT_OFFLINE_MODE", "TRAIGENT_OFFLINE"):
        env.pop(key, None)
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
    windows = [
        line.split("=", 1)[1]
        for line in proc.stdout.splitlines()
        if line.startswith("EXIT_WINDOW=")
    ]
    assert len(windows) == 1, f"no exit window: {proc.stdout!r} {proc.stderr!r}"
    return proc, float(windows[0])


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
    base, base_window = _run(dead, exit_flush=False)
    with_hook, window = _run(dead, exit_flush=True, timeout_s=0.3)
    assert base.returncode == 0 and with_hook.returncode == 0
    # retry budget alone would be ~minutes (5 attempts, backoff to 30s);
    # the hard deadline keeps the extra exit time near exit_flush_timeout_s.
    # Both numbers time only the exit hooks, so start-up and teardown jitter
    # between the two launches cannot fail (or mask) the bound.
    _failure_detail = f"exit hooks took {window:.3f}s vs control {base_window:.3f}s"
    assert window - base_window < 1.5, _failure_detail
