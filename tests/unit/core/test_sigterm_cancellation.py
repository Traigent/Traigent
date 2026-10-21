"""SIGTERM is translated into the Ctrl+C cancellation path (#2667)."""

from __future__ import annotations

import signal
import threading

import pytest

from traigent.core.sigterm_cancellation import sigterm_as_cancellation


def test_sigterm_raises_keyboard_interrupt_and_restores_handler():
    before = signal.getsignal(signal.SIGTERM)
    with sigterm_as_cancellation():
        assert signal.getsignal(signal.SIGTERM) is not before
        with pytest.raises(KeyboardInterrupt):
            signal.raise_signal(signal.SIGTERM)
    assert signal.getsignal(signal.SIGTERM) is before


def test_host_installed_handler_is_never_overridden():
    calls: list[int] = []

    def host(signum, frame):
        calls.append(signum)

    prev = signal.signal(signal.SIGTERM, host)
    try:
        with sigterm_as_cancellation():
            assert signal.getsignal(signal.SIGTERM) is host
            signal.raise_signal(signal.SIGTERM)
        assert calls == [signal.SIGTERM]
        assert signal.getsignal(signal.SIGTERM) is host
    finally:
        signal.signal(signal.SIGTERM, prev)


def test_ignored_sigterm_is_left_ignored():
    prev = signal.signal(signal.SIGTERM, signal.SIG_IGN)
    try:
        with sigterm_as_cancellation():
            assert signal.getsignal(signal.SIGTERM) == signal.SIG_IGN
    finally:
        signal.signal(signal.SIGTERM, prev)


def test_non_main_thread_is_a_noop():
    seen: list[object] = []

    def worker():
        with sigterm_as_cancellation():
            seen.append("entered")

    t = threading.Thread(target=worker)
    t.start()
    t.join()
    assert seen == ["entered"]


def test_orchestrator_optimize_wires_sigterm_scope():
    import inspect

    from traigent.core.orchestrator import OptimizationOrchestrator

    src = inspect.getsource(OptimizationOrchestrator.optimize)
    assert "sigterm_as_cancellation()" in src
    assert "sigterm_scope.__exit__" in src


def test_sigterm_mid_run_finalizes_as_cancelled_in_subprocess(tmp_path):
    """The child signals itself mid-trial; the run must end cancelled, not die."""
    import os
    import subprocess
    import sys

    (tmp_path / "d.jsonl").write_text(
        '{"input":{"q":"1"},"output":"ok"}\n{"input":{"q":"2"},"output":"ok"}\n'
    )
    (tmp_path / "app.py").write_text(
        "import os, signal, traigent\n"
        "@traigent.optimize(configuration_space={'v': ['a', 'b', 'c']},\n"
        "    objectives=['accuracy'], eval_dataset='d.jsonl', offline=True,\n"
        "    algorithm='grid', max_trials=3)\n"
        "def agent(q: str) -> str:\n"
        "    os.kill(os.getpid(), signal.SIGTERM)\n"
        "    return 'ok'\n"
        "r = agent.optimize_sync()\n"
        "print('RESULT', r.status)\n"
    )
    env = {
        **os.environ,
        "HOME": str(tmp_path),
        "TRAIGENT_RESULTS_FOLDER": str(tmp_path / "res"),
        "TRAIGENT_MOCK_LLM": "true",
        "PYTHONPATH": os.getcwd(),
    }
    for k in ("TRAIGENT_API_KEY", "TRAIGENT_BACKEND_URL"):
        env.pop(k, None)
    proc = subprocess.run(
        [sys.executable, "app.py"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode != -signal.SIGTERM, proc.stderr[-500:]
    assert "RESULT" in proc.stdout, proc.stderr[-500:]
    assert "cancel" in proc.stdout.lower()


def test_overlapping_scopes_keep_handler_until_last_exit():
    """Review: A exiting first must not restore the default under active B."""
    before = signal.getsignal(signal.SIGTERM)
    a = sigterm_as_cancellation()
    b = sigterm_as_cancellation()
    a.__enter__()
    b.__enter__()
    try:
        a.__exit__(None, None, None)
        assert signal.getsignal(signal.SIGTERM) is not before
        with pytest.raises(KeyboardInterrupt):
            signal.raise_signal(signal.SIGTERM)
    finally:
        b.__exit__(None, None, None)
    assert signal.getsignal(signal.SIGTERM) is before
