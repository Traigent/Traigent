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


@pytest.mark.skipif(not hasattr(signal, "SIGHUP"), reason="no SIGHUP on this platform")
def test_sighup_raises_keyboard_interrupt_and_restores_handler():
    """#3259: a closed terminal (SIGHUP) cancels like SIGTERM.

    Runs in a subprocess so a missing handler cannot kill the pytest process.
    """
    import subprocess
    import sys

    code = (
        "import signal\n"
        "from traigent.core.sigterm_cancellation import sigterm_as_cancellation\n"
        "b_h, b_t = signal.getsignal(signal.SIGHUP), signal.getsignal(signal.SIGTERM)\n"
        "with sigterm_as_cancellation():\n"
        "    try:\n"
        "        signal.raise_signal(signal.SIGHUP)\n"
        "    except KeyboardInterrupt:\n"
        "        print('CANCELLED')\n"
        "assert signal.getsignal(signal.SIGHUP) is b_h\n"
        "assert signal.getsignal(signal.SIGTERM) is b_t\n"
        "print('RESTORED')\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=60
    )
    assert proc.returncode == 0, (proc.returncode, proc.stderr[-500:])
    assert "CANCELLED" in proc.stdout and "RESTORED" in proc.stdout


@pytest.mark.skipif(not hasattr(signal, "SIGHUP"), reason="no SIGHUP on this platform")
def test_failed_install_midway_leaks_no_handler(monkeypatch):
    """Review: a raise during the second install must roll back the first."""
    import traigent.core.sigterm_cancellation as mod

    before_term = signal.getsignal(signal.SIGTERM)
    before_hup = signal.getsignal(signal.SIGHUP)
    real = signal.signal

    def flaky(sig, handler):
        if sig == signal.SIGHUP and handler is mod._raise_cancel:
            raise KeyboardInterrupt
        return real(sig, handler)

    monkeypatch.setattr(mod.signal, "signal", flaky)
    try:
        with pytest.raises(KeyboardInterrupt):
            with mod.sigterm_as_cancellation():
                pytest.fail("body must not run")
        monkeypatch.undo()
        assert signal.getsignal(signal.SIGTERM) is before_term
        assert signal.getsignal(signal.SIGHUP) is before_hup
        assert mod._active_scopes == 0
        assert mod._saved_handlers == {}
    finally:
        monkeypatch.undo()
        real(signal.SIGTERM, before_term)
        real(signal.SIGHUP, before_hup)


@pytest.mark.skipif(not hasattr(signal, "SIGHUP"), reason="no SIGHUP on this platform")
def test_host_sighup_handler_is_never_overridden_but_sigterm_still_is():
    calls: list[int] = []

    def host(signum, frame):
        calls.append(signum)

    prev = signal.signal(signal.SIGHUP, host)
    try:
        with sigterm_as_cancellation():
            assert signal.getsignal(signal.SIGHUP) is host
            signal.raise_signal(signal.SIGHUP)
            with pytest.raises(KeyboardInterrupt):
                signal.raise_signal(signal.SIGTERM)
        assert calls == [signal.SIGHUP]
        assert signal.getsignal(signal.SIGHUP) is host
    finally:
        signal.signal(signal.SIGHUP, prev)


def test_scope_is_a_noop_when_sighup_is_missing(monkeypatch):
    """Windows has no SIGHUP: SIGTERM handling must still work, no AttributeError."""
    monkeypatch.delattr(signal, "SIGHUP", raising=False)
    before = signal.getsignal(signal.SIGTERM)
    with sigterm_as_cancellation():
        with pytest.raises(KeyboardInterrupt):
            signal.raise_signal(signal.SIGTERM)
    assert signal.getsignal(signal.SIGTERM) is before
