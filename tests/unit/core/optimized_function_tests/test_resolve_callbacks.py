"""Auto-injection rules for the managed-path heartbeat (Traigent#1601).

``_resolve_callbacks`` injects a ``ManagedProgressCallback`` on the managed
(HYBRID) path so a long non-interactive run is not silent. These tests pin the
conditions under which it must NOT inject -- the review of #1601 found the
duplicate guard recognised only two of the four progress-reporting callbacks.
"""

from __future__ import annotations

import pytest


@pytest.mark.parametrize(
    "supplied",
    ["SimpleProgressCallback", "DetailedProgressCallback"],
)
def test_managed_heartbeat_defers_to_any_progress_capable_callback(
    monkeypatch, supplied
):
    """The heartbeat fills a silence; if the user already reports, there is none.

    The guard originally recognised only ``ProgressBarCallback`` and
    ``ManagedProgressCallback``, so supplying either of the other two
    progress-reporting callbacks on the managed path got a heartbeat injected
    alongside it -- two reporters interleaving on the same stream. Measured
    before the fix: ``reporters=2`` for both of these.
    """
    import sys as _sys

    from traigent.config.types import ExecutionMode
    from traigent.core.optimized_function import _resolve_callbacks
    from traigent.utils import callbacks as cb_module

    monkeypatch.setattr(_sys.stdin, "isatty", lambda: False)
    supplied_callback = getattr(cb_module, supplied)()

    resolved = _resolve_callbacks(
        [supplied_callback], None, None, execution_mode=ExecutionMode.HYBRID.value
    )

    progress_capable = (
        cb_module.ProgressBarCallback,
        cb_module.ManagedProgressCallback,
        cb_module.SimpleProgressCallback,
        cb_module.DetailedProgressCallback,
    )
    reporters = [c for c in resolved if isinstance(c, progress_capable)]
    assert len(reporters) == 1, (
        f"expected the supplied {supplied} to be the only progress reporter, got "
        f"{[type(c).__name__ for c in resolved]}"
    )
    assert isinstance(reporters[0], type(supplied_callback))


def test_managed_heartbeat_still_injects_when_nothing_reports(monkeypatch):
    """The widened guard must not disable the feature it guards."""
    import sys as _sys

    from traigent.config.types import ExecutionMode
    from traigent.core.optimized_function import _resolve_callbacks
    from traigent.utils.callbacks import ManagedProgressCallback

    monkeypatch.setattr(_sys.stdin, "isatty", lambda: False)

    resolved = _resolve_callbacks(
        [], None, None, execution_mode=ExecutionMode.HYBRID.value
    )

    assert any(isinstance(c, ManagedProgressCallback) for c in resolved)


def test_heartbeat_still_injects_for_a_silent_simple_progress_callback(monkeypatch):
    """Membership in the guard list is not the question; reporting is.

    ``SimpleProgressCallback(show_details=False)`` emits nothing from
    ``on_trial_complete`` (measured: ``''``). Treating it as a reporter would
    suppress the heartbeat and leave the managed run silent -- the exact problem
    the heartbeat exists to solve, reintroduced by the guard meant to stop
    duplicate output.
    """
    import sys as _sys

    from traigent.config.types import ExecutionMode
    from traigent.core.optimized_function import _resolve_callbacks
    from traigent.utils.callbacks import ManagedProgressCallback, SimpleProgressCallback

    monkeypatch.setattr(_sys.stdin, "isatty", lambda: False)

    resolved = _resolve_callbacks(
        [SimpleProgressCallback(show_details=False)],
        None,
        None,
        execution_mode=ExecutionMode.HYBRID.value,
    )

    assert any(isinstance(c, ManagedProgressCallback) for c in resolved), (
        "a silent SimpleProgressCallback must not suppress the heartbeat"
    )


def test_heartbeat_defers_to_a_reporting_simple_progress_callback(monkeypatch):
    """The same callback WITH details on does report, so it must suppress."""
    import sys as _sys

    from traigent.config.types import ExecutionMode
    from traigent.core.optimized_function import _resolve_callbacks
    from traigent.utils.callbacks import ManagedProgressCallback, SimpleProgressCallback

    monkeypatch.setattr(_sys.stdin, "isatty", lambda: False)

    resolved = _resolve_callbacks(
        [SimpleProgressCallback(show_details=True)],
        None,
        None,
        execution_mode=ExecutionMode.HYBRID.value,
    )

    assert not any(isinstance(c, ManagedProgressCallback) for c in resolved)


def test_no_progress_callback_combination_leaves_a_managed_run_silent(monkeypatch):
    """Exhaustive: for EVERY constructor combination, the user sees something.

    This guard has now been wrong twice in opposite directions -- first too
    narrow (a second reporter was injected alongside Simple/Detailed), then too
    broad (a silent SimpleProgressCallback suppressed the heartbeat). Both were
    found by review rather than by a test, because each fix was a hand-picked
    case. So assert the actual invariant over the whole space instead: if a
    supplied callback emits nothing visible for a trial, the heartbeat MUST be
    injected.

    Known and accepted: Simple(show_details=True) prints on completed trials but
    not failed ones, so it still reports on the normal path. This checks the
    completed-trial case, which is what "does it report at all" turns on.
    """
    import contextlib
    import io
    import logging
    import sys as _sys
    from datetime import UTC, datetime

    from traigent.api.types import TrialResult, TrialStatus
    from traigent.config.types import ExecutionMode
    from traigent.core.optimized_function import _resolve_callbacks
    from traigent.utils.callbacks import (
        DetailedProgressCallback,
        ManagedProgressCallback,
        ProgressBarCallback,
        ProgressInfo,
        SimpleProgressCallback,
    )

    logging.basicConfig(level=logging.WARNING, force=True)
    monkeypatch.setattr(_sys.stdin, "isatty", lambda: False)

    trial = TrialResult(
        trial_id="t",
        config={},
        metrics={"accuracy": 0.9},
        status=TrialStatus.COMPLETED,
        duration=1.0,
        timestamp=datetime.now(UTC),
    )
    progress = ProgressInfo(
        current_trial=1,
        total_trials=3,
        completed_trials=1,
        successful_trials=1,
        failed_trials=0,
        best_score=0.9,
        best_config={},
        elapsed_time=5.0,
        estimated_remaining=10.0,
        current_algorithm="grid",
    )

    factories = [
        (
            f"Simple(output={out!r}, show_details={detail})",
            lambda o=out, d=detail: SimpleProgressCallback(output=o, show_details=d),
        )
        for out in ("print", "log")
        for detail in (True, False)
    ]
    factories += [
        (
            f"Detailed(config={cfg}, metrics={met})",
            lambda c=cfg, m=met: DetailedProgressCallback(
                show_config_details=c, show_metrics=m
            ),
        )
        for cfg in (True, False)
        for met in (True, False)
    ]
    factories += [
        ("ProgressBar()", ProgressBarCallback),
        ("Managed()", ManagedProgressCallback),
    ]

    silent_runs = []
    for label, make in factories:
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
            make().on_trial_complete(trial, progress)
        user_sees_something = bool(buffer.getvalue().strip())

        resolved = _resolve_callbacks(
            [make()], None, None, execution_mode=ExecutionMode.HYBRID.value
        )
        heartbeat = any(isinstance(c, ManagedProgressCallback) for c in resolved)

        if not (user_sees_something or heartbeat):
            silent_runs.append(label)

    assert not silent_runs, (
        "these callbacks emit nothing per trial AND suppress the heartbeat, so a "
        f"managed run would be silent: {silent_runs}"
    )
