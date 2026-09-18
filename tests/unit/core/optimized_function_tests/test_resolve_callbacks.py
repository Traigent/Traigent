"""Auto-injection rules for the managed-path heartbeat (Traigent#1601).

``_resolve_callbacks`` injects a ``ManagedProgressCallback`` on the managed
(HYBRID) path so a long non-interactive run is not silent. These tests pin the
conditions under which it must NOT inject -- the review of #1601 found the
duplicate guard recognised only two of the four progress-reporting callbacks.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime

import pytest


@pytest.mark.parametrize(
    "supplied",
    ["DetailedProgressCallback"],
)
def test_managed_heartbeat_defers_to_any_progress_capable_callback(
    monkeypatch, supplied
):
    """The heartbeat fills a silence; if the user already reports, there is none.

    The guard originally recognised only ``ProgressBarCallback`` and
    ``ManagedProgressCallback``, so supplying a genuinely reporting callback on
    the managed path got a heartbeat injected alongside it -- two reporters
    interleaving on the same stream. Measured before the fix: ``reporters=2``.

    ``SimpleProgressCallback`` was dropped from this list after the #2361
    review: unlike ``DetailedProgressCallback``, whose per-trial line is
    unconditional, it emits nothing on a failed trial, so it does not remove
    the silence and must not suppress the heartbeat. See
    ``test_heartbeat_does_not_defer_even_to_a_detailed_simple_progress_callback``.
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


def test_heartbeat_does_not_defer_even_to_a_detailed_simple_progress_callback(
    monkeypatch,
):
    """``show_details=True`` is still not enough, and this test used to say it was.

    It previously asserted the opposite -- that ``SimpleProgressCallback`` with
    details on "does report, so it must suppress". Driving its
    ``on_trial_complete`` shows that is only true for a *successful* trial with
    a score: the method is guarded on ``trial.status == "completed"``, so a
    failed trial prints nothing, and a completed trial with no recognized score
    and no best score yet prints nothing either. Both measurements are pinned
    just below.

    A run whose trials fail is exactly when a user needs to see that anything is
    still happening, so this callback must not buy silence. The cost of the
    corrected answer is one duplicated line when trials do succeed.
    """
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

    assert any(isinstance(c, ManagedProgressCallback) for c in resolved), (
        "SimpleProgressCallback is silent on failed trials, so it must not "
        "suppress the heartbeat"
    )


def test_no_progress_callback_combination_leaves_a_managed_run_silent(monkeypatch):
    """Exhaustive: for EVERY constructor combination, the user sees something.

    This guard has now been wrong twice in opposite directions -- first too
    narrow (a second reporter was injected alongside Simple/Detailed), then too
    broad (a silent SimpleProgressCallback suppressed the heartbeat). Both were
    found by review rather than by a test, because each fix was a hand-picked
    case. So assert the actual invariant over the whole space instead: if a
    supplied callback emits nothing visible for a trial, the heartbeat MUST be
    injected.

    This case covers the COMPLETED-trial half of the space. The failed-trial
    half -- which this docstring used to record as "known and accepted" -- is no
    longer accepted: it is measured in
    ``test_suppressing_the_heartbeat_implies_the_callback_actually_reports``,
    and it is the reason SimpleProgressCallback no longer suppresses the
    heartbeat at all.
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


# ---------------------------------------------------------------------------
# The invariant, measured rather than asserted about types.
#
# Review of #2361 found the previous tests pinned which callback types get a
# heartbeat without ever invoking a completion or looking at what was emitted,
# so a callback that is silent in practice could be classified as a reporter
# and every test would stay green. These drive on_trial_complete for real, on
# BOTH a successful and a failed trial, and check the only property that
# matters:
#
#     the heartbeat is suppressed  =>  the supplied callback really did emit
#                                      on every trial outcome
#
# The two directions are not symmetric. Suppressing wrongly makes a long
# managed run print nothing, which is the bug this feature exists to fix.
# Injecting wrongly costs one duplicated line.
# ---------------------------------------------------------------------------


def _trial(*, succeeded: bool, metrics=None):
    from traigent.api.types import TrialResult, TrialStatus

    return TrialResult(
        trial_id="t1",
        config={"temperature": 0.0},
        metrics=metrics
        if metrics is not None
        else ({"accuracy": 0.9} if succeeded else {}),
        status=TrialStatus.COMPLETED if succeeded else TrialStatus.FAILED,
        duration=1.0,
        timestamp=datetime.now(UTC),
        error_message=None if succeeded else "boom",
        score=0.9 if succeeded else None,
    )


def _progress(*, best_score, failed_trials):
    from traigent.utils.callbacks import ProgressInfo

    return ProgressInfo(
        current_trial=1,
        total_trials=4,
        completed_trials=1,
        successful_trials=0 if failed_trials else 1,
        failed_trials=failed_trials,
        best_score=best_score,
        best_config=None,
        elapsed_time=12.0,
        estimated_remaining=None,
        current_algorithm="random",
    )


def _emits(callback, *, succeeded, capsys, caplog) -> bool:
    """Drive one trial completion and report whether ANYTHING reached the user."""
    capsys.readouterr()  # drop anything buffered from a previous case
    callback.total_trials = 4
    callback.current_trial = 1
    progress = _progress(
        best_score=0.9 if succeeded else None,
        failed_trials=0 if succeeded else 1,
    )
    with caplog.at_level(logging.INFO):
        caplog.clear()
        callback.on_trial_complete(_trial(succeeded=succeeded), progress)
    captured = capsys.readouterr()
    printed = bool((captured.out + captured.err).strip())
    # logger.info output only counts if the user would actually see it; the
    # CLI configures WARNING by default (traigent/cli/main.py), so an INFO
    # record is captured here but invisible in a real run. Deliberately NOT
    # counted as emitting.
    return printed


def _heartbeat_injected(supplied, monkeypatch) -> bool:
    import sys as _sys

    from traigent.config.types import ExecutionMode
    from traigent.core.optimized_function import _resolve_callbacks
    from traigent.utils.callbacks import ManagedProgressCallback

    monkeypatch.setattr(_sys.stdin, "isatty", lambda: False)
    resolved = _resolve_callbacks(
        [supplied], None, None, execution_mode=ExecutionMode.HYBRID.value
    )
    return any(
        c is not supplied and isinstance(c, ManagedProgressCallback) for c in resolved
    )


def _candidates():
    from traigent.utils import callbacks as cb

    return [
        ("ManagedProgressCallback", cb.ManagedProgressCallback()),
        ("DetailedProgressCallback", cb.DetailedProgressCallback()),
        (
            "DetailedProgressCallback(no metrics)",
            cb.DetailedProgressCallback(show_metrics=False),
        ),
        ("SimpleProgressCallback(default)", cb.SimpleProgressCallback()),
        (
            "SimpleProgressCallback(show_details=False)",
            cb.SimpleProgressCallback(show_details=False),
        ),
        ("SimpleProgressCallback(output=log)", cb.SimpleProgressCallback(output="log")),
    ]


@pytest.mark.parametrize(
    "label,callback", _candidates(), ids=lambda v: v if isinstance(v, str) else ""
)
def test_suppressing_the_heartbeat_implies_the_callback_actually_reports(
    label, callback, monkeypatch, capsys, caplog
):
    """The load-bearing invariant: never go quiet on the strength of a guess."""
    emits_on_success = _emits(callback, succeeded=True, capsys=capsys, caplog=caplog)
    emits_on_failure = _emits(callback, succeeded=False, capsys=capsys, caplog=caplog)
    suppressed = not _heartbeat_injected(callback, monkeypatch)

    if suppressed:
        assert emits_on_success and emits_on_failure, (
            f"{label} suppresses the managed heartbeat but does not emit on every "
            f"trial outcome (success={emits_on_success}, failure={emits_on_failure}). "
            "A managed run using it would print nothing for those trials."
        )


def test_simple_progress_callback_is_silent_on_a_failed_trial(capsys, caplog):
    """The measurement the classification now rests on.

    ``SimpleProgressCallback.on_trial_complete`` guards its output on
    ``trial.status == "completed"``, so a run whose trials all fail prints
    nothing -- which is exactly when a heartbeat matters most. If this ever
    becomes untrue, the exclusion above should be revisited.
    """
    from traigent.utils.callbacks import SimpleProgressCallback

    callback = SimpleProgressCallback()
    assert _emits(callback, succeeded=True, capsys=capsys, caplog=caplog) is True
    assert _emits(callback, succeeded=False, capsys=capsys, caplog=caplog) is False


def test_simple_progress_callback_is_silent_without_a_score(capsys, caplog):
    """A completed trial with no recognized score and no best score yet."""
    from traigent.api.types import TrialResult, TrialStatus
    from traigent.utils.callbacks import SimpleProgressCallback

    callback = SimpleProgressCallback()
    callback.total_trials = 4
    callback.current_trial = 1
    capsys.readouterr()
    callback.on_trial_complete(
        TrialResult(
            trial_id="t1",
            config={},
            metrics={"unrecognised_metric": 1.0},
            status=TrialStatus.COMPLETED,
            duration=1.0,
            timestamp=datetime.now(UTC),
            score=None,
        ),
        _progress(best_score=None, failed_trials=0),
    )
    assert not capsys.readouterr().out.strip()


def test_a_silent_subclass_cannot_suppress_the_heartbeat(monkeypatch):
    """Exact-type classification, stated as behaviour.

    A subclass can override ``on_trial_complete`` to emit nothing, which is
    invisible to any type check. It must therefore not be treated as a
    reporter.
    """
    from traigent.utils.callbacks import ManagedProgressCallback

    class MuteManaged(ManagedProgressCallback):
        def on_trial_complete(self, trial, progress) -> None:  # noqa: D102
            return None

    assert _heartbeat_injected(MuteManaged(), monkeypatch), (
        "a subclass that emits nothing suppressed the heartbeat"
    )


def test_a_callback_whose_show_details_raises_does_not_break_resolution(monkeypatch):
    """Regression: the predicate used to read attributes off the user's object.

    ``getattr(callback, "show_details", True)`` runs a property, and a property
    that raises took down callback resolution -- turning a cosmetic
    progress-reporting decision into a failed ``.optimize()`` call.
    """
    from traigent.utils.callbacks import SimpleProgressCallback

    class Exploding(SimpleProgressCallback):
        @property
        def show_details(self):  # type: ignore[override]
            raise RuntimeError("must never be read during callback resolution")

        @show_details.setter
        def show_details(self, value):
            pass

    # Must not raise.
    assert _heartbeat_injected(Exploding(), monkeypatch) is True
