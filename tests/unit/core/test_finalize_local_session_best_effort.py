"""``finalize_local_session`` must honour its own "never break the run" contract.

Traigent#2048.

The docstring says *"Best-effort: storage failures never break the run"*, but only
the final ``storage.finalize_session(...)`` call was inside the ``try``. The
prologue — ``_egress_disabled()``, ``_remote_session_completed()``,
``_local_storage()`` — ran unguarded, so a raise from any of them propagated out
of a method whose contract says it cannot.

That matters more than a normal leak because this runs on the FAILURE path: an
incidental bookkeeping error would replace the optimization error the caller was
about to receive, which is the same "the diagnostic destroys the diagnosis" shape
as #2050 and #2029.
"""

from __future__ import annotations

import logging

import pytest

from traigent.core import backend_session_manager as bsm
from traigent.core.backend_session_manager import BackendSessionManager
from traigent.api.types import OptimizationStatus


class _Boom(Exception):
    """Distinguishable from any incidental error the method might raise."""


def _manager() -> BackendSessionManager:
    """A manager with the tracking flags set so the prologue is actually reached."""
    manager = BackendSessionManager.__new__(BackendSessionManager)
    manager._backend_client = object()
    manager._backend_tracking_enabled = True
    return manager


@pytest.mark.parametrize(
    "attribute",
    ["_egress_disabled", "_remote_session_completed", "_local_storage"],
)
def test_a_raising_prologue_step_does_not_escape(attribute, monkeypatch):
    """Each of the three previously-unguarded calls, one at a time."""
    manager = _manager()

    def _raise(*_args, **_kwargs):
        raise _Boom("prologue exploded")

    # The prologue is an `and` chain, so the stubs needed to REACH each target
    # differ. Getting this wrong makes the test vacuous: the raise never runs
    # and the method trivially "does not escape".
    #
    #   remote_finalize_owns_mirror = client and tracking
    #                                 and not _egress_disabled()
    #                                 and not _remote_session_completed(sid)
    #
    reachable = {
        # called first; nothing to arrange
        "_egress_disabled": {},
        # needs the chain to CONTINUE past egress
        "_remote_session_completed": {"_egress_disabled": lambda: False},
        # needs the chain to be FALSE so we fall through to storage
        "_local_storage": {
            "_egress_disabled": lambda: True,
            "_remote_session_completed": lambda _sid: False,
        },
    }[attribute]
    for name, stub in reachable.items():
        monkeypatch.setattr(manager, name, stub, raising=False)
    monkeypatch.setattr(manager, attribute, _raise, raising=False)

    # Captured on the module logger directly: the SDK's logger does not
    # propagate to root, so caplog alone sees nothing.
    records: list[logging.LogRecord] = []

    class _Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    handler = _Capture()
    bsm.logger.addHandler(handler)
    try:
        # Must not raise. Before the fix this propagated _Boom.
        manager.finalize_local_session("sess-1", OptimizationStatus.FAILED)
    finally:
        bsm.logger.removeHandler(handler)

    # The failure is swallowed for the CALLER, but never silently: it is still
    # reported, with the session id, so the run is diagnosable.
    assert any("sess-1" in record.getMessage() for record in records)


def test_a_storage_write_failure_still_does_not_escape(monkeypatch):
    """The one path that was already guarded -- pinned so the refactor kept it."""
    manager = _manager()

    class _Storage:
        def finalize_session(self, *_args, **_kwargs):
            raise _Boom("write exploded")

    monkeypatch.setattr(manager, "_egress_disabled", lambda: True, raising=False)
    monkeypatch.setattr(
        manager, "_remote_session_completed", lambda _sid: False, raising=False
    )
    monkeypatch.setattr(manager, "_local_storage", lambda: _Storage(), raising=False)

    manager.finalize_local_session("sess-2", OptimizationStatus.COMPLETED)


def test_an_exception_that_cannot_render_itself_still_does_not_escape(monkeypatch):
    """The #2050/#2029 shape, on this path.

    The handler renders the caught exception into its warning. If that render is
    the thing that raises, the guard would be defeated by its own diagnostic.
    """

    class _Unprintable(Exception):
        def __str__(self) -> str:
            raise RuntimeError("cannot render")

    manager = _manager()

    def _raise(*_args, **_kwargs):
        raise _Unprintable()

    monkeypatch.setattr(manager, "_local_storage", _raise, raising=False)
    monkeypatch.setattr(manager, "_egress_disabled", lambda: True, raising=False)
    monkeypatch.setattr(
        manager, "_remote_session_completed", lambda _sid: False, raising=False
    )

    manager.finalize_local_session("sess-3", OptimizationStatus.FAILED)


def test_a_keyboard_interrupt_is_still_allowed_through(monkeypatch):
    """Boundary: best-effort bookkeeping must not swallow a stop request."""
    manager = _manager()

    def _interrupt(*_args, **_kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(manager, "_local_storage", _interrupt, raising=False)
    monkeypatch.setattr(manager, "_egress_disabled", lambda: True, raising=False)
    monkeypatch.setattr(
        manager, "_remote_session_completed", lambda _sid: False, raising=False
    )

    with pytest.raises(KeyboardInterrupt):
        manager.finalize_local_session("sess-4", OptimizationStatus.FAILED)


def test_an_empty_session_id_is_still_a_no_op(monkeypatch):
    """The early return must not have been swallowed into the try."""
    manager = _manager()

    def _should_not_run(*_args, **_kwargs):
        raise AssertionError("prologue ran for an empty session id")

    monkeypatch.setattr(manager, "_egress_disabled", _should_not_run, raising=False)

    manager.finalize_local_session("", OptimizationStatus.FAILED)
    manager.finalize_local_session(None, OptimizationStatus.FAILED)
