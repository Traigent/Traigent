"""Treat SIGTERM and SIGHUP like Ctrl+C for the duration of an optimization run.

A run killed by SIGTERM (CI, Kubernetes; #2667) or SIGHUP (closed terminal;
#3259) used to die without finalizing its session, leaving it ``pending``.
Raising ``KeyboardInterrupt`` from the handler sends it down the existing
user-cancel path, which finalizes the session as cancelled with a partial result.

Safety: installed only from the main thread, only for signals that still have
the default disposition (each signal is decided independently; a host-installed
handler or SIG_IGN is never overridden, nor chained), SIGHUP is skipped where
the platform lacks it (Windows), and previous handlers are restored on exit.
"""

from __future__ import annotations

import signal
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from types import FrameType
from typing import Any


def _raise_cancel(signum: int, frame: FrameType | None) -> None:
    raise KeyboardInterrupt


_CANCEL_SIGNAL_NAMES = ("SIGTERM", "SIGHUP")

_LOCK = threading.Lock()
_active_scopes = 0
_saved_handlers: dict[int, Any] = {}


def _cancel_signals() -> list[int]:
    """Cancellation signals this platform defines (SIGHUP is absent on Windows)."""
    found = (getattr(signal, name, None) for name in _CANCEL_SIGNAL_NAMES)
    return [sig for sig in found if sig is not None]


def _restore_saved_handlers() -> None:
    """Restore and forget every handler this module installed (caller holds _LOCK)."""
    for sig, handler in list(_saved_handlers.items()):
        try:
            signal.signal(sig, handler)
        except (ValueError, OSError):
            pass
    _saved_handlers.clear()


@contextmanager
def sigterm_as_cancellation() -> Iterator[None]:
    """Context manager translating SIGTERM and SIGHUP into ``KeyboardInterrupt``.

    Re-entrant: overlapping scopes share the installed handlers, which are
    installed on the first enter and restored on the last exit.
    """
    global _active_scopes
    signals = _cancel_signals()
    if not signals or threading.current_thread() is not threading.main_thread():
        yield
        return
    counted = False
    joined = False
    with _LOCK:
        try:
            if _active_scopes > 0:
                joined = True
                _active_scopes += 1
                counted = True
            else:
                for sig in signals:
                    try:
                        previous = signal.getsignal(sig)
                        if previous == signal.SIG_DFL:
                            # Record first: an interrupt right after the install
                            # must still find this signal when rolling back.
                            _saved_handlers[sig] = previous
                            signal.signal(sig, _raise_cancel)
                    except (ValueError, OSError):
                        continue
                if _saved_handlers:
                    _active_scopes = 1
                    counted = True
        except (ValueError, OSError):
            counted = False
        except BaseException:
            # E.g. KeyboardInterrupt from an already-installed handler firing
            # mid-install: undo this call's installs so none leaks, then re-raise.
            if joined:
                # Handlers belong to the still-active outer scope: keep them,
                # only undo our own count.
                if counted:
                    _active_scopes -= 1
            else:
                _restore_saved_handlers()
            raise
    try:
        yield
    finally:
        if counted:
            with _LOCK:
                _active_scopes -= 1
                if _active_scopes == 0:
                    _restore_saved_handlers()
