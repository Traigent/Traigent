"""Treat SIGTERM like Ctrl+C for the duration of an optimization run (#2667).

A run killed by SIGTERM (CI, Kubernetes) used to die without finalizing its
session, leaving it ``pending``. Raising ``KeyboardInterrupt`` from the handler
sends it down the existing user-cancel path, which finalizes the session as
cancelled with a partial result.

Safety: installed only from the main thread, only when SIGTERM still has the
default disposition (a host-installed handler or SIG_IGN is never overridden,
nor chained), and the previous handler is restored on exit.
"""

from __future__ import annotations

import signal
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from types import FrameType


def _raise_cancel(signum: int, frame: FrameType | None) -> None:
    raise KeyboardInterrupt


_LOCK = threading.Lock()
_active_scopes = 0
_saved_handler: object = signal.SIG_DFL


@contextmanager
def sigterm_as_cancellation() -> Iterator[None]:
    """Context manager translating SIGTERM into ``KeyboardInterrupt``.

    Re-entrant: overlapping scopes share one installed handler, which is
    installed on the first enter and restored on the last exit.
    """
    global _active_scopes, _saved_handler
    sigterm = getattr(signal, "SIGTERM", None)
    if sigterm is None or threading.current_thread() is not threading.main_thread():
        yield
        return
    counted = False
    with _LOCK:
        try:
            if _active_scopes > 0:
                _active_scopes += 1
                counted = True
            else:
                previous = signal.getsignal(sigterm)
                if previous == signal.SIG_DFL:
                    signal.signal(sigterm, _raise_cancel)
                    _saved_handler = previous
                    _active_scopes = 1
                    counted = True
        except (ValueError, OSError):
            counted = False
    try:
        yield
    finally:
        if counted:
            with _LOCK:
                _active_scopes -= 1
                if _active_scopes == 0:
                    try:
                        signal.signal(sigterm, _saved_handler)  # type: ignore[arg-type]
                    except (ValueError, OSError):
                        pass
