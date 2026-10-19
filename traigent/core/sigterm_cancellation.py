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


@contextmanager
def sigterm_as_cancellation() -> Iterator[None]:
    """Context manager translating SIGTERM into ``KeyboardInterrupt``."""
    sigterm = getattr(signal, "SIGTERM", None)
    if sigterm is None or threading.current_thread() is not threading.main_thread():
        yield
        return
    try:
        previous = signal.getsignal(sigterm)
        if previous != signal.SIG_DFL:
            yield
            return
        signal.signal(sigterm, _raise_cancel)
    except (ValueError, OSError):
        yield
        return
    try:
        yield
    finally:
        try:
            signal.signal(sigterm, previous)
        except (ValueError, OSError):
            pass
