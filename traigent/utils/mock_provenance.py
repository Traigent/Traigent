"""Run-scoped record of where LLM responses were actually served (#2516).

Mock mode is process state that an agent or a callback can flip mid-run, so a
flag sampled at trial dispatch or result build can contradict what happened.
Every mock interception point serves its response through
``MockAdapter.get_mock_response``, which calls :func:`note_mock_response`; the
LiteLLM wrappers call :func:`note_real_call` when a call goes down the
non-mock path. A recorder is bound to one ``optimize()`` run.

Scope: a ``ContextVar`` binds the run for the task and anything that copies its
context (``asyncio`` tasks, ``asyncio.to_thread``). A thread that does not copy
the context (a plain ``ThreadPoolExecutor``) falls back to every active
recorder: concurrent runs may then over-attribute, and the only effect is the
safe direction for mock (labelled) and a spurious "mixed" note.
"""

from __future__ import annotations

import threading
from contextvars import ContextVar, Token

_current: ContextVar[MockProvenanceRecorder | None] = ContextVar(
    "traigent_mock_provenance_recorder", default=None
)
_active: list[MockProvenanceRecorder] = []
_active_lock = threading.Lock()


class MockProvenanceRecorder:
    """Counts mock responses served and non-mock LLM calls observed in one run."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.mock_responses_served = 0
        self.real_calls_observed = 0
        #: Provider hosts the mock transport guard blocked during this run (#2812).
        self.blocked_hosts: dict[str, int] = {}
        self._token: Token[MockProvenanceRecorder | None] | None = None

    def _bump(self, mock: bool) -> None:
        with self._lock:
            if mock:
                self.mock_responses_served += 1
            else:
                self.real_calls_observed += 1

    def _bump_blocked(self, host: str) -> None:
        with self._lock:
            self.blocked_hosts[host] = self.blocked_hosts.get(host, 0) + 1

    def activate(self) -> None:
        self._token = _current.set(self)
        with _active_lock:
            _active.append(self)

    def deactivate(self) -> None:
        with _active_lock:
            if self in _active:
                _active.remove(self)
        if self._token is not None:
            try:
                _current.reset(self._token)
            except ValueError:  # reset from a different context; leave it bound
                _current.set(None)
            self._token = None


def _targets() -> list[MockProvenanceRecorder]:
    bound = _current.get()
    if bound is not None:
        return [bound]
    with _active_lock:
        return list(_active)


def note_mock_response() -> None:
    """A mock response was served to a caller."""
    for recorder in _targets():
        recorder._bump(True)


def note_real_call() -> None:
    """An LLM call took the non-mock path (it may still be simulated downstream)."""
    for recorder in _targets():
        recorder._bump(False)


def note_mock_blocked(host: str) -> None:
    """The mock transport guard blocked a raw call to ``host`` (#2812)."""
    for recorder in _targets():
        recorder._bump_blocked(host)
