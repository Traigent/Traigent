"""Pickle/copy support for exceptions whose ``__init__`` differs from ``args``.

``BaseException.__reduce__`` rebuilds an exception as ``cls(*self.args)``.
Many Traigent exceptions format a message and call ``super().__init__(message)``
while requiring other positional or keyword-only arguments, so unpickling (for
example when a worker raises inside ``ProcessPoolExecutor``, joblib, Celery or
dask) raised ``TypeError`` or re-wrapped the message.

``PicklableExceptionMixin`` records the exact constructor call in ``__new__``
and replays it on unpickle/copy, then restores instance attributes. Put it
before the exception base: ``class MyError(PicklableExceptionMixin, Exception)``.
"""

from __future__ import annotations

import copy
from collections.abc import MutableMapping, MutableSequence, MutableSet
from typing import Any

_CTOR_ARGS = "_ctor_args"
_CTOR_KWARGS = "_ctor_kwargs"


def _snapshot(value: Any) -> Any:
    """Shallow-copy mutable containers so later caller mutation cannot skew a replay."""
    if isinstance(value, (MutableSequence, MutableMapping, MutableSet, bytearray)):
        try:
            return copy.copy(value)
        except Exception:  # noqa: BLE001 - uncopyable input: keep the reference
            return value
    return value


def _rebuild_exception(
    cls: type[BaseException], args: tuple[Any, ...], kwargs: dict[str, Any]
) -> BaseException:
    return cls(*args, **kwargs)


class PicklableExceptionMixin:
    """Make an exception round-trip through pickle/copy via its constructor call."""

    def __new__(cls, *args: Any, **kwargs: Any):
        # BaseException.__new__ is next in the MRO when this mixin precedes it.
        instance = super().__new__(cls, *args)  # type: ignore[call-arg]
        instance.__dict__[_CTOR_ARGS] = tuple(_snapshot(a) for a in args)
        instance.__dict__[_CTOR_KWARGS] = {k: _snapshot(v) for k, v in kwargs.items()}
        return instance

    def __reduce__(self) -> tuple[Any, ...]:
        state = {
            key: value
            for key, value in self.__dict__.items()
            if key not in (_CTOR_ARGS, _CTOR_KWARGS)
        }
        return (
            _rebuild_exception,
            (
                type(self),
                self.__dict__.get(_CTOR_ARGS, self.args),  # type: ignore[attr-defined]
                self.__dict__.get(_CTOR_KWARGS, {}),
            ),
            state or None,
        )
