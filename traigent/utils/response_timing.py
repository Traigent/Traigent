"""Side-channel for per-response latency that never mutates the response.

Pydantic provider responses (LiteLLM's ``ModelResponse``) are serialised on a
background thread by LiteLLM's success handler (``model_dump`` in
``pydantic-core``) while the caller already holds the object. Assigning a new
attribute to such a response (``response.response_time_ms = ...``) inserts into
the model's extras dict and, if it lands mid-serialisation, pydantic-core's
Rust iterator panics with ``dictionary changed size during iteration``
(Traigent#1812). Latency for those responses is therefore recorded here,
keyed by object identity, instead of on the response.
"""

from __future__ import annotations

import threading
import weakref
from typing import Any

_LOCK = threading.Lock()
# id(response) -> (weakref to response, latency in ms); the weakref callback
# drops the entry when the response is collected, so ids are never reused.
_TIMES: dict[int, tuple[weakref.ref[Any], float]] = {}


def _is_pydantic_model(response: Any) -> bool:
    return hasattr(type(response), "model_dump") and hasattr(
        type(response), "__pydantic_serializer__"
    )


def record_response_time_ms(response: Any, response_time_ms: float) -> None:
    """Attach latency to ``response`` without mutating a pydantic response."""
    if _is_pydantic_model(response):
        key = id(response)

        def _drop(_ref: Any, key: int = key) -> None:
            with _LOCK:
                _TIMES.pop(key, None)

        try:
            ref = weakref.ref(response, _drop)
        except TypeError:
            return  # cannot track safely; latency stays unknown, never guessed
        with _LOCK:
            _TIMES[key] = (ref, float(response_time_ms))
        return
    try:
        response.response_time_ms = response_time_ms
    except Exception:  # noqa: BLE001 - timing is best-effort metadata
        pass


def recorded_response_time_ms(response: Any) -> float | None:
    """Latency recorded by :func:`record_response_time_ms`, else ``None``."""
    with _LOCK:
        entry = _TIMES.get(id(response))
    if entry is None or entry[0]() is not response:
        return None
    return entry[1]
