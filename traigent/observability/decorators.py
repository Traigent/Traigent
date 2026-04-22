"""Decorator and context-manager helpers for observability instrumentation."""

from __future__ import annotations

import contextvars
import functools
import inspect
import threading
from collections.abc import Callable
from datetime import datetime
from typing import Any, Literal

from traigent.observability.client import ObservabilityClient
from traigent.observability.dtos import ObservationType, utc_now
from traigent.utils.logging import get_logger

logger = get_logger(__name__)

_current_client: contextvars.ContextVar[ObservabilityClient | None] = (
    contextvars.ContextVar(
        "traigent_observability_client",
        default=None,
    )
)
_current_trace_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "traigent_observability_trace_id",
    default=None,
)
_current_observation_stack: contextvars.ContextVar[tuple[str, ...]] = (
    contextvars.ContextVar(
        "traigent_observability_observation_stack",
        default=(),
    )
)
_default_client: ObservabilityClient | None = None
_default_client_lock = threading.Lock()


def _redacted_input_payload() -> dict[str, bool]:
    return {"redacted": True}


def get_default_observability_client() -> ObservabilityClient:
    global _default_client
    if _default_client is None:
        with _default_client_lock:
            if _default_client is None:
                _default_client = ObservabilityClient()
    return _default_client


def set_default_observability_client(client: ObservabilityClient) -> None:
    global _default_client
    with _default_client_lock:
        _default_client = client


class ObserveContext:
    """Context manager used by the `observe` helper."""

    def __init__(
        self,
        *,
        name: str,
        client: ObservabilityClient | None = None,
        observation_type: ObservationType | str = ObservationType.SPAN,
        input_data: Any = None,
        metadata: dict[str, Any] | None = None,
        environment: str | None = None,
        release: str | None = None,
        tags: list[str] | None = None,
        redact_input: bool = False,
    ) -> None:
        self.name = name
        self.client = client
        self.observation_type = (
            observation_type
            if isinstance(observation_type, ObservationType)
            else ObservationType(observation_type)
        )
        self.input_data = input_data
        self.metadata = dict(metadata or {})
        self.environment = environment
        self.release = release
        self.tags = list(tags or [])
        self.redact_input = redact_input

        self._started_at: datetime | None = None
        self._trace_id: str | None = None
        self._observation_id: str | None = None
        self._created_trace = False
        self._trace_token: contextvars.Token[str | None] | None = None
        self._stack_token: contextvars.Token[tuple[str, ...]] | None = None
        self._client_token: contextvars.Token[ObservabilityClient | None] | None = None
        self._finished = False
        self._result: Any = None

    def __enter__(self) -> ObserveContext:
        self._started_at = utc_now()
        client = (
            self.client or _current_client.get() or get_default_observability_client()
        )
        self._client_token = _current_client.set(client)

        trace_id = _current_trace_id.get()
        if trace_id is None:
            trace_id = client.start_trace(
                self.name,
                environment=self.environment,
                release=self.release,
                tags=self.tags,
                metadata={"source": "observe"},
                started_at=self._started_at,
                input_data=self.input_data,
            )
            self._created_trace = True
            self._trace_token = _current_trace_id.set(trace_id)

        stack = _current_observation_stack.get()
        parent_observation_id = stack[-1] if stack else None
        self._observation_id = client.record_observation(
            trace_id,
            observation_id=self._observation_id,
            name=self.name,
            observation_type=self.observation_type,
            parent_observation_id=parent_observation_id,
            status="running",
            started_at=self._started_at,
            input_data=self.input_data,
            metadata=self.metadata,
        )
        self._trace_id = trace_id
        self._stack_token = _current_observation_stack.set(
            stack + (self._observation_id,)
        )
        return self

    def __exit__(self, exc_type, exc, exc_tb) -> Literal[False]:
        self._finish(result=self._result, error=exc)
        return False

    async def __aenter__(self) -> ObserveContext:
        return self.__enter__()

    async def __aexit__(self, exc_type, exc, exc_tb) -> Literal[False]:
        self._finish(result=self._result, error=exc)
        return False

    def _finish(self, *, result: Any = None, error: Exception | None = None) -> None:
        if self._finished:
            return
        self._finished = True
        client = (
            self.client or _current_client.get() or get_default_observability_client()
        )
        ended_at = utc_now()
        status = "failed" if error is not None else "completed"
        metadata = dict(self.metadata)
        if error is not None:
            metadata["error_type"] = type(error).__name__
            metadata["error_message"] = str(error)

        if self._trace_id and self._observation_id:
            client.record_observation(
                self._trace_id,
                observation_id=self._observation_id,
                name=self.name,
                observation_type=self.observation_type,
                status=status,
                ended_at=ended_at,
                input_data=self.input_data,
                output_data=result,
                metadata=metadata,
            )
            if self._created_trace:
                client.end_trace(
                    self._trace_id,
                    status=status,
                    output_data=result,
                    ended_at=ended_at,
                )

        if self._stack_token is not None:
            _current_observation_stack.reset(self._stack_token)
        if self._trace_token is not None:
            _current_trace_id.reset(self._trace_token)
        if self._client_token is not None:
            _current_client.reset(self._client_token)


class _ObserveFactory:
    """Object that acts as both decorator factory and context manager."""

    def __init__(
        self,
        *,
        name: str | None,
        client: ObservabilityClient | None,
        observation_type: ObservationType | str,
        metadata: dict[str, Any] | None,
        environment: str | None,
        release: str | None,
        tags: list[str] | None,
        redact_input: bool,
    ) -> None:
        self.name = name
        self.client = client
        self.observation_type = observation_type
        self.metadata = metadata
        self.environment = environment
        self.release = release
        self.tags = tags
        self.redact_input = redact_input
        self._context: ObserveContext | None = None

    def __call__(self, func: Callable[..., Any]):
        observation_name = self.name or func.__name__

        if inspect.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                input_data = (
                    _redacted_input_payload()
                    if self.redact_input
                    else {"args": args, "kwargs": kwargs}
                )
                async with ObserveContext(
                    name=observation_name,
                    client=self.client,
                    observation_type=self.observation_type,
                    input_data=input_data,
                    metadata=self.metadata,
                    environment=self.environment,
                    release=self.release,
                    tags=self.tags,
                    redact_input=self.redact_input,
                ) as ctx:
                    result = await func(*args, **kwargs)
                    ctx._result = result
                    return result

            return async_wrapper

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            input_data = (
                _redacted_input_payload()
                if self.redact_input
                else {"args": args, "kwargs": kwargs}
            )
            with ObserveContext(
                name=observation_name,
                client=self.client,
                observation_type=self.observation_type,
                input_data=input_data,
                metadata=self.metadata,
                environment=self.environment,
                release=self.release,
                tags=self.tags,
                redact_input=self.redact_input,
            ) as ctx:
                result = func(*args, **kwargs)
                ctx._result = result
                return result

        return wrapper

    def __enter__(self) -> ObserveContext:
        self._context = ObserveContext(
            name=self.name or "observation",
            client=self.client,
            observation_type=self.observation_type,
            metadata=self.metadata,
            environment=self.environment,
            release=self.release,
            tags=self.tags,
            redact_input=self.redact_input,
        )
        return self._context.__enter__()

    def __exit__(self, exc_type, exc, exc_tb) -> bool:
        if self._context is None:
            raise RuntimeError(
                "observe context has not been entered; __exit__ called before __enter__"
            )
        return self._context.__exit__(exc_type, exc, exc_tb)

    async def __aenter__(self) -> ObserveContext:
        self._context = ObserveContext(
            name=self.name or "observation",
            client=self.client,
            observation_type=self.observation_type,
            metadata=self.metadata,
            environment=self.environment,
            release=self.release,
            tags=self.tags,
            redact_input=self.redact_input,
        )
        return await self._context.__aenter__()

    async def __aexit__(self, exc_type, exc, exc_tb) -> bool:
        if self._context is None:
            raise RuntimeError(
                "observe context has not been entered; __aexit__ called before __aenter__"
            )
        return await self._context.__aexit__(exc_type, exc, exc_tb)


def observe(
    name: str | None = None,
    *,
    client: ObservabilityClient | None = None,
    observation_type: ObservationType | str = ObservationType.SPAN,
    metadata: dict[str, Any] | None = None,
    environment: str | None = None,
    release: str | None = None,
    tags: list[str] | None = None,
    redact_input: bool = False,
):
    """Create a decorator or context manager for observability instrumentation.

    By default the decorator captures function arguments as trace input. Set
    `redact_input=True` when arguments may include secrets, credentials, or PII.
    """

    if callable(name):
        func = name
        return observe()(func)

    return _ObserveFactory(
        name=name,
        client=client,
        observation_type=observation_type,
        metadata=metadata,
        environment=environment,
        release=release,
        tags=tags,
        redact_input=redact_input,
    )
