"""Decorator and context-manager helpers for observability instrumentation."""

from __future__ import annotations

import contextvars
import functools
import inspect
import threading
from collections.abc import Callable
from datetime import datetime
from typing import Any, Literal

from traigent.config.context import (
    applied_config_context,
    config_context,
    get_trial_context,
)
from traigent.observability.client import ObservabilityClient
from traigent.observability.dtos import ObservationType, to_jsonable, utc_now
from traigent.utils.logging import get_logger

logger = get_logger(__name__)

_SENSITIVE_KEY_FRAGMENTS = (
    "api_key",
    "apikey",
    "auth",
    "credential",
    "password",
    "private_key",
    "secret",
    "token",
)
_ACTIVE_CONFIG_METADATA_KEY = "traigent_active_config"
_OPTIMIZATION_CONTEXT_METADATA_KEY = "traigent_optimization_context"

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


def _is_sensitive_key(key: str) -> bool:
    normalized = key.lower().replace("-", "_")
    return any(fragment in normalized for fragment in _SENSITIVE_KEY_FRAGMENTS)


def _coerce_mapping_payload(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, dict):
        return value
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        candidate = to_dict()
        if isinstance(candidate, dict):
            return candidate
    return None


def _sanitize_metadata_value(value: Any, *, key: str | None = None) -> Any:
    if key is not None and _is_sensitive_key(key):
        return "[REDACTED]"

    mapping = _coerce_mapping_payload(value)
    if mapping is not None:
        sanitized: dict[str, Any] = {}
        for raw_key, raw_value in mapping.items():
            normalized_key = str(raw_key)
            if normalized_key.startswith("_"):
                continue
            sanitized_value = _sanitize_metadata_value(raw_value, key=normalized_key)
            sanitized[normalized_key] = sanitized_value
        return sanitized

    if isinstance(value, (list, tuple, set)):
        return [_sanitize_metadata_value(item) for item in value]

    return to_jsonable(value)


def _build_observe_enrichment_metadata() -> dict[str, Any]:
    raw_applied_config = applied_config_context.get(None)
    raw_active_config = config_context.get(None)
    active_config_source: str | None = None

    active_config = _sanitize_metadata_value(raw_applied_config)
    if isinstance(active_config, dict) and active_config:
        active_config_source = "applied-config"
    else:
        fallback_config = _sanitize_metadata_value(raw_active_config)
        if isinstance(fallback_config, dict) and fallback_config:
            active_config = fallback_config
            active_config_source = "context-config"
        else:
            active_config = None

    raw_trial_context = get_trial_context()
    optimization_context: dict[str, Any] = {}
    if isinstance(raw_trial_context, dict):
        for raw_key, raw_value in raw_trial_context.items():
            normalized_key = str(raw_key)
            if normalized_key.startswith("_") or raw_value is None:
                continue
            optimization_context[normalized_key] = _sanitize_metadata_value(
                raw_value, key=normalized_key
            )

    if isinstance(active_config, dict) and active_config:
        config_snapshot = optimization_context.get("config_snapshot")
        if config_snapshot == active_config:
            optimization_context.pop("config_snapshot", None)

    if raw_trial_context is not None:
        optimization_context["config_source"] = "trial-config"
    elif active_config_source is not None:
        optimization_context["config_source"] = active_config_source

    enriched: dict[str, Any] = {}
    if isinstance(active_config, dict) and active_config:
        enriched[_ACTIVE_CONFIG_METADATA_KEY] = active_config
    if optimization_context:
        enriched[_OPTIMIZATION_CONTEXT_METADATA_KEY] = optimization_context
    return enriched


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
        session_id: str | None = None,
        user_id: str | None = None,
        custom_trace_id: str | None = None,
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
        self.session_id = session_id
        self.user_id = user_id
        self.custom_trace_id = custom_trace_id
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
        self._enriched_metadata = _build_observe_enrichment_metadata()

    def __enter__(self) -> ObserveContext:
        self._started_at = utc_now()
        client = (
            self.client or _current_client.get() or get_default_observability_client()
        )
        self._client_token = _current_client.set(client)

        trace_id = _current_trace_id.get()
        if trace_id is None:
            trace_metadata = {"source": "observe"}
            trace_metadata.update(self.metadata)
            trace_metadata.update(self._enriched_metadata)
            trace_id = client.start_trace(
                self.name,
                session_id=self.session_id,
                user_id=self.user_id,
                environment=self.environment,
                release=self.release,
                tags=self.tags,
                metadata=trace_metadata,
                started_at=self._started_at,
                input_data=self.input_data,
                custom_trace_id=self.custom_trace_id,
            )
            self._created_trace = True
            self._trace_token = _current_trace_id.set(trace_id)

        observation_metadata = dict(self.metadata)
        observation_metadata.update(self._enriched_metadata)
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
            metadata=observation_metadata,
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
        metadata.update(self._enriched_metadata)
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
        session_id: str | None = None,
        user_id: str | None = None,
        custom_trace_id: str | None = None,
        environment: str | None = None,
        release: str | None = None,
        tags: list[str] | None = None,
        redact_input: bool = False,
    ) -> None:
        self.name = name
        self.client = client
        self.observation_type = observation_type
        self.metadata = metadata
        self.session_id = session_id
        self.user_id = user_id
        self.custom_trace_id = custom_trace_id
        self.environment = environment
        self.release = release
        self.tags = tags
        self.redact_input = redact_input
        self._context: ObserveContext | None = None

    def _require_context(self) -> ObserveContext:
        if self._context is None:
            raise RuntimeError(
                "observe context has not been entered; __exit__ called before __enter__"
            )
        return self._context

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
                    session_id=self.session_id,
                    user_id=self.user_id,
                    custom_trace_id=self.custom_trace_id,
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
                session_id=self.session_id,
                user_id=self.user_id,
                custom_trace_id=self.custom_trace_id,
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
            session_id=self.session_id,
            user_id=self.user_id,
            custom_trace_id=self.custom_trace_id,
            environment=self.environment,
            release=self.release,
            tags=self.tags,
            redact_input=self.redact_input,
        )
        return self._context.__enter__()

    def __exit__(self, exc_type, exc, exc_tb) -> bool:
        return self._require_context().__exit__(exc_type, exc, exc_tb)

    async def __aenter__(self) -> ObserveContext:
        self._context = ObserveContext(
            name=self.name or "observation",
            client=self.client,
            observation_type=self.observation_type,
            metadata=self.metadata,
            session_id=self.session_id,
            user_id=self.user_id,
            custom_trace_id=self.custom_trace_id,
            environment=self.environment,
            release=self.release,
            tags=self.tags,
            redact_input=self.redact_input,
        )
        return await self._context.__aenter__()

    async def __aexit__(self, exc_type, exc, exc_tb) -> bool:
        return await self._require_context().__aexit__(exc_type, exc, exc_tb)


def observe(
    name: str | None = None,
    *,
    client: ObservabilityClient | None = None,
    observation_type: ObservationType | str = ObservationType.SPAN,
    metadata: dict[str, Any] | None = None,
    session_id: str | None = None,
    user_id: str | None = None,
    custom_trace_id: str | None = None,
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
        session_id=session_id,
        user_id=user_id,
        custom_trace_id=custom_trace_id,
        environment=environment,
        release=release,
        tags=tags,
        redact_input=redact_input,
    )
