"""Public API of the OpenTelemetry observability layer.

``init`` attaches Traigent's processor/exporter to a tracer provider (yours or
one it creates), ``observe`` / ``attributes`` create and enrich spans,
``instrument`` installs third-party instrumentors behind an exposure gate.

Environment: ``OTEL_EXPORTER_OTLP_*`` variables are deliberately ignored - an
environment variable must not be able to redirect the Traigent API key.
"""

from __future__ import annotations

import contextlib
import functools
import inspect
import json
import os
import threading
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.trace import Span, SpanKind, Status, StatusCode, use_span

from traigent.cloud.url_security import validate_cloud_base_url
from traigent.config.backend_config import BackendConfig
from traigent.observability.config import (
    most_restrictive_content_mode,
    resolve_client_content_mode,
    validate_content_mode_override,
)
from traigent.observability.otel import contract as C
from traigent.observability.otel.exporter import TraigentOTLPExporter
from traigent.observability.otel.instrument import install
from traigent.observability.otel.lineage import attributes as attributes  # re-export
from traigent.observability.otel.processor import FlushOutcome, TraigentSpanProcessor
from traigent.observability.otel.sampling import (
    parent_based_ratio_sampler,
    validate_rate,
)
from traigent.observability.otel.transport import Transport, UrllibTransport
from traigent.security.redaction import redact_sensitive_data
from traigent.utils.env_config import is_backend_offline, is_truthy
from traigent.utils.logging import get_logger

logger = get_logger(__name__)

OTLP_PATH = "/api/v1beta/observability/otlp"
SAMPLE_RATE_ENV = "TRAIGENT_OBSERVABILITY_SAMPLE_RATE"


class ObservabilityHandle:
    """A live initialisation: provider, processor, exporter and helpers."""

    def __init__(
        self,
        *,
        provider: Any,
        processor: TraigentSpanProcessor | None,
        exporter: TraigentOTLPExporter | None,
        content_mode: str,
        created_provider: bool,
        allow_unverified_exporters: bool,
    ) -> None:
        self.provider = provider
        self.processor = processor
        self.exporter = exporter
        self.content_mode = content_mode
        self.created_provider = created_provider
        self._allow_unverified = allow_unverified_exporters
        self._instrumentors: list[Any] = []
        self._closed = False
        self._final_stats: dict[str, Any] | None = None

    @property
    def enabled(self) -> bool:
        return self.processor is not None

    @property
    def tracer(self) -> trace.Tracer:
        return self.provider.get_tracer(C.TRAIGENT_SCOPE_NAME)

    def instrument(
        self, *specs: Any, allow_unverified_exporters: bool | None = None
    ) -> None:
        allow = (
            self._allow_unverified
            if allow_unverified_exporters is None
            else allow_unverified_exporters
        )
        self._instrumentors += install(
            self.provider,
            specs,
            mode=self.content_mode,
            allow_unverified_exporters=allow,
        )

    def flush(self, timeout: float = 5.0) -> FlushOutcome:
        if self.processor is None:
            return FlushOutcome(True, False, 0)
        return self.processor.flush(timeout)

    def stats(self) -> dict[str, Any]:
        """Counters of this pipeline; ``enabled`` says whether it is live.

        After ``shutdown`` the final counters are kept (``enabled`` False,
        ``shutdown`` True); an offline handle reports ``reason: offline``.
        """
        if self._final_stats is not None:
            return dict(self._final_stats)
        if self.processor is None:
            return {"enabled": False, "reason": "offline"}
        return {"enabled": True, **self.processor.stats()}

    def shutdown(self) -> None:
        global _handle, _last_handle
        if self._closed:
            return
        self._closed = True
        if self.processor is not None:
            # captured before the worker stops; refreshed after the final drain
            self._final_stats = {"enabled": False, "shutdown": True}
        for instrumentor in self._instrumentors:
            with contextlib.suppress(Exception):
                instrumentor.uninstrument()
        if self.processor is not None:
            self.processor.shutdown()
            self._final_stats = {
                **self.processor.stats(),
                "enabled": False,
                "shutdown": True,
            }
        if self.created_provider:
            with contextlib.suppress(Exception):
                self.provider.shutdown()
        with _lock:
            if _handle is self:
                _handle = None
                _last_handle = self


_lock = threading.Lock()
_handle: ObservabilityHandle | None = None
_last_handle: ObservabilityHandle | None = None  # most recent shut-down handle


def get_handle() -> ObservabilityHandle | None:
    return _handle


def _resolve_rate(sample_rate: float | None) -> float:
    if sample_rate is not None:
        return validate_rate(sample_rate)
    raw = os.getenv(SAMPLE_RATE_ENV)
    if raw is None:
        return 1.0
    try:
        return validate_rate(float(raw))
    except ValueError:
        raise ValueError(
            f"{SAMPLE_RATE_ENV} must be a number between 0 and 1"
        ) from None


def init(
    *,
    api_key: str | None = None,
    project: str | None = None,
    endpoint: str | None = None,
    content_mode: str | None = None,
    sample_rate: float | None = None,
    environment: str | None = None,
    release: str | None = None,
    service_name: str | None = None,
    instrument: Sequence[Any] | str | None = None,
    tracer_provider: Any | None = None,
    set_global: bool = False,
    allow_unverified_exporters: bool = False,
    max_queue_spans: int = 10_000,
    schedule_delay_s: float = 5.0,
    max_batch_spans: int = 512,
    max_batch_bytes: int = 4 * 1024 * 1024,
    export_timeout_s: float = 10.0,
    exit_flush: bool = True,
    exit_flush_timeout_s: float = 5.0,
    on_drop: Callable[[str, int], None] | None = None,
    transport: Transport | None = None,
) -> ObservabilityHandle:
    """Initialise OTel-based observability.  Raises if already initialised.

    ``sample_rate`` applies only to a provider Traigent creates; a provider you
    pass in keeps its own sampler (which is authoritative).
    ``content_mode`` resolves most-restrictive-wins against the environment,
    exactly like the legacy client; the default is ``metadata``.
    """
    global _handle, _last_handle
    with _lock:
        if _handle is not None:
            raise RuntimeError("Traigent observability is already initialised")
        mode, _explicit = resolve_client_content_mode(content_mode)
        rate = _resolve_rate(sample_rate)
        offline = is_backend_offline() or is_truthy(
            os.getenv("TRAIGENT_DISABLE_TELEMETRY")
        )
        if tracer_provider is not None and not hasattr(
            tracer_provider, "add_span_processor"
        ):
            raise TypeError(
                "tracer_provider must be an OpenTelemetry SDK TracerProvider "
                "(it needs add_span_processor)"
            )

        created = tracer_provider is None
        if created:
            attrs: dict[str, Any] = {}
            if service_name:
                attrs["service.name"] = service_name
            if environment:
                attrs["deployment.environment.name"] = environment
            if release:
                attrs["service.version"] = release
            provider = TracerProvider(
                sampler=parent_based_ratio_sampler(rate),
                resource=Resource.create(attrs),
                # Traigent flushes on exit itself (exit_flush); the provider's own
                # atexit hook would make exit_flush=False ineffective.
                shutdown_on_exit=False,
            )
        else:
            provider = tracer_provider

        processor = exporter = None
        if not offline:
            key = api_key if api_key is not None else BackendConfig.get_api_key()
            if not key or not key.strip():
                raise ValueError(
                    "api_key is required (argument or TRAIGENT_API_KEY); set "
                    "TRAIGENT_OFFLINE_MODE=true to run without exporting"
                )
            base = (endpoint or f"{BackendConfig.get_backend_url()}{OTLP_PATH}").rstrip(
                "/"
            )
            url = validate_cloud_base_url(base, purpose="observability OTLP endpoint")
            headers = {"X-API-Key": key.strip(), "User-Agent": "traigent-otel/1"}
            if project:
                headers["X-Project-Id"] = project
            exporter = TraigentOTLPExporter(
                transport or UrllibTransport(f"{url}/v1/traces", headers),
                content_mode=mode,
                max_batch_bytes=max_batch_bytes,
                export_timeout=export_timeout_s,
            )
            processor = TraigentSpanProcessor(
                exporter,
                max_queue_spans=max_queue_spans,
                schedule_delay_s=schedule_delay_s,
                max_batch_spans=max_batch_spans,
                max_batch_bytes=max_batch_bytes,
                exit_flush=exit_flush,
                exit_flush_timeout_s=exit_flush_timeout_s,
                on_drop=on_drop,
            )
            provider.add_span_processor(processor)
        if set_global:
            trace.set_tracer_provider(provider)
        handle = ObservabilityHandle(
            provider=provider,
            processor=processor,
            exporter=exporter,
            content_mode=mode,
            created_provider=created,
            allow_unverified_exporters=allow_unverified_exporters,
        )
        _handle = handle
        _last_handle = None
    if instrument:
        names = (
            ("openai", "anthropic", "langchain", "bedrock")
            if instrument == "auto"
            else instrument
        )
        specs = (names,) if isinstance(names, str) else tuple(names)
        if instrument == "auto":
            specs = tuple(_installed_auto_names())
        try:
            handle.instrument(*specs)
        except Exception:
            handle.shutdown()
            raise
    return handle


def _installed_auto_names() -> list[str]:
    import importlib.util

    from traigent.observability.otel.instrument import INSTRUMENTORS

    found = []
    for name, (module, _cls, _extra) in INSTRUMENTORS.items():
        try:
            if importlib.util.find_spec(module) is not None:
                found.append(name)
        except (ImportError, ValueError):
            continue
    return found


def instrument(*specs: Any, allow_unverified_exporters: bool | None = None) -> None:
    handle = _require_handle()
    handle.instrument(*specs, allow_unverified_exporters=allow_unverified_exporters)


def flush(timeout: float = 5.0) -> FlushOutcome:
    handle = _handle
    return handle.flush(timeout) if handle else FlushOutcome(True, False, 0)


def shutdown() -> None:
    handle = _handle
    if handle is not None:
        handle.shutdown()


def stats() -> dict[str, Any]:
    handle = _handle or _last_handle
    return handle.stats() if handle else {"enabled": False}


def _require_handle() -> ObservabilityHandle:
    if _handle is None:
        raise RuntimeError("call traigent.observability.otel.init() first")
    return _handle


# ---------------------------------------------------------------------------
# observe
# ---------------------------------------------------------------------------

_MAX_CAPTURE_BYTES = C.MAX_RECORD_ATTR_BYTES


def _tracer() -> trace.Tracer:
    handle = _handle
    if handle is not None:
        return handle.tracer
    return trace.get_tracer(C.TRAIGENT_SCOPE_NAME)


def _serialise(value: Any) -> str:
    try:
        text = json.dumps(redact_sensitive_data(value), default=str)
    except Exception:
        text = str(type(value).__name__)
    encoded = text.encode("utf-8")
    if len(encoded) > _MAX_CAPTURE_BYTES:
        text = encoded[:_MAX_CAPTURE_BYTES].decode("utf-8", "ignore")
    return text


class _Observe:
    """Decorator and (async) context manager producing one OTel span."""

    def __init__(
        self,
        name: str | None,
        *,
        as_type: str = "span",
        tool_name: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        session_id: str | None = None,
        user_id: str | None = None,
        tags: Sequence[str] | None = None,
        prompt_reference: Mapping[str, Any] | None = None,
        redact_input: bool = False,
        redact_output: bool = False,
        content_mode: str | None = None,
    ) -> None:
        if as_type not in C.OBSERVATION_TYPES:
            raise ValueError(f"as_type must be one of {sorted(C.OBSERVATION_TYPES)}")
        self._name = name
        self._as_type = as_type
        self._tool_name = tool_name
        self._metadata = dict(metadata) if metadata else None
        self._ctx_attrs = {
            "session_id": session_id,
            "user_id": user_id,
            "tags": tags,
            "prompt_reference": prompt_reference,
        }
        self._redact_input = redact_input
        self._redact_output = redact_output
        self._override = validate_content_mode_override(content_mode)
        self._stack: list[tuple[Span, Any, Any]] = []

    # -- mode -----------------------------------------------------------
    def _mode(self) -> str:
        handle = _handle
        base = handle.content_mode if handle else "metadata"
        if self._override is None:
            return base
        return most_restrictive_content_mode(base, self._override)  # only tightens

    # -- span lifecycle ---------------------------------------------------
    def _begin(self, name: str, args: tuple, kwargs: dict) -> tuple[Span, Any]:
        mode = self._mode()
        handle = _handle
        # Enter the caller-attribute scope first so this span itself is stamped
        # by the processor's on_start, not only its children.
        scope = contextlib.ExitStack()
        if any(v is not None for v in self._ctx_attrs.values()):
            scope.enter_context(attributes(**self._ctx_attrs))
        span = _tracer().start_span(name, kind=SpanKind.INTERNAL)
        span.set_attribute(C.ATTR_OBSERVATION_TYPE, self._as_type)
        if self._tool_name:
            span.set_attribute("gen_ai.tool.name", self._tool_name)
        if handle is not None and mode != handle.content_mode:
            span.set_attribute(C.CONTENT_MODE_ATTRIBUTE, mode)
        if mode == "record":
            if self._redact_input:
                span.set_attribute(C.ATTR_INPUT, C.REDACTED_PLACEHOLDER)
            elif args or kwargs:
                span.set_attribute(
                    C.ATTR_INPUT, _serialise({"args": args, "kwargs": kwargs})
                )
        elif mode == "redacted":
            span.set_attribute(C.ATTR_INPUT, C.REDACTED_PLACEHOLDER)
        if self._metadata and mode != "metadata":
            for key, value in self._metadata.items():
                if isinstance(value, (str, bool, int, float)):
                    span.set_attribute(
                        f"{C.CONTENT_METADATA_PREFIX}{key}",
                        value if mode == "record" else C.REDACTED_PLACEHOLDER,
                    )
        return span, scope

    def _finish(
        self,
        span: Span,
        scope: Any,
        *,
        result: Any = None,
        error: BaseException | None = None,
        has_result: bool = False,
    ) -> None:
        try:
            mode = self._mode()
            if has_result and mode != "metadata":
                if mode == "record" and not self._redact_output:
                    span.set_attribute(C.ATTR_OUTPUT, _serialise(result))
                else:
                    span.set_attribute(C.ATTR_OUTPUT, C.REDACTED_PLACEHOLDER)
            if error is not None:
                etype = type(error).__name__
                span.set_attribute("error.type", etype)
                event_attrs: dict[str, Any] = {"exception.type": etype}
                description = None
                if mode == "record":
                    event_attrs["exception.message"] = str(error)[:1024]
                    description = str(error)[:1024]
                span.add_event("exception", event_attrs)
                span.set_status(Status(StatusCode.ERROR, description))
        finally:
            scope.close()
            span.end()

    # -- context manager --------------------------------------------------
    def __enter__(self) -> Span:
        span, scope = self._begin(self._name or "observe", (), {})
        cm = use_span(
            span,
            end_on_exit=False,
            record_exception=False,
            set_status_on_exception=False,
        )
        cm.__enter__()
        self._stack.append((span, scope, cm))
        return span

    def __exit__(self, exc_type, exc, tb) -> bool:
        span, scope, cm = self._stack.pop()
        cm.__exit__(None, None, None)
        self._finish(span, scope, error=exc if isinstance(exc, BaseException) else None)
        return False

    async def __aenter__(self) -> Span:
        return self.__enter__()

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        return self.__exit__(exc_type, exc, tb)

    # -- decorator ----------------------------------------------------------
    def __call__(self, func: Callable[..., Any]) -> Callable[..., Any]:
        name = self._name or getattr(func, "__name__", "observe")
        run = self

        if inspect.isasyncgenfunction(func):

            @functools.wraps(func)
            async def agen_wrapper(*args: Any, **kwargs: Any):
                span, scope = run._begin(name, args, kwargs)
                agen = func(*args, **kwargs)
                to_send: Any = None
                to_throw: BaseException | None = None
                try:
                    while True:
                        with use_span(
                            span,
                            end_on_exit=False,
                            record_exception=False,
                            set_status_on_exception=False,
                        ):
                            try:
                                if to_throw is not None:
                                    exc, to_throw = to_throw, None
                                    item = await agen.athrow(exc)
                                else:
                                    item = await agen.asend(to_send)
                            except StopAsyncIteration:
                                run._finish(span, scope)
                                span = None
                                return
                        try:
                            to_send = yield item
                        except GeneratorExit:
                            await agen.aclose()
                            raise
                        except BaseException as exc:  # thrown into the wrapper
                            to_throw, to_send = exc, None
                except GeneratorExit:
                    raise
                except BaseException as exc:
                    if span is not None:
                        run._finish(span, scope, error=exc)
                        span = None
                    raise
                finally:
                    if span is not None:  # abandoned / closed early
                        run._finish(span, scope)

            return agen_wrapper

        if inspect.isgeneratorfunction(func):

            @functools.wraps(func)
            def gen_wrapper(*args: Any, **kwargs: Any):
                span, scope = run._begin(name, args, kwargs)
                gen = func(*args, **kwargs)
                to_send: Any = None
                to_throw: BaseException | None = None
                try:
                    while True:
                        with use_span(
                            span,
                            end_on_exit=False,
                            record_exception=False,
                            set_status_on_exception=False,
                        ):
                            try:
                                if to_throw is not None:
                                    exc, to_throw = to_throw, None
                                    item = gen.throw(exc)
                                else:
                                    item = gen.send(to_send)
                            except StopIteration as stop:
                                run._finish(
                                    span, scope, result=stop.value, has_result=True
                                )
                                span = None
                                return stop.value
                        try:
                            to_send = yield item
                        except GeneratorExit:
                            gen.close()
                            raise
                        except BaseException as exc:
                            to_throw, to_send = exc, None
                except GeneratorExit:
                    raise
                except BaseException as exc:
                    if span is not None:
                        run._finish(span, scope, error=exc)
                        span = None
                    raise
                finally:
                    if span is not None:
                        run._finish(span, scope)

            return gen_wrapper

        if inspect.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any):
                span, scope = run._begin(name, args, kwargs)
                try:
                    with use_span(
                        span,
                        end_on_exit=False,
                        record_exception=False,
                        set_status_on_exception=False,
                    ):
                        result = await func(*args, **kwargs)
                except BaseException as exc:
                    run._finish(span, scope, error=exc)
                    raise
                run._finish(span, scope, result=result, has_result=True)
                return result

            return async_wrapper

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any):
            span, scope = run._begin(name, args, kwargs)
            try:
                with use_span(
                    span,
                    end_on_exit=False,
                    record_exception=False,
                    set_status_on_exception=False,
                ):
                    result = func(*args, **kwargs)
            except BaseException as exc:
                run._finish(span, scope, error=exc)
                raise
            run._finish(span, scope, result=result, has_result=True)
            return result

        return wrapper


def observe(name: Any = None, **options: Any) -> Any:
    """Decorator / context manager creating one span (sync, async, generators).

    Content: by default (``metadata``) inputs and outputs are NOT placed on the
    span at all.  ``content_mode`` may only tighten the client's mode.  In
    ``record`` mode the current receiver still drops content (see docs); the SDK
    sends it only because you asked for ``record``.

    ``name`` is NOT a content channel: outside ``record`` mode a name that is not
    identifier-shaped (letters, digits and ``_ . - / :``, at most 80 characters)
    is replaced by the observation type and counted as a dropped attribute.
    Never put user text in a span name.
    """
    if callable(name):
        return _Observe(None)(name)
    return _Observe(name, **options)
