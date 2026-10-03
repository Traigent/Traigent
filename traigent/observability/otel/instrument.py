"""Third-party instrumentor installation with an exposure gate.

A Traigent content policy protects only Traigent's exporter.  Another exporter
attached to the same tracer provider would receive whatever an instrumentor
captures.  Therefore ``check_exporters`` refuses to install instrumentors onto
a provider whose exporters are not verified to be Traigent's, unless the caller
passes ``allow_unverified_exporters=True`` (documented consent to exposure).  A
provider that cannot be inspected counts as unknown.

Instrumentors are also configured to hide content at the source unless the
content mode is ``record`` (best effort: depends on the instrumentor accepting
a ``config`` argument).
"""

from __future__ import annotations

import importlib
import inspect
from dataclasses import dataclass
from typing import Any

from traigent.observability.otel.exporter import TraigentOTLPExporter
from traigent.observability.otel.processor import TraigentSpanProcessor
from traigent.utils.logging import get_logger

logger = get_logger(__name__)

# name -> (module, class).  Public package/class names from each package's
# documented usage; installed only through the optional extras.
INSTRUMENTORS: dict[str, tuple[str, str, str]] = {
    "openai": (
        "openinference.instrumentation.openai",
        "OpenAIInstrumentor",
        "observability-openai",
    ),
    "anthropic": (
        "openinference.instrumentation.anthropic",
        "AnthropicInstrumentor",
        "observability-anthropic",
    ),
    "langchain": (
        "openinference.instrumentation.langchain",
        "LangChainInstrumentor",
        "observability-langchain",
    ),
    "bedrock": (
        "openinference.instrumentation.bedrock",
        "BedrockInstrumentor",
        "observability-bedrock",
    ),
}


class UnverifiedExporterError(RuntimeError):
    """Raised when instrumenting a provider that has unknown exporters."""


class InstrumentationStateError(RuntimeError):
    """An instrumentor is already installed, or did not actually install.

    OpenTelemetry instrumentors return silently when they are already
    instrumented (possibly onto an unaudited provider) and when a dependency
    conflict stops them; neither is visible from the return value.
    """


@dataclass(frozen=True)
class ExporterAudit:
    verified: bool
    unknown: tuple[str, ...]


def audit_exporters(provider: Any) -> ExporterAudit:
    """Classify the span processors/exporters currently attached to ``provider``."""
    multi = getattr(provider, "_active_span_processor", None)
    processors = getattr(multi, "_span_processors", None)
    if processors is None:
        return ExporterAudit(False, ("<provider cannot be inspected>",))
    unknown: list[str] = []
    for proc in processors:
        if isinstance(proc, TraigentSpanProcessor):
            continue
        exporter = getattr(proc, "span_exporter", None)
        if isinstance(exporter, TraigentOTLPExporter):
            continue
        detail = (
            type(exporter).__name__ if exporter is not None else type(proc).__name__
        )
        unknown.append(detail)
    return ExporterAudit(not unknown, tuple(unknown))


def check_exporters(
    provider: Any, *, allow_unverified_exporters: bool
) -> ExporterAudit:
    audit = audit_exporters(provider)
    if not audit.verified and not allow_unverified_exporters:
        raise UnverifiedExporterError(
            "Refusing to install instrumentors: the tracer provider has exporters "
            f"Traigent cannot verify ({', '.join(audit.unknown)}). Instrumentors "
            "capture content that Traigent's content policy cannot remove from "
            "other exporters. Pass allow_unverified_exporters=True to consent to "
            "that exposure."
        )
    return audit


def _masking_config(mode: str) -> Any | None:
    if mode == "record":
        return None
    try:  # public OpenInference TraceConfig; optional
        base = importlib.import_module("openinference.instrumentation")
        return base.TraceConfig(
            hide_inputs=True,
            hide_outputs=True,
            hide_input_messages=True,
            hide_output_messages=True,
            hide_llm_invocation_parameters=True,
            hide_embedding_vectors=True,
        )
    except Exception:
        return None


def _accepts(func: Any, name: str) -> bool:
    try:
        params = inspect.signature(func).parameters
    except (TypeError, ValueError):
        return False
    return name in params or any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()
    )


def resolve_instrumentor(spec: Any) -> Any:
    if isinstance(spec, str):
        try:
            module_name, class_name, extra = INSTRUMENTORS[spec]
        except KeyError:
            raise ValueError(
                f"unknown instrumentor {spec!r}; known: {sorted(INSTRUMENTORS)}"
            ) from None
        try:
            module = importlib.import_module(module_name)
        except ImportError as exc:
            raise ImportError(
                f"instrumentor {spec!r} needs the optional package for it: "
                f'pip install "traigent[{extra}]"'
            ) from exc
        return getattr(module, class_name)()
    if inspect.isclass(spec):
        return spec()
    return spec


def _is_instrumented(instrumentor: Any) -> bool | None:
    """The instrumentor's own install state, or ``None`` if it exposes none."""
    try:
        state = getattr(instrumentor, "_is_instrumented_by_opentelemetry", None)
    except Exception:
        return None
    return state if isinstance(state, bool) else None


def _undo_failed(instrumentor: Any) -> None:
    """Undo a FAILED instrument() call's own partial patches.

    OpenTelemetry's ``BaseInstrumentor.instrument`` flips its installed flag only
    AFTER ``_instrument()`` returns, and its public ``uninstrument()`` is a
    no-op while the flag is False, so a raise midway leaves patches applied that
    the public API will not remove.  Safe to run: Traigent only calls
    ``instrument()`` on an instrumentor verified not installed (or explicitly
    adopted), so whatever is patched now came from this call.
    """
    try:
        if _is_instrumented(instrumentor) is True:
            instrumentor.uninstrument()
            return
        private = getattr(instrumentor, "_uninstrument", None)
        if callable(private):
            private()
        else:
            instrumentor.uninstrument()
    except Exception:
        logger.debug("failed-install cleanup raised", exc_info=True)


def _rollback(installed: list[Any]) -> None:
    for instrumentor in reversed(installed):
        try:
            instrumentor.uninstrument()
        except Exception:
            logger.debug("instrumentor rollback failed", exc_info=True)


def install(
    provider: Any,
    specs: tuple[Any, ...],
    *,
    mode: str,
    allow_unverified_exporters: bool,
    adopt_unverifiable: bool = False,
) -> list[Any]:
    """Install instrumentors onto ``provider`` explicitly (never via globals).

    Returns only instrumentors THIS call installed (the caller owns exactly
    those).  An instrumentor that is already installed is rejected, never
    adopted: it may hold an unaudited provider, and uninstrumenting it later
    would remove instrumentation Traigent did not install.  An install that
    leaves the instrumentor not installed (dependency conflict) raises.  If any
    instrumentor fails, every one installed earlier in this call is rolled back,
    and the failing one's own partial patches are undone too.

    An instrumentor that exposes no install state (``_is_instrumented`` is
    ``None``) cannot be verified, so it is treated as NOT owned and refused,
    failing closed; pass ``adopt_unverifiable=True`` to take responsibility for
    it explicitly.
    """
    check_exporters(provider, allow_unverified_exporters=allow_unverified_exporters)
    installed: list[Any] = []
    try:
        for spec in specs:
            instrumentor = resolve_instrumentor(spec)
            name = type(instrumentor).__name__
            state = _is_instrumented(instrumentor)
            if state is None and not adopt_unverifiable:
                raise InstrumentationStateError(
                    f"{name} exposes no install state, so Traigent cannot verify "
                    "that it is not already instrumented or that it installed; "
                    "refusing to treat it as owned. Pass adopt_unverifiable=True "
                    "to adopt it explicitly."
                )
            if state is True:
                raise InstrumentationStateError(
                    f"{name} is already instrumented (by the application or another "
                    "library); Traigent will not adopt it because it may hold an "
                    "unaudited tracer provider. Uninstrument it first."
                )
            kwargs: dict[str, Any] = {"tracer_provider": provider}
            config = _masking_config(mode)
            if config is not None and _accepts(instrumentor.instrument, "config"):
                kwargs["config"] = config
            elif mode != "record":
                logger.info(
                    "instrumentor %s was not given a source-masking config; content "
                    "is still removed by Traigent's exporter",
                    name,
                )
            try:
                instrumentor.instrument(**kwargs)
            except BaseException:
                _undo_failed(instrumentor)  # works before the installed flag flips
                raise
            if _is_instrumented(instrumentor) is False:
                raise InstrumentationStateError(
                    f"{name}.instrument() returned without installing (dependency "
                    "conflict or unsupported library version); nothing was "
                    "instrumented."
                )
            installed.append(instrumentor)
    except BaseException:
        _rollback(installed)
        raise
    return installed
