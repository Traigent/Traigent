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
        detail = type(exporter).__name__ if exporter is not None else type(proc).__name__
        unknown.append(detail)
    return ExporterAudit(not unknown, tuple(unknown))


def check_exporters(provider: Any, *, allow_unverified_exporters: bool) -> ExporterAudit:
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


def install(
    provider: Any,
    specs: tuple[Any, ...],
    *,
    mode: str,
    allow_unverified_exporters: bool,
) -> list[Any]:
    """Install instrumentors onto ``provider`` explicitly (never via globals)."""
    check_exporters(provider, allow_unverified_exporters=allow_unverified_exporters)
    installed: list[Any] = []
    for spec in specs:
        instrumentor = resolve_instrumentor(spec)
        kwargs: dict[str, Any] = {"tracer_provider": provider}
        config = _masking_config(mode)
        if config is not None and _accepts(instrumentor.instrument, "config"):
            kwargs["config"] = config
        elif mode != "record":
            logger.info(
                "instrumentor %s was not given a source-masking config; content "
                "is still removed by Traigent's exporter",
                type(instrumentor).__name__,
            )
        instrumentor.instrument(**kwargs)
        installed.append(instrumentor)
    return installed
