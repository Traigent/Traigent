"""Issue #1894: an exception escaping a Traigent span must not reach OTLP raw.

OpenTelemetry's ``start_as_current_span`` defaults (``record_exception=True``,
``set_status_on_exception=True``) attach the raw exception message and
stacktrace as an ``exception`` event and a raw ``"<Type>: <message>"`` status
description, bypassing ``_scrub_error_text``. Canary: an exception whose message
carries PII sentinels is raised inside each span, and the sentinels must be
absent from everything the span exports (events, attributes, status), while a
scrubbed ``exception`` event is still recorded.

Covers both the embedded implementation in ``traigent.core.tracing`` and the
``traigent-tracing`` plugin, which core delegates to wholesale when installed.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Callable
from contextlib import AbstractContextManager
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

pytest.importorskip("opentelemetry.sdk.trace")

from opentelemetry.sdk.trace import TracerProvider  # noqa: E402
from opentelemetry.sdk.trace.export import SimpleSpanProcessor  # noqa: E402
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (  # noqa: E402
    InMemorySpanExporter,
)
from opentelemetry.trace import StatusCode  # noqa: E402

import traigent.core.tracing as core_tracing  # noqa: E402

EMAIL_SENTINEL = "canary-1894@example.com"
AWS_KEY_SENTINEL = "AKIACANARY1894EXAMPL"  # AKIA + 16 upper-case alphanumerics

_PLUGIN_TRACING = (
    Path(__file__).resolve().parents[3]
    / "plugins"
    / "traigent-tracing"
    / "traigent_tracing"
    / "tracing.py"
)


def _load_plugin_tracing() -> ModuleType:
    """Load the plugin module from source without installing it or touching sys.path.

    Installing it (or adding it to sys.path) would switch
    ``traigent.core.tracing`` to the plugin for every other test.
    """
    spec = importlib.util.spec_from_file_location(
        "_traigent_tracing_plugin_under_test_1894", _PLUGIN_TRACING
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _core_module() -> ModuleType:
    if getattr(core_tracing, "_PLUGIN_AVAILABLE", False):
        pytest.skip("traigent-tracing plugin installed; embedded path not in use")
    return core_tracing


_SPANS: dict[str, Callable[[ModuleType], AbstractContextManager[Any]]] = {
    "session": lambda m: m.optimization_session_span("fn", max_trials=1),
    "trial": lambda m: m.trial_span("trial-1", 0, {"model": "m"}),
    "example": lambda m: m.example_evaluation_span("ex-1", 0, {"q": "hello"}),
}


def _exported_text(span: Any) -> str:
    parts: list[str] = [span.name, str(span.status.description)]
    parts.extend(f"{k}={v}" for k, v in (span.attributes or {}).items())
    for event in span.events:
        parts.append(event.name)
        parts.extend(f"{k}={v}" for k, v in (event.attributes or {}).items())
    return "\n".join(parts)


@pytest.mark.parametrize("span_kind", sorted(_SPANS))
@pytest.mark.parametrize("implementation", ["core", "plugin"])
def test_escaping_exception_is_scrubbed_on_export(
    monkeypatch: pytest.MonkeyPatch, implementation: str, span_kind: str
) -> None:
    module = _core_module() if implementation == "core" else _load_plugin_tracing()
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(module, "get_tracer", lambda: provider.get_tracer("t-1894"))

    with pytest.raises(RuntimeError):
        with _SPANS[span_kind](module):
            raise RuntimeError(
                f"provider rejected {EMAIL_SENTINEL} using key {AWS_KEY_SENTINEL}"
            )

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    exported = _exported_text(span)
    assert EMAIL_SENTINEL not in exported
    assert AWS_KEY_SENTINEL not in exported

    exception_events = [e for e in span.events if e.name == "exception"]
    assert len(exception_events) == 1
    attributes = dict(exception_events[0].attributes or {})
    assert attributes["exception.type"] == "RuntimeError"
    assert "***EMAIL***" in attributes["exception.message"]
    assert "***AWS_ACCESS_KEY***" in attributes["exception.message"]
    assert "exception.stacktrace" not in attributes

    assert span.status.status_code is StatusCode.ERROR
    assert "***EMAIL***" in (span.status.description or "")


@pytest.mark.parametrize("implementation", ["core", "plugin"])
def test_span_without_exception_records_no_exception_event(
    monkeypatch: pytest.MonkeyPatch, implementation: str
) -> None:
    module = _core_module() if implementation == "core" else _load_plugin_tracing()
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(module, "get_tracer", lambda: provider.get_tracer("t-1894"))

    with module.trial_span("trial-1", 0, {"model": "m"}):
        pass

    (span,) = exporter.get_finished_spans()
    assert [e.name for e in span.events] == []
    assert span.status.status_code is StatusCode.UNSET
