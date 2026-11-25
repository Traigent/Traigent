"""Partial OpenTelemetry install: ERROR status must survive a missing OTLP exporter.

When the OpenTelemetry API is installed but the OTLP HTTP exporter is not
(``opentelemetry-exporter-otlp-proto-http`` absent), the embedded tracing
module used to bind ``trace`` to ``None`` because the exporter import shared a
``try`` block with ``from opentelemetry import trace``. ``_set_error_status``
then returned early and spans silently lost ``StatusCode.ERROR``.

These tests block the exporter import, reload/load the tracing module, and
assert that the API ``trace`` module stays bound and that an escaping
exception still marks the span ERROR. Covers the embedded implementation in
``traigent.core.tracing`` and the ``traigent-tracing`` plugin source.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
from collections.abc import Iterator
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

_EXPORTER_PREFIX = "opentelemetry.exporter.otlp.proto.http"
_MISSING = object()

_PLUGIN_TRACING = (
    Path(__file__).resolve().parents[3]
    / "plugins"
    / "traigent-tracing"
    / "traigent_tracing"
    / "tracing.py"
)


def _block_exporter_modules() -> dict[str, Any]:
    """Make the OTLP HTTP exporter unimportable; return the saved sys.modules entries."""
    saved: dict[str, Any] = {}
    for name in (
        _EXPORTER_PREFIX,
        f"{_EXPORTER_PREFIX}.trace_exporter",
        f"{_EXPORTER_PREFIX}.trace_exporter.trace_exporter",
    ):
        saved[name] = sys.modules.get(name, _MISSING)
        sys.modules[name] = None  # type: ignore[assignment]
    return saved


def _restore_modules(saved: dict[str, Any]) -> None:
    for name, value in saved.items():
        if value is _MISSING:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = value


@pytest.fixture
def core_without_exporter() -> Iterator[ModuleType]:
    """Reload ``traigent.core.tracing`` with the OTLP exporter blocked.

    The original module state is restored afterwards so other tests are unaffected.
    """
    saved = _block_exporter_modules()
    try:
        module = importlib.reload(core_tracing)
        yield module
    finally:
        _restore_modules(saved)
        importlib.reload(core_tracing)


def _plugin_without_exporter() -> ModuleType:
    """Load the plugin module from source with the OTLP exporter blocked.

    Loaded from file, not installed, so ``traigent.core.tracing`` keeps using the
    embedded implementation for every other test.
    """
    saved = _block_exporter_modules()
    try:
        spec = importlib.util.spec_from_file_location(
            "_traigent_tracing_plugin_under_test_exporter_missing", _PLUGIN_TRACING
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        _restore_modules(saved)


def _error_span_status(module: ModuleType, monkeypatch: pytest.MonkeyPatch) -> Any:
    """Run a trial span that raises and return the exported span's status."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(module, "get_tracer", lambda: provider.get_tracer("t-exporter"))

    with pytest.raises(RuntimeError):
        with module.trial_span("trial-1", 0, {"model": "m"}):
            raise RuntimeError("provider failed")

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    return spans[0].status


def test_core_keeps_api_trace_module_when_exporter_missing(
    core_without_exporter: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = core_without_exporter
    # Confirm the simulated partial install actually took effect.
    assert module.OTEL_AVAILABLE is False
    assert module.trace is not None

    status = _error_span_status(module, monkeypatch)
    assert status.status_code is StatusCode.ERROR


def test_plugin_keeps_api_trace_module_when_exporter_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _plugin_without_exporter()
    assert module.OTEL_AVAILABLE is False
    assert module.trace is not None

    status = _error_span_status(module, monkeypatch)
    assert status.status_code is StatusCode.ERROR
