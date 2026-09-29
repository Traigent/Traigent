"""OpenTelemetry-based observability for Traigent (optional extra).

Install with ``pip install "traigent[observability]"``.  Public names:
``init``, ``instrument``, ``observe``, ``attributes``, ``flush``, ``shutdown``,
``stats``.  Import them from this package; the legacy client API in
``traigent.observability`` is unchanged.
"""

from traigent.observability.otel.api import (
    ObservabilityHandle,
    attributes,
    flush,
    get_handle,
    init,
    instrument,
    observe,
    shutdown,
    stats,
)
from traigent.observability.otel.instrument import UnverifiedExporterError
from traigent.observability.otel.processor import FlushOutcome

__all__ = [
    "FlushOutcome",
    "ObservabilityHandle",
    "UnverifiedExporterError",
    "attributes",
    "flush",
    "get_handle",
    "init",
    "instrument",
    "observe",
    "shutdown",
    "stats",
]
