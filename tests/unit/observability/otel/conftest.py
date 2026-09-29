"""Shared fixtures for the OTel observability tests."""

from __future__ import annotations

import pytest

pytest.importorskip("opentelemetry.sdk")
pytest.importorskip("opentelemetry.exporter.otlp.proto.common")

from opentelemetry.sdk.resources import Resource  # noqa: E402
from opentelemetry.sdk.trace import TracerProvider  # noqa: E402
from opentelemetry.sdk.trace.export import SimpleSpanProcessor  # noqa: E402
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (  # noqa: E402
    InMemorySpanExporter,
)

CANARY = "CANARY-7f3a91c2-do-not-egress"


@pytest.fixture
def canary() -> str:
    return CANARY


@pytest.fixture
def memory_provider():
    """A real SDK provider with an in-memory exporter (its own resource canary)."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider(
        resource=Resource.create(
            {
                "service.name": "svc",
                "canary.resource.attr": CANARY,
                "process.command_line": CANARY,
            }
        ),
        shutdown_on_exit=False,
    )
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    yield provider, exporter
    provider.shutdown()
