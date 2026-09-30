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


# ---------------------------------------------------------------------------
# Local OTLP/HTTP collector stub (decodes with the official opentelemetry-proto)
# ---------------------------------------------------------------------------
import gzip  # noqa: E402
import threading  # noqa: E402
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer  # noqa: E402

from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import (  # noqa: E402
    ExportTraceServiceRequest,
)


class CollectorStub:
    def __init__(self):
        self.requests: list[dict] = []
        self.raw_bodies: list[bytes] = []
        self.raw_wire: list[bytes] = []  # as received (possibly gzipped)
        self.statuses: list[int] = []  # scripted, popped left; default 200
        self.redirect_to: str | None = None
        self._lock = threading.Lock()
        stub = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):  # silence
                return

            def do_POST(self):  # noqa: N802
                length = int(self.headers.get("Content-Length", 0))
                raw = self.rfile.read(length)
                body = (
                    gzip.decompress(raw)
                    if self.headers.get("Content-Encoding") == "gzip"
                    else raw
                )
                with stub._lock:
                    if stub.redirect_to:
                        self.send_response(307)
                        self.send_header("Location", stub.redirect_to)
                        self.end_headers()
                        return
                    req = ExportTraceServiceRequest()
                    req.ParseFromString(body)
                    stub.raw_bodies.append(body)
                    stub.raw_wire.append(raw)
                    stub.requests.append(
                        {
                            "path": self.path,
                            "headers": {k.lower(): v for k, v in self.headers.items()},
                            "req": req,
                        }
                    )
                    status = stub.statuses.pop(0) if stub.statuses else 200
                self.send_response(status)
                self.send_header("Content-Type", "application/x-protobuf")
                self.end_headers()

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_address[1]}"

    def spans(self):
        out = []
        for entry in list(self.requests):
            for rs in entry["req"].resource_spans:
                for ss in rs.scope_spans:
                    for span in ss.spans:
                        out.append((rs, ss, span))
        return out

    @staticmethod
    def attrs(span) -> dict:
        def val(v):
            kind = v.WhichOneof("value")
            return getattr(v, kind) if kind else None

        return {a.key: val(a.value) for a in span.attributes}

    def close(self):
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def collector():
    stub = CollectorStub()
    yield stub
    from traigent.observability.otel import api

    api.shutdown()  # flush while the stub is still listening
    stub.close()


@pytest.fixture(autouse=True)
def _otel_env(monkeypatch):
    """Development env (local stub allowed), no ambient credentials/redirects."""
    monkeypatch.setenv("TRAIGENT_ENV", "development")
    monkeypatch.delenv("ENVIRONMENT", raising=False)
    monkeypatch.delenv("TRAIGENT_DISABLE_TELEMETRY", raising=False)
    monkeypatch.delenv("TRAIGENT_OBSERVABILITY_CONTENT", raising=False)
    monkeypatch.delenv("TRAIGENT_OBSERVABILITY_CAPTURE_CONTENT", raising=False)
    monkeypatch.delenv("TRAIGENT_OBSERVABILITY_SAMPLE_RATE", raising=False)
    yield
    from traigent.observability.otel import api

    api.shutdown()
