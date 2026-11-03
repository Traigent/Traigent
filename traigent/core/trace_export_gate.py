"""Send-time optimization_only gate for OTLP trace export.

Shared by the SDK's embedded tracer and the bundled ``traigent-tracing`` add-on so
every OTLP exporter that can reach a Traigent destination is built the same way:
explicit, frozen headers and key, a session that re-checks every PREPARED request,
and an exporter that re-checks every batch.
"""

from __future__ import annotations

from typing import Any

try:  # pragma: no cover - optional dependency
    import requests
except ImportError:  # pragma: no cover
    requests = None  # type: ignore[assignment]


def _tracing() -> Any:
    import traigent.core.tracing as _t

    return _t


class _GatedSession(requests.Session if requests is not None else object):  # type: ignore[misc]
    """requests.Session that re-checks the gate before EVERY request.

    OpenTelemetry retries inside one ``export``; a batch-level check cannot see a
    credential that latches between attempts. ``send`` is the single choke point
    (``post``/``request`` both land here), so a restricted gate raises instead of
    sending; OTel treats the error as non-retryable and drops the batch.
    """

    def __init__(self, endpoint: str, api_key: Any) -> None:
        super().__init__()
        self._gate_endpoint = endpoint
        self._gate_api_key = api_key

    def send(self, request: Any, **kwargs: Any) -> Any:
        if _tracing()._optimization_only_blocks_export(
            self._gate_endpoint, self._gate_api_key
        ) or not self._prepared_request_allowed(request):
            raise ConnectionAbortedError(
                "optimization_only: trace export blocked at send time"
            )
        # Never follow redirects: a collector could bounce the (credentialed)
        # request to a different, e.g. Traigent, origin the gate never saw.
        kwargs["allow_redirects"] = False
        return super().send(request, **kwargs)

    def _prepared_request_allowed(self, request: Any) -> bool:
        """Authorize the PREPARED request: its own origin and its own X-API-Key."""
        try:
            from urllib.parse import urlsplit

            from traigent.utils import optimization_only_egress as oo_egress

            parts = urlsplit(str(request.url))
            if not parts.scheme or not parts.netloc:
                return False
            origin = f"{parts.scheme}://{parts.netloc}"
            headers = {str(k): v for k, v in dict(request.headers).items()}
            req_key = _tracing()._api_key_from_headers(headers)
            has_bearer = any(k.strip().lower() == "authorization" for k in headers)
            if req_key is None and not has_bearer:
                # carries no credential at all: the frozen key (checked by the
                # caller just before) is the only one in play
                return bool(
                    oo_egress.ensure_trace_egress_allowed(self._gate_api_key, origin)
                )
            if req_key != self._gate_api_key and oo_egress.is_traigent_trace_endpoint(
                origin
            ):
                return False  # credential differs from the frozen, validated one
            return bool(
                oo_egress.ensure_trace_egress_allowed(req_key, origin, headers=headers)
            )
        except Exception:  # noqa: BLE001 - fail closed
            return False


class _GatedSpanExporter:
    """Span exporter that re-checks optimization_only on every batch.

    Any tracer already handed out keeps working through this exporter, so the gate
    lives in the export path: when restricted the batch is dropped (reported as
    SUCCESS, nothing sent). Shutdown/flush therefore discards queued spans instead
    of draining them to Traigent.
    """

    def __init__(self, inner: Any, endpoint: str, api_key: Any = None) -> None:
        self._inner = inner
        self._endpoint = endpoint
        self._api_key = api_key  # frozen at construction; never re-read per batch

    def export(self, spans: Any) -> Any:
        from opentelemetry.sdk.trace.export import SpanExportResult

        if _tracing()._optimization_only_blocks_export(self._endpoint, self._api_key):
            return SpanExportResult.SUCCESS
        return self._inner.export(spans)

    def shutdown(self) -> None:
        self._inner.shutdown()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return bool(self._inner.force_flush(timeout_millis))


def build_gated_otlp_exporter(otlp_endpoint: str) -> tuple[Any, str | None] | None:
    """Build an OTLP/HTTP span exporter that is gated at send time.

    Returns ``(exporter, frozen_key)``, or ``None`` when optimization_only forbids
    export now (the caller must not cache that: the denial may be transient).
    Headers are resolved ONCE and passed explicitly, so the key validated is the
    only key ever sent, even when it is empty.
    """
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
        OTLPSpanExporter,
    )

    from traigent.utils import optimization_only_egress as oo_egress

    t = _tracing()
    headers = t._resolve_trace_headers()
    frozen_key = t._effective_trace_api_key(headers)
    if not oo_egress.ensure_trace_egress_allowed(frozen_key, otlp_endpoint):
        return None
    exporter = _GatedSpanExporter(
        OTLPSpanExporter(
            endpoint=f"{otlp_endpoint}/v1/traces",
            headers=headers,
            session=_GatedSession(otlp_endpoint, frozen_key),
        ),
        otlp_endpoint,
        frozen_key,
    )
    return exporter, frozen_key
