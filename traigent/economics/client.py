"""Async transport for economics telemetry ingestion (WI-B).

A thin authenticated, project-scoped emitter for
``POST /api/v1/economics/telemetry``. It follows the SDK's canonical transport
conventions (the same credential resolution and ``X-Project-Id`` scoping as the
analytics read client), REQUIRES the exact economics Schema to validate every
batch before transmission (fail closed — see :mod:`traigent.economics.schema`),
sends a caller-stable ``Idempotency-Key`` that matches the body, retries only
transient failures (safe because the key replays), and validates the backend's
response before returning an honest acceptance/rejection outcome.

Idempotency and cross-call recovery. A prepared batch is IMMUTABLE: it holds the
exact bytes and key sent on every attempt, so built-in retries replay rather
than double-count. To recover across processes/calls, resubmit the SAME
:class:`PreparedTelemetryBatch` (or rebuild from the SAME complete stable tuple —
events, batch_id, sent_at, idempotency_key); anything less produces a different
key and a new batch, not a replay.

Privacy: request/event bodies, evidence pointers, response bodies, and secrets
are never logged. Only the HTTP status, disposition counts, and closed reason
codes are.
"""

from __future__ import annotations

import asyncio
import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from types import MappingProxyType
from typing import Any, cast

from traigent.cloud.url_security import validate_cloud_base_url
from traigent.cloud.user_agent import get_sdk_user_agent
from traigent.economics.contract import (
    IDEMPOTENCY_KEY_HEADER,
    PROJECT_ID_HEADER,
    TELEMETRY_ENDPOINT,
)
from traigent.economics.errors import (
    EconomicsBatchTooLarge,
    EconomicsIdempotencyConflict,
    EconomicsResponseError,
    EconomicsTelemetryAuthError,
    EconomicsTelemetryContractError,
    EconomicsTelemetryTransportError,
)
from traigent.economics.egress import enforce_characterization_egress
from traigent.economics.payload import build_telemetry_request, canonical_json
from traigent.economics.result import TelemetryIngestResult
from traigent.economics.schema import validate_request_or_fail

try:  # pragma: no cover - exercised only when httpx is absent
    import httpx

    HTTPX_AVAILABLE = True
except ImportError:  # pragma: no cover
    HTTPX_AVAILABLE = False
    httpx = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

_JSON_CONTENT_TYPE = "application/json"
# Response statuses that carry the ingest-result schema (not an error envelope).
_INGEST_RESULT_STATUSES = frozenset({200, 201, 422})
# Transient statuses worth a safe retry under the stable idempotency key.
_TRANSIENT_STATUSES = frozenset({408, 429, *range(500, 600)})
# Backoff bounds. Retry-After is honored but clamped to a finite safe ceiling so
# a hostile or malformed header cannot park the caller for an unbounded delay.
_MAX_BACKOFF_SECONDS = 2.0
_MAX_RETRY_AFTER_SECONDS = 30.0


@dataclass(frozen=True)
class PreparedTelemetryBatch:
    """An immutable, already-validated batch ready to transmit or resubmit.

    ``content`` is the exact serialized payload sent on every attempt; ``body``
    is a read-only view for inspection. Resubmitting the SAME prepared batch is
    the honest cross-call recovery path: identical bytes and key mean the
    backend replays rather than writing a second batch.
    """

    project_id: str
    idempotency_key: str
    batch_id: str
    submitted: int
    content: bytes
    body: Mapping[str, Any]
    #: Ordered submitted event ids (identifiers only, no payload) used to
    #: reconcile the backend's per-event rejections against this exact batch.
    event_ids: tuple[str, ...]

    @property
    def headers(self) -> dict[str, str]:
        """Per-request headers: project scope, idempotency key, content type."""
        return {
            PROJECT_ID_HEADER: self.project_id,
            IDEMPOTENCY_KEY_HEADER: self.idempotency_key,
            "Content-Type": _JSON_CONTENT_TYPE,
        }


class EconomicsTelemetryClient:
    """Async client for emitting economics telemetry batches."""

    def __init__(
        self,
        backend_url: str | None = None,
        *,
        api_key: str | None = None,
        jwt_token: str | None = None,
        timeout: float = 30.0,
        max_retries: int = 3,
    ) -> None:
        """Initialize the emitter.

        Args:
            backend_url: Backend origin. Defaults to the SDK's resolved URL.
            api_key: Explicit API key; the SDK's credential resolution is used
                when None.
            jwt_token: Explicit JWT bearer token; ``api_key`` wins when both are
                present, matching the SDK header builder.
            timeout: Per-request timeout in seconds.
            max_retries: Max attempts for transient failures (>= 1 total).

        Raises:
            ImportError: If ``httpx`` is not installed.
        """
        if not HTTPX_AVAILABLE:
            raise ImportError(
                "httpx is required for EconomicsTelemetryClient. "
                "Install with: pip install 'traigent[hybrid]'"
            )

        from traigent.config.backend_config import (
            BackendConfig,
            get_no_credentials_hint,
        )

        resolved_url = (backend_url or BackendConfig.get_backend_url()).rstrip("/")
        self.backend_url = validate_cloud_base_url(
            resolved_url, purpose="economics telemetry request"
        )
        self.timeout = timeout
        self.max_retries = max(1, int(max_retries))

        self.api_key = api_key
        self.jwt_token = jwt_token
        if self.api_key is None and self.jwt_token is None:
            self.api_key = self._resolve_api_key()
        if not self.api_key and not self.jwt_token:
            logger.warning(
                "No API key or JWT found for EconomicsTelemetryClient. %s",
                get_no_credentials_hint(),
            )

        self._client: Any = None

    # -- credentials / transport plumbing ---------------------------------------

    @staticmethod
    def _resolve_api_key() -> str | None:
        from traigent.cloud.credential_manager import CredentialManager

        return cast("str | None", CredentialManager.get_api_key())

    def _auth_headers(self) -> dict[str, str]:
        """Build auth headers via the canonical API-key/JWT header builder."""
        from traigent.cloud.auth import _build_api_key_auth_headers

        api_key_headers: dict[str, str] = _build_api_key_auth_headers(self.api_key)
        if api_key_headers:
            return api_key_headers
        if self.jwt_token:
            return {"Authorization": f"Bearer {self.jwt_token}"}
        return {}

    def _get_client(self) -> Any:
        from traigent.utils.env_config import raise_if_backend_offline

        raise_if_backend_offline("EconomicsTelemetryClient request")
        if self._client is None:
            headers = self._auth_headers()
            headers.setdefault("User-Agent", get_sdk_user_agent())
            headers.setdefault("Content-Type", _JSON_CONTENT_TYPE)
            self._client = httpx.AsyncClient(
                base_url=self.backend_url,
                headers=headers,
                timeout=self.timeout,
                # Never auto-follow redirects: a 3xx would otherwise re-send the
                # auth credential (X-API-Key / Authorization) to a redirect target.
                # A redirect surfaces as a transport error instead.
                follow_redirects=False,
            )
        return self._client

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self) -> EconomicsTelemetryClient:
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.close()

    # -- public API -------------------------------------------------------------

    def prepare(
        self,
        events: Sequence[Mapping[str, Any]],
        *,
        project_id: str,
        batch_id: str | None = None,
        sent_at: str | None = None,
        idempotency_key: str | None = None,
    ) -> PreparedTelemetryBatch:
        """Build, egress-check, and schema-validate an immutable batch.

        No transport happens here. The returned batch is fully validated against
        the exact economics Schema (fail closed if it is absent/old/raises) and
        carries the exact bytes and key that will be sent, so it can be
        resubmitted for an honest cross-call replay.

        Raises:
            EconomicsTelemetryContractError: Malformed batch / project mismatch /
                schema-invalid body / non-serializable payload.
            EconomicsSchemaUnavailable: Exact economics Schema unavailable.
            EgressPolicyError: Characterization egress violation.
        """
        scope = _require_non_empty(project_id, "project_id")
        body = build_telemetry_request(
            events,
            batch_id=batch_id,
            sent_at=sent_at,
            idempotency_key=idempotency_key,
        )
        self._guard_project_scope(body["events"], scope)
        # Fail closed: the exact Schema must validate the request before any bytes
        # are serialized for transport. Arbitrary extra keys are rejected here.
        validate_request_or_fail(body)
        content = _serialize(body)
        # Identifiers only (schema-validated present), for response reconciliation.
        event_ids = tuple(str(event["event_id"]) for event in body["events"])
        return PreparedTelemetryBatch(
            project_id=scope,
            idempotency_key=body["idempotency_key"],
            batch_id=body["batch_id"],
            submitted=len(body["events"]),
            content=content,
            body=MappingProxyType(dict(body)),
            event_ids=event_ids,
        )

    async def submit(
        self,
        events: Sequence[Mapping[str, Any]],
        *,
        project_id: str,
        batch_id: str | None = None,
        sent_at: str | None = None,
        idempotency_key: str | None = None,
    ) -> TelemetryIngestResult:
        """Prepare and submit one idempotent batch; return the honest outcome.

        A backend rejection (including an all-rejected 422) is returned as a
        result, never disguised as success. Transport/auth/conflict/malformed-
        response failures raise typed errors.
        """
        prepared = self.prepare(
            events,
            project_id=project_id,
            batch_id=batch_id,
            sent_at=sent_at,
            idempotency_key=idempotency_key,
        )
        return await self.submit_prepared(prepared)

    async def submit_prepared(
        self, prepared: PreparedTelemetryBatch
    ) -> TelemetryIngestResult:
        """Submit an already-prepared batch, replaying on retry with identical bytes.

        The prepared batch is treated as UNTRUSTED input: ``PreparedTelemetryBatch``
        is a public, constructible/``dataclasses.replace``-able dataclass, so every
        transport field is re-derived and re-validated from its body immediately
        before transport (see :meth:`_reverify_prepared`). A directly-constructed
        or tampered batch — forged bytes, withheld/non-schema JSON, a mismatched
        key, event ids, or project scope, or an absent Schema — fails closed here,
        so no forged byte reaches the wire.
        """
        content, headers = self._reverify_prepared(prepared)
        response = await self._post_with_retry(content, headers)
        return self._interpret(response, prepared)

    def _reverify_prepared(
        self, prepared: PreparedTelemetryBatch
    ) -> tuple[bytes, dict[str, str]]:
        """Re-derive and validate every transport field from the batch body.

        Returns the canonical bytes and headers to transmit, computed FROM the
        re-validated body (never the object's precomputed ``content``/headers).
        Fails closed on any tamper or on an unavailable exact Schema.
        """
        body_map = prepared.body
        if not isinstance(body_map, Mapping):
            raise EconomicsTelemetryContractError(
                "prepared batch body is not a mapping"
            )
        body = dict(body_map)

        events = body.get("events")
        if not isinstance(events, list) or not events:
            raise EconomicsTelemetryContractError("prepared batch carries no events")

        # Re-run egress on every run_economics characterization (fail closed) so a
        # withheld value can never leave even via a hand-built batch.
        for event in events:
            if (
                isinstance(event, Mapping)
                and event.get("event_type") == "run_economics"
            ):
                enforce_characterization_egress(event.get("characterization") or {})

        # Re-validate against the exact Schema: rejects arbitrary non-schema JSON
        # and fails closed when the exact Schema is unavailable.
        validate_request_or_fail(body)

        # Project binding: the scope must be present and every event must name it.
        scope = _require_non_empty(prepared.project_id, "project_id")
        self._guard_project_scope(events, scope)

        # Derive the transport fields from the validated body and require the
        # object's claimed fields to match exactly.
        content = canonical_json(body).encode("utf-8")
        event_ids = tuple(str(event["event_id"]) for event in events)
        key = body.get("idempotency_key")
        batch_id = body.get("batch_id")
        if content != prepared.content:
            raise EconomicsTelemetryContractError(
                "prepared content does not match its body"
            )
        if prepared.idempotency_key != key:
            raise EconomicsTelemetryContractError(
                "prepared idempotency key does not match its body"
            )
        if prepared.batch_id != batch_id:
            raise EconomicsTelemetryContractError(
                "prepared batch id does not match its body"
            )
        if prepared.event_ids != event_ids:
            raise EconomicsTelemetryContractError(
                "prepared event ids do not match its body"
            )
        if prepared.submitted != len(events):
            raise EconomicsTelemetryContractError(
                "prepared submitted count does not match its body"
            )

        headers = {
            PROJECT_ID_HEADER: scope,
            IDEMPOTENCY_KEY_HEADER: str(key),
            "Content-Type": _JSON_CONTENT_TYPE,
        }
        return content, headers

    # -- request / response -----------------------------------------------------

    async def _post_with_retry(self, content: bytes, headers: dict[str, str]) -> Any:
        client = self._get_client()
        last_exc: Exception | None = None
        for attempt in range(self.max_retries):
            is_last = attempt + 1 >= self.max_retries
            try:
                response = await client.post(
                    TELEMETRY_ENDPOINT, content=content, headers=headers
                )
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_exc = exc
                logger.debug(
                    "economics telemetry transport error (attempt %d/%d)",
                    attempt + 1,
                    self.max_retries,
                )
                if is_last:
                    break
                await self._backoff(attempt, None)
                continue

            if response.status_code in _TRANSIENT_STATUSES and not is_last:
                logger.debug(
                    "economics telemetry transient HTTP %d (attempt %d/%d)",
                    response.status_code,
                    attempt + 1,
                    self.max_retries,
                )
                await self._backoff(attempt, response)
                continue
            return response

        raise EconomicsTelemetryTransportError(
            f"economics telemetry request failed after {self.max_retries} attempts"
        ) from last_exc

    @staticmethod
    async def _backoff(attempt: int, response: Any) -> None:
        delay = min(_MAX_BACKOFF_SECONDS, 0.25 * (2**attempt))
        if response is not None:
            retry_after = _parse_retry_after(
                response.headers.get("Retry-After"), _utcnow()
            )
            if retry_after is not None:
                # Honor the server, but never longer than a finite safe ceiling.
                delay = min(_MAX_RETRY_AFTER_SECONDS, max(delay, retry_after))
        await asyncio.sleep(delay)

    def _interpret(
        self, response: Any, prepared: PreparedTelemetryBatch
    ) -> TelemetryIngestResult:
        status = response.status_code
        if status in _INGEST_RESULT_STATUSES:
            try:
                body = response.json()
            except ValueError as exc:
                # Never log the body; report a stable, payload-free error.
                raise EconomicsResponseError(
                    f"economics telemetry response (HTTP {status}) was not JSON"
                ) from exc
            result = TelemetryIngestResult.from_response(
                http_status=status,
                body=body,
                expected_idempotency_key=prepared.idempotency_key,
                expected_batch_id=prepared.batch_id,
                expected_submitted=prepared.submitted,
                expected_event_ids=prepared.event_ids,
            )
            logger.info(
                "economics telemetry ingested: status=%d replayed=%s "
                "submitted=%d accepted=%d duplicate=%d rejected=%d reasons=%s",
                status,
                result.replayed,
                result.submitted,
                result.accepted,
                result.duplicate,
                result.rejected,
                sorted(set(result.rejection_reasons)),
            )
            return result

        # Error envelopes: map to typed errors. Never log or surface the body.
        if status == 400:
            raise EconomicsTelemetryContractError(
                "economics telemetry batch was rejected as malformed (HTTP 400)"
            )
        if status in (401, 403):
            raise EconomicsTelemetryAuthError(
                f"economics telemetry request was not authorized (HTTP {status})"
            )
        if status == 409:
            raise EconomicsIdempotencyConflict(
                "economics telemetry idempotency key was reused with a different body "
                "(HTTP 409)"
            )
        if status == 413:
            raise EconomicsBatchTooLarge(
                "economics telemetry batch exceeded the event limit (HTTP 413)"
            )
        raise EconomicsTelemetryTransportError(
            f"economics telemetry request failed with HTTP {status}"
        )

    # -- guards -----------------------------------------------------------------

    @staticmethod
    def _guard_project_scope(
        events: Sequence[Mapping[str, Any]], project_id: str
    ) -> None:
        """Fail fast when an event's project_ref differs from the submitted scope.

        A client-side consistency guard only: the backend is the authority that
        verifies ownership against the request context. Catching the mismatch
        here avoids a guaranteed tenant-scope rejection round-trip.
        """
        for index, event in enumerate(events):
            ref = event.get("project_ref")
            if ref is not None and ref != project_id:
                raise EconomicsTelemetryContractError(
                    f"event {index} project_ref does not match the submitted project scope"
                )


# -- module helpers -------------------------------------------------------------


def _utcnow() -> datetime:
    """Current UTC instant. Indirected so tests can inject a deterministic clock."""
    return datetime.now(UTC)


def _serialize(body: Mapping[str, Any]) -> bytes:
    """Serialize the batch to the exact CANONICAL bytes sent on every attempt.

    Uses the same canonical (sorted-key, compact) representation the idempotency
    key and batch id are derived from, so two equivalent mappings that differ
    only in insertion order produce identical batch id, idempotency key, AND wire
    bytes — the key can never agree while the bytes diverge.
    """
    return canonical_json(body).encode("utf-8")


def _parse_retry_after(value: str | None, now: datetime) -> float | None:
    """Parse a Retry-After header to a finite, non-negative delay (seconds).

    Supports both HTTP forms: delta-seconds and an HTTP-date. A past date is
    treated as zero; an invalid or non-finite value returns None (caller falls
    back to computed backoff). The caller clamps the result to a finite ceiling.
    """
    if not value:
        return None
    text = value.strip()
    # delta-seconds
    try:
        parsed = float(text)
    except (TypeError, ValueError):
        parsed = None
    if parsed is not None:
        if not math.isfinite(parsed):
            return None
        return max(0.0, parsed)
    # HTTP-date
    try:
        retry_at = parsedate_to_datetime(text)
    except (TypeError, ValueError, IndexError, OverflowError):
        return None
    if retry_at is None:
        return None
    if retry_at.tzinfo is None:
        retry_at = retry_at.replace(tzinfo=UTC)
    delta = (retry_at - now).total_seconds()
    if not math.isfinite(delta):
        return None
    return max(0.0, delta)


def _require_non_empty(value: str, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise EconomicsTelemetryContractError(f"{field} must be a non-empty string")
    return value


__all__ = ["EconomicsTelemetryClient", "HTTPX_AVAILABLE", "PreparedTelemetryBatch"]
