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
import hashlib
import hmac
import logging
import math
import secrets
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
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
from traigent.economics.payload import build_telemetry_request, canonical_json_bytes
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

# Sentinel: response body failed to decode as JSON. Used so the decode exception
# is confined to its handler and never chained into the public error.
_JSON_PARSE_FAILED = object()

# Attribute name under which prepare() stamps an issued batch's HMAC token. The
# NAME is not secret (knowing it does not help forge a token); the token VALUE is
# an HMAC over the batch's object identity keyed by a process-random secret that
# lives ONLY inside the closures created by ``_install_issuance`` below.
_ISSUANCE_ATTR = "_economics_issuance_token"


@dataclass(frozen=True)
class PreparedTelemetryBatch:
    """An immutable, prepare()-issued batch ready to transmit or resubmit.

    ``content`` is the exact serialized payload sent on every attempt; ``body``
    is a read-only view for inspection. Resubmitting the SAME prepared batch is
    the honest cross-call recovery path: identical bytes and key mean the
    backend replays rather than writing a second batch.

    The class is publicly constructible for API compatibility, but only the exact
    object returned by :meth:`EconomicsTelemetryClient.prepare` is submittable:
    a directly-constructed, ``copy.copy``/``copy.deepcopy``-ed,
    ``dataclasses.replace``-d, or unpickled instance is a different identity, so
    it carries no valid identity-bound issuance token and is refused before any
    transport (fail closed).

    Repr safety: ``content`` (the raw serialized payload) and ``body`` (the raw
    payload mapping, including shared evidence pointers) are marked
    ``field(repr=False)`` so they never appear in ``repr()``/``str()`` or a ``%r``
    log record — a batch payload could otherwise leak the exact values the
    characterization sharing policy governs. Only the identifiers (project_id,
    idempotency_key, batch_id, submitted, event_ids) are shown. This changes the
    repr ONLY: the fields remain ordinary init/eq fields, the exact wire bytes are
    unchanged, and the identity-bound issuance token (an instance attribute set
    outside the dataclass fields) is unaffected.
    """

    project_id: str
    idempotency_key: str
    batch_id: str
    submitted: int
    #: The exact serialized payload sent on every attempt. Redacted from repr:
    #: it carries payload values / evidence pointers that must not be logged.
    content: bytes = field(repr=False)
    #: A read-only view of the raw payload. Redacted from repr for the same reason.
    body: Mapping[str, Any] = field(repr=False)
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

    def _has_credential(self) -> bool:
        """True when an API key or JWT is configured for this client.

        The single fact ``_post_with_retry`` gates transport on: a client
        constructed (or resolved, via ``_resolve_api_key``) without either
        credential must never reach the wire, matching the fail-safe convention
        other SDK write paths use (e.g. ``BackendSyncManager`` refuses to sync
        rather than attempt an uncredentialed backend call).
        """
        return bool(self.api_key) or bool(self.jwt_token)

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
        """Build, egress-check, and schema-validate an immutable, ISSUED batch.

        No transport happens here. The returned batch is fully validated against
        the exact economics Schema (fail closed if it is absent/old/raises),
        carries the exact bytes and key that will be sent, and is stamped with the
        issuance token that makes it submittable.

        This method is REPLACED at import time by the issuance-capable variant
        installed in :func:`_install_issuance` (the token secret lives only in
        that closure, never in module globals). This in-class definition is the
        fail-closed fallback: it returns an UNSTAMPED batch, which
        :meth:`submit_prepared` refuses — so even if installation were skipped,
        nothing unissued could be transmitted.

        Raises:
            EconomicsTelemetryContractError: Malformed batch / project mismatch /
                schema-invalid body / non-serializable payload.
            EconomicsSchemaUnavailable: Exact economics Schema unavailable.
            EgressPolicyError: Characterization egress violation.
        """
        return self._build_prepared(
            events,
            project_id=project_id,
            batch_id=batch_id,
            sent_at=sent_at,
            idempotency_key=idempotency_key,
        )

    def _build_prepared(
        self,
        events: Sequence[Mapping[str, Any]],
        *,
        project_id: str,
        batch_id: str | None = None,
        sent_at: str | None = None,
        idempotency_key: str | None = None,
    ) -> PreparedTelemetryBatch:
        """Build and validate a batch WITHOUT stamping issuance (unsubmittable).

        The heavy lifting shared by ``prepare``. Kept separate so the import-time
        issuance installer can wrap it with the token stamp. A batch returned by
        this method alone is not submittable.
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

        Raises:
            EconomicsTelemetryAuthError: No API key or JWT is configured (raised
                locally, before transport — see :meth:`_post_with_retry`), or the
                backend rejected the request as unauthorized (401 / 403).
        """
        self._require_issued(prepared)
        content, headers = self._reverify_prepared(prepared)
        response = await self._post_with_retry(content, headers)
        return self._interpret(response, prepared)

    @staticmethod
    def _require_issued(prepared: PreparedTelemetryBatch) -> None:
        """Refuse a batch that is not the exact object issued by ``prepare()``.

        The primary trust gate, checked before any field work or transport. It
        verifies the issuance token: an HMAC over the batch's object identity
        keyed by a process-random secret held only inside the issuance closures.
        A directly-constructed, copied, ``dataclasses.replace``-d, or unpickled
        batch — even one whose every public field is internally consistent — has
        no valid token for its identity (a copy carries the original's token but
        a different id, so the HMAC no longer matches), and is refused.

        This in-class definition is the fail-closed FALLBACK: it refuses
        unconditionally. At import time :func:`_install_issuance` REPLACES it with
        a verifier-backed closure installed directly on the class. The verifier,
        the secret, and the mint are captured in that closure and never bound to
        module globals, so there is no module-reachable verifier/key/mint name to
        monkeypatch into an always-accept bypass. If installation were skipped,
        every submission refuses. The error is payload-free.
        """
        raise EconomicsTelemetryContractError(
            "telemetry batch issuance verifier is not installed; no batch can be "
            "submitted (fail closed)"
        )

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
        content = canonical_json_bytes(body)
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
        # Fail-safe gate, checked immediately before the only transport call in
        # this class: never send an economics telemetry request without a
        # credential attached. A missing credential is not a reason to POST and
        # let the backend police it (that would put a real, schema-validated
        # payload on the wire unauthenticated even if the backend later
        # rejects it) -- it is refused here, before any bytes leave the
        # machine, the same way EgressPolicyError and
        # EconomicsSchemaUnavailable already fail closed pre-transport.
        if not self._has_credential():
            logger.debug(
                "economics telemetry request not sent: no API key or JWT is "
                "configured for EconomicsTelemetryClient (fail-safe no-op, "
                "not an unauthenticated request)"
            )
            raise EconomicsTelemetryAuthError(
                "economics telemetry request was not sent: no API key or JWT "
                "is configured for EconomicsTelemetryClient"
            )
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
            # Decode inside the handler but RAISE outside it, so the invalid-JSON
            # exception (whose JSONDecodeError.doc carries the raw payload) never
            # rides along as __cause__ or __context__. Never log the body either.
            try:
                body: Any = response.json()
            except ValueError:
                body = _JSON_PARSE_FAILED
            if body is _JSON_PARSE_FAILED:
                raise EconomicsResponseError(
                    f"economics telemetry response (HTTP {status}) was not JSON"
                )
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

    Routes through the single payload-safe ``canonical_json_bytes`` chokepoint
    (shared with the batch-id and idempotency-key derivations), so the wire path
    never diverges and a lone surrogate cannot leak the payload via a raw
    ``UnicodeEncodeError``.
    """
    return canonical_json_bytes(body)


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


def _install_issuance() -> None:
    """Install the issuance capability. NOTHING escapes to module scope.

    A process-random secret is created here and captured by three closures — the
    stamping ``prepare`` and the verifier-backed ``_require_issued`` (both
    installed onto the class) and the internal token helpers. The secret, the
    token function, the verifier, and the stamping wrapper are all LOCALS of this
    factory: none is bound to module globals and nothing is returned. There is no
    importable minting authority, no writable issuance registry, and — crucially —
    **no module-global verifier/key/mint name to reassign** into an always-accept
    bypass. The prior ``_ISSUED_BATCHES`` registry and the ``_verify_issuance``
    module global are both gone.

    One-shot installer: this function is called exactly once at import time and
    then ``del``-eted from module globals (see below), so ordinary import cannot
    re-invoke it. That closes the re-install bypass — monkeypatching
    ``_build_prepared`` and calling ``_install_issuance()`` again to recapture a
    hostile builder into the mint — because after import there is no installer
    name left to call.

    Dispatch-free build (closes the normal-call mint bypass): the GENUINE builder
    is captured BEFORE any wrapper is installed and is called directly by the
    issuing ``prepare``. The wrapper never calls a dynamically dispatched
    ``self._build_prepared``, so a caller-supplied ``self`` or a subclass cannot
    override ``_build_prepared`` to return a pre-forged batch and have the mint
    HMAC-stamp it. The mint only ever stamps a batch produced by this captured,
    override-proof builder.

    Introspection boundary (stated honestly): the secret is reachable via
    deliberate closure introspection (e.g. ``EconomicsTelemetryClient.prepare.
    __closure__`` / ``EconomicsTelemetryClient._require_issued.__closure__``),
    exactly as ``object.__setattr__`` can stamp any attribute. This is an API
    trust boundary, not a cryptographic seal against a caller who reaches into
    interpreter internals. It DOES close the normal-import and normal-call mint
    paths the findings require.

    No retained object state: the token lives on the batch instance (bounded,
    fixed-size bytes) and is collected with it. The secret is one fixed-size
    constant. There is no per-batch module state, so there is nothing to leak and
    no lifetime bookkeeping to get wrong. The secret is module-scoped, so a batch
    issued by one client instance verifies for any client in the SAME process; it
    does not survive copy, replace, pickling, or a process boundary.
    """
    # Capture the GENUINE builder BEFORE any wrapper is installed. The issuing
    # prepare calls THIS captured function directly (never a dispatched
    # self._build_prepared), so no caller-supplied self/subclass override can
    # substitute the object that gets stamped.
    _genuine_build = EconomicsTelemetryClient._build_prepared

    secret = secrets.token_bytes(32)

    def _expected_token(batch: PreparedTelemetryBatch) -> bytes:
        # Bind to the exact object identity. A copy/replace/unpickle has a
        # different id(), so its (copied or absent) token cannot match.
        return hmac.new(secret, str(id(batch)).encode("ascii"), hashlib.sha256).digest()

    def _verify(batch: PreparedTelemetryBatch) -> bool:
        actual = getattr(batch, _ISSUANCE_ATTR, None)
        if not isinstance(actual, bytes):
            return False
        return hmac.compare_digest(actual, _expected_token(batch))

    def _issuing_prepare(
        self: EconomicsTelemetryClient,
        events: Sequence[Mapping[str, Any]],
        *,
        project_id: str,
        batch_id: str | None = None,
        sent_at: str | None = None,
        idempotency_key: str | None = None,
    ) -> PreparedTelemetryBatch:
        # Build via the captured genuine builder (dispatch-free): a caller cannot
        # substitute the returned object through a _build_prepared override.
        prepared = _genuine_build(
            self,
            events,
            project_id=project_id,
            batch_id=batch_id,
            sent_at=sent_at,
            idempotency_key=idempotency_key,
        )
        # The ONLY place a valid token is minted — inlined here so no standalone,
        # module-reachable "stamp arbitrary batch" callable exists.
        object.__setattr__(prepared, _ISSUANCE_ATTR, _expected_token(prepared))
        return prepared

    def _require_issued(prepared: PreparedTelemetryBatch) -> None:
        # Verifier-backed, closure-held. There is no module-global verifier name
        # to monkeypatch into an always-accept bypass; the verifier is reachable
        # only by deliberately introspecting this closure.
        if not _verify(prepared):
            raise EconomicsTelemetryContractError(
                "telemetry batch was not issued by EconomicsTelemetryClient.prepare(); "
                "only the exact object returned by prepare() may be submitted "
                "(copies, dataclasses.replace results, and unpickled batches are refused)"
            )

    _issuing_prepare.__doc__ = EconomicsTelemetryClient.prepare.__doc__
    _require_issued.__doc__ = EconomicsTelemetryClient._require_issued.__doc__
    # Install both wrappers on the class. setattr (not plain assignment) rebinds
    # the methods to these closure-capturing functions while keeping the verifier,
    # secret, and mint inside the closure — never module/class-visible standalone
    # callables. B010 is suppressed deliberately: direct assignment to a method
    # trips mypy's method-assign, and the attribute names are fixed by design.
    setattr(EconomicsTelemetryClient, "prepare", _issuing_prepare)  # noqa: B010
    setattr(  # noqa: B010
        EconomicsTelemetryClient, "_require_issued", staticmethod(_require_issued)
    )
    # Return nothing: no verifier/secret/mint is exposed to module scope.


# Install the issuance capability onto the class exactly once, then DELETE the
# installer from module globals. This is a one-shot: after import there is no
# ``_install_issuance`` name to re-invoke, so an attacker cannot monkeypatch
# ``_build_prepared`` and re-run the installer to recapture a hostile builder into
# the mint. Nothing is returned to module scope either, so no verifier/key/mint/
# registry is module-reachable — only the class-attribute closures remain.
_install_issuance()
del _install_issuance


__all__ = ["EconomicsTelemetryClient", "HTTPX_AVAILABLE", "PreparedTelemetryBatch"]
