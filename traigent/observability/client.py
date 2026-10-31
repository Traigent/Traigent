"""Sync-friendly observability client built on a buffered batch transport."""

from __future__ import annotations

import atexit
import copy
import hashlib
import io
import json
import sys
import threading
import time
import uuid
from collections import OrderedDict, deque
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any, cast
from urllib import error, request
from urllib.parse import urlencode

from traigent.cloud.async_batch_transport import BatchFlushResult
from traigent.identity.agent_scope import ScopeLedger, new_trace_ledger
from traigent.identity.workload import resolve_environment
from traigent.observability.config import (
    INGEST_MAX_OBSERVATIONS_PER_REQUEST,
    INGEST_MAX_TRACES_PER_REQUEST,
    NOT_SUPPLIED,
    OBSERVABILITY_CONTENT_MODES,
    ObservabilityConfig,
    apply_content_mode,
    apply_content_mode_to_text,
    describe_offline_mode_cause,
    most_restrictive_content_mode,
    nonblank_credential,
    redacted_content_marker,
    validate_content_mode_override,
    wire_value_for_tightened_update,
)
from traigent.observability.dtos import (
    OBSERVABILITY_STATUSES,
    CorrelationIds,
    ExecutionContextDTO,
    FlushResult,
    ObservationDTO,
    ObservationType,
    PromptReferenceDTO,
    SessionDTO,
    SessionListResponse,
    SessionRecord,
    ThumbRating,
    TraceCollaborationState,
    TraceCommentRecord,
    TraceCommentsResponse,
    TraceDTO,
    TraceFeedbackResponse,
    TraceListResponse,
    TraceObservationsResponse,
    TraceRecord,
    utc_now,
)
from traigent.security.redaction import redact_sensitive_data, redact_sensitive_text
from traigent.utils import optimization_only_egress as oo_egress
from traigent.utils.exceptions import (
    AuthenticationError,
    ClientError,
    ContentDisabledError,
    TraigentConnectionError,
)
from traigent.utils.logging import get_logger
from traigent.utils.pagination import iter_pages
from traigent.utils.retry import CLOUD_API_RETRY_CONFIG, RetryHandler

logger = get_logger(__name__)

_TRACE_BATCH_PREFIX_BYTES = len(b'{"traces": [')
_TRACE_BATCH_SUFFIX_BYTES = len(b"]}")
_TRACE_BATCH_ITEM_SEPARATOR_BYTES = len(b", ")


class _HealthDispatchStop:
    """Typed sentinel that terminates the per-transport health dispatcher."""


_HEALTH_DISPATCH_STOP = _HealthDispatchStop()


class _ClientErrorWithRetryAfter(ClientError):
    """Client error carrying a parsed retry delay."""

    retry_after: float

    def __init__(
        self,
        message: str,
        *,
        status_code: int,
        details: dict[str, Any],
        retry_after: float,
    ) -> None:
        super().__init__(message, status_code=status_code, details=details)
        self.retry_after = retry_after


def _gate_opener(opener: Any) -> Any:
    """Route this module's urllib opener through the optimization_only transport backstop."""
    from traigent.labels.transport_gate import gate_opener

    return gate_opener(opener)


class _NoRedirectHandler(request.HTTPRedirectHandler):
    """Prevent credentialed observability requests from replaying headers elsewhere."""

    def redirect_request(
        self,
        req: request.Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> None:
        return None


def _is_client_timeout(exc: BaseException) -> bool:
    """True when ``exc`` is, or was caused by, a client-side socket timeout."""
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, TimeoutError):
            return True
        if isinstance(current, error.URLError) and isinstance(
            current.reason, TimeoutError
        ):
            return True
        current = current.__cause__ or current.__context__
    return False


def _new_trace_id() -> str:
    return f"trace_{uuid.uuid4().hex}"


def _new_observation_id() -> str:
    return f"obs_{uuid.uuid4().hex}"


_CONTENT_BAG_FIELDS = ("metadata", "input_data", "output_data")


def _safe_deepcopy_or_default(value: Any, default: Any, *, context: str) -> Any:
    """B2 (addendum R3, A5 completion): `copy.deepcopy` can raise on a
    caller-supplied structure that is uncopyable (`threading.Lock()`, an
    open socket/file handle, ...) or that holds a throwing `__deepcopy__`/
    property. Every INTAKE copy that detaches from caller-owned state
    (`_detach_metadata`, `_coerce_session`, `_coerce_correlation_ids`,
    `_coerce_prompt_reference`) must go through this helper rather than
    calling `copy.deepcopy` directly: a failure here must withhold the field
    (fall back to `default`) and continue, never raise out of the public
    `start_trace`/`record_observation`/`end_trace`/`submit_feedback` call
    that was only trying to detach from the caller's own object (M8's
    "drop the content field, never break the caller's application").
    """
    try:
        return copy.deepcopy(value)
    except Exception:
        logger.exception(
            "Observability intake deep-copy failed for %s; withholding it "
            "rather than raising into the caller's application.",
            context,
        )
        return default


def _safe_str(value: Any) -> str:
    """B2: `str(value)` can raise for a hostile object with a broken
    `__str__`/`__repr__` (a throwing "getter") -- e.g. building
    `error_message` from a caller-raised exception whose subclass overrides
    `__str__` to raise. Must never raise out of a public observability
    call; falls back to a fixed placeholder instead.
    """
    try:
        return str(value)
    except Exception:
        logger.exception(
            "Observability failed to stringify a value (broken __str__); "
            "using a placeholder instead of raising into the caller's "
            "application.",
        )
        return "<unstringifiable value>"


def _detach_metadata(metadata: dict[str, Any] | None) -> dict[str, Any]:
    """M4/A8: never retain a caller-owned reference for a metadata bag.

    `dict(metadata or {})` (the previous behavior) only copies the TOP-LEVEL
    dict -- every nested dict/list/set the caller passed in is still the
    SAME object, so a caller mutating their own nested structure AFTER the
    call returns (but before a later `flush()`/`close()`/update
    re-serializes this trace/observation from the stored DTO) can still
    change what eventually gets sent (finding 7's confirmed probe: mutating
    nested trace metadata after `start_trace` changed the delivered
    payload). A deep copy at intake detaches from every caller-owned nested
    structure in one step.

    B2: the deep copy itself can raise on a hostile value (`threading.
    Lock()` nested in metadata, a cyclic structure, ...) -- `_safe_deepcopy_
    or_default` catches that and withholds the whole bag (`{}`) rather than
    raising out of `start_trace`/`record_observation`/`end_trace`.
    """
    if not metadata:
        return {}
    copied = _safe_deepcopy_or_default(metadata, None, context="metadata")
    return copied if isinstance(copied, dict) else {}


def _harden_content_bags(payload: Any) -> Any:
    """Apply credential-key-name hardening to every arbitrary, caller-supplied
    data bag on a trace/observation payload dict -- `metadata` / `input_data`
    / `output_data` directly on the payload, `session.metadata` (A4: session
    metadata is its own arbitrary bag, distinct from the trace/observation
    metadata `_CONTENT_BAG_FIELDS` already covered -- finding 3's confirmed
    probe: `session.metadata.password` reached the sender unredacted because
    this function never looked inside `session` at all), and
    `prompt_reference.variables` (also an arbitrary caller-chosen
    `dict[str, Any]`, per `PromptReferenceDTO.variables`) -- recursing into
    `observations` and `children` so nested observations get the same
    treatment -- rather than to the whole structural payload.

    `redact_sensitive_data(..., redact_credential_keys=True)` treats any key
    whose NAME looks credential-like as a secret subtree: necessary for
    catching a secret hiding inside a caller-supplied bag under a plausible
    key name (M5), but wrong for the DTO's OWN structural field names --
    `input_tokens` / `output_tokens` / `total_tokens` all substring-match the
    `token` credential root. Those are typed, schema-fixed numeric telemetry
    at a FIXED position in every payload, never an arbitrary caller-chosen
    key, so the credential-key check must never even consider them. Scoping
    *where* that check applies (here) -- rather than special-casing these
    field names inside the generic redactor -- is what keeps a genuinely
    arbitrary `total_tokens` key placed inside a caller-supplied bag still
    non-exempt (see `redact_sensitive_data`'s `usage.*`/`max_tokens`
    exemption, which stays narrow because it never has to reach these
    fields in the first place).

    B4 (addendum R3, A4 completion): the common key-based scrubber now also
    covers `correlation_ids`, `tags`, and the COMPLETE `prompt_reference`
    structure (previously only its `.variables` sub-dict) -- fixed correlation fields and string-only tags
    mean key scanning is a no-op for well-typed values TODAY, but binding
    A4 requires every arbitrary-data bag to flow through the SAME scrubber
    regardless, so a future extension of any of these shapes is covered
    from day one rather than needing another fix. The fixed numeric DTO
    fields (`latency_ms`, `*_tokens`, `cost_usd`, timestamps, ids, ...) are
    never routed through this function at all.

    The caller still runs the result through `redact_sensitive_data(...,
    redact_credential_keys=False)` for value-pattern hygiene (PII-looking
    strings anywhere) without a second key-name pass.
    """
    if not isinstance(payload, dict):
        return payload
    hardened = dict(payload)
    for field_name in _CONTENT_BAG_FIELDS:
        if field_name in hardened:
            hardened[field_name] = redact_sensitive_data(
                hardened[field_name], redact_credential_keys=True
            )
    session = hardened.get("session")
    if isinstance(session, dict) and "metadata" in session:
        hardened_session = dict(session)
        hardened_session["metadata"] = redact_sensitive_data(
            session["metadata"], redact_credential_keys=True
        )
        hardened["session"] = hardened_session
    if isinstance(hardened.get("prompt_reference"), dict):
        hardened["prompt_reference"] = redact_sensitive_data(
            hardened["prompt_reference"], redact_credential_keys=True
        )
    if isinstance(hardened.get("correlation_ids"), dict):
        hardened["correlation_ids"] = redact_sensitive_data(
            hardened["correlation_ids"], redact_credential_keys=True
        )
    if isinstance(hardened.get("tags"), list):
        hardened["tags"] = redact_sensitive_data(
            hardened["tags"], redact_credential_keys=True
        )
    for list_field in ("observations", "children"):
        items = hardened.get(list_field)
        if isinstance(items, list):
            hardened[list_field] = [_harden_content_bags(item) for item in items]
    return hardened


_SAFE_SNAPSHOT_TIMESTAMP_FIELDS = ("started_at", "ended_at")
_SAFE_SNAPSHOT_NUMERIC_FIELDS = (
    "latency_ms",
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "cost_usd",
)
_SAFE_SNAPSHOT_ID_FIELDS = ("session_id", "custom_trace_id")
# B3 (addendum R3, A2 completion): content fields covered by
# `_TraceState.content_history` -- see `_TraceState` for why "did this field
# carry content before" must be tracked independently of the value
# currently held (which may itself be the pathological one that made
# serialization/redaction fail).
_SAFE_SNAPSHOT_CONTENT_FIELDS = ("input_data", "output_data")
# Key `_TraceState.content_history` uses for the trace's OWN content
# fields, distinct from any observation id (observation ids are generated
# by `_new_observation_id`, always prefixed `obs_`, so this never collides).
_TRACE_CONTENT_KEY = "__trace__"


def _safe_scalar(value: Any) -> Any:
    """Only trust an already-plain JSON scalar; anything else becomes `None`.

    The M8/A5 fallback runs precisely because the payload may hold
    pathological or cyclic caller-supplied objects, so even an "allowlisted"
    field name is only passed through if its value is already a plain
    `str`/`int`/`float`/`bool` -- never serialized, recursed into, or
    trusted to have a safe `__repr__`/`__str__`.
    """
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return None


_REDACTED = "[REDACTED]"


def _safe_allowlisted_observation_snapshot(
    descriptor: _ObservationFallbackDescriptor,
) -> dict[str, Any]:
    """Minimal, explicitly allowlisted snapshot for ONE observation, used
    when the trace-level M8/A5 fallback fires (`_safe_allowlisted_snapshot`).

    B3 (addendum R3, finding 1's confirmed probe): the OLD fallback dropped
    every observation outright, so a failed serialization/redaction pass
    over an update to observation X sent a trace snapshot that never
    mentioned X at all -- under the backend's skip-omitted-fields contract,
    X's PRIOR content (from an earlier successful send) stayed stored,
    exactly the "never omit the clearing field" defect B3 forbids. This
    function always keeps the observation (by id) in the outbound update.

    Takes an already-detached
    `_ObservationFallbackDescriptor` -- frozen at snapshot-enqueue time,
    while the client lock was held -- rather than the live `ObservationDTO`
    plus a live, mutable `_TraceState`. Reading `id`/`status`/`type`/
    `parent_observation_id`/`had_prior_*` off a fresh `ObservationDTO`/
    `_TraceState` here would let a fallback that fires later (e.g. in the
    transport's own defensive redaction pass, well after this call's lock
    was released) reflect whatever those mutate into by then, rather than
    this snapshot's own state -- see `_TraceFallbackDescriptor`.

    `id`/`status`/`type`/`parent_observation_id` were themselves read off
    the DTO when the descriptor was built -- SDK-controlled, schema-
    validated fields (`ObservationDTO.__post_init__` requires `id`/`name`
    to be validated strings and `status`/`type` to be validated
    enums/statuses), never arbitrary caller content, so they carry the same
    trust level `_safe_allowlisted_snapshot` already gives the trace's own
    `id`/`status`. `name` is still replaced with the redaction placeholder
    (a caller-chosen string, exactly like the trace-level `name`).

    Content fields (`input_data`/`output_data`) are included -- as the
    clearing placeholder, NEVER the real value -- only when the descriptor
    shows this observation's field carried content BEFORE the value
    currently held (which may itself be the pathological value that made
    serialization/redaction fail). A genuine first write (nothing ever
    recorded for this field) is still safe to omit -- there is nothing
    stored server-side yet to leak (M1/A2/B3 test 4).

    `error_message`
    gets the SAME clearing treatment, inside `metadata` (where the normal,
    non-fallback update path -- `record_observation`'s
    `merged_metadata["error_message"] = "[REDACTED]"` branch -- already
    writes it). The old fallback omitted `metadata` (indeed the whole
    observation) entirely, so an observation that previously carried real
    `error_message` text, then hit this fallback on a later update, sent no
    `metadata` at all -- under the backend's skip-omitted-fields contract,
    the PRIOR error text stayed stored server-side even though local state
    (`existing.metadata["error_message"]`) already held "[REDACTED]".
    """
    safe: dict[str, Any] = {
        "id": descriptor.id,
        "name": _REDACTED,
        "status": descriptor.status,
    }
    if descriptor.type:
        safe["type"] = descriptor.type
    if descriptor.parent_observation_id:
        safe["parent_observation_id"] = descriptor.parent_observation_id
    if descriptor.had_prior_input_data:
        safe["input_data"] = redacted_content_marker()
    if descriptor.had_prior_output_data:
        safe["output_data"] = redacted_content_marker()
    if descriptor.had_prior_error_message:
        safe["metadata"] = {"error_message": _REDACTED}
    return safe


def _keep_if_none(value: Any, fallback: Any) -> Any:
    """``value`` unless it is None, in which case ``fallback``."""
    return value if value is not None else fallback


def _updated_content_value(
    state: _TraceState,
    observation_id: str,
    field_name: str,
    existing_value: Any,
    *,
    supplied: bool,
    gated: Any,
    effective_content_mode: Any,
) -> Any:
    # M4 fix: fall back to `existing` only when this call never supplied the
    # field (NOT_SUPPLIED) -- a supplied value, gated, always overwrites, even
    # when gating withheld it to `None` (tightening wins; see the
    # `apply_content_mode` docstring). `wire_value_for_tightened_update`
    # additionally forces the explicit placeholder (not `None`) when this is a
    # tightening update over PRIOR content -- the ingest contract's
    # `apply_updates` keeps stored content on an omitted/`None` field, so
    # merely omitting would not clear it server-side.
    if not supplied:
        return existing_value
    return wire_value_for_tightened_update(
        gated,
        effective_content_mode=effective_content_mode,
        existing_value=(
            existing_value
            if existing_value is not None
            or not state.had_prior_content(observation_id, field_name)
            else redacted_content_marker()
        ),
    )


def _safe_allowlisted_snapshot(
    payload: dict[str, Any], *, state: _TraceFallbackDescriptor | None = None
) -> dict[str, Any]:
    """M8/A5 last-resort fallback: when a trace snapshot cannot be safely
    redacted (redaction, or the DTO serialization that built `payload`
    itself, raised on a malformed/cyclic caller-supplied structure), never
    send the raw payload -- and never send the full structural payload
    either, since fields like `metadata`/`input_data`/`output_data` are
    exactly what failed to redact. Construct a MINIMAL, explicitly
    allowlisted snapshot instead: identifiers, timestamps, status, and
    numeric DTO counters pass through (only when they are already plain
    scalars -- see `_safe_scalar`); `name`/`user_id` are replaced with the
    redaction placeholder rather than dropped (the backend requires a
    non-empty `name`). Every arbitrary bag (`metadata`, `session`,
    `prompt_reference`, `correlation_ids`, `tags`) is dropped outright --
    recursing into them is exactly the operation that already failed once
    for this payload, and cyclic structures make recursion itself unsafe.

    B3 (addendum R3, A2 completion): `input_data`/`output_data` and the
    nested `observations` are handled differently from the other bags --
    dropping them unconditionally is exactly finding 1's confirmed defect
    (a failed update over PRIOR content silently left the old content
    stored, because the backend's `apply_updates` skips fields/observations
    the update never mentions). When `state` is supplied (the trace/
    observation snapshot path -- never the generic comment/feedback/
    transport-level redaction fallback, which has no descriptor to
    consult), this instead:
      * includes `input_data`/`output_data` as the clearing placeholder
        (never the real value, and never read off `payload`) whenever
        `state` shows the trace carried that content BEFORE the value now
        failing to serialize/redact; a genuine first write still omits (B3
        test 4).
      * always includes every known observation (by id, via
        `_safe_allowlisted_observation_snapshot`), so a failure never
        drops an observation from the update entirely -- and, per that
        function, never drops a prior `error_message` either (finding 1).
    `payload` itself is deliberately NOT consulted for content-field
    presence: at redaction-failure time `payload` is the fully-serialized
    but UNREDACTED dict, so its content fields are present for every write
    (including a genuine first write) -- reading presence off `payload`
    cannot tell "first write" apart from "update over prior content" the
    way the descriptor can.

    `state` is a
    `_TraceFallbackDescriptor` -- an immutable, detached snapshot of exactly
    what this function needs, frozen at snapshot-enqueue time while the
    client lock was held -- never the live, mutable `_TraceState` itself.
    The live object is shared with every later call on the same trace, so
    consulting it here (which can run much later, e.g. from the transport's
    own defensive redaction pass, well after the enqueueing call released
    the lock) risked reflecting whatever the trace mutated into by then --
    a completed status, a newly added observation -- rather than THIS
    snapshot's own state. See `_TraceFallbackDescriptor`/
    `_build_fallback_descriptor`.

    This never raises and never returns `None`: a missing/unsafe `id` gets a
    freshly generated one rather than causing the whole snapshot to be
    dropped, so callers never have to thread a "no snapshot to send" case
    through the transport/flush/close paths.
    """
    safe_id = _safe_scalar(payload.get("id"))
    safe: dict[str, Any] = {
        "id": safe_id if isinstance(safe_id, str) and safe_id else _new_trace_id(),
        "name": _REDACTED,
        "metadata": {},
    }
    # #2588: keep an explicitly supplied status; never fabricate one for a
    # trace that carried none (downstream would otherwise record "failed").
    safe_status = _safe_scalar(payload.get("status"))
    if isinstance(safe_status, str) and safe_status:
        safe["status"] = safe_status
    if payload.get("user_id") is not None:
        safe["user_id"] = _REDACTED
    for field_name in _SAFE_SNAPSHOT_ID_FIELDS:
        value = _safe_scalar(payload.get(field_name))
        if isinstance(value, str) and value:
            safe[field_name] = value
    for field_name in (
        *_SAFE_SNAPSHOT_TIMESTAMP_FIELDS,
        *_SAFE_SNAPSHOT_NUMERIC_FIELDS,
    ):
        value = _safe_scalar(payload.get(field_name))
        if value is not None:
            safe[field_name] = value
    if state is not None:
        _apply_fallback_descriptor(safe, state)
    return safe


def _apply_fallback_descriptor(
    safe: dict[str, Any], state: _TraceFallbackDescriptor
) -> None:
    if state.had_prior_input_data:
        safe["input_data"] = redacted_content_marker()
    if state.had_prior_output_data:
        safe["output_data"] = redacted_content_marker()
    safe["observations"] = [
        _safe_allowlisted_observation_snapshot(observation_descriptor)
        for observation_descriptor in state.observations
    ]


def _safe_allowlisted_snapshot_or_static_fallback(
    payload: Any, *, state: _TraceFallbackDescriptor | None = None
) -> dict[str, Any]:
    """Belt-and-suspenders wrapper around `_safe_allowlisted_snapshot`: even
    reading allowlisted fields off a sufficiently pathological `payload`
    (e.g. a `dict` subclass whose `.get` raises) must never itself raise out
    of the caller's application (M8/A5). Falls back to a fully static,
    payload-independent minimal snapshot as the absolute last resort -- at
    that point `state` is no longer consulted either, since building even
    the allowlisted view already failed.
    """
    if isinstance(payload, dict):
        try:
            return _safe_allowlisted_snapshot(payload, state=state)
        except Exception:
            logger.exception(
                "Observability safe-snapshot fallback itself failed; sending "
                "a static minimal snapshot instead of the caller's payload.",
            )
    # #2588: no "status" here -- it is trace status, and a snapshot-builder
    # failure says nothing about whether the traced operation failed.
    return {
        "id": _new_trace_id(),
        "name": _REDACTED,
        "metadata": {},
    }


def _redact_trace_payload(
    payload: dict[str, Any], *, state: _TraceFallbackDescriptor | None = None
) -> dict[str, Any]:
    """Full redaction pass for an outbound trace/observation snapshot.

    Hardens `metadata`/`input_data`/`output_data` (M5's credential-key-name
    sweep, scoped to just those bags -- see `_harden_content_bags`), then
    value-scans the ENTIRE payload (`redact_credential_keys=False`) for
    PII-shaped text without a second key-name pass -- preserving structural
    fields (ids, timestamps, token counts, cost) everywhere else.

    M8/A5 (fail closed): if redaction itself raises -- a caller-supplied
    payload can hold arbitrary, even pathological, objects -- this never
    propagates the exception (which would break the caller's application
    inside their own `start_trace`/`record_observation`/`end_trace` call)
    and never returns the unredacted payload (which would leak content). It
    logs and falls back to a minimal, explicitly allowlisted snapshot
    instead (`_safe_allowlisted_snapshot`) -- not just the previous bags,
    since a redaction failure means those bags could not be trusted at all,
    and other fields (e.g. `name`, `user_id`) are arbitrary caller content
    too.

    `state` (optional): a `_TraceFallbackDescriptor` -- an immutable,
    detached snapshot of the trace's content-history/observation data,
    frozen (under the client lock) at the moment THIS `payload` was built
    -- so the M8 fallback can consult it (B3) rather than `payload` itself
    -- `payload` here is the fully-serialized but UNREDACTED dict, so it
    cannot distinguish a genuine first write from an update over prior
    content the way the descriptor can. Threaded through from
    `ObservabilityClient._submit_trace_snapshot` into the transport's own
    defensive second redaction pass (`_SyncBatchTransport._prepare_payload`)
    as well, so a scrubbing failure THERE (re-redacting the already-safe
    payload this function just produced) -- which can run well after the
    enqueueing call released the client lock -- still reaches the SAME
    frozen descriptor, never a live `_TraceState` that may have mutated in
    the meantime (finding 2/A8/M4 snapshot immutability) and never a
    state-blind fallback. Left as `None` for callers with no associated
    trace state (e.g. a test that submits directly to `_SyncBatchTransport`
    without going through `ObservabilityClient`).
    """
    try:
        return cast(
            dict[str, Any],
            redact_sensitive_data(
                _harden_content_bags(payload), redact_credential_keys=False
            ),
        )
    except Exception:
        logger.exception(
            "Observability content redaction failed; sending a minimal "
            "safe-allowlisted snapshot for this trace/observation payload "
            "rather than sending it unredacted.",
        )
        return _safe_allowlisted_snapshot_or_static_fallback(payload, state=state)


def _parse_retry_after(headers: Any) -> float | None:
    if not headers:
        return None
    try:
        value = headers.get("Retry-After")
    except AttributeError:
        return None
    if value is None:
        return None

    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        pass

    try:
        retry_at = parsedate_to_datetime(str(value))
    except (TypeError, ValueError, IndexError, OverflowError):
        return None
    if retry_at.tzinfo is None:
        retry_at = retry_at.replace(tzinfo=UTC)
    return max(0.0, (retry_at - datetime.now(UTC)).total_seconds())


@dataclass
class _ObservationCall:
    """The arguments of one ``record_observation`` call after content gating."""

    observation_id: str
    effective_type: ObservationType
    name: str
    parent_observation_id: str | None
    status: str
    started_at: datetime | None
    ended_at: datetime | None
    latency_ms: int | None
    input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None
    cost_usd: float | None
    model_name: str | None
    tool_name: str | None
    input_supplied: bool
    output_supplied: bool
    gated_input_data: Any
    gated_output_data: Any
    effective_content_mode: Any
    error_supplied: bool
    gated_error_message: str | None
    metadata: dict[str, Any]
    correlation_ids: Any
    prompt_reference: Any


@dataclass
class _CloseClaim:
    """What one ``ObservabilityClient.close`` call decided to do (under the lock)."""

    close_in_progress: bool = False
    already_closed: bool = False
    offline_close: bool = False
    trace_payloads: list[tuple[str, dict[str, Any], _TraceFallbackDescriptor]] = field(
        default_factory=list
    )
    inflight_snapshot_submissions: int = 0


@dataclass
class _TraceState:
    trace: TraceDTO
    observations: dict[str, ObservationDTO] = field(default_factory=dict)
    observation_order: list[str] = field(default_factory=list)
    # A1 (addendum R2): the content_mode this TRACE resolved to at
    # `start_trace` time -- the BASE that `record_observation`/`end_trace`/
    # `add_comment`/`submit_feedback` calls on this trace resolve against
    # when they supply no per-call override, and the value a per-call
    # override may only TIGHTEN (never loosen) when `content_mode_locked`.
    resolved_content_mode: str = "metadata"
    # True iff `resolved_content_mode` came from an explicit source: either
    # `start_trace`'s own `content_mode` argument was supplied, or (that
    # argument being absent) the CLIENT-level policy was itself explicit
    # (`ObservabilityConfig.content_mode_explicit`). See
    # `ObservabilityClient._resolve_trace_scoped_content_mode`.
    content_mode_locked: bool = False
    # B3 (addendum R3, A2 completion): per-entity record of which content
    # fields (`_SAFE_SNAPSHOT_CONTENT_FIELDS`, plus -- finding 1,
    # observations only -- `"error_message"`) have carried real content (or
    # an earlier clearing placeholder) BEFORE the value now held on
    # `trace`/`observations` -- keyed by `_TRACE_CONTENT_KEY` for the trace
    # itself, or by observation id. Deliberately NOT derived from the
    # CURRENT `trace`/`observations` field values: a serialization/
    # redaction failure is caused by the value CURRENTLY held (this call's
    # own, possibly cyclic/hostile write), so reading presence off that same
    # value cannot tell a genuine first write (nothing stored server-side
    # yet -- safe to omit, M1/A2) apart from an update over content a PRIOR
    # call already delivered (must be cleared with the placeholder, never
    # silently left stored -- B3's confirmed defect). Populated by
    # `ObservabilityClient.record_observation`/`end_trace` from each
    # merge's own PRE-overwrite `existing` value -- mirroring exactly the
    # `existing_value is not None` check `wire_value_for_tightened_update`
    # already performs for the normal (non-failure) tightening path --
    # never from the freshly-gated value that might be the one about to
    # fail. Monotonic: once a field is marked, it is never unmarked.
    #
    # This dict/set structure is itself mutable and lives on the shared,
    # lock-guarded `_TraceState` -- never read directly by the M8/A5
    # fallback machinery below. `_build_fallback_descriptor` snapshots it
    # (as plain booleans, on a frozen `_TraceFallbackDescriptor`/
    # `_ObservationFallbackDescriptor`) while the client lock is held, and
    # that frozen snapshot -- never this dict -- is what threads through to
    # the transport (finding 2/A8/M4 snapshot immutability).
    content_history: dict[str, set[str]] = field(default_factory=dict)
    # Number of snapshots captured from this trace so far (#2609). Advanced by
    # `_build_fallback_descriptor`, under the client lock, so a snapshot's
    # `capture_sequence` is its capture order whatever order the (lock-free)
    # transport submissions later run in. One int per trace; never on the wire.
    snapshot_sequence: int = 0
    # Agent scopes seen inside this trace; decides the scope-derived agent_id
    # that only the final (end_trace) snapshot may carry. Never on the wire.
    scope_ledger: ScopeLedger = field(default_factory=ScopeLedger)
    # Set by `end_trace`; only an ended trace can be evicted from the client's
    # state map (#2604). Reset by a restart (`start_trace` builds a new state).
    ended: bool = False
    # True while an entry for this state sits in the client's eviction queue,
    # so repeated `end_trace` calls cannot enqueue it twice.
    eviction_scheduled: bool = False

    def mark_content_history(self, entity_key: str, field_name: str) -> None:
        self.content_history.setdefault(entity_key, set()).add(field_name)

    def had_prior_content(self, entity_key: str, field_name: str) -> bool:
        return field_name in self.content_history.get(entity_key, ())

    def to_payload(self) -> dict[str, Any]:
        ordered: dict[str, ObservationDTO] = {}
        for observation_id in self.observation_order:
            observation = self.observations[observation_id]
            ordered[observation_id] = replace(observation, children=[])

        roots: list[ObservationDTO] = []
        for observation_id in self.observation_order:
            observation = ordered[observation_id]
            parent_id = observation.parent_observation_id
            if parent_id and parent_id in ordered:
                ordered[parent_id].children.append(observation)
            else:
                roots.append(observation)

        payload_trace = replace(self.trace, observations=roots)
        return cast(dict[str, Any], payload_trace.to_dict())


@dataclass(frozen=True)
class _ObservationFallbackDescriptor:
    """Detached, per-observation data the M8/A5 fallback needs, frozen off
    a live `ObservationDTO`/`_TraceState` at snapshot-enqueue time (see
    `_build_fallback_descriptor`) -- part of the fix where a fallback consulting the live,
    mutable `_TraceState` after the client lock that guards it was
    released could observe a LATER call's mutation (a completed status, a
    newly added observation) instead of the state as of the snapshot this
    fallback is standing in for.

    `id`/`status`/`type`/`parent_observation_id` are read once, off the
    schema-validated DTO, at build time -- SDK-controlled fields, never
    arbitrary caller content (see `_safe_allowlisted_observation_snapshot`).
    """

    id: str
    status: str
    type: str | None
    parent_observation_id: str | None
    had_prior_input_data: bool
    had_prior_output_data: bool
    # Whether this
    # observation previously carried real `error_message` text, so the
    # fallback can clear it (inside `metadata`, matching the normal
    # non-fallback update path) instead of omitting `metadata` -- and thus
    # the prior error text -- entirely.
    had_prior_error_message: bool


@dataclass(frozen=True)
class _TraceFallbackDescriptor:
    """Detached, snapshot-specific data the M8/A5 trace-level fallback
    needs (`_safe_allowlisted_snapshot`), frozen off a live `_TraceState` at
    snapshot-enqueue time, under the client lock (see
    `_build_fallback_descriptor`). Passed through
    `ObservabilityClient._submit_trace_snapshot` and
    `_SyncBatchTransport.submit`/`_prepare_payload` in place of the live
    `_TraceState` itself -- those calls can run well after the enqueueing call
    released the lock, so consulting the live, shared `_TraceState` there
    risked reflecting a LATER call's mutation of the SAME trace rather than
    the state as of this particular snapshot. Everything on this dataclass
    is a plain, already-copied scalar/tuple -- no reference back into
    `_TraceState` survives past `_build_fallback_descriptor`.
    """

    had_prior_input_data: bool
    had_prior_output_data: bool
    observations: tuple[_ObservationFallbackDescriptor, ...]
    # Capture order of this snapshot within its trace (#2609); the transport
    # uses it, never the payload, to tell which of two same-trace snapshots is
    # the newer. ``None`` for a caller that did not capture it (no ordering
    # known: the transport then treats the submission as the newest).
    capture_sequence: int | None = None


def _build_fallback_descriptor(state: _TraceState) -> _TraceFallbackDescriptor:
    """Freeze everything `_safe_allowlisted_snapshot`/
    `_safe_allowlisted_observation_snapshot` might later need from `state`
    into a `_TraceFallbackDescriptor` (finding 2/A8/M4 snapshot
    immutability).

    MUST be called while holding the lock that guards `state` (every call
    site does -- `_safe_trace_snapshot` is only ever invoked under
    `ObservabilityClient._lock`). Reads happen exactly once, at this
    instant; nothing returned here holds a reference back into `state` --
    every field is a plain `str`/`bool`/`None`, or a `tuple` of the same --
    so no caller downstream (in particular the transport, which runs
    OUTSIDE this lock and potentially long after it was released) can ever
    observe a subsequent mutation of the live, shared `_TraceState`.
    """
    observation_descriptors: list[_ObservationFallbackDescriptor] = []
    for observation_id in state.observation_order:
        observation = state.observations.get(observation_id)
        if observation is None:
            continue
        observation_type = getattr(observation.type, "value", None)
        parent_observation_id = observation.parent_observation_id
        observation_descriptors.append(
            _ObservationFallbackDescriptor(
                id=observation.id,
                status=observation.status,
                type=(
                    observation_type
                    if isinstance(observation_type, str) and observation_type
                    else None
                ),
                parent_observation_id=(
                    parent_observation_id
                    if isinstance(parent_observation_id, str) and parent_observation_id
                    else None
                ),
                had_prior_input_data=state.had_prior_content(
                    observation.id, "input_data"
                ),
                had_prior_output_data=state.had_prior_content(
                    observation.id, "output_data"
                ),
                had_prior_error_message=state.had_prior_content(
                    observation.id, "error_message"
                ),
            )
        )
    state.snapshot_sequence += 1
    return _TraceFallbackDescriptor(
        had_prior_input_data=state.had_prior_content(_TRACE_CONTENT_KEY, "input_data"),
        had_prior_output_data=state.had_prior_content(
            _TRACE_CONTENT_KEY, "output_data"
        ),
        observations=tuple(observation_descriptors),
        capture_sequence=state.snapshot_sequence,
    )


def _safe_trace_snapshot(
    state: _TraceState,
) -> tuple[dict[str, Any], _TraceFallbackDescriptor]:
    """Build and redact an outbound trace snapshot from `state`, catching
    failures from EITHER step: `state.to_payload()` (DTO serialization --
    e.g. `to_jsonable`'s unbounded recursion on a cyclic caller-supplied
    `metadata`/`input_data`/`output_data` structure, which raises
    `RecursionError`) or `_redact_trace_payload` itself (M8/A5). Without this,
    `state.to_payload()` used to run OUTSIDE `_redact_trace_payload`'s own
    try/except at every call site, so a cyclic structure raised straight out
    of `start_trace`/`record_observation`/`end_trace`/`flush`/`close` and
    into the caller's application -- exactly what M8 forbids (finding 4's
    confirmed probe: a cyclic `input_data` raised `RecursionError` through
    this path).

    Also builds and
    returns the `_TraceFallbackDescriptor` for this snapshot
    (`_build_fallback_descriptor`), frozen from `state` at THIS instant --
    every one of this function's three call sites holds
    `ObservabilityClient._lock` while calling it. Callers must thread the
    returned descriptor (never `state` itself) into
    `ObservabilityClient._submit_trace_snapshot`, so any later fallback --
    including one firing in the transport, outside this lock and possibly
    long after it was released -- reflects this snapshot, never whatever
    `state` mutates into by then.
    """
    return _capture_trace_snapshot(state)


# Test seam: called between serialization and the identity decision.
_after_serialize_hook: Callable[[], None] | None = None


def _stamp_scope_agent_id(state: _TraceState, payload: dict[str, Any]) -> None:
    """Set or clear the generated ``execution_context.agent_id`` on ``payload``.

    ``payload`` is the detached, already-serialized snapshot; the live trace is
    never modified, so a generated id is "withdrawn" simply by not stamping it
    again. Only a finalized (ended) trace may carry a scope-derived id, and only
    the one the sealed, untainted ledger yields. An explicit ``agent_id``
    (per-trace, ``TRAIGENT_AGENT_ID``, including an explicit clear) is present in
    the payload already and is never touched. Nothing here runs caller code, so
    it is safe under the ledger lock.
    """
    hook = _after_serialize_hook
    if hook is not None:
        hook()
    if state.trace.ended_at is None:
        return

    def stamp(key: str | None) -> None:
        if key is None:
            return
        context = payload.get("execution_context")
        if context is None:
            payload["execution_context"] = {"schema_version": "1.0", "agent_id": key}
        elif isinstance(context, dict) and "agent_id" not in context:
            context["agent_id"] = key

    state.scope_ledger.finalize(stamp)


def _capture_trace_snapshot(
    state: _TraceState,
) -> tuple[dict[str, Any], _TraceFallbackDescriptor]:
    descriptor = _build_fallback_descriptor(state)
    try:
        payload = state.to_payload()
    except Exception:
        logger.exception(
            "Observability trace/observation serialization failed for "
            "trace '%s'; sending a minimal safe-allowlisted snapshot rather "
            "than raising into the caller's code path.",
            state.trace.id,
        )
        return (
            _safe_allowlisted_snapshot_or_static_fallback(
                {
                    key: value
                    for key, value in (
                        ("id", state.trace.id),
                        ("name", state.trace.name),
                        ("status", state.trace.status),
                    )
                    if value is not None
                },
                state=descriptor,
            ),
            descriptor,
        )
    _stamp_scope_agent_id(state, payload)
    return _redact_trace_payload(payload, state=descriptor), descriptor


def _count_payload_observations(payload: dict[str, Any]) -> int:
    """Count observations the way the backend's ingest cap does: every
    observation object, nested ``children`` included (#2600)."""
    count = 0
    pending: list[Any] = [payload.get("observations")]
    while pending:
        observations = pending.pop()
        if not isinstance(observations, list):
            continue
        for observation in observations:
            if isinstance(observation, dict):
                count += 1
                pending.append(observation.get("children"))
    return count


@dataclass(frozen=True)
class _BufferedPayload:
    payload: dict[str, Any]
    payload_bytes: int
    observation_count: int = 0
    # Content digest of the encoded payload; lets the transport skip a
    # snapshot identical to the one it last delivered for the item (#2604).
    digest: bytes = b""
    # Capture order of the snapshot (see `_TraceFallbackDescriptor.
    # capture_sequence`), resolved to an int at admission (#2609).
    generation: int = 0


class _SyncBatchTransport:
    """Thread-safe batch transport that avoids cross-thread asyncio coordination.

    Failed snapshots remain in a bounded memory outbox for a later explicit
    flush or submission after connectivity recovers. A submitted snapshot
    identical to the one last delivered for the same item is not queued again
    (counted in ``skipped_unchanged_items``), so repeated flushes of unchanged
    traces cost no requests (#2604). This transport does not
    detect reconnects; exhausted attempts stop the current drain and its timer.

    ``health_callback`` receives snapshots after transport state has been
    updated and after transport locks have been released. A dedicated daemon
    dispatcher delivers snapshots in state-update order. Callback exceptions are
    swallowed so observability delivery is never disrupted, and user callbacks
    never extend a caller's flush or close deadline. Close drains snapshots
    accepted before dispatcher shutdown; later snapshots increment
    ``dropped_health_events``.
    """

    def __init__(
        self,
        sender: Callable[[list[dict[str, Any]]], dict[str, Any] | None],
        *,
        batch_size: int,
        max_buffer_age: float,
        max_queue_size: int,
        max_batch_bytes: int,
        health_callback: Callable[[str, dict[str, Any]], None] | None = None,
        max_traces_per_request: int | None = INGEST_MAX_TRACES_PER_REQUEST,
        max_observations_per_request: int | None = INGEST_MAX_OBSERVATIONS_PER_REQUEST,
    ) -> None:
        self._sender = sender
        self._health_callback = health_callback
        self.batch_size = batch_size
        self.max_buffer_age = max_buffer_age
        self.max_queue_size = max_queue_size
        self.max_batch_bytes = max_batch_bytes
        # Backend ingest caps (#2600); None disables a cap, for a custom
        # sender whose sink is not the Traigent backend.
        for name, cap in (
            ("max_traces_per_request", max_traces_per_request),
            ("max_observations_per_request", max_observations_per_request),
        ):
            if cap is not None and cap <= 0:
                raise ValueError(f"{name} must be greater than 0 or None")
        self.max_traces_per_request = max_traces_per_request
        self.max_observations_per_request = max_observations_per_request
        # A client-side timeout against a slow-but-working server must not count
        # toward the breaker (#2599): ingest upserts by trace id, so the retry is
        # safe, and a timeout alone does not show the server is down.
        self._retry_handler = RetryHandler(
            replace(
                CLOUD_API_RETRY_CONFIG,
                callback_on_retry=self._record_retry,
                circuit_breaker_ignore_failure=_is_client_timeout,
            )
        )
        self._buffer: OrderedDict[str, _BufferedPayload] = OrderedDict()
        self._lock = threading.RLock()
        self._send_lock = threading.Lock()
        self._health_event_queue: deque[
            tuple[int, str, dict[str, Any]] | threading.Event | _HealthDispatchStop
        ] = deque()
        self._health_event_condition = threading.Condition(self._lock)
        self._next_health_event_sequence = 0
        self._dropped_health_events = 0
        self._health_dispatch_closed = False
        self._timer: threading.Timer | None = None
        self._delivery_failure_epoch = 0
        self._flush_thread: threading.Thread | None = None
        self._closed = False
        self._errors: list[str] = []
        self._warnings: list[str] = []
        self._inflight_items = 0
        # Item ids whose in-flight snapshot was superseded by a rejected update
        # (#2601): a failed send must not retain them for another attempt.
        self._superseded_inflight_ids: set[str] = set()
        # Digests of the in-flight batch's snapshots, and of the last snapshot
        # delivered per item (LRU, bounded by max_queue_size). Forgetting a
        # delivered digest only costs a redundant resend, never a lost update.
        self._inflight_digests: dict[str, bytes] = {}
        # Generation of the in-flight batch's snapshots (bounded by the batch,
        # cleared with it): a rejected update only supersedes a queued or
        # in-flight snapshot of a lower generation (#2609).
        self._inflight_generations: dict[str, int] = {}
        self._delivered_digests: OrderedDict[str, bytes] = OrderedDict()
        # Items whose latest snapshot the backend accepted (#2604). Unlike
        # `_delivered_digests` it survives a response with warnings, so the
        # client can tell "confirmed" from "pending". Not an LRU: an entry
        # leaves when a newer snapshot is buffered or the client releases it
        # (`release_settled`, on eviction), so settlement can never expire
        # under sustained traffic and strand an evictable trace.
        self._settled_items: set[str] = set()
        self._oldest_inflight_started_at: float | None = None
        self._active_send_completion: threading.Event | None = None
        self._stats: dict[str, int] = {
            "submitted_items": 0,
            "sent_items": 0,
            "dropped_items": 0,
            "successful_batches": 0,
            "failed_batches": 0,
            "pending_items": 0,
            "background_flushes": 0,
            "retry_attempts": 0,
            "skipped_unchanged_items": 0,
        }
        self._dropped_by_reason: dict[str, int] = {}
        self._health_dispatcher_thread: threading.Thread | None = None
        if self._health_callback is not None:
            self._health_dispatcher_thread = threading.Thread(
                target=self._dispatch_health_events,
                name="traigent-observability-health-dispatcher",
                daemon=True,
            )
            self._health_dispatcher_thread.start()

    def submit(
        self,
        item_id: str,
        payload: dict[str, Any],
        *,
        deadline: float | None = None,
        state: _TraceFallbackDescriptor | None = None,
    ) -> bool:
        # Capture order of this snapshot within its item, or None when the
        # caller (a direct transport user) has none: it is then the newest.
        capture_sequence = getattr(state, "capture_sequence", None)
        try:
            buffered_payload = self._prepare_payload(payload, state=state)
        except TypeError as exc:
            message = (
                "observability payload for item "
                f"'{item_id}' is not JSON serializable: {exc}; dropped"
            )
            self._reject_update(
                "payload_not_json_serializable",
                message,
                item_id=item_id,
                generation=capture_sequence,
            )
            logger.warning(
                "Observability transport dropped payload '%s' because "
                "it is not JSON serializable: %s",
                item_id,
                exc,
            )
            return False

        item_bytes = self._batch_payload_size_from_body(buffered_payload.payload_bytes)
        if item_bytes > self.max_batch_bytes:
            message = (
                "observability payload for item "
                f"'{item_id}' is {item_bytes} bytes, exceeding "
                f"max_batch_bytes={self.max_batch_bytes}; dropped"
            )
            self._reject_update(
                "payload_too_large",
                message,
                item_id=item_id,
                item_bytes=item_bytes,
                max_batch_bytes=self.max_batch_bytes,
                generation=capture_sequence,
            )
            logger.warning(
                "Observability transport dropped payload '%s' because "
                "its encoded size %d exceeds max_batch_bytes=%d",
                item_id,
                item_bytes,
                self.max_batch_bytes,
            )
            return False

        max_observations = self.max_observations_per_request
        if (
            max_observations is not None
            and buffered_payload.observation_count > max_observations
        ):
            message = (
                "observability payload for item "
                f"'{item_id}' has {buffered_payload.observation_count} "
                "observations, exceeding the backend's limit of "
                f"{max_observations} per ingest request; dropped"
            )
            self._reject_update(
                "too_many_observations",
                message,
                item_id=item_id,
                observation_count=buffered_payload.observation_count,
                max_observations=max_observations,
                generation=capture_sequence,
            )
            logger.warning(
                "Observability transport dropped payload '%s' because it has %d "
                "observations, above the backend limit of %d per request",
                item_id,
                buffered_payload.observation_count,
                max_observations,
            )
            return False

        drop_event: tuple[str, dict[str, Any]] | None = None
        if not self._acquire_lock_until(deadline):
            return False
        try:
            if self._closed:
                drop_event = self._record_drop_locked(
                    "transport_closed",
                    f"transport closed; dropped payload for item '{item_id}'",
                    item_id=item_id,
                )
            elif (
                item_id not in self._buffer
                and self._delivered_digests.get(item_id) == buffered_payload.digest
                and self._inflight_digests.get(item_id, buffered_payload.digest)
                == buffered_payload.digest
            ):
                # Already delivered exactly this snapshot, and nothing different
                # is queued or in flight that would land after it: re-sending it
                # would only repeat the same write.
                self._stats["skipped_unchanged_items"] += 1
            elif (
                item_id not in self._buffer and len(self._buffer) >= self.max_queue_size
            ):
                drop_event = self._record_drop_locked(
                    "queue_full",
                    f"transport queue full; dropped payload for item '{item_id}'",
                    item_id=item_id,
                )
                # A full backlog (possibly a retained outbox whose timer was
                # cancelled after a failure) still gets one bounded retry
                # opportunity from this submission.
                self._ensure_timer_locked()
            else:
                self._buffer.pop(item_id, None)
                self._settled_items.discard(item_id)
                self._buffer[item_id] = replace(
                    buffered_payload,
                    generation=self._resolve_generation_locked(
                        item_id, capture_sequence
                    ),
                )
                self._stats["submitted_items"] += 1
                self._stats["pending_items"] = len(self._buffer)

                if len(self._buffer) >= self.batch_size:
                    self._ensure_flush_thread_locked()
                # Always keep an age timer armed while the buffer is non-empty. On the
                # batch-size path this is a backstop: if the flush thread is between its
                # buffer-empty check and exit when this item arrives, is_alive() still
                # reads True so _ensure_flush_thread_locked spawns nothing — the timer
                # then guarantees the tail item is not stranded until the next
                # submit/close. When the thread does drain everything, the leftover
                # timer fires once, finds an empty buffer, and returns harmlessly.
                self._ensure_timer_locked()
        finally:
            self._lock.release()

        if drop_event is not None:
            self._emit_health_events([drop_event], deadline=deadline)
            return False

        return True

    def is_settled(self, item_id: str) -> bool:
        """True iff the backend accepted this item and nothing newer for it is
        queued or in flight. A failed send leaves the item buffered, so a
        failed or unconfirmed item is never settled (#2604)."""
        with self._lock:
            return (
                item_id in self._settled_items
                and item_id not in self._buffer
                and item_id not in self._inflight_digests
            )

    def release_settled(self, item_id: str) -> None:
        """Forget an item's settled mark once the client has evicted it."""
        with self._lock:
            self._settled_items.discard(item_id)

    def flush(
        self,
        timeout: float | None = None,
        *,
        warn_on_deadline_expiry: bool | None = None,
        timer_epoch: int | None = None,
    ) -> BatchFlushResult:
        """Flush queued payloads, waiting no longer than ``timeout`` seconds.

        Results retain cumulative failure/error history, even after a later
        flush successfully delivers the previously retained snapshots.
        """
        deadline = None if timeout is None else time.monotonic() + max(0.0, timeout)
        if warn_on_deadline_expiry is None:
            warn_on_deadline_expiry = timeout != 0
        if not self._cancel_timer(deadline, timer_epoch=timer_epoch):
            return self._deadline_expiry_result(
                "Observability transport flush deadline exceeded while acquiring "
                "the timer state lock"
            )
        completed = self._send_available(
            deadline=deadline,
            warn_on_deadline_expiry=warn_on_deadline_expiry,
            timer_epoch=timer_epoch,
        )
        return self._build_result(
            deadline,
            deadline_expired=not completed,
            warn_on_deadline_expiry=warn_on_deadline_expiry,
        )

    def close(
        self,
        timeout: float | None = None,
        *,
        warn_on_deadline_expiry: bool | None = None,
    ) -> BatchFlushResult:
        """Prevent new submissions, flush, and stop health-event dispatch."""
        deadline = None if timeout is None else time.monotonic() + max(0.0, timeout)
        if not self._acquire_lock_until(deadline):
            return self._deadline_expiry_result(
                "Observability transport close deadline exceeded while acquiring "
                "the transport state lock"
            )
        try:
            self._closed = True
        finally:
            self._lock.release()
        result = self.flush(
            timeout=self._remaining_time(deadline),
            warn_on_deadline_expiry=warn_on_deadline_expiry,
        )
        self._stop_health_dispatcher(deadline)
        return result

    def get_stats(self, timeout: float | None = None) -> dict[str, Any]:
        """Return a thread-safe transport health snapshot.

        ``dropped_by_reason`` identifies local data loss,
        ``dropped_health_events`` counts callback snapshots discarded because
        the dispatch queue overflowed or had stopped, ``queue_depth`` is the
        number of buffered payloads, and ``inflight_items`` covers payloads that
        have left the queue but whose sender has not completed.
        """
        deadline = None if timeout is None else time.monotonic() + max(0.0, timeout)
        if not self._acquire_lock_until(deadline):
            return self._stats_snapshot_without_lock()
        try:
            snapshot: dict[str, Any] = dict(self._stats)
            snapshot["queue_depth"] = len(self._buffer)
            snapshot["inflight_items"] = self._inflight_items
            snapshot["send_in_progress"] = (
                self._active_send_completion is not None or self._send_lock.locked()
            )
            snapshot["oldest_inflight_age_seconds"] = (
                None
                if self._oldest_inflight_started_at is None
                else max(0.0, time.monotonic() - self._oldest_inflight_started_at)
            )
            snapshot["dropped_by_reason"] = dict(self._dropped_by_reason)
            snapshot["dropped_health_events"] = self._dropped_health_events
            snapshot["errors"] = list(self._errors)
            snapshot["warnings"] = list(self._warnings)
            return snapshot
        finally:
            self._lock.release()

    def _ensure_timer_locked(self) -> None:
        if self._timer is not None and self._timer.is_alive():
            return
        self._timer = threading.Timer(
            self.max_buffer_age,
            self._flush_from_timer,
            args=(self._delivery_failure_epoch,),
        )
        self._timer.daemon = True
        self._timer.start()

    def _ensure_flush_thread_locked(self) -> None:
        if self._flush_thread is not None and self._flush_thread.is_alive():
            return
        self._flush_thread = threading.Thread(
            target=self._flush_from_worker,
            args=(self._delivery_failure_epoch,),
            name="traigent-observability-flush",
            daemon=True,
        )
        self._stats["background_flushes"] += 1
        self._flush_thread.start()

    def _cancel_timer_locked(self) -> None:
        timer = self._timer
        self._timer = None
        if timer is not None and timer is not threading.current_thread():
            timer.cancel()

    def _cancel_timer(
        self, deadline: float | None = None, *, timer_epoch: int | None = None
    ) -> bool:
        if not self._acquire_lock_until(deadline):
            return False
        try:
            # A stale timer/worker must not cancel a timer armed by a newer
            # submission after the failure it predates.
            if timer_epoch is None or timer_epoch == self._delivery_failure_epoch:
                self._cancel_timer_locked()
            return True
        finally:
            self._lock.release()

    def _flush_from_timer(self, epoch: int | None = None) -> None:
        # A timer armed before a delivery failure must not resend the retained
        # outbox: only a later explicit flush or fresh submission may retry.
        if epoch is not None and epoch != self._delivery_failure_epoch:
            return
        try:
            self.flush(timer_epoch=epoch)
        except Exception as exc:
            self._append_error(f"timer flush failed: {exc}")

    def _flush_from_worker(self, epoch: int | None = None) -> None:
        try:
            self._send_available(timer_epoch=epoch)
        except Exception as exc:
            self._append_error(f"background flush failed: {exc}")

    def _send_available(
        self,
        *,
        deadline: float | None = None,
        warn_on_deadline_expiry: bool = True,
        timer_epoch: int | None = None,
    ) -> bool:
        events: list[tuple[str, dict[str, Any]]] = []
        caller_released = threading.Event()
        if not self._acquire_send_lock(deadline):
            self._expire_send(events, deadline, warn_on_deadline_expiry)
            self._emit_health_events(events, deadline=deadline)
            return False

        try:
            while True:
                if self._send_blocked(deadline):
                    return self._expire_send(events, deadline, warn_on_deadline_expiry)
                if not self._acquire_lock_until(deadline):
                    return False
                try:
                    done, batch_items = self._collect_batch_locked(timer_epoch)
                finally:
                    self._lock.release()
                if done:
                    return True

                if not batch_items:
                    continue
                outcome = self._deliver_collected_batch(
                    batch_items,
                    deadline,
                    warn_on_deadline_expiry,
                    caller_released,
                    events,
                )
                if outcome is not None:
                    return outcome
        finally:
            self._send_lock.release()
            caller_released.set()
            self._emit_health_events(events, deadline=deadline)

    def _acquire_send_lock(self, deadline: float | None) -> bool:
        if deadline is None:
            return self._send_lock.acquire()
        return self._send_lock.acquire(timeout=max(0.0, deadline - time.monotonic()))

    def _send_blocked(self, deadline: float | None) -> bool:
        if not self._wait_for_active_send(deadline):
            return True
        return deadline is not None and time.monotonic() >= deadline

    def _expire_send(
        self,
        events: list[tuple[str, dict[str, Any]]],
        deadline: float | None,
        warn_on_deadline_expiry: bool,
    ) -> bool:
        if warn_on_deadline_expiry:
            self._note_deadline_expiry(events, deadline)
        return False

    def _deliver_collected_batch(
        self,
        batch_items: list[tuple[str, dict[str, Any]]],
        deadline: float | None,
        warn_on_deadline_expiry: bool,
        caller_released: threading.Event,
        events: list[tuple[str, dict[str, Any]]],
    ) -> bool | None:
        """Send one collected batch; ``None`` means keep draining."""
        if deadline is None:
            try:
                delivered, batch_events = self._send_batch(batch_items)
                events.extend(batch_events)
            finally:
                self._complete_inflight_batch(len(batch_items))
            if not delivered:
                # Keep failed snapshots for a later caller-driven retry,
                # rather than retrying this outbox forever in this drain.
                return True
            return None
        timed_result = self._send_batch_until(batch_items, deadline, caller_released)
        if timed_result is None:
            return self._expire_send(events, deadline, warn_on_deadline_expiry)
        delivered, queued_events = timed_result
        events.extend(queued_events)
        if not delivered:
            return True
        return None

    def _note_deadline_expiry(
        self, events: list[tuple[str, dict[str, Any]]], deadline: float | None
    ) -> None:
        event = self._record_deadline_expiry(deadline)
        if event is not None:
            events.append(event)

    def _collect_batch_locked(
        self, timer_epoch: int | None
    ) -> tuple[bool, list[tuple[str, dict[str, Any]]]]:
        """Pop the next batch off the outbox; the caller holds ``self._lock``.

        Returns ``(done, batch_items)``: ``done`` means the drain is finished
        and ``_send_available`` returns True without sending anything.
        """
        if timer_epoch is not None and (timer_epoch != self._delivery_failure_epoch):
            # A failure retained the outbox while this timer drain
            # waited for send serialization; do not resend it.
            return True, []
        if not self._buffer:
            self._stats["pending_items"] = self._inflight_items
            return True, []

        batch_items: list[tuple[str, dict[str, Any]]] = []
        batch_body_bytes = 0
        batch_observations = 0
        # Every request fits the backend's per-request caps
        # (#2600), whatever batch_size the caller configured.
        max_batch_items = min(self.batch_size, len(self._buffer))
        if self.max_traces_per_request is not None:
            max_batch_items = min(max_batch_items, self.max_traces_per_request)
        for _ in range(max_batch_items):
            item_id, buffered = next(iter(self._buffer.items()))
            projected_body_bytes = batch_body_bytes + buffered.payload_bytes
            if batch_items:
                projected_body_bytes += _TRACE_BATCH_ITEM_SEPARATOR_BYTES
            projected_bytes = self._batch_payload_size_from_body(projected_body_bytes)
            projected_observations = batch_observations + buffered.observation_count
            if batch_items and (
                projected_bytes > self.max_batch_bytes
                or (
                    self.max_observations_per_request is not None
                    and projected_observations > self.max_observations_per_request
                )
            ):
                break
            self._buffer.popitem(last=False)
            self._inflight_digests[item_id] = buffered.digest
            self._inflight_generations[item_id] = buffered.generation
            # Once a send starts, the backend may end up holding
            # this snapshot whatever the outcome (a timeout after
            # commit, an exception escaping the send), so the
            # older delivered one is forgotten until this send
            # succeeds. The cost is at most one redundant resend.
            self._delivered_digests.pop(item_id, None)
            batch_items.append((item_id, buffered.payload))
            batch_body_bytes = projected_body_bytes
            batch_observations = projected_observations
        self._inflight_items += len(batch_items)
        if self._oldest_inflight_started_at is None:
            self._oldest_inflight_started_at = time.monotonic()
        self._stats["pending_items"] = len(self._buffer) + self._inflight_items
        return False, batch_items

    def _send_batch_until(
        self,
        batch_items: list[tuple[str, dict[str, Any]]],
        deadline: float,
        caller_released: threading.Event,
    ) -> tuple[bool, list[tuple[str, dict[str, Any]]]] | None:
        completed = threading.Event()
        events: list[tuple[str, dict[str, Any]]] = []
        delivered = False

        if not self._acquire_lock_until(deadline):
            return None
        try:
            self._active_send_completion = completed
        finally:
            self._lock.release()

        def send() -> None:
            nonlocal delivered
            try:
                delivered, batch_events = self._send_batch(batch_items)
                events.extend(batch_events)
            except Exception as exc:
                # Flush and close are best effort; background sender failures must
                # be visible in the result but must never escape user code.
                self._append_error(f"batch delivery failed unexpectedly: {exc}")
            finally:
                self._complete_inflight_batch(len(batch_items), completed)
                completed.set()

        threading.Thread(
            target=send,
            name="traigent-observability-send",
            daemon=True,
        ).start()
        if completed.wait(timeout=max(0.0, deadline - time.monotonic())):
            return delivered, events

        threading.Thread(
            target=self._emit_orphan_health_events,
            args=(completed, caller_released, events),
            name="traigent-observability-send-events",
            daemon=True,
        ).start()
        return None

    def _wait_for_active_send(self, deadline: float | None) -> bool:
        """Wait for a timed-out sender without ever starting a concurrent sender."""
        while True:
            if not self._acquire_lock_until(deadline):
                return False
            try:
                completion = self._active_send_completion
            finally:
                self._lock.release()
            if completion is None:
                return True
            if deadline is None:
                completion.wait()
                continue
            remaining = max(0.0, deadline - time.monotonic())
            if remaining <= 0 or not completion.wait(timeout=remaining):
                return False

    def _emit_orphan_health_events(
        self,
        completed: threading.Event,
        caller_released: threading.Event,
        events: list[tuple[str, dict[str, Any]]],
    ) -> None:
        completed.wait()
        caller_released.wait()
        self._emit_health_events(events)

    def _complete_inflight_batch(
        self, item_count: int, completion: threading.Event | None = None
    ) -> None:
        with self._lock:
            self._inflight_items -= item_count
            if self._inflight_items == 0:
                self._oldest_inflight_started_at = None
                self._superseded_inflight_ids.clear()
                self._inflight_digests.clear()
                self._inflight_generations.clear()
            self._stats["pending_items"] = len(self._buffer) + self._inflight_items
            if completion is not None and self._active_send_completion is completion:
                self._active_send_completion = None

    def _record_deadline_expiry(
        self, deadline: float | None
    ) -> tuple[str, dict[str, Any]] | None:
        if not self._acquire_lock_until(deadline):
            return None
        try:
            remaining = len(self._buffer) + self._inflight_items
            self._stats["pending_items"] = remaining
            message = (
                "Observability transport flush deadline exceeded with "
                f"{remaining} items remaining"
            )
            self._warnings.append(message)
            if len(self._warnings) > 50:
                self._warnings = self._warnings[-50:]
            event = self._queue_health_event_locked("warning", {"message": message})
        finally:
            self._lock.release()
        logger.warning(message)
        return event

    def _prepare_payload(
        self, payload: dict[str, Any], *, state: _TraceFallbackDescriptor | None = None
    ) -> _BufferedPayload:
        # Direct transport callers bypass ObservabilityClient's trace redaction.
        # Preserve credential-key-name redaction (c4d874c3): this is a confirmed
        # egress path carrying arbitrary user keys, so opt in as the other two
        # egress call sites do. `_redact_trace_payload` scopes the key-name
        # sweep to the content bags so a directly-submitted trace/observation
        # payload's own `input_tokens`/`output_tokens`/`total_tokens` fields
        # are never mistaken for a `token`-rooted credential key (M5).
        # B3: `state` (a `_TraceFallbackDescriptor`, never the live
        # `_TraceState` -- finding 2), when the caller has it (see
        # `submit`), lets a redaction failure on THIS (already-once-
        # redacted) pass still reach the content-history-aware fallback
        # instead of a state-blind one -- see
        # `ObservabilityClient._submit_trace_snapshot`.
        redacted_payload = _redact_trace_payload(payload, state=state)
        return self._buffered_payload(copy.deepcopy(redacted_payload))

    @staticmethod
    def _buffered_payload(payload: dict[str, Any]) -> _BufferedPayload:
        encoded = json.dumps(payload).encode("utf-8")
        return _BufferedPayload(
            payload=payload,
            payload_bytes=len(encoded),
            observation_count=_count_payload_observations(payload),
            digest=hashlib.blake2b(encoded, digest_size=16).digest(),
        )

    @staticmethod
    def _batch_payload_size_from_body(body_bytes: int) -> int:
        return _TRACE_BATCH_PREFIX_BYTES + body_bytes + _TRACE_BATCH_SUFFIX_BYTES

    def _record_batch_success(
        self,
        batch_items: list[tuple[str, dict[str, Any]]],
        payloads: list[dict[str, Any]],
        result: Any,
        events: list[tuple[str, dict[str, Any]]],
    ) -> list[tuple[str, dict[str, Any]]]:
        raw_warnings = (
            result.result.get("warnings") if isinstance(result.result, dict) else None
        )
        # Same tolerance as `_log_ingest_warnings_from_data`: a malformed
        # warnings value must not raise out of a flush.
        response_warnings = raw_warnings if isinstance(raw_warnings, list) else []
        for warning in response_warnings:
            events.append(self._append_warning(str(warning)))
        with self._lock:
            self._stats["successful_batches"] += 1
            self._stats["sent_items"] += len(payloads)
            for item_id, _ in batch_items:
                digest = self._inflight_digests.pop(item_id, None)
                if item_id not in self._buffer:
                    self._settled_items.add(item_id)
                if digest is None or response_warnings:
                    # A response with warnings (e.g. an unresolved prompt
                    # link) may be resolved by a later re-send, so it does
                    # not count as delivered for dedup.
                    self._delivered_digests.pop(item_id, None)
                    continue
                self._delivered_digests[item_id] = digest
                self._delivered_digests.move_to_end(item_id)
            while len(self._delivered_digests) > self.max_queue_size:
                self._delivered_digests.popitem(last=False)

        return events

    def _record_batch_failure(
        self,
        batch_items: list[tuple[str, dict[str, Any]]],
        payloads: list[dict[str, Any]],
        result: Any,
        events: list[tuple[str, dict[str, Any]]],
    ) -> list[tuple[str, dict[str, Any]]]:
        with self._lock:
            self._stats["failed_batches"] += 1
            # The sender receives already prepared snapshots. Retain those exact
            # payloads; do not re-redact them using mutable live trace state.
            retained = OrderedDict(
                (
                    item_id,
                    replace(
                        self._buffered_payload(payload),
                        generation=self._inflight_generations.get(item_id, 0),
                    ),
                )
                for item_id, payload in batch_items
                if item_id not in self._buffer
                and item_id not in self._superseded_inflight_ids
            )
            # Submissions made during the send are newer and must win on ID
            # collisions. Old retained entries come first for bounded eviction.
            retained.update(self._buffer)
            self._buffer = retained
            while len(self._buffer) > self.max_queue_size:
                item_id, _ = self._buffer.popitem(last=False)
                message = f"observability outbox full; evicted oldest item '{item_id}'"
                events.append(
                    self._record_drop_locked("outbox_full", message, item_id=item_id)
                )
                events.append(self._append_warning(message))
                logger.warning(message)
            # A failed-only outbox must not create an automatic retry storm.
            # A later explicit flush, or fresh submission after reconnection,
            # provides another bounded retry opportunity.
            self._delivery_failure_epoch += 1
            self._cancel_timer_locked()
            dropped_items = self._stats["dropped_items"]
            queue_depth = len(self._buffer)
            event = self._queue_health_event_locked(
                "batch_delivery_failed",
                {
                    "dropped_items": dropped_items,
                    "queue_depth": queue_depth,
                    "message": str(result.error or "batch delivery failed"),
                    "item_count": len(payloads),
                    "trace_ids": [item_id for item_id, _ in batch_items[:20]],
                    "trace_ids_truncated": len(batch_items) > 20,
                },
            )
        self._append_error(str(result.error or "batch delivery failed"))
        events.append(event)
        logger.warning(
            "Observability transport retained failed batch of %d payloads after retries: %s",
            len(payloads),
            result.error,
        )

        return events

    def _send_batch(
        self, batch_items: list[tuple[str, dict[str, Any]]]
    ) -> tuple[bool, list[tuple[str, dict[str, Any]]]]:
        events: list[tuple[str, dict[str, Any]]] = []
        payloads = [payload for _, payload in batch_items]
        result = self._retry_handler.execute_with_result(self._sender, payloads)

        if result.success:
            return True, self._record_batch_success(
                batch_items, payloads, result, events
            )
        return False, self._record_batch_failure(batch_items, payloads, result, events)

    def _build_result(
        self,
        deadline: float | None = None,
        *,
        deadline_expired: bool = False,
        warn_on_deadline_expiry: bool = True,
    ) -> BatchFlushResult:
        if not self._acquire_lock_until(deadline):
            return self._deadline_expiry_result(
                "Observability transport flush deadline exceeded while acquiring "
                "the transport state lock"
            )
        try:
            remaining = len(self._buffer) + self._inflight_items
            self._stats["pending_items"] = remaining
            warnings = list(self._warnings)
            if (
                warn_on_deadline_expiry
                and deadline_expired
                and not any(
                    "flush deadline exceeded" in warning for warning in warnings
                )
            ):
                warnings.append(
                    "Observability transport flush deadline exceeded before "
                    "delivery completed"
                )
            return BatchFlushResult(
                success=self._stats["failed_batches"] == 0 and remaining == 0,
                items_sent=self._stats["sent_items"],
                items_pending=remaining,
                items_dropped=self._stats["dropped_items"],
                successful_batches=self._stats["successful_batches"],
                failed_batches=self._stats["failed_batches"],
                errors=list(self._errors),
                warnings=warnings,
            )
        finally:
            self._lock.release()

    def _append_error(self, message: str) -> None:
        with self._lock:
            self._errors.append(message)
            if len(self._errors) > 20:
                self._errors = self._errors[-20:]

    def _append_warning(self, message: str) -> tuple[str, dict[str, Any]]:
        with self._lock:
            self._warnings.append(message)
            if len(self._warnings) > 50:
                self._warnings = self._warnings[-50:]
            return self._queue_health_event_locked("warning", {"message": message})

    def _resolve_generation_locked(self, item_id: str, generation: int | None) -> int:
        """Return ``generation``, or for a submission with no capture order a
        value above every snapshot of ``item_id`` currently known (it is the
        newest, as it was before capture order existed)."""
        if generation is not None:
            return generation
        queued = self._buffer.get(item_id)
        return 1 + max(
            queued.generation if queued is not None else 0,
            self._inflight_generations.get(item_id, 0),
        )

    def _reject_update(
        self,
        event_type: str,
        message: str,
        *,
        item_id: str,
        generation: int | None = None,
        **details: Any,
    ) -> None:
        """Drop a submitted snapshot that can never be delivered, and the older
        snapshot of the same item it was meant to replace (#2601).

        The buffer holds the latest snapshot per item. When the latest one is
        rejected (too large, not serializable, too many observations), sending
        the older one still queued would report success for state the
        application has already moved past, so it is evicted too. If that older
        snapshot is already in flight it cannot be recalled, but it is marked so
        a failed send does not put it back into the buffer.

        Only snapshots captured before the rejected one are superseded
        (#2609): submissions run outside the client's capture lock, so a newer
        valid snapshot of the item may already be queued or in flight when an
        older, rejected one reaches here, and it must survive.
        """
        with self._lock:
            rejected = self._resolve_generation_locked(item_id, generation)
            queued = self._buffer.get(item_id)
            evicted = queued is not None and queued.generation < rejected
            if evicted:
                del self._buffer[item_id]
                self._stats["pending_items"] = len(self._buffer) + self._inflight_items
            if self._inflight_generations.get(item_id, rejected) < rejected:
                self._superseded_inflight_ids.add(item_id)
            event = self._record_drop_locked(
                event_type,
                message,
                item_id=item_id,
                evicted_stale_snapshot=evicted,
                **details,
            )
        self._emit_health_events([event])

    def _record_drop_locked(
        self, event_type: str, message: str, **details: Any
    ) -> tuple[str, dict[str, Any]]:
        self._stats["dropped_items"] += 1
        self._dropped_by_reason[event_type] = (
            self._dropped_by_reason.get(event_type, 0) + 1
        )
        self._errors.append(message)
        if len(self._errors) > 20:
            self._errors = self._errors[-20:]
        return self._queue_health_event_locked(
            event_type,
            {
                "drop_reason": event_type,
                "dropped_items": self._stats["dropped_items"],
                "queue_depth": len(self._buffer),
                "message": message,
                **details,
            },
        )

    def _record_retry(self, error: Exception, attempt: int) -> None:
        del error, attempt
        with self._lock:
            self._stats["retry_attempts"] += 1

    def _queue_health_event_locked(
        self, event_type: str, payload: dict[str, Any]
    ) -> tuple[str, dict[str, Any]]:
        """Queue a state snapshot while holding ``_lock`` for ordered dispatch."""
        event = (event_type, payload)
        if self._health_callback is None:
            return event
        if self._health_dispatch_closed:
            self._dropped_health_events += 1
            return event
        self._next_health_event_sequence += 1
        if len(self._health_event_queue) >= 256:
            self._health_event_queue.popleft()
            self._dropped_health_events += 1
        self._health_event_queue.append(
            (self._next_health_event_sequence, event_type, payload)
        )
        return event

    def _emit_health_events(
        self,
        events: list[tuple[str, dict[str, Any]]],
        *,
        deadline: float | None = None,
    ) -> None:
        del events
        if self._health_callback is None:
            return
        if deadline is None:
            acquired = self._health_event_condition.acquire()
        else:
            acquired = self._health_event_condition.acquire(
                timeout=max(0.0, deadline - time.monotonic())
            )
        if not acquired:
            return
        try:
            self._health_event_condition.notify()
        finally:
            self._health_event_condition.release()

    def _dispatch_health_events(self) -> None:
        """Deliver health events from one daemon thread, preserving queue order."""
        while True:
            with self._health_event_condition:
                while not self._health_event_queue:
                    self._health_event_condition.wait()
                queued = self._health_event_queue.popleft()
            if isinstance(queued, _HealthDispatchStop):
                return
            if isinstance(queued, threading.Event):
                queued.set()
                continue
            _, event_type, payload = queued
            callback = self._health_callback
            if callback is None:
                continue
            try:
                callback(event_type, payload)
            except Exception:
                logger.debug("Observability health callback failed", exc_info=True)

    def _stop_health_dispatcher(self, deadline: float | None) -> None:
        """Drain accepted events, then stop the per-transport dispatcher."""
        dispatcher = self._health_dispatcher_thread
        if dispatcher is None:
            return
        if not self._health_event_condition.acquire(
            timeout=(-1 if deadline is None else max(0.0, deadline - time.monotonic()))
        ):
            return
        try:
            if not self._health_dispatch_closed:
                self._health_dispatch_closed = True
                final_drain = threading.Event()
                self._health_event_queue.append(final_drain)
                self._health_event_queue.append(_HEALTH_DISPATCH_STOP)
                self._health_event_condition.notify_all()
        finally:
            self._health_event_condition.release()
        if dispatcher is not threading.current_thread():
            dispatcher.join(timeout=self._remaining_time(deadline))

    @staticmethod
    def _remaining_time(deadline: float | None) -> float | None:
        if deadline is None:
            return None
        return max(0.0, deadline - time.monotonic())

    def _acquire_lock_until(self, deadline: float | None) -> bool:
        if deadline is None:
            return self._lock.acquire()
        return self._lock.acquire(timeout=max(0.0, deadline - time.monotonic()))

    def _stats_snapshot_without_lock(self) -> dict[str, Any]:
        """Return the last state values when a bounded caller cannot lock them.

        All writers update these values while holding ``_lock``. A caller reaches
        this fallback only because another thread still owns that lock, so the
        values are a stable, if necessarily partial, snapshot for the deadline
        result rather than a fabricated success.
        """
        return {
            **self._stats,
            "queue_depth": len(self._buffer),
            "inflight_items": self._inflight_items,
            "send_in_progress": (
                self._active_send_completion is not None or self._send_lock.locked()
            ),
            "oldest_inflight_age_seconds": None,
            "dropped_by_reason": dict(self._dropped_by_reason),
            "dropped_health_events": self._dropped_health_events,
            "errors": list(self._errors),
            "warnings": list(self._warnings),
        }

    def _deadline_expiry_result(self, warning: str) -> BatchFlushResult:
        stats = self._stats_snapshot_without_lock()
        return BatchFlushResult(
            success=False,
            items_sent=int(stats["sent_items"]),
            items_pending=int(stats["pending_items"]),
            items_dropped=int(stats["dropped_items"]),
            successful_batches=int(stats["successful_batches"]),
            failed_batches=int(stats["failed_batches"]),
            errors=list(stats["errors"]),
            warnings=[*stats["warnings"], warning],
        )


DEFAULT_TRACE_STATE_GRACE_SECONDS = 300.0
_MAX_EVICTION_STEPS_PER_PASS = 256
_MAX_EVICTED_TRACE_TOMBSTONES = 4096


@dataclass(frozen=True)
class _EvictedTrace:
    """What a later update needs to restart an evicted trace id (#2604).

    Carries only the trace's identity, its content policy and per-field
    "content was delivered" flags, never its content, so memory per entry is
    small and bounded by the tombstone cap.
    """

    name: str
    environment: str | None
    release: str | None
    snapshot_sequence: int
    resolved_content_mode: str
    content_mode_locked: bool
    content_history: dict[str, frozenset[str]]


class ObservabilityClient:
    """Client for capturing and shipping generic application traces.

    Delivery happens on a background daemon thread. Normal interpreter shutdown
    will trigger a best-effort atexit flush, but abrupt termination (for example
    `SIGKILL` or `os._exit()`) can still drop buffered payloads. Short-lived or
    serverless processes should call `flush()` or `close()` explicitly.

    Ended, delivered traces are evicted after `trace_state_grace_seconds` (#2604).
    A late update to an evicted trace revives it from a tombstone that keeps its
    content policy and delivered-content flags. Only the last 4096 evictions are
    remembered: an update to a trace evicted longer ago than that raises
    `ValueError("Unknown trace_id ...")`. Use a longer grace window if late
    updates are expected.
    """

    def __init__(
        self,
        config: ObservabilityConfig | None = None,
        *,
        sender: Callable[[list[dict[str, Any]]], None] | None = None,
        request_sender: (
            Callable[[str, str, dict[str, Any] | None], dict[str, Any]] | None
        ) = None,
        trace_state_grace_seconds: float = DEFAULT_TRACE_STATE_GRACE_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if trace_state_grace_seconds < 0:
            raise ValueError("trace_state_grace_seconds must be >= 0")
        self.config = config or ObservabilityConfig()
        self._sender_override = sender
        self._request_sender_override = request_sender
        self._trace_states: dict[str, _TraceState] = {}
        # Ended traces awaiting eviction (#2604): (eligible_at, trace_id), in
        # eligible-time order (the clock is monotonic and every entry is
        # appended with now + grace), so a pass only ever looks at the head.
        self._trace_state_grace = float(trace_state_grace_seconds)
        self._clock = clock
        self._eviction_queue: deque[tuple[float, str]] = deque()
        self._evicted_traces: OrderedDict[str, _EvictedTrace] = OrderedDict()
        self._lock = threading.RLock()
        self._snapshot_submission_complete = threading.Event()
        self._snapshot_submission_complete.set()
        self._inflight_snapshot_submissions = 0
        self._closing = False
        # A close may stop new snapshot submissions before the transport can
        # acquire its own state lock. Keep that recoverable state separate
        # from a completed close so a later close() can retry transport.close().
        self._close_pending = False
        self._close_complete = threading.Event()
        self._close_complete.set()
        self._closed = False
        self._offline_notice_logged = False
        self._missing_generation_usage_notice_logged = False
        self._credential_egress_disabled = False
        self._http_opener = _gate_opener(request.build_opener(_NoRedirectHandler))
        self._transport = self._create_transport()
        self._disable_egress_if_credential_missing()

        if self.config.enable_atexit_flush:
            atexit.register(self._atexit_close)

    def _resolve_content_mode(self, content_mode: str | None) -> str:
        """Resolve a `content_mode` override against the CLIENT-level default.

        Used by `start_trace`, which is always establishing a NEW trace's own
        base mode and so has no prior trace state to inherit from -- see
        `_resolve_trace_scoped_content_mode` (A1) for `record_observation`/
        `end_trace`/`add_comment`/`submit_feedback`, which resolve against
        that TRACE's own already-fixed base instead. `None` means "use this
        client's configured default" (`self.config.content_mode`).

        M2: an explicit override may only TIGHTEN an explicitly-set
        client-level result, never loosen it -- an explicit privacy-on
        setting anywhere (constructor option, env, or legacy env) must never
        lose to a looser `content_mode=`. When the client-level result is
        just the bare "nothing configured" default
        (`self.config.content_mode_explicit` is `False`), there is no
        explicit policy to protect, so the override is used as-is for just
        this call -- this is the normal, documented way a caller opts a
        single trace into `record`/`redacted` capture.
        """
        normalized = validate_content_mode_override(content_mode)
        return self._resolve_against_client_default(normalized)

    def _resolve_against_client_default(self, normalized: str | None) -> str:
        """Shared M2 client-level resolution; `normalized` must already have
        passed `validate_content_mode_override` (or be `None`)."""
        if normalized is None:
            return self.config.content_mode
        if self.config.content_mode_explicit:
            return most_restrictive_content_mode(normalized, self.config.content_mode)
        return normalized

    def _resolve_trace_scoped_content_mode(
        self, trace_id: str, content_mode: str | None
    ) -> str:
        """ADDENDUM R2 A1: resolve a per-call `content_mode` override for a
        call SCOPED TO AN EXISTING TRACE (`record_observation`, `end_trace`,
        `add_comment`, `submit_feedback`) against that trace's own resolved
        mode -- fixed once, at `start_trace` -- rather than straight against
        the client config the way `start_trace` itself does.

        `None` (no override on this call) resolves to the trace's own
        `resolved_content_mode` -- the base a later un-annotated call on the
        same trace continues. This is the fix for finding 8's confirmed
        parity probe: starting a trace with `content_mode="redacted"` and
        then calling `record_observation` with no override must emit the
        `redacted` placeholder (inheriting the TRACE's mode), not silently
        fall back to the CLIENT's own default/config the way the pre-fix
        implementation did (which, for an unconfigured client, is `metadata`
        and so OMITTED the field instead -- the exact TS/Python divergence).

        A supplied override may TIGHTEN that base only when the trace's own
        mode was itself set from an explicit source (`state.content_mode_
        locked` -- either `start_trace`'s own `content_mode` argument was
        given, or the client-level policy was explicit); with neither, the
        override is used as-is for this call -- the same "no explicit policy
        to protect" rule `_resolve_against_client_default` applies at the
        client level, just against the trace's base instead.

        A `trace_id` this client holds no local `_TraceState` for (created by
        another process or SDK instance, or a bare `add_comment`/
        `submit_feedback` call that never went through `start_trace` here)
        falls back to client-level resolution, exactly like a bare
        `start_trace` -- A1: "Top-level startTrace/addComment/submitFeedback
        without a trace use the client-level result."
        """
        normalized = validate_content_mode_override(content_mode)
        with self._lock:
            state = self._trace_states.get(trace_id)
            tombstone = self._evicted_traces.get(trace_id) if state is None else None
        if state is not None:
            base_mode = state.resolved_content_mode
            locked = state.content_mode_locked
        elif tombstone is not None:
            # #2604: an evicted trace keeps its own content policy -- the
            # revival that follows restores it, so resolve against it now
            # instead of the (possibly looser) client default. A malformed
            # tombstone fails closed to the strictest mode.
            base_mode = tombstone.resolved_content_mode
            locked = tombstone.content_mode_locked
            if base_mode not in OBSERVABILITY_CONTENT_MODES:
                base_mode, locked = "metadata", True
        else:
            return self._resolve_against_client_default(normalized)
        if normalized is None:
            return base_mode
        if locked:
            return most_restrictive_content_mode(normalized, base_mode)
        return normalized

    def start_trace(
        self,
        name: str,
        *,
        trace_id: str | None = None,
        session_id: str | None = None,
        user_id: str | None = None,
        environment: str | None = None,
        release: str | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        input_data: Any = NOT_SUPPLIED,
        output_data: Any = NOT_SUPPLIED,
        custom_trace_id: str | None = None,
        started_at: datetime | None = None,
        session: SessionDTO | dict[str, Any] | None = None,
        correlation_ids: CorrelationIds | dict[str, str] | None = None,
        prompt_reference: PromptReferenceDTO | dict[str, Any] | None = None,
        execution_context: ExecutionContextDTO | dict[str, Any] | None = None,
        status: str = "running",
        content_mode: str | None = None,
        redact_input: bool = False,
        redact_output: bool = False,
    ) -> str:
        trace_id = trace_id or _new_trace_id()
        # Read the ambient agent scope before any span of this trace opens: a
        # scope that already encloses the trace belongs to it.
        scope_ledger = new_trace_ledger()
        session_dto = self._coerce_session(session)
        if session_dto is not None and session_id is None:
            session_id = session_dto.id

        effective_content_mode = self._resolve_content_mode(content_mode)
        gated_input_data = (
            None
            if input_data is NOT_SUPPLIED
            else apply_content_mode(
                input_data, effective_content_mode, force_redact=redact_input
            )
        )
        gated_output_data = (
            None
            if output_data is NOT_SUPPLIED
            else apply_content_mode(
                output_data, effective_content_mode, force_redact=redact_output
            )
        )

        trace = TraceDTO(
            id=trace_id,
            name=name,
            status=status,
            session_id=session_id,
            user_id=user_id,
            environment=(
                resolve_environment(environment).environment
                if environment is not None and environment.strip()
                else self.config.default_environment
            ),
            release=release or self.config.default_release,
            custom_trace_id=custom_trace_id,
            tags=list(tags or []),
            metadata=_detach_metadata(metadata),
            input_data=gated_input_data,
            output_data=gated_output_data,
            started_at=started_at or utc_now(),
            session=session_dto,
            correlation_ids=self._coerce_correlation_ids(correlation_ids),
            prompt_reference=self._coerce_prompt_reference(prompt_reference),
            execution_context=self._coerce_execution_context(execution_context),
        )

        # A1: fix this trace's own base content_mode and whether a later
        # per-call override on this trace may only tighten it (see
        # `_resolve_trace_scoped_content_mode`).
        content_mode_locked = (
            content_mode is not None or self.config.content_mode_explicit
        )
        with self._lock:
            previous = self._trace_states.get(trace_id)
            tombstone = self._evicted_traces.pop(trace_id, None)
            if previous is not None:
                previous_sequence = previous.snapshot_sequence
            elif tombstone is not None:
                previous_sequence = tombstone.snapshot_sequence
            else:
                previous_sequence = 0
            self._trace_states[trace_id] = _TraceState(
                trace=trace,
                resolved_content_mode=effective_content_mode,
                content_mode_locked=content_mode_locked,
                # Restarting a trace id keeps its capture order monotonic: the
                # transport may still hold a snapshot of the earlier state.
                snapshot_sequence=previous_sequence,
                scope_ledger=scope_ledger,
            )

        self._queue_trace_snapshot(trace_id)
        return trace_id

    @staticmethod
    def _gate_observation_error(
        error: BaseException | None,
        metadata: dict[str, Any],
        effective_content_mode: Any,
    ) -> str | None:
        gated_error_message: str | None = None
        error_supplied = error is not None
        if error_supplied:
            # error_message is free-form content: exception strings routinely
            # interpolate the very inputs the content gate withholds (prompts,
            # records, PII), so it must honor `content_mode` exactly like
            # input_data/output_data. error_type is only a class name, so it
            # is safe to keep in every mode. B2: `str(error)` is a property/
            # getter access on caller-supplied state -- a hostile exception
            # subclass can override `__str__` to raise -- so it must go
            # through `_safe_str` rather than raising straight out of this
            # call.
            metadata["error_type"] = type(error).__name__
            gated_error_message = apply_content_mode_to_text(
                _safe_str(error), effective_content_mode
            )
            if gated_error_message is not None:
                metadata["error_message"] = gated_error_message
            # A2/finding 2: if this mode withholds the error text
            # (`gated_error_message is None`), do NOT set "error_message" in
            # `metadata` here -- whether an existing "error_message" must be
            # force-cleared to "[REDACTED]" (tightening over prior content)
            # or left absent (first write, nothing stored yet) depends on
            # `existing`, only known once the trace lock below is held; see
            # the merge branch.
        return gated_error_message

    @staticmethod
    def _note_prior_observation_content(
        state: _TraceState, observation_id: str, existing: ObservationDTO
    ) -> None:
        # B3: record, BEFORE this call's own (possibly gated-to-None
        # or freshly-hostile) write lands, whether this observation
        # already carried content -- see `_TraceState.
        # content_history`. Read directly off `existing`, never off
        # the value this call is about to write.
        if existing.input_data is not None:
            state.mark_content_history(observation_id, "input_data")
        if existing.output_data is not None:
            state.mark_content_history(observation_id, "output_data")
        # Same tracking for `error_message` -- it is free-form
        # content (an exception string) subject to the exact same
        # "never omit the clearing field on a later failure" rule as
        # `input_data`/`output_data`, but it lives inside `metadata`
        # rather than as its own top-level field, so it needs its
        # own history entry (consulted by
        # `_safe_allowlisted_observation_snapshot` via
        # `_build_fallback_descriptor`).
        if existing.metadata.get("error_message") is not None:
            state.mark_content_history(observation_id, "error_message")

    def _build_new_observation(
        self, state: _TraceState, call: _ObservationCall
    ) -> ObservationDTO:
        observation_id = call.observation_id
        effective_type = call.effective_type
        name = call.name
        parent_observation_id = call.parent_observation_id
        status = call.status
        started_at = call.started_at
        ended_at = call.ended_at
        latency_ms = call.latency_ms
        input_tokens = call.input_tokens
        output_tokens = call.output_tokens
        total_tokens = call.total_tokens
        cost_usd = call.cost_usd
        model_name = call.model_name
        tool_name = call.tool_name
        input_supplied = call.input_supplied
        output_supplied = call.output_supplied
        gated_input_data = call.gated_input_data
        gated_output_data = call.gated_output_data
        effective_content_mode = call.effective_content_mode
        error_supplied = call.error_supplied
        gated_error_message = call.gated_error_message
        metadata = call.metadata
        correlation_ids = call.correlation_ids
        prompt_reference = call.prompt_reference
        # A revived trace (#2604) holds no observations, but its
        # tombstone remembers which fields were delivered: send the
        # clearing marker for those exactly as a live update would.
        if input_supplied and state.had_prior_content(observation_id, "input_data"):
            gated_input_data = wire_value_for_tightened_update(
                gated_input_data,
                effective_content_mode=effective_content_mode,
                existing_value=redacted_content_marker(),
            )
        if output_supplied and state.had_prior_content(observation_id, "output_data"):
            gated_output_data = wire_value_for_tightened_update(
                gated_output_data,
                effective_content_mode=effective_content_mode,
                existing_value=redacted_content_marker(),
            )
        if (
            error_supplied
            and gated_error_message is None
            and state.had_prior_content(observation_id, "error_message")
        ):
            metadata["error_message"] = _REDACTED
        observation = ObservationDTO(
            id=observation_id,
            type=effective_type,
            name=name,
            parent_observation_id=parent_observation_id,
            status=status,
            started_at=started_at or utc_now(),
            ended_at=ended_at,
            latency_ms=latency_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            cost_usd=cost_usd,
            model_name=model_name,
            tool_name=tool_name,
            input_data=gated_input_data,
            output_data=gated_output_data,
            metadata=dict(metadata or {}),
            correlation_ids=self._coerce_correlation_ids(correlation_ids),
            prompt_reference=self._coerce_prompt_reference(prompt_reference),
        )
        return observation

    def _build_updated_observation(
        self,
        state: _TraceState,
        existing: ObservationDTO,
        call: _ObservationCall,
    ) -> ObservationDTO:
        observation_id = call.observation_id
        effective_type = call.effective_type
        name = call.name
        parent_observation_id = call.parent_observation_id
        status = call.status
        started_at = call.started_at
        ended_at = call.ended_at
        latency_ms = call.latency_ms
        input_tokens = call.input_tokens
        output_tokens = call.output_tokens
        total_tokens = call.total_tokens
        cost_usd = call.cost_usd
        model_name = call.model_name
        tool_name = call.tool_name
        input_supplied = call.input_supplied
        output_supplied = call.output_supplied
        gated_input_data = call.gated_input_data
        gated_output_data = call.gated_output_data
        effective_content_mode = call.effective_content_mode
        error_supplied = call.error_supplied
        gated_error_message = call.gated_error_message
        metadata = call.metadata
        correlation_ids = call.correlation_ids
        prompt_reference = call.prompt_reference
        merged_metadata = dict(existing.metadata)
        if metadata:
            merged_metadata.update(metadata)
        # A2/finding 2: a tightening update over an error this SAME
        # observation already recorded must never let the stale
        # `error_message` survive the merge. `metadata` only carries
        # an "error_message" key when this call's gating actually
        # produced text (see above) -- when it didn't (this mode
        # withholds it) but `existing.metadata` still has one from
        # an earlier, less restrictive call, `merged_metadata.
        # update(metadata)` alone would silently keep it (the
        # original bug: nothing in `metadata` ever overwrote it).
        # Force the explicit placeholder instead, never merge old
        # error text back. A genuine first write (no prior
        # "error_message" at all) still omits cleanly -- there is
        # nothing stored to leak.
        if (
            error_supplied
            and gated_error_message is None
            and (
                "error_message" in existing.metadata
                or state.had_prior_content(observation_id, "error_message")
            )
        ):
            merged_metadata["error_message"] = _REDACTED
        observation = ObservationDTO(
            id=observation_id,
            type=effective_type,
            name=name,
            parent_observation_id=_keep_if_none(
                parent_observation_id, existing.parent_observation_id
            ),
            status=status,
            started_at=started_at or existing.started_at,
            ended_at=ended_at or existing.ended_at,
            latency_ms=_keep_if_none(latency_ms, existing.latency_ms),
            input_tokens=_keep_if_none(input_tokens, existing.input_tokens),
            output_tokens=_keep_if_none(output_tokens, existing.output_tokens),
            total_tokens=_keep_if_none(total_tokens, existing.total_tokens),
            cost_usd=_keep_if_none(cost_usd, existing.cost_usd),
            model_name=model_name or existing.model_name,
            tool_name=tool_name or existing.tool_name,
            # M4 fix: fall back to `existing` only when this call
            # never supplied the field (NOT_SUPPLIED) -- a supplied
            # value, gated, always overwrites, even when gating
            # withheld it to `None` (tightening wins; see the
            # `apply_content_mode` docstring). `wire_value_for_
            # tightened_update` additionally forces the explicit
            # placeholder (not `None`) when this is a tightening
            # update over PRIOR content -- the ingest contract's
            # `apply_updates` keeps stored content on an
            # omitted/`None` field, so merely omitting would not
            # clear it server-side.
            input_data=_updated_content_value(
                state,
                observation_id,
                "input_data",
                existing.input_data,
                supplied=input_supplied,
                gated=gated_input_data,
                effective_content_mode=effective_content_mode,
            ),
            output_data=_updated_content_value(
                state,
                observation_id,
                "output_data",
                existing.output_data,
                supplied=output_supplied,
                gated=gated_output_data,
                effective_content_mode=effective_content_mode,
            ),
            metadata=merged_metadata,
            correlation_ids=(
                self._coerce_correlation_ids(correlation_ids)
                or existing.correlation_ids
            ),
            prompt_reference=(
                self._coerce_prompt_reference(prompt_reference)
                or existing.prompt_reference
            ),
        )
        return observation

    def record_observation(
        self,
        trace_id: str,
        *,
        name: str,
        observation_type: ObservationType | str | None = None,
        observation_id: str | None = None,
        parent_observation_id: str | None = None,
        status: str = "running",
        started_at: datetime | None = None,
        ended_at: datetime | None = None,
        latency_ms: int | None = None,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        total_tokens: int | None = None,
        cost_usd: float | None = None,
        model_name: str | None = None,
        tool_name: str | None = None,
        input_data: Any = NOT_SUPPLIED,
        output_data: Any = NOT_SUPPLIED,
        metadata: dict[str, Any] | None = None,
        correlation_ids: CorrelationIds | dict[str, str] | None = None,
        prompt_reference: PromptReferenceDTO | dict[str, Any] | None = None,
        content_mode: str | None = None,
        redact_input: bool = False,
        redact_output: bool = False,
        error: BaseException | None = None,
    ) -> str:
        requested_type = (
            ObservationType(observation_type)
            if isinstance(observation_type, str)
            else observation_type
        )

        # A1: resolve against the TRACE's own base mode (fixed at
        # `start_trace`), not straight against the client config -- see
        # `_resolve_trace_scoped_content_mode`.
        effective_content_mode = self._resolve_trace_scoped_content_mode(
            trace_id, content_mode
        )
        # M4: NOT_SUPPLIED (the default) means "this call never touched the
        # field" -- skip gating entirely and, on an update, keep whatever was
        # previously recorded. A SUPPLIED value (including an explicit
        # `None`, e.g. a decorated function that legitimately returned
        # `None`) is real content subject to `content_mode` gating, and on
        # an update OVERWRITES the existing field with the freshly gated
        # result -- this is the fix for the historical bug where a
        # metadata-mode update could not tell "I have nothing new" apart
        # from "I have new content this mode forbids", so it silently kept
        # content an earlier, less restrictive call had recorded.
        input_supplied = input_data is not NOT_SUPPLIED
        output_supplied = output_data is not NOT_SUPPLIED
        gated_input_data = (
            apply_content_mode(
                input_data, effective_content_mode, force_redact=redact_input
            )
            if input_supplied
            else None
        )
        gated_output_data = (
            apply_content_mode(
                output_data, effective_content_mode, force_redact=redact_output
            )
            if output_supplied
            else None
        )
        # A8: detach from the caller's own metadata dict/nested structures
        # (deep copy) rather than a shallow `dict(metadata or {})`.
        metadata = _detach_metadata(metadata)
        error_supplied = error is not None
        gated_error_message = self._gate_observation_error(
            error, metadata, effective_content_mode
        )

        observation_id = observation_id or _new_observation_id()
        with self._lock:
            state = self._state_or_revive_locked(trace_id)
            if state is None:
                raise ValueError(f"Unknown trace_id '{trace_id}'")
            existing = state.observations.get(observation_id)
            if existing is not None:
                self._note_prior_observation_content(state, observation_id, existing)
            effective_type = requested_type or (
                existing.type if existing is not None else ObservationType.SPAN
            )
            if tool_name is None and effective_type in {
                ObservationType.TOOL,
                ObservationType.TOOL_CALL,
            }:
                tool_name = name
            self._validate_observation_relationship(
                state,
                observation_id=observation_id,
                observation_type=effective_type,
                parent_observation_id=parent_observation_id,
            )

            call = _ObservationCall(
                observation_id=observation_id,
                effective_type=effective_type,
                name=name,
                parent_observation_id=parent_observation_id,
                status=status,
                started_at=started_at,
                ended_at=ended_at,
                latency_ms=latency_ms,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                cost_usd=cost_usd,
                model_name=model_name,
                tool_name=tool_name,
                input_supplied=input_supplied,
                output_supplied=output_supplied,
                gated_input_data=gated_input_data,
                gated_output_data=gated_output_data,
                effective_content_mode=effective_content_mode,
                error_supplied=error_supplied,
                gated_error_message=gated_error_message,
                metadata=metadata,
                correlation_ids=correlation_ids,
                prompt_reference=prompt_reference,
            )
            if existing is None:
                observation = self._build_new_observation(state, call)
                state.observation_order.append(observation_id)
            else:
                observation = self._build_updated_observation(state, existing, call)

            state.observations[observation_id] = observation

            if (
                observation.type is ObservationType.GENERATION
                and observation.status in {"completed", "failed"}
                and observation.input_tokens is None
                and observation.output_tokens is None
                and observation.total_tokens is None
                and observation.cost_usd is None
                and not self._missing_generation_usage_notice_logged
            ):
                self._missing_generation_usage_notice_logged = True
                logger.warning(
                    "Generation observation '%s' has no token/cost usage fields; "
                    "usage will be reported as unknown. Pass input_tokens, "
                    "output_tokens, total_tokens, or cost_usd from your provider "
                    "response when recording generation spans.",
                    observation.name,
                )

        self._queue_trace_snapshot(trace_id)
        return observation_id

    def end_trace(
        self,
        trace_id: str,
        *,
        status: str = "completed",
        output_data: Any = NOT_SUPPLIED,
        ended_at: datetime | None = None,
        started_at: datetime | None = None,
        metadata: dict[str, Any] | None = None,
        content_mode: str | None = None,
        redact_output: bool = False,
    ) -> None:
        if status not in OBSERVABILITY_STATUSES:
            allowed = ", ".join(sorted(OBSERVABILITY_STATUSES))
            raise ValueError(f"status must be one of: {allowed}")
        # A1: resolve against the trace's own base mode, not the client
        # config directly -- see `_resolve_trace_scoped_content_mode`.
        effective_content_mode = self._resolve_trace_scoped_content_mode(
            trace_id, content_mode
        )
        # M4: same NOT_SUPPLIED-vs-supplied fix as `record_observation` --
        # `end_trace` is itself "a later update" of the trace's output_data,
        # so it is subject to the same tightening-must-win requirement. A
        # status-only `end_trace(trace_id)` call must not disturb
        # `output_data`; a call that supplies output_data (even under a mode
        # that withholds it) must overwrite whatever was recorded before.
        output_supplied = output_data is not NOT_SUPPLIED
        gated_output_data = (
            apply_content_mode(
                output_data, effective_content_mode, force_redact=redact_output
            )
            if output_supplied
            else None
        )
        with self._lock:
            state = self._state_or_revive_locked(trace_id)
            if state is None:
                raise ValueError(f"Unknown trace_id '{trace_id}'")
            state.trace.status = status
            if started_at is not None:
                state.trace.started_at = started_at
            state.trace.ended_at = ended_at or utc_now()
            if output_supplied:
                # B3: record, BEFORE this call's own write lands, whether
                # the trace already carried `output_data` content -- see
                # `_TraceState.content_history`. Read directly off the
                # PRE-overwrite value on the line below, never off
                # `gated_output_data` (this call's own, possibly
                # freshly-hostile write).
                if state.trace.output_data is not None:
                    state.mark_content_history(_TRACE_CONTENT_KEY, "output_data")
                # Ingest-contract fix: force the explicit placeholder (not
                # `None`) when this tightens over prior content -- see
                # `wire_value_for_tightened_update`.
                state.trace.output_data = wire_value_for_tightened_update(
                    gated_output_data,
                    effective_content_mode=effective_content_mode,
                    existing_value=(
                        state.trace.output_data
                        if state.trace.output_data is not None
                        or not state.had_prior_content(
                            _TRACE_CONTENT_KEY, "output_data"
                        )
                        else redacted_content_marker()
                    ),
                )
            if metadata:
                # A8: detach (deep copy) the caller's metadata before merging
                # -- see `_detach_metadata`.
                merged = dict(state.trace.metadata)
                merged.update(_detach_metadata(metadata))
                state.trace.metadata = merged

        self._queue_trace_snapshot(trace_id)
        self._schedule_trace_state_eviction(trace_id)
        self._evict_expired_trace_states()

    def _schedule_trace_state_eviction(self, trace_id: str) -> None:
        with self._lock:
            state = self._trace_states.get(trace_id)
            if state is None:
                return
            state.ended = True
            if not state.eviction_scheduled:
                state.eviction_scheduled = True
                self._eviction_queue.append(
                    (self._clock() + self._trace_state_grace, trace_id)
                )

    @staticmethod
    def _delivered_content_history(state: _TraceState) -> dict[str, frozenset[str]]:
        """Per-field "content was delivered" flags to carry on a tombstone.

        The history recorded so far, plus every content field the evicted
        state still holds (a delivered value, or an earlier clearing marker),
        so a revived trace sends the clearing marker exactly as a live one.
        """
        history = {key: set(fields) for key, fields in state.content_history.items()}
        for field_name in _SAFE_SNAPSHOT_CONTENT_FIELDS:
            if getattr(state.trace, field_name, None) is not None:
                history.setdefault(_TRACE_CONTENT_KEY, set()).add(field_name)
        for observation_id, observation in state.observations.items():
            for field_name in _SAFE_SNAPSHOT_CONTENT_FIELDS:
                if getattr(observation, field_name, None) is not None:
                    history.setdefault(observation_id, set()).add(field_name)
            if observation.metadata.get("error_message") is not None:
                history.setdefault(observation_id, set()).add("error_message")
        return {key: frozenset(fields) for key, fields in history.items()}

    def _remember_evicted_locked(self, trace_id: str, state: _TraceState) -> None:
        self._evicted_traces[trace_id] = _EvictedTrace(
            name=state.trace.name,
            environment=state.trace.environment,
            release=state.trace.release,
            snapshot_sequence=state.snapshot_sequence,
            resolved_content_mode=state.resolved_content_mode,
            content_mode_locked=state.content_mode_locked,
            content_history=self._delivered_content_history(state),
        )
        self._evicted_traces.move_to_end(trace_id)
        while len(self._evicted_traces) > _MAX_EVICTED_TRACE_TOMBSTONES:
            self._evicted_traces.popitem(last=False)

    def _state_or_revive_locked(self, trace_id: str) -> _TraceState | None:
        """The live state, or a minimal fresh one for a recently evicted id.

        An update to an evicted trace starts a fresh snapshot under the same
        id (#2604); an id this client never held (or whose tombstone aged out
        of the bounded set) still returns None.
        """
        state = self._trace_states.get(trace_id)
        if state is not None:
            return state
        tombstone = self._evicted_traces.pop(trace_id, None)
        if tombstone is None:
            return None
        state = _TraceState(
            trace=TraceDTO(
                id=trace_id,
                name=tombstone.name,
                environment=tombstone.environment,
                release=tombstone.release,
                started_at=utc_now(),
            ),
            resolved_content_mode=tombstone.resolved_content_mode,
            content_mode_locked=tombstone.content_mode_locked,
            content_history={
                key: set(fields) for key, fields in tombstone.content_history.items()
            },
            snapshot_sequence=tombstone.snapshot_sequence,
        )
        self._trace_states[trace_id] = state
        return state

    def _evict_expired_trace_states(self) -> int:
        """Drop ended, delivered traces whose grace window has passed (#2604).

        Bounded work: only the head of the time-ordered queue is examined, at
        most `_MAX_EVICTION_STEPS_PER_PASS` entries per call. A trace that is
        not ended, or whose latest snapshot is queued, in flight, failed or
        otherwise unconfirmed, is kept and rechecked one grace window later.
        A later update to an evicted trace id starts a fresh snapshot.
        """
        evicted = 0
        with self._lock:
            now = self._clock()
            steps = 0
            while (
                self._eviction_queue
                and self._eviction_queue[0][0] <= now
                and steps < _MAX_EVICTION_STEPS_PER_PASS
            ):
                steps += 1
                _, trace_id = self._eviction_queue.popleft()
                state = self._trace_states.get(trace_id)
                if state is None or not state.ended:
                    if state is not None:
                        state.eviction_scheduled = False
                    continue
                if self._ingest_suppressed or self._transport.is_settled(trace_id):
                    del self._trace_states[trace_id]
                    self._transport.release_settled(trace_id)
                    self._remember_evicted_locked(trace_id, state)
                    evicted += 1
                else:
                    self._eviction_queue.append(
                        (now + self._trace_state_grace, trace_id)
                    )
        return evicted

    def scope_ledger(self, trace_id: str) -> ScopeLedger | None:
        """The agent-scope ledger of a live trace (None for an unknown trace)."""
        with self._lock:
            state = self._trace_states.get(trace_id)
            return state.scope_ledger if state is not None else None

    def flush(self, timeout: float | None = None) -> FlushResult:
        """Flush queued traces.

        Supplying ``timeout`` bounds delivery with a monotonic deadline. Omitting
        it preserves the historical synchronous flush behavior; ``timeout=0`` is
        an immediate poll that does not emit a deadline warning.
        Health callbacks are asynchronous, so this can return before callbacks
        for delivery events from this flush have run.

        When delivery is suppressed (offline mode, ``TRAIGENT_DISABLE_TELEMETRY``,
        or no credential), the result keeps ``success=True`` so telemetry never
        breaks the host application, but ``items_sent`` is 0 and ``warnings``
        names the cause. Check ``items_sent`` rather than ``success`` alone to
        confirm delivery.
        """
        deadline = self._flush_deadline(timeout)
        if self._ingest_suppressed:
            self._log_offline_mode_once()
            return self._suppressed_flush_result()
        if not self._acquire_client_lock_until(deadline):
            return self._flush_lock_timeout_result(deadline)
        try:
            # Finding 2 (A8/M4 snapshot immutability): `_safe_trace_snapshot`
            # returns `(payload, descriptor)` -- the descriptor is a frozen
            # `_TraceFallbackDescriptor` built from `state` while THIS lock is
            # still held. Thread the descriptor, never `state` itself, into
            # `_submit_trace_snapshot` below (which runs after the lock is
            # released).
            trace_payloads = [
                (trace_id, *_safe_trace_snapshot(state))
                for trace_id, state in self._trace_states.items()
            ]
        finally:
            self._lock.release()

        for trace_id, payload, descriptor in trace_payloads:
            self._submit_trace_snapshot(
                trace_id, payload, deadline=deadline, state=descriptor
            )
        result = self._flush_transport_until(deadline)
        self._evict_expired_trace_states()
        return result

    def get_stats(self) -> dict[str, Any]:
        """Return a snapshot of transport delivery and local-drop health.

        The snapshot includes ``dropped_by_reason``, ``dropped_health_events``,
        ``queue_depth``, ``inflight_items``, ``oldest_inflight_age_seconds``,
        and ``retry_attempts``. Configure
        :attr:`ObservabilityConfig.health_callback` to receive structured
        batch-level drop events. They run on one daemon dispatcher thread after
        internal transport locks are released; a bounded flush or close can
        return before callbacks for its events have run.
        """
        stats = cast(dict[str, Any], self._transport.get_stats())
        with self._lock:
            stats["active_trace_count"] = len(self._trace_states)
        return stats

    def _claim_close_locked(self) -> _CloseClaim:
        """Decide this close call's role; the caller holds ``self._lock``."""
        claim = _CloseClaim()
        if self._closing:
            claim.close_in_progress = True
        elif self._closed:
            claim.already_closed = True
        elif self._close_pending:
            self._closing = True
            self._close_complete.clear()
            claim.offline_close = self._ingest_suppressed
        else:
            self._closing = True
            self._close_pending = True
            self._close_complete.clear()
            claim.offline_close = self._ingest_suppressed
            if not claim.offline_close:
                # Finding 2 (A8/M4 snapshot immutability): thread the
                # frozen descriptor (built from `state` under THIS lock),
                # never `state` itself -- see the matching comment in
                # `flush`.
                claim.trace_payloads = [
                    (trace_id, *_safe_trace_snapshot(state))
                    for trace_id, state in self._trace_states.items()
                ]
                claim.inflight_snapshot_submissions = (
                    self._inflight_snapshot_submissions
                )
        return claim

    def _deliver_close(
        self, claim: _CloseClaim, deadline: float | None, timeout: float | None
    ) -> tuple[FlushResult, bool]:
        if claim.offline_close:
            self._log_offline_mode_once()
            transport_result = self._transport.close(
                timeout=self._remaining_flush_time(deadline),
                warn_on_deadline_expiry=timeout != 0,
            )
            result = self._with_suppressed_delivery_warning(
                self._flush_result_from_transport(transport_result)
            )
        else:
            if claim.inflight_snapshot_submissions:
                self._wait_for_inflight_snapshots(
                    claim.inflight_snapshot_submissions, deadline
                )

            for trace_id, payload, descriptor in claim.trace_payloads:
                self._submit_trace_snapshot(
                    trace_id, payload, deadline=deadline, state=descriptor
                )
            transport_result = self._transport.close(
                timeout=self._remaining_flush_time(deadline)
            )
            result = self._flush_result_from_transport(transport_result)
        transport_close_completed = self._transport_close_completed(transport_result)
        if not transport_close_completed:
            result = self._transport_close_pending_result(result)
        return result, transport_close_completed

    def _wait_for_inflight_snapshots(
        self, inflight_snapshot_submissions: int, deadline: float | None
    ) -> None:
        remaining = self._remaining_flush_time(deadline)
        completed = (
            self._snapshot_submission_complete.wait()
            if remaining is None
            else self._snapshot_submission_complete.wait(timeout=remaining)
        )
        if not completed:
            logger.warning(
                "Observability close deadline expired while waiting for "
                "%d snapshot submissions",
                inflight_snapshot_submissions,
            )

    def close(self, timeout: float | None = None) -> FlushResult:
        """Stop submissions and flush queued traces.

        An explicit ``timeout`` bounds delivery; ``timeout=0`` is an immediate,
        warning-free poll. The atexit handler explicitly uses
        :attr:`ObservabilityConfig.flush_timeout` so interpreter shutdown stays
        bounded even though a normal no-argument close remains synchronous.
        Queued health callbacks accepted before an unbounded close are drained
        before it returns. A bounded close stops waiting at its deadline. Health
        events produced after dispatcher shutdown are counted in
        ``dropped_health_events``.
        """
        deadline = self._flush_deadline(timeout)
        if not self._acquire_client_lock_until(deadline):
            return self._close_lock_timeout_result(deadline)
        try:
            claim = self._claim_close_locked()
        finally:
            self._lock.release()
        if claim.close_in_progress:
            return self._wait_for_concurrent_close(deadline)
        if claim.already_closed:
            closed_result = self._flush_transport_until(deadline)
            if self._ingest_suppressed:
                return self._with_suppressed_delivery_warning(closed_result)
            return closed_result

        result: FlushResult | None = None
        transport_close_completed = False
        finalizer_completed = False
        try:
            result, transport_close_completed = self._deliver_close(
                claim, deadline, timeout
            )
        finally:
            if self._acquire_client_lock_until(deadline):
                try:
                    self._finalize_close_state(transport_close_completed)
                    finalizer_completed = True
                finally:
                    self._lock.release()
            else:
                logger.warning(
                    "Observability close deadline exceeded while reacquiring the "
                    "client state lock for finalization"
                )
                threading.Thread(
                    target=self._finalize_close_when_client_lock_releases,
                    args=(transport_close_completed,),
                    name="traigent-observability-close-finalizer",
                    daemon=True,
                ).start()
        if not finalizer_completed:
            return self._close_finalizer_lock_timeout_result(deadline)
        if result is None:  # defensive: finalizer_completed implies a result was set
            return self._close_finalizer_lock_timeout_result(deadline)
        return result

    @staticmethod
    def _flush_result_from_transport(result: BatchFlushResult) -> FlushResult:
        return FlushResult(
            success=result.success,
            items_sent=result.items_sent,
            items_pending=result.items_pending,
            items_dropped=result.items_dropped,
            successful_batches=result.successful_batches,
            failed_batches=result.failed_batches,
            errors=result.errors,
            warnings=result.warnings,
        )

    def _flush_transport_until(self, deadline: float | None) -> FlushResult:
        """Return the current transport truth without waiting past ``deadline``."""
        result = self._transport.flush(timeout=self._remaining_flush_time(deadline))
        return self._flush_result_from_transport(result)

    def _transport_close_completed(self, result: BatchFlushResult) -> bool:
        """Whether transport close finished delivery and dispatcher teardown."""
        if not result.success:
            return False
        if not getattr(self._transport, "_closed", True):
            return False
        dispatcher = getattr(self._transport, "_health_dispatcher_thread", None)
        return dispatcher is None or not dispatcher.is_alive()

    @staticmethod
    def _transport_close_pending_result(result: FlushResult) -> FlushResult:
        """Keep a partial transport close visibly retryable to callers."""
        warning = (
            "Observability transport close is pending; the client remains closed "
            "to new snapshots until a later close completes teardown"
        )
        return FlushResult(
            success=False,
            items_sent=result.items_sent,
            items_pending=result.items_pending,
            items_dropped=result.items_dropped,
            successful_batches=result.successful_batches,
            failed_batches=result.failed_batches,
            errors=result.errors,
            warnings=[*result.warnings, warning],
        )

    def _flush_lock_timeout_result(self, deadline: float | None) -> FlushResult:
        """Report that flush never inspected client state or submitted snapshots."""
        stats = self._transport.get_stats(timeout=self._remaining_flush_time(deadline))
        warning = (
            "Observability flush deadline exceeded while acquiring the client "
            "state lock; no client snapshots were submitted"
        )
        return FlushResult(
            success=False,
            items_sent=int(stats["sent_items"]),
            items_pending=int(stats["pending_items"]),
            items_dropped=int(stats["dropped_items"]),
            successful_batches=int(stats["successful_batches"]),
            failed_batches=int(stats["failed_batches"]),
            errors=list(stats["errors"]),
            warnings=[*stats["warnings"], warning],
        )

    def _close_lock_timeout_result(self, deadline: float | None) -> FlushResult:
        """Report that close never acquired client state and changed nothing."""
        stats = self._transport.get_stats(timeout=self._remaining_flush_time(deadline))
        warning = (
            "Observability close deadline exceeded while acquiring the client "
            "state lock; client and transport remain open"
        )
        return FlushResult(
            success=False,
            items_sent=int(stats["sent_items"]),
            items_pending=int(stats["pending_items"]),
            items_dropped=int(stats["dropped_items"]),
            successful_batches=int(stats["successful_batches"]),
            failed_batches=int(stats["failed_batches"]),
            errors=list(stats["errors"]),
            warnings=[*stats["warnings"], warning],
        )

    def _close_finalizer_lock_timeout_result(
        self, deadline: float | None
    ) -> FlushResult:
        """Report a close whose transport work finished but state finalization did not."""
        stats = self._transport.get_stats(timeout=self._remaining_flush_time(deadline))
        warning = (
            "Observability close deadline exceeded while reacquiring the client "
            "state lock for finalization; close state remains in progress"
        )
        return FlushResult(
            success=False,
            items_sent=int(stats["sent_items"]),
            items_pending=int(stats["pending_items"]),
            items_dropped=int(stats["dropped_items"]),
            successful_batches=int(stats["successful_batches"]),
            failed_batches=int(stats["failed_batches"]),
            errors=list(stats["errors"]),
            warnings=[*stats["warnings"], warning],
        )

    def _finalize_close_state(self, transport_close_completed: bool) -> None:
        """Record whether transport teardown completed while holding client state."""
        self._closing = False
        self._close_pending = not transport_close_completed
        self._closed = transport_close_completed
        self._close_complete.set()

    def _finalize_close_when_client_lock_releases(
        self, transport_close_completed: bool
    ) -> None:
        """Complete state cleanup after a bounded close returned its partial result."""
        with self._lock:
            self._finalize_close_state(transport_close_completed)

    def _wait_for_concurrent_close(self, deadline: float | None) -> FlushResult:
        remaining = self._remaining_flush_time(deadline)
        completed = (
            self._close_complete.wait()
            if remaining is None
            else self._close_complete.wait(timeout=remaining)
        )
        result = self._flush_transport_until(deadline)
        if completed:
            return result
        warning = (
            "Observability close deadline exceeded while another close was running"
        )
        return FlushResult(
            success=False,
            items_sent=result.items_sent,
            items_pending=result.items_pending,
            items_dropped=result.items_dropped,
            successful_batches=result.successful_batches,
            failed_batches=result.failed_batches,
            errors=result.errors,
            warnings=[*result.warnings, warning],
        )

    def list_comments(self, trace_id: str) -> TraceCommentsResponse:
        payload = self._request_json("GET", f"/traces/{trace_id}/comments")
        return TraceCommentsResponse.from_dict(
            self._unwrap_data(payload, "trace comments")
        )

    def add_comment(
        self,
        trace_id: str,
        content: str,
        *,
        content_mode: str | None = None,
    ) -> TraceCommentRecord:
        """Post a free-text comment on a trace (M3).

        A comment IS content -- there is no non-content way to post one --
        so, unlike `record_observation`/`end_trace`, `metadata` mode cannot
        just withhold a field and still send the call: it refuses locally,
        before anything is sent. `redacted` sends the placeholder; `record`
        sends the text after mandatory secret scrubbing.

        `content_mode` (A1/finding 8, Python/TS parity): a per-call override,
        resolved against this TRACE's own base mode (if `trace_id` is
        locally tracked) or the client config otherwise -- see
        `_resolve_trace_scoped_content_mode`.
        """
        effective_content_mode = self._resolve_trace_scoped_content_mode(
            trace_id, content_mode
        )
        if effective_content_mode == "metadata":
            raise ContentDisabledError(
                "add_comment content is disabled because content_mode="
                "'metadata'; use content_mode='redacted' or 'record' on this "
                "client, or attach non-content metadata instead."
            )
        gated_content = apply_content_mode_to_text(content, effective_content_mode)
        if effective_content_mode == "record" and gated_content is not None:
            # M8/A6 fail closed: a scrub failure must withhold, never send
            # the unscrubbed comment or raise out of this call. Unlike
            # `record_observation`'s content fields, the backend's
            # `TraceCommentCreateRequest.content` REQUIRES a non-empty
            # string -- sending `None` here does not withhold, it sends an
            # invalid request the backend rejects. Fall back to the
            # `"[REDACTED]"` text placeholder instead (finding 9's confirmed
            # test gap: the old fallback of `None` was never actually
            # exercised against that requirement).
            try:
                gated_content = redact_sensitive_text(gated_content)
            except Exception:
                logger.exception(
                    "Comment secret scrubbing failed; sending the redaction "
                    "placeholder rather than the unscrubbed comment.",
                )
                gated_content = _REDACTED
        payload = self._request_json(
            "POST",
            f"/traces/{trace_id}/comments",
            {"content": gated_content},
        )
        return TraceCommentRecord.from_dict(self._unwrap_data(payload, "trace comment"))

    def _gate_correction_output(
        self, correction_output: Any, effective_content_mode: str
    ) -> Any:
        # B2 (addendum R3, A5 completion): a hostile `correction_output`
        # (uncopyable, cyclic, not JSON serializable) must never raise
        # out of this public call -- `_ensure_json_serializable` used to
        # raise `ClientError` straight into the caller's application.
        # Withhold the
        # field instead, same as a `record`-mode scrub failure below.
        try:
            correction_output = self._ensure_json_serializable(
                correction_output,
                field_name="correction_output",
            )
            gated_correction_output = apply_content_mode(
                correction_output, effective_content_mode
            )
        except Exception:
            logger.exception(
                "Feedback correction_output failed JSON-serializability "
                "validation; withholding it rather than raising into "
                "the caller's application.",
            )
            gated_correction_output = None
        if effective_content_mode == "record" and gated_correction_output is not None:
            try:
                gated_correction_output = redact_sensitive_data(
                    gated_correction_output, redact_credential_keys=True
                )
            except Exception:
                logger.exception(
                    "Feedback correction_output secret scrubbing failed; "
                    "withholding it rather than sending it unscrubbed.",
                )
                gated_correction_output = None
        return gated_correction_output

    def submit_feedback(
        self,
        trace_id: str,
        rating: ThumbRating | str,
        *,
        comment: str | None = None,
        correction_output: Any = NOT_SUPPLIED,
        content_mode: str | None = None,
    ) -> TraceFeedbackResponse:
        """Submit thumbs up/down feedback, an optional comment and correction (M3).

        Unlike `add_comment`, this call always has a non-content payload
        (`rating`) it can send, so `metadata` mode never raises here: it
        sends `rating` and silently omits `comment`/`correction_output`.
        `redacted` sends placeholders for both; `record` sends them after
        mandatory secret scrubbing. `correction_output` is validated as
        JSON-serializable regardless of mode (fail fast on malformed input
        even when the mode would go on to withhold it).

        `content_mode` (A1/finding 8, Python/TS parity): same per-call
        override as `add_comment`.
        """
        if isinstance(rating, str):
            rating = ThumbRating(rating)

        effective_content_mode = self._resolve_trace_scoped_content_mode(
            trace_id, content_mode
        )

        gated_comment = apply_content_mode_to_text(comment, effective_content_mode)
        if effective_content_mode == "record" and gated_comment is not None:
            # M8 fail closed: see `add_comment`.
            try:
                gated_comment = redact_sensitive_text(gated_comment)
            except Exception:
                logger.exception(
                    "Feedback comment secret scrubbing failed; withholding "
                    "comment content rather than sending it unscrubbed.",
                )
                gated_comment = None

        gated_correction_output: Any = None
        if correction_output is not NOT_SUPPLIED:
            gated_correction_output = self._gate_correction_output(
                correction_output, effective_content_mode
            )

        feedback_payload: dict[str, Any] = {"rating": rating.value}
        if gated_comment is not None:
            feedback_payload["comment"] = gated_comment
        if gated_correction_output is not None:
            feedback_payload["correction_output"] = gated_correction_output

        payload = self._request_json(
            "PUT",
            f"/traces/{trace_id}/feedback",
            feedback_payload,
        )
        return TraceFeedbackResponse.from_dict(
            self._unwrap_data(payload, "trace feedback")
        )

    def update_collaboration(
        self,
        trace_id: str,
        *,
        is_bookmarked: bool | None = None,
        is_published: bool | None = None,
    ) -> TraceCollaborationState:
        payload = self._request_json(
            "PATCH",
            f"/traces/{trace_id}/collaboration",
            {
                "is_bookmarked": is_bookmarked,
                "is_published": is_published,
            },
        )
        return TraceCollaborationState.from_dict(
            self._unwrap_data(payload, "trace collaboration")
        )

    def set_bookmarked(
        self, trace_id: str, is_bookmarked: bool
    ) -> TraceCollaborationState:
        return self.update_collaboration(trace_id, is_bookmarked=is_bookmarked)

    def set_published(
        self, trace_id: str, is_published: bool
    ) -> TraceCollaborationState:
        return self.update_collaboration(trace_id, is_published=is_published)

    def list_traces(
        self,
        *,
        page: int = 1,
        per_page: int = 20,
        search: str | None = None,
        environment: str | None = None,
        status: str | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
        tags: list[str] | str | None = None,
        release: str | None = None,
        model: str | None = None,
        start_time_from: datetime | None = None,
        start_time_to: datetime | None = None,
    ) -> TraceListResponse:
        query = self._build_query_string(
            page=page,
            per_page=per_page,
            search=search,
            environment=environment,
            status=status,
            user_id=user_id,
            session_id=session_id,
            tags=tags,
            release=release,
            model=model,
            start_time_from=start_time_from,
            start_time_to=start_time_to,
        )
        payload = self._request_json("GET", f"/traces{query}")
        return TraceListResponse.from_dict(self._unwrap_data(payload, "trace list"))

    def iter_traces(
        self,
        *,
        per_page: int = 100,
        search: str | None = None,
        environment: str | None = None,
        status: str | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
        tags: list[str] | str | None = None,
        release: str | None = None,
        model: str | None = None,
        start_time_from: datetime | None = None,
        start_time_to: datetime | None = None,
    ) -> Iterator[TraceRecord]:
        """Iterate over *all* traces across pages, fetching lazily.

        Filter arguments are forwarded unchanged on every page request.

        Parameters
        ----------
        per_page:
            Items per page.  Defaults to 100 to minimise round-trips.
        search, environment, status, user_id, session_id, tags, release,
        model, start_time_from, start_time_to:
            Same filter semantics as :meth:`list_traces`.
        """
        yield from iter_pages(
            self.list_traces,
            per_page=per_page,
            search=search,
            environment=environment,
            status=status,
            user_id=user_id,
            session_id=session_id,
            tags=tags,
            release=release,
            model=model,
            start_time_from=start_time_from,
            start_time_to=start_time_to,
        )

    def get_trace(self, trace_id: str) -> TraceRecord:
        payload = self._request_json("GET", f"/traces/{trace_id}")
        return TraceRecord.from_dict(self._unwrap_data(payload, "trace detail"))

    def get_trace_observations(self, trace_id: str) -> TraceObservationsResponse:
        payload = self._request_json("GET", f"/traces/{trace_id}/observations")
        return TraceObservationsResponse.from_dict(
            self._unwrap_data(payload, "trace observations")
        )

    def list_sessions(
        self,
        *,
        page: int = 1,
        per_page: int = 20,
        search: str | None = None,
        environment: str | None = None,
        user_id: str | None = None,
        tags: list[str] | str | None = None,
        release: str | None = None,
        start_time_from: datetime | None = None,
        start_time_to: datetime | None = None,
    ) -> SessionListResponse:
        query = self._build_query_string(
            page=page,
            per_page=per_page,
            search=search,
            environment=environment,
            user_id=user_id,
            tags=tags,
            release=release,
            start_time_from=start_time_from,
            start_time_to=start_time_to,
        )
        payload = self._request_json("GET", f"/sessions{query}")
        return SessionListResponse.from_dict(self._unwrap_data(payload, "session list"))

    def iter_sessions(
        self,
        *,
        per_page: int = 100,
        search: str | None = None,
        environment: str | None = None,
        user_id: str | None = None,
        tags: list[str] | str | None = None,
        release: str | None = None,
        start_time_from: datetime | None = None,
        start_time_to: datetime | None = None,
    ) -> Iterator[SessionRecord]:
        """Iterate over *all* sessions across pages, fetching lazily.

        Filter arguments are forwarded unchanged on every page request.

        Parameters
        ----------
        per_page:
            Items per page.  Defaults to 100 to minimise round-trips.
        search, environment, user_id, tags, release, start_time_from,
        start_time_to:
            Same filter semantics as :meth:`list_sessions`.
        """
        yield from iter_pages(
            self.list_sessions,
            per_page=per_page,
            search=search,
            environment=environment,
            user_id=user_id,
            tags=tags,
            release=release,
            start_time_from=start_time_from,
            start_time_to=start_time_to,
        )

    def get_session(self, session_id: str) -> SessionRecord:
        payload = self._request_json("GET", f"/sessions/{session_id}")
        return SessionRecord.from_dict(self._unwrap_data(payload, "session detail"))

    def _create_transport(self) -> _SyncBatchTransport:
        # The backend's per-request caps (#2600) bind only the SDK's own ingest
        # sender; a custom sender's sink has its own limits.
        backend_sink = self._sender_override is None
        return _SyncBatchTransport(
            self._send_payload_batch,
            batch_size=self.config.batch_size,
            max_buffer_age=self.config.max_buffer_age,
            max_queue_size=self.config.max_queue_size,
            max_batch_bytes=self.config.max_batch_bytes,
            health_callback=self.config.health_callback,
            max_traces_per_request=(
                INGEST_MAX_TRACES_PER_REQUEST if backend_sink else None
            ),
            max_observations_per_request=(
                INGEST_MAX_OBSERVATIONS_PER_REQUEST if backend_sink else None
            ),
        )

    def _send_payload_batch(
        self, traces: list[dict[str, Any]]
    ) -> dict[str, Any] | None:
        if self._sender_override is not None:
            return self._sender_override(traces)

        return self._post_batch_sync(traces)

    def _optimization_only_policy(self) -> oo_egress.Policy:
        """Policy inputs for this client's credential (no explicit setting of its own)."""
        if not oo_egress.scope_required(None):
            return oo_egress.Policy()
        try:
            headers = self.config.build_headers()
            origin = self.config.backend_origin
        except Exception:  # noqa: BLE001 - silent-ok: unattributable, env/explicit only
            return oo_egress.Policy()
        return oo_egress.Policy(None, oo_egress.credential_scope(origin, headers))

    @property
    def _has_custom_trace_sender(self) -> bool:
        return self._sender_override is not None

    @property
    def _ingest_suppressed(self) -> bool:
        """Trace-snapshot delivery is suppressed for this client.

        True when offline mode or the missing-credential guard is active and
        no custom sender owns delivery — a custom sender is a local lane the
        SDK never authenticates, so it keeps working in both cases.
        """
        return (
            self.config.offline_mode
            or self._credential_egress_disabled
            or oo_egress.is_active(self._optimization_only_policy())
        ) and not self._has_custom_trace_sender

    def _post_batch_sync(self, traces: list[dict[str, Any]]) -> dict[str, Any] | None:
        if oo_egress.is_active(self._optimization_only_policy()):
            # optimization_only: the server refuses trace ingest; nothing is sent.
            return None
        if self.config.offline_mode or self._credential_egress_disabled:
            self._log_offline_mode_once()
            return None

        payload = json.dumps({"traces": traces}).encode("utf-8")
        http_request = request.Request(
            self.config.ingest_url,
            data=payload,
            headers=self.config.build_headers(),
            method="POST",
        )
        try:
            with self._open_http_request(
                http_request,
                timeout=self.config.effective_request_timeout(len(traces)),
            ) as response:
                status_code = getattr(response, "status", 200)
                body = response.read().decode("utf-8") if response else ""
                if status_code >= 400:
                    raise self._client_error_with_retry_after(
                        f"Observability ingest failed with status {status_code}",
                        status_code=status_code,
                        details={"body": body},
                        headers=getattr(response, "headers", None),
                    )
                return self._parse_ingest_response(body)
        except error.HTTPError as exc:
            body = self._read_http_error_body(exc)
            oo_egress.note_response(
                exc.code, body, self._optimization_only_policy().scope
            )
            if exc.code in {401, 403}:
                raise AuthenticationError(
                    f"Observability ingest rejected with status {exc.code}"
                ) from exc
            raise self._client_error_with_retry_after(
                f"Observability ingest failed with status {exc.code}",
                status_code=exc.code,
                details={"body": body},
                headers=getattr(exc, "headers", None),
            ) from exc
        except error.URLError as exc:
            raise TraigentConnectionError(
                f"Failed to connect to observability backend at {self.config.ingest_url}"
            ) from exc

    def _log_ingest_warnings(self, body: str) -> None:
        payload = self._decode_ingest_response(body)
        self._log_ingest_warnings_from_data(payload)

    def _log_ingest_warnings_from_data(self, payload: dict[str, Any] | None) -> None:
        warnings = (payload or {}).get("warnings")
        if not isinstance(warnings, list) or not warnings:
            return

        for warning in warnings:
            logger.warning("Observability ingest warning: %s", warning)

    def _parse_ingest_response(self, body: str) -> dict[str, Any] | None:
        data = self._decode_ingest_response(body)
        self._log_ingest_warnings_from_data(data)
        return data

    def _decode_ingest_response(self, body: str) -> dict[str, Any] | None:
        if not body:
            return None
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError:
            return None
        data = parsed.get("data")
        if not isinstance(data, dict):
            return None
        return data

    def _request_json(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            if self._closed or self._close_pending:
                raise ClientError("Observability client is closed")
        if self.config.offline_mode:
            self._log_offline_mode_once()
            raise ClientError(
                "Observability backend request skipped because offline mode is "
                f"enabled ({describe_offline_mode_cause()})"
            )
        if self._request_sender_override is not None:
            return cast(
                dict[str, Any], self._request_sender_override(method, path, payload)
            )
        oo_egress.refuse_if_active(
            "observability_request", explicit=self._optimization_only_policy()
        )
        if self._credential_egress_disabled:
            raise ClientError(
                "Observability backend request skipped: no credential resolved "
                "(set TRAIGENT_API_KEY or run `traigent auth login`)"
            )
        return self._request_json_sync(method, path, payload)

    def _request_json_sync(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        encoded_payload = None
        if payload is not None:
            encoded_payload = json.dumps(payload).encode("utf-8")

        http_request = request.Request(
            f"{self.config.backend_origin}{self.config.api_path}{path}",
            data=encoded_payload,
            headers=self.config.build_headers(),
            method=method,
        )
        try:
            with self._open_http_request(http_request) as response:
                status_code = getattr(response, "status", 200)
                body = response.read().decode("utf-8") if response else ""
                parsed = json.loads(body) if body else {}
                if status_code >= 400:
                    raise self._client_error_with_retry_after(
                        f"Observability request failed with status {status_code}",
                        status_code=status_code,
                        details={"body": body},
                        headers=getattr(response, "headers", None),
                    )
                return parsed
        except error.HTTPError as exc:
            body = self._read_http_error_body(exc)
            oo_egress.note_response(
                exc.code, body, self._optimization_only_policy().scope
            )
            if exc.code in {401, 403}:
                raise AuthenticationError(
                    f"Observability request rejected with status {exc.code}"
                ) from exc
            raise self._client_error_with_retry_after(
                f"Observability request failed with status {exc.code}",
                status_code=exc.code,
                details={"body": body},
                headers=getattr(exc, "headers", None),
            ) from exc
        except error.URLError as exc:
            raise TraigentConnectionError(
                f"Failed to connect to observability backend at {self.config.backend_origin}"
            ) from exc

    def _unwrap_data(self, payload: dict[str, Any], label: str) -> dict[str, Any]:
        data = payload.get("data")
        if not isinstance(data, dict):
            raise ClientError(f"Unexpected response structure for {label}")
        return data

    def _open_http_request(
        self, http_request: request.Request, timeout: float | None = None
    ) -> Any:
        return self._http_opener.open(
            http_request,
            timeout=self.config.request_timeout if timeout is None else timeout,
        )

    @staticmethod
    def _validate_observation_relationship(
        state: _TraceState,
        *,
        observation_id: str,
        observation_type: ObservationType,
        parent_observation_id: str | None,
    ) -> None:
        if parent_observation_id is not None:
            parent = state.observations.get(parent_observation_id)
            if parent is not None and parent.type is ObservationType.EVENT:
                raise ValueError("event observations cannot have children")

        if observation_type is ObservationType.EVENT:
            has_children = any(
                existing_id != observation_id
                and observation.parent_observation_id == observation_id
                for existing_id, observation in state.observations.items()
            )
            if has_children:
                raise ValueError("event observations cannot have children")

    def _queue_trace_snapshot(self, trace_id: str) -> None:
        if self._ingest_suppressed:
            return
        with self._lock:
            state = self._trace_states.get(trace_id)
            if state is None or self._closed or self._close_pending:
                return
            # Snapshot immutability: `_safe_trace_snapshot` freezes a
            # `_TraceFallbackDescriptor` from `state` right here, while this
            # lock is still held. `_submit_trace_snapshot` below runs AFTER
            # the lock is released -- possibly well after, and on a
            # different thread -- so it must receive that frozen
            # `descriptor`, never a reference to the live, shared `state`,
            # which a concurrent `record_observation`/`end_trace` call on
            # this same trace could keep mutating in the meantime (the
            # confirmed probe: completing a running observation and adding a
            # new one leaked into what should have been a stale snapshot's
            # fallback).
            payload, descriptor = _safe_trace_snapshot(state)
            self._inflight_snapshot_submissions += 1
            self._snapshot_submission_complete.clear()
        try:
            self._submit_trace_snapshot(trace_id, payload, state=descriptor)
        finally:
            with self._lock:
                self._inflight_snapshot_submissions -= 1
                if self._inflight_snapshot_submissions == 0:
                    self._snapshot_submission_complete.set()

    def _submit_trace_snapshot(
        self,
        trace_id: str,
        payload: dict[str, Any],
        *,
        deadline: float | None = None,
        state: _TraceFallbackDescriptor | None = None,
    ) -> None:
        # B3: thread `state` (a `_TraceFallbackDescriptor`, frozen at
        # enqueue time under the client lock -- never the live `_TraceState`
        # -- finding 2) through to the transport's own defensive redaction
        # pass (`_SyncBatchTransport._prepare_payload`) too -- that pass
        # re-runs `_redact_trace_payload` on the ALREADY-safe payload
        # `_safe_trace_snapshot` just built. It is normally a no-op (the
        # payload is already redacted/fallback-built), but if it were ever
        # to fail too (e.g. the same scrubbing outage), it must reach the
        # SAME `state.content_history`-aware fallback, not a state-blind one
        # that would drop the placeholder this call just carefully built.
        submitted = (
            self._transport.submit(trace_id, payload, state=state)
            if deadline is None
            else self._transport.submit(
                trace_id, payload, deadline=deadline, state=state
            )
        )
        if not submitted:
            stats = (
                self._transport.get_stats()
                if deadline is None
                else self._transport.get_stats(
                    timeout=self._remaining_flush_time(deadline)
                )
            )
            errors = stats.get("errors") or []
            reason = errors[-1] if errors else "transport rejected trace snapshot"
            reason = redact_sensitive_text(str(reason))
            logger.warning(
                "Observability trace snapshot for %s was not queued: %s",
                trace_id,
                reason,
            )

    def _disable_egress_if_credential_missing(self) -> None:
        """Fail fast (quietly) when no credential resolves for network egress.

        Without a credential the backend rejects every ingest batch with 401,
        which the retry handler turns into a per-batch retry storm and, once the
        circuit breaker opens, an opaque "ingest rejected with status 401" that
        never names the missing key. When neither an API key, a JWT, nor a
        non-blank Authorization/X-API-Key in ``extra_headers`` resolved, log one
        actionable warning and disable the SDK's own network egress for the
        process. Suppression is lane-aware rather than a global offline flip:
        a custom ``sender`` keeps owning ingest delivery and a custom
        ``request_sender`` keeps owning control-plane calls — only the
        network lanes the SDK itself would drive unauthenticated are disabled.
        ``config.offline_mode`` is left untouched so it keeps reflecting the
        caller's explicit setting. Telemetry must never break the host app, so
        this never raises.
        """
        if self.config.offline_mode:
            return
        if (
            self._sender_override is not None
            and self._request_sender_override is not None
        ):
            # Every egress lane is override-covered; the SDK itself will never
            # emit network traffic, authenticated or not.
            return
        if nonblank_credential(self.config.api_key) or nonblank_credential(
            self.config.jwt_token
        ):
            return
        # A caller supplying auth through extra_headers (gateway/proxy setups)
        # has a working credential the backend will accept — the 401-storm
        # premise this guard exists for does not apply. Header names alone do
        # not count: a blank value is as unauthenticated as a missing header.
        if any(
            name.strip().lower() in ("authorization", "x-api-key")
            and nonblank_credential(value)
            for name, value in self.config.extra_headers.items()
        ):
            return

        logger.warning(
            "No Traigent credential resolved (set TRAIGENT_API_KEY or run "
            "`traigent auth login`); observability network egress disabled "
            "for this process"
        )
        self._credential_egress_disabled = True
        # The warning above already told the user egress is disabled; suppress
        # the vaguer INFO offline notice so exactly one message is emitted.
        self._offline_notice_logged = True

    def _log_offline_mode_once(self) -> None:
        if self._offline_notice_logged:
            return
        self._offline_notice_logged = True
        logger.info("Observability transport in offline mode; traces will not be sent")

    def _suppressed_delivery_warning(self) -> str:
        """Explain why trace delivery is suppressed, naming the real cause.

        Suppression deliberately keeps ``FlushResult.success`` true (telemetry
        must never break the host application), so this warning is what lets a
        caller tell "nothing was delivered" apart from a real delivery (#2602).
        """
        if self.config.offline_mode:
            cause = f"offline mode is enabled ({describe_offline_mode_cause()})"
        else:
            cause = (
                "no credential resolved (set TRAIGENT_API_KEY or run "
                "`traigent auth login`)"
            )
        return (
            f"Observability trace delivery is suppressed because {cause}; "
            "no traces were sent to the backend"
        )

    def _suppressed_flush_result(self) -> FlushResult:
        return FlushResult(
            success=True,
            items_sent=0,
            items_pending=0,
            items_dropped=0,
            successful_batches=0,
            failed_batches=0,
            errors=[],
            warnings=[self._suppressed_delivery_warning()],
        )

    def _with_suppressed_delivery_warning(self, result: FlushResult) -> FlushResult:
        return replace(
            result, warnings=[*result.warnings, self._suppressed_delivery_warning()]
        )

    def _build_query_string(self, **params: Any) -> str:
        serialized: dict[str, str | list[str]] = {}
        for key, value in params.items():
            if value is None or value == "":
                continue
            if isinstance(value, datetime):
                serialized[key] = value.isoformat()
            elif isinstance(value, list):
                # Repeated keys (`tags=a&tags=b`), never a comma-join: the backend reads
                # list filters with `getlist` and keeps each value as one literal item,
                # so `tags=a,b` would be the single tag "a,b" and match nothing.
                cleaned = [str(item).strip() for item in value if str(item).strip()]
                if cleaned:
                    serialized[key] = cleaned
            else:
                serialized[key] = str(value)

        if not serialized:
            return ""
        return "?" + urlencode(serialized, doseq=True)

    def _ensure_json_serializable(self, value: Any, *, field_name: str) -> Any:
        if value is None:
            return None
        try:
            json.dumps(value)
        except (TypeError, ValueError) as exc:
            raise ClientError(f"{field_name} must be JSON serializable") from exc
        return value

    @staticmethod
    def _client_error_with_retry_after(
        message: str,
        *,
        status_code: int,
        details: dict[str, Any],
        headers: Any,
    ) -> ClientError:
        retry_after = _parse_retry_after(headers)
        if retry_after is not None:
            return _ClientErrorWithRetryAfter(
                message,
                status_code=status_code,
                details=details,
                retry_after=retry_after,
            )
        return ClientError(message, status_code=status_code, details=details)

    def _read_http_error_body(self, exc: error.HTTPError) -> str:
        try:
            if exc.fp is None:
                return ""
            body = exc.read()
            if isinstance(body, bytes):
                return body.decode("utf-8")
            if isinstance(body, io.BytesIO):
                return body.getvalue().decode("utf-8")
            return str(body)
        finally:
            exc.close()

    def _coerce_session(
        self, session: SessionDTO | dict[str, Any] | None
    ) -> SessionDTO | None:
        # A8: detach from the caller's own object/dict (deep copy) -- a
        # `SessionDTO` instance is mutable, and its `metadata` field is a
        # caller-owned dict either way, so a caller mutating it after this
        # call returns must never change a later re-serialized snapshot
        # (finding 7). B2: the deep copy (and, for a dict input, the
        # `SessionDTO(**...)` construction) must never raise out of
        # `start_trace` on a hostile value -- a failure drops `session`
        # entirely (withheld) rather than propagating.
        if session is None:
            return None
        try:
            if isinstance(session, SessionDTO):
                copied = _safe_deepcopy_or_default(session, None, context="session")
                return copied if isinstance(copied, SessionDTO) else None
            copied_dict = _safe_deepcopy_or_default(session, None, context="session")
            if not isinstance(copied_dict, dict):
                return None
            return SessionDTO(**copied_dict)
        except Exception:
            logger.exception(
                "Observability session coercion failed; withholding the "
                "session rather than raising into the caller's application.",
            )
            return None

    def _coerce_correlation_ids(
        self,
        correlation_ids: CorrelationIds | dict[str, str] | None,
    ) -> CorrelationIds | None:
        # B5 (addendum R3, A8 completion): an already-`CorrelationIds`
        # caller value was previously returned BY REFERENCE -- the confirmed
        # bug where mutating the caller's own `CorrelationIds` instance
        # after this call changed a later re-serialized snapshot. Always
        # detach (deep copy), same as `_coerce_session`/`_coerce_prompt_
        # reference`. B2: the copy/construction must never raise out of
        # `start_trace`/`record_observation` on a hostile value.
        if correlation_ids is None:
            return None
        try:
            if isinstance(correlation_ids, CorrelationIds):
                copied = _safe_deepcopy_or_default(
                    correlation_ids, None, context="correlation_ids"
                )
                return copied if isinstance(copied, CorrelationIds) else None
            copied_dict = _safe_deepcopy_or_default(
                correlation_ids, None, context="correlation_ids"
            )
            if not isinstance(copied_dict, dict):
                return None
            return CorrelationIds(**copied_dict)
        except Exception:
            logger.exception(
                "Observability correlation_ids coercion failed; withholding "
                "it rather than raising into the caller's application.",
            )
            return None

    def _coerce_prompt_reference(
        self,
        prompt_reference: PromptReferenceDTO | dict[str, Any] | None,
    ) -> PromptReferenceDTO | None:
        # A8: `variables` is an arbitrary caller-chosen `dict[str, Any]` --
        # detach it the same way `_coerce_session` detaches session metadata
        # (finding 7). B2: the copy/construction must never raise out of
        # `start_trace`/`record_observation` on a hostile value -- a
        # failure drops `prompt_reference` entirely (withheld).
        if prompt_reference is None:
            return None
        try:
            if isinstance(prompt_reference, PromptReferenceDTO):
                copied = _safe_deepcopy_or_default(
                    prompt_reference, None, context="prompt_reference"
                )
                return copied if isinstance(copied, PromptReferenceDTO) else None
            copied_dict = _safe_deepcopy_or_default(
                prompt_reference, None, context="prompt_reference"
            )
            if not isinstance(copied_dict, dict):
                return None
            return PromptReferenceDTO(**copied_dict)
        except Exception:
            logger.exception(
                "Observability prompt_reference coercion failed; "
                "withholding it rather than raising into the caller's "
                "application.",
            )
            return None

    def _coerce_execution_context(
        self,
        execution_context: ExecutionContextDTO | dict[str, Any] | None,
    ) -> ExecutionContextDTO | None:
        defaults = dict(self.config.default_execution_context)
        if isinstance(execution_context, ExecutionContextDTO):
            explicit = execution_context.to_dict()
        else:
            explicit = dict(execution_context or {})
        merged = {**defaults, **explicit}
        if not merged:
            return None
        merged.setdefault("schema_version", "1.0")
        return ExecutionContextDTO.from_dict(merged)

    def _atexit_close(self) -> None:
        with self._lock:
            if self._closed:
                return
        try:
            result = self.close(timeout=self.config.flush_timeout)
        except Exception as exc:
            logger.warning("Observability atexit flush failed: %s", exc)
            return
        pending = getattr(result, "items_pending", 0)
        if (
            self.config.warn_on_undelivered_at_exit
            and pending > 0
            and not getattr(self, "_exit_warning_written", False)
        ):
            self._exit_warning_written = True
            try:
                sys.stderr.write(
                    f"traigent: {pending} observability trace(s) were not "
                    "delivered before exit; call flush() before exit or raise "
                    "ObservabilityConfig.flush_timeout.\n"
                )
            except Exception:  # noqa: BLE001 - stderr may be closed at exit
                pass

    def _flush_deadline(self, timeout: float | None) -> float | None:
        if timeout is None:
            return None
        return time.monotonic() + max(0.0, timeout)

    def _acquire_client_lock_until(self, deadline: float | None) -> bool:
        """Acquire client state without allowing a bounded call to exceed its budget."""
        if deadline is None:
            return self._lock.acquire()
        return self._lock.acquire(timeout=max(0.0, deadline - time.monotonic()))

    @staticmethod
    def _remaining_flush_time(deadline: float | None) -> float | None:
        if deadline is None:
            return None
        return max(0.0, deadline - time.monotonic())
