"""Batch/request construction + idempotency for economics telemetry (WI-B).

This module wraps caller-supplied, schema-shaped events into the authoritative
batch envelope and enforces the client-side egress contract on every
``run_economics`` event before the request is handed to transport. It also
derives a caller-stable idempotency key that MATCHES the body, so a retried
batch replays rather than double-counting a funnel.

It deliberately does NOT re-encode the full closed contract as typed builders:
the backend today accepts only the project-level ``eligible`` funnel entry (no
authoritative producer exists for advice-bearing stages, run settlements, or
receipts yet), so the one usable path gets a typed constructor
(:func:`funnel_eligible_event`) and everything else is accepted as a
schema-shaped dict and validated at the boundary. That keeps the SDK from
fabricating lifecycle it cannot authoritatively produce, while still letting a
caller represent the full contract when a producer exists.
"""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any

from traigent.economics.contract import (
    CONTRACT_ID,
    CONTRACT_VERSION,
    MAX_BATCH_EVENTS,
    SOURCE_KIND,
    SOURCE_NAME,
)
from traigent.economics.egress import enforce_characterization_egress
from traigent.economics.errors import EconomicsTelemetryContractError

# ShortLabel grammar the ``source.version`` must satisfy.
_SHORT_LABEL_RE = re.compile(r"^[A-Za-z0-9]([A-Za-z0-9._:/+-]*[A-Za-z0-9])?$")
_IDEMPOTENCY_KEY_RE = re.compile(r"^[A-Za-z0-9._:-]{8,128}$")


def utc_now_z() -> str:
    """Return an RFC-3339 UTC timestamp with millisecond precision and a Z.

    Matches the contract's ``UtcTimestamp`` form; a local-offset timestamp would
    silently move an event by hours, reordering the funnel.
    """
    now = datetime.now(UTC)
    return now.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def _sdk_source_version() -> str:
    """Resolve a ShortLabel-safe SDK version for the ``source`` block."""
    version: str | None = None
    try:  # pragma: no cover - resolution path varies by install
        from importlib import metadata as importlib_metadata

        version = importlib_metadata.version("traigent")
    except Exception:  # noqa: BLE001 - fall through to the file-based resolver
        try:
            from traigent._version import get_version

            version = get_version()
        except Exception:  # noqa: BLE001 - last resort below
            version = None
    if isinstance(version, str) and _SHORT_LABEL_RE.fullmatch(version):
        return version
    return "0.0.0"


def build_source(
    *, name: str = SOURCE_NAME, version: str | None = None
) -> dict[str, str]:
    """Build the ``source`` block identifying this emitting surface."""
    resolved_version = version if version is not None else _sdk_source_version()
    if not _SHORT_LABEL_RE.fullmatch(name):
        raise EconomicsTelemetryContractError("source name is not a valid ShortLabel")
    if not _SHORT_LABEL_RE.fullmatch(resolved_version):
        raise EconomicsTelemetryContractError(
            "source version is not a valid ShortLabel"
        )
    return {"kind": SOURCE_KIND, "name": name, "version": resolved_version}


def funnel_eligible_event(
    project_ref: str,
    *,
    event_id: str | None = None,
    occurred_at: str | None = None,
    occurred_in_environment: str | None = None,
) -> dict[str, Any]:
    """Build the one funnel event the backend accepts through public ingest today.

    The project-level ``eligible``/``entered`` stage carries no advice_id and no
    run_id (both exempt at eligibility), so it is the single stage that has an
    authoritative producer and can be accepted rather than failed closed.

    ``project_ref`` must equal the project the batch is submitted under; the
    backend verifies this against the authenticated request context and rejects
    a mismatch as a tenant-scope violation.
    """
    ref = _require_non_empty(project_ref, "project_ref")
    event: dict[str, Any] = {
        "event_type": "funnel_event",
        "event_id": event_id or f"evt-{uuid.uuid4().hex}",
        "occurred_at": occurred_at or utc_now_z(),
        "project_ref": ref,
        "stage": "eligible",
        "outcome": "entered",
    }
    if occurred_in_environment is not None:
        event["occurred_in_environment"] = occurred_in_environment
    return event


def build_telemetry_request(
    events: Sequence[Mapping[str, Any]],
    *,
    batch_id: str | None = None,
    sent_at: str | None = None,
    idempotency_key: str | None = None,
    source: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Assemble one economics telemetry batch request body.

    Enforces the client-side egress contract on every ``run_economics`` event
    (raising before any bytes leave the machine), then wraps the events in the
    authoritative envelope with a caller-stable idempotency key that matches the
    body.

    Args:
        events: Schema-shaped event objects (1..500). Deep-copied into the body.
        batch_id: Opaque batch id; a content-stable default is derived when None.
        sent_at: RFC-3339 UTC send time; defaults to now.
        idempotency_key: Explicit key; a stable key derived from the body is used
            when None. When supplied, the caller owns pairing a stable key with a
            stable body (a differing body under the same key conflicts, 409).
        source: Explicit source block; the SDK's own is used when None.

    Returns:
        The batch request body, ready to POST.

    Raises:
        EconomicsTelemetryContractError: On a malformed batch shape.
        EgressPolicyError: On a characterization egress violation.
    """
    if not isinstance(events, Sequence) or isinstance(events, (str, bytes)):
        raise EconomicsTelemetryContractError("events must be a sequence")
    event_list = list(events)
    if not event_list:
        raise EconomicsTelemetryContractError(
            "a telemetry batch requires at least one event"
        )
    if len(event_list) > MAX_BATCH_EVENTS:
        raise EconomicsTelemetryContractError(
            f"a telemetry batch cannot exceed {MAX_BATCH_EVENTS} events"
        )

    normalized: list[dict[str, Any]] = []
    for index, event in enumerate(event_list):
        if not isinstance(event, Mapping):
            raise EconomicsTelemetryContractError(f"event {index} must be an object")
        copied = _json_roundtrip(event, index)
        if copied.get("event_type") == "run_economics":
            # Egress enforcement runs on the settled characterization BEFORE the
            # batch can be built, so a withheld value can never be transmitted.
            enforce_characterization_egress(copied.get("characterization") or {})
        normalized.append(copied)

    body: dict[str, Any] = {
        "contract": CONTRACT_ID,
        "contract_version": CONTRACT_VERSION,
        "batch_id": batch_id or _derive_batch_id(normalized),
        "sent_at": sent_at or utc_now_z(),
        "source": dict(source) if source is not None else build_source(),
        "events": normalized,
    }

    key = (
        idempotency_key
        if idempotency_key is not None
        else _derive_idempotency_key(body)
    )
    if not isinstance(key, str) or not _IDEMPOTENCY_KEY_RE.fullmatch(key):
        raise EconomicsTelemetryContractError(
            "idempotency_key is not a valid IdempotencyKey"
        )
    # The key lives in the body AND is sent in the header; the two must match.
    body["idempotency_key"] = key
    return body


def canonical_json(value: Any) -> str:
    """Deterministic JSON encoding for stable hashing (sorted keys, compact).

    Rejects NaN/Infinity (``allow_nan=False``): a non-finite number cannot be
    represented in the contract's bounded numeric types and must fail locally,
    not travel as an invalid JSON token.

    The serialization exception is confined to its handler and NOT chained: a
    ``json`` error can carry the offending value, so the raised contract error is
    payload-free (no ``__cause__``/``__context__``).
    """
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError):
        pass
    raise EconomicsTelemetryContractError(
        "telemetry payload is not serializable to canonical JSON"
    )


def canonical_json_bytes(value: Any) -> bytes:
    """Canonical JSON encoded to UTF-8 bytes — the ONE payload-safe chokepoint.

    Every batch-id, idempotency-key, and wire serialization path routes through
    here so none diverge. Beyond the ``json`` errors ``canonical_json`` catches,
    this also catches ``UnicodeError`` at the ``.encode`` step: with
    ``ensure_ascii=False`` a lone surrogate is a valid ``str`` but not valid
    UTF-8, and the resulting ``UnicodeEncodeError.object`` would carry the ENTIRE
    canonical payload. The failure is confined to its handler and a fresh
    contract error is raised OUTSIDE it, so the payload never rides along in the
    message, in ``__cause__``/``__context__``, or in an exception attribute.
    """
    try:
        text = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        return text.encode("utf-8")
    except (TypeError, ValueError, UnicodeError):
        pass
    raise EconomicsTelemetryContractError(
        "telemetry payload is not serializable to canonical UTF-8 JSON"
    )


def _derive_idempotency_key(body_without_key: Mapping[str, Any]) -> str:
    """Derive a stable idempotency key from the body (excluding the key itself).

    The key is a pure function of the rest of the body, so header and body agree
    by construction and re-sending the identical built body replays. A byte-level
    change yields a fresh key (a distinct batch), never a silent 409.
    """
    digest = hashlib.sha256(canonical_json_bytes(body_without_key)).hexdigest()
    return f"econ-tel-{digest[:48]}"


def _derive_batch_id(events: Sequence[Mapping[str, Any]]) -> str:
    """Derive a content-stable batch id from the events."""
    digest = hashlib.sha256(canonical_json_bytes(list(events))).hexdigest()
    return f"batch-{digest[:32]}"


def _json_roundtrip(event: Mapping[str, Any], index: int) -> dict[str, Any]:
    """Deep-copy an event through JSON so the body carries no live references.

    Rejects non-serializable and non-finite (NaN/Infinity) values as a typed
    local contract error rather than letting them reach the wire. The
    serialization exception is confined to its handler and NOT chained, so the
    raised error cannot carry the offending value.
    """
    try:
        return json.loads(json.dumps(event, allow_nan=False))  # type: ignore[no-any-return]
    except (TypeError, ValueError):
        pass
    raise EconomicsTelemetryContractError(
        f"event {index} is not JSON-serializable or contains a non-finite value"
    )


def _require_non_empty(value: str, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise EconomicsTelemetryContractError(f"{field} must be a non-empty string")
    return value


__all__ = [
    "build_source",
    "build_telemetry_request",
    "canonical_json",
    "canonical_json_bytes",
    "funnel_eligible_event",
    "utc_now_z",
]
