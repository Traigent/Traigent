"""Honest, fail-closed result types for economics telemetry ingestion (WI-B).

The backend today fails closed for every path lacking an authoritative producer
(advice-bearing funnel stages, run settlements, receipts), accepting only the
project-level ``eligible`` funnel entry. So the result MUST NOT flatten a
rejection into a boolean success, and MUST NOT coerce a malformed response into
an apparently-accepted batch: the parser validates the response's identity,
version, status/flag agreement, count invariants, and rejection shape before it
will hand back a result, and raises otherwise. The response body is never logged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from traigent.economics.contract import (
    CONTRACT_ID,
    CONTRACT_VERSION,
    REJECTION_REASONS,
)
from traigent.economics.errors import EconomicsResponseError
from traigent.economics.schema import validate_response_or_fail

# HTTP status -> the value the contract binds `replayed` to on that path.
_STATUS_REPLAYED = {200: True, 201: False, 422: False}


@dataclass(frozen=True)
class Rejection:
    """One rejected event, with the closed backend reason code.

    The backend's advisory ``detail`` string is validated for shape on the wire
    but then DISCARDED — it is deliberately NOT a field of this public type. Per
    the response schema ``detail`` is ``x-privacy-classification: user_content``
    and its no-payload-echo rule is prose-only (not machine-enforced), so a
    ``detail`` that quoted a value the client's characterization sharing policy
    withheld would hand that value straight back through the public result, its
    ``repr``, or logs. Only the machine-readable identifiers survive here. The
    accepted JS SDK likewise drops ``detail`` from its public rejection type.
    """

    event_index: int
    reason: str
    event_id: str | None = None


@dataclass(frozen=True)
class TelemetryIngestResult:
    """The parsed, validated outcome of one ingest call.

    Distinguishes a fresh ingest from a replay, and — within acceptance —
    a fresh write from an already-stored duplicate, so no caller mistakes an
    all-duplicate replay for new state.
    """

    http_status: int
    replayed: bool
    batch_id: str
    idempotency_key: str
    contract: str
    contract_version: str
    received_at: str
    submitted: int
    accepted: int
    duplicate: int
    rejected: int
    rejections: tuple[Rejection, ...]

    @property
    def no_rejections(self) -> bool:
        """True when nothing was rejected (says nothing about fresh vs duplicate)."""
        return self.rejected == 0

    @property
    def fully_accepted(self) -> bool:
        """True only when every submitted event was a FRESH write (no dup/reject).

        An all-duplicate batch is deliberately NOT fully_accepted: it wrote no
        new state. Use :attr:`no_rejections` for 'nothing was rejected'.
        """
        return self.submitted > 0 and self.accepted == self.submitted

    @property
    def any_rejected(self) -> bool:
        return self.rejected > 0

    @property
    def all_rejected(self) -> bool:
        """True when the batch was non-empty and every event was rejected (422)."""
        return self.submitted > 0 and self.rejected == self.submitted

    @property
    def all_duplicate(self) -> bool:
        """True when the batch was non-empty and every event was already stored."""
        return self.submitted > 0 and self.duplicate == self.submitted

    @property
    def rejection_reasons(self) -> tuple[str, ...]:
        """The closed reason codes for the rejected events, in event order."""
        return tuple(r.reason for r in self.rejections)

    @classmethod
    def from_response(
        cls,
        *,
        http_status: int,
        body: Any,
        expected_idempotency_key: str,
        expected_batch_id: str,
        expected_submitted: int,
        expected_event_ids: tuple[str, ...],
    ) -> TelemetryIngestResult:
        """Parse and VALIDATE an ingest-response body (200 / 201 / 422).

        The body is first validated against the exact per-status economics
        response schema (shape, unknown keys, timestamps, closed reasons), then
        semantically reconciled against the request (identity, counts, and — per
        rejection — a unique index and an event_id that matches the request event
        at that index). The body is never included in an error message.

        Raises:
            EconomicsResponseError: On any schema or reconciliation violation.
            EconomicsSchemaUnavailable: If the exact response schema is
                unavailable/mismatched (fail closed).
        """
        if http_status not in _STATUS_REPLAYED:
            raise EconomicsResponseError(
                f"unexpected ingest status {http_status} for a result body"
            )
        # Schema-validate the exact response shape BEFORE any hand parsing.
        validate_response_or_fail(body, http_status=http_status)
        if not isinstance(body, dict):
            raise EconomicsResponseError("ingest response was not a JSON object")

        if body.get("contract") != CONTRACT_ID:
            raise EconomicsResponseError("ingest response names a different contract")
        if body.get("contract_version") != CONTRACT_VERSION:
            raise EconomicsResponseError(
                "ingest response names a different contract version"
            )

        replayed = body.get("replayed")
        if not isinstance(replayed, bool):
            raise EconomicsResponseError(
                "ingest response replayed flag is missing or non-boolean"
            )
        if replayed != _STATUS_REPLAYED[http_status]:
            # A 200 that says replayed=false (or a 201/422 that says true) would
            # lie about whether state was written.
            raise EconomicsResponseError(
                "ingest response status and replayed flag disagree"
            )

        batch_id = body.get("batch_id")
        key = body.get("idempotency_key")
        if batch_id != expected_batch_id:
            raise EconomicsResponseError(
                "ingest response batch_id does not echo the request"
            )
        if key != expected_idempotency_key:
            raise EconomicsResponseError(
                "ingest response idempotency_key does not echo the request"
            )

        received_at = body.get("received_at")
        if not isinstance(received_at, str) or not received_at:
            raise EconomicsResponseError("ingest response received_at is missing")

        submitted, accepted, duplicate, rejected = _parse_counts(body.get("counts"))
        if submitted != expected_submitted:
            raise EconomicsResponseError(
                "ingest response submitted count does not match the batch"
            )
        if accepted + duplicate + rejected != submitted:
            raise EconomicsResponseError("ingest response counts do not reconcile")
        if http_status == 422 and not (submitted > 0 and rejected == submitted):
            raise EconomicsResponseError("422 response is not an all-rejected batch")
        # All-rejected is the 422 status contract, never a fresh 201. (A 200 may
        # legitimately be all-rejected: it replays a prior 422 ingest.)
        if http_status == 201 and submitted > 0 and rejected == submitted:
            raise EconomicsResponseError(
                "all-rejected batch must be reported as 422, not a fresh 201"
            )

        rejections = _parse_rejections(
            body.get("rejections"),
            submitted=submitted,
            rejected=rejected,
            expected_event_ids=expected_event_ids,
        )

        return cls(
            http_status=http_status,
            replayed=replayed,
            batch_id=str(batch_id),
            idempotency_key=str(key),
            contract=CONTRACT_ID,
            contract_version=CONTRACT_VERSION,
            received_at=received_at,
            submitted=submitted,
            accepted=accepted,
            duplicate=duplicate,
            rejected=rejected,
            rejections=rejections,
        )


def _parse_counts(counts: Any) -> tuple[int, int, int, int]:
    if not isinstance(counts, dict):
        raise EconomicsResponseError("ingest response counts are missing")
    values: list[int] = []
    for name in ("submitted", "accepted", "duplicate", "rejected"):
        value = counts.get(name)
        # bool is an int subclass; reject it so a True count is not read as 1.
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise EconomicsResponseError(f"ingest response count '{name}' is invalid")
        values.append(value)
    return values[0], values[1], values[2], values[3]


def _parse_rejections(
    raw: Any, *, submitted: int, rejected: int, expected_event_ids: tuple[str, ...]
) -> tuple[Rejection, ...]:
    if not isinstance(raw, list):
        raise EconomicsResponseError(
            "ingest response rejections is missing or not a list"
        )
    if len(raw) != rejected:
        raise EconomicsResponseError(
            "ingest response rejections length does not match count"
        )

    parsed: list[Rejection] = []
    seen_indices: set[int] = set()
    for item in raw:
        if not isinstance(item, dict):
            raise EconomicsResponseError("ingest response rejection is not an object")
        index = item.get("event_index")
        if (
            not isinstance(index, int)
            or isinstance(index, bool)
            or not (0 <= index < submitted)
        ):
            raise EconomicsResponseError(
                "ingest response rejection event_index is out of range"
            )
        # A stage may reject an event once; two rejections for one index is a
        # malformed disposition, not two failures of the same event.
        if index in seen_indices:
            raise EconomicsResponseError(
                "ingest response has duplicate rejection event_index"
            )
        seen_indices.add(index)
        reason = item.get("reason")
        if reason not in REJECTION_REASONS:
            raise EconomicsResponseError(
                "ingest response rejection names an unknown reason"
            )
        event_id = item.get("event_id")
        if event_id is not None and not isinstance(event_id, str):
            raise EconomicsResponseError(
                "ingest response rejection event_id is not a string"
            )
        # When the backend echoes the rejected event_id, it must be the event WE
        # submitted at that index — otherwise the rejection addresses something
        # other than our batch.
        if (
            event_id is not None
            and index < len(expected_event_ids)
            and event_id != expected_event_ids[index]
        ):
            raise EconomicsResponseError(
                "ingest response rejection event_id does not match the request "
                "event at that index"
            )
        # Validate `detail`'s shape exactly as before (a non-string still fails
        # closed), then DISCARD it — see the Rejection docstring. It is never
        # copied into the public result, so no user_content value can leak through
        # a field, repr, or log line downstream.
        detail = item.get("detail")
        if detail is not None and not isinstance(detail, str):
            raise EconomicsResponseError(
                "ingest response rejection detail is not a string"
            )
        parsed.append(Rejection(event_index=index, reason=reason, event_id=event_id))
    return tuple(parsed)


__all__ = ["Rejection", "TelemetryIngestResult"]
