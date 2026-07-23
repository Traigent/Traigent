"""Typed errors for the economics telemetry emitter (WI-B).

Every error here is safe to surface and safe to log: none carries a payload
value, an evidence pointer, raw survey data, or a secret. Egress violations name
the offending characterization FIELD (an allowlisted vocabulary term) and the
rule it broke — never the withheld value.
"""

from __future__ import annotations


class EconomicsTelemetryError(Exception):
    """Base class for all economics telemetry emitter errors."""


class EgressPolicyError(EconomicsTelemetryError):
    """A characterization payload violated the client-side egress contract.

    Raised BEFORE any bytes leave the machine, so a withheld value can never be
    transmitted: the batch is refused locally rather than sent and rejected. The
    message names the field and the rule, never the value.
    """


class EconomicsTelemetryContractError(EconomicsTelemetryError):
    """The request could not be constructed or was rejected as malformed (400).

    Covers client-side construction errors (bad batch shape, project mismatch,
    non-serializable / non-finite values), local schema-validation failure, and
    the backend's malformed-batch / unknown-contract 400.
    """


class EconomicsSchemaUnavailable(EconomicsTelemetryError):
    """The exact economics contract Schema is required but unusable.

    Raised — and the batch is NOT transmitted — when the authoritative
    ``traigent-schema`` economics contract is absent, too old to carry the
    economics schemas, or raises while validating. This surface fails closed on
    it rather than emitting an unvalidated payload: without the exact schema the
    client cannot prove the closed-pipe egress rules or that no arbitrary extra
    key leaves the machine.
    """


class EconomicsResponseError(EconomicsTelemetryError):
    """The backend response did not conform to the ingest response contract.

    Raised when the response fails identity/version/count/rejection-shape or
    status/replayed-flag invariants. A malformed response is never coerced into
    an apparently-accepted result, and its body is never logged.
    """


class EconomicsIdempotencyConflict(EconomicsTelemetryError):
    """The idempotency key was already used with a DIFFERENT body (409).

    Stored events are immutable; the key is never re-applied to a new body. A
    retry of the SAME built request replays instead of conflicting.
    """


class EconomicsBatchTooLarge(EconomicsTelemetryError):
    """The batch exceeded the contract's per-batch event limit (413)."""


class EconomicsTelemetryAuthError(EconomicsTelemetryError):
    """Authentication failed or permission was insufficient (401 / 403).

    Also raised LOCALLY, before any bytes leave the machine, when no API key or
    JWT is configured on the client: the SDK never sends an economics telemetry
    request without a credential attached (fail closed rather than emit an
    unauthenticated request for the backend to police).
    """


class EconomicsTelemetryTransportError(EconomicsTelemetryError):
    """A transport or server error prevented a definitive ingest result.

    Raised after retries are exhausted for transient failures (timeouts,
    connection errors, 5xx / 503). Because the emitter sends a stable
    Idempotency-Key, retrying is safe: a duplicate delivery replays.
    """


__all__ = [
    "EconomicsBatchTooLarge",
    "EconomicsIdempotencyConflict",
    "EconomicsResponseError",
    "EconomicsSchemaUnavailable",
    "EconomicsTelemetryAuthError",
    "EconomicsTelemetryContractError",
    "EconomicsTelemetryError",
    "EconomicsTelemetryTransportError",
    "EgressPolicyError",
]
