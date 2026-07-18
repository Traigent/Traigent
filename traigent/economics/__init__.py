"""Economics telemetry emission (WI-B).

The smallest safe public surface for emitting one idempotent batch of economics
telemetry to ``POST /api/v1/economics/telemetry``. It constructs the authoritative
closed-contract request, enforces the client-side characterization egress rules
before transmission, validates every batch against the exact economics Schema
(fail closed when it is unavailable), sends a caller-stable ``Idempotency-Key``
that matches the body, and exposes the backend's acceptance/rejection outcome
honestly.

Scope (WI-B): funnel/run/receipt telemetry EMISSION only. Survey submission, the
calculator/recommendations, preflight economics, credits, and pricing behavior
are out of scope and are not implemented here. The backend accepts only the
project-level ``eligible`` funnel entry today and fails closed for stages and
receipts lacking an authoritative producer; this emitter can represent the full
contract but never disguises a backend rejection as success.

Public-surface policy — ROOT LAZY EXPORT + subpackage. ``EconomicsTelemetryClient``
is exported both as ``traigent.EconomicsTelemetryClient`` (a lazy root export, so
plain ``import traigent`` does not eagerly import this subpackage) and as
``traigent.economics.EconomicsTelemetryClient``; both names resolve to the same
class. Root symbols are governed by the schema-owned Python/JS parity manifest
(``TraigentSchema parity/python-js-sdk.json``, pinned to a target SHA), which now
classifies ``EconomicsTelemetryClient`` as a ``matched`` root export, and the JS
SDK root-exports the same symbol from ``@traigent/sdk`` (``traigent-js``
``src/index.ts``) — so the Python root export is true parity, not a unilateral
add. The subpackage import remains supported for callers that prefer it.
"""

from __future__ import annotations

from traigent.economics.client import (
    EconomicsTelemetryClient,
    PreparedTelemetryBatch,
)
from traigent.economics.contract import (
    CONTRACT_ID,
    CONTRACT_VERSION,
    TELEMETRY_ENDPOINT,
)
from traigent.economics.egress import enforce_characterization_egress
from traigent.economics.errors import (
    EconomicsBatchTooLarge,
    EconomicsIdempotencyConflict,
    EconomicsResponseError,
    EconomicsSchemaUnavailable,
    EconomicsTelemetryAuthError,
    EconomicsTelemetryContractError,
    EconomicsTelemetryError,
    EconomicsTelemetryTransportError,
    EgressPolicyError,
)
from traigent.economics.payload import (
    build_telemetry_request,
    funnel_eligible_event,
)
from traigent.economics.result import Rejection, TelemetryIngestResult

__all__ = [
    "CONTRACT_ID",
    "CONTRACT_VERSION",
    "TELEMETRY_ENDPOINT",
    "EconomicsBatchTooLarge",
    "EconomicsIdempotencyConflict",
    "EconomicsResponseError",
    "EconomicsSchemaUnavailable",
    "EconomicsTelemetryAuthError",
    "EconomicsTelemetryClient",
    "EconomicsTelemetryContractError",
    "EconomicsTelemetryError",
    "EconomicsTelemetryTransportError",
    "EgressPolicyError",
    "PreparedTelemetryBatch",
    "Rejection",
    "TelemetryIngestResult",
    "build_telemetry_request",
    "enforce_characterization_egress",
    "funnel_eligible_event",
]
