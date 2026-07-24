"""Frozen constants for the economics telemetry contract (WI-B).

These mirror the authoritative closed contract in ``TraigentSchema``
(``traigent_schema/schemas/economics/*``). They are declared here — rather than
imported from the schema package — because ``traigent-schema`` is an OPTIONAL
runtime dependency of the SDK, so the emitter cannot assume it is installed.

To keep this from becoming a competing contract, a drift test binds every value
below to the authoritative local Schema worktree (see the economics contract
tests); a schema change that these constants do not track fails that test rather
than silently shipping a divergent client contract.

Nothing here carries or logs a payload value: these are contract identifiers,
closed vocabularies, and header/route names only.
"""

from __future__ import annotations

# --- Route + contract identity -------------------------------------------------

#: POST route that ingests one idempotent batch of economics telemetry.
TELEMETRY_ENDPOINT = "/api/v1/economics/telemetry"

#: Stable contract family id. A payload not naming this contract is rejected.
CONTRACT_ID = "economics_telemetry"

#: Stable contract version. Ingestion rejects unknown versions; a breaking
#: change ships a new const value here and in the schema together.
CONTRACT_VERSION = "1.0.0"

#: The surface kind this SDK emits as (SourceKind enum member).
SOURCE_KIND = "python_sdk"

#: Presentation name/version for the ``source`` block. ShortLabel-shaped.
SOURCE_NAME = "traigent-python-sdk"

# --- Transport headers ---------------------------------------------------------

#: Idempotency-Key header name (Planner V2 convention). The value also travels
#: in the body ``idempotency_key`` and the two MUST match.
IDEMPOTENCY_KEY_HEADER = "Idempotency-Key"

#: Project-scoping header the backend resolves the project from. The contract
#: carries NO tenant/project field precisely so a client cannot assert one.
PROJECT_ID_HEADER = "X-Project-Id"

# --- Batch bounds --------------------------------------------------------------

#: Max events per batch. Larger emissions page through multiple idempotent
#: batches; the backend enforces this authoritatively (413).
MAX_BATCH_EVENTS = 500

#: Idempotency key grammar (IdempotencyKey definition).
IDEMPOTENCY_KEY_PATTERN = r"^[A-Za-z0-9._:-]{8,128}$"

# --- Closed vocabularies (mirror the schema enums / consts) --------------------

EVENT_TYPES = frozenset({"funnel_event", "run_economics", "receipt"})

FUNNEL_STAGES = (
    "eligible",
    "advice_shown",
    "budget_allocated",
    "run_started",
    "completed",
    "recommendation_accepted",
    "executed",
    "promoted",
    "production_retained",
)

FUNNEL_OUTCOMES = frozenset({"entered", "exited"})

FUNNEL_ENVIRONMENTS = frozenset({"development", "staging", "production"})

#: The allowlist of characterization field names that may EVER be named in
#: telemetry (CharacterizationFieldName). Egress enforcement rejects anything
#: outside this set.
CHARACTERIZATION_FIELD_NAMES = frozenset(
    {
        "value_channel",
        "daily_volume_band",
        "error_cost_band",
        "lifecycle_stage",
        "human_cycle_hours_band",
        "value_per_task_usd",
        "loss_per_bad_output_usd",
        "observed_daily_volume",
        "forecast_daily_volume",
        "human_minutes_per_example",
    }
)

PROVENANCE_VALUES = frozenset({"asked", "inferred", "defaulted"})

SHARING_OUTCOMES = frozenset({"shared", "withheld_by_policy"})

EVIDENCE_STATUSES = frozenset({"provided", "withheld_by_policy"})

#: Closed rejection reason codes the ingest response may carry per event.
REJECTION_REASONS = frozenset(
    {
        "schema_violation",
        "unknown_event_type",
        "unsupported_contract_version",
        "duplicate_event_id",
        "tenant_scope_violation",
        "unknown_reference",
        "funnel_order_violation",
        "receipt_verification_failed",
        "attestation_not_independent",
        "meter_reconciliation_failed",
        "budget_authorship_mismatch",
        "batch_limit_exceeded",
        "duplicate_characterization_field",
        "interval_bounds_inconsistent",
        "support_counts_inconsistent",
        "withheld_field_value_present",
        "winner_receipt_reconciliation_failed",
    }
)


__all__ = [
    "CHARACTERIZATION_FIELD_NAMES",
    "CONTRACT_ID",
    "CONTRACT_VERSION",
    "EVENT_TYPES",
    "EVIDENCE_STATUSES",
    "FUNNEL_ENVIRONMENTS",
    "FUNNEL_OUTCOMES",
    "FUNNEL_STAGES",
    "IDEMPOTENCY_KEY_HEADER",
    "IDEMPOTENCY_KEY_PATTERN",
    "MAX_BATCH_EVENTS",
    "PROJECT_ID_HEADER",
    "PROVENANCE_VALUES",
    "REJECTION_REASONS",
    "SHARING_OUTCOMES",
    "SOURCE_KIND",
    "SOURCE_NAME",
    "TELEMETRY_ENDPOINT",
]
