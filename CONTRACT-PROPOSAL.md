# Observability ingest `client_stats` proposal

## Status

Not sent by this SDK change. The current contract rejects unknown top-level
fields, so adding `client_stats` to an ingest request today would return `422`
rather than being safely ignored.

Evidence checked on 2026-07-12:

- `TraigentBackend/src/shared_infrastructure/schemas/observability.py` defines
  `ObservabilityIngestRequest` with `model_config = ConfigDict(extra="forbid")`.
- `TraigentSchema/traigent_schema/schemas/observability/observability_ingest_request_schema.json`
  sets `additionalProperties: false` and only permits `traces` and `source`.

## Proposed additive field

Add this optional top-level member to the schema and backend Pydantic model:

```json
{
  "client_stats": {
    "schema_version": 1,
    "sdk_version": "0.21.3",
    "dropped_items": 3,
    "dropped_by_reason": {
      "queue_full": 2,
      "payload_too_large": 1
    },
    "queue_depth": 12,
    "inflight_items": 1,
    "retry_attempts": 4
  }
}
```

All counts are non-negative integers. `schema_version` is required when the
object is present and initially has the sole allowed value `1`. `sdk_version`
is a non-empty string with a 128-character maximum. `dropped_by_reason` is an
object with at most 20 keys; each key is a lower-case ASCII reason identifier
(`^[a-z][a-z0-9_]{0,63}$`) and each value is a non-negative integer.

`client_stats` is a point-in-time, client-process snapshot taken immediately
before the ingest payload is serialized. It contains no trace content,
identifiers, tenant data, or exception messages. It is not an acknowledgement
of backend persistence, and it must not change the ingest response semantics.

## Coordinated rollout

1. Add a strict `ClientStats` Pydantic model and optional
   `client_stats: ClientStats | None = None` to `ObservabilityIngestRequest` in
   TraigentBackend; keep the request model's `extra="forbid"` setting.
2. Add the matching optional object to TraigentSchema with
   `additionalProperties: false`, its size bounds, and acceptance/rejection
   contract tests.
3. Add a backend ingest test proving the field is accepted but does not alter
   trace persistence. Decide separately whether to persist aggregates, emit
   metrics, or both, with retention/cardinality limits.
4. Only after the coordinated backend and schema release is deployed, make the
   SDK include this capped object in outgoing batches. Preserve SDK-side
   `get_stats()` and `health_callback` as the immediate local observability
   path for older backends.
