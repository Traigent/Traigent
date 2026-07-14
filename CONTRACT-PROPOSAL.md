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

## Proposed versioned, idempotent health snapshot

Fold this into the ingest-v2 (T4) protocol work; do not ship it as a standalone
additive field on the current ingest contract. The object is a **versioned,
idempotent health snapshot**, not an increment operation or an ingest
acknowledgement.

The T4 schema and backend model should add this optional member:

```json
{
  "client_stats": {
    "schema_version": 1,
    "client_instance_id": "a2f2f5c0-6f7c-4d3b-a750-3d5b7c4c4f7d",
    "snapshot_seq": 42,
    "observed_at": "2026-07-12T12:34:56.789Z",
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

`schema_version`, `client_instance_id`, `snapshot_seq`, and `observed_at` are
required when the object is present. `schema_version` initially has the sole
allowed value `1`; `client_instance_id` is a UUID generated once per client
instance; `snapshot_seq` is a strictly monotonic, non-negative integer within
that client instance; and `observed_at` is the UTC time at which that snapshot
was read. `sdk_version` is a non-empty string with a 128-character maximum.
All counts are non-negative integers. `dropped_by_reason` is an object with at
most 20 keys; each key is a lower-case ASCII reason identifier
(`^[a-z][a-z0-9_]{0,63}$`) and each value is a non-negative integer.

`client_stats` is a cumulative point-in-time client-process health snapshot
taken immediately before the ingest payload is serialized. It contains no trace
content, tenant data, or exception messages. A newly constructed client gets a
new `client_instance_id`, which explicitly starts a new lineage; counters may
therefore reset only across client-instance lineages, never within one.

Backends must deduplicate and aggregate by snapshot identity: apply
immutable identity rules for each `(client_instance_id, snapshot_seq)`. A retry
with the same identity and the same content hash is an idempotent no-op. A
submission with the same identity and a different content hash is an integrity
conflict: the backend must reject it and alert, and must never overwrite the
stored snapshot. A sequence lower than the stored maximum for its
`client_instance_id` is an out-of-order no-op. Never sum cumulative snapshot
counters across retries or successive snapshots. Cross-client reporting must
derive values from the latest accepted snapshot per client lineage (or a
separately defined time-series calculation), not from request-level addition.
The object is not an acknowledgement of backend persistence and must not change
ingest response semantics.

## Coordinated rollout

1. During T4 ingest-v2 design, add a strict `ClientStats` Pydantic model and
   optional `client_stats: ClientStats | None = None` to the new request model;
   keep the request model's `extra="forbid"` setting.
2. Add the matching versioned object to TraigentSchema with
   `additionalProperties: false`, its size bounds, and acceptance/rejection
   contract tests, including duplicate and out-of-order snapshot handling.
3. Add backend ingest tests proving retries do not overcount, immutable snapshot
   identity semantics hold (including conflict rejection and alerting), and trace
   persistence is unchanged. Decide separately whether to persist latest
   snapshots, emit derived metrics, or both, with retention/cardinality limits.
4. Only after the coordinated T4 backend and schema release is deployed, make
   the SDK include this capped object in outgoing batches. Preserve SDK-side
   `get_stats()` and `health_callback` as the immediate local observability
   path for older backends.
