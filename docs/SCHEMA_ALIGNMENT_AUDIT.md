# Schema Alignment Audit

This document compares the key JS SDK DTOs and optimization-facing entities in
this checkout against the canonical schema definitions in:

- [TraigentSchema/traigent_schema/schemas](../../TraigentSchema/traigent_schema/schemas)

The goal is to distinguish:

- true schema alignment
- intentional SDK-level divergences
- deferred platform/schema features that this native-first JS checkout does not
  claim to implement

## Summary

- Metric-value dictionaries in this SDK are aligned with the canonical schema
  contracts.
- Trial/runtime DTOs are intentionally SDK-internal transport objects, not
  1:1 platform persistence entities.
- The JS `OptimizationSpec` is intentionally an ergonomic SDK authoring API,
  not a direct in-memory copy of the canonical TVL `config_space_schema.json`.
- One real mismatch was corrected in this pass: metric values now require
  finite numbers, which matches JSON-schema-compatible numeric transport.

## Exact Alignment

### Metric value maps

Canonical references:

- [execution/metric_submission_schema.json](../../TraigentSchema/traigent_schema/schemas/execution/metric_submission_schema.json)
- [evaluation/configuration_run_schema.json](../../TraigentSchema/traigent_schema/schemas/evaluation/configuration_run_schema.json)

Local JS DTOs:

- [src/dtos/measures.ts](../src/dtos/measures.ts)
- [src/dtos/trial.ts](../src/dtos/trial.ts)

Aligned behavior:

- keys must match `^[a-zA-Z_][a-zA-Z0-9_]*$`
- maximum 50 keys
- values must be numeric or `null`
- values are now restricted to finite JSON-compatible numbers

Important nuance:

- This aligns with metric-value dictionaries, not with
  [measures/measure_schema.json](../../TraigentSchema/traigent_schema/schemas/measures/measure_schema.json),
  which defines measure catalog metadata, not runtime metric results.

## Intentional Divergences

### TrialConfig / TrialResultPayload are SDK runtime contracts

Local JS DTOs:

- [src/dtos/trial.ts](../src/dtos/trial.ts)

Closest canonical schemas:

- [execution/hybrid_session_schema.json](../../TraigentSchema/traigent_schema/schemas/execution/hybrid_session_schema.json)
- [execution/metric_submission_schema.json](../../TraigentSchema/traigent_schema/schemas/execution/metric_submission_schema.json)
- [evaluation/configuration_run_schema.json](../../TraigentSchema/traigent_schema/schemas/evaluation/configuration_run_schema.json)

Why this is intentionally different:

- `TrialConfig` is the local SDK execution context for one trial, not a
  persisted backend entity.
- `TrialResultPayload` is the SDK bridge/runtime result envelope, not the same
  thing as persisted metric submission or configuration run records.
- The schema repo models storage/API entities; the SDK models a runtime
  execution handshake.

This divergence is justified.

### `OptimizationSpec` is an SDK authoring surface, not canonical TVL config space

Local types:

- [src/optimization/types.ts](../src/optimization/types.ts)

Canonical schemas:

- [optimization/config_space_schema.json](../../TraigentSchema/traigent_schema/schemas/optimization/config_space_schema.json)
- [optimization/tvar_definition_schema.json](../../TraigentSchema/traigent_schema/schemas/optimization/tvar_definition_schema.json)
- [optimization/objective_definition_schema.json](../../TraigentSchema/traigent_schema/schemas/optimization/objective_definition_schema.json)
- [optimization/typed_constraints_schema.json](../../TraigentSchema/traigent_schema/schemas/optimization/typed_constraints_schema.json)

Intentional differences:

- JS uses `configurationSpace: Record<string, ParameterDefinition>` instead of
  a full `config_space` entity with `id`, `schema_version`, `tunable_id`,
  `metadata`, and timestamps.
- JS objective objects use `{ metric, direction, weight? }`, while canonical
  objective schema uses `{ name, direction, weight?, unit?, band? }`.
- JS native parameter definitions in this checkout only support `enum`, `int`,
  and `float`. Canonical TVAR schema also includes `bool`, `str`, `agent`,
  `is_tool`, and per-TVAR textual constraints.
- JS native constraints are callback predicates, not canonical typed TVL
  structural/derived constraint objects.
- JS native runtime options such as `trialConcurrency`, `plateau`,
  `checkpoint`, `timeoutMs`, `signal`, and Bayesian settings are optimizer
  runtime controls, not a 1:1 implementation of
  `exploration_config_schema.json`.

This divergence is also justified. The JS surface is an ergonomic SDK API for
authoring and executing native optimization, not a persistence schema.

### `HybridConfigSpace` is a legacy wire adapter

Local type:

- [src/optimization/types.ts](../src/optimization/types.ts)

Related canonical schema:

- [optimization/config_space_schema.json](../../TraigentSchema/traigent_schema/schemas/optimization/config_space_schema.json)

Why it differs:

- `HybridConfigSpace` exists only to serialize the older hybrid wire shape used
  by legacy routes in this checkout.
- It is intentionally not the canonical TVL config-space entity.

This divergence is expected and already documented elsewhere in the repo.

## Deferred Canonical Features

These are real schema/platform features that are not fully implemented in this
native-first JS checkout:

- full canonical typed TVL constraints schema fidelity
  - JS now supports structural and derived TVL constraints, but compiles a
    supported subset into callback predicates instead of materializing the raw
    canonical schema objects end-to-end
- full canonical TVL/entity fidelity for `band` and `promotion_policy`
  - JS now supports banded objectives and promotion-policy semantics in native
    execution, but still exposes an ergonomic SDK spec instead of a 1:1
    canonical persistence model
- free-form canonical TVAR shapes such as `str`, `agent`, and schema-level
  textual constraint metadata
- hybrid session persistence/API entities as first-class JS runtime types
- canonical config-space entity ids/metadata/timestamps

These are deferred, not accidental mismatches.

## Conclusion

- The value-level DTOs that need to align with schema-governed runtime payloads
  now do align.
- The bigger optimization entities diverge from the schema repo by design
  because this SDK exposes an ergonomic authoring/runtime API instead of raw
  persistence entities.
- Those divergences are justified, but they should stay documented so the SDK
  does not accidentally present itself as a 1:1 TVL/config-space schema model.
