# Hybrid Config-Space Token Estimate Rollout

Last updated: 2026-03-13

## Purpose

`estimated_tokens_per_example` is an optional field on `GET /traigent/v1/config-space`
for OpenAPI-based `hybrid_api` integrations.

It is strongly recommended because it materially improves the SDK's pre-run cost
approval estimate. In the validation scenario used for `#246`, the same workload
estimate dropped from `$12.96` to `$0.14` once the SDK had both:

- the candidate model set from config-space
- `estimated_tokens_per_example` from the client service

That is roughly a `90x` improvement in estimate quality.

## Contract Snippet

Minimal JSON response fragment:

```json
{
  "estimated_tokens_per_example": {
    "input_tokens": 100,
    "output_tokens": 50
  }
}
```

Minimal YAML schema fragment:

```yaml
estimated_tokens_per_example:
  type: object
  required: [input_tokens, output_tokens]
  properties:
    input_tokens:
      type: integer
      minimum: 0
    output_tokens:
      type: integer
      minimum: 0
```

## Adoption Guidance

- This field is optional. Existing client services continue to work without it.
- If omitted, the SDK falls back to conservative token assumptions and may
  substantially overestimate pre-approval cost.
- If present, the SDK can make much tighter pre-run approval checks in
  `hybrid_api` mode.
- Implementers should return a stable, representative estimate for one example
  under normal service operation. It does not need to be exact per request.

## First-Party Status

| Implementer | Endpoint ownership | Status | Notes |
| --- | --- | --- | --- |
| `Traigent/examples/experimental/hybrid_api_demo` | Demo Flask server | Updated locally | Returns `estimated_tokens_per_example = {100, 50}`. |
| `JS-Mastra-APIs-Validation` | First-party validation server | Updated locally | `getConfigSpace()` now returns the field for both tunables. |
| `BazakDemo-Apis-Validation` | First-party validation server | Updated locally | `getConfigSpace()` now returns the field. |
| `Traigent/traigent/wrapper` | SDK-hosted wrapper server | Not updated | Wrapper exposes `/traigent/v1/*`, but generic wrapper code cannot infer a representative token estimate yet. Treat as follow-up if wrapper deployments are part of release scope. |
| `traigent-api` | Contract repo only | Updated locally | OpenAPI schema and README include the optional field. |
| `TraigentBackend` | Not an OpenAPI client implementer in this workspace | No change needed | Search found no `/traigent/v1/config-space`, `/execute`, or `/evaluate` routes. |

## Remaining Work

- Push and merge the SDK changes in `Traigent`.
- Push and merge the contract change in `traigent-api`.
- Push and merge first-party adopter changes in:
  - `JS-Mastra-APIs-Validation`
  - `BazakDemo-Apis-Validation`
- Ask external client implementers to add the field to their
  `GET /traigent/v1/config-space` responses.
