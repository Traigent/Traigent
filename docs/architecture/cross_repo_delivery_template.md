# Cross-Repo Delivery Template

Date: 2026-03-12

## Summary

Use this template for every cross-repo platform wave. It exists to keep
`TraigentSchema`, backend, SDKs, and frontend aligned.

## Required Delivery Order

1. `TraigentSchema`
2. `TraigentBackend`
3. `Traigent`
4. `TraigentFrontend`
5. `traigent-js` or `traigent-api` only if the wave genuinely requires them

## Minimum Requirements Per Wave

Every wave must declare:

- depends on
- introduces migrations
- backfill required
- customer-visible changes

Every wave must ship:

- schema contract tests
- backend authz and isolation tests
- privacy-mode tests
- audit-event tests for sensitive operations
- telemetry for adoption and failure paths

## Contract Rules

- `TraigentSchema` is the only public wire-contract source of truth
- local DTOs in backend, SDK, or frontend are adapters, not canonical definitions
- no frontend or service may invent its own analytics/export wire shape outside `TraigentSchema`

## Default NFRs

Measurement point:

- backend response boundary, excluding browser render time, on seeded standard project-size datasets in local or staging environments

Targets:

- list/detail endpoints: p95 under `500ms`
- dashboard summary/trend endpoints: p95 under `1.5s`
- export job creation: synchronous response under `500ms`
- high-cardinality endpoints must paginate

## Rollout Checklist

- compatibility path documented
- backfill strategy documented if migrations touch historical rows
- legacy-route or compatibility-adapter removal threshold defined if applicable
- privacy-mode behavior documented and test-covered
