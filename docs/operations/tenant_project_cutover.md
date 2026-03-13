# Tenant/Project Enforcement Cutover Notes

This document describes the beta cutover path from legacy single-tenant assumptions to backend-enforced tenant/project isolation.

## Scope

This cutover applies to:
- agents
- benchmarks
- measures
- experiments
- experiment runs
- configuration runs
- project-scoped analytics

It does **not** rely on frontend-only filtering. The backend enforcement kernel is the source of truth.

## Preconditions

Before cutover:
- Wave 3A enforcement code is merged
- dual-run discrepancy telemetry is active
- safe vs unsafe legacy routes are inventoried
- unsafe legacy routes are disabled
- seeded backend + Playwright validation passes on a live stack

## Migration-Chain Mitigation

The local/live risk came from legacy global `alembic_version` rows such as `v3.2.0.a`, while the beta backend expects a dedicated backend migration table and reconciliation flow.

Current mitigation components:
- backend migration bootstrap logic
- schema-anchor detection
- dedicated backend version table support
- reconciliation script for legacy DB state

Operational policy:
1. attempt reconciliation of the existing DB
2. validate startup against:
   - fresh DB
   - legacy DB with global `alembic_version`
   - partially migrated local DB
3. if reconciliation is not resolved quickly, switch release validation to a fresh disposable seeded DB and continue legacy repair in parallel

## Backfill Policy

Legacy rows are handled with a strict policy:
- deterministically mappable rows are assigned to a **default tenant + default project**
- backfill must be **idempotent**
- unmappable rows are **quarantined**
- runtime code is not allowed to infer or guess ownership

Quarantined rows must not be served until remediated.

## Dual-Run Validation

Before cutover, old and new enforcement paths run in parallel where supported and emit discrepancy telemetry.

Expected discrepancy classes to review:
- legacy owner allowed cross-project access
- legacy access checker allowed out-of-scope resource
- legacy list path returned broader results than scoped path

Wave 3A is considered stable only when:
- discrepancy telemetry is active
- seeded discrepancy cases are clean
- no unresolved discrepancy class remains
- soak period of 5 business days completes successfully

## Legacy Route Policy

Each surviving legacy route must be classified as:

- `safe`
  - already flows through the tenant/project enforcement kernel
- `unsafe`
  - bypasses enforcement or depends on legacy single-tenant behavior

Rules:
- safe legacy routes may remain temporarily
- unsafe legacy routes must be disabled before beta

## Rollback Procedure

Rollback is for emergency recovery only.

Steps:
1. re-enable the compatibility path temporarily
2. keep discrepancy logging active
3. validate that in-scope users regain expected access
4. continue root-cause analysis on the failed enforcement path

Rollback is not a substitute for fixing data-mapping or scope bugs.

## Seeded Validation

The beta gate requires an automated seeded validation harness that:
- seeds at least two tenants and two projects
- creates realistic:
  - experiments
  - experiment runs
  - configuration runs
  - measures/scores
  - observability traces
  - export jobs
- verifies:
  - in-scope resources are visible
  - out-of-scope resources return `404` or empty lists
  - dashboards remain scoped
  - UI views render only the active tenant/project data

The seeded harness must run in CI or staging without manual intervention.

## Monitoring and Alerting

Before beta onboarding:
- cross-tenant/project monitoring must be active
- audit alerts/health should surface:
  - scope violations
  - scope discrepancies

These are release gates, not optional observability improvements.

## Documentation and Review Gates

Before beta cut:
- API reference exists for dashboards, exports, roles, and policies
- privacy-mode behavior is documented
- lightweight threat-model review is completed
- findings are resolved or explicitly accepted

## Related Documents
- [enterprise_beta_execution_plan.md](/home/nimrodbu/Traigent_enterprise/Traigent/docs/feature_matrices/enterprise_beta_execution_plan.md)
- [enterprise_beta_execution_tracker.md](/home/nimrodbu/Traigent_enterprise/Traigent/docs/feature_matrices/enterprise_beta_execution_tracker.md)
- [enterprise_beta_threat_model.md](/home/nimrodbu/Traigent_enterprise/Traigent/docs/architecture/enterprise_beta_threat_model.md)
