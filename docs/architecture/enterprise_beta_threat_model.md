# Enterprise Beta Threat Model

Date: 2026-03-13

Related documents:
- [langfuse_target_architecture.md](./langfuse_target_architecture.md)
- [cross_repo_delivery_template.md](./cross_repo_delivery_template.md)
- [../feature_matrices/enterprise_beta_execution_plan.md](../feature_matrices/enterprise_beta_execution_plan.md)
- [../feature_matrices/enterprise_beta_execution_tracker.md](../feature_matrices/enterprise_beta_execution_tracker.md)

## Scope

This is the early lightweight threat-model pass required after Wave 3A
stabilization. It focuses on the beta-critical surfaces:

- tenant/project enforcement on core resources
- project-level RBAC on protected mutations
- export and raw-content policy transitions
- legacy route exposure
- seeded validation and automated release verification

It is not a full SOC 2 or ISO 27001 control catalog. It is the architecture and
abuse-case review that should happen before beta hardening becomes schedule
critical.

## Assets and Trust Boundaries

### Protected assets

- core project-scoped resources:
  - agents
  - benchmarks
  - measures
  - experiments
  - experiment runs
  - configuration runs
- project-scoped observability data
- evaluator results and score records
- export artifacts and their metadata
- project memberships and role assignments
- API keys and user authentication state

### Trust boundaries

1. **Authentication boundary**
   - browser JWT/cookie users
   - API key callers
   - platform-admin/internal tooling
2. **Context resolution boundary**
   - tenant resolution
   - project resolution
3. **Authorization boundary**
   - project roles
   - tenant-admin override
4. **Data-access boundary**
   - DALs, services, and routes that touch tenant/project-scoped models
5. **Export boundary**
   - privacy-safe manifest paths
   - materialized/raw-content export paths

## Security Objectives

- no cross-tenant or cross-project data access on beta-critical paths
- no public API key path may act as a cross-tenant super-admin
- protected mutations must be backend-authorized, not frontend-authorized
- export paths must default to privacy-safe behavior
- any legacy route that bypasses the new enforcement kernel must be disabled
- failures in export retention or cleanup must not silently preserve artifacts forever

## Main Threat Scenarios

### 1. Cross-tenant read via unscoped route or DAL

Risk:
- a request path reads a core resource by primary key or broad query without
  tenant/project filtering

Current mitigations:
- live single-tenant bypasses removed from the v2 resource access checker
- core DAL hardening for `ExperimentRun`, `ConfigurationRun`, `Agent`, and related
  clone/helper paths
- release-review guard tests prevent reintroducing common unscoped
  `db.session.get(...)` patterns on tenant/project-bound models
- seeded backend + Playwright validation now verifies positive and negative
  cross-tenant/cross-project behavior against the live local worktree stack

Residual risk:
- transitive-scope assumptions can still reappear in new list/read code if not
  caught by review or tests

Required release action:
- complete remaining route/service audit
- keep cross-scope seeded validation in the release gate

### 2. Cross-tenant write via unscoped update/delete path

Risk:
- a caller updates or deletes a scoped resource by ID without tenant/project
  validation

Current mitigations:
- `ConfigurationRunDAL` mutators now use scoped lookup
- project-role enforcement now guards multiple protected mutation surfaces

Residual risk:
- new write paths may regress if they bypass scoped DAL helpers

Required release action:
- keep protected-mutation authz tests mandatory for every new route

### 3. Admin API key bypass of tenant/project scope

Risk:
- “admin” API keys act as super-admin and bypass tenant/project checks

Current mitigations:
- beta decision is explicit: public admin keys are tenant-scoped
- v2 resource access checker resolves scope before permission grant

Residual risk:
- internal or legacy admin-key code paths could drift if they do not share the
  same resolver

Required release action:
- keep “no public API key bypass” as a release gate

### 4. Legacy route bypass

Risk:
- an old `/api/v1/...` route still touches core resources without going through
  the tenant/project enforcement kernel

Current mitigations:
- route inventory and safe/unsafe classification are part of Wave 3A
- non-operational legacy session compatibility route is hard-disabled

Residual risk:
- route coverage can drift as new compatibility shims are added

Required release action:
- no unsafe legacy route may remain enabled before beta

### 5. Export raw-content policy bypass

Risk:
- a project that should allow only privacy-safe exports receives raw content

Current mitigations:
- project export policy exists and defaults to manifest/reference mode
- export contracts carry `privacy_classification`
- materialized export is explicitly policy-gated

Residual risk:
- future export or sink adapters could bypass the policy if implemented outside
  the shared service path

Required release action:
- all export paths must stay behind the shared project export policy check
- privacy-mode integration tests must cover all beta export surfaces

### 6. Expired artifacts remain accessible

Risk:
- export artifact metadata expires, but the underlying artifact remains readable

Current mitigations:
- export jobs carry `expires_at`
- retention metadata is enforced in the model and normalization paths
- artifact store is blob-oriented behind a dedicated abstraction

Residual risk:
- cleanup enforcement must still be verified under concurrent/idempotent runs

Required release action:
- retention-aware cleanup must be part of the release gate

### 7. Context confusion and stale authorization decisions

Risk:
- a cached or previously resolved authorization result is reused across tenant or
  project changes

Current mitigations:
- active tenant/project is resolved per request
- seeded validation explicitly switches projects and tenants in UI and verifies
  scoped rendering

Residual risk:
- discrepancy telemetry is not yet exhaustive on all list/read flows
- short-lived cache keys that do not include scope remain a review watch item

Required release action:
- continue Phase 1 telemetry expansion where scoped-vs-legacy behavior still
  differs

## Current Findings

### No new blocker found from this early review

The current architecture direction is acceptable for Enterprise Beta provided the
remaining release gates are satisfied.

### Open items that still matter before beta

1. Complete the remaining route/service audit so transitive-scope assumptions do
   not survive on core resources.
2. Finish cross-tenant/project monitoring and alerting.
3. Complete privacy-mode integration tests for all beta analytics/export
   surfaces.
4. Complete the final audit-event coverage set for membership/role/export/admin
   changes.
5. Keep changed-code coverage above `85%` as the export and release-hardening
   slices land.

## Release Position

This threat-model pass supports the current beta sequencing:

- finish Wave 3A completely
- keep Wave 3B governance hardening in place
- finish export durability and cleanup
- then run final release validation

The main conclusion is:

- **the enforcement model is now credible**
- **the remaining work is mostly completeness, observability, and release-hardening**
- **beta should not ship until the open release-gate items above are closed**
