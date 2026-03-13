# Enterprise Beta Execution Tracker

Last updated: 2026-03-13

## Purpose
Track the remaining Enterprise Beta work across the active repos while implementation continues:

- `Traigent`
- `TraigentBackend`
- `TraigentFrontend`
- `TraigentSchema`

This file is the working checklist for the remaining execution waves.

## Status Legend
- `[ ]` not started
- `[~]` in progress
- `[x]` complete
- `[!]` blocked

## Current Focus
- `[~]` Step 0 secure-and-push baseline across all active repos
- `[~]` Wave 3A core tenant/project enforcement completion
- `[~]` Wave 3B project RBAC kernel
- `[~]` Automated seeded release-validation harness
- `[~]` Wave 3B basic project membership UI
- `[~]` Wave 3B beta retention controls

## Step 0: Secure the Current Baseline
- `[x]` Commit and push all currently implemented but uncommitted work in `Traigent`
- `[x]` Commit and push all currently implemented but uncommitted work in `TraigentSchema`
- `[x]` Commit and push all currently implemented but uncommitted work in backend worktree
- `[x]` Commit and push all currently implemented but uncommitted work in SDK worktree
- `[x]` Commit and push all currently implemented but uncommitted work in frontend worktree
- `[x]` Record pushed commit SHAs in this tracker
- `[x]` Confirm tracker no longer mixes local-only implementation with committed baseline

### Step 0 pushed baseline commits
- `TraigentSchema` `feat/pr52-contract-sync-schema`: `de2f0bf`
- `TraigentBackend` `feature/langfuse-epic1-observability`: `b77ceae`
- `Traigent` SDK worktree `feature/langfuse-epic1-observability`: `e40eac1d`
- `TraigentFrontend` `feature/langfuse-epic1-observability`: `cd97ae1`
- `Traigent` main repo `develop`: `b0244ab5`

## Wave 3A: Core Tenant/Project Enforcement
- `[x]` Remove live single-tenant core-resource bypass in access checker v2
- `[x]` Scope admin API keys to tenant/project enforcement
- `[x]` Scope ExperimentRun and ConfigurationRun DALs
- `[x]` Fail closed when tenant/project context is missing in run DALs
- `[x]` Scope experiment-run routes and clone helpers
- `[x]` Scope example scoring routes
- `[x]` Scope benchmark/example DAL access
- `[x]` Scope evaluator target adapters
- `[x]` Scope trace span ingestion
- `[x]` Scope workflow trace trial span loading
- `[x]` Scope session-service ExperimentRun access
- `[x]` Scope optimization comparison run lookups
- `[x]` Scope trace graph experiment lookup
- `[x]` Scope evaluator backfill target queries
- `[x]` Audit and scope `trace_attribution_service.py`
- `[x]` Audit and scope `configuration_run_saga.py`
- `[x]` Scope agent DAL mutators and foreign measure/model-parameter attachment paths
- `[x]` Scope comparison, saga, evaluator-backfill, and trace-attribution read paths
- `[x]` Harden benchmark processor benchmark/file/config existence checks under active scope
- `[x]` Prevent unscoped core-model primary-key fetches from reappearing in runtime code
- `[~]` Audit remaining experiment/core runtime paths for transitive-scope assumptions
- `[ ]` Expand dual-run discrepancy telemetry beyond current access-checker coverage
- `[ ]` Verify no legacy route bypasses the enforcement kernel
- `[ ]` Inventory surviving legacy routes touching core resources
- `[ ]` Classify surviving legacy routes as safe vs unsafe
- `[ ]` Disable any unsafe legacy route before beta

### Migration-chain mitigation
- `[ ]` Commit the migration bootstrap/reconciliation path in Step 0
- `[ ]` Wire docker/dev startup to reconcile legacy DB state before `alembic upgrade head`
- `[ ]` Validate fresh DB startup path
- `[ ]` Validate legacy DB with global `alembic_version`
- `[ ]` Validate partially migrated local worktree DB
- `[ ]` If needed, provision fresh disposable seeded-validation DB fallback
- `[ ]` Record whether beta validation is satisfied via reconciled legacy DB or fresh fallback DB

## Seeded Validation Harness
- `[x]` Seed at least two tenants and two projects with realistic data
- `[x]` Seed experiments / experiment runs / configuration runs / observability traces / export jobs
- `[x]` Provide seeded validation user and memberships
- `[x]` Add backend fixture integration test
- `[x]` Add frontend Playwright seeded validation suite
- `[!]` Run seeded validation successfully against the live local worktree stack
- `[x]` Add explicit negative UI assertions for cross-project/cross-tenant absence
- `[x]` Fix fixture measure scoping so score records do not reference cross-tenant measures
- `[x]` Add safe handling/documentation for fixture credentials output

## Monitoring / Cost Baseline
- `[x]` Shared analytics query layer foundation
- `[x]` Cost provenance fields and explicit unpriced state
- `[x]` Pricing catalog contract and backend fallback path
- `[x]` Optimization overview dashboard
- `[ ]` Evaluator quality trends dashboard
- `[ ]` Project cost/token/latency/volume dashboard completion
- `[ ]` Observability summary dashboard

## Minimal Export Product
- `[x]` On-demand export jobs
- `[x]` Export history/detail/status
- `[x]` Privacy-safe manifest export by default
- `[ ]` Provider-neutral artifact-store abstraction
- `[ ]` Local artifact-store sink support
- `[ ]` Cloud sink adapter only if required by a beta customer
- `[ ]` Policy validation for raw-content export/materialization
- `[ ]` Coverage checkpoint after export artifact-storage slice

## Wave 3B: Project RBAC / Governance
- `[x]` Project role model: admin/editor/viewer
- `[~]` Backend authorization kernel
- `[~]` Membership APIs
- `[~]` Membership UI
- `[ ]` Audit events for membership/role/export/admin actions
- `[x]` Rate-limit policy product surface
- `[~]` Retention controls for export/raw-content surfaces

### Wave 3B kernel progress
- `[x]` Add project membership model and migration
- `[x]` Add project role resolver with tenant-admin override
- `[x]` Add project membership backend routes
- `[x]` Add project membership integration tests
- `[x]` Enforce viewer/editor roles on analytics and export routes
- `[~]` Expand project-role enforcement beyond analytics/export to additional protected mutations
- `[x]` Add audit events for project membership and role changes
- `[x]` Enforce viewer/editor roles on prompt-management routes
- `[x]` Enforce viewer/editor roles on evaluation routes
- `[x]` Enforce viewer/editor roles on annotation queue routes
- `[x]` Enforce viewer/editor roles on observability routes
- `[x]` Add focused integration coverage for the protected route surfaces above
- `[x]` Add project-membership frontend service
- `[x]` Add basic project-membership management UI on the Projects page
- `[x]` Add focused frontend tests for project-membership management
- `[x]` Add backend project rate-limit policy routes and audit logging
- `[x]` Add SDK project rate-limit policy DTOs and client methods
- `[x]` Add frontend project rate-limit policy service and Projects-page UI
- `[x]` Add focused backend/SDK/frontend tests for project rate-limit policy flows
- `[x]` Add backend project retention policy routes and audit logging
- `[x]` Add export-job retention metadata (`retention_category`, `expires_at`)
- `[x]` Add SDK project retention policy DTOs and client methods
- `[x]` Add frontend project retention policy service and Projects-page UI
- `[x]` Add focused backend/SDK/frontend/schema tests for retention policy flows
- `[ ]` Add retention-aware artifact cleanup/expiry enforcement beyond metadata

## Release Validation / Security
- `[ ]` Early lightweight threat-model review after Wave 3A stabilization
- `[ ]` Threat-model review notes for Wave 3A/3B
- `[ ]` Cross-tenant/project monitoring and alerting
- `[ ]` API reference for dashboards, roles, exports, retention/rate limits
- `[ ]` Migration/cutover notes for tenant/project enforcement
- `[ ]` Privacy-mode integration tests for all new beta surfaces
- `[ ]` Full CI and repo-wide validation rerun
- `[ ]` Changed-code coverage at or above 85%
- `[ ]` Claude review after each completed wave

## Notes
- Browser/JS feedback SDK, prompt playground, scheduled exports, export webhooks, and self-serve dashboards remain intentionally out of beta scope.
- Any legacy route that bypasses tenant/project enforcement must be disabled before beta, even if cosmetic route cleanup is deferred.
- Admin API keys are tenant-scoped in beta; cross-tenant operations are reserved for platform-admin JWT/internal tooling.
- The seeded validation fixture and Playwright suite are implemented and tested, but the live local run remains blocked until the worktree DB migration-chain issue is repaired.
