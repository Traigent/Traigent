# Enterprise Beta Execution Plan

Date: 2026-03-13

Related documents:
- [competitor_feature_gap_analysis.md](competitor_feature_gap_analysis.md)
- [langfuse_replacement_implementation_plan.md](langfuse_replacement_implementation_plan.md)
- [enterprise_beta_execution_tracker.md](enterprise_beta_execution_tracker.md)
- [../architecture/langfuse_target_architecture.md](../architecture/langfuse_target_architecture.md)
- [../architecture/cross_repo_delivery_template.md](../architecture/cross_repo_delivery_template.md)
- [../architecture/enterprise_beta_threat_model.md](../architecture/enterprise_beta_threat_model.md)

## Summary

Ship an **Enterprise Beta** focused on enterprise-safe isolation, governance,
monitoring, and operational exports rather than broad feature parity.

### Progress snapshot

- tracker completion is roughly **74%**
- effort-weighted beta readiness is roughly **60-65%**
- risk-adjusted time remaining is roughly **10-14 weeks**
  - lower if the migration-chain blocker is resolved quickly
  - higher if migration repair or threat-model review surfaces more code changes

### Locked decisions

- **Admin API keys are tenant-scoped.**
  - `admin` means tenant-admin, not cross-tenant super-admin.
  - Cross-tenant access is reserved for platform-admin JWT/internal tooling.
  - No public API key path may bypass the tenant/project enforcement kernel.
- **Seeded demo verification is automated.**
  - The seeded validation path is a CI/staging-runnable seed plus assertion suite,
    not a manual QA walkthrough.
- **Privacy mode stays default.**
  - Analytics, exports, and monitoring must work with ids, hashes, refs, and
    aggregates by default.
  - Raw-content export or materialization is explicit and policy-gated.
- **Export storage stays provider-neutral.**
  - Beta does not assume customer AWS usage or require S3 accounts.
  - Export artifacts are written through a backend artifact-store abstraction.
  - Beta ships with a local artifact-store sink first.
  - Cloud-specific sinks are added only if a beta customer explicitly requires them.

## Step 0: Secure the current baseline

Before further implementation starts:

- commit and push all currently implemented but uncommitted work across:
  - `Traigent`
  - `TraigentSchema`
  - backend worktree
  - SDK worktree
  - frontend worktree
- record the pushed commit set in
  [enterprise_beta_execution_tracker.md](enterprise_beta_execution_tracker.md)
- ensure tracker state no longer mixes “implemented locally” with “committed baseline”

This is mandatory because too much high-value security and governance work is
still only in working trees.

## Must Ship Before Beta

### 1. Monitoring and cost baseline

Ship:
- one shared analytics query layer
- cost provenance with precedence:
  1. observed provider usage
  2. recorded model/provider/version metadata
  3. pricing catalog fallback
  4. explicit `unknown/unpriced`
- prebuilt dashboards only:
  - optimization overview
  - evaluator quality trends
  - project cost/token/latency/volume
  - observability summary

Required behavior:
- dashboards are query outputs, not entities
- unpriced data is explicit, never silently zeroed

### 2. Wave 3A: Core tenant/project enforcement

Ship:
- remove remaining single-tenant core-resource bypasses
- enforce tenant/project scoping on:
  - agents
  - benchmarks
  - measures
  - experiments
  - experiment runs
  - configuration runs
  - analytics queries
- harden transitive-scope assumptions during route audit
- backfill legacy single-tenant rows using:
  - default tenant + default project assignment for deterministic rows
  - idempotent backfill
  - quarantine for unmappable rows
  - no runtime ownership guessing

Cutover and rollback:
- dual-run validation before cutover
- old and new enforcement paths log discrepancies in parallel
- rollback is an emergency compatibility toggle only
- beta cutover is blocked until:
  - seeded discrepancy cases are clean
  - rollback drill has run once
  - no unresolved discrepancy classes remain

Release gate:
- no request path may reach core resources without tenant/project enforcement
- legacy routes may remain only if they flow through the same enforcement kernel
- any bypassing legacy route must be disabled before beta

### Migration-chain mitigation

The current live local worktree DB is carrying a legacy global
`alembic_version` state, while the backend now expects a dedicated
`traigent_alembic_version` flow through the bootstrap/reconciliation path.

What already exists in the backend worktree:
- backend migration bootstrap logic
- schema-anchor detection
- dedicated backend version table support
- reconciliation script for legacy DB state

Mitigation plan:
1. commit the bootstrap/reconciliation code in Step 0
2. make the docker/dev startup path run reconciliation before `alembic upgrade head`
3. validate three cases:
   - fresh DB
   - legacy DB with global `alembic_version`
   - partially migrated local worktree DB
4. if in-place repair fails, immediately switch seeded validation to a fresh disposable DB
5. continue repairing the legacy local DB in parallel, but do not block Wave 3A on it forever

Time-box:
- legacy DB repair gets **2 working days** after Step 0
- if unresolved, live seeded validation proceeds on the fresh validated DB path

Success criteria:
- backend starts cleanly against fresh and reconciled DBs
- seeded validation passes on at least one real live stack
- the tracker records whether the beta gate is satisfied via:
  - repaired legacy DB
  - or fresh validated DB fallback

### 3. Wave 3B: Project-level RBAC and enterprise controls

Ship:
- project roles:
  - admin
  - editor
  - viewer
- backend authorization kernel first
- membership APIs and basic UI second
- audit events for:
  - membership changes
  - role changes
  - exports
  - admin/security changes
- rate-limit policies
- retention controls limited in beta to:
  - export artifacts
  - materialized/raw-content export payloads

Deferred from beta:
- destructive auto-deletion for core optimization history

### 4. Minimal operational export product

Ship:
- on-demand export jobs
- export history/detail/status
- privacy-safe manifest export by default
- one provider-neutral artifact-store abstraction
- local artifact-store sink for beta by default
- one cloud sink only if a beta customer explicitly requires it
- explicit materialized/raw-content export only when policy allows it

Artifact-store contract for beta:
- `put(artifact_id, content_bytes, metadata) -> artifact_ref`
- `get(artifact_ref) -> content_bytes`
- `delete(artifact_ref) -> None`
- listing/query remains the responsibility of `project_export_jobs`; the sink is blob-oriented, not the system of record

Deferred to Release +1:
- scheduled exports
- export webhooks
- additional cloud sinks (for example S3-compatible, Azure Blob, or GCS)

## Automated Release Validation

Build one seeded validation harness that:
- seeds at least two tenants and two projects with:
  - experiments
  - experiment runs
  - configuration runs
  - measures/scores
  - observability traces
  - export jobs
- runs backend assertions for:
  - in-scope visibility
  - out-of-scope `404` or empty-list behavior
  - dashboard aggregation staying in-scope
- runs Playwright UI assertions for:
  - optimization views showing seeded runs
  - observability views showing seeded traces
  - dashboard cards/tables reflecting only the active tenant/project
  - membership/role changes affecting visible access
  - export history/status being visible and scoped

This suite must be runnable in CI or staging without manual intervention.

Privacy-mode validation is also a release requirement:
- every new analytics, export, and monitoring surface added for beta must have
  privacy-mode integration coverage
- those tests must prove the surface works correctly with ids, hashes, refs, and
  aggregates only, without requiring raw content

## Release Hardening Work

### Early lightweight threat-model pass

Start this immediately after Wave 3A is stable, not only at the end.

Focus on:
- auth boundary
- tenant/project resolution
- legacy route exposure
- export/raw-content policy transitions

Any findings that require code changes should feed back before final release
validation becomes schedule-critical.

## Sequencing

1. Step 0: secure the current baseline
2. finish Wave 3A enforcement
3. finish the minimal export product
4. finish the monitoring/cost baseline
5. complete release hardening and final validation

Definition of **Wave 3A stable**:
- enforcement code merged
- dual-run discrepancy telemetry active
- seeded discrepancy coverage clean
- safe vs unsafe legacy route inventory completed
- unsafe legacy routes disabled
- clean soak period of 5 business days
- rollback drill completed successfully

Allowed overlap:
- only after Wave 3A is code-complete and seeded validation is unblocked
- Stream A: minimal export product
- Stream B: monitoring/cost baseline
- dashboard work is allowed to proceed in parallel with export work after that
  gate, because it depends on the shared analytics layer and scoped data, not on
  artifact storage

Coverage checkpoint:
- at the end of the export-product phase, run a changed-code coverage checkpoint
- if coverage is below **85%**, close the gap immediately instead of deferring it to final release validation

## Explicitly Deferred to Release +1

- prompt testing and prompt experiments
- dataset ops and regression/testing workflows
- guardrails and safety-result tracing
- reviewer workflow depth and richer observability UX
- browser/JS feedback SDK
- scheduled exports and export webhooks
- self-serve dashboard builder
- multi-cloud sink breadth

## Release Gates

Beta is only eligible when all of these are true:

### Security and isolation
- tenant/project enforcement covers all core resources
- no unsafe legacy route remains enabled
- monitoring and alerting for cross-tenant/project failures is active

### Validation
- seeded backend plus Playwright suite passes on a live stack
- full CI is green
- changed-code coverage is **>= 85%**
- privacy-mode integration tests pass for all new beta surfaces

### Documentation
- API reference exists for dashboards, roles, exports, retention, and rate limits
- migration/cutover notes exist
- privacy-mode behavior is documented

### Review
- lightweight threat-model review is complete
- findings are resolved or explicitly accepted before beta cut
