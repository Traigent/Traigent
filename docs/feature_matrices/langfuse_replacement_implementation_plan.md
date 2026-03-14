# Langfuse Gap-Closure Implementation Plan

Date: 2026-03-12

Depends on:
- [langfuse_replacement_gap_analysis.md](langfuse_replacement_gap_analysis.md)
- [langfuse_target_architecture.md](../architecture/langfuse_target_architecture.md)
- [cross_repo_delivery_template.md](../architecture/cross_repo_delivery_template.md)

## Summary

This is the plan of record for closing the remaining Langfuse-replacement gaps from
the codebase as it exists today, not from stale historical matrix rows.

### Current status taxonomy

- `shipped-foundation`: implemented and structurally reusable
- `shipped-but-not-productized`: implemented, but still missing product depth
- `partial`: meaningful capability exists, but major gaps remain
- `not-started`: no meaningful product surface yet
- `blocked`: depends on another wave first

### Verified current baseline

- observability: `shipped-but-not-productized`
- prompt management: `shipped-but-not-productized`
- evaluation ops and annotation queues: `shipped-but-not-productized`
- analytics foundation and privacy-safe exports: `partial`
- tenant and project isolation: `partial`
- project-level RBAC: `partial`
- browser and JS feedback instrumentation: `not-started`
- prompt playground: `not-started`

### Target deployment model

- primary target: one shared deployment with backend-enforced `tenant -> project -> resource` isolation
- compatibility mode: single-tenant deployments continue to work as one default tenant plus one default project
- no future work should assume tenant-per-deployment as the strategic model

### Program timing

- parallelized path: approximately `12.5 sprints` (`~25 weeks`)
- mostly sequential path: approximately `16.5 sprints` (`~33 weeks`)
- this excludes any extra rollback, migration, or procurement hardening discovered during Wave 3A

## Architectural red lines

- auth chain is always: `authenticate -> resolve tenant -> resolve project -> authorize -> scope data -> audit`
- `Measure` is the only canonical metric or score-definition entity
- `EvaluatorDefinition` is the only canonical automated scorer-definition entity
- `ScoreRecord` is the only canonical measured outcome entity
- dashboards are query outputs, not domain entities
- no public wire contract may be introduced outside `TraigentSchema`
- privacy mode remains default
- no route may bypass the tenant/project auth chain
- browser feedback may not introduce a second score/event taxonomy when existing score/feedback entities can represent it
- no new foundational abstraction is allowed in Waves 5 or 6

## Wave plan

### Wave 0: Normalize and lock the baseline

Duration: `0.5 sprint`

Depends on:
- none

Introduces migrations:
- no

Backfill required:
- no

Customer-visible changes:
- login reliability
- local admin bootstrap consistency
- authoritative roadmap docs

Key work:

- commit the local-only hardening still outside recorded branch history:
  - auth CORS allow-headers for tenant/project login flows
  - admin alias bootstrap support
  - latest demo/observability logging fixes
- refresh the gap-analysis matrix so prompt management, observability, and evaluation are no longer described as missing
- publish two standalone reference docs:
  - target architecture
  - cross-repo delivery template

Acceptance:

- every major area is tagged with one status from the approved taxonomy
- no local-only auth/scoping fix remains outside source control
- the target architecture and delivery template are referenced by later waves

### Wave 1A: Shared analytics query layer

Duration: `2 sprints`

Depends on:
- Wave 0

Introduces migrations:
- additive only if needed

Backfill required:
- no

Customer-visible changes:
- existing project analytics and optimization analytics move to shared queries

Key work:

- `TraigentSchema`
  - summary, trend, breakdown, and distribution contracts
  - explicit `bucket`, `requested_bucket`, and `resolved_bucket`
  - machine-checkable `privacy_classification`
- `TraigentBackend`
  - one PostgreSQL-backed aggregation/query module
  - migrate existing project analytics and optimization views onto it
- `Traigent`
  - typed analytics DTOs and client support
- `TraigentFrontend`
  - migrate existing analytics pages and optimization result views to the shared query layer

Acceptance:

- all analytics surfaces read from one aggregation layer
- no frontend or service defines analytics wire shapes outside `TraigentSchema`
- bucket resolution is deterministic and test-covered
- privacy-safe analytics works with no raw content

### Wave 1B: Cost productization and prebuilt dashboards

Duration: `2 sprints`

Depends on:
- Wave 1A

Introduces migrations:
- yes

Backfill required:
- likely historical pricing metadata

Customer-visible changes:
- cost-aware dashboards
- consistent cost and token totals across surfaces

Key work:

- `TraigentSchema`
  - pricing catalog contracts
  - cost-attribution contracts
  - provenance fields such as `cost_source` and `pricing_resolution_mode`
- `TraigentBackend`
  - pricing catalog by provider/model/version
  - consistent cost attribution for experiments, evaluators, prompts, and observability
  - prebuilt dashboards:
    - optimization overview
    - measure quality trends
    - evaluator score trends
    - project cost/token/latency/volume
    - prompt/version usage
    - observability summary
- `TraigentFrontend`
  - cost becomes first-class in analytics everywhere it already exists

Cost precedence:

1. observed provider usage
2. recorded model/provider/version metadata
3. pricing catalog fallback
4. explicit `unknown/unpriced`

Acceptance:

- the same underlying activity yields consistent cost totals across surfaces within defined rounding tolerance
- unpriced data is explicit and never silently zeroed
- dashboards remain query outputs, not entities

### Wave 2: Data platform and export productization

Duration: `2 sprints`

Depends on:
- Wave 1A

Introduces migrations:
- yes

Backfill required:
- no

Customer-visible changes:
- scheduled exports
- export history/status
- provider-neutral artifact storage
- webhook automation

Key work:

- `TraigentSchema`
  - export job, schedule, sink, artifact, and webhook contracts
  - explicit `privacy_mode` on export contracts
- `TraigentBackend`
  - scheduled exports
  - provider-neutral artifact-store abstraction
  - local sink first, cloud-specific adapters only when needed
  - export webhooks
  - privacy-safe manifest export by default
  - explicit materialized/raw-content export only when policy and environment allow it
  - artifact-store contract kept deliberately small:
    - `put(artifact_id, content_bytes, metadata) -> artifact_ref`
    - `get(artifact_ref) -> content_bytes`
    - `delete(artifact_ref) -> None`
    - export-job tables remain the index; the sink does not own listing/query semantics
- `Traigent`
  - export job client
- `TraigentFrontend`
  - export management page with schedule/history/status

Acceptance:

- scheduled and manual exports both work
- every export artifact includes machine-readable privacy metadata
- webhook delivery is idempotent and audited
- artifact-store configuration is validated before job execution

### Wave 3A: Core tenant/project enforcement completion

Duration: `3 sprints`

Depends on:
- Wave 1A

Introduces migrations:
- yes

Backfill required:
- yes, for any remaining single-tenant core rows

Customer-visible changes:
- core optimization and analytics surfaces become uniformly tenant/project enforced

This wave fixes the real backend gap before project-level RBAC.

Key work:

- remove any remaining single-tenant core-resource bypasses such as `_SINGLE_TENANT_CORE_TYPES`
- complete or retire the remaining legacy customer-isolation path in favor of the current tenant/project context chain
- make core-resource access paths uniformly enforce tenant/project scoping for:
  - agents
  - benchmarks
  - measures
  - experiments
  - experiment runs
  - configuration runs
  - analytics queries introduced in Wave 1A
- finish the migration/backfill strategy for existing single-tenant data

Rollback and cutover requirements:

- support a dual-run validation period where old and new enforcement paths log discrepancies before final cutover
- provide a fast compatibility toggle or rollback procedure for the old access path during rollout
- add validation that previously valid user access does not disappear for in-scope resources after cutover

Acceptance:

- no core-resource read or write path bypasses tenant/project enforcement
- direct cross-tenant/project access returns `404`
- list endpoints return only in-scope rows
- the analytics query layer is tenant/project safe by default
- dual-run validation is completed and logged discrepancies are resolved before final cutover
- rollback procedure is documented and testable in staging

### Wave 3B: Project-level RBAC and enterprise controls

Duration: `2 sprints`

Depends on:
- Wave 3A

Introduces migrations:
- yes

Backfill required:
- maybe for default memberships and roles

Customer-visible changes:
- project access management
- retention and rate-limit policy surfaces

This is the strategic enterprise release gate.

Key work:

- `TraigentSchema`
  - project membership and role contracts
  - retention and rate-limit policy contracts
- `TraigentBackend`
  - authorization kernel first:
    - project admin/editor/viewer minimum
    - backend enforcement on all project-protected writes
  - membership and policy APIs second
  - retention enforcement
  - productized rate-limit policies
- `Traigent`
  - project role and admin clients
- `TraigentFrontend`
  - project access management UI
  - clearer forbidden vs not-found UX

Acceptance:

- no project-protected mutation relies only on tenant roles or frontend checks
- every protected mutation endpoint has project-level authz tests
- membership and policy changes are audited

### Wave 4: Evaluation depth and reviewer operations

Duration: `2 sprints`

Depends on:
- Wave 3B

Introduces migrations:
- maybe

Backfill required:
- minimal

Customer-visible changes:
- queue analytics
- richer reviewer workflow
- bulk assignment and triage

Key work:

- queue analytics
- bulk assignment and triage
- richer reviewer workflow
- corrected-output lifecycle

Corrected-output semantics:

- corrected output is reviewer-authored supplemental data by default
- it does not silently replace original model output
- promotion to reference/gold artifact is an explicit state transition

Acceptance:

- reviewer and queue actions have auditable lifecycle transitions
- corrected-output states are contract-tested and unambiguous

### Wave 5: Browser/JS feedback and ecosystem parity

Duration: `2 sprints`

Depends on:
- Wave 2
- Wave 3B

Introduces migrations:
- maybe

Backfill required:
- no

Customer-visible changes:
- browser/end-user feedback capture
- JS/browser instrumentation parity

Key work:

- `TraigentSchema`
  - browser feedback contracts
  - privacy-safe local-content reference contracts
- `TraigentBackend`
  - end-user feedback ingestion linked to trace/session/project
- `TraigentFrontend`
  - embeddable feedback components and admin review
- `traigent-js`
  - brought in here because this wave genuinely depends on browser parity

Acceptance:

- browser capture is privacy-safe by default
- free-text capture is opt-in and policy-gated
- JS/browser parity is documented and materially improved

### Wave 6: Remaining parity gaps

Duration: `2 sprints`

Depends on:
- Waves 1 through 5

Introduces migrations:
- additive only

Backfill required:
- avoid if possible

Customer-visible changes:
- final parity closures

Hard boundary:

- no new foundational abstraction may be introduced in this wave

Key work:

- prompt playground and prompt experiments
- guardrail integrations and result tracing
- explicit legacy-route removal
- remaining procurement and deployment gaps

Legacy-route removal policy:

- instrument usage
- emit warnings/telemetry
- publish the removal threshold and target release
- remove only after migration completion or low usage

Acceptance:

- all remaining parity items are enumerated before the wave starts
- Wave 6 contains no new foundational entity or taxonomy

## Parallelism and delivery order

Default plan is sequential until Wave 1A is stable.

After Wave 1A lands, two streams may run in parallel if staffing allows:

- Stream A: Wave 1B -> Wave 2
- Stream B: Wave 3A -> Wave 3B

Wave 4 may begin exploratory work after Wave 3A is stable, but it depends on Wave 3B for release because reviewer operations require project-level authorization.

## Delivery template and NFRs

For every wave, use this order:

1. `TraigentSchema`
2. `TraigentBackend`
3. `Traigent`
4. `TraigentFrontend`
5. `traigent-js` or `traigent-api` only if the wave truly requires them

Every wave must declare:

- depends on
- introduces migrations
- backfill required
- customer-visible changes

Every wave must also satisfy these mandatory release gates:

- schema contract tests
- backend authorization and isolation tests
- privacy-mode tests
- audit-event tests for sensitive operations
- adoption and failure telemetry for the new surface

Default NFR measurement point:

- measure latency at the backend response boundary, excluding browser render time, on seeded standard project-size datasets in local/staging environments

Default targets:

- list/detail endpoints: p95 under `500ms`
- dashboard summary/trend endpoints: p95 under `1.5s`
- export job creation: synchronous response under `500ms`
- all high-cardinality endpoints must paginate

## Epic mapping

This plan supersedes the old numbered-wave narrative but keeps it mappable:

- Wave 1A / 1B -> the remaining analytics platform work from `EPIC-08`
- Wave 2 -> `EPIC-10` plus the export/webhook slice of `EPIC-11`
- Wave 3A / 3B -> the remaining enterprise identity and governance work from `EPIC-09`
- Wave 4 -> the maturity work still open from `EPIC-05` and `EPIC-06`
- Wave 5 -> browser/JS instrumentation plus the ecosystem gap previously deferred under `EPIC-11`
- Wave 6 -> long-tail parity from `EPIC-04`, `EPIC-12`, and remaining deployment/integration cleanup

## Assumptions and defaults

- prompt management and observability remain treated as shipped foundations unless Wave 0 evidence contradicts that
- tenant/project isolation is not considered fully shipped until Wave 3A removes the single-tenant bypasses
- Wave 2 depends on Wave 1A because exports must use shared analytics/query primitives
- privacy mode remains default across analytics, exports, evaluator workflows, and browser feedback
