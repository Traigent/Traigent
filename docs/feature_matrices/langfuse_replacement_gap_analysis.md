# Langfuse Replacement Gap Analysis

Date: 2026-03-12

Scope:
- `TraigentSchema`
- `Traigent`
- `TraigentBackend`
- `TraigentFrontend`
- `traigent-js`
- `traigent-api`

Depends on:
- [langfuse_target_architecture.md](../architecture/langfuse_target_architecture.md)

Related documents:
- [langfuse_replacement_implementation_plan.md](langfuse_replacement_implementation_plan.md)
- [cross_repo_delivery_template.md](../architecture/cross_repo_delivery_template.md)

## Executive Summary

Traigent is no longer missing the same Langfuse-replacement foundations it was missing
at the start of the program. The current gap is less about absent primitives and more
about productization, analytics, export workflows, ecosystem breadth, and enterprise
governance depth.

The most important planning rule is to use the codebase, not stale historical matrix
rows, as the source of truth.

## Status Taxonomy

- `shipped-foundation`: implemented and structurally reusable
- `shipped-but-not-productized`: implemented, but still missing product depth
- `partial`: meaningful capability exists, but major gaps remain
- `not-started`: no meaningful product surface
- `blocked`: depends on another wave first

## Current State Snapshot

### Shipped baseline

- observability: `shipped-but-not-productized`
- prompt management: `shipped-but-not-productized`
- evaluation ops and annotation queues: `shipped-but-not-productized`
- tenant and project scoping: `partial`
- analytics foundation and privacy-safe exports: `partial`

### Highest-priority remaining gaps

1. shared analytics query layer completion
2. cost productization and prebuilt dashboards
3. data platform and export productization
4. core tenant/project enforcement completion
5. project-level RBAC and enterprise controls

### Explicitly tracked deferred gaps

- browser and JS feedback instrumentation
- webhooks and downstream event automation
- rate limiting productization
- data retention controls
- prompt playground and prompt experiments
- guardrail integrations and monitoring
- legacy-route removal

## Architectural Rules

- `Measure` is the single canonical metric or score-definition entity
- `EvaluatorDefinition` is the single canonical automated scorer-definition entity
- `ScoreRecord` is the single canonical measured outcome entity
- dashboards are query outputs, not domain entities
- privacy mode remains default
- no public wire contract lives outside `TraigentSchema`

## Category Matrix

| Langfuse category | Current status | What already exists | Biggest remaining gaps | Priority |
| --- | --- | --- | --- | --- |
| Observability and tracing | `shipped-but-not-productized` | live list/detail UI, trace comments and feedback, prompt linkage, sessions, nested observations, bookmarks/publishing, project-aware trace analytics | browser/end-user feedback SDK, richer attachments/multimodal support, webhook/event automation, further frontend instrumentation breadth | P1 |
| Prompt management | `shipped-but-not-productized` | prompt repo, versions, labels, analytics, prompt-trace linkage, frontend prompt management pages, SDK and backend contracts | playground/testing workflow, cache/webhook ecosystem, richer productized deployment flows | P1 |
| Evaluation and reviewer ops | `shipped-but-not-productized` | typed scores, evaluator definitions/runs, real judge execution, annotation queues, score panels, review primitives | queue analytics, bulk assignment, richer reviewer workflow, corrected-output lifecycle depth | P1 |
| Metrics and dashboards | `partial` | project analytics page, experiment dashboards, cost/token primitives, fine-tuning manifest export base, optimization analytics surfaces | one shared aggregation layer, prebuilt dashboards, unified dashboard contracts, cost provenance/productization | P0 |
| API and data platform | `partial` | privacy-safe manifest export base, project-scoped analytics/export contracts, broad backend API surface | scheduled exports, sink integrations, webhook automation, stronger reporting/export APIs | P0 |
| Administration and security | `partial` | tenant/project context, SSO enforcement, enterprise admin control plane, project-scoped routes and helpers | project-level RBAC, retention controls, rate-limit productization, SCIM-class admin depth | P0 |
| Core tenant/project enforcement | `partial` | tenant/project context and scoping helpers exist across the newer product surfaces | remove single-tenant core-resource bypasses, complete core-resource tenant/project enforcement, backfill remaining compatibility paths | P0 |
| Browser and JS ecosystem | `not-started` | Python SDK and frontend admin/operator surfaces are strong | browser/end-user feedback instrumentation, first-class JS/browser parity, privacy-safe browser capture | P1 |
| Playground and prompt experimentation | `not-started` | no first-class in-product playground | prompt testing, save-back workflows, side-by-side prompt/model experiments | P2 |
| Guardrails and monitoring | `partial` | security monitoring, rate limiting primitives, audit/security surfaces | guardrail integrations, guardrail result tracing, prompt-injection/content-safety productization | P2 |

## What Is Already Shipped

This section is intentionally explicit so the roadmap does not re-plan completed work.

### Observability

- beta observability API surface
- list/detail UI
- live list/detail refresh
- comments, thumbs feedback, bookmark/publish
- prompt linkage and prompt analytics joins

### Prompt management

- prompt definitions, immutable versions, mutable labels
- prompt detail/list/create/version flows
- usage analytics and prompt-trace linking

### Evaluation

- typed score system
- evaluator definitions/runs
- real LLM-as-judge execution
- annotation queues

### Tenant and project foundations

- tenant context
- project context
- project-scoped routes and helpers
- active tenant/project UX

These foundations are real, but they are not yet enough to claim “full enterprise-ready isolation” because the core-resource single-tenant bypasses still need to be removed.

## Ground-Truth Caveat

Prompt management, observability, and evaluator operations should not be planned as if
they are absent. The roadmap should treat them as shipped foundations that need more
product depth.

By contrast, tenant/project enforcement on the core optimization resources should not
be over-credited. The context and route scaffolding exists, but the program still
needs one dedicated enforcement wave to remove the remaining single-tenant access
paths from the core backend resource model.
