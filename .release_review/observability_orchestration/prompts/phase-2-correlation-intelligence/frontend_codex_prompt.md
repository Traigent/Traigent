# Phase 2 - Frontend Agent Prompt (Hardening + Correlation)

## Repo
- `/home/nimrodbu/Traigent_enterprise/TraigentFrontend`

## Branch
- Work branch: `observability-frontend-phase-2-correlation-intelligence`

## Mission
- First port missing workflow trace UX hardening.
- Then build correlation intelligence UX.

## Phase 2A: Hardening (Required Before Correlation)
- File targets:
  - `src/services/workflowTraceService.ts`
  - `src/components/experiment/workflow/WorkflowTracesTab.tsx`
  - `src/services/__tests__/workflowTraceService.test.ts`
  - `src/components/experiment/workflow/__tests__/WorkflowTracesTab.test.tsx`
- Implement:
  - Remove redundant `hasWorkflowTraces()` preflight request from load path.
  - Add trace response cache TTL of `60s`.
  - Preserve manual trial selection on refresh unless selected trial disappears.
  - Separate empty-data UX from backend error UX.
  - Add large-trace rendering mode trigger (`>1k` spans) with graceful fallback.

## Phase 2B: Correlation UX
- Build benchmark-to-training correlation views with:
  - confidence and sample-size context
  - segment filters and URL state persistence
  - drill-down into example-level evidence
- Ensure filter state is deep-linkable and shareable.

## Required Tests
- Ensure single trace API call per tab load.
- Cache TTL hit/miss/expiry tests.
- Trial selection persistence tests.
- Error-path tests (`5xx`, malformed response payloads).
- Run relevant Vitest suites and report exact commands/results.

## Done Criteria
- One network round-trip for initial trace load.
- User-selected trial is stable across refreshes.
- Large traces remain navigable with fallback rendering mode.
- Correlation screens expose confidence/sample-size context and drill-down evidence.

## Reporting
- Update status file:
  - `/home/nimrodbu/Traigent_enterprise/Traigent/.release_review/observability_orchestration/status/phase-2-correlation-intelligence/frontend.md`
- Include:
  - latest commit SHA
  - exact files changed
  - test commands + results
  - blockers/risks
