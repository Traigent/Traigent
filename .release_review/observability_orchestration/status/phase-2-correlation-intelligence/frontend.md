# Phase 2 - Frontend Status

## Branch
- `observability-frontend-phase-2-correlation-intelligence`

## Progress
- status: done
- completed_tasks:
  - Kept single-fetch load path (no preflight dependency in main traces tab flow).
  - Switched workflow trace fetches to canonical observability endpoint (`/api/observability/runs/<run_id>?include_spans=true`).
  - Added adaptive trace cache TTL policy (`5s` for active runs, `60s` for stable runs).
  - Added strict malformed-payload validation in `workflowTraceService`.
  - Added selection-preservation logic for previously selected trial.
  - Replaced naive large-trace slicing with topology-preserving truncation (`<=1000` spans without orphaning children).
  - Expanded service/tab tests for canonical endpoint usage, adaptive TTL hit/miss/expiry, malformed payload handling, and large-trace banner behavior.
- tests:
  - `npx eslint src/services/workflowTraceService.ts src/services/__tests__/workflowTraceService.test.ts src/components/experiment/workflow/WorkflowTracesTab.tsx src/components/experiment/workflow/__tests__/WorkflowTracesTab.test.tsx` -> PASS
  - `npm run test -- src/services/__tests__/workflowTraceService.test.ts src/components/experiment/workflow/__tests__/WorkflowTracesTab.test.tsx` -> PASS (`75 passed`)
  - `npm run type-check` -> PASS
- blockers:
  - None.
- notes:
  - latest_commit: `e60c07fae008769864139b6e4c937744ab5cd280`
