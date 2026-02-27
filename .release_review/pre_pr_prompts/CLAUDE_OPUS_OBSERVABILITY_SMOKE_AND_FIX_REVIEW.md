# Claude Code Opus 4.6 Extended-Thinking Review Prompt

You are reviewing a cross-repo observability stabilization implementation for Traigent.

## Review goal
Perform a **deep code review** (security, correctness, reliability, performance, API/contract consistency, test quality) and identify any remaining issues after these fixes were applied.

Prioritize findings by severity:
1. `CRITICAL` (must block merge)
2. `HIGH` (should fix before release)
3. `MEDIUM` (next sprint)
4. `LOW` (improvement)

For each finding include:
- Repo + file path + line reference
- Why it is a problem
- How to reproduce / failure mode
- Concrete fix recommendation

If no issue in an area, say so explicitly.

---

## Scope and context
This is the first unreleased single-version observability release (no v1/v2 compatibility needed).

### Repos
- SDK: `/home/nimrodbu/Traigent_enterprise/Traigent`
- Backend: `/home/nimrodbu/Traigent_enterprise/TraigentBackend`
- Frontend: `/home/nimrodbu/Traigent_enterprise/TraigentFrontend`

### Targeted issues addressed in this patch set
1. SDK: lifecycle cleanup for async observability client sessions (prevent unclosed aiohttp sessions)
2. SDK: bound SQLite spool growth
3. Backend: remove ingest N+1 duplicate checks by batching idempotency key lookup
4. Frontend: route readiness should require both bootstrap requests, not either
5. Frontend: dashboard cache TTL consistency (60s)

---

## Files changed (focus review here)

### SDK
- `traigent/api/decorators.py`
- `traigent/core/optimization_pipeline.py`
- `traigent/core/orchestrator.py`
- `traigent/integrations/observability/spool.py`
- `traigent/integrations/observability/workflow_traces.py`
- `tests/unit/integrations/observability/test_spool.py`
- `tests/unit/integrations/observability/test_workflow_traces.py`
- `tests/unit/core/test_orchestrator_cleanup.py`

### Backend
- `src/services/observability_service.py`
- `tests/unit/services/test_observability_v2_service.py`

### Frontend
- `src/services/observabilityDashboardService.ts`
- `src/pages/ObservabilityDashboard.tsx`
- `src/pages/__tests__/ObservabilityDashboard.test.tsx`
- `src/services/__tests__/observabilityDashboardService.test.ts`

---

## Architectural expectations to validate

1. **SDK cleanup correctness**
- Tracker resources are always cleaned in orchestrator `finally` path.
- No async session leak in success/failure/cancellation flows.
- Cleanup failures do not crash optimization teardown.

2. **SDK spool durability and bounds**
- Spool size cannot grow unbounded.
- Capacity trimming is deterministic and FIFO-oldest.
- Replay deletion happens only after successful ack.

3. **Backend ingest idempotency and scaling**
- Idempotency key behavior remains deterministic.
- Batch duplicate prefetch cannot create false negatives/positives.
- Duplicate handling inside same request is correct.
- Fallback path behavior is safe.

4. **Frontend bootstrap and UX correctness**
- Route does not mask partial bootstrap failures.
- Error state behavior remains correct.
- Cache TTL aligns with the rest of observability flows.

5. **Cross-repo contract safety**
- SDK emitted payloads still accepted by backend.
- FE normalization still matches backend output shapes for dashboard and run/trial endpoints.

---

## Executed validation evidence (already run)

### SDK tests
Command:
`pytest -q tests/unit/integrations/observability/test_spool.py tests/unit/integrations/observability/test_workflow_traces.py tests/unit/core/test_orchestrator_cleanup.py tests/unit/api/test_decorators.py tests/unit/core/test_optimization_pipeline.py`

Result:
- `174 passed`

### Backend tests
Command:
`pytest -q tests/unit/routes/test_observability_v2_routes.py tests/unit/services/test_observability_v2_service.py`

Result:
- `44 passed`

### Frontend tests
Command:
`npm run test -- src/pages/__tests__/ObservabilityDashboard.test.tsx src/services/__tests__/observabilityDashboardService.test.ts`

Result:
- `9 passed`

### Runtime smoke (SDK -> BE observability)
Executed:
- `python examples/core/simple-prompt/run.py` with:
  - `TRAIGENT_OFFLINE_MODE=false`
  - `TRAIGENT_TRACES_ENABLED=true`
  - `TRAIGENT_MOCK_LLM=true`

Observed run:
- `experiment_id=361afbabf6aebabbf38d82d97e815cc1`
- `run_id=1c64efa5-a13e-4aa5-be5a-ec2ba4db80ee`
- SDK log: `Submitted 5 workflow spans`

Backend endpoint checks (all HTTP 200):
- `/api/observability/runs/{run_id}?include_spans=true` -> `trials=5`, `total_spans=5`
- `/api/observability/runs/{run_id}/trials?include_spans=true&limit=50` -> `items=5`
- `/api/observability/dashboard/runs/{run_id}/cost-latency?granularity=hour&limit=100` -> `items=1`
- `/api/observability/dashboard/run-health?limit=10` -> `items=10`

---

## Required output format

1. `Findings` section sorted by severity (CRITICAL -> LOW)
2. `Open Questions / Assumptions`
3. `Overall Risk Assessment`
4. `Recommended immediate fixes before merge`

Be strict and specific. Flag anything that could break production behavior even if tests pass.
