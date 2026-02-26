# Phase 2 Port Plan: Hardening + Correlation

## Why This Plan Exists
- Phase 2 branches were cut from observability base branches, not from Phase 1 completion commits.
- Result: hardening fixes from Phase 1 review are missing in current Phase 2 codepaths.
- Direct cherry-pick conflicted heavily and was aborted.

## Goal
- Port critical hardening into current Phase 2 codepaths first.
- Then implement Phase 2 correlation intelligence on top.

## Sequencing
1. SDK hardening baseline
2. Backend hardening baseline
3. Frontend hardening baseline
4. Cross-repo contract and e2e verification
5. Phase 2 correlation feature work

## SDK Port Map (`/home/nimrodbu/Traigent_enterprise/Traigent`)
- Target files:
  - `traigent/core/workflow_trace_manager.py`
  - `traigent/integrations/observability/workflow_traces.py`
  - `tests/unit/core/test_workflow_trace_manager.py`
  - `tests/unit/integrations/observability/test_workflow_traces.py`
- Required hardening:
  - Add lock-protected access for `_collected_spans` (collect, snapshot, clear).
  - Add required span `idempotency_key` and deterministic formula: `{trace_id}:{configuration_run_id}:{span_id}`.
  - Ensure SDK always sends `idempotency_key` in ingest payload.
  - Add recursive redaction support for dict/list/tuple/set/dataclass/model objects.
  - Remove `aiohttp` client session leakage in trace client flow.
- Required tests:
  - Real concurrent `collect_span` test using `threading.Barrier`.
  - Payload contract parity including `idempotency_key`.
  - Redaction coverage for tuple/set/dataclass payloads.

## Backend Port Map (`/home/nimrodbu/Traigent_enterprise/TraigentBackend`)
- Target files:
  - `src/routes/trace_routes.py`
  - `src/services/trace_span_service.py`
  - `src/models/trace_span.py`
  - `src/routes/experiment_run_routes.py`
  - `tests/unit/routes/*trace*`
  - `tests/unit/services/*trace*`
- Required hardening:
  - Accept and validate required `idempotency_key` per span.
  - Enforce ingest bounds (request size, span count, metadata limits, numeric/timestamp bounds).
  - Reject malformed timestamps and negative duration/token/cost fields.
  - Make span ingestion atomic for the whole request (no partial commit on per-span failures).
  - Add idempotent behavior for retries/replays (dedupe by idempotency key).
  - Ensure run traces endpoint returns deterministic error semantics (`404` only for true not-found).
- Required tests:
  - Ingest validation matrix for bound violations.
  - Duplicate replay test for idempotency.
  - Atomic rollback test on mid-batch failure.
  - `GET /runs/{id}/traces` behavior tests for no-graph/no-span conditions.

## Frontend Port Map (`/home/nimrodbu/Traigent_enterprise/TraigentFrontend`)
- Target files:
  - `src/services/workflowTraceService.ts`
  - `src/components/experiment/workflow/WorkflowTracesTab.tsx`
  - `src/services/__tests__/workflowTraceService.test.ts`
  - `src/components/experiment/workflow/__tests__/WorkflowTracesTab.test.tsx`
- Required hardening:
  - Remove redundant preflight `hasWorkflowTraces()` request from tab load path.
  - Add cache TTL (`60s`) for trace responses.
  - Preserve user trial selection across reloads unless selection becomes invalid.
  - Differentiate empty-data state vs backend-error state.
  - Add volume-based rendering mode fallback (full vs large-trace mode indicator).
- Required tests:
  - Assert one network call per load.
  - Cache hit/miss/expiry tests.
  - Selection persistence tests.
  - Explicit error-path tests (`5xx`, malformed payload).

## Cross-Repo Gate (Must Pass Before Phase 2 Feature Merge)
- SDK payload fixture validates in backend ingest validator.
- End-to-end path verified:
  - SDK run emits traces
  - Backend stores and serves run traces
  - Frontend `WorkflowTracesTab` renders real backend data without fallback mocks
- Regression checks:
  - No unclosed client session warnings.
  - No duplicate spans under retry.
  - No extra preflight request from frontend trace tab.
