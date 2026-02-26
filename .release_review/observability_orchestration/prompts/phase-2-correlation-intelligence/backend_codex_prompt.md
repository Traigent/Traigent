# Phase 2 - Backend Agent Prompt (Hardening + Correlation)

## Repo
- `/home/nimrodbu/Traigent_enterprise/TraigentBackend`

## Branch
- Work branch: `observability-backend-phase-2-correlation-intelligence`

## Mission
- First port missing ingest/query hardening into current Phase 2 codepaths.
- Then build correlation intelligence APIs.

## Phase 2A: Hardening (Required Before Correlation)
- File targets:
  - `src/routes/trace_routes.py`
  - `src/services/trace_span_service.py`
  - `src/models/trace_span.py`
  - `src/routes/experiment_run_routes.py`
  - related tests under `tests/unit/routes` and `tests/unit/services`
- Implement:
  - Require per-span `idempotency_key` and enforce deterministic dedupe on replay.
  - Add strict ingest validators:
    - max request size
    - max spans per request
    - metadata shape/depth bounds
    - non-negative token/cost fields
    - valid timezone-aware ISO timestamps
    - `end_time >= start_time` if both provided
  - Make ingest request atomic (no partial writes on per-span failure).
  - Ensure run trace fetch endpoint has deterministic 404 vs 200 behavior.

## Phase 2B: Correlation Intelligence
- Build correlation engine with sample-size/confidence gating.
- Expose drill-down endpoints for:
  - benchmark metric -> training outcome correlation
  - example-level evidence rows
  - segment filters
- Return confidence intervals and gating metadata in payloads.

## Required Tests
- Add unit tests for ingest validation failures.
- Add idempotent replay tests for duplicate spans.
- Add atomic rollback test for mid-batch failure.
- Add correlation correctness tests with deterministic fixture datasets.
- Run targeted suites for changed routes/services and report exact commands.

## Done Criteria
- Ingest rejects malformed payloads with deterministic `400`.
- Duplicate replay does not create duplicate spans.
- Failed ingest batch leaves no partial persisted rows.
- Correlation endpoints return confidence/sample-size metadata.

## Reporting
- Update status file:
  - `/home/nimrodbu/Traigent_enterprise/Traigent/.release_review/observability_orchestration/status/phase-2-correlation-intelligence/backend.md`
- Include:
  - latest commit SHA
  - exact files changed
  - test commands + results
  - blockers/risks
