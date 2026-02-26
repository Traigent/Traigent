# Phase 2 - Backend Status

## Branch
- `observability-backend-phase-2-correlation-intelligence`

## Progress
- status: in_progress
- completed_tasks:
  - Added ingest payload-size guard in trace route (5 MB limit).
  - Added strict span validation in `TraceSpanService` for required fields, idempotency key, timestamps, numeric bounds, metadata shape, and IO payload size.
  - Added deterministic replay behavior by skipping already-ingested spans for same trace/config/span.
  - Updated route error mapping: validation errors -> `400`, missing configuration run -> `404`.
  - Added new route/service unit tests for hardening behavior.
- tests:
  - `pytest tests/unit/services/test_trace_span_service.py tests/unit/routes/test_trace_routes_ingest.py tests/unit/routes/test_experiment_run_traces.py -q` (pass)
- blockers:
  - None in backend hardening slice.
- notes:
  - latest_commit: `78bb827`
