# Phase 2 - SDK Status

## Branch
- `observability-sdk-phase-2-correlation-intelligence`

## Progress
- status: in_progress
- completed_tasks:
  - Added lock-protected span buffer access in `WorkflowTraceManager`.
  - Added deterministic span `idempotency_key` generation and emission.
  - Added recursive normalization/redaction coverage for tuple/set/dataclass/model payloads.
  - Added explicit async client-session close path in workflow trace client.
  - Added concurrency and payload/redaction tests.
- tests:
  - `pytest tests/unit/core/test_workflow_trace_manager.py tests/unit/integrations/observability/test_workflow_traces.py -q` (pass)
- blockers:
  - None in SDK hardening slice.
- notes:
  - latest_commit: `325e917`
