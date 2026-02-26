# Phase 2 - SDK Status

## Branch
- `observability-sdk-phase-2-correlation-intelligence`

## Progress
- status: done
- completed_tasks:
  - Added lock-protected span buffer access in `WorkflowTraceManager`.
  - Added deterministic span `idempotency_key` generation and emission.
  - Extended recursive normalization/redaction with sensitive string-value masking (not only key-based masking).
  - Added explicit async client-session close path in workflow trace client.
  - Emitted richer lineage/training outcome metadata from trial lifecycle including `example_outcomes`, `sample_size`, and confidence context.
  - Raised workflow span collection failure logging to warning with traceback to avoid silent loss during rollout.
  - Added/updated payload and lifecycle tests for new metadata and redaction behavior.
- tests:
  - `.venv/bin/ruff check traigent/integrations/observability/workflow_traces.py traigent/core/trial_lifecycle.py tests/unit/integrations/observability/test_workflow_traces.py tests/unit/core/test_trial_lifecycle.py` -> PASS
  - `.venv/bin/pytest tests/unit/core/test_workflow_trace_manager.py tests/unit/integrations/observability/test_workflow_traces.py tests/unit/core/test_trial_lifecycle.py -q` -> PASS (`150 passed`)
- blockers:
  - None.
- notes:
  - latest_commit: `6c0c6530799f550c17d4d0f53ec0daaf6c970abd`
