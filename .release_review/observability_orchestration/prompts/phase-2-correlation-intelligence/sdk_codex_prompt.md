# Phase 2 - SDK Agent Prompt (Hardening + Correlation)

## Repo
- `/home/nimrodbu/Traigent_enterprise/Traigent`

## Branch
- Work branch: `observability-sdk-phase-2-correlation-intelligence`

## Mission
- First port missing hardening fixes into the current Phase 2 codepaths.
- Then deliver SDK-side correlation telemetry contracts.

## Phase 2A: Hardening (Required Before Correlation)
- File targets:
  - `traigent/core/workflow_trace_manager.py`
  - `traigent/integrations/observability/workflow_traces.py`
  - `tests/unit/core/test_workflow_trace_manager.py`
  - `tests/unit/integrations/observability/test_workflow_traces.py`
- Implement:
  - Thread-safe span buffer access for `WorkflowTraceManager` using a lock around all `_collected_spans` mutations/snapshots/clears.
  - Required `idempotency_key` on each span with formula `{trace_id}:{configuration_run_id}:{span_id}`.
  - SDK must always send `idempotency_key` in ingest payload.
  - Recursive redaction coverage for dict/list/tuple/set/dataclass/model payloads.
  - Remove trace client session leakage (`aiohttp` unclosed session path).

## Phase 2B: Correlation Payloads
- Emit training outcome events with:
  - `training_run_id`
  - `dataset_id`
  - `example_id`
  - pre/post benchmark metrics
  - confidence/sample-size metadata
- Add contract fixtures for backend correlation ingestion.

## Required Tests
- `pytest tests/unit/core/test_workflow_trace_manager.py -q`
- `pytest tests/unit/integrations/observability/test_workflow_traces.py -q`
- Add concurrency test with `threading.Barrier` to force overlap.
- Add payload parity tests for required `idempotency_key`.

## Done Criteria
- No unsynchronized access to shared span buffers.
- No span payload emitted without `idempotency_key`.
- No unclosed `aiohttp` session warnings in test/demo runs.
- Correlation contract fixtures validated and committed.

## Reporting
- Update status file:
  - `/home/nimrodbu/Traigent_enterprise/Traigent/.release_review/observability_orchestration/status/phase-2-correlation-intelligence/sdk.md`
- Include:
  - latest commit SHA
  - exact files changed
  - test commands + results
  - blockers/risks
