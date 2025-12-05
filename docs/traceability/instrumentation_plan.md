# Runtime Instrumentation Plan (TraceSync)

Purpose: add runtime trace hooks using `tools.trace_sync.runtime` so `verify-runtime` can validate flows against specs. This plan is executable next session.

## Prereqs
- Ensure CodeSync package is importable: from the CodeSync repo run `pip install -e .[dev]` or add its path to `PYTHONPATH`.
- Runtime helpers to import in Traigent:  
  `from tools.trace_sync.runtime import flow_context, set_flow_id, concept_action, sync_rule, record_action`
- Default log: `<repo>/runtime/traces/runtime.log`; override via env `TRACE_SYNC_LOG_PATH` if needed.

## Instrumentation principles
- Instrument concept/requirement boundaries, not every helper.
- Emit:
  - `flow_context` decorator at flow entry (CLI/API/orchestrator main).
  - `set_flow_id` if an id already exists in context/request.
  - `concept_action(concept_id=..., req_ids=[...], func_ids=[...], details=...)` where concept logic executes.
  - `record_action(kind="sync", sync_ids=[...], details=...)` at SYNC-* boundaries.
  - Optional `record_action(kind="event", ...)` for key steps (suggest/record/save/load).
- Include: flow/session id, trial id, dataset id, optimizer name, objectives/metrics, execution_mode, provider/model, storage id; exclude PII/prompts.

## Targeted entrypoints & hooks
- **API/CLI** (`traigent/api/decorators.py`, `api/functions.py`, `cli/main.py`, `cli/auth_commands.py`, `cli/local_commands.py`): `flow_context` + `concept_action` (Layer-API, REQ-API-001, FUNC-API-ENTRY); sync with OptimizationFlow.
- **Orchestration/Core** (`core/optimized_function.py`, `core/orchestrator.py`): `flow_context` around optimize, `concept_action` for orchestration (REQ-ORCH-003), `record_action(kind="sync", sync_ids=["SYNC-OptimizationFlow"])` at loop stages.
- **Invokers** (`invokers/base.py`, `invokers/local.py`, `invokers/batch.py`): `concept_action` around invoke/invoke_batch (REQ-INV-006, REQ-INJ-002), include execution_mode/status; sync with orchestrator as needed.
- **Evaluators** (`evaluators/base.py`, `evaluators/local.py`): `concept_action` before batch eval and after metrics aggregation (REQ-EVAL-005), include dataset id/metrics; sync with OptimizationFlow.
- **Optimizers** (`optimizers/*` suggest/record paths): `record_action(kind="event")` per trial, `concept_action` for optimizer concept (REQ-OPT-ALG-004); remote/interactive/cloud emit sync for `SYNC-CloudHybrid`.
- **Storage** (`storage/local_storage.py`, `utils/persistence.py`, `utils/file_versioning.py`): `concept_action` on save/load, include storage path/id; `record_action(kind="event")` for writes.
- **Cloud/Security** (`cloud/*`, `optigen_integration.py`, `security/headers.py`, `security/jwt_validator.py`): `concept_action` for security/compliance; `record_action(kind="sync")` for cloud interactions.
- **Integrations** (`integrations/*`, `utils/langchain_interceptor.py`): `concept_action` when calling provider adapters; `record_action(kind="sync", sync_ids=["SYNC-IntegrationHook"])`.
- **Analytics/Telemetry** (`analytics/*`, `telemetry/optuna_metrics.py`, `utils/optimization_logger.py`, `utils/local_analytics.py`): `concept_action` for analytics; sync to observability; include metrics metadata (no PII).
- **TVL/Config** (`tvl/spec_loader.py`, `tvl/options.py`, `config/types.py` overlays): `concept_action` when loading/applying specs; include spec id/version.
- **Experimental** (`experimental/simple_cloud/*`): only entrypoints if needed; mark Layer-Experimental.

## Coordination
- Use `docs/traceability/tagging_tracking.md` to claim instrumentation rows (set `in-progress` with agent_id/UUID).
- Make minimal edits per file; add imports + decorators/calls; avoid behavior changes.
- After instrumenting a batch, run a flow to produce runtime log, then `python -m tools.trace_sync.cli verify-runtime --log runtime/traces/runtime.log --repo /path/to/specs`.

## Notes
- Keep new instrumentation in-place in existing modules (no separate folder needed because hooks must wrap the real entrypoints). If config helpers are needed, add a small module under `traigent/config` or `traigent/utils` with clear naming.
