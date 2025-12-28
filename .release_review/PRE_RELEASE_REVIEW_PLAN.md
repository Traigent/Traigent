# Traigent Pre-Release Review Plan (SDK v0.8.0)

This plan breaks the repo into reviewable components with concrete checklists and required evidence.
Assignments and sign-offs live in `.release_review/PRE_RELEASE_REVIEW_TRACKING.md`.

## Scope

- **In-scope**: anything shipped in the SDK distribution (`traigent/`), plus docs/examples/CLI needed to onboard users.
- **Also in-scope**: release engineering (packaging, deps, versioning, CI) and security release blockers.
- **Optional**: `tools/`, `scripts/`, `playground/`, `walkthrough/`—review if they are part of the release story.

## Release Gates (Stop-Ship)

- [ ] All items in `RELEASE_BLOCKERS_TODO.md` resolved (or explicitly accepted risk + documented).
- [ ] Tests: `pytest` passes (unit + integration + e2e + security as configured).
- [ ] Lint/type: `ruff check` passes; `mypy` passes for its configured scope.
- [ ] Mock mode works fully offline (`TRAIGENT_MOCK_MODE=true`) for quickstarts.
- [ ] Version consistency: `pyproject.toml`, `traigent.__version__`, and CLI show the same version.
- [ ] Packaging smoke: build + install + CLI entrypoint works.

Reference success criteria: `docs/testing/RELEASE_READINESS_TESTING.md`.

## Release Completion Criteria

The release is approved when:
- [ ] All **P0** and **P1** components have Status = **Approved** in `.release_review/PRE_RELEASE_REVIEW_TRACKING.md`.
- [ ] All **Release Gates** above pass on the merged release-review branch (evidence recorded).
- [ ] `RELEASE_BLOCKERS_TODO.md` has all **Critical/High** items resolved or explicitly accepted with documented rationale.
- [ ] Captain ran final `pytest`, `ruff check`, and configured `mypy` scope on merged state and recorded PASS output.
- [ ] A human release owner signed off (name + date recorded in `.release_review/PRE_RELEASE_REVIEW_TRACKING.md`).

## Review Workflow

1. **Release captain** assigns owners/approvers in `.release_review/PRE_RELEASE_REVIEW_TRACKING.md`.
2. Each **component owner**:
   - Reviews code using the component checklist below.
   - Runs the most relevant tests for that component (or the full suite if unsure).
   - Verifies docs/examples relevant to the component are accurate (where applicable).
3. Owner records **evidence**:
   - A short written summary (what was checked, findings, follow-ups).
   - Links to PRs/issues/commits for any fixes.
   - Commands + PASS output (or CI run links).
4. **Approver** signs off and marks the component **Approved** in `.release_review/PRE_RELEASE_REVIEW_TRACKING.md`.

## Cross-Cutting Sweeps (Do Once, Then Spot-Check Per Component)

- [ ] Search for `TODO`, `FIXME`, `WIP`, `HACK`, `legacy`, `deprecated`, `print(` and decide: remove, ticket, or accept.
- [ ] Secrets hygiene sweep (align with `.secrets.baseline` and `docs/guides/secrets_management.md`).
- [ ] Confirm mock mode produces **no unexpected network calls** during quickstarts and key integration tests.

## Component Checklists

### 1) Public API & DX (`traigent/`, `traigent/api/`, `traigent/cli/`)

Paths:
- `traigent/__init__.py`
- `traigent/api/`
- `traigent/cli/`

Checklist:
- [ ] Public exports are intentional; deprecations are warned + documented.
- [ ] `@traigent.optimize(...)` parameters match docs and validate inputs cleanly.
- [ ] Errors are actionable (clear message + remediation hints).
- [ ] CLI commands have stable flags, good `--help`, and fail fast on missing env/config.

Suggested tests:
- `pytest tests/test_documentation.py tests/cli/ -q`
- `pytest tests/e2e/test_agent_flow.py -q` (smoke)

### 2) Configuration & Injection (`traigent/config/`)

Paths:
- `traigent/config/context.py`
- `traigent/config/runtime_injector.py`
- `traigent/config/seamless_injection.py`
- `traigent/config/ast_transformer.py`
- `traigent/config/types.py`
- `traigent/config/feature_flags.py`

Checklist:
- [ ] Injection modes match docs (`docs/features/seamless_injection.md`, `docs/user-guide/injection_modes.md`).
- [ ] Thread/async context propagation works (e.g., `copy_context_to_thread`).
- [ ] No hidden globals/import-time side effects that unexpectedly change user code.
- [ ] Edge cases: missing keys, partial defaults, reset/apply flows.

Suggested tests:
- `pytest tests/integration/test_injection_modes_apply_config.py -q`
- `pytest tests/integration/test_cost_enforcement_* -q`
- `pytest tests/unit/test_config.py -q`

### 3) Core Orchestration (`traigent/core/`)

Paths:
- `traigent/core/orchestrator.py`
- `traigent/core/optimized_function.py`
- `traigent/core/trial_lifecycle.py`
- `traigent/core/trial_context.py`
- `traigent/core/stop_condition_manager.py`
- `traigent/core/stop_conditions.py`
- `traigent/core/parallel_execution_manager.py`
- `traigent/core/cost_enforcement.py`
- `traigent/core/result_selection.py`
- `traigent/core/metrics_aggregator.py`

Checklist:
- [ ] Trial lifecycle/state machine is correct; resets are safe.
- [ ] Stop conditions and budgets enforce limits without race conditions.
- [ ] Parallel execution is bounded; cancellation is respected; cleanup always happens.
- [ ] Results selection/aggregation matches objective orientations/weights.

Suggested tests:
- `pytest tests/integration/test_full_optimization.py -q`
- `pytest tests/integration/test_execution_modes.py -q`
- `pytest tests/integration/test_cost_enforcement_* -q`

### 4) Optimizers (`traigent/optimizers/`)

Paths:
- `traigent/optimizers/` (base/registry/random/grid/bayesian/optuna*/pruners/checkpoint/remote/cloud/interactive)

Checklist:
- [ ] Base optimizer contract is implemented consistently (sync/async, failures, cancellation).
- [ ] Search-space handling is correct (categorical vs continuous; reproducible seeding).
- [ ] Optuna integration: checkpoint/resume, pruners, trial accounting, and storage behavior.
- [ ] Remote/cloud/interactive optimizers are either complete or clearly guarded/flagged.

Suggested tests:
- `pytest tests/unit/test_enhanced_base_optimizer.py -q`
- `pytest tests/e2e/test_interactive_flow.py -q`
- `pytest tests/e2e/test_hybrid_flow.py -q`

### 5) Evaluators & Metrics (`traigent/evaluators/`, `traigent/metrics/`)

Paths:
- `traigent/evaluators/local.py`
- `traigent/evaluators/metrics.py`
- `traigent/evaluators/metrics_tracker.py`
- `traigent/evaluators/dataset_registry.py`
- `traigent/metrics/registry.py`
- `traigent/metrics/ragas_metrics.py`

Checklist:
- [ ] Dataset formats validate cleanly and error messages point to the expected schema (see `docs/guides/evaluation.md`).
- [ ] Metrics are consistent across modes; `None`/missing handling matches `docs/features/strict_metrics_nulls.md`.
- [ ] Optional metrics (RAGAS, etc.) fail gracefully when deps aren't installed.

Suggested tests:
- `pytest tests/integration/test_metrics_flow.py -q`
- `pytest tests/integration/test_mock_mode_metrics.py -q`
- `pytest tests/performance/test_metrics_performance.py -q`

### 6) Invokers & Execution (`traigent/invokers/`, `traigent/adapters/`)

Paths:
- `traigent/invokers/local.py`
- `traigent/invokers/batch.py`
- `traigent/invokers/streaming.py`
- `traigent/adapters/execution_adapter.py`

Checklist:
- [ ] Sync/async behavior is consistent; streaming doesn't leak resources.
- [ ] Concurrency limits/timeouts are enforced; cancellation propagates.
- [ ] Batch mode handles partial failures deterministically.

Suggested tests:
- `pytest tests/unit/test_batch_processing.py -q`
- `pytest tests/integration/test_cost_enforcement_parallel.py -q`

### 7) Integrations (LLMs/frameworks/vector stores/observability) (`traigent/integrations/`)

Paths:
- Core: `traigent/integrations/framework_override.py`, `base.py`, `base_plugin.py`, `plugin_registry.py`, `wrappers.py`
- LLMs: `traigent/integrations/llms/`
- Model discovery: `traigent/integrations/model_discovery/`
- Vector stores: `traigent/integrations/vector_stores/`
- Observability: `traigent/integrations/observability/`
- Utilities: `traigent/integrations/utils/parameter_normalizer.py`, `validation.py`, `version_compat.py`

Checklist:
- [ ] Provider integrations are consistent (parameter mapping, validation rules, async handling).
- [ ] No legacy integration paths remain active; discovery/registry is deterministic.
- [ ] Override contexts restore originals reliably (no leaked monkey patches).
- [ ] Mock mode covers integrations that would otherwise require API keys.
- [ ] Provider ops docs are correct: `docs/operations/*.md`.

Suggested tests:
- `pytest tests/unit/test_enhanced_framework_override.py -q`
- `pytest tests/unit/test_plugins.py -q`
- `pytest tests/integration/test_optimization_with_platforms.py -q`
- `pytest tests/integration/test_plugin_ctd.py -q`

Reference inventories:
- `integrations_comprehensive_inventory.md`
- `docs/architecture/plugin_architecture.md`

### 8) Security & Privacy (`traigent/security/`, plus cross-cutting)

Paths:
- `traigent/security/` (jwt_validator, rate_limiter, headers, credentials, encryption, auth/*, input_validation)
- `RELEASE_BLOCKERS_TODO.md`

Checklist:
- [ ] No committed secrets; secrets are loaded from env/config and redacted in logs.
- [ ] Auth/session/token handling matches intended threat model; rate limiting is enabled where required.
- [ ] Crypto choices are safe (no MD5/weak defaults).
- [ ] Security headers applied where HTTP is served (if applicable).

Suggested tests:
- `pytest tests/security/ -q`
- `pytest tests/integration/test_privacy_* -q`

### 9) Storage & Persistence (`traigent/storage/`, `traigent/utils/persistence.py`)

Paths:
- `traigent/storage/local_storage.py`
- `traigent/utils/secure_path.py`
- `traigent/utils/file_versioning.py`
- `traigent/utils/persistence.py`

Checklist:
- [ ] File paths are validated/sandboxed; no unsafe temp dirs or path traversal.
- [ ] Concurrent read/write patterns are safe and atomic enough.
- [ ] Stored artifacts contain no secrets/PII by default.

Suggested tests:
- `pytest tests/storage/test_local_storage.py -q`

### 10) Telemetry, Analytics, Visualization, TVL (`traigent/telemetry/`, `traigent/analytics/`, `traigent/visualization/`, `traigent/tvl/`)

Checklist:
- [ ] Optional deps are handled gracefully (import guards, clear error messages).
- [ ] "Experimental" modules are clearly labeled and not required for core flows.
- [ ] Plots/analytics do not leak sensitive data by default.
- [ ] TVL spec loading is deterministic (`runtime/tvl_demo.yaml` is a useful smoke input).

Suggested tests:
- `pytest tests/unit/test_analytics.py -q`
- `pytest tests/utils/test_local_analytics.py -q`

### 11) Cloud / Hybrid / Experimental (`traigent/cloud/`, `traigent/optigen_integration.py`, `traigent/experimental/`)

Checklist:
- [ ] Cloud/hybrid code paths are either functional or clearly guarded/disabled in OSS flows.
- [ ] Clear errors when backend is not configured; no hidden network calls in mock mode.
- [ ] Docs reflect reality (see `docs/architecture/ARCHITECTURE.md` notes on OSS vs cloud).

Suggested tests:
- `pytest tests/cloud/ -q`
- `pytest tests/unit/test_optigen_integration.py -q`
- `pytest tests/mcp/ -q` (if MCP is part of the release surface)

### 12) Docs, Examples, Walkthrough, Playground (user-facing)

Paths:
- Docs: `docs/`, `README.md`, `DISCLAIMER.md`
- Examples: `examples/`, `walkthrough/`, `use-cases/`, `playground/`

Checklist:
- [ ] "First 10 minutes" path works (mock mode) with no guesswork.
- [ ] All referenced paths exist; env vars are documented; runtime estimates reasonable.
- [ ] Playground is clearly optional and has install/run instructions.
- [ ] Walkthrough scripts and demos still run or are clearly marked optional.

Suggested smoke runs:
- `export TRAIGENT_MOCK_MODE=true`
- `python examples/quickstart/01_simple_qa.py`
- `python examples/quickstart/02_customer_support_rag.py`
- `python examples/quickstart/03_custom_objectives.py`

Reference:
- `REVIEW_READY.md`
- `docs/testing/RELEASE_READINESS_TESTING.md`

### 13) Release Engineering (packaging, deps, CI)

Paths:
- `pyproject.toml`, `uv.lock`, `requirements/`, `MANIFEST.in`, `LICENSE`, `NOTICE`, `.github/`

Checklist:
- [ ] Optional extras install cleanly; dependency constraints reflect security fixes.
- [ ] Entry points work (`traigent` CLI).
- [ ] License/disclaimer files included in sdist/wheel.
- [ ] CI workflows cover core test matrix; no stale scripts.

Suggested commands:
- `python -m build`
- `pip install -e ".[dev,integrations,analytics,security]"`

## Optional Accelerators (Good for Reviewers)

- Automated review tracks: `tools/code_review/automation/run_all_validations.py`
- Contract compliance scans: `tools/traceability/contract_extractor.py`
- Codebase analysis: `python scripts/code_analysis/run_analysis.py`
