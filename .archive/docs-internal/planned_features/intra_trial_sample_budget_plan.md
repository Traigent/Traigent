# Intra-Trial Sample Budget Enforcement: Implementation Plan

## Objectives
- Enforce `max_total_examples` _within_ a running trial by allowing evaluators to stop once the global sample budget is exhausted.
- Support all execution modes (local/edge analytics, hybrid, remote SaaS), sync/async evaluators, and parallel trial scheduling.
- Maintain compatibility with existing stop conditions, optimizers (including Optuna), and telemetry surfaces.
- Deliver comprehensive test coverage and instrumentation for observability.

## Scope Overview
- Core orchestration: new budget manager, lease lifecycle, orchestration hooks.
- Evaluators: budget-aware batching for sequential, async, and parallel paths.
- Optimizers/adapters: Optuna pruning, hybrid adapter propagation, remote DTO extensions.
- Telemetry & reporting: standardised exhaustion metadata, logging, and metrics.
- Testing, documentation, and configuration validation.

Out of scope for this iteration: advanced sampler implementations (stratified/curriculum/etc.), non-critical telemetry dashboards, UI treatment.

## Key Concepts

### SampleBudgetManager
- Thread-safe controller owned by `OptimizationOrchestrator`.
- Tracks the global budget (`max_total_examples`) and cumulative consumption.
- Issues `SampleBudgetLease` objects per trial.

### SampleBudgetLease
- Records allocation, consumption, and exhaustion state for a specific trial.
- Methods:
  - `remaining()` – current unconsumed budget visible to the trial.
  - `try_take(n)` – atomically reserves `n` examples; returns `False` when the global budget is depleted.
  - `rollback(n)` – returns unused slots (e.g., on cancellation/error).
  - `finalize(status)` – closes the lease, reports final consumption, and flags `sample_budget_exhausted` when appropriate.
- Internally communicates actual deltas back to the manager so `OptimizationOrchestrator` only counts confirmed consumption.

### Allocation Model
- Default: dynamic acquisition (`try_take`) with no upfront reservation to avoid starvation.
- Optional preallocation hook (`reserve(n)`) for future fairness strategies, but initial implementation will emphasise simplicity.
- Fairness guard: orchestrator clamp ensures each launched trial can consume at least one example when budget is positive.
- Parallel batches: orchestrator calculates per-trial _upper bounds_ based on remaining budget and dataset size; leases enforce the ceiling dynamically.

## Architecture Changes

### 1. OptimizationOrchestrator
1. Instantiate `SampleBudgetManager` during `__init__` when `max_total_examples` is set.
2. Expose `acquire_budget_lease(trial_id, dataset_size, parallel_hint)` to trial runners.
3. In `_run_trial`, request a lease and pass it through to the evaluator (`sample_lease` kwarg).
4. After evaluation, finalise the lease, update `_consumed_examples`, and persist `examples_attempted`, `sample_budget_exhausted`, `budget_remaining`, etc., on the resulting `TrialResult`.
5. Parallel execution (`_run_parallel_batch`):
   - Calculate per-trial ceilings using `min(dataset_size, remaining_budget - (num_trials - idx - 1))`.
   - Abort launching additional trials once the budget cannot guarantee one example per trial.
6. Update `_register_examples_attempted` to prefer lease metrics (enforced by finalisation) and to avoid double counting.
7. When a lease reports exhaustion, set `_stop_reason = "max_samples_reached"` and increment `_examples_capped`.

### 2. Evaluators
#### BaseEvaluator
- Update `evaluate` signature to accept optional `sample_lease`.
- Propagate the lease to `_evaluate_batch`.
- Return `(outputs, errors, example_results, consumed_count, exhausted_flag)` from `_evaluate_batch`.
- Maintain backwards compatibility: when no lease, behaviour unchanged.

#### LocalEvaluator
- Sequential path: before processing each example, call `sample_lease.try_take(1)`. Exit loop when `False`.
- Detailed mode: mark truncated examples with `success=False`, `error="sample_budget_exhausted"`, and stop iterating.
- Async path (`max_workers > 1`):
  - Replace bulk `asyncio.gather` with worker queue that only schedules when `try_take(1)` succeeds.
  - Cancel in-flight tasks on exhaustion; ensure results/metrics are gathered from completed tasks only.
- Batch helpers: if `Dataset` exposes `get_batch(size)`, request the entire batch, clamp to remaining budget, and handle partial batches gracefully.
- Ensure `progress_callback` fires with `{"stop_reason": "sample_budget_exhausted"}` when breaking early.
- Populate `EvaluationResult.total_examples`, `EvaluationResult.metadata["sample_budget_exhausted"]`, and `EvaluationResult.metadata["examples_consumed"]`.

### 3. Execution Modes
- **Edge Analytics / Local**: covered by LocalEvaluator changes.
- **HybridPlatformAdapter**:
  - Inject leases when delegating to `LocalExecutionAdapter`.
  - Attach budget metadata to outgoing results.
  - Prevent platform-specific retries after exhaustion.
- **Remote SaaS (OptiGen)**:
  - Extend session DTOs with optional `max_total_examples` and `remaining_budget`.
  - Backend client must include the remaining budget on execution requests.
  - Ensure remote runners mirror lease semantics (placeholder stub with TODO for backend alignment; guard with feature flag if remote support lags).

### 4. Optimizer Integration
- On `sample_budget_exhausted`, map trial outcome to `TrialStatus.PRUNED` with `trial_result.metadata["stop_reason"] = "sample_budget_exhausted"`.
- Optuna bridge:
  - Call `trial.report` with partial metrics and consumed examples at exhaustion.
  - Raise `optuna.TrialPruned` to stop the Optuna trial.
  - Prevent Optuna from requeueing the same config when exhaustion occurs due to global cap (mark as `should_retry=False`).
- Update `StopConditionManager` to rely on lease totals (metadata override) without double counting.

### 5. Telemetry & Observability
- Introduce `BudgetMetrics` dataclass for logging and telemetry emission:
  - `total_budget`, `consumed`, `remaining`, `wasted`, `efficiency`.
- Emit log entries on:
  - Lease creation and finalisation (debug level).
  - Budget exhaustion events (info level with summary metrics).
- Record aggregated metrics in orchestrator result summary for CLI display.

### 6. Configuration & CLI
- Validate combinations of `max_trials`, `max_examples`, `max_total_examples` via helper `validate_budget_config`.
- Extend CLI help text to describe intra-trial enforcement behaviour.
- Expose runtime toggle for `samples_include_pruned` (if not already surfaced).

## Testing Strategy

### Unit Tests
1. `tests/core/test_sample_budget_manager.py`
   - Creation, concurrent `try_take`, rollback/finalise semantics.
   - Fairness guard (minimum allocation).
2. `tests/evaluators/test_local_evaluator_budget.py`
   - Sequential exhaustion (`max_workers=1`).
   - Parallel exhaustion (`max_workers>1`).
   - Detailed mode metadata propagation.
   - Async cancellation handling.
3. `tests/core/test_orchestrator_budget.py`
   - Lease acquisition and finalisation workflow.
   - Parallel batch allocation fairness.
   - Stop-condition triggering from lease metadata.

### Integration / End-to-End
1. CLI or orchestrator integration test verifying:
   - Combined `max_trials` + `max_total_examples` behaviour.
   - Optuna pruning path (mocked trial object).
2. Hybrid adapter smoke test ensuring lease metadata reaches result payloads.
3. Regression verifying legacy behaviour when `max_total_examples` is `None`.

### Performance / Stress (optional but recommended)
- Benchmark sequential vs parallel evaluation with budget enforcement enabled to quantify overhead and ensure no significant regression.

## Implementation Phases
1. **Phase 1 – Infrastructure (Day 1–2)**
   - Implement `SampleBudgetManager` and `SampleBudgetLease`.
   - Provide unit tests covering concurrent operations and edge cases.
2. **Phase 2 – Orchestrator Integration (Day 3–4)**
   - Wire manager into orchestrator runtime.
   - Update trial execution paths and metadata handling.
   - Add orchestrator-level tests.
3. **Phase 3 – Local Evaluator & Base Plumbing (Day 5–7)**
   - Update evaluator signatures, `_evaluate_batch`, and LocalEvaluator logic.
   - Introduce queue-based async execution.
   - Add evaluator tests for sequential/parallel/detailed flows.
4. **Phase 4 – Optimizer & Stop Condition Alignment (Day 8)**
   - Update Optuna integration and stop condition usage.
   - Add Optuna pruning tests.
5. **Phase 5 – Hybrid & Remote Support (Day 9–10)**
   - Extend hybrid adapter, remote DTOs, and backend client calls.
   - Provide integration tests (mocked backend).
6. **Phase 6 – Telemetry, Validation, Documentation (Day 11)**
   - Add budget metrics logging, configuration validation helpers.
   - Update CLI/docs to describe behaviour.
7. **Phase 7 – Regression & Polish (Day 12)**
   - Run full test suite, address performance issues.
   - Final documentation touch-up.

## Risks & Mitigations
- **Async cancellation complexity**: introduce helper to cancel and await outstanding tasks; cover in tests.
- **Fairness/starvation**: start with dynamic acquisition but ensure orchestrator refuses to launch more trials than remaining budget permits.
- **Remote alignment timing**: guard remote enforcement behind capability flag; fall back to coarse-grained stop if remote does not yet support intra-trial leases.
- **Performance overhead**: measure and cache frequently accessed lease state; use `RLock` sparingly.

## Deliverables Checklist
- [ ] `SampleBudgetManager` + tests.
- [ ] Orchestrator wiring with lease lifecycle.
- [ ] Evaluator budget enforcement (sequential + async) + tests.
- [ ] Optuna pruning alignment + tests.
- [ ] Hybrid/remote propagation + mocked tests.
- [ ] Telemetry/logging + configuration validation.
- [ ] Documentation updates (README, CLI help, architecture doc).
- [ ] Full test suite green.
