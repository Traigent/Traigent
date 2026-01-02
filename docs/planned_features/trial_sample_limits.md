# Trial Sample Limit Control

## Background

- **Current state:** the SDK exposes `max_trials` (total configurations explored) and `max_examples` (per-trial dataset cap).
- **Gap:** reviewers and CI need a **global sample budget** (e.g., stop after 50k evaluated examples across all trials) to manage compute/network costs. Today there is no enforcement across trials.
- **Observed issues:** multi-objective runs in cloud and edge modes can blow past expected usage when optimizers propose many low-batch configurations. Governance tooling only sees post-hoc totals.

## Goal

Introduce a first-class **`max_total_examples`** limit that:

1. Stops the orchestrator when the cumulative `examples_attempted` reaches the cap.
2. Propagates through API/CLI/OptiGen so every execution path honours it.
3. Reports the requested cap and consumed total in telemetry & backend session metadata.

## Scope & Interfaces

- **Sampling abstraction:** introduce `traigent.core.samplers.BaseSampler` with a canonical `sample(**kwargs)` method returning a sample or `None`. Register implementations (starting with `RandomSampler`) through `SamplerFactory` so evaluators/optimizers can request consistent sampling strategies across local, hybrid, synchronous, and async flows.
  - Each sampler exposes `clone()` so parallel executors can create per-worker instances without sharing mutable state.
- **Decorator / SDK surface:** new optional kwarg `max_total_examples` (alias `max_examples_total`) alongside existing `max_trials`, `max_examples`.
- **Runtime override helpers:** `traigent.override_config(max_total_examples=...)`, CLI `--max-total-examples`, OptiGen integration knobs.
- **Stop conditions:** add `MaxSamplesStopCondition` (mirrors `MaxTrialsStopCondition`), summing `examples_attempted` (uses trial metrics -> metadata fallback). Honour `include_pruned` gating.
- **Orchestrator:** accept/store the new limit, wire it into `StopConditionManager`, clamp parallel batch sizes based on remaining sample budget, set `_stop_reason="max_samples"` as needed.
- **Trial accounting:** ensure `examples_attempted` is always captured via `trial_result_factory` for success, failure, and pruned trials to keep the counter accurate.
- **Backend/analytics:** include the new limit and final totals in `BackendSessionManager.create_session` metadata, session DTOs, optimization logger checkpoints, and summary payloads.
- **Cloud/hybrid (OptiGen) flows:** propagate the limit when creating remote sessions and make remote schedulers respect it. For local fallback, rely on orchestrator enforcement.

## Implementation Steps

1. **API & decorator plumbing – In progress**
   - Decorator + CLI now accept `max_total_examples` / `samples_include_pruned` and forward them to `OptimizedFunction`.
   - Remaining work: surface the same controls through OptiGen/cloud DTOs.
2. **Stop-condition engine – Completed**
   - `MaxSamplesStopCondition` added with support for pruned-trial inclusion/exclusion.
   - `StopConditionManager` now manages the new condition and propagates updates.
   - Sampler hooks remain on the roadmap for richer strategies.
3. **Orchestrator wiring – Completed**
   - `OptimizationOrchestrator` tracks `max_total_examples`, clamps parallel batches, and stops with `max_samples_reached` when the budget is exhausted.
   - Remaining work: ensure backend telemetry/reporting captures consumed vs. requested samples.
4. **Telemetry / backend**
   - Pass the cap to `BackendSessionManager.create_session` and cloud DTOs (e.g., `traigent/cloud/dtos.py`, `backend_client.py`).
   - Emit requested vs. consumed samples in the final `OptimizationResult.metadata` and analytics logs.
5. **Docs & discoverability**
   - Update public docs, CLI help, and case-study READMEs to describe the new option and its relation to `max_trials` / `max_examples`.
   - Document sampler configuration (e.g., YAML/kwargs: `sampler={"type": "random", "params": {"sample_limit": 1000}}`) and how evaluators/optimizers can request sampler instances in local vs. hybrid runs.
6. **Testing**
   - Unit tests for the stop condition (include/exclude pruned, missing metrics, exact cap hit).
   - Orchestrator tests for early stops, parallel scheduling, failure/pruned accounting.
   - CLI & OptiGen integration tests ensuring the flag is respected.
   - Smoke runs in edge + cloud modes capturing stop reasons and telemetry.

## Risks & Mitigations

- **Incomplete metrics:** if `examples_attempted` is missing from evaluators, the stop condition should fall back to metadata and raise a clear error if still unavailable.
- **Parallel overshoot:** reduce scheduled trial batches when remaining sample budget is smaller than `parallel_config.trial_concurrency`.
- **Backwards compatibility:** default behaviour remains unchanged when the new option is omitted. Document the distinction between per-trial `max_examples` and global `max_total_examples`.
- **Sampler consistency:** guarantee sampler instances are isolated per worker/batch so parallel trials do not share mutable state, and surface factory hooks for custom implementations.

## Status

- ✅ Plan drafted (this document).
- ✅ Sampler abstraction scaffolded (`BaseSampler`, `RandomSampler`, `SamplerFactory`).
- ✅ MaxSamplesStopCondition implemented and wired into `OptimizationOrchestrator` (sample budget tracking plus stop reason propagation).
- ⚙️ API changes (OptiGen/cloud propagation pending)
- ✅ Orchestrator wiring
- ☐ Telemetry/back-end updates
- ☐ Documentation & tests

Tracking: https://linear.app/traigent/ISSUE-0000 (placeholder)
