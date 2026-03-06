# Internal Release Notes: Native JS Optimization Phase 1

## Scope

This release isolates the phase-1 native JS parity slice for `@traigent/sdk`.

## SDK Changes

- Root exports continue to include `optimize`, `param`, `getOptimizationSpec`, and `toHybridConfigSpace`.
- Wrapped functions now support native `.optimize({ algorithm, maxTrials, randomSeed, timeoutMs })`.
- Native optimization currently supports:
  - Node-only execution
  - `grid`, `random`, and sequential `bayesian`
  - log-scale `float` and `int` parameters
  - budget enforcement from numeric `metrics.cost`
  - `stopReason: 'timeout' | 'error'`

## Cross-SDK Validation

- Python-owned oracle fixtures validate deterministic grid, seeded random, and budget cutoff behavior.
- JS reports async scheduler performance against Python with a shared synthetic workload.
- The lightweight Python oracle exporter is stdlib-only; full Python Bayesian oracle parity remains deferred in this environment.

## Intentional Phase-1 Limits

- No public `trialConcurrency`, `signal`, `plateau`, or `checkpoint` options.
- No wrapper-local `applyBestConfig()` or `currentConfig()`.
- No Optuna-family algorithms, example-level concurrency, or cloud orchestration.
- `budget.maxCostUsd` requires numeric `metrics.cost` on every trial.

## Release Validation

```bash
npm test
npm run test:cross-sdk
npm run smoke:example
npm run benchmark:cross-sdk
npm run build
npm pack --dry-run
```
