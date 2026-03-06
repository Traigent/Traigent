# Internal Release Notes: Native JS Optimization

## Scope

This release extends the native JS optimization surface in `@traigent/sdk` beyond the phase-1 slice.

## SDK Changes

- Root exports continue to include `optimize`, `param`, `getOptimizationSpec`, and `toHybridConfigSpace`.
- Wrapped functions now support native `.optimize({ algorithm, maxTrials, randomSeed, timeoutMs, trialConcurrency, plateau, checkpoint, signal })`.
- Native optimization currently supports:
  - Node-only execution
  - `grid`, `random`, and sequential `bayesian`
  - log-scale `float` and `int` parameters
  - budget enforcement from numeric `metrics.cost`
  - `stopReason: 'timeout' | 'error' | 'cancelled' | 'plateau'`
  - wrapper-local `applyBestConfig()` / `currentConfig()`

## Cross-SDK Validation

- Python-owned oracle fixtures validate deterministic grid, seeded random, and budget cutoff behavior.
- JS reports async optimization scheduling against Python with a shared synthetic workload.
- The lightweight Python oracle exporter is stdlib-only; full Python Bayesian oracle parity remains deferred in this environment.

## Intentional v1 Limits

- `trialConcurrency` is currently limited to `grid` and `random`.
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
