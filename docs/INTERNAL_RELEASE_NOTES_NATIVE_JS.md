# Internal Release Notes: Native JS Optimization

## Scope

This release hardens the current native JS optimization surface in `@traigent/sdk` and the hybrid demo consumer that validates migration from JSON tunables to code-defined specs.

## SDK Changes

- Root exports now include `optimize`, `param`, `getOptimizationSpec`, and `toHybridConfigSpace`.
- Wrapped functions can expose native `.optimize({ algorithm, maxTrials, randomSeed, timeoutMs, trialConcurrency, plateau, checkpoint })`.
- Native optimization currently supports:
  - Node-only execution
  - `grid`, `random`, and sequential `bayesian`
  - spec-driven objectives and config spaces
  - budget enforcement from numeric `metrics.cost`

## Demo / Migration State

- `child-age-agent-a` is code-defined and serialized with `toHybridConfigSpace()`.
- `child-age-agent-b` remains JSON-backed for one migration cycle.
- Startup fails when spec-backed and JSON-backed config spaces disagree.
- Runtime config is bound only inside capability execution through `TrialContext.run(...)`.

## Intentional v1 Limits

- Log-scale grid search requires a multiplicative `step > 1`.
- `trialConcurrency` is currently limited to `grid` and `random`.
- No worker pools, Optuna-family algorithms, example-level concurrency, or hybrid API orchestration in the native runner.
- `budget.maxCostUsd` requires numeric `metrics.cost` on every trial.

## Release Validation

Run in `traigent-js`:

```bash
npm test
npm run smoke:example
npm run build
npm pack --dry-run
```

Run in `JS-Mastra-APIs-Validation`:

```bash
npm test
DEMO_MOCK_GENERATION=1 npm run api:dev
npm run api:test
```

## Branch Isolation

Cut the release from an isolated branch or PR that excludes unrelated local SDK work already present in the `traigent-js` worktree.
