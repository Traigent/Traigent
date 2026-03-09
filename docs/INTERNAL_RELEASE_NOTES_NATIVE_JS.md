# Internal Release Notes: JS Optimization Parity

## Scope

This release aligns the JS SDK more closely with the sibling Python SDK where
the optimization contract is shared.

## SDK Changes

- Root exports include `optimize`, `param`, `getOptimizationSpec`,
  `toHybridConfigSpace`, `loadTvlSpec`, and `parseTvlSpec`.
- Wrapped functions support native
  `.optimize({ mode: "native", algorithm, maxTrials, randomSeed, timeoutMs, trialConcurrency, plateau, checkpoint, signal })`.
- Wrapped functions also support backend-guided
  `.optimize({ algorithm: "optuna", maxTrials, backendUrl, apiKey, timeoutMs, requestTimeoutMs, signal })`.

## Native Optimization

- Node-only execution
- `grid`, `random`, and sequential `bayesian`
- log-scale `float` and `int` parameters
- budget enforcement from numeric `metrics.cost`
- `stopReason: "timeout" | "error" | "cancelled" | "plateau"`
- wrapper-local `applyBestConfig()` / `currentConfig()`

## Hybrid Optimization

- local JS trial execution with backend-driven configuration suggestions
- backend session orchestration via `/sessions`, `/next-trial`, `/results`, and `/finalize`
- env-backed backend discovery through `TRAIGENT_BACKEND_URL` /
  `TRAIGENT_API_URL` and `TRAIGENT_API_KEY`
- snake_case transport serialization for hybrid `optimizationStrategy` keys
- backend finalization fallback for `bestConfig` / `bestMetrics`
- fast compatibility failure when a backend still exposes the legacy TraiGent
  `/sessions` contract instead of the typed interactive one
- weighted and banded objectives
- conditional parameters with default fallbacks
- structural constraints and derived constraints
- spec-level `maxTrials`, `maxCostUsd`, and `maxWallclockMs`
- promotion-policy metadata transport

## Intentional Limits

- `toHybridConfigSpace()` still rejects conditional parameters because the legacy
  hybrid wire format cannot express them.
- Promotion policy is currently transport-only metadata. It is parsed and sent,
  but not behaviorally enforced yet.
- No native in-process Optuna-family algorithms or example-level concurrency.
- `budget.maxCostUsd` requires numeric `metrics.cost` on every trial.
- Hybrid mode still rejects native-only runtime options such as
  `trialConcurrency`, `plateau`, `checkpoint`, and `randomSeed`.

## Release Validation

```bash
npm run ci
npm run test:cross-sdk
npm run smoke:example
npm run smoke:hybrid-live
npm run benchmark:cross-sdk
npm run build
npm pack --dry-run
```

`smoke:hybrid-live` requires `TRAIGENT_BACKEND_URL` or `TRAIGENT_API_URL` plus
`TRAIGENT_API_KEY`.
