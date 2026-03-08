# Internal Release Notes: Native JS Optimization

## Scope

This release extends the native JS optimization surface in `@traigent/sdk` beyond the phase-1 slice.

## SDK Changes

- Root exports continue to include `optimize`, `param`, `getOptimizationSpec`, and `toHybridConfigSpace`.
- Wrapped functions now support native `.optimize({ algorithm, maxTrials, randomSeed, timeoutMs, trialConcurrency, plateau, checkpoint, signal })`.
- Wrapped functions also support backend-guided `.optimize({ mode: 'hybrid', algorithm: 'optuna', maxTrials, backendUrl, apiKey, timeoutMs, requestTimeoutMs, signal })`.
- Native optimization currently supports:
  - Node-only execution
  - `grid`, `random`, and sequential `bayesian`
  - log-scale `float` and `int` parameters
  - conditional parameters with equality conditions and required default fallback
  - budget enforcement from numeric `metrics.cost`
  - `stopReason: 'timeout' | 'error' | 'cancelled' | 'plateau'`
  - wrapper-local `applyBestConfig()` / `currentConfig()`
- Hybrid optimization currently supports:
  - local JS trial execution with backend-driven configuration suggestions
  - backend session orchestration via `/sessions`, `/next-trial`, `/results`, and `/finalize`
  - env-backed backend discovery through `TRAIGENT_BACKEND_URL` / `TRAIGENT_API_URL` and `TRAIGENT_API_KEY`
  - snake_case transport serialization for hybrid `optimizationStrategy` keys
  - backend finalization fallback for `bestConfig` / `bestMetrics`
  - fast compatibility failure when a backend still exposes the legacy TraiGent `/sessions` contract instead of the typed interactive one

## Cross-SDK Validation

- Python-owned oracle fixtures validate deterministic grid, conditional grid, seeded random, conditional random, and budget cutoff behavior.
- The new LangChain E2E parity lane runs the same fixture-backed chain in JS and Python from the neutral harness repo `../traigent-cross-sdk-benchmarks` and writes comparison reports under `../traigent-cross-sdk-benchmarks/tmp/langchain-e2e/`.
- JS reports async optimization scheduling against Python with a shared synthetic workload.
- The lightweight Python oracle exporter is stdlib-only; full Python Bayesian oracle parity remains deferred in this environment.

## Intentional v1 Limits

- `trialConcurrency` is currently limited to `grid` and `random`.
- Conditional parameters are native-only for now; `toHybridConfigSpace()` rejects them.
- No native in-process Optuna-family algorithms or example-level concurrency yet.
- `budget.maxCostUsd` requires numeric `metrics.cost` on every trial.
- Hybrid mode rejects weighted objectives, explicit direction overrides that do not match backend inference, conditional parameters, and native-only runtime options.
- The current local `TraigentBackend` codebase exposes two different session APIs:
  - `/api/v1/sessions` is still the older TraiGent grid/random contract and is incompatible with JS hybrid Optuna mode.
  - `/api/v1/hybrid/sessions` is an IRT round-based service and is also not the Optuna-style one-trial-at-a-time contract used by this client.
- Live backend validation is therefore blocked until the backend publishes the typed interactive session API expected by the JS hybrid client.

## Release Validation

```bash
npm test
npm run test:cross-sdk
npm run report:langchain-e2e
npm run report:langchain-e2e:online -- --env-file ../Traigent/walkthrough/examples/real/.env --max-trials 2 --dataset-size 2
npm run smoke:example
npm run smoke:hybrid-live
npm run benchmark:cross-sdk
npm run build
npm pack --dry-run
```

`smoke:hybrid-live` requires `TRAIGENT_BACKEND_URL` or `TRAIGENT_API_URL` plus `TRAIGENT_API_KEY`.
`report:langchain-e2e:online` forces `TRAIGENT_OFFLINE_MODE=false`, prefers OpenRouter when `OPENROUTER_API_KEY` is present, otherwise uses OpenAI, and reports native plus hybrid runs to the backend.
