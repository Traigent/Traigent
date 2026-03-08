# Client Code Guide

`@traigent/sdk` now supports three distinct JS flows:

- spec authoring for existing Traigent-compatible hybrid API services with `toHybridConfigSpace()`
- native in-process optimization in Node with `wrapped.optimize({ algorithm: 'grid' | 'random' | 'bayesian' })`
- backend-guided hybrid optimization with `wrapped.optimize({ mode: 'hybrid', algorithm: 'optuna', ... })`

## Native Node Optimization

Use native mode when the optimizer should run inside the Node process.

```ts
import { optimize, param } from '@traigent/sdk';

const runTrial = optimize({
  configurationSpace: {
    model: param.enum(['cheap', 'accurate']),
    temperature: param.float({ min: 0, max: 0.5, step: 0.25 }),
  },
  objectives: ['accuracy', 'cost'],
  evaluation: {
    data: [{ id: 1 }, { id: 2 }],
  },
})(async (trialConfig) => ({
  metrics: {
    accuracy: trialConfig.config.model === 'accurate' ? 0.92 : 0.75,
    cost: trialConfig.config.model === 'accurate' ? 0.4 : 0.1,
  },
}));

const result = await runTrial.optimize({
  algorithm: 'grid',
  maxTrials: 10,
});
```

## Backend-Guided Hybrid Optimization

Use hybrid mode when configuration selection should come from the Traigent backend while trial execution stays local in JS.

```ts
import { optimize, param } from '@traigent/sdk';

const runTrial = optimize({
  configurationSpace: {
    model: param.enum(['gpt-4o-mini', 'gpt-4o']),
    temperature: param.float({ min: 0, max: 1, step: 0.2 }),
  },
  objectives: ['accuracy', 'cost'],
  evaluation: {
    data: [{ id: 1 }, { id: 2 }, { id: 3 }],
  },
})(async (trialConfig) => {
  return {
    metrics: {
      accuracy: 0.85,
      cost: 0.2,
    },
  };
});

const result = await runTrial.optimize({
  mode: 'hybrid',
  algorithm: 'optuna',
  maxTrials: 12,
  backendUrl: process.env.TRAIGENT_BACKEND_URL,
  apiKey: process.env.TRAIGENT_API_KEY,
  timeoutMs: 5_000,
});
```

Hybrid mode uses the backend interactive session API:

- `POST /sessions`
- `POST /sessions/{session_id}/next-trial`
- `POST /sessions/{session_id}/results`
- `POST /sessions/{session_id}/finalize`

Compatibility note:

- The legacy TraiGent `/api/v1/sessions` route that expects `problem_statement`, `dataset`, `search_space`, and `optimization_config` is not this contract.
- The separate `/api/v1/hybrid/sessions` route in `TraigentBackend` is an IRT round-based workflow, not the Optuna-style single-trial session flow used here.
- `wrapped.optimize({ mode: 'hybrid', algorithm: 'optuna' })` now fails fast against the legacy `/sessions` contract instead of surfacing a vague HTTP 400 later in the run.

`backendUrl` may be either the backend origin or the `/api/v1` base URL. Resolution order is:

1. `backendUrl`
2. `TRAIGENT_BACKEND_URL`
3. `TRAIGENT_API_URL`

API key resolution order is:

1. `apiKey`
2. `TRAIGENT_API_KEY`

For a real backend session smoke run, use `npm run smoke:hybrid-live` after setting `TRAIGENT_BACKEND_URL` or `TRAIGENT_API_URL` plus `TRAIGENT_API_KEY`.
For full local-vs-hybrid live LangChain runs with `TRAIGENT_OFFLINE_MODE=false`, use `npm run report:langchain-e2e:online -- --env-file ../Traigent/walkthrough/examples/real/.env --max-trials 2 --dataset-size 2`. That command delegates to the neutral cross-SDK harness in `../traigent-cross-sdk-benchmarks`.
Hybrid `optimizationStrategy` options stay camelCase in JS and are serialized to snake_case on the backend request.

## Hybrid v1 Limits

- only `mode: 'hybrid'` + `algorithm: 'optuna'`
- requires a backend that exposes the typed interactive session API on `/api/v1/sessions`
- no client-side resume
- no `trialConcurrency`, `plateau`, `checkpoint`, or `randomSeed`
- explicit objective objects must match backend direction inference and cannot be weighted
- conditional parameters remain native-only
- `spec.evaluation.data` or `spec.evaluation.loadData()` is still required so the SDK can derive dataset size
