# Client Code Guide

`@traigent/sdk` supports three distinct JS flows:

- spec authoring for existing Traigent-compatible hybrid API services with `toHybridConfigSpace()`
- backend-guided hybrid optimization with `wrapped.optimize({ algorithm: "optuna", ... })`
- native in-process optimization in Node with `wrapped.optimize({ mode: "native", algorithm: "grid" | "random" | "bayesian" })`

## Native Node Optimization

Use native mode when the optimizer should run inside the Node process or when
the spec is limited to native-safe features.

```ts
import { optimize, param } from "@traigent/sdk";

const runTrial = optimize({
  configurationSpace: {
    model: param.enum(["cheap", "accurate"]),
    temperature: param.float({ min: 0, max: 0.5, step: 0.25 }),
  },
  objectives: ["accuracy", "cost"],
  evaluation: {
    data: [{ id: 1 }, { id: 2 }],
  },
})(async (trialConfig) => ({
  metrics: {
    accuracy: trialConfig.config.model === "accurate" ? 0.92 : 0.75,
    cost: trialConfig.config.model === "accurate" ? 0.4 : 0.1,
  },
}));

const result = await runTrial.optimize({
  mode: "native",
  algorithm: "grid",
  maxTrials: 10,
});
```

## Backend-Guided Hybrid Optimization

Use hybrid mode when configuration selection should come from the Traigent
backend while trial execution stays local in JS. This is the default if you do
not pass `mode`.

```ts
import { optimize, param } from "@traigent/sdk";

const runTrial = optimize({
  configurationSpace: {
    model: param.enum(["gpt-4o-mini", "gpt-4o"]),
    temperature: param.float({ min: 0, max: 1, step: 0.2 }),
  },
  objectives: [
    { metric: "accuracy", direction: "maximize", weight: 2 },
    { metric: "cost", direction: "minimize", weight: 1 },
  ],
  evaluation: {
    data: [{ id: 1 }, { id: 2 }, { id: 3 }],
  },
})(async (trialConfig) => ({
  metrics: {
    accuracy: 0.85,
    cost: 0.2,
  },
}));

const result = await runTrial.optimize({
  algorithm: "optuna",
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

- The legacy TraiGent `/api/v1/sessions` route that expects
  `problem_statement`, `dataset`, `search_space`, and `optimization_config` is
  not this contract.
- The separate `/api/v1/hybrid/sessions` route in `TraigentBackend` is an IRT
  round-based workflow, not the Optuna-style single-trial session flow used
  here.
- `wrapped.optimize({ algorithm: "optuna" })` fails fast against the legacy
  `/sessions` contract instead of surfacing a vague HTTP 400 later in the run.

`backendUrl` may be either the backend origin or the `/api/v1` base URL.
Resolution order is:

1. `backendUrl`
2. `TRAIGENT_BACKEND_URL`
3. `TRAIGENT_API_URL`

API key resolution order is:

1. `apiKey`
2. `TRAIGENT_API_KEY`

For a real backend session smoke run, use `npm run smoke:hybrid-live` after
setting `TRAIGENT_BACKEND_URL` or `TRAIGENT_API_URL` plus `TRAIGENT_API_KEY`.
Hybrid `optimizationStrategy` options stay camelCase in JS and are serialized to
snake_case on the backend request.

## Hybrid Support Surface

- typed backend session API on `/api/v1/sessions`
- `algorithm: "optuna"`
- weighted objectives
- banded objectives
- conditional parameters with required defaults
- structural constraints
- derived constraints
- spec-level `maxTrials`, `maxCostUsd`, and `maxWallclockMs`
- promotion-policy metadata transport
- explicit top-level backend `stop_reason`

## Native Limits

- weighted objectives are rejected
- conditional parameters are rejected
- structural and derived constraints are rejected
- spec-level `budget.maxTrials` and `budget.maxWallclockMs` are rejected
- `spec.evaluation.data` or `spec.evaluation.loadData()` is still required
