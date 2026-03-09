# @traigent/sdk

TypeScript SDK for Traigent optimization in JavaScript and TypeScript.

## Supported Flows

- Hybrid spec authoring for services that expose Traigent-compatible `/config-space`, `/execute`, and `/evaluate` routes.
- Backend-guided hybrid optimization as the default `.optimize(...)` execution path.
- Explicit native Node optimization with `wrapped.optimize({ mode: "native", ... })`.
- Legacy Python-to-Node bridge execution through the CLI runner.

## Installation

```bash
npm install @traigent/sdk
```

## Hybrid Mode Authoring

Use `optimize(...)` and `toHybridConfigSpace(...)` to define tunables in code while
keeping the existing hybrid API wire format unchanged.

```ts
import {
  getTrialParam,
  optimize,
  param,
  toHybridConfigSpace,
} from "@traigent/sdk";

export const childAgeTrial = optimize({
  configurationSpace: {
    model: param.enum(["gpt-4o-mini", "gpt-4o"]),
    temperature: param.float({ min: 0, max: 1, scale: "linear" }),
    maxRetries: param.int({ min: 0, max: 3, scale: "linear" }),
  },
  objectives: ["accuracy", "cost"],
})(async () => ({
  metrics: {
    accuracy: 0,
    cost: 0,
  },
}));

export const configSpace = toHybridConfigSpace(childAgeTrial);

export function resolveRuntimeConfig() {
  return {
    model: getTrialParam("model", "gpt-4o-mini"),
    temperature: getTrialParam("temperature", 0.2),
    maxRetries: getTrialParam("maxRetries", 0),
  };
}
```

## Native Node Optimization

Native optimization runs in-process in Node and reuses the same spec metadata.
Use it when you want fully local execution or when the spec is limited to
native-safe features.

```ts
import { optimize, param } from "@traigent/sdk";

const evaluatePrompt = optimize({
  configurationSpace: {
    model: param.enum(["cheap", "accurate"]),
    temperature: param.float({
      min: 0,
      max: 0.5,
      step: 0.5,
      scale: "linear",
    }),
  },
  objectives: ["accuracy", "cost"],
  budget: {
    maxCostUsd: 2,
  },
  evaluation: {
    data: [{ id: 1 }, { id: 2 }],
  },
})(async (trialConfig) => {
  const model = String(trialConfig.config.model);
  const accuracy = model === "accurate" ? 0.95 : 0.72;
  const cost = model === "accurate" ? 0.4 : 0.1;

  return {
    metrics: {
      accuracy,
      cost,
      latency: model === "accurate" ? 1.2 : 0.6,
    },
  };
});

const result = await evaluatePrompt.optimize({
  mode: "native",
  algorithm: "grid",
  maxTrials: 10,
  timeoutMs: 5_000,
});

console.log(result.bestConfig);
console.log(result.bestMetrics);
evaluatePrompt.applyBestConfig(result);
console.log(evaluatePrompt.currentConfig());
```

See [`examples/native-optimization.mjs`](./examples/native-optimization.mjs) for
the runnable smoke example.

## Backend-Guided Hybrid Optimization

Hybrid optimization keeps trial execution local in JS while the Traigent backend
drives Optuna-style configuration suggestions through the session API. This is
the default when you omit `mode`.

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
  constraints: {
    structural: [
      {
        when: 'params.model == "gpt-4o"',
        then: "params.temperature <= 0.6",
      },
    ],
  },
  evaluation: {
    data: [{ id: 1 }, { id: 2 }, { id: 3 }],
  },
})(async (trialConfig) => ({
  metrics: {
    accuracy: trialConfig.config.model === "gpt-4o" ? 0.9 : 0.82,
    cost: trialConfig.config.model === "gpt-4o" ? 0.3 : 0.08,
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

Resolution order:

- `backendUrl`
- `TRAIGENT_BACKEND_URL`
- `TRAIGENT_API_URL`

API key resolution order:

- `apiKey`
- `TRAIGENT_API_KEY`

Hybrid transport keeps JS options camelCase at the public API and serializes
`optimizationStrategy` keys to snake_case for the backend session request.

The hybrid session client talks to:

- `POST /sessions`
- `POST /sessions/{session_id}/next-trial`
- `POST /sessions/{session_id}/results`
- `POST /sessions/{session_id}/finalize`

See [`examples/hybrid-optuna.mjs`](./examples/hybrid-optuna.mjs) for a runnable
example that expects backend env vars.
Use `npm run smoke:hybrid-live` to run a real backend session smoke test once
`TRAIGENT_BACKEND_URL` or `TRAIGENT_API_URL` and `TRAIGENT_API_KEY` are
configured.

Compatibility note:

- This client expects a typed interactive session contract on `/api/v1/sessions`.
- The older TraiGent `/api/v1/sessions` contract that expects
  `problem_statement`, `dataset`, `search_space`, and `optimization_config` is
  not compatible and fails fast with a validation error.
- The separate `/api/v1/hybrid/sessions` API in `TraigentBackend` is an IRT
  round-based service, not the Optuna-style one-trial-at-a-time contract used by
  this JS client.

The wrapped function must satisfy the JS trial contract:

- Input: `TrialConfig`
- Output: `{ metrics, metadata?, duration? }`

## Objectives, Constraints, and Parameters

Built-in objective strings:

- `accuracy` -> `maximize`
- `cost` -> `minimize`
- `latency` -> `minimize`

Any other metric must use an explicit object:

```ts
objectives: [{ metric: "quality_score", direction: "maximize", weight: 1 }]
```

Hybrid mode also supports:

- weighted objectives
- banded objectives
- structural constraints
- derived constraints
- budgets for `maxCostUsd`, `maxTrials`, and `maxWallclockMs`
- conditional parameters with required default fallbacks
- TVL-derived promotion-policy metadata

Parameter helpers:

- `param.enum(values, { conditions?, default? })`
- `param.float({ min, max, scale, step?, conditions?, default? })`
- `param.int({ min, max, scale, step?, conditions?, default? })`
- `param.bool({ conditions?, default? })`

## Trial Context Access

Use `TrialContext`, `getTrialConfig()`, and `getTrialParam()` inside a bound
trial or capability execution path.

`TrialContext` is only guaranteed inside:

- native `wrapped.optimize(...)` trial execution
- host-managed capability execution where your app calls `TrialContext.run(...)`
- backend-guided hybrid trial execution after the SDK binds the received
  suggestion config

## Current Native Limits

- Native optimization is Node-only.
- Supported algorithms are `grid`, `random`, and sequential `bayesian`.
- `evaluation.data` or `evaluation.loadData` is required for `.optimize()`.
- `budget.maxCostUsd` is enforced only from numeric `metrics.cost`.
- `timeoutMs`, `trialConcurrency`, `plateau`, `checkpoint`, and wrapper-local
  `applyBestConfig()` / `currentConfig()` are supported in native mode.
- Log-scale grid search requires a multiplicative `step > 1`.
- Native mode intentionally rejects backend-only semantics such as weighted
  objectives, conditional parameters, structural constraints, derived
  constraints, and spec-level `maxTrials` / `maxWallclockMs`.
- `trialConcurrency` is currently limited to `grid` and `random`.
- Worker pools, example-level concurrency, and cloud orchestration are still out
  of scope.

## Current Hybrid Limits

- Hybrid optimization currently supports the typed session contract on
  `/api/v1/sessions` and `algorithm: "optuna"`.
- Hybrid optimization requires a backend that actually exposes the typed
  interactive session API on `/api/v1/sessions`.
- `spec.evaluation.data` or `spec.evaluation.loadData()` is required so the SDK
  can derive dataset size for backend sessions.
- Weighted objectives, banded objectives, conditional parameters, constraints,
  spec-level budgets, and promotion-policy metadata are supported through the
  typed backend contract.
- `trialConcurrency`, `plateau`, `checkpoint`, and `randomSeed` are native-only
  options.
- `toHybridConfigSpace()` remains the legacy hybrid wire-format serializer and
  still rejects conditional parameters because that older wire format cannot
  encode them.

## Documentation, Examples, and Walkthrough

- [Docs index](./docs/README.md)
- [Examples](./examples/README.md)
- [Walkthrough](./walkthrough/README.md)
