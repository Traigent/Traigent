# @traigent/sdk

TypeScript SDK for Traigent optimization in JavaScript and TypeScript.

## Supported Flows

- Hybrid spec authoring for services that expose Traigent-compatible `/config-space`, `/execute`, and `/evaluate` routes.
- Native Node optimization with `optimize(spec)(trialFn)` and `await wrapped.optimize(...)`.
- Legacy Python-to-Node bridge execution through the CLI runner.

## Installation

```bash
npm install @traigent/sdk
```

## Hybrid Mode Authoring

Use `optimize(...)` and `toHybridConfigSpace(...)` to define tunables in code while keeping the existing hybrid API wire format unchanged.

```ts
import {
  getTrialParam,
  optimize,
  param,
  toHybridConfigSpace,
} from '@traigent/sdk';

export const childAgeTrial = optimize({
  configurationSpace: {
    model: param.enum(['gpt-4o-mini', 'gpt-4o']),
    temperature: param.float({ min: 0, max: 1, scale: 'linear' }),
    max_retries: param.int({ min: 0, max: 3, scale: 'linear' }),
  },
  objectives: ['accuracy', 'cost'],
})(async () => ({
  metrics: {
    accuracy: 0,
    cost: 0,
  },
}));

export const configSpace = toHybridConfigSpace(childAgeTrial);

export function resolveRuntimeConfig() {
  return {
    model: getTrialParam('model', 'gpt-4o-mini'),
    temperature: getTrialParam('temperature', 0.2),
    maxRetries: getTrialParam('max_retries', 0),
  };
}
```

## Native Node Optimization

Native optimization runs in-process in Node and reuses the same spec metadata.

```ts
import { optimize, param } from '@traigent/sdk';

const evaluatePrompt = optimize({
  configurationSpace: {
    model: param.enum(['cheap', 'accurate']),
    temperature: param.float({ min: 0, max: 0.5, step: 0.5, scale: 'linear' }),
  },
  objectives: ['accuracy', 'cost'],
  budget: {
    maxCostUsd: 2,
  },
  evaluation: {
    data: [{ id: 1 }, { id: 2 }],
  },
})(async (trialConfig) => {
  const model = String(trialConfig.config.model);
  const accuracy = model === 'accurate' ? 0.95 : 0.72;
  const cost = model === 'accurate' ? 0.4 : 0.1;

  return {
    metrics: {
      accuracy,
      cost,
      latency: model === 'accurate' ? 1.2 : 0.6,
    },
  };
});

const result = await evaluatePrompt.optimize({
  algorithm: 'grid',
  maxTrials: 10,
  timeoutMs: 5_000,
});

console.log(result.bestConfig);
console.log(result.bestMetrics);
evaluatePrompt.applyBestConfig(result);
console.log(evaluatePrompt.currentConfig());
```

See [`examples/native-optimization.mjs`](./examples/native-optimization.mjs) for the runnable smoke example.

The wrapped function must satisfy the JS trial contract:

- Input: `TrialConfig`
- Output: `{ metrics, metadata?, duration? }`

## Objectives and Parameters

Built-in objective strings:

- `accuracy` -> `maximize`
- `cost` -> `minimize`
- `latency` -> `minimize`

Any other metric must use an explicit object:

```ts
objectives: [{ metric: 'quality_score', direction: 'maximize', weight: 1 }]
```

Parameter helpers:

- `param.enum(values, { conditions?, default? })`
- `param.float({ min, max, scale, step?, conditions?, default? })`
- `param.int({ min, max, scale, step?, conditions?, default? })`

## Trial Context Access

Use `TrialContext`, `getTrialConfig()`, and `getTrialParam()` inside a bound trial or capability execution path.

`TrialContext` is only guaranteed inside:

- native `wrapped.optimize(...)` trial execution
- host-managed capability execution where your app calls `TrialContext.run(...)`

## Current Native v1 Limits

- Native optimization is Node-only.
- Supported algorithms are `grid`, `random`, and sequential `bayesian`.
- `evaluation.data` or `evaluation.loadData` is required for `.optimize()`.
- `budget.maxCostUsd` is enforced only from numeric `metrics.cost`.
- `timeoutMs`, `trialConcurrency`, `plateau`, `checkpoint`, and wrapper-local `applyBestConfig()` / `currentConfig()` are supported in native mode.
- Log-scale grid search requires a multiplicative `step > 1`.
- Conditional parameters support simple equality conditions with a required `default` fallback in native optimization.
- `toHybridConfigSpace()` currently rejects conditional parameters because the existing hybrid wire format does not encode them.
- `trialConcurrency` is currently limited to `grid` and `random`.
- Worker pools, Optuna-family optimizers, example-level concurrency, and hybrid API orchestration are still out of scope.

## Cross-SDK Validation

Internal validation:

```bash
npm test
npm run test:cross-sdk
npm run smoke:example
npm run benchmark:cross-sdk
npm run build
npm pack --dry-run
```

See [docs/NATIVE_JS_PARITY_MATRIX.md](./docs/NATIVE_JS_PARITY_MATRIX.md) and [docs/CROSS_SDK_VALIDATION.md](./docs/CROSS_SDK_VALIDATION.md) for the current parity and methodology.

## Requirements

- Node.js >= 18.0.0

## License

MIT
