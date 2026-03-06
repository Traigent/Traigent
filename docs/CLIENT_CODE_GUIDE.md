# Client Code Guide

This guide describes the current JavaScript and TypeScript integration surface for `@traigent/sdk`.

## Choose a Flow

Use the SDK in one of two current ways:

| Flow | When to use it | Main APIs |
| --- | --- | --- |
| Hybrid mode authoring | Your app exposes Traigent-compatible HTTP routes and needs a code-defined config space | `optimize`, `param`, `toHybridConfigSpace`, `TrialContext`, `getTrialParam`, `getTrialConfig` |
| Native Node optimization | Your optimization loop should run in-process in Node | `optimize`, `param`, `wrapped.optimize(...)` |

The older Python-orchestrated bridge flow still exists, but it is now documented as a legacy reference in [docs/REAL_MODE_SEQUENCE_FLOW.md](./REAL_MODE_SEQUENCE_FLOW.md).

## Flow 1: Hybrid Mode Authoring

In hybrid mode, the SDK is responsible for authoring the spec and exposing runtime config access. Your service is still responsible for HTTP routes, validation, dispatch, and error shaping.

### What you write

1. A code-defined optimization spec.
2. A host app that converts the spec with `toHybridConfigSpace()`.
3. Capability or business logic that runs inside `TrialContext.run(...)`.

### Minimal example

```ts
import {
  TrialContext,
  getTrialParam,
  optimize,
  param,
  toHybridConfigSpace,
} from '@traigent/sdk';

export const childAgeSpec = optimize({
  configurationSpace: {
    model: param.enum(['gpt-4o-mini', 'gpt-4o']),
    temperature: param.float({ min: 0, max: 1, scale: 'linear' }),
    system_prompt_version: param.enum(['v1', 'v2', 'v3']),
    max_retries: param.int({ min: 0, max: 3, scale: 'linear' }),
  },
  objectives: ['accuracy', 'cost'],
})(async () => ({
  metrics: {
    accuracy: 0,
    cost: 0,
  },
}));

export const hybridConfigSpace = toHybridConfigSpace(childAgeSpec);

export async function executeCapability(body: {
  request_id?: string;
  session_id?: string;
  config?: Record<string, unknown>;
  inputs: unknown[];
}) {
  const trialConfig = {
    trial_id: body.request_id ?? crypto.randomUUID(),
    trial_number: 0,
    experiment_run_id: body.session_id ?? body.request_id ?? 'hybrid-api',
    config: body.config ?? {},
    dataset_subset: {
      indices: [],
      total: body.inputs.length,
    },
  };

  return TrialContext.run(trialConfig, async () => {
    const model = getTrialParam('model', 'gpt-4o-mini');
    const temperature = getTrialParam('temperature', 0.2);

    return {
      model,
      temperature,
    };
  });
}
```

### Hybrid mode rules

- Keep `configurationSpace`, `objectives`, `budget`, and `evaluation` in camelCase.
- Keep your existing hybrid wire contract unchanged by serializing with `toHybridConfigSpace()`.
- Resolve runtime config with `getTrialParam()` and `getTrialConfig()` inside the execution path, not in top-level route handlers.
- If you temporarily support both code-defined specs and JSON config files, choose one source at startup and fail fast when they disagree.

## Flow 2: Native Node Optimization

In native mode, the wrapped function is the trial function and `.optimize(...)` runs the optimizer directly in Node.

### What you write

1. A spec with `optimize(...)`.
2. A trial function that accepts `TrialConfig`.
3. An evaluation dataset supplied via `evaluation.data` or `evaluation.loadData`.

### Minimal example

```ts
import { optimize, param } from '@traigent/sdk';

const runTrial = optimize({
  configurationSpace: {
    model: param.enum(['cheap', 'accurate']),
    temperature: param.float({ min: 0, max: 0.5, step: 0.5, scale: 'linear' }),
  },
  objectives: ['accuracy', 'cost'],
  budget: {
    maxCostUsd: 1,
  },
  evaluation: {
    data: [{ id: 'a' }, { id: 'b' }],
  },
})(async (trialConfig) => {
  const model = String(trialConfig.config.model);

  return {
    metrics: {
      accuracy: model === 'accurate' ? 0.95 : 0.7,
      cost: model === 'accurate' ? 0.4 : 0.1,
    },
    metadata: {
      evaluatedRows: trialConfig.dataset_subset.total,
    },
  };
});

const result = await runTrial.optimize({
  algorithm: 'grid',
  maxTrials: 12,
  randomSeed: 42,
  timeoutMs: 5_000,
  trialConcurrency: 2,
});

runTrial.applyBestConfig(result);
console.log(runTrial.currentConfig());
```

### Native mode rules

- The wrapped function must return `{ metrics, metadata?, duration? }`.
- Built-in objective strings are limited to `accuracy`, `cost`, and `latency`.
- Any other objective must use `{ metric, direction, weight? }`.
- `budget.maxCostUsd` only works when every trial returns numeric `metrics.cost`.

## Parameter and Objective Reference

### Parameters

```ts
param.enum(['gpt-4o-mini', 'gpt-4o']);
param.float({ min: 0, max: 1, scale: 'linear', step: 0.25 });
param.int({ min: 0, max: 3, scale: 'linear', step: 1 });
```

### Objectives

```ts
objectives: ['accuracy', 'cost'];
objectives: [{ metric: 'quality_score', direction: 'maximize' }];
```

## Runtime Access Rules

`getTrialConfig()` and `getTrialParam()` are the canonical runtime accessors.

Use them only when a trial context is active:

- inside `TrialContext.run(...)`
- inside a native `.optimize(...)` trial
- inside host-managed capability logic that has already bound the synthetic `TrialConfig`

Do not introduce a second global context mechanism for config lookup.

## Current v1 Limits

- Native optimization supports `grid`, `random`, and sequential `bayesian`.
- Native optimization is Node-only.
- `budget.maxCostUsd` still depends on numeric `metrics.cost`.
- Log-scale grid search requires a multiplicative `step > 1`.
- `trialConcurrency` is currently supported for `grid` and `random` only.
- Worker pools, Optuna-family optimizers, example-level concurrency, and hybrid API orchestration are not part of this v1 surface.

## Legacy Bridge Flow

If you still need the Python-to-Node bridge:

- Use the CLI runner with `npx traigent-js --module ./dist/trial.js --function runTrial`
- Treat [docs/REAL_MODE_SEQUENCE_FLOW.md](./REAL_MODE_SEQUENCE_FLOW.md) and the Mermaid diagrams as bridge-only reference material

That flow is no longer the primary entry point for the current JS-native or hybrid authoring APIs.
