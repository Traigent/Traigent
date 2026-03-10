# @traigent/sdk

TypeScript SDK for Traigent optimization in JavaScript and TypeScript.

## Supported Flows

- Native Node optimization of plain agent functions with `optimize(spec)(agentFn)` and `await wrapped.optimize(...)`.
- Hybrid spec authoring for services that expose Traigent-compatible `/config-space`, `/execute`, and `/evaluate` routes.
- Legacy Python-to-Node bridge execution through the CLI runner.

This checkout is native-first. It supports hybrid spec authoring via
`toHybridConfigSpace()`, but it does not implement backend-guided
`execution.mode = 'hybrid'`.

## Installation

```bash
npm install @traigent/sdk
```

## Native Node Optimization

The primary JS path is a plain agent function plus an `evaluation` block. The
SDK owns dataset iteration, scoring, aggregation, and optimization.

```ts
import { getTrialParam, optimize, param } from '@traigent/sdk';

const answerQuestion = optimize({
  configurationSpace: {
    model: param.enum(['cheap', 'accurate']),
    temperature: param.float({ min: 0, max: 0.5, step: 0.5, scale: 'linear' }),
  },
  objectives: ['accuracy', 'cost'],
  budget: {
    maxCostUsd: 2,
  },
  execution: {
    maxTotalExamples: 100,
    repsPerTrial: 3,
    repsAggregation: 'median',
  },
  evaluation: {
    data: [
      { input: 'What is 2+2?', output: '4' },
      { input: 'What is the capital of France?', output: 'Paris' },
    ],
    scoringFunction: (output, expectedOutput) =>
      output === expectedOutput ? 1 : 0,
    metricFunctions: {
      cost: (_output, _expectedOutput, _runtimeMetrics, row) =>
        row.input.includes('capital') ? 0.2 : 0.1,
    },
  },
})(async (question: string) => {
  const model = String(getTrialParam('model', 'cheap'));
  const temperature = Number(getTrialParam('temperature', 0));

  if (model === 'accurate' && temperature === 0) {
    return question.includes('capital') ? 'Paris' : '4';
  }

  return 'unknown';
});

const result = await answerQuestion.optimize({
  algorithm: 'grid',
  maxTrials: 10,
  timeoutMs: 5_000,
});

console.log(result.bestConfig);
console.log(result.bestMetrics);

answerQuestion.applyBestConfig(result);
console.log(await answerQuestion('What is 2+2?'));
console.log(answerQuestion.currentConfig());
```

See [examples/native-optimization.mjs](/home/nimrodbu/Traigent_enterprise/traigent-js/examples/native-optimization.mjs) for the runnable smoke example.

## Focused TVL Loading

This checkout now supports a focused native TVL subset through
`parseTvlSpec()` and `loadTvlSpec()`:

```ts
import { optimize, parseTvlSpec } from '@traigent/sdk';

const loaded = parseTvlSpec(`
spec:
  id: rag-demo
tvars:
  - name: model
    type: enum[str]
    domain: ["cheap", "accurate"]
objectives:
  - name: accuracy
    direction: maximize
exploration:
  strategy: random
  budgets:
    max_trials: 8
`);

const runTrial = optimize({
  ...loaded.spec,
  execution: {
    ...(loaded.spec.execution ?? {}),
    contract: 'trial',
  },
  evaluation: {
    data: [{ id: 1 }],
  },
})(async (trialConfig) => ({
  metrics: {
    accuracy: trialConfig.config.model === 'accurate' ? 1 : 0.6,
  },
}));

const result = await runTrial.optimize({
  algorithm: loaded.optimizeOptions?.algorithm ?? 'random',
  maxTrials: loaded.optimizeOptions?.maxTrials ?? 8,
});
```

Supported TVL subset:

- typed `tvars` for `bool`, `enum`, `int`, `float`, `tuple[...]`, and `callable[...]`
- standard and banded objectives
- structural and derived constraints
  - compiled through a parsed safe-expression subset, not `new Function()`
  - calls, computed property access, and unsupported syntax are rejected at load time
- exploration strategy mapping
- budgets:
  - `max_trials`
  - `max_spend_usd`
  - `max_wallclock_s`
- promotion-policy parsing
- explicit native compatibility reporting via `loaded.nativeCompatibility`

Still reduced vs full Python TVL/runtime semantics:

- promotion policy reporting/lifecycle parity beyond native best-trial selection and trial rejection
- full Python TVL statistical layer
- registry-backed domains and other advanced TVL runtime features

## Injection Modes

JS supports three runtime injection modes:

- `context` (default): use `getTrialParam()` / `getTrialConfig()` inside the wrapped function.
- `parameter`: the SDK calls your function as `agentFn(input, config?)`.
- `seamless`: the SDK supports framework interception, explicit code rewrites for tuned locals, and an experimental runtime rewrite fallback for self-contained functions.

`seamless` in this checkout is no longer framework-only. It means:

- [createTraigentOpenAI](/home/nimrodbu/Traigent_enterprise/traigent-js/src/integrations/openai/index.ts)
- [withTraigentModel](/home/nimrodbu/Traigent_enterprise/traigent-js/src/integrations/langchain/model.ts)
- [withTraigent](/home/nimrodbu/Traigent_enterprise/traigent-js/src/integrations/vercel-ai/index.ts)
- [autoWrapFrameworkTarget](/home/nimrodbu/Traigent_enterprise/traigent-js/src/integrations/auto-wrap.ts)
- [autoWrapFrameworkTargets](/home/nimrodbu/Traigent_enterprise/traigent-js/src/integrations/auto-wrap.ts)
- build-time transformed tuned variables via the Babel plugin export
- one-time source rewrites via `traigent migrate seamless`
- an experimental runtime rewrite for self-contained plain Node functions

For framework-mediated params, you can now batch-wrap supported targets instead
of calling each wrapper manually:

```ts
import {
  autoWrapFrameworkTargets,
  describeFrameworkAutoOverride,
  optimize,
  param,
} from '@traigent/sdk';

const { openaiClient, chatModel } = autoWrapFrameworkTargets({
  openaiClient,
  chatModel,
});

console.log(describeFrameworkAutoOverride(undefined, true));
```

For hardcoded local tuned variables, the recommended path is:

```bash
traigent migrate seamless src/agent.ts --write
```

or the build-time plugin:

```js
import traigentSeamless from '@traigent/sdk/babel-plugin-seamless';
```

Both the codemod and Babel plugin are fail-closed:

- unsupported patterns are reported explicitly
- the CLI will not modify blocked files
- the Babel plugin throws instead of emitting a partial transform

You can inspect which seamless path the SDK chose at runtime:

```js
console.log(optimizedAgent.seamlessResolution());
```

## Advanced Trial Contract

The old low-level manual trial contract still exists as an advanced path:

```ts
const runTrial = optimize({
  configurationSpace: {
    model: param.enum(['cheap', 'accurate']),
  },
  objectives: ['accuracy'],
  execution: {
    contract: 'trial',
  },
  evaluation: {
    data: [{ id: 1 }],
  },
})(async (trialConfig) => ({
  metrics: {
    accuracy: trialConfig.config.model === 'accurate' ? 1 : 0.5,
  },
}));
```

That path is deprecated in this checkout. Prefer the high-level agent contract
unless you intentionally want to manage metrics at the trial-function level.

## Hybrid Mode Authoring

Use `optimize(...)` and `toHybridConfigSpace(...)` to define tunables in code
while keeping the existing hybrid API wire format unchanged.

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
  execution: {
    contract: 'trial',
  },
  evaluation: {
    data: [{ id: 1 }],
  },
})(async () => ({
  metrics: {
    accuracy: 0,
    cost: 0,
  },
}));

export const configSpace = toHybridConfigSpace(childAgeSpec);

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

Notes:

- `configurationSpace`, `objectives`, `budget`, `evaluation`, `injection`, and `execution` are camelCase JS APIs.
- `toHybridConfigSpace()` preserves parameter names and emits the current hybrid `tunables` wire shape.
- `getTrialParam()` and `getTrialConfig()` remain the canonical runtime accessors when your host app binds `TrialContext`.
- This repo does not implement backend-guided hybrid optimization in `execution.mode = 'hybrid'`.

## Objectives and Parameters

Built-in objective strings:

- `accuracy` → `maximize`
- `cost` → `minimize`
- `latency` → `minimize`

Any other metric must use an explicit object:

```ts
objectives: [{ metric: 'quality_score', direction: 'maximize', weight: 1 }]
```

Parameter helpers:

- `param.enum(values)`
- `param.float({ min, max, scale, step? })`
- `param.int({ min, max, scale, step? })`

## Runtime Access Rules

Use `TrialContext`, `getTrialConfig()`, and `getTrialParam()` inside a bound
trial or capability execution path:

```ts
import { TrialContext, getTrialConfig, getTrialParam } from '@traigent/sdk';

await TrialContext.run(trialConfig, async () => {
  const config = getTrialConfig();
  const model = getTrialParam('model', 'gpt-4o-mini');
  console.log(config, model);
});
```

`TrialContext` is only guaranteed inside:

- native `wrapped.optimize(...)` execution
- wrapper calls that use `applyBestConfig()` in `context` mode
- host-managed capability execution where your app calls `TrialContext.run(...)`

Do not assume context is available in top-level route handlers or unrelated code paths unless you bind it yourself.

## Current Checkout Limits

- Native optimization is Node-only.
- Supported algorithms are `grid`, `random`, and sequential `bayesian`.
- `evaluation.data` or `evaluation.loadData` is required for high-level agent optimization.
- `budget.maxCostUsd` is enforced from numeric `metrics.total_cost` or `metrics.cost`.
- Native guardrails include `timeoutMs`, `trialConcurrency`, `plateau`, `checkpoint`, `execution.maxTotalExamples`, `execution.repsPerTrial`, and `execution.repsAggregation`.
- Runtime/provider metrics now preserve `input_cost`, `output_cost`, `total_cost`, and the compatibility alias `cost`.
- Log-scale grid search requires a multiplicative `step > 1`.
- `trialConcurrency` is limited to `grid` and `random`.
- `execution.mode = 'hybrid'` is not implemented in this branch.
- `seamless` now includes framework interception, codemod/build-time rewritten tuned variables, and an experimental runtime rewrite fallback for self-contained functions.
- seamless framework auto-override defaults to all active wrapped targets and can be narrowed with `frameworkTargets` or disabled with `autoOverrideFrameworks: false`.
- Native JS supports `defaultConfig`, callback-based `constraints`, and callback-based `safetyConstraints`.
- TVL loading is now supported for a focused native subset:
  - typed `tvars`
  - banded objectives
  - structural and derived constraints compiled into native callbacks
  - exploration strategy/budget mapping
  - promotion-policy parsing
  - `nativeCompatibility` reporting on loaded TVL artifacts
- Promotion policy is partially enforced in this checkout: native best-trial
  selection honors `minEffect` and `tieBreakers`, uses statistical promotion
  when both trials expose per-objective metric samples, and `chanceConstraints`
  reject trials when they have explicit `{successes, trials}` counts or binary
  metric samples. Native results/trials now expose bounded `promotionDecision`
  reports, but full Python promotion-gate lifecycle parity is still deferred.
- Python's safety preset/statistical layer and backend-guided Optuna orchestration are not implemented in this checkout.

## Documentation

Start here for the current repo shape:

- [docs/README.md](/home/nimrodbu/Traigent_enterprise/traigent-js/docs/README.md)
- [docs/CLIENT_CODE_GUIDE.md](/home/nimrodbu/Traigent_enterprise/traigent-js/docs/CLIENT_CODE_GUIDE.md)
- [docs/NATIVE_JS_PARITY_MATRIX.md](/home/nimrodbu/Traigent_enterprise/traigent-js/docs/NATIVE_JS_PARITY_MATRIX.md)
- [examples/README.md](/home/nimrodbu/Traigent_enterprise/traigent-js/examples/README.md)
- [walkthrough/README.md](/home/nimrodbu/Traigent_enterprise/traigent-js/walkthrough/README.md)

Legacy bridge references:

- [docs/REAL_MODE_SEQUENCE_FLOW.md](/home/nimrodbu/Traigent_enterprise/traigent-js/docs/REAL_MODE_SEQUENCE_FLOW.md)
- [docs/diagrams/real_mode_sequence.mmd](/home/nimrodbu/Traigent_enterprise/traigent-js/docs/diagrams/real_mode_sequence.mmd)
- [docs/diagrams/real_mode_flowchart.mmd](/home/nimrodbu/Traigent_enterprise/traigent-js/docs/diagrams/real_mode_flowchart.mmd)

## Development Validation

```bash
npm test
npm run typecheck
npm run smoke:example
npm run smoke:examples
npm run smoke:walkthrough
npm run build
npm pack --dry-run
```

## Requirements

- Node.js >= 18.0.0

## License

MIT
