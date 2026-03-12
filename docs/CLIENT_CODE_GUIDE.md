# Client Code Guide

This guide describes the current JavaScript and TypeScript integration surface for `@traigent/sdk` in this checkout.

## Choose a Flow

Use the SDK in one of two supported ways:

| Flow                     | When to use it                                                                         | Main APIs                                                                                     |
| ------------------------ | -------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------- |
| Hybrid mode authoring    | Your app exposes Traigent-compatible HTTP routes and needs a code-defined config space | `optimize`, `param`, `toHybridConfigSpace`, `TrialContext`, `getTrialParam`, `getTrialConfig` |
| Native Node optimization | Your optimization loop should run in-process in Node                                   | `optimize`, `param`, `wrapped.optimize(...)`, `applyBestConfig()`                             |

This repo does not implement backend-guided `execution.mode = 'hybrid'`. The older Python-orchestrated bridge flow still exists, but it is documented as a legacy reference in [REAL_MODE_SEQUENCE_FLOW.md](./REAL_MODE_SEQUENCE_FLOW.md).

## Flow 1: Hybrid Mode Authoring

In hybrid mode authoring, the SDK defines the spec and exposes runtime config access. Your service still owns the HTTP routes, validation, dispatch, and error shaping.

### What you write

1. A code-defined optimization spec.
2. A host app that converts the spec with `toHybridConfigSpace()`.
3. Capability or business logic that runs inside `TrialContext.run(...)`.

### Minimal example

```ts
import { TrialContext, getTrialParam, optimize, param, toHybridConfigSpace } from '@traigent/sdk';

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

    return { model, temperature };
  });
}
```

### Hybrid authoring rules

- Keep `configurationSpace`, `objectives`, `budget`, `evaluation`, `injection`, and `execution` in camelCase.
- Keep your existing hybrid wire contract unchanged by serializing with `toHybridConfigSpace()`.
- Resolve runtime config with `getTrialParam()` and `getTrialConfig()` inside the execution path, not in top-level route handlers.
- If you temporarily support both code-defined specs and JSON config files, choose one source at startup and fail fast when they disagree.

## Flow 2: Native Node Optimization

In native mode, the primary JS contract is a plain agent function plus an `evaluation` block. The SDK owns dataset iteration, output scoring, metric aggregation, and optimization.

### What you write

1. A spec with `optimize(...)`.
2. A plain agent function returning output.
3. An evaluation dataset supplied via `evaluation.data` or `evaluation.loadData`.

### Minimal example

```ts
import { getTrialParam, optimize, param } from '@traigent/sdk';

const answerQuestion = optimize({
  configurationSpace: {
    model: param.enum(['cheap', 'accurate']),
    temperature: param.float({ min: 0, max: 0.5, step: 0.5, scale: 'linear' }),
  },
  objectives: ['accuracy', 'cost'],
  budget: {
    maxCostUsd: 1,
  },
  execution: {
    maxTotalExamples: 100,
    exampleConcurrency: 4,
    repsPerTrial: 3,
    repsAggregation: 'median',
  },
  evaluation: {
    data: [
      { input: 'What is 2+2?', output: '4' },
      { input: 'What is the capital of France?', output: 'Paris' },
    ],
    scoringFunction: (output, expectedOutput) => (output === expectedOutput ? 1 : 0),
    metricFunctions: {
      cost: (_output, _expectedOutput, _runtimeMetrics, row) =>
        row.input.includes('capital') ? 0.2 : 0.1,
    },
  },
})(async (question: string) => {
  const model = String(getTrialParam('model', 'cheap'));
  return model === 'accurate' ? (question.includes('capital') ? 'Paris' : '4') : 'unknown';
});

const result = await answerQuestion.optimize({
  algorithm: 'grid',
  maxTrials: 12,
  randomSeed: 42,
  timeoutMs: 5_000,
  trialConcurrency: 2,
});

answerQuestion.applyBestConfig(result);
console.log(await answerQuestion('What is 2+2?'));
console.log(answerQuestion.currentConfig());
```

### Injection modes

- `context`:
  - default mode
  - use `getTrialParam()` / `getTrialConfig()` in the wrapped function
- `parameter`:
  - the SDK calls your function as `agentFn(input, config?)`
  - use this when you want explicit tuned-variable flow in user code
- `seamless`:
  - first-class through framework wrappers for OpenAI, LangChain, and Vercel AI
  - the fastest framework path is now:
    - `autoWrapFrameworkTarget(...)` for one client/model
    - `autoWrapFrameworkTargets({ ... })` for a small object map of supported targets
    - `discoverFrameworkTargets(...)` to inspect an explicit object graph
    - `prepareFrameworkTargets(...)` for one-call discover + wrap + status
  - also supported for hardcoded local tuned variables through:
    - `traigent migrate seamless`
    - `@traigent/sdk/babel-plugin-seamless`
    - an experimental runtime rewrite fallback for self-contained plain Node functions when `TRAIGENT_ENABLE_EXPERIMENTAL_RUNTIME_SEAMLESS=1`
  - defaults `autoOverrideFrameworks` to `true`, so all active wrapped targets
    are eligible unless you narrow them with `frameworkTargets`
  - fail-closed semantics:
    - the codemod reports rejected patterns and will not modify blocked files
    - the Babel plugin throws instead of emitting a partial transform
  - `frameworkTargets` is still useful when you want to constrain seamless interception to specific SDKs
  - set `autoOverrideFrameworks: false` when you want seamless to ignore active
    wrapped SDK targets and rely only on transformed code paths

### Tuned-variable discovery

The native checkout now includes a bounded tuned-variable discovery helper for
local JS functions:

```ts
import { discoverTunedVariables } from '@traigent/sdk';

const report = discoverTunedVariables(answerQuestion);
console.log(report.candidates);
```

Use it when you want a heuristic report of likely tunable locals before moving
to explicit `context` / `parameter` injection or the seamless codemod path.

The matching CLI is:

```bash
traigent detect tuned-variables src/agent.ts --function answerQuestion
```

For native TVL artifacts, the CLI can also inspect the loaded feature set and
native-compatibility report:

```bash
traigent inspect tvl path/to/spec.yml
```

Current scope:

- native checkout only
- self-contained function bodies
- literal defaults that map to the JS config-space model
- high/medium/low confidence reporting with warnings for skipped reassigned vars

This is intentionally still behind Python’s fuller tuned-variable discovery and
config-generation pipeline.

Example:

```ts
import {
  describeFrameworkAutoOverride,
  prepareFrameworkTargets,
  optimize,
  param,
} from '@traigent/sdk';

const prepared = prepareFrameworkTargets({
  openaiClient,
  chatModel,
});
const wrapped = prepared.wrapped;

const agent = optimize({
  configurationSpace: {
    model: param.enum(['gpt-4o-mini', 'gpt-4o']),
    temperature: param.float({ min: 0, max: 1, step: 0.2, scale: 'linear' }),
  },
  objectives: ['accuracy'],
  injection: {
    mode: 'seamless',
  },
  evaluation: {
    data: [{ input: 'hello', output: 'ok' }],
    scoringFunction: () => 1,
  },
})(async (input) => wrapped.chatModel.invoke(input));

console.log(describeFrameworkAutoOverride(undefined, true));
console.log(prepared.discovered);
```

### Bounded framework discovery

The native checkout now also supports bounded framework discovery for explicit
object graphs:

```ts
import { discoverFrameworkTargets, prepareFrameworkTargets } from '@traigent/sdk';

const discovered = discoverFrameworkTargets({
  services: { openaiClient, chatModel },
});

const prepared = prepareFrameworkTargets({
  services: { openaiClient, chatModel },
});
```

Current scope:

- direct supported targets
- nested arrays and plain-object graphs
- cycle-safe traversal

Out of scope:

- scanning module globals
- auto-patching arbitrary imported clients you did not pass explicitly

Identity note:

- OpenAI clients are wrapped in place
- LangChain and Vercel AI targets return wrapped objects
- all wrapper paths are idempotent, so repeated wrapping is safe

### Native mode rules

- Built-in objective strings are limited to `accuracy`, `cost`, and `latency`.
- Any other objective must use `{ metric, direction, weight? }`.
- `budget.maxCostUsd` only works when trials produce numeric `metrics.total_cost` or `metrics.cost`.
- `execution.maxTotalExamples` caps the total number of evaluated examples across all trials.
- `execution.exampleConcurrency` runs multiple examples from the same trial in parallel while preserving stable aggregation and metric-sample order.
- `execution.repsPerTrial` and `execution.repsAggregation` provide repetition-based stability for noisy evaluations.
- `execution.contract = 'trial'` still supports the old low-level trial-function style, but it is deprecated and documented as advanced-only.

### Advanced low-level trial contract

If you intentionally want to manage metrics yourself, you can still opt into the old contract:

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

Treat that as an advanced compatibility path, not the primary JS usage model.

## Runtime Access Rules

`getTrialConfig()` and `getTrialParam()` are the canonical runtime accessors.

Use them only when a trial context is active:

- inside `TrialContext.run(...)`
- inside a native `.optimize(...)` run
- during normal calls after `applyBestConfig()` when using context-based injection
- inside host-managed capability logic that has already bound a synthetic `TrialConfig`

Do not introduce a second global config mechanism for JS.

## Current Checkout Limits

- Native optimization supports `grid`, `random`, and sequential `bayesian`.
- Native optimization is Node-only.
- `budget.maxCostUsd` still depends on numeric `metrics.total_cost` or `metrics.cost`.
- Cost-aware wrappers record `input_cost`, `output_cost`, `total_cost`, and `cost` in runtime metrics.
- Log-scale grid search requires a multiplicative `step > 1`.
- `trialConcurrency` is supported for `grid` and `random` only.
- `execution.mode = 'hybrid'` is not implemented in this branch.
- `seamless` prefers explicit rewrite tooling over runtime magic:
  - use framework wrappers for SDK-mediated params
  - use the codemod or Babel plugin for hardcoded locals
  - treat runtime rewriting as experimental
  - inspect `wrapped.seamlessResolution()` when you need to confirm which path was actually selected
- Native JS supports `defaultConfig`, callback-based `constraints`, and callback-based `safetyConstraints`.
- TVL loading is supported for a focused native subset:
  - typed `tvars`
  - banded objectives
  - structural and derived constraints compiled through a parsed safe-expression subset
  - exploration strategy/budget mapping
  - promotion-policy parsing
  - `parseTvlSpec()` and `loadTvlSpec()` expose artifact-specific
    `nativeCompatibility`
    so the supported vs reduced-semantics native envelope is explicit in the
    loaded file, including `usedFeatures` and summarized `warnings`
- Promotion policy is partially enforced in this checkout: native best-trial
  selection honors `minEffect` and `tieBreakers`, uses statistical promotion
  when both trials expose per-objective metric samples, and `chanceConstraints`
  reject trials when they have explicit `{successes, trials}` counts or binary
  metric samples. Native results/trials now expose bounded `promotionDecision`
  reports plus a small `result.reporting` summary for trial counts, evaluated
  examples, and whether chance constraints / statistical comparison /
  tie-breakers were involved. Full Python promotion-gate lifecycle parity is
  still deferred.
- Python's safety preset/statistical layer and backend-guided Optuna orchestration are not part of this checkout.

## Legacy Bridge Flow

If you still need the Python-to-Node bridge:

- Use the CLI runner with `npx traigent-js --module ./dist/trial.js --function runTrial`
- Treat [REAL_MODE_SEQUENCE_FLOW.md](./REAL_MODE_SEQUENCE_FLOW.md) and the Mermaid diagrams as bridge-only reference material
