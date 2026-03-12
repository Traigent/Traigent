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

const answerQuestion = optimize({
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
    data: [
      { input: "capital of france", output: "Paris" },
      { input: "capital of japan", output: "Tokyo" },
    ],
    scoringFunction: (output, expectedOutput) =>
      output === expectedOutput ? 1 : 0,
    metricFunctions: {
      cost: (_output, _expected, runtimeMetrics) => runtimeMetrics.cost ?? 0,
      latency: (_output, _expected, runtimeMetrics) => runtimeMetrics.latency ?? 0,
    },
  },
  injection: {
    mode: "parameter",
  },
})(async (input, config) => {
  const model = String(config?.model ?? "cheap");
  if (model === "accurate") {
    return input === "capital of france" ? "Paris" : "Tokyo";
  }
  return input === "capital of france" ? "Paris" : "Osaka";
});

const result = await answerQuestion.optimize({
  mode: "native",
  algorithm: "grid",
  maxTrials: 10,
  timeoutMs: 5_000,
});

console.log(result.bestConfig);
console.log(result.bestMetrics);
answerQuestion.applyBestConfig(result);
console.log(answerQuestion.currentConfig());
```

See [`examples/native-optimization.mjs`](./examples/native-optimization.mjs) for
the runnable smoke example.

## Backend-Guided Hybrid Optimization

Hybrid optimization keeps trial execution local in JS while the Traigent backend
drives Optuna-style configuration suggestions through the session API. This is
the default when you omit `mode`.

```ts
import { optimize, param } from "@traigent/sdk";

const answerQuestion = optimize({
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
    data: [{ input: "hello", output: "HELLO!" }],
    scoringFunction: (output, expectedOutput) =>
      output === expectedOutput ? 1 : 0,
  },
  injection: {
    mode: "parameter",
  },
})(async (input, config) =>
  config?.temperature && Number(config.temperature) > 0.5
    ? `${String(input).toUpperCase()}?`
    : `${String(input).toUpperCase()}!`,
);

const result = await answerQuestion.optimize({
  algorithm: "optuna",
  maxTrials: 12,
  backendUrl: process.env.TRAIGENT_BACKEND_URL,
  apiKey: process.env.TRAIGENT_API_KEY,
  timeoutMs: 5_000,
  includeFullHistory: true,
});
```

Hybrid results expose backend finalization reporting directly on
`result.reporting`, including:

- `totalTrials`
- `successfulTrials`
- `totalDuration`
- `costSavings`
- `convergenceHistory`
- `fullHistory` when `includeFullHistory: true`

For follow-up session control, the hybrid SDK also exposes:

- `createOptimizationSession(request, options?)`
- `getNextOptimizationTrial(sessionId, options?)`
- `submitOptimizationTrialResult(sessionId, result, options?)`
- `listOptimizationSessions(options?)`
- `checkOptimizationServiceStatus(options?)`
- `getOptimizationSessionStatus(sessionId, options?)`
- `finalizeOptimizationSession(sessionId, options?)`
- `deleteOptimizationSession(sessionId, options?)`

`deleteOptimizationSession(...)` defaults `cascade` to `false`; opt in
explicitly if you want recursive backend cleanup.

The low-level `createOptimizationSession(...)`,
`getNextOptimizationTrial(...)`, and
`submitOptimizationTrialResult(...)` helpers are advanced session-control APIs.
Use them when you want to drive the typed session lifecycle directly instead of
going through `wrapped.optimize(...)`.

`listOptimizationSessions(...)` forwards `pattern` directly to the current
typed-session backend, where it behaves like a substring-style session-id
filter. Its `total` field reflects the backend-reported count before SDK-side
filtering of malformed entries, so it may be larger than `sessions.length`.
`getOptimizationSessionStatus(...)` and listed session entries also surface the
current backend's known session detail fields directly when present:
`createdAt`, `functionName`, `datasetSize`, `objectives`, `experimentId`, and
`experimentRunId`, while still preserving the raw `metadata` object.

See [`examples/core/hybrid-session-control/run.mjs`](./examples/core/hybrid-session-control/run.mjs)
for an executable session-control flow that demonstrates:

1. env-based hybrid optimize/session helpers
2. explicit options-based list/health/status/finalize/delete helper calls
3. wrapped, auto-wrapped, or discovered framework clients in `injection.mode = "seamless"`
4. shared `reporting` shape between `.optimize()` and `finalizeOptimizationSession(...)`
5. seamless diagnostics through `frameworkAutoOverrideStatus()` and `seamlessResolution()`

For supported frameworks, seamless mode now works in hybrid too. Wrap the
framework target once, then let the backend-suggested config override the local
framework call while the SDK records provider usage metrics:

```ts
import OpenAI from "openai";
import {
  autoWrapFrameworkTargets,
  discoverFrameworkTargets,
  optimize,
  param,
} from "@traigent/sdk";

const runtime = {
  providers: {
    primary: new OpenAI({ apiKey: process.env.OPENAI_API_KEY }),
  },
};

console.log(discoverFrameworkTargets(runtime));

const wrappedRuntime = autoWrapFrameworkTargets(runtime);
const client = wrappedRuntime.providers.primary;

const answerQuestion = optimize({
  configurationSpace: {
    model: param.enum(["gpt-4o-mini", "gpt-4o"]),
    temperature: param.float({ min: 0, max: 1, step: 0.2 }),
    maxTokens: param.int({ min: 32, max: 256, step: 32 }),
  },
  objectives: ["accuracy", "cost"],
  evaluation: {
    data: [{ input: "hello", output: "HELLO!" }],
    scoringFunction: (output, expectedOutput) =>
      output === expectedOutput ? 1 : 0,
  },
  injection: {
    mode: "seamless",
  },
})(async (input) => {
  const response = await client.chat.completions.create({
    model: "gpt-3.5-turbo",
    temperature: 0.9,
    max_tokens: 16,
    messages: [{ role: "user", content: input }],
  });

  return response.choices[0]?.message?.content ?? "";
});

console.log(answerQuestion.frameworkAutoOverrideStatus());
console.log(answerQuestion.seamlessResolution());
```

The seamless diagnostics surface is available on every optimized function:

- `frameworkAutoOverrideStatus()`
  - reports active registered targets, requested targets, selected targets, and
    why framework auto-override is or is not enabled
  - reflects the current framework registry state, not a historical trial snapshot
- `seamlessResolution()`
  - reports the resolved seamless path for the function, including framework
    targets when the framework interception path is active
  - returns `undefined` when seamless mode is not configured, or when seamless
    mode is configured but no active framework targets are currently registered
  - use `frameworkAutoOverrideStatus()` to distinguish those cases

For bounded convenience discovery, the hybrid worktree also supports:

- `discoverFrameworkTargets(value)`
  - inspects explicitly passed arrays and plain-object graphs
  - reports discovered target paths like `providers.primary`
- `autoWrapFrameworkTargets(value)`
  - now recursively wraps those explicit object graphs
  - preserves cycles and repeated references
  - does not scan arbitrary module/global state

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

Primary contract:

- Input: a plain agent function like `agentFn(input, config?)`
- Output: raw model/application output
- Evaluation: local `scoringFunction`, `metricFunctions`, or `customEvaluator`
- Seamless wrappers: OpenAI, LangChain, and Vercel AI can now apply backend
  suggestions automatically and feed token/cost/latency metrics back into the
  hybrid evaluation pipeline

Advanced compatibility contract:

- set `execution.contract = "trial"`
- input: `TrialConfig`
- output: `{ metrics, metadata?, duration? }`

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
