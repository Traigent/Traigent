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

const answerQuestion = optimize({
  configurationSpace: {
    model: param.enum(["cheap", "accurate"]),
    temperature: param.float({ min: 0, max: 0.5, step: 0.25 }),
  },
  objectives: ["accuracy", "cost"],
  evaluation: {
    data: [
      { input: "capital of france", output: "Paris" },
      { input: "capital of japan", output: "Tokyo" },
    ],
    scoringFunction: (output, expectedOutput) =>
      output === expectedOutput ? 1 : 0,
    metricFunctions: {
      cost: (_output, _expectedOutput, runtimeMetrics) => runtimeMetrics.cost ?? 0,
    },
  },
  injection: {
    mode: "parameter",
  },
})(async (input, config) => {
  if (config?.model === "accurate") {
    return input === "capital of france" ? "Paris" : "Tokyo";
  }
  return input === "capital of france" ? "Paris" : "Osaka";
});

const result = await answerQuestion.optimize({
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

const answerQuestion = optimize({
  configurationSpace: {
    model: param.enum(["gpt-4o-mini", "gpt-4o"]),
    temperature: param.float({ min: 0, max: 1, step: 0.2 }),
  },
  objectives: [
    { metric: "accuracy", direction: "maximize", weight: 2 },
    { metric: "cost", direction: "minimize", weight: 1 },
  ],
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

For hybrid runs, the SDK exposes the backend finalization summary on
`result.reporting`. Use it for session-level reporting like total trial count,
successful trial count, total duration, cost savings, convergence history, and
optional full history payloads. `result.reporting.fullHistory` uses the backend
trial-result record shape (`session_id`, `trial_id`, `metrics`, `duration`,
`status`, `error_message`, `metadata`) so consumers can inspect raw session
history without guessing the payload format.

If you need session lifecycle helpers outside the optimize loop, use:

- `createOptimizationSession(request, options?)`
- `getNextOptimizationTrial(sessionId, options?)`
- `submitOptimizationTrialResult(sessionId, result, options?)`
- `listOptimizationSessions(options?)`
- `checkOptimizationServiceStatus(options?)`
- `getOptimizationSessionStatus(sessionId, options?)`
- `finalizeOptimizationSession(sessionId, options?)`
- `deleteOptimizationSession(sessionId, options?)`

Both helpers normalize backend field naming:

- `createOptimizationSession(...)` mirrors the reachable typed backend create
  route using JS-friendly camelCase request fields and a normalized `sessionId`
  response
- `getNextOptimizationTrial(...)` returns a normalized `suggestion`,
  `shouldContinue`, `stopReason`, and `sessionStatus` shape for low-level trial
  orchestration
- `submitOptimizationTrialResult(...)` accepts JS-friendly trial result input
  and normalizes the backend response to `{ success, continueOptimization, ... }`
- `listOptimizationSessions(...)` always returns a normalized `{ sessions, total }`
  shape and filters malformed session entries instead of guessing IDs
- `pattern` is forwarded directly to the backend list route; on the current
  typed-session backend it behaves like a substring-style session-id filter
- `total` reflects the backend-reported count before SDK-side filtering, so it
  may be larger than `sessions.length` when malformed entries are dropped
- `checkOptimizationServiceStatus(...)` returns `{ status, error? }` and mirrors
  the Python cloud-client behavior by returning `"unavailable"` instead of
  throwing when the backend health check itself fails
- `getOptimizationSessionStatus(...)` always exposes `sessionId`
- `getOptimizationSessionStatus(...)` and each normalized listed session entry
  also expose the current backend's known detail fields directly when present:
  `createdAt`, `functionName`, `datasetSize`, `objectives`, `experimentId`, and
  `experimentRunId`
- `finalizeOptimizationSession(...)` always exposes normalized `sessionId`,
  `bestConfig`, `bestMetrics`, optional `reporting`, and supports
  `includeFullHistory`
- `deleteOptimizationSession(...)` always exposes `success` and `sessionId`,
  whether the backend responds with a raw delete payload or the standard
  `{ success, message, data }` envelope. It defaults `cascade` to `false`, so
  recursive cleanup is opt-in.

For an executable control-plane example, see
[examples/core/hybrid-session-control/run.mjs](../examples/core/hybrid-session-control/run.mjs).
That example demonstrates:

- env-based `optimize(...)`
- explicit helper options for list/status/finalize/delete
- low-level session lifecycle parity is available through advanced helper APIs
- wrapped, auto-wrapped, or explicitly discovered framework clients in seamless mode
- `frameworkAutoOverrideStatus()` and `seamlessResolution()` diagnostics
- the shared `reporting` shape returned by `.optimize()` and `finalizeOptimizationSession(...)`

For supported frameworks, use seamless mode with a wrapped target:

```ts
import OpenAI from "openai";
import {
  optimize,
  param,
  prepareFrameworkTargets,
} from "@traigent/sdk";

const prepared = prepareFrameworkTargets({
  providers: {
    primary: new OpenAI({ apiKey: process.env.OPENAI_API_KEY }),
  },
});

console.log(prepared.discovered);
console.log(prepared.autoOverrideStatus);

const wrappedRuntime = prepared.wrapped;
const client = wrappedRuntime.providers.primary;

const answerQuestion = optimize({
  configurationSpace: {
    model: param.enum(["gpt-4o-mini", "gpt-4o"]),
    temperature: param.float({ min: 0, max: 1, step: 0.2 }),
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
    messages: [{ role: "user", content: input }],
  });

  return response.choices[0]?.message?.content ?? "";
});

console.log(answerQuestion.frameworkAutoOverrideStatus());
console.log(answerQuestion.seamlessResolution());
```

Use the diagnostics helpers when you want to understand why a seamless function
did or did not select framework interception:

- `frameworkAutoOverrideStatus()`
  - active registered targets
  - requested targets
  - selected targets
  - whether auto-override is enabled and why
  - reflects the current framework registry state, not a historical trial snapshot
- `seamlessResolution()`
  - the resolved seamless path for the optimized function
  - currently reports the framework path when seamless interception is active
  - returns `undefined` when seamless mode is not configured or when no active
    framework targets are currently registered
  - use `frameworkAutoOverrideStatus()` to tell those cases apart
- `discoverFrameworkTargets(value)`
  - bounded discovery for explicitly passed arrays and plain-object graphs
  - useful when your runtime bundles framework clients inside nested objects
- `autoWrapFrameworkTargets(value)`
  - recursively wraps those explicit object graphs with cycle safety
  - does not scan arbitrary module/global state
- `prepareFrameworkTargets(value)`
  - combines discovery, wrapping, and current auto-override diagnostics in one
    explicit-object setup step

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
- high-level agent functions with local `evaluation` and `injection`
- seamless framework interception for OpenAI, LangChain, and Vercel AI, with
  automatic runtime token/cost/latency collection

## Execution Contracts

Default path:

- plain agent/runnable function
- the SDK iterates the evaluation dataset
- local scoring produces trial metrics

Advanced compatibility path:

- set `execution.contract = "trial"`
- function input is `TrialConfig`
- function output is `{ metrics, metadata?, duration? }`

## Native Limits

- weighted objectives are rejected
- conditional parameters are rejected
- structural and derived constraints are rejected
- spec-level `budget.maxTrials` and `budget.maxWallclockMs` are rejected
- `spec.evaluation.data` or `spec.evaluation.loadData()` is still required
