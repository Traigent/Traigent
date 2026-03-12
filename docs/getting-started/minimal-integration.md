# Minimal Integration

Use `optimize(spec)(fn)` to attach optimization metadata to a plain agent
function. The SDK can own dataset iteration and evaluation for you.

```ts
import { optimize, param } from "@traigent/sdk";

const answerQuestion = optimize({
  configurationSpace: {
    model: param.enum(["cheap", "accurate"]),
  },
  objectives: ["accuracy"],
  evaluation: {
    data: [{ input: "capital of france", output: "Paris" }],
    scoringFunction: (output, expectedOutput) =>
      output === expectedOutput ? 1 : 0,
  },
  injection: {
    mode: "parameter",
  },
})(async (input, config) =>
  config?.model === "accurate" ? "Paris" : String(input).toUpperCase(),
);
```

Then choose either:

- backend-guided hybrid: `answerQuestion.optimize({ algorithm: "optuna", ... })`
- local native: `answerQuestion.optimize({ mode: "native", algorithm: "grid", ... })`

For hybrid runs, inspect `result.reporting` for the backend finalization
summary. Set `includeFullHistory: true` if you want the backend full-history
payload included there.

If you keep the returned `result.sessionId`, you can later query, finalize, or
inspect and manage the typed backend session with
`checkOptimizationServiceStatus(...)`, `listOptimizationSessions(...)`, `getOptimizationSessionStatus(...)`,
`finalizeOptimizationSession(...)`, and `deleteOptimizationSession(...)`.

See the executable session-control walkthrough in
[examples/core/hybrid-session-control/run.mjs](../../examples/core/hybrid-session-control/run.mjs)
for both:

- env-based auth/config
- explicit helper options-based auth/config
- wrapped or discovered framework interception in `injection.mode = "seamless"`
- `frameworkAutoOverrideStatus()` / `seamlessResolution()` diagnostics

If your agent already uses a supported framework client or model, wrap it once
and switch to `injection.mode = "seamless"`. The wrapped framework call will
pick up optimized params from the active trial and report runtime usage metrics
automatically. Use `autoWrapFrameworkTarget(...)` when you want the SDK to pick
the right OpenAI/LangChain/Vercel wrapper for a direct framework object. If
your runtime stores clients/models inside nested arrays or plain-object graphs,
use `prepareFrameworkTargets(...)` for the one-call path, or
`discoverFrameworkTargets(...)` and `autoWrapFrameworkTargets(...)` if you want
the discovery and wrapping steps separately.

For production debugging, call:

- `optimizedFn.frameworkAutoOverrideStatus()`
- `optimizedFn.seamlessResolution()`

to inspect the currently active framework targets and the seamless path chosen
for that optimized function.

If you need the older low-level contract, set `execution.contract = "trial"` and
return metrics directly.
