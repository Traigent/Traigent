# `optimize()`

`optimize(spec)(fn)` wraps a function and attaches optimization metadata.

Primary contract:

- `fn(input, config?) => output`
- the SDK owns evaluation via `evaluation.scoringFunction`,
  `evaluation.metricFunctions`, or `evaluation.customEvaluator`

Advanced compatibility contract:

- set `execution.contract = "trial"`
- `fn(trialConfig) => { metrics, metadata?, duration? }`

Key spec fields:

- `configurationSpace`
- `objectives`
- `budget`
- `constraints`
- `defaultConfig`
- `execution`
- `evaluation`
- `promotionPolicy`
- `autoLoadBest`
- `loadFrom`
- `injection`

Execution modes:

- default hybrid execution
- explicit native execution with `mode: "native"`

Hybrid-only options:

- `includeFullHistory?: boolean`
  Requests backend full-history payloads during finalization and exposes them on
  `result.reporting.fullHistory`.

Hybrid result reporting:

- `result.reporting.totalTrials`
- `result.reporting.successfulTrials`
- `result.reporting.totalDuration`
- `result.reporting.costSavings`
- `result.reporting.convergenceHistory`
- `result.reporting.fullHistory`

Reporting shape:

- `convergenceHistory` is an array of backend-provided convergence snapshots
- `fullHistory` is an array of trial-result records with backend field names such
  as `session_id`, `trial_id`, `metrics`, `duration`, `status`, and
  `error_message`

Related hybrid session helpers:

- `getOptimizationSessionStatus(sessionId, options?)`
- `finalizeOptimizationSession(sessionId, options?)`
- `deleteOptimizationSession(sessionId, options?)`

Helper normalization:

- `getOptimizationSessionStatus(...)` always returns `sessionId`, even when the
  backend payload uses `session_id`
- `finalizeOptimizationSession(...)` always returns `sessionId`, normalized
  `bestConfig` / `bestMetrics`, optional `reporting`, and supports
  `includeFullHistory?: boolean`
- `deleteOptimizationSession(...)` normalizes both raw backend delete payloads
  and standard `{ success, message, data }` envelopes into one JS shape with
  `success`, `sessionId`, and optional `deleted` / `cascade`. It defaults
  `cascade` to `false`; set it explicitly when you want recursive cleanup.

Executable example:

- [examples/core/hybrid-session-control/run.mjs](../../examples/core/hybrid-session-control/run.mjs)

That example uses only public exports and shows both:

- env-based hybrid optimize/session control
- explicit helper options for status/finalize/delete
- wrapped or auto-wrapped seamless framework usage

Related seamless diagnostics on the optimized function:

- `frameworkAutoOverrideStatus()`
  Returns active registered framework targets, requested targets, selected
  targets, and the reason framework auto-override is or is not enabled. This
  reflects the current framework registry state, not a historical trial
  snapshot.
- `seamlessResolution()`
  Returns the resolved seamless path for the optimized function. In the current
  hybrid worktree this surfaces the framework interception path and its active
  targets when applicable. It returns `undefined` when seamless mode is not
  configured, or when seamless mode is configured but no active framework
  targets are currently registered. Use `frameworkAutoOverrideStatus()` to
  distinguish those cases.
