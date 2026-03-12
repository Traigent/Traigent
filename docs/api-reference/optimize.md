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

- `createOptimizationSession(request, options?)`
- `getNextOptimizationTrial(sessionId, options?)`
- `submitOptimizationTrialResult(sessionId, result, options?)`
- `listOptimizationSessions(options?)`
- `checkOptimizationServiceStatus(options?)`
- `getOptimizationSessionStatus(sessionId, options?)`
- `finalizeOptimizationSession(sessionId, options?)`
- `deleteOptimizationSession(sessionId, options?)`

Helper normalization:

- `createOptimizationSession(...)` accepts camelCase request fields and returns
  normalized `sessionId`, `status`, and optional strategy / metadata fields even
  when the backend payload uses snake_case keys
- `getNextOptimizationTrial(...)` normalizes raw and wrapped next-trial payloads
  into one JS-friendly shape with `suggestion`, `shouldContinue`, `stopReason`,
  and `sessionStatus`
- `submitOptimizationTrialResult(...)` accepts JS-friendly result input and
  normalizes the backend response into `{ success, continueOptimization, ... }`
- `listOptimizationSessions(...)` normalizes raw and wrapped list payloads into
  `{ sessions, total }`, preserves `sessionId` on each listed session, and
  filters malformed session entries instead of inventing synthetic IDs
- `listOptimizationSessions(..., { pattern })` forwards `pattern` verbatim to the
  backend list route; in the current backend this behaves like a substring-style
  session-id filter
- `listOptimizationSessions(...).total` reflects the backend-reported total count
  before SDK-side filtering of malformed entries, so it may exceed
  `sessions.length`
- `checkOptimizationServiceStatus(...)` queries the backend root `/health` route
  and returns `{ status, error? }`. Like the Python cloud client, it reports
  `status: "unavailable"` instead of throwing when the health check itself
  fails
- `getOptimizationSessionStatus(...)` always returns `sessionId`, even when the
  backend payload uses `session_id`, and surfaces the current backend's known
  detail fields directly when present: `createdAt`, `functionName`,
  `datasetSize`, `objectives`, `experimentId`, and `experimentRunId`
- `listOptimizationSessions(...)` uses the same normalized session-entry shape
  as `getOptimizationSessionStatus(...)` for the per-session fields it can
  derive from the list response
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
- explicit helper options for create/next-trial/submit/list/status/finalize/delete
- wrapped, auto-wrapped, or explicitly discovered seamless framework usage

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
- `discoverFrameworkTargets(value)`
  Inspects explicitly passed arrays and plain-object graphs and reports the
  wrappable target paths it finds, such as `providers.primary`.
- `autoWrapFrameworkTargets(value)`
  Recursively wraps those explicit object graphs with cycle safety. It does not
  scan arbitrary module/global state.
- `prepareFrameworkTargets(value)`
  Performs bounded explicit-object discovery, wraps any supported targets it
  finds, and returns the current auto-override diagnostics alongside the
  wrapped value.
