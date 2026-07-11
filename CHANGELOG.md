# Changelog

All notable changes to Traigent SDK are documented in this file.

Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

## [0.21.3] - 2026-07-11

### Added

- Additive SmartOps Planner V2 client and `traigent guidance` CLI with strict
  treatment/profile provenance, opaque decision execution, server-integrity-tagged structured
  resolution, explicit idempotent receipt submission, and lifecycle reopen
  commands.
  The existing `traigent next-steps` v1 surface remains unchanged.

### Fixed

- Correct the registered observability issue-list MCP wrapper so it matches the
  six-argument helper contract instead of raising a runtime ``TypeError``.
- Enforce Planner V2 treatment isolation even outside strict experiment mode;
  strict mode additionally rejects fallback and incomplete context.
- Validate exact action economics, implementation hash, attempt-bound argv,
  receipt scope, and live equal execution/lease expiries before presenting a
  resolved operation. The server integrity tag remains opaque to public clients.
- Describe rule parity and certified session-utility advantage without implying
  policy agreement or a product KPI guarantee, and validate reopen/receipt
  lifecycle transitions fail-closed.

- Declare `tenacity` as a dependency so litellm's retry helpers
  (`completion_with_retries`) work on a clean install (#1824).

## [0.21.0] - 2026-07-07

Observability hardening release. Enterprise-grade privacy posture for the
`@observe` instrumentation plus latency/usage accuracy fixes across the
SDK, backend, and portal.

### Changed

- **BREAKING (behavioral): `@observe` is now metadata-only by default.** The
  decorator and `ObserveContext` no longer send function arguments or return
  values to the observability backend by default — only span metadata (name,
  type, timing, tags, session/environment, and token/cost fields you pass).
  To restore raw input/output capture, opt in per span with
  `@observe(..., content_mode="record")` or globally with
  `TRAIGENT_OBSERVABILITY_CONTENT=record`. Use `content_mode="redacted"` to
  emit redaction placeholders. This closes an egress path where prompt/response
  content left the process by default. **Migration:** if you relied on seeing
  prompts/outputs in the dashboard, set `content_mode="record"` on those spans
  (only for spans whose I/O is safe to store).
- Recorded span latency now measures only the wrapped function body. Previously
  the span start was stamped before SDK trace-setup work, so transport/setup
  overhead (tens of ms) was misattributed to application latency.
- Token and cost usage fields are nullable end-to-end (SDK DTO → backend schema
  and model → portal). Unknown usage now renders as `—` instead of a misleading
  `0` / `$0.0000`; a measured zero still renders as `0`. Completed generation
  spans with no usage log a one-time warning rather than implying auto-capture.

### Fixed

- Observability transport no longer blocks the calling application thread on
  batch-size flushes: sends run on a daemon flush thread, and payload
  redaction/serialization happen before acquiring the transport lock. An age
  timer stays armed as a backstop so a tail item is never stranded if the flush
  thread is in its death window when the next item arrives.
- Telemetry drops (queue full, oversized/non-serializable payloads, closed
  transport, failed batches) are no longer silent: each updates stats/errors and
  emits a `health_callback` event.
- Unknown `ENVIRONMENT`/`TRAIGENT_ENV` labels now log a warning (without leaking
  the raw value) instead of being silently ignored; a missing `session_id`
  renders as `—`.

## [0.20.1] - 2026-07-06

Patch release for the managed optimization dispatch fix wave and adjacent
SDK honesty/hardening fixes shipped on develop after 0.20.0.

### Changed
- Named smart optimization for managed-backend users now binds supported
  algorithm names (`bayesian`, `tpe`, `optuna`, `optuna_tpe`, and
  `optuna_random`) to the typed backend Optuna strategy and serializes
  `optimization_strategy` on session creation. Unsupported smart names such
  as `nsga2` and `cmaes` now fail before backend session creation with a
  capability message instead of creating an orphaned session or returning a
  misleading zero-trial result. (#1758, #1752)

### Fixed
- Dict-input evaluation now warns when metadata-only parameters are defaulted
  instead of silently accepting the defaulted shape. (#1756, #1746)
- Local run-log persistence preserves trial/configuration fidelity and keeps
  orchestrator builder compatibility. (#1757, #1747, #1748)
- CLI algorithm/model surfaces now match public behavior and remove unsourced
  Claude model guesses. (#1759, #1751)
- Callback failures derived from `BaseException` are contained without
  breaking optimization control flow. (#1760, #1749)
- Objective reporting now honestly records unmatched objective metrics instead
  of implying a measurement exists. (#1761, #1691)
- API-key validation failures are classified as auth/permission failures rather
  than generic backend/network errors. (#1762, #1754)
- Prior unreleased develop fixes since 0.20.0: public onboarding skills repo
  selection (#1737, #1623), early-stop decorator propagation (#1739, #1692),
  unpriceable OpenRouter cost handling (#1740, #1597), redaction fail-open
  tightening (#1742, #1649, #1650), exact ruff pin extraction in preflight
  (#1743, #1550), analytics insight value validation (#1744, #1662), and
  companion error message threading (#1745, #1885).

## [0.20.0] - 2026-07-05

Fail-closed hardening release. A systematic silent-failure audit of the
SDK↔backend integration (25 findings) closed the classes of error that were
being suppressed or silently degraded without telling the user, and added
reusable enforcement so they cannot silently recur.

### Fixed
- Metric/scoring-function exceptions no longer silently score an objective
  0.0: a failing objective metric now fails the trial (excluded from
  best-config selection) instead of fabricating a score; informational-metric
  failures are recorded as structured `metric_errors`. (#1722)
- CLI commands now return honest exit codes — `optimize`/`validate`/`check`
  exit non-zero on real failure (was always exit 0); `auth login`/`refresh`
  fail when credential persistence fails; `plan`/`next-steps` use stored
  credentials and fall back to localhost, not the prod cloud. (#1721)
- Decorator/session contract: `ExecutionOptions` now rejects misspelled option
  keys (was `extra="allow"`, silently swallowed); `@optimize(default_config=)`
  is now sent on the session-create wire (was materialized locally only);
  `validate_providers` docs corrected to the real standalone function. (#1723)
- Trial/result persistence made honest: failed-trial error messages now use the
  wire key the backend reads (`error_message`), so they are no longer dropped;
  a session rollup that was never transmitted is reported `persistence_status=
  "degraded"` instead of `"succeeded"`; measures-contract violations fail closed
  instead of submitting unvalidated metrics; weighted-score partial failures
  now record attempted/failed counts. (#1724)
- Warm-start visibility: seed drops are surfaced (drop-reason histogram +
  candidate count) instead of debug-logged and discarded; the seed cap keeps
  the most recent (not oldest) configs and signals truncation; a classic-REST
  seed-build error reports `degraded` instead of silently omitting warm-start.
  (#1725)
- `create_optimization_workflow` now raises on a missing `agent_id` /
  `experiment_run_id` (previously only `experiment_id` was checked). (#1726)
- Backend session finalization is retried before reporting failure. (#1730)

### Added
- Reusable silent-failure checkers (Phase D): a report-only, ratcheted
  AST lint that flags new swallowed/silently-substituted wire and cost
  failures in `traigent/cloud`, `traigent/cli`, and the pricing modules, plus
  unknown-parameter rejection tests pinning that public entrypoints reject
  unknown kwargs. (#1727)

## [0.19.3] - 2026-07-04

Hotfix: user-declared objective weights now reach the backend.

### Fixed
- `ObjectiveSchema` / ACL weights are now serialized to the backend at
  session creation as `{name, orientation, weight}` objective dicts.
  Previously only bare metric names crossed the wire, so backend-routed
  optimization silently equal-weighted all objectives and the portal
  showed "0 weights / optimization objective: not reported" with a
  0/80/20 display fallback. Pre-existing in all prior releases; the
  backend counterpart mirrors the declared objectives into the run
  metadata the portal reads. (#1715, #1716; TraigentBackend#1987)

## [0.19.2] - 2026-07-04

Multi-run analytics release: agents can now read the experiment-group
cohort — one aggregated, source-preserving table of results across runs
for the same agent + dataset.

### Added
- Three analytics MCP tools exposing the experiment-group cohort
  endpoints: `analytics_list_experiment_groups`,
  `analytics_get_experiment_group`, and
  `analytics_list_experiment_group_configuration_runs` (one row per
  configuration-run across the cohort's runs, carrying `configuration`,
  `measures`, status, `trial_number`, and source ids). Rows remain
  individual source runs — grouped, never merged or deduped by config
  hash. (#1707)
- Server/tool parity guard: every `ANALYTICS_TOOL_NAMES` entry must be
  registered on the FastMCP server, so a defined-but-unregistered tool
  can no longer ship silently. (#1707)

### Fixed
- `traigent next-steps` now honors `TRAIGENT_BACKEND_URL` /
  `TRAIGENT_API_URL` — the `--backend-url` option previously ignored the
  documented environment variable and silently fell back to
  `http://localhost:5000`. Wiring and help text now match
  `traigent plan`. (#1707)

## [0.19.1] - 2026-07-03

No-silent-legacy hardening release: five classes of silent failure now either
work correctly or fail loudly.

### Fixed
- `ObjectiveSchema` weights now govern `best_config` selection for grid/random
  runs via observed-range (min-max, orientation-aware) weighted scoring;
  per-trial `metrics["score"]` is populated and post-hoc
  `calculate_weighted_scores` uses the same scorer. Flipping weights
  0.9/0.1 ↔ 0.1/0.9 on an accuracy/cost tradeoff now flips the winner. (#1682)
- `warm_start_transfer` returned in the session-create response now reaches
  `result.metadata` (previously dropped; runs reported `no_seed_configs` from
  valid cloud-tracked priors). An explicit `warm_start_from` that applies 0
  seeds emits a warning naming the refusal reason. (#1683)
- Dataset file-not-found errors name the resolution rule
  (`TRAIGENT_DATASET_ROOT` vs cwd), the registry-resolved path, and the exact
  candidate tried — the doubled-path case is self-diagnosing. (#1684)

### Changed (fail loud instead of silent)
- Smart algorithms (`bayesian`, `tpe`, `cmaes`, `nsga2`, `optuna*`) raise an
  actionable error when the managed cloud path cannot execute them, instead of
  returning a silent 0-trial COMPLETED result with `best_config=None`. Runtime
  `optimize(algorithm=...)` smart overrides re-resolve to cloud-required
  policy (no silent local fallback). (#1681)
- `.optimize()` rejects decorator-only arguments at call time with a
  move-it-to-the-decorator `TypeError` — `warm_start_from` and 39 sibling
  options were previously swallowed silently. (#1683)
- Deprecated `execution_mode` selectors (notably `"edge_analytics"`) raise
  `ConfigurationError` with `algorithm=` + `offline=` migration guidance on
  every entry path (decorator, `initialize`, `TraigentConfig`, client) —
  previously a silent 0-trial run. On the wire, `metadata["mode"]` now carries
  the canonical `local`/`hybrid`/`hybrid_api` vocabulary. (#1684, #1393)
- Nonpositive cost limits raise at configuration time; a budget-gate block
  logs "budget gate blocked N planned trials (estimated $X > limit $Y)"
  instead of silently completing with 0 trials. (#1684)
- Flagship multi-objective examples and docs migrated off removed/deprecated
  parameters (`execution_mode`, `budget_limit`). (#1684)

## [0.19.0] - 2026-07-03

### Added
- `SemanticSaturationStopCondition` rebuilt from the shipped schema plan
  (FR-SDK-SEMANTIC-SATURATION-V1).
- Surrogate (pre-screen) evaluator: a cheap second scorer over captured
  outputs, with an e2e co-evolution demo in mock mode.
- Smart-pruning producer wired end-to-end: per-example pruning is reachable
  from the SDK, with real per-example accuracy threaded into pruning progress
  and live-e2e-confirmed correctness/durability fixes.
- `list_sessions()` with typed `SessionSummary`; `delete_session` defaults to
  non-destructive and `cascade=True` now calls the backend DELETE.
- Experiment group analytics client.

### Changed
- Deprecation warnings for the legacy session contract and the
  `edge_analytics` wire value (now `local`).
- Self-describing default experiment names.
- Real streaming token-cost accounting with a plausibility clamp on
  self-reported costs; `budget.max_cost_usd` is sent on typed session create
  so the backend cost cap arms.

### Fixed
- Abort on systematic provider failures instead of reporting fake
  convergence; fail fast when the optimizer is unavailable instead of
  hanging.
- Offline-sync finalization is status-aware and skips terminal non-COMPLETED
  experiments; sync CLI surfaces terminal-skip warnings.
- Trial-results table no longer double-scales 0-100 metrics; cost capture
  works when the config has no `model` key.
- `FunctionDescriptor` identity, metric batching, and audit-alert
  ratchet-surfaced bugs.

### Security
- `RestrictedUnpickler` pinned to a precise (module, name) allowlist,
  closing a pickle deserialization RCE vector.
- CLI login rate limiting enforced; file-read containment for CWE-23.
- Sensitive tuned config values are redacted on the privacy-mode submission
  path; safety-sensitive-key blocking is non-bypassable on best-config
  drift; CI-approval token expiry compared in UTC.

## [0.18.0] - 2026-06-27

### Added
- Added a no-content-egress cloud-brain canary test that captures serialized
  session, next-trial, metrics, and finalize payloads and asserts dataset
  example content never crosses the backend boundary.

### Changed
- Rewrote execution-mode guidance around `algorithm` and `offline`: cloud-first
  `auto`, explicit local `grid`/`random`, zero-egress offline mode, result
  provenance, and legacy selector deprecations.

### Removed
- Removed public pruner exports: `CeilingPruner`, `CeilingPrunerConfig`,
  `StatisticalInferiorityPruner`, and `StatisticalInferiorityPrunerConfig`.
- Dropped the `optuna` runtime dependency.
- Removed `register_optuna_optimizers` and the legacy Optuna feature flag/env
  var (`OPTUNA_ROLLOUT`, renamed during the managed-routing migration).
- Smart algorithms (`bayesian` and the Optuna family) now route to the Traigent
  cloud; the local SDK supports `grid` and `random` only.
- Removed 7 production-dead internal modules (`config_builder`,
  `llm_processor`, `refactoring_utils`, `service_registry`, `event_manager`,
  `langchain/discovery`, empty `telemetry` package) and their tests.

### Fixed
- Per-config `cost` metric is now wired to the authoritative `total_cost`, so
  the `minimize cost` objective is no longer inert and the portal no longer
  shows `$0` per config on a real paid run; a model unpriced at runtime is
  surfaced on the result (and fails closed under
  `TRAIGENT_STRICT_COST_ACCOUNTING`).
- `traigent sync` / `traigent local sync` no longer fail every completed
  session with `409 EXPERIMENT_HAS_NO_RUNS`; sync now creates the experiment
  `PENDING`, uploads runs, finalizes to `COMPLETED`, resumes idempotently, and
  returns a non-zero exit on failure.
- Runtime `optimize(algorithm="grid"|"random")` with a portal key now stays
  local and exhaustive instead of routing to the cloud sampler.
- `traigent quickstart` runs on a bare install (uses base `litellm`, not the
  `integrations` LangChain extra) and fails closed when no trial succeeds.
- `ObjectiveDefinition` fails closed on unimplemented normalization strategies
  (`z_score`/`robust`) instead of silently behaving like `min_max`.

### Dependencies
- Bumped `langsmith` 0.8.3 → 0.8.18 and `pydantic-settings` 2.14.1 → 2.14.2.

## [0.14.1] - 2026-06-19

### Fixed
- **Cloud session creation HTTP 400 resolved.** `_typed_configuration_space()` now
  normalizes scalar fixed values (e.g. `temperature=0.0`) to single-choice categoricals and
  infers `"float"`/`"int"` types for untyped range dicts (e.g. `{"low": 0.0, "high": 1.0}`).
  Previously these passed through to the backend un-typed, triggering `VALIDATION_ERROR`
  and silent fallback to local-only mode (`cloud_url=None`).

## [0.14.0] - 2026-06-19

### Added
- **Composite telemetry rides the measures channel.** `composite_measures(run)`
  (`traigent.knobs.telemetry`, re-exported from `traigent.knobs`) flattens a composite
  run's RFC 0002 §3.10 content-free telemetry — `escalation_rate`, `stage_selected`, and
  the per-gate `gate_margin_pass_rate` map — into flat, identifier-safe, numeric-only keys
  (e.g. `composite_escalation_rate`, `composite_stage_selected`,
  `composite_gate_0_margin_pass_rate`). Merge it into the metrics your decorated function
  returns and the `composite_*` keys ride the existing per-trial measures wire channel as
  ordinary numeric metrics — no new wire surface. Keys are capped with headroom below the
  backend `MeasuresDict` 50-key ceiling (truncated deterministically with a logged warning,
  never raised mid-trial), and the output is content-free by construction (the adapter reads
  `run.measures` only, never `run.output`). New docs page
  `docs/concepts/composite-knobs.md` covers the pattern catalog, executing a composite,
  certified selection with a `binary_cascade`, and telemetry-to-measures; runnable offline
  example at `examples/advanced/composite-knobs/composite_telemetry.py`.

### Changed
- **Timing metrics now emit canonical millisecond keys** (#1225). Model latency is
  reported as `response_time_ms`, evaluation wall time is reported as
  `execution_time_ms`, and execution duration is no longer copied into
  `response_time_ms`. Legacy seconds keys `model_response_time` and
  `function_duration` remain populated for one minor-version compatibility window;
  remove them after the next minor once downstream dashboards have migrated.
- **Unpriced models now block instead of warn-and-continue.** When a real run includes
  models with no known pricing, the SDK now requires explicit confirmation: interactive
  terminals get a blocking prompt; non-interactive runs fail closed before any trial.
  Pre-approve with `cost_approved=True` (must be a real boolean) or
  `TRAIGENT_COST_APPROVED=true` (exact value), or supply custom pricing via
  `TRAIGENT_CUSTOM_MODEL_PRICING_JSON`/`_FILE`. `TRAIGENT_STRICT_COST_ACCOUNTING=true`
  still hard-fails without prompting; mock runs are unaffected.

### Removed
- **`BudgetStopCondition` class removed.** The class is no longer exported from any
  public module. Replace usage with `cost_limit=<value>` in `.optimize()` calls, or
  use `metric_limit=<value>` + `metric_name=<name>` for metric-based stopping.
- **`TRAIGENT_TRACES_ENABLED` environment variable removed.** The deprecated plural-form
  alias is no longer read. Use `TRAIGENT_TRACE_ENABLED=true` (singular) instead. Setting
  the old variable will have no effect and tracing will remain disabled.
- **`OptimizedFunction.config_space` property removed.** The property has been removed
  from the public API. Access the configuration space through the `TVLParameterAgent`
  interface or via decorator introspection helpers.
- **`strategy=` non-preset alias removed.** Passing an unrecognized string to `strategy=`
  in `.optimize()` now raises `ValueError` immediately rather than falling back silently.
  Use one of the documented preset names.
- **`budget_limit`, `budget_metric`, `budget_include_pruned` runtime override keys
  removed.** Passing any of these as keyword arguments to `.optimize()` now raises
  `TypeError`. Migrate to `cost_limit=<value>` for cost-based stopping, or
  `metric_limit=<value>` + `metric_name=<name>` for metric-based stopping.

## [0.12.0] - 2026-06-06

### Added
- **Content-logging opt-out for optimization logs** (#1069). `TRAIGENT_LOG_EXAMPLE_CONTENT=false`
  (or `OptimizationLogger(..., log_example_content=False)`) keeps per-trial ids and metrics on
  disk while omitting the per-example `query` / `response` / `expected` content — without having
  to enable full privacy mode. Defaults to on for DX. The logger now also writes a `.gitignore`
  (`*`) into the log root (`./.traigent/optimization_logs/` by default) so prompt/response
  content isn't accidentally committed, and `docs/api-reference/telemetry.md` documents the
  default location, what is persisted, and that on-disk redaction covers structured PII only.

### Removed
- **Breaking in 0.12.0:** removed Python-orchestrated JavaScript optimization through the temporary JS bridge. `ExecutionOptions.runtime`, all `ExecutionOptions.js_*` fields, `traigent.bridges.*`, and `traigent.evaluators.JSEvaluator` are no longer available. JavaScript/TypeScript users should migrate to native `@traigent/sdk` optimization with `optimize(spec)(agentFn)` and `await wrapped.optimize(...)`; see https://github.com/Traigent/traigent-js/blob/main/docs/getting-started/minimal-integration.md and https://github.com/Traigent/traigent-js/blob/main/docs/MIGRATION_FROM_PYTHON.md.

### Fixed
- **`BenchmarkClient` now attributes generation to the active project** (#1066). It read
  `TRAIGENT_PROJECT_ID` but applied it to the bare `/api/v1` root, which `scope_api_path`
  leaves unrewritten — so `generate_sync` posted to the unscoped `/api/v1/datasets/generate`
  and the benchmark was not project-scoped (a tenancy / data-isolation gap for multi-project
  tenants). With a project set it now targets the project-scoped backend route
  `POST /api/v1beta/projects/{id}/benchmarks/generate` (the `/datasets/generate` alias is not
  project-scoped on the backend; the project-scoped generate route is registered under
  `benchmarks`). With no project set, behavior is unchanged.

### Security
- **Standalone backend clients now honor `TRAIGENT_OFFLINE_MODE`** (#1068). Seven publicly-exported clients — `EvaluationClient`, `PromptManagementClient`, `EnterpriseAdminClient`, `ProjectManagementClient`, `CoreMetricsClient`, `BenchmarkClient`, `ExampleInsightsClient` — previously ignored offline mode and performed outbound HTTP (carrying evaluator definitions, prompt text, tenant/admin records, per-example features) even with `TRAIGENT_OFFLINE_MODE=true`, breaking the documented air-gapped guarantee. They now **fail closed** at the request boundary with a clear `OfflineModeError` (new, `traigent.utils.error_handler`) via the shared `raise_if_backend_offline()` guard — so a caller learns the request was deliberately blocked instead of silently leaking off-box. Behavior is unchanged when offline mode is off.
- SDK auth and security policy checks now use `treat_as_production_policy()`, which treats unset or unknown environment names as production-safe by default. Local development JWT validation and mock password auth now require an explicit non-production environment such as `ENVIRONMENT=development` or `TRAIGENT_ENV=development`.
- Bump `litellm` floor from `>=1.83.0` to `>=1.83.7` to fix [GHSA-xqmj-j6mv-4862](https://github.com/BerriAI/litellm/security/advisories/GHSA-xqmj-j6mv-4862) — prompt-template SSTI in `POST /prompts/test` that could run arbitrary code in the LiteLLM Proxy process. 1.83.7 renders templates in a sandboxed Jinja environment.
- Bump `langchain-core` floor `>=1.2.22` → `>=1.2.28` (CVE-2026-40087 — improper element neutralization in templates).
- Bump `langchain-openai` floor `>=0.3.30` → `>=1.1.14` (GHSA-r7w7-9xr2-qq2r — SSRF). Major-version-line bump; lock currently resolves to 1.2.x.
- Lock-only bumps for transitive deps to clear Aikido CVEs: `fonttools` 4.60.1 → 4.62.1 (CVE-2025-66034), `langgraph-checkpoint` 3.0.0 → 4.0.2 (CVE-2026-27794, unsafe deserialization → RCE), `filelock` 3.20.0 → 3.29.0 (CVE-2025-68146, CVE-2026-22701), `sqlparse` 0.5.3 → 0.5.5 (GHSA-27jp-wm6q-gp25), `pygments` 2.19.2 → 2.20.0 (CVE-2026-4539). Two further uv.lock advisories — `ragas` (CVE-2026-6587, SSRF) and `diskcache` (CVE-2025-69872, code injection) — have no upstream patch yet and remain open; tracked separately.

### Migration notes
- **`langgraph-checkpoint` 3.x → 4.x:** the 4.0 release drops default deserialization of payloads serialized with the legacy `"json"` serde mode. Users running `langgraph` with a persistent checkpoint saver (SQLite, Postgres, Redis, etc.) whose stored state contains JSON-format blobs will see deserialization failures unless they configure `serde` with an explicit `allowed_json_modules` list. Traigent does not bind `langgraph-checkpoint` directly — it arrives transitively through the `langgraph` dependency — so this only affects projects that also use `langgraph` checkpointers in their own code. See [CVE-2026-27794](https://github.com/langchain-ai/langgraph/security/advisories) for the underlying RCE that motivated removing the default.

### Changed
- **CI approval gate:** running `optimize()` in local/offline mode within a CI
  environment now requires explicit approval via `TRAIGENT_RUN_APPROVED=1`.
  `TRAIGENT_MOCK_LLM` no longer bypasses this gate — mock and offline runs in
  CI are intentionally included (a purpose-built approval signal replaces an
  env-var bypass of a security control). Cloud-mode runs are unaffected.
  Migration: set `TRAIGENT_RUN_APPROVED=1` in CI for legitimately-approved
  runs.
- **Strict evidence modes now fail closed in promotion** (#1103,
  FR-SDK-FAIL-CLOSED-PROMOTION-V1). When a TVL promotion policy declares a strict
  evidence mode — `require_calibration.enabled: true` or `chance_constraints` —
  the verdicts *no decision*, *insufficient samples*, and *gate exception* now
  WITHHOLD promotion instead of silently falling back to the permissive simple
  comparison, and the terminal selector returns either the gate-certified
  incumbent verbatim or an explicit no-winner result
  (`reason_code="NO_CERTIFIED_SELECTION"`, empty `best_config`,
  `best_score=None`) instead of re-deriving a winner by raw score.
  **Behavior change for existing specs that already declare
  `chance_constraints`:** runs whose evidence never certifies a winner now
  return the explicit no-winner result; no best config is applied or
  snapshotted, so `export_best_config()` raises `ConfigurationError` instead
  of exporting an uncertified config. Runs with certified winners, and all
  specs without strict declarations, are unchanged. First-trial incumbency
  counts as initialization, not certification.
- **Dual-licensed** under `AGPL-3.0-only OR LicenseRef-Traigent-Commercial`: declared the SPDX
  license expression in package metadata (PEP 639 via `setuptools>=77`), added `license-files`,
  per-module SPDX headers, and `COMMERCIAL-LICENSE.md`; aligned `NOTICE`, `README`, and
  `DISCLAIMER`. Commercial terms remain available under separate agreement; see
  `COMMERCIAL-LICENSE.md` and `CONTRIBUTOR-LICENSING.md`. No version bump in this change.
- Agent platform helpers now distinguish executable platforms from configuration-mapping-only platforms: `get_supported_platforms()` reports only executor-backed platforms, while `get_mapping_platforms()` returns all registered mappings.
- Agent executors without real cost-estimation support now raise `NotImplementedError` instead of returning an authoritative-looking zero-cost estimate.
- Langfuse trace metrics now include an `observations_partial` numeric flag in `to_measures_dict()` when observation pagination returns incomplete metrics.
- `OptimizationJob.wait()` now raises `PlatformCapabilityError` with an actionable
  experimental-feature message instead of `NotImplementedError`. This is an
  intentional public exception-type change to remove a concrete public
  `NotImplementedError` stub; callers that handled the old error should catch
  `PlatformCapabilityError` or `TraigentError`.
- Cost enforcement invariant checks now treat the cost-limit bound as an admission-time permit rule. `assert_invariants()` still detects stranded permits and reservation drift, but it no longer flags valid post-trial actual-cost overruns while other admitted permits remain in flight.
- `metric_limit` in parallel mode is evaluated after each batch and may overshoot by up to `parallel_trials - 1` trials; use `cost_limit` when a hard spend bound is required.
- Deprecated `budget_limit` stop-condition aliases now report `OptimizationResult.stop_reason="metric_limit"` instead of `"cost_limit"`. Use `cost_limit` for hard USD spend control and `metric_limit` with `metric_name` for soft cumulative metric stopping.
- Built-in hypervolume convergence now reports `OptimizationResult.stop_reason="convergence"`; arbitrary custom stop conditions still report the generic `"condition"` reason unless explicitly mapped.

### Fixed
- Trial tenants now pass quota checks until `trial_ends_at` and are blocked after expiry.
- Unsupported hybrid keep-alive now marks the session status as `unsupported` instead of treating every tracked session as alive.

## [0.11.4] - 2026-04-04

### Fixed
- Mock mode reported 0% accuracy when callers passed custom `metric_functions` because the mock evaluator returned a constant value that didn't satisfy the user-provided metric closure. The mock now honors registered `metric_functions` so mock-mode optimizations produce non-trivial scores (#649).

## [0.11.3] - 2026-04-04

### Changed
- Release-prep version bump on top of 0.11.2; no user-visible behavior change. Published to PyPI at 2026-04-03T22:27:47Z (2026-04-04 local release date) to align metadata with the next-patch release pipeline.

## [0.11.2] - 2026-04-01

### Fixed
- Quickstart accuracy always 0% with real LLMs - add system prompt for concise answers and use contains-match instead of exact-match (#642)
- CI publish verification retry improvements (#640)

## [0.11.1] - 2026-04-01

### Fixed
- Quickstart fails when run from any directory other than project root - set `TRAIGENT_DATASET_ROOT` to package directory (#636)
- PyPI publish verification broken pipe - save curl response to file instead of piping (#635)
- Consistent `resolve()` for dataset path in quickstart

## [0.11.0] - 2026-03-30

### Added
- **DeepEval metric integration** - `DeepEvalScorer` bridges DeepEval metrics into Traigent's `metric_functions` system
- **Rich results table** - Formatted ASCII table with box-drawing characters for optimization output
- **Interactive vendor error handling** - Pause/resume prompts on rate limits and budget exhaustion
- **Evaluation kwargs** - Configurable per-evaluation parameters for hybrid wrapper service
- **Run labeling** - Optional `run_label` on `OptimizationResult` for experiment tagging
- **Session creation circuit breaker** - Graceful fallback when backend is unreachable
- **RAG walkthrough demo** - New `walkthrough/demo/optimize_rag.py` example

### Changed
- Default backend URL changed from `localhost:5000` to `https://portal.traigent.ai` - SDK devs should set `TRAIGENT_BACKEND_URL=http://localhost:5000` explicitly
- `ExperimentRunDTO.status` default changed from `"pending"` to `"not_started"`
- `HybridEvaluateRequest.config` type narrowed from `dict[str, Any]` to `dict[str, ScalarValue]` (`config` is now a deprecated alias for `kwargs`)
- Keyring credential storage removed in favor of file-only storage (`~/.traigent/credentials.json` with `0600` permissions) to avoid macOS Keychain prompts
- Schema validator package renamed from `optigen_schemas` to `traigent-schema`

### Removed
- **`benchmark_id` from `HybridExecuteRequest` and `HybridEvaluateRequest`** - Breaking change for external hybrid service consumers
- **`benchmarks_revision` from hybrid request DTOs**
- `hello.py` replaced by `hello_world.py`
- `traigent/experimental/` package (simple_cloud platforms)
- `traigent/conftest.py` (moved to `tests/conftest.py`)

### Fixed
- Credential file write TOCTOU race fixed with atomic `os.open()` + mode `0o600`
- Credential file read now verifies and tightens permissions if overly permissive
- Auth rejection log spam suppressed with instance-scoped deduplication at DEBUG level
- `pydantic` added to core dependencies
- `traigent-schema` made optional for public installs
- Missing `aiohttp.ClientTimeout` added to `DatasetConverter` and OAuth2 sessions
- Vendor error classification anchored to contextual patterns (`"status 429"` instead of bare `"429"`)
- `_safe_copy` in DeepEval integration now raises `TypeError` instead of silently returning shared mutable instance
- Silent `except Exception: pass` in `_finalize_optimization` replaced with debug logging
- Dead methods removed from orchestrator (`_check_batch_vendor_failures`, `_maybe_pause_on_cost_limit`, `_handle_vendor_pause_in_loop`)

### Breaking Changes
- `create_session()` return type changed from `str` to `SessionCreationResult` dataclass.
  `SessionCreationResult.__str__` returns the session ID so most existing code continues
  to work without changes, but explicit `isinstance(result, str)` checks will need updating.

  **Before (v0.10.x):**
  ```python
  session_id: str = create_session()
  ```
  **After (v0.11.0+):**
  ```python
  result = create_session()
  session_id: str = result.session_id   # explicit
  # or: str(result)                      # via __str__
  ```
- `HybridExecuteRequest` and `HybridEvaluateRequest` no longer accept `benchmark_id` parameter
- Default backend URL is now cloud instead of localhost

## [0.10.0] - 2026-02-07

### Added
- **LangChain/LangGraph native callback handler** - Native instrumentation for LangChain and LangGraph workflows via `TraigentCallbackHandler`
- **Langfuse observability bridge** - Integration with Langfuse for unified observability across platforms
- **Namespace parsing utilities** - Agent namespace extraction from span names for multi-agent optimization
- **Agent-specific metrics utilities** - Automatic per-agent metric aggregation for multi-agent workflows
- **Workflow traces visualization** - Graph-based visualization of multi-agent workflow execution
- Multi-agent parameter and measure mapping system
- `AgentConfiguration` types for explicit agent groupings
- `agent` parameter on `Range`, `Choices`, `IntRange` classes
- `agent_prefixes` for prefix-based agent inference
- TVL support for `agent` field on tvars
- Content scoring and data integrity improvements
- Pre-rendered architecture diagram (replaces inline Mermaid)
- Click-to-play demo thumbnails with animated SVG playback

### Changed
- **Root directory reorganization** - Consolidated 40+ root items to ~14
- Moved `baselines/` to `configs/baselines/`, `runtime/` to `configs/runtime/`
- Consolidated `mypy.ini` and `pytest.ini` into `pyproject.toml`
- Backend session metadata now includes `agent_configuration`
- Improved type safety across API types
- Improved node context restoration on chain end to prevent metric misattribution
- README trimmed ~30% with all examples fixed to include required params

### Fixed
- Agent ID validation to fix edge cases in multi-agent workflows
- Mock mode metrics simulation for cost and accuracy in demos

## [0.9.0] - 2025-01-09

### Added
- Core optimization decorator (`@traigent.optimize`)
- Grid, Random, and Bayesian optimization strategies
- TVL (Traigent Variable Language) specification support
- LangChain and DSPy integration adapters
- Edge analytics execution mode
- Cloud and hybrid execution modes
- Privacy-preserving optimization mode
- Parallel trial execution support
- Stop conditions (plateau detection, max trials)
- Plugin architecture for extensibility

### Changed
- Refactored plugin architecture for modularity

### Security
- JWT-based authentication for cloud operations
- Input validation and sanitization
