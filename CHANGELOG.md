# Changelog

All notable changes to Traigent SDK are documented in this file.

Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Fixed

- **`traigent hooks install --force` no longer destroys the hook you already
  had (#2707).** Backing up used `hook_path.rename(backup_path)`, and `rename`
  overwrites: a second `--force` replaced the saved original with Traigent's
  own hook, and `uninstall` then restored that instead, leaving the
  customer's hook unrecoverable. Three ordinary commands were enough.
  Ownership is now recorded rather than guessed. `install` writes the digest
  of each artifact it publishes, and of any original it preserves, to
  `.git/traigent-hooks.json`; `status` and `uninstall` read it. The SDK
  removes exactly what it generated and never deletes or resurrects a hook it
  did not: a preserved original is recorded before the backup is created, a
  refused install claims nothing, a saved original is never collected as a
  stale artifact even if a later release recognises its bytes, and a backup
  with no record is left exactly where it is rather than restored over your
  hook. Publication no longer overwrites: hooks are staged and linked, and
  every mutation re-checks the file it is about to touch. `traigent hooks
  status` now also tells outdated, unsafe and unreadable hooks apart, and
  reports a backup left behind. The record lives inside `.git`, is never
  committed, is read without following symlinks, and degrades to the previous
  content-only behaviour with a warning if it is missing or unreadable. The
  generated hook scripts themselves are unchanged.

  Two consequences of deciding ownership exactly, both user-visible. A
  Traigent hook **you have edited** is no longer removed by `uninstall`: it no
  longer matches an artifact Traigent generated, so it is reported skipped and
  left for you, where before the marker in its header was enough to delete it.
  And `uninstall` now removes the **pre-commit** hook: ownership was previously
  decided by looking for the *pre-push* header in both files, so a generated
  pre-commit hook never matched and was always left behind.

- **Trial error text and three metadata maps now cross to the backend as closed codes and allowlisted keys (#2241).** `error_message` on trial submission and config-run status is reduced to one fixed code (for example `trial_failed` or `no_examples_succeeded`); the full text stays on the local result. `metadata.agent_configuration`, `metadata.surrogate_evaluator` and `summary_stats.metadata` are rebuilt from a key allowlist, so unlisted keys and judge prompts no longer leave the SDK.
- **TVL derived constraints are documented as per-trial only; the loader warns when a spec relies on `environment.context` (#1618).** `load_tvl_spec` emits one value-free warning naming the derived constraint ids; `docs/tvl/CONSTRAINT_EXPRESSIONS.md` states the rule.
- **Keyless quickstart is labelled a plumbing check, with no fake winner (#1690).** `traigent quickstart` now ends with a message saying the run used a mock LLM and simulated scores, is not a model comparison, and found no best configuration; the `best config ... scored 0.850` line is gone. Achievements stay blocked in mock runs.
- **Run-group table across runs (#1711, slice 1).** `client.experiment_groups.table(group_id, project_id)`
  returns one `RunGroupTable` of every configuration-run in a run group: run id, run name,
  config, metrics, cost, status and `created_at`. An unmeasured cost is `None` (with
  `cost_unmeasured`), never `0`. `to_dataframe()` needs the
  optional pandas extra. Also fixes the group list/configuration-runs reads to send
  `per_page` and read nested `pagination` (they were ignored before). See `docs/user-guide/run_group_table.md`.

- **Anthropic key probe uses a current Claude id (#2651).** `validate_providers` probes with `claude-haiku-5-5` (was `claude-haiku-4-5-20251001`); the known-models list carries the Claude 5.5 ids and no retired `claude-3*` ids.
- **Dev tooling now agrees with CI (#2469).** `release-review.yml` formats with
  `ruff format --check` (as the required PR gate does) instead of black; the `dev`
  extra exact-pins `isort==9.0.1` and `mypy==2.3.1` (the versions in
  `.pre-commit-config.yaml` and `uv.lock`); `scripts/local_gate.sh` tests generated
  schema types against the commit in `scripts/ci/schema-pin.txt` (exporting it from
  the sibling checkout when that is at another commit, failing closed when the pin
  is unreadable or absent).
- **BREAKING (packaging): the SDK no longer bundles `traigent/skills` (#1609).**
  The folder and its package-data entry are removed; the skills live only in
  `Traigent/traigent-skills` (`npx skills add Traigent/traigent-skills`, which
  `traigent onboard` already runs). The runtime message that pointed at the bundled
  folder now points to installing `traigent-skills`. The bundle-only content guards
  (lifecycle-vocabulary lint, decorator-skill snippet test, MLflow skill example
  check) go with it; the tracker no-local-default check stays.
- **Nous provider auth flow is marked UNVERIFIED (#2067).** The six "confirm
  before merge" OWNER markers are now explicit UNVERIFIED notes (token endpoint,
  `auth.json` schema, refresh request shape, base URL, model-id form), the
  provider logs one value-free warning at first use, and
  `docs/guides/nous-provider.md` states the flow was not checked against the live
  service.

- **`generate-config -o tvl` structural constraints now load (#2647).** The
  generator's known constraint shapes (`require(...)`, `implies(...)`,
  `lambda config: ...`) are translated to the `expr` / `when`-`then` forms the
  SDK's own `load_tvl_spec` accepts. A shape outside that grammar is skipped
  with a value-free warning naming only the constraint id, instead of being
  written as an unloadable `code` entry.

### Changed

- **Honest run cost totals; unmeasured-cost floor in its own field (#2688).** When
  any trial's cost is unmeasured, `OptimizationResult.total_cost` and
  `metrics["total_cost"]` are now unknown (`None` / absent) instead of a partial
  sum; `metrics["cost_floor_usd"]` (known costs plus the floors of unmeasured
  trials) and `metrics["cost_unmeasured_trials"]` report the exact lower bound
  and count. Unmeasured trials are submitted with `cost_unmeasured = 1.0` and
  their floor as top-level `cost_floor_usd` in the results body; budget-armed
  sessions additionally keep `metrics.cost = floor` as a temporary bridge until
  the backend reads `cost_floor_usd`; other sessions send no `metrics.cost`.
  `SimpleScoringEvaluator`
  no longer hands placeholder `$0` costs (unmeasured or unpriced usage) to the
  scoring function, progress callback or trial metrics.

- **Finalize propagates ownership/auth errors (#2510).** HTTP 401/403 from the
  session finalize call now raise `CloudServiceError` instead of being swallowed
  as "endpoint unavailable"; only an unavailable endpoint or transport failure
  falls back to backend auto-finalization.
- **Safety compliance target is documented as derived (#2204).** The required
  compliance rate stays derived from a constraint's operator and value
  (`above(x)` needs `x`, `below(x)` needs `1 - x`); explained in
  `docs/features/safety-compliance-target.md`, with a regression test that
  `above()` and `below()` at the same threshold derive different rates.
- **Mock-provenance metadata is documented and guarded (#2493).** Results keep
  the `mock_llm_run` / `mock_llm_mixed` metadata keys; they are described in
  `docs/user-guide/cost_capture.md` and the `traigent-analyze-results` skill, and
  tests show they survive save and load while a legacy result without them is
  reported as unknown, not as a real run.
- **No-model-call examples point at `with_usage(total_cost=0)` (#2449).** The
  unmeasured-cost error and warning now say plainly that a path making no model
  calls should wrap its return value in `traigent.with_usage(value,
  total_cost=0)` to record a measured $0; unknown stays the default. Documented in
  `docs/guides/execution-budget.md`, with a test that a wrapped no-call example
  counts as measured $0 and an unwrapped one stays unknown.

- **User-cancelled runs are now clearly labelled (#2666).** `optimize()` still
  returns the partial result after a Ctrl+C (the caller owns the exit code),
  with `status` CANCELLED and `stop_reason` `user_cancelled`. It now logs one
  warning with counts only (N of M trials finished, result is partial), and
  `traigent results list` has a Status column. Results saved before this
  change show `unknown`. Progress, summary and logging callbacks and the CLI
  summary now say "CANCELLED BY USER" and that results are partial, instead of
  "timeout reached" or "complete".

- **`traigent hooks validate --exit-code` fails on an empty explicit target
  (#2639).** When a target path is given explicitly and contains no
  `@traigent.optimize` functions, the command now exits non-zero; pass
  `--allow-empty` to keep exit 0. With no path (default `.`, as the installed
  pre-push hook runs it) behaviour is unchanged. Without `--exit-code`
  nothing changes.
- **Partially acknowledged runs now report `local_fallback` (#2027).** When the
  backend acknowledges only some of a run's trials, `result_source` /
  `source` is `local_fallback` instead of `cloud_brain`, because the run was
  not fully cloud-confirmed.
- **A run where every trial failed now ends `FAILED` with
  `ALL_TRIALS_FAILED` (#2477),** regardless of the stop reason, including
  `cost_limit` and `execution_budget` stops. The `stop_reason` value itself is
  unchanged.
- **Mock mode no longer skips the cost preflights (#2681).** The pre-run cost
  approval and the model-price coverage preflight now run under plain
  `TRAIGENT_MOCK_LLM`, unless `TRAIGENT_COST_PREFLIGHT_MOCK_EXEMPT=true` is set
  (honored only in mock mode outside production). Fully mocked tests and keyless
  walkthroughs that relied on the skip must set the exemption.
- **BREAKING: non-string hybrid example ids are refused (#2575).** When the
  id a row resolves to (`example_id`, else `input_id`) is not a string (an
  int, float, bool, list or dict), the trial now fails with a
  `ValidationError` naming the position, the field and its type. The hybrid
  protocol defines these ids as strings echoed exactly; a dataset with unique
  int ids that ran before must convert them to strings. A falsy non-string
  `example_id` or `input_id` (`0`, `False`) is refused too; it used to be
  replaced by the `input_id` or the fallback id.

- **BREAKING: a run with `cost` as an objective now refuses to record an unpriced
  call as $0 (#2517).** Strict cost accounting is on by default, so such a run
  raises `OptimizationError` (with `CostTrackingRequiredError` as its
  `__cause__`) instead of ranking the unpriced model cheapest. Remedy: report usage with `traigent.with_usage(...)`, or give
  per-token prices through `TRAIGENT_CUSTOM_MODEL_PRICING_JSON` /
  `TRAIGENT_CUSTOM_MODEL_PRICING_FILE`; set
  `TRAIGENT_STRICT_COST_ACCOUNTING=false` only to turn the error into a warning:
  the unpriced call stays flagged `cost_unmeasured` and is never recorded as $0.
  A trial whose cost is only partly measured (some example reported no usage,
  or a call inside one example reported none) stops the run the same way, and
  the error names up to 5 such examples. A failed example with no cost
  telemetry counts as unmeasured too, unless it reports a cost (`0` included)
  or `provider_attempts: 0`. Fix it
  by wrapping a path that makes no model call in
  `traigent.with_usage(value, total_cost=0)`, adding a price for an unpriced
  model (`TRAIGENT_CUSTOM_MODEL_PRICING_JSON` / `_FILE`), or setting
  `TRAIGENT_STRICT_COST_ACCOUNTING=false`. Without a cost objective and without
  an explicit cost limit the run continues with a warning; an explicit cost
  limit fails closed the same way, because it cannot bound an unknown spend.
  Spend that was reported for a trial flagged `cost_unmeasured` (its
  `cost_lower_bound`) is charged to `cost_limit`, `ExecutionBudget` and the
  cost enforcer as a minimum; the trial keeps its `cost_unmeasured` flag and its
  objective value is unchanged.

### Fixed

- **Pre-run cost estimate samples the dataset and shows its assumptions
  (#2498).** Input tokens per example are now estimated from the first 20
  dataset examples (~4 ASCII chars/token, 1 token per non-ASCII char), plus a
  200-token prompt-template allowance (minimum 256), so short inputs can
  estimate below the old 2000-token default; a configured `max_tokens` in the config space is
  the output cap; backend overrides still win and the 1.2x buffer is kept. The
  cost-approval refusal lists model(s), example count, token estimates, source
  and buffer, and states that real billing may differ and the runtime cost
  limit still applies. No example text appears in the message.

- **Provider-billed cost is no longer replaced by the plausibility floor (#2505).**
  A cost read from the provider's own response (`usage.cost`, e.g. OpenRouter's
  billed charge) is tagged `provider_billed` and skips the 50%-of-table-price
  floor, so a cheaper upstream route is recorded as billed instead of at a stale
  table price. Self-reported costs (`with_usage()`, injected `__traigent_meta__`,
  raw paths) and any untagged cost keep the floor and its warning. Each cost now
  carries a source (`provider_billed`, `self_reported`, `table`); the trial's
  `summary_stats["metadata"]["cost_source"]` holds it (or `mixed`) and the
  numeric measure `cost_provider_billed_fraction` gives the billed share.

- **Model defaults no longer point at retiring models (#2650).** The `model`
  range preset drops `gpt-4.1-mini`/`gpt-4.1-nano` for `gpt-5.4-mini`/
  `gpt-5.4-nano`; the OpenAI executor's model allow-list is now derived from the
  SDK's known-models table (case-insensitive, retired families still rejected);
  and `Choices.model()` raises `ValueError` for a provider with no defaults
  instead of silently returning an OpenAI model. `DEFAULT_MODEL` is unchanged.

- **Revived traces keep their content policy and clearing markers (#2604).**
  A late update to an evicted trace now resolves its content mode from the
  trace's own tombstone (a malformed one fails closed to `metadata`), so a
  trace set to `metadata` never sends content under a looser client `record`
  policy. The tombstone also keeps per-field "content was delivered" flags, so
  an `end_trace`/`record_observation` that withholds content sends the clearing
  marker as it would for a live trace. Only the last 4096 evictions are
  remembered: an update to a trace evicted longer ago raises `Unknown
  trace_id`; set a longer `trace_state_grace_seconds` if late updates are
  expected.
- **Ended, delivered traces are evicted from the observability client (#2604).**
  `ObservabilityClient` no longer keeps every trace it ever started. A trace
  that has ended and whose latest snapshot the backend accepted is dropped from
  memory after a grace window (`trace_state_grace_seconds`, default 300 s).
  Traces that are still running, queued, in flight or failed stay for retry.
  Restarting an evicted trace id starts a fresh snapshot.
  Eviction depends on a per-item settled mark kept until the trace is evicted
  (not a bounded delivery cache), and a later update to an evicted trace id
  (`record_observation`, `end_trace`) starts a fresh snapshot instead of raising;
  only ids the client never held still raise `Unknown trace_id`.

- **TVL float ranges that declare `resolution` are now a grid (#1612).** A
  float `range` with a positive `resolution` loads as the discrete values
  `min, min+resolution, ...` (never above `max`); without `resolution` it stays
  continuous (unlike the official TVL default step of 0.001). Grids over 10,000
  points are rejected.
- **TVL registry domains without a `registry_resolver` fail at load on purpose
  (#1616).** The docs now say so: it avoids starting an optimisation with a
  missing setting and differs from the official TVL lazy placeholders.
- **A strict cost-objective run now stops after its first successful trial
  that captured no usage, and keeps its result (#2421).** Previously the run
  billed every trial before raising `UnknownModelError`, and the raise
  discarded the result. The run now stops with `stop_reason="no_usage_captured"`
  (scheduling of new trials stops; trials already in flight may finish), and
  the `UnknownModelError` carries the finished trials as `err.result`. Mock
  LLM mode and `TRAIGENT_STRICT_COST_ACCOUNTING=false` keep the warning
  behaviour; failed trials never trigger the stop.
- **Mock mode now blocks real requests to known LLM provider hosts (#2418).**
  Raw `openai` / `anthropic` clients inside an optimised function, and LiteLLM
  calls that supply their own `litellm_logging_obj`, used to reach the real
  provider (billed if a key was set) while `TRAIGENT_MOCK_LLM=true` or
  `enable_mock_mode_for_quickstart()` was on. While mock mode is on, any
  httpx (including `httpx2`), aiohttp or `requests` request to a fixed list of
  provider hosts now raises `MockModeTransportBlockedError` before connecting.
  The error names only the host, never the path, query, headers or body. The
  guard is inert when mock mode is off, and enabling mock mode fails closed if
  it cannot be installed.

- **`auto_override_frameworks=True` was a silent no-op for sync functions on
  0.27.0 and earlier (#2420).** The override gate is a `ContextVar`, and the
  sync evaluator lane ran the function in a worker thread without the caller's
  context, so the patched framework constructor built the hard-coded model for
  every trial while the results table showed distinct configs. The sync lane now
  runs the function inside `copy_context()` of the caller (commit 02413c33,
  first in the v0.28.0 tag). Async functions were never affected. A regression
  test runs `optimize_sync()` on a sync function and asserts each trial's
  constructor values
  (`tests/unit/evaluators/test_sync_auto_override_context_2420.py`).

  | traigent | auto-override (sync fn) |
  |---|---|
  | 0.27.0 and earlier (checked: no context hand-off in the 0.23.0 to 0.27.0 sources) | no-op, silent |
  | next release containing 02413c33 (the v0.28.0 tag does) | applied |

- **A real `publish_best_config()` no longer raises `CloudBestConfigIntegrityError` after the
  Backend has already committed the publish.** The SDK hashed `provenance` and the unfiltered
  `validation` into a best-config `spec_hash`, the Backend (since #819) hashes neither, so every
  publish looked failed while the server had moved the head. `compute_spec_hash` now applies the
  one frozen `traigent.best_config.v1` rule (TraigentSchema `best_config_schema.json`
  `x-traigent-spec-hash`): provenance not covered, claim/result keys removed from `validation`
  recursively, trimmed `config_id`, effective environment always hashed. Publish verification
  hashes the spec the Backend returned, not the request. A repo or cloud-cache manifest written
  with the old rule now fails closed with "written with an older spec_hash rule" (regenerate it
  or refetch); no data is migrated silently.

- **Observability ingest requests now fit the backend's per-request limits
  (#2600).** The transport used to cut requests only by `batch_size` (up to
  10,000) and bytes. The backend accepts at most 100 traces (422 above it) and
  1,000 observations, nested children included (413 above it), so batches
  above either limit failed and were retained and re-sent unchanged on every
  flush. Each request now carries at most 100 traces and 1,000 observations.
  `batch_size` above 100 is still accepted; it sets how many buffered traces
  trigger a background flush, and that flush is sent as several requests. A
  single trace with more than 1,000 observations can never be accepted, so it
  is now dropped locally (`dropped_by_reason["too_many_observations"]`, named
  in `FlushResult.errors`) instead of blocking the traces batched with it.
  These limits apply only to the SDK's own ingest path; a custom `sender=`
  receives batches cut by `batch_size` and `max_batch_bytes` as before.
- **A rejected trace update no longer lets an older snapshot of the same trace
  go out (#2601).** When an update was dropped (`payload_too_large`, not JSON
  serializable, or `too_many_observations`), the older snapshot already queued
  for that trace was still sent, and the backend stored a stale partial trace
  behind a successful flush. That older snapshot is now dropped with the
  update; if it is already being sent it cannot be recalled, but a failed send
  no longer queues it for a retry. The trace id is named in
  `FlushResult.errors` as before.
- **`flush()` / `close()` no longer re-send traces the backend already has
  (#2604).** Every flush re-submitted the latest snapshot of every trace the
  client had started, and each was sent again. A snapshot identical to the one
  last delivered for that trace is now skipped
  (`get_stats()["skipped_unchanged_items"]`); a changed or never-delivered
  trace is sent as before, and so is any trace whose last delivery came back
  with backend warnings. The transport remembers delivered snapshots for the
  most recent `max_queue_size` traces (10,000 by default); beyond that, older
  traces are sent again on the next flush, as before. `FlushResult.items_sent`
  no longer counts skipped repeats. The client still keeps every trace it
  started in memory; that is unchanged.
- **Suppressed observability delivery is now reported (#2602).** With offline
  mode, `TRAIGENT_DISABLE_TELEMETRY`, or no credential, `flush()` and
  `close()` returned `success=True, items_sent=0` and no warning, the same as
  a real delivery. `success` stays `True` so telemetry never breaks the host
  application, but `warnings` now names the cause and says no traces were
  sent.
- **Offline-mode errors from observability control-plane calls name the real
  cause (#2603).** `submit_feedback`, `add_comment` and the `list_*` / `get_*`
  calls said "skipped because TRAIGENT_OFFLINE_MODE=true" even when
  `TRAIGENT_DISABLE_TELEMETRY`, `TRAIGENT_OFFLINE` or
  `ObservabilityConfig(offline_mode=True)` caused it. The message now names
  each offline variable that is set, or the config flag.
- **Dead and non-public links removed from published surfaces (#2598).** The
  PyPI Documentation URL and the docs site's `site_url` pointed at a domain
  that does not resolve; both now use the docs site
  (https://traigent.github.io/Traigent/). The "API keys detected in code"
  warning linked a domain Traigent does not own; it now links the secret
  management guide. The docs nav's dead "JavaScript SDK" entry is removed
  until JS SDK docs are published, and doc links to a dead community forum and to
  repositories the public cannot open were removed. A new offline test fails
  if any of these come back.
- **`Dataset.to_jsonl` keeps an existing file's permission bits when it
  replaces it.** Re-exporting over an owner-only (`0600`) file under a `022`
  umask used to leave it `0644`. A new file still follows the umask, like
  `open(..., "w")`. Owner and group are those of a new file, as before.
- **The outcome-matrix merge warning no longer logs trial ids.** A trial id
  is a free-form string set by whoever built the result and can carry content,
  so the warning now names the trial by its matrix column index.
- **`TraceIngestionResponse(success=True)` without `spans_ingested` is now an
  unconfirmed count, not a confirmed zero**, so `submit_traces` no longer warns
  that every span went missing. Pass `spans_ingested=0` to state zero. The
  two fields are annotated `int | None` / `bool | None` so an omitted value is
  detectable; after construction they are always an `int` and a `bool`.
- **An unmeasured cost is no longer stored, shown or ranked as $0 (#2517, #2518, #2446).**
  When no LLM usage is captured for an example (for example a raw provider
  client the interceptors do not wrap), the local lane used to estimate tokens
  and record `cost: 0.0` as a measurement. The trial now omits `cost` /
  `total_cost`, sets `cost_unmeasured: 1.0`, keeps the estimated tokens
  labelled with `tokens_estimated`, and the results table prints `unmeasured`.
  The Pareto front excludes a trial missing a configured objective and the batch
  composite score no longer rescales over the objectives that remain. A run
  stopped by the SDK default `max_trials` or the unmeasured-cost safety limit
  records and prints why (`metadata["stop_detail"]`).

- **A hybrid service that reports no cost is unmeasured, not $0 (#2517).**
  `TraigentService` no longer sends `cost_usd: 0.0` for an execute handler that
  returned no cost: the per-item field is omitted, and a batch where any handler
  (or a handler that raised) reported none sends `operational_metrics.cost_unmeasured`
  next to the priced lower bound it must still send as `total_cost_usd`.
  `HybridAPIEvaluator` treats that flag, an omitted or negative cost, and a
  failed row with no reported cost as unknown: the trial carries `cost_unmeasured`
  and no cost total, a per-item `metrics.cost_unmeasured` from the service can no
  longer overwrite that verdict, and the optimization logger and outcome matrix
  never write `0.0` for it. A workflow span sends the known spend floor (0.0 when
  none was reported) as `cost_usd`, always with
  `metadata["traigent.cost_unmeasured"] = true` so consumers can tell it apart
  from a measured cost. Run summaries
  (`get_summary()`, `experiment_stats`) withhold `total_cost` when a trial is
  unmeasured and report `cost_unmeasured_trials` / `cost_lower_bound`.

- **`ObservabilityClient` multi-tag trace/session filters no longer return an
  empty page (#2573).** `list_traces` / `iter_traces` / `list_sessions` /
  `iter_sessions` now send a list `tags` filter as repeated keys
  (`tags=vip&tags=beta`) instead of one comma-joined value, which the backend
  read as the single literal tag `"vip,beta"`. A single string is still sent
  as one value.
- **TVL specs with out-of-range chance thresholds or budgets now fail to load
  (#1616).** `load_tvl_spec` raises `TVLValidationError` when a
  `promotion_policy.chance_constraints[].threshold` is outside `[0, 1]`, or when
  `exploration.budgets` has a `max_trials` or `max_wallclock_s` below 1 or not a
  whole number, or a `max_spend_usd` below 0 or NaN. These are the canonical TVL
  bounds. A threshold above 1 used to make the constraint always pass. The same
  check applies to a `promotion_policy` returned by hybrid config-space
  discovery. `max_spend_usd: 0` still loads, as the TVL schema allows, but a run
  that takes its cost limit from the spec still fails the cost-limit check,
  which needs a value above 0.
- **Hybrid API examples no longer share one id (#2575).** `HybridAPIEvaluator`
  numbered the fallback example id from the position inside the batch, so at
  the default `batch_size=1` every example was sent and recorded as `ex_0`, and
  the outcome matrix merged them into one row. The fallback is now
  `ex_<dataset position>`; a string `example_id` (else `input_id`) in the
  inputs still wins. A dataset that fits in one batch sends the same ids as
  before, and the ids the backend stores for hybrid rows are unchanged. A
  hybrid dataset whose ids repeat now fails the trial with a `ValidationError`
  naming the positions, before any request is sent. `build_outcome_matrix`
  logs a warning when one trial repeats an example id instead of merging the
  cells silently. Statistical-significance tests, the best-config margin and
  the semantic-saturation stop condition, which key per-example scores by id,
  also saw only one example per in-batch position (a single example at the
  default `batch_size=1`) when the rows carry no ids; they now see every
  example, so their results for such hybrid runs can change.

### Added

- **`compare()` evaluates fixed configs as one experiment (#1952).** `OptimizedFunction.compare(configs, eval_dataset=None, repeats=1)` (and `compare_async`) calls the evaluator `optimize()` builds directly (no orchestrator, backend session, trial submission or `save_to` persistence) for each fully specified config on the same examples, and returns one `ComparisonResult` with per-config mean/spread, per-config cost (unmeasured cost is `None` plus a `cost_lower_bound`, never `0.0`), and a holdout check against the tuning dataset. Configs outside the declared space raise a value-free `ValueError`; cost estimate/approval, cumulative cost limit, strict cost tracking, CI approval, timeout, `max_total_examples` and a shared `ExecutionBudget` apply as in `optimize()` (a reached limit returns a partial result with `stop_reason`); the winner follows the objective's declared orientation; the wrapper's applied config and history are unchanged. See `docs/user-guide/compare_configs.md`.

- **`optimize(resume_from=...)` resumes a local run (#1697).** Pass a prior
  `OptimizationResult`, or a result saved with `PersistenceManager.save_result`
  (integrity manifest verified; tampering raises `ValueError`). Trials whose
  dataset, scorer, agent and configuration-space `fp1` fingerprints all match
  and that completed with a measured or unknown cost are reused, count toward
  `max_trials`, and are not charged to the new run's cost limit; in-flight
  trials are never reused or retried. Cloud and hybrid runs raise `ValueError`.
  See `docs/features/resume-from.md`.
- **`resume_from` reuses completed examples of partial trials (#2541, #1697).**
  A prior trial that finished only some examples is re-run for the same config
  with the completed examples (matching example id, input and expected output)
  reused and only the missing ones evaluated, aggregated as one trial. Reused
  examples are not charged again to the cost limit, stay in totals, keep unknown
  cost unknown, and use none of `max_total_examples`. Results without
  per-example data fall back to trial-level reuse. Call-time `eval_dataset`
  remains rejected (#1683).
- **`example_timeout` option, decoupled from the run timeout (#2509).** New
  `@optimize(example_timeout=...)` / `ExecutionOptions(example_timeout=...)`
  (positive number of seconds; default 60) sets the per-example limit for the
  local evaluator and custom-evaluator wrapper. Timed-out examples are counted
  per trial (`trial.metadata["timed_out_examples"]`, run total in
  `result.metadata["timed_out_examples"]`), shown in the progress line, and the
  limit is printed in the run banner. A timed-out example is a failed example
  (quality metrics 0.0, no fabricated cost). **Behaviour change:** the run-level
  `optimize(timeout=...)` no longer sets the per-example limit (it used to), and
  custom evaluators now actually enforce the per-example limit.

- **Paid runs check provider keys before cost approval (#1781).** When a cost
  limit is set or cost is approved (mock mode off), the optimizer validates the
  keys of only the providers the run's models use, once, and raises
  `ProviderValidationError` (provider name and error type only, no key
  material) before any trial. Adds an OpenRouter key check and an explicit
  "skipped" status for Bedrock (`ProviderStatus.skipped`). Honours
  `TRAIGENT_SKIP_PROVIDER_VALIDATION`; `validate_providers` stays callable
  standalone.

- **Cached input tokens and cache-hit rate per trial (#2069).** Trial metadata
  now carries `cached_input_tokens` and `cache_hit_rate` (cached / total input
  tokens), and `OptimizationResult.get_summary()` adds a `cache_note` when hit
  rates differ by more than 20 percentage points across trials, since cost
  comparisons may then reflect run order. Both values are `None` (unknown, never
  0) when a provider does not report cache usage. No behaviour change.

- **RAGAS reports row coverage and fails below a minimum scored share (#2677).**
  For each RAGAS metric the trial metrics now carry integer
  `ragas_<name>_rows_scored` and `ragas_<name>_rows_total`. Rows RAGAS leaves
  undefined are no longer dropped silently: if the scored share is below
  `ragas_min_scored_share` (evaluator config key, or
  `configure_ragas_defaults(min_scored_share=...)`; default 0.5, `0.0`
  disables) the trial fails with a counts-only error. Agent-failed rows still
  count as 0.0 and as scored. The metric value itself is unchanged: rows RAGAS
  leaves undefined are excluded from the mean, not counted as zero. Existing
  runs where fewer than half the rows were scored will now fail instead of
  reporting a partial mean.
- **New stop reason `safety_constraint_insufficient_evidence` (#2481).** A run
  halts with it when a safety constraint is neither proven nor violated for
  lack of enough evidence. `OptimizationResult.stop_reason` can now carry this
  value; code that switches on stop reasons should handle it.
- **Failed custom-evaluator rows are now visible (#2422).** When a
  `custom_evaluator` raises on some but not all rows of a trial, the trial
  metadata records `custom_evaluator_failed_rows` and
  `custom_evaluator_total_rows`, and `result.warning_codes` contains
  `CUSTOM_EVALUATOR_ROWS_FAILED` (per-trial counts in
  `result.metadata["session_summary"]["custom_evaluator_failed_rows_by_trial"]`).
  Counts only. Scoring is unchanged: a raising row still scores 0.0 on quality
  metrics.
- **`TraceRecord` and `SessionRecord` expose `tenant_id` and `project_id`
  (#2605).** Both are optional and filled from the backend's trace and session
  read responses, so a caller can confirm which project its traces landed in
  when the scope was resolved from the API key. They are `None` when the
  response does not carry them.
- **Export a run's failing examples as a regression dataset.**
  `OptimizationResult.failing_examples(threshold=0.5, mode="all_configs")`
  returns the ids of examples that failed in every trial (or, with
  `mode="best_config"`, in every trial of the winning configuration), read from
  the per-example outcome matrix without re-running anything; the selector is
  `traigent.utils.outcome_matrix.select_failing_example_ids`.
  `traigent results export-failures RUN --dataset PATH --out PATH` joins each
  id to the one dataset row whose input and expected_output equal what the run
  recorded for that id (never by guessing the evaluator's id scheme; both are
  compared in the redacted form the run stores, so export with the SDK version
  that ran the optimization), and
  writes the rows as JSONL that `Dataset.from_jsonl` reads back. It writes
  nothing if any id has no recorded input, matches no row, or matches more than
  one. Exported rows without a `metadata.example_id` gain their source id
  (`example_<index>`) so the subset's ids stay unique. `RUN` is a saved result
  name; logger run directories (which record no example inputs) and `local:`
  sessions are refused. New helpers: `Dataset.to_jsonl()` (atomic, refuses
  values JSON would alter) and
  `traigent.utils.regression_dataset.select_examples_by_id(dataset, ids, run)`.
  Nothing is sent over the network.
- **Trace search filters: `model`, `tags`, `outcome_kind`, `has_outcome` (#2564).**
  `BackendAnalyticsClient.search_observability_traces` and the
  `observability_search_traces` MCP tool accept them and validate them before
  any request: tags are exact, sent as repeated `tags` keys and must all match;
  `outcome_kind` is one of the backend's outcome kinds; `has_outcome=False` is
  sent explicitly. Free-text `search`, `user_id` and `session_id` remain
  deliberately unsupported to keep the reader content-free.
- **`ProjectUsageExperimentDTO`: optional `priced_configuration_runs` /
  `unpriced_configuration_runs`.** Both default to `None` and are read from the
  matching keys when present, so clients can tell an unmeasured cost (priced 0,
  unpriced > 0) from a real zero.
- **`traigent.observability.otel`: OpenTelemetry-based observability (extra `observability`).**
  `init` / `instrument` / `observe` / `attributes` / `flush` / `shutdown` /
  `stats`, a bounded best-effort span processor, an OTLP/HTTP exporter (official
  protobuf encoders, retry only on 429/502/503/504 and network errors), an
  egress-authoritative content policy built on a typed allowlist, parent-based
  head sampling with shared vectors, an exposure gate for third-party
  instrumentors, and automatic stamping of active optimization-trial ids on
  every span. The legacy client API is unchanged. See
  `docs/observability/otel.md`.
- **`publish_best_config()` refuses to overwrite an agent head that moved during the run.**
  When the backend discloses the agent a session is bound to, the SDK reads that
  agent's head generation once, when the optimization step starts, and sends it
  with the promoting publish. If another writer advanced the head in the
  meantime the backend answers 409 and the SDK raises the new
  `CloudBestConfigStaleHeadError` (a `CloudPublishUnavailable` subclass, reason
  `stale_head_generation`) carrying `expected_generation`, `current_generation`,
  `agent_id`, `environment` and the refused `best_config`. The published head is
  left untouched and the run's `OptimizationResult` stays available to publish
  again deliberately. Runs against a backend that reports no generation (older
  backend, offline run, no bound agent) publish exactly as before.

### Security

- **Saved optimization results are integrity-checked at load (#1961).**
  `PersistenceManager.save_result` now writes `integrity.sha256.json` with
  sha256 digests of `metadata.json` and `trials.json.gz`; `load_result`
  verifies them and raises a value-free `ValueError` on a mismatch, a missing
  listed file or an unreadable manifest. Results saved before this change have
  no manifest and load with one warning. `load_result` also raises
  `ValueError` when `best_config` lies outside the saved configuration space.
  This detects corruption and casual edits only; it is not tamper-proof,
  since anyone who can rewrite the files can rewrite the manifest too.
  The membership check judges the `best_config` actually returned (a
  disagreeing outer copy raises), reads a plain list as a set of choices (not
  an interval), and rejects a value of the wrong type against numeric bounds;
  two-number tuple ranges are now saved as `{"low", "high"}` in the curated
  `configuration_space`.
- **Old `traigent-tracing` add-ons are ignored (#1894).** An installed
  `traigent-tracing` older than 0.1.1 (or with an unreadable version) lacks the
  span-attribute scrubbers, so the SDK now logs one warning and uses the
  built-in scrubbed tracing instead. The add-on is bumped to 0.1.1.

### Removed

- **BREAKING: `outputs_sample` removed from `InteractiveOptimizer.report_results`
  and the trial-result payload (#2239).** Model outputs are customer content and
  are never sent to the backend. Passing `outputs_sample=` now raises
  `TypeError`, and `error_message`/`metadata` on `report_results` and
  `submit_trial_result` are now keyword-only so a legacy positional
  `outputs_sample` cannot be silently rebound; drop the argument. The SDK no longer puts `outputs_sample` on
  the wire.

- **Candidate runs (`optimize(apply=False)`) are temporarily withdrawn.** Their
  config isolation was not provably consistent under concurrent config
  changes. `apply=False` now raises `ConfigurationError` immediately in every
  entry point that accepts it -- `optimize()`, `optimize_sync()` (checked
  before creating any event loop, coroutine or thread, not deferred to the
  underlying `optimize()` call), and `optimize_with_guidance()` (checked
  before resolving `rewrite_llm`, loading the dataset, or touching any
  dataset override) -- with: "optimize(apply=False) (candidate runs) is
  temporarily unavailable; it will return in a future release. Use
  apply=True (the default)." The `apply` parameter remains in the signature
  and `apply=True` (the default) is unaffected.
- **BREAKING: the inert `mock_mode_config` and `mock` (`MockModeOptions`) parameters
  are removed** (Traigent#1370 Item 6). Both were accepted and ignored since the
  mock-mode retirement. Passing either to `@traigent.optimize`, runtime
  `.optimize()`, `OptimizedFunction(...)` or `validate_optimize_parameters()` now
  raises an error naming the removed parameter (even when the value is `None` or
  `TRAIGENT_DISABLED` is set), and `traigent validate-config` reports a top-level
  `mock_mode_config` / `mock` key as a failure. `MockModeOptions` is no longer
  exported. Mock LLM behaviour is unchanged: use
  `traigent.testing.enable_mock_mode_for_quickstart()` (or the `TRAIGENT_MOCK_LLM`
  env var) instead.
- **BREAKING: `OptimizedFunction(...)` rejects `use_cloud_service` and the singular
  `framework_target`.** Presence alone raises (even `False` / `None`), instead of
  being absorbed. Migration: drop `use_cloud_service` (it had no effect); use
  `framework_targets` in place of `framework_target`.
- **BREAKING: optimizer constructors and `get_optimizer(...)` reject `mock` and
  `mock_mode_config`.** `get_optimizer` wraps the constructor `TypeError` in
  `OptimizationError`; `validate_optimize_parameters()` raises `ValidationError`.
- **BREAKING: evaluators reject `mock` and `mock_mode_config`.** `LocalEvaluator(...)`
  (and every `BaseEvaluator` subclass) now raises `TypeError` for either name,
  even when the value is `None`, instead of storing it in `evaluator.config`.

## [0.30.0] - 2026-09-26

### Added

- **Raw OpenAI SDK calls are cost-captured (#2441, #2442).** With the OpenAI
  override active (`enable_openai_optimization()` or
  `framework_targets=["openai.OpenAI", "openai.AsyncOpenAI"]`), every
  non-streaming `chat.completions.create` / `completions.create` response that
  carries `usage` is charged to the running example, sync and async, including
  every call of a multi-call agent. Streaming calls and responses without
  `usage` are reported as unmeasured. See `docs/user-guide/cost_capture.md`.
- **`COST_OBJECTIVE_PARTIAL_USAGE_CAPTURED` and
  `COST_UNMEASURED_TRIAL_LIMIT_REACHED` warning codes (#2441, #2442).** The
  first marks a cost-objective run where only some trials captured usage. The
  second marks a run stopped by the unmeasured-cost trial limit and carries a
  message naming the ways to continue.

### Changed

- **An unmeasured cost is reported as unknown, not $0 (#2441, #2442).** In the
  custom-evaluator lane, a trial with no captured LLM usage no longer carries
  `input/output/total_tokens`, `input/output/total_cost` or a `cost` objective
  of `0.0`: the keys are left out (or `None` with
  `TRAIGENT_STRICT_METRICS_NULLS=true`). Weighted `best_config` selection
  scores the missing cost as the worst value, so an unmeasured configuration
  no longer wins a cost objective there. The Pareto front, the batch composite
  score, `metrics.get("cost", 0)` constraints and workflow spans do not treat
  it as unknown yet (#2446). The results table prints `n/a` for it. A
  `metric_limit` on a cost metric skips unmeasured trials instead of failing
  the run. Consumers that read these keys must expect them to be absent.
- **Unmeasured-cost runs stop at a default safety limit unless you size them
  (#2441, #2442, #2447).** When trial cost cannot be measured, the cost limit
  cannot bound spend. A run that did not set `max_trials` or
  `max_total_examples` explicitly stops after the default safety limit of 10
  trials (`TRAIGENT_FALLBACK_TRIAL_LIMIT` changes it) with
  `stop_reason="cost_limit"`. **A single unmeasured trial is enough**, even
  when every other trial was measured. This now also applies to
  custom-evaluator runs that previously reported `$0`, including async
  LangChain agents, whose `ainvoke`/`abatch` calls are not captured yet
  (#2445). The result carries `COST_UNMEASURED_TRIAL_LIMIT_REACHED` and a
  message that counts the unmeasured and measured trials and says to set
  `max_trials` explicitly to run more trials, or to capture usage. A run that
  sets `max_trials` or `max_total_examples` explicitly runs up to that size
  and carries `COST_UNMEASURED_TRIALS_RAN` instead. `max_total_examples` caps
  total examples and counts as explicit consent, but does not raise the
  default `max_trials` of 10. A `max_total_examples` passed to `.optimize()`
  stays on the function object for later calls.
  `TRAIGENT_FALLBACK_TRIAL_LIMIT` is now also honoured when `cost_limit` or
  `cost_approved` is set; those paths used to fall back to 10.
  `OptimizedFunction(max_trials=None)` and an MCP `run_optimization` call
  without `max_trials` do not count as explicitly sized.
- **The framework override no longer passes call-time parameters to a client
  constructor that does not accept them (#2441, #2442).** Building
  `openai.OpenAI()` inside an optimized function with
  `framework_targets=["openai.OpenAI"]` used to raise `TypeError: __init__()
  got an unexpected keyword argument 'model'`.

### Fixed

- **`import traigent` no longer imports MLflow (#2436).** With MLflow installed, every
  `import traigent` (and every `@traigent.optimize` run, via
  `traigent/integrations/__init__.py` and
  `traigent/integrations/observability/__init__.py`) eagerly imported the real
  `mlflow` package -- including its `uname` process spawn on some platforms --
  whether or not MLflow tracking was ever configured. `MLFLOW_AVAILABLE` is now
  a cheap `importlib.util.find_spec("mlflow")` check, and the real package is
  imported once, lazily, on first actual use of the MLflow integration. Also
  drops the unused `mlflow.sklearn` / `mlflow.pytorch` flavor-module imports
  entirely (this integration only ever called MLflow's tracking surface).

## [0.29.0] - 2026-09-25

### Added

- **Candidate runs: `optimize(apply=False)`.** Returns the result, with
  `result.best_config` as the candidate, without applying it to the wrapper. A
  candidate run executes on an isolated copy of the wrapper taken at rest: it does
  not change the wrapper's state, results history, current config or runtime
  overrides. Any number of candidate runs may run in parallel with each other and
  with an applying run. Promote a candidate later with `apply_best_config(result)`.
  The default (`apply=True`) is unchanged. The candidate's config and search space
  are snapshotted at candidate start by a type-preserving structural copy
  (classified by type identity: exact `dict`/`list`/`tuple` rebuilt, tuples kept
  as tuples, `None`/`str`/`int`/`float`/`bool` kept, NumPy scalars normalized to
  Python scalars). Any other value -- including container subclasses (a known
  restriction), functions, bound methods, classes, `Enum` members, locks and live
  clients -- makes `optimize(apply=False)` raise
  `traigent.utils.exceptions.CandidateIsolationError` naming the key path; no
  user code runs during the copy.
- **Overlapping applying runs on one wrapper are refused.** A second
  `optimize()` (`apply=True`) on an `OptimizedFunction` that already has one in
  flight now raises `traigent.utils.exceptions.OverlappingOptimizationError` (a
  subclass of `OptimizationStateError`) instead of racing it, where the last
  finisher silently replaced the first winner. `apply_best_config()` takes the
  same exclusive slot and holds it across its commit, so a promotion and an
  applying run can never interleave.

### Changed

- **`import traigent` no longer lets LiteLLM reach the network by default.**
  LiteLLM downloads its model-cost table (and, for Anthropic, its beta-header
  config) from `raw.githubusercontent.com` on its own first import unless
  `LITELLM_LOCAL_MODEL_COST_MAP` / `LITELLM_LOCAL_ANTHROPIC_BETA_HEADERS` are
  already set — a ~5s stall plus a warning on a blocked or restricted network.
  Traigent now sets both (`os.environ.setdefault`, so an explicit value of
  yours always wins) as soon as `traigent` is imported, before anything else
  in the package can import `litellm`. Set `TRAIGENT_LITELLM_LIVE_PRICES=1` to
  opt back into LiteLLM's live fetch. The pin only works if `traigent` is
  imported before `litellm` in your process; if `litellm` was already
  imported first, Traigent logs a debug notice instead of reloading it. See
  `traigent/skills/traigent-quickstart/references/environment-variables.md`.

- **Counting tokens no longer downloads a Hugging Face tokenizer.** Pricing a
  Llama-family model (also Cohere `command-r` and older non-`claude-3`
  Anthropic ids) through `litellm.token_counter()` used to lazily fetch that
  model's "exact" tokenizer from `huggingface.co` on first use — an egress
  audit measured 8 blocked-network attempts and ~24s added latency pricing
  one such model with the host unreachable. Traigent now sets LiteLLM's own
  `disable_hf_tokenizer_download` flag as soon as `traigent` is imported, so
  counting for these models falls back to LiteLLM's tiktoken-based count
  instead — approximate for exactly these models, not an outage. This
  affects only tokenizer selection for counting/encoding; it does not touch
  model calls, so it does not block a customer's own Hugging Face model
  downloads elsewhere in the SDK (Traigent still never sets `HF_HUB_OFFLINE`
  globally). Set `TRAIGENT_HF_TOKENIZER_DOWNLOAD=1` to opt back into
  LiteLLM's original, download-capable exact tokenizer selection. See
  `traigent/skills/traigent-quickstart/references/environment-variables.md`.

- **Declaring a reserved metric's direction opposite its default now warns.** An explicit
  `orientation`/`orientations=` still wins over an SDK-owned default (e.g. declaring
  `total_cost` as `maximize` instead of its `minimize` default), but it now emits
  `traigent.utils.exceptions.ObjectiveDirectionOverrideWarning` naming the reserved
  keyword, its default direction, and the direction you declared. The Traigent Backend
  certificate for that run will refuse to assert a comparison claim for that objective,
  because the declared direction contradicts the reserved keyword's preset; this warning
  surfaces that at declaration time instead of only when the certificate is issued. No
  warning for a bare name taking its canonical default, or for any custom (non-reserved) name.

### Fixed


- **Per-run cost accounting state.** The unpriced-at-runtime model registry and
  the usage-capture counter are now scoped to each `optimize()` run (a
  `ContextVar` inherited by the run's asyncio tasks and SDK worker threads)
  instead of being process-global and reset at every run start. Two agents
  optimizing concurrently in one process no longer wipe or pick up each other's
  records. Code outside a run keeps the previous process-level behaviour.
  `copy_context_to_thread()` / `snapshot.restore()` carry the run's cost state
  into user-created worker threads.
- **A `CancelledError` or `KeyboardInterrupt` that escapes an `optimize()` run no
  longer leaves the wrapper stuck in `OPTIMIZING`** (where `current_config`
  raises); the lifecycle moves to `ERROR`, as for any other failure.

- **A custom evaluator's failed or unpriced examples no longer lower a trial's cost.**
  A failed example was recorded with `cost: 0.0`, and an example that omitted `cost`
  was read as `0.0`, so a trial with half its examples failed reported about half its
  true cost and could win a cost objective (#2404). Cost keys (`cost`, `input_cost`,
  `output_cost`, `total_cost`) are now averaged only over examples that measured
  them, with a warning naming the coverage. A failed example does not fail its
  trial or the run, including under strict cost accounting. Quality metrics keep
  their existing behaviour (a failed example still scores 0.0), and a run where no
  example reports cost is unchanged.

## [0.28.0] - 2026-09-25

### Changed

- **Breaking: bare custom objective names now require an explicit orientation.**
  Traigent previously guessed from spelling and ultimately defaulted every unknown
  name to `maximize`. That made `total_cost` select the most expensive configuration
  and misread names such as `cost_savings` and `accuracy_loss`. Bare names now receive
  a default only for exact SDK-owned metrics: `accuracy`, `success`, `success_rate`,
  and `exact_match_default` maximize; `cost`, `total_cost`,
  `cost_per_example_mean`, `input_cost`, `output_cost`, `latency`, `duration`,
  `response_time_ms`, `avg_response_time`, `avg_response_time_ms`,
  `execution_time_ms`, `error_rate`, `empty_output_rate`, and
  `truncated_output_rate` minimize. Any other bare name raises `ValueError` before a
  trial or hosted session begins. Declare custom metrics with `ObjectiveSchema`, or
  call `create_default_objectives(["my_metric"], orientations={"my_metric":
  "maximize"})`. Explicit valid directions always win. Invalid directions now raise;
  target-banded objectives keep their non-directional band semantics.

- **Unrelated inline datasets no longer share one portal history.** Every inline
  example list was named `inline_dataset`, and that name was sent as the dataset
  identity, so all inline runs of one agent grouped together in history. The generated
  name is no longer sent as an identity. New `EvaluationOptions(dataset_id="...")` lets you
  declare a stable dataset id that survives content edits and `Dataset.name` renames
  (stripped and Unicode-NFC-normalized, 1-255 characters after normalization, otherwise a
  validation error). Two spellings of one name — composed and decomposed — are therefore one
  id, not two histories. An explicit id containing a control or zero-width character is
  rejected naming the codepoint, rather than silently rewritten; the same characters in a
  `Dataset.name` are cleaned out instead, so a dataset name can never fail a run. It is sent on the connected
  grid/random and managed session-create paths. A run with neither an explicit id nor a
  named `Dataset` logs one warning that its history will show "Dataset not linked".
  Offline runs record the identity locally and `traigent sync` sends it; sessions recorded
  before this change sync with no identity. A real `Dataset(name="support-v1")` keeps the
  same identity as before.

- **Breaking: `.optimize()` now rejects a keyword argument it does not recognize.** A
  call-time keyword that is neither a `@traigent.optimize` decorator option nor consumed by
  an optimizer was absorbed into the optimizer's `algorithm_config` and had no effect, so a
  typo silently ran a different optimization from the one you wrote. Such a keyword now
  raises `TypeError` naming it. Remove the argument at the call site, or correct the
  spelling if you meant a real option; there is no compatibility flag, because a keyword
  that reaches this error never had an effect.
  **Options of the optimizer you selected still work**, including every batch- and
  remote-optimizer option (`batch_config`, `pareto_frontier_size`, `base_optimizer`,
  `remote_enabled`, ...). The accepted set is derived from the registered optimizers'
  constructor signatures rather than hand-written, so registering a plugin optimizer — or
  adding a parameter to an existing one — cannot break its callers.
  The shipped `walkthrough/` examples passed
  `show_progress=` to `.optimize()`, which was never a parameter of it and never did
  anything; it is removed from all 21 example call sites and no example's behaviour changes.
  The parameter that does control the live progress bar is `progress_bar`: `True` forces one,
  `False` suppresses it, `None` (the default) auto-enables it in an interactive terminal.

- **Breaking: `PromptRewriter.rewrite()` no longer accepts `plan`, and `TextDocument` no
  longer exposes `trainable`.** Both were dead: `rewrite()` ignored `plan` entirely and
  nothing in the package ever read `trainable`. `PromptRewriter` is exported from
  `traigent.generation` (`__all__`), so this is a source-breaking change for any caller
  passing `plan=` or a third positional argument — those now raise `TypeError` rather
  than being silently ignored, and reading `.trainable` raises `AttributeError`. The
  TypeScript SDK's rewriter never had the parameter, so this restores cross-SDK parity.
  Remove the argument at the call site; there is no replacement, because it never had an
  effect.

- **Breaking: a model with no price is no longer scored as free when you optimize for cost.**
  Previously a call whose model had no price table entry and no provider-reported cost
  was recorded as `$0.00`, and because the optimizer minimizes cost it ranked that
  configuration cheapest: an unpriced model won `best_config` over a priced one of
  equal accuracy. When `TRAIGENT_STRICT_COST_ACCOUNTING` is unset, runs whose
  objectives include `cost` are now strict at runtime and stop at the first such call
  with the fix in the message. Provider-reported costs (OpenRouter) still price the
  call, and the pre-run coverage preflight is unchanged. Set
  `TRAIGENT_STRICT_COST_ACCOUNTING=false` to restore the previous behavior, which
  records `$0` and warns (`UNPRICED_MODEL_RUNTIME`). Runs without a cost objective are
  unchanged. Two more ways a cost run recorded `$0` are covered: a response that carries
  token counts but no model name to price them now fails the same way, and a cost-objective
  run that captured no LLM usage at all fails too, because an unmeasured `$0` is not a
  cheap configuration. Strict mode raises `CostTrackingRequiredError` at the first such
  trial, mock-LLM runs included (mock mode never bypasses cost enforcement); fix it with
  `traigent.with_usage(value, total_cost=0)` for a path that makes no model call, custom
  pricing (`TRAIGENT_CUSTOM_MODEL_PRICING_JSON`), or `TRAIGENT_STRICT_COST_ACCOUNTING=false`,
  under which the run only warns (`COST_OBJECTIVE_NO_USAGE_CAPTURED`);
  `result.metadata["pricing"]["usage_captured"]` reports whether any usage was measured.
  `cost_per_1k` now counts as a cost objective. The LangChain and Pydantic AI callback
  handlers read the setting when constructed, so set
  `TRAIGENT_STRICT_COST_ACCOUNTING=true` explicitly when optimizing through them.
  Upgrade note (#2421): 0.27.x only warned for both this case and an objective
  orientation defaulted to `maximize`; from 0.28.0 both are errors.

### Fixed

- **Dataset row ids you wrote yourself were sent to the Traigent service.** If your
  dataset rows carried an `example_id` (or `dataset_example_id` / `input_id` / a nested
  id in row metadata), that string was used verbatim as the per-example id on every
  trial submission — so a row keyed by its question text, ticket subject, or account
  number put that text on the wire. Per-example ids sent to the service are now always
  generated by the SDK: either the positional `example_<index>` id, or the opaque
  `ex_<hash>_<index>` id derived from the dataset name and the row's position. Your own
  id is unchanged in the results you get back and in the on-disk trial logs, and you can
  recompute the opaque id locally to join service-side insights back to your rows.

- **The path of your source file could leave your machine.** When the file defining the
  optimized function was outside both the working directory and the installed package —
  running a script by absolute path from another directory, or a notebook with a
  temporary working directory — the absolute path was sent, which includes your
  operating-system username and the directory layout above your project. The same value
  also shaped the reported module name. Only a project-relative path is sent now; when
  none can be derived, just the file name is sent.

- **Non-numeric metrics are now rejected instead of being sent.** Metrics are numeric by
  contract so that no text ever crosses to the Traigent service, but a metric holding a
  string (an LLM judge's rationale, a captured response, an error excerpt) was only
  warned about and then submitted — including through two secondary copies of the trial
  metrics that skipped the check entirely. A non-numeric metric in a trial's measures now
  causes that trial's submission to be rejected and logged at error level — the trial will
  not appear on the backend until the evaluator returns a number for that metric. You will
  not see a raised exception; look for "Rejecting submission (measures contract
  violation)" in your logs. Any non-numeric value in the secondary trial-metric copies is
  dropped with a warning instead, key by key, rather than rejecting the whole trial. Keep
  rationale and explanation text local, in your own logging or the on-disk trial logs.

- **`offline=True` and OpenTelemetry.** Documented, in `docs/api-reference/telemetry.md`,
  that `offline=True` does not turn off OpenTelemetry export: with
  `TRAIGENT_TRACE_ENABLED=true` and `OTEL_EXPORTER_OTLP_ENDPOINT` set, evaluation spans
  still go to the collector you configured.

- **`SafetyValidator.validate()` ignored the constraint operator, so `below()` safety
  gates were trivially satisfiable** (#2204). Aggregate validation compared the
  Clopper-Pearson lower bound on the per-trial compliance rate directly against the
  constraint's raw metric threshold, regardless of `above()`/`below()`. That comparison
  happened to read sensibly for `above()`, but for `below()` a *strict* (small) metric
  threshold produced a *lenient* compliance bar — e.g. `hallucination_rate().below(0.1)`
  only required ~10-18% of trials to comply. **Behavior change existing users will
  notice: any `below()` safety gate that was reporting "satisfied" may now correctly
  report "not satisfied"** — this is the fix, not a regression, but it changes the
  pass/fail verdict of runs that rely on it. `above()`/`>=`/`>` gates are unaffected: the
  required compliance rate for those operators is unchanged. See the PR that closes #2204
  for the exact compliance-rate rule now used for `below()`/`<=`/`<`, and an open question
  for a future owner decision about whether that rule should become an explicit API
  parameter.

### Added

- **The run's finalize request now carries a selection receipt.** In non-strict modes,
  when the SDK selects a winner, the `session_aggregation` sent on session finalize
  includes `selection`: the winning trial id, the exact ranking-eligible trial ids it
  was chosen over (unique, sorted), their count and a `sha256:` digest, the SDK's selection reason code, and the
  winner-vs-runner-up margin as already computed for `result.best_config_margin`. The
  Backend binds it to the session's own trials; the SDK never recomputes ranking. The
  receipt carries only ids, counts, a digest, bounded numbers and labels — no config
  values, prompts, or example content — and is rebuilt from an allowlist at egress. No
  winner means no `selection` key, and strict (certified) mode sends no receipt at all.
  Duplicate ids in the eligible set are never merged: the receipt is withheld with a
  count-only warning. A margin with a non-finite number, an inverted `ci95`, a winner
  different from the receipt's, or more configs than eligible trials is omitted rather
  than sent. Requires the TraigentSchema build pinned in `scripts/ci/schema-pin.txt`
  (now `df06e6dc`).
- **Each run records where its prices came from.** `result.metadata["pricing"]` holds
  the LiteLLM price-table source (`local` or `remote`, from LiteLLM 1.93), any fetch
  fallback reason, and whether strict cost accounting was on and why. It is attached
  after the run and is not sent to the Traigent service.
- **Docs: restricted networks and reproducible cost.** The environment-variable
  reference now explains LiteLLM's import-time GitHub fetch, the Hugging Face
  tokenizer download for Llama-family models, and the variables that turn both off.
- **`EvaluationOptions(task_type=...)`** — a coarse task category for the run
  (`multiple_choice`, `exact_match`, `text2sql`, `code_generation`, `summarization`, …)
  sent as `task_type` on `POST /api/v1/sessions` from both session-create paths. The
  Traigent service maps it to an evaluator-quality anchor policy — which verifiable
  ground truth, if any, an audit of your evaluator may use. You never name an anchor;
  unknown values resolve to "no anchor". Without it the evaluator-quality audit abstains
  as `no_anchor_designation` on every run, which is what the first real audit did on
  2026-09-03.

## [0.27.0] - 2026-08-22

### Changed

- **Cryptography 49.0.0 → cryptography 50.0.1.** The published floor is now
  `cryptography>=50.0.0` and the shipped lock selects 50.0.1, closing
  CVE-2026-69247 / GHSA-g6cj-pr64-35w5. This is possible because the integrations
  extra now uses `mlflow-skinny` instead of full MLflow 3.15.1 and its
  incompatible `cryptography<50` constraint. Pandas is a direct integrations
  dependency so `mlflow.search_runs(..., output_format="pandas")` remains
  available.
- `traigent[integrations]` now bundles MLflow's tracking/client runtime, not the
  full MLflow server and model stack. Install the supported client surface with
  `python -m pip install "traigent[integrations]"`.
- The dependency graph no longer adds full MLflow, `mlflow-tracing`, Alembic,
  Flask and Flask-CORS, Gunicorn and Waitress, Docker, PyArrow, Graphene and its
  GraphQL packages, Huey, Mako, PrettyTable, Skops, Blinker, ItsDangerous,
  wcwidth, or Werkzeug. SQLAlchemy is deliberately not on this list: it remains
  present through `langchain-community`, although Traigent does not promise it
  as an MLflow SQL-backend extra.
- Model-flavor modules remain importable, but their execution dependencies are
  not supplied by the skinny runtime. Add only the flavor dependency you use,
  for example `python -m pip install "traigent[integrations]" scikit-learn`.
  PyTorch users can add their platform-appropriate `torch` build instead.
- Full MLflow 3.15.1 cannot share the current Traigent environment because its
  `cryptography<50` requirement conflicts with Traigent's security floor. The
  skinny-only environment does not support local SQLite or other SQL tracking
  URIs. Those require a separate environment or service running a compatible
  full MLflow stack; point Traigent at that service's remote tracking URI.
  Otherwise wait for a full MLflow release that supports `cryptography>=50`;
  do not install full MLflow alongside this Traigent release.

### Removed

- **The public `traigent[dspy]` extra and its locked dependency chain are
  removed.** The import-optional DSPy adapter remains available to applications
  that provide their own compatible DSPy installation, but Traigent no longer
  brings `dspy-ai`, `dspy`, or `diskcache` into its shipped lock. This does not
  attest a user-managed DSPy installation as safe: DSPy's current dependency
  chain includes `diskcache`, and CVE-2025-69872 has no patched release at the
  time of this release. Users who deliberately retain DSPy must manage and review
  that dependency risk separately instead of installing `traigent[dspy]`.
- **Named strategy presets removed** (#2100, #2101). The advisory selection rules
  behind `max_accuracy_then_cheapest_within_epsilon`, `quality_floor_min_cost`, and
  `pareto_frontier` are the technique, not something a public package ships readable
  source for; that logic belongs in the backend. The presets are gone outright — there
  is no shim and no compatibility alias — which removes the surface
  both #2100 and #2101 were reported against and stops it being published again. The
  removal takes effect from this release onward and does not alter anything already
  published. `traigent.api.strategy_presets` is deleted along with all eight symbols
  it exported from `traigent` and `traigent.api`
  (`ADVISORY_SELECTION_NOTICE`, `NormalizedStrategyPreset`, `StrategyPresetError`,
  `StrategyPresetValidationError`, `UnknownStrategyPresetError`, `VALID_PRESET_NAMES`,
  `normalize_strategy_preset`, `select_strategy_preset`). Use `algorithm=<optimizer>`
  to choose a search algorithm and `objectives=[...]` to declare what is being
  optimized.

  Scope note for anyone tracking **#2101**: removing the presets removes the surface that
  reported the defect, not its underlying cause. #2101's root cause — the built-in
  exact-match rate occupying `metrics["accuracy"]` when `accuracy` is not the ranked
  metric, so a graded `metric_functions` scorer registered under another name still leaves
  `metrics["accuracy"] == 0.0` — is **not** fixed here, and is being addressed separately.
  What is fixed is that no preset silently floors or ranks on that value any more.
- **Decorator `strategy=` / `strategy_params=` now raise `TypeError`.** Both parameters
  stay in the signature so the error can say what happened instead of reading as a typo.
  Passing one of the three retired preset names reports the removal explicitly.
- **Runtime `.optimize(strategy=<preset name>)` and `.optimize(strategy_params=...)` now
  raise `TypeError`.** `strategy_params` raises on every non-`None` value, whether or not
  `strategy` is also passed. `strategy=<optimizer name>` is unaffected: it remains the
  pre-existing, unrelated deprecated alias for `algorithm=`. The three retired names are
  matched exactly, so a name you have registered yourself via `register_optimizer` still
  resolves to your optimizer — including a casing such as `"PARETO_FRONTIER"`, which the
  registry treats as a distinct name from `"pareto_frontier"`.
- **`OptimizedFunction(...)` raises `TypeError` for `strategy=<preset name>`,
  `strategy_params=` and the former internal `strategy_preset=`.** The constructor takes
  `**kwargs`, so these were accepted as unknown keywords, stored, and never read — the
  call returned an ordinary result for a request that asked for a preset. Other `**kwargs`
  keys, and a non-preset `strategy=` value, are unchanged.
- **Migration — the exception *type* changed, so an `except ValueError:` around one of
  these calls no longer catches it.** The deleted preset errors were declared as
  `StrategyPresetError(ValueError)`, with `UnknownStrategyPresetError` and
  `StrategyPresetValidationError` deriving from it. Everything above now raises
  **`TypeError`**, which is not a `ValueError`:

  ```python
  try:
      result = fn.optimize_sync(strategy_params={"epsilon": 0.02})
  except ValueError:
      result = fall_back()
  ```

  now lets the `TypeError` propagate. Widening the `except` is not the fix: no preset name
  and no `strategy_params` value works any more, so the call itself has to go. Replace it
  with `algorithm=<optimizer>` and `objectives=[...]`.

  The TVL-spec route raises the same `TypeError`, which means it is the one spec failure
  that does **not** arrive as `TVLValidationError`/`ValidationError`: an `except
  ValidationError:` wrapped around a spec-driven `optimize()` catches every other
  malformed-spec case but lets this one through. Catch `TypeError` as well if a spec whose
  `strategy` names a retired preset has to be handled rather than fixed.
- **A TVL spec naming a retired preset in `optimization.strategy` / `exploration.strategy`
  now fails when the spec is loaded**, with the same by-name message, instead of loading
  clean and dying mid-run as `OptimizationError: Unknown optimizer '<preset>'` after a
  session had already been created.
- **`traigent validate-config` fails a config whose `"algorithm"` is a retired preset
  name**, with the same by-name message, instead of reporting "Configuration validation
  passed" for a config that cannot run. Only the three retired names are checked; the
  route still does not validate algorithm names in general.
- **`traigent results rerank` loses `--preset`, `--epsilon`, and `--floor`.** `--weights`
  is the only remaining mode and is now required, so a bare invocation fails with Click's
  missing-option error rather than the SDK's former "use exactly one of" message.
- **Supersedes the 0.14.0 entry "`strategy=` non-preset alias removed"**, whose closing
  guidance — "Use one of the documented preset names" — is no longer valid advice; there
  are no documented preset names. That historical entry is left as written.
- Not removed: `OptimizationResult.preset_selection` and `traigent.api.types.PresetSelection`,
  including their persistence round-trip and the CLI's display-only rendering. They are
  retained solely so a result persisted by an older SDK build still loads and still shows
  its selection record. No code path in this build populates them. The `strategy_preset`
  key is no longer written to session or trial metadata, and is no longer reconstructed
  when reading a pre-#2031 legacy artifact (such artifacts still load).

### Changed

- **The eval audit reports a review tier instead of its scoring model** (#1881). Same
  reasoning as the strategy-preset removal above: the scorer is the technique, not
  something a public package ships readable. `OptimizationResult.eval_audit.scored`
  previously returned, for every scored item, a continuous `defect_score`, its exact
  `defect_percentile`, the raw `features` map, and `contributing_signals` — each carrying
  the model coefficient (`weight`) and its additive `contribution`. Together with the
  documented functional form, that published the estimator itself and a ranked dataset
  to fit against.

  `ItemDefectScore` now carries `example_id`, `review_tier` (`"high"` / `"elevated"` /
  `"normal"`, relative to this run) and `primary_reason` (`"never_correct"` /
  `"mean_wrong"` / `"instability"`). `DefectSignal` is removed from the public surface.
  The ranked worklist is unchanged — items still arrive most-in-need-of-review first, and
  **the list order is the ranking**, so "work the top of the list" behaves exactly as
  before. `flagged` and `summary` are untouched.

  The scorer itself is unchanged in behaviour and still refittable via
  `defect_score_intercept` / `defect_score_weights` on `compute_eval_audit` — pass a dict
  fitted on your own labeled subset to calibrate the ranking to your data. It is now
  **private**, along with everything that carries the model: `compute_defect_scores`,
  `DEFECT_SCORE_WEIGHTS`, `DEFECT_SCORE_INTERCEPT` and `DEFECT_SCORE_FEATURES` are renamed
  with a leading underscore, and the two refit parameters default to `None` rather than to
  the shipped coefficients, so the signature does not publish them either. A projection is
  only a boundary if there is no supported way around it.

  Code reading `item.defect_score` or `item.defect_percentile` should switch to
  `item.review_tier`, and code reading `item.contributing_signals[0].feature` to
  `item.primary_reason`. Code that only iterated `scored` in order needs no change. Code
  importing `compute_defect_scores` or the `DEFECT_SCORE_*` constants directly was reading
  the estimator and has no replacement; use `compute_eval_audit` and the refit parameters.

## [0.26.0] - 2026-08-02

Security and correctness release. Hardens every egress path the SDK owns
(cloud URLs, telemetry, tracing headers, dataset paths) to fail closed, and
corrects the multi-objective and early-stopping machinery that decides which
configuration wins.

### Added

- **Cumulative execution budgets.** `ExecutionBudget` (root-exported,
  experimental) is shared across the evaluation and optimization phases of a
  run, so a budget means the same thing whichever phase spends it (#1980).
- **No-execution evaluation-compatibility contract.** `validate_evaluation_contract()`
  statically checks an evaluator/dataset pair before any model call, reporting
  typed `ContractFinding`s. New root exports: `validate_evaluation_contract`,
  `EvaluationContractReport`, `ContractFinding`, `ContractCode`,
  `EVALUATION_CONTRACT_VERSION`.
- **Economics telemetry.** Contract-first telemetry emission with a sealed
  prepared-batch identity and provenance. `EconomicsTelemetryClient` is now a
  root export, matching the schema-owned Python/JS parity manifest; it stays
  lazy, so `import traigent` does not pull in `traigent.economics`.
- **Observability receipts.** Telemetry receipts plus a run-level zero-span
  invariant, so a run that emitted nothing says so instead of looking healthy.
  `SpanResult` is now root-exported.
- **W3C trace context propagation.** When an OpenTelemetry span is active, the
  SDK injects `traceparent` on backend, hybrid, analytics, and evaluation HTTP
  calls, so backend spans join the caller's trace (#1882, #1893).
- **Statistical ties on `best_config`.** Results now mark when the winner is not
  significantly better than the runner-up, using a paired significance test with
  confidence intervals and a multiplicity correction for post-hoc runner-up
  selection (#1866).
- **Syncable local session id.** `OptimizationResult` exposes the local session
  id so an offline or partially-synced run can be reconciled later (#2020).
- **Backend attribution block.** Guidance and decision surfaces accept and
  surface the optional `attribution` block returned by the backend, with the
  contract bounds enforced client-side (#2018).

### Fixed

- **Multi-objective selection was wrong in several ways.** Dominance and
  hypervolume are now driven by the *configured* objective set rather than
  whatever a trial happened to report; hypervolume handles the 1-D and
  mixed-completeness cases; frontier trimming preserves completeness; and the
  public no-arg hypervolume default is safe. NaN could previously leak through
  scalarization (#1940, #1941, #1942, #1944, #1945).
- **Early stopping stopped at the wrong time.** The plateau window is now
  compared against a pre-window baseline instead of itself, thresholds are
  derived from the configured orientation instead of being hardcoded, a declared
  orientation is honored in every selection path, and an invalid
  `early_stopping_min_delta` is rejected at the config boundary rather than
  silently coerced. The `OptimizationStrategy` positional ABI is preserved.
- **Retired model IDs.** Retired IDs are swept from `_KNOWN_MODELS` and the
  model tables, with a runtime denylist and an assertion that every finite table
  is complete (#1936, #1937).
- **Cost constraints priced inconsistently.** `model_cost_constraint` now prices
  canonical-first and falls back to litellm, matching the estimation path (#1958).
- **Idempotency on retry.** A retried `/optimize` POST sends a stable
  `Idempotency-Key`, so a network retry cannot create a second run.
- **Local-fallback reporting.** When a run falls back to local search the SDK
  says so in user terms and picks the remedy from the typed reason rather than
  parsing backend prose (#2024). Non-fatal config-space warnings from
  `@optimize` are surfaced, and a degenerate config space warns once instead of
  once per pinned knob.
- **Strict session-id contract in the CLI.** Whitespace-only and padded session
  ids are rejected, and `sync --clean` requires the resolved id (#2030).
- **Dataset path resolution.** A root-relative dataset path resolves against the
  cwd when the root join misses, nested relative datasets pass an absolute path
  to `safe_open`, and the evaluator's cwd retry no longer raises out of the
  registry.
- Submission attempts, not `result.trials`, drive the sync-id predicate (#2020).
- `math.isnan` replaces a self-comparison NaN check in `ExecutionBudget`.
- Runtime and CI are pinned to TraigentSchema 5.0.0.
- Mock multi-objective walkthroughs now demonstrate the cost/latency trade-off,
  and real-mode example summaries print latency in ms rather than seconds.

### Security

- **Cloud URL SSRF guard hardened against IMDS and link-local egress.** The
  guard now blocks the cloud-metadata endpoint and link-local ranges, including
  IPv6-transition encodings of them (`::ffff:169.254.169.254` and friends). This
  is the highest-severity fix in the release.
- **Telemetry egress requires SDK auth.** The economics telemetry client sends
  nothing without a credential — no anonymous egress path remains. Telemetry is
  revalidated at the transport boundary, exact contracts are enforced, and
  prepared-batch issuance is sealed with bound identity and provenance.
- **No `tracestate` forwarding.** Only `traceparent` is propagated to the
  backend; `tracestate` can carry third-party vendor data and is dropped.
- **Repr/detail leaks closed** in the economics surfaces, with root-export
  parity so an error's string form cannot expose payload contents.
- **Dataset validation fails closed.** Relative dataset-path escapes are
  rejected, resolution `OSError` fails closed against a concrete containment
  base, and every dataset row is validated — not just the first five.

### Dependencies

- `pyasn1` 0.6.4 — CVE-2026-59884, CVE-2026-59885, CVE-2026-59886.
- `gitpython` 3.1.54 — two GHSA advisories.
- `pillow` 12.3.0.

### Docs

- `add_agent_span` documentation aligned with the `SpanResult` return value.

## [0.25.0] - 2026-07-18

Client-sequenced grid/sync release: online mode is the default and always
supported for grid, random, and auto — local decision-making never blocks
transmission of results to the backend.

### Added

- **Grid & random runs finish locally, then sync.** Connected `grid`/`random`
  runs now enumerate every planned configuration even when the backend closes
  the tracking session early — the run no longer truncates at ~4 configs. Such
  a run completes locally, persists every trial, and is marked
  `backend_tracking="partial"` / `persistence_status="degraded"` so you can tell
  it was client-sequenced. `auto`/cloud-brain behavior is unchanged (#1938,
  #1947).
- **Offline runs are now syncable.** An offline optimization mints a local
  session, mirrors every trial to `sessions/<id>.json`, and finalizes to a
  sync-eligible status — so `traigent local list` shows it and
  `traigent sync --all` uploads it once you reconnect. Local storage is no
  longer coupled to backend connectivity (#1939, #1947).

### Fixed

- Hybrid MCP transport now classifies transport errors correctly and falls back
  cleanly when a model has been retired; retired model IDs removed from the
  provider tier tables (#1930, #1931, #1932, #1933, #1934).
- `update_best` no longer errors when a run declares no objectives (#1909).
- The optimizer meta-skill is no longer re-injected unbounded each epoch — its
  injection is gated and capped (#1929).

### Security

- Skill-training input is hardened against invisible-character prompt injection:
  default-ignorable Unicode (including non-`Cf` `Default_Ignorable_Code_Point`)
  is stripped before the meta-skill injection scan (#1926, #1929).
- MCP floor raised to `>=1.28.1,<2` to pick up CVE-2026-59950 (websocket
  Host/Origin validation) and CVE-2026-52870 (task-handler session scoping)
  (#1946).

### Dependencies

- `anyio>=4.14.2` (new floor; IDNA2008 TLS hostname certificate matching,
  AIKIDO-2026-889297) and `mcp>=1.28.1,<2.0.0` in the `hybrid` and `mcp` extras
  (#1943, #1946).

### Docs

- Examples and public docs use current-catalog model IDs, validated against
  `models.yaml`; assorted public-doc accuracy fixes (#1903, #1909, #1917, #1921).

## [0.24.0] - 2026-07-16

### Added

- **Eval-defect audit** (`result.eval_audit`, opt-in, zero-cost by default): a
  per-example × per-config outcome matrix persisted with each run (#1838/#1889),
  deterministic dataset-defect detectors — never-correct, token-leak,
  cross-family consensus-on-wrong (#1880/#1897) — and a continuous 0-1 defect
  score with percentile ranking per example via `result.eval_audit.scored`
  (#1881/#1901). Pure functions of already-persisted run data: no LLM calls,
  no network.
- `setup_logging(logger_name=...)`: confine SDK logging to a scoped logger
  instead of the root logger (opt-in; default behavior unchanged) (#1883/#1899).
- `TRAIGENT_OPTIMIZATION_LOG_MAX_RUNS`: opt-in retention pruning of old local
  run directories — never touches the active run, errs on keeping data
  (#1884/#1899).

### Security

- Span error strings are scrubbed before OTLP export in both the core and
  plugin tracing paths: exception messages/payloads are redacted of emails,
  keys, and secrets before entering any trace backend (#1885/#1898).
- The outcome-matrix loader refuses artifacts resolving outside the artifacts
  directory (symlink containment).

- `@observe` exception metadata now honors the content gate. `error_message`
  carries free-form content — exception strings routinely interpolate prompts,
  records, and PII that pattern redaction cannot catch — so it now follows the
  same rule as `input_data`/`output_data`: withheld entirely under the
  metadata-only default, replaced with `[REDACTED]` under
  `content_mode="redacted"`, and sent verbatim only under
  `content_mode="record"`. `error_type` (a class name, never content) is
  retained in every mode. Closes the error-path egress gap left open when
  0.21.0 made `@observe` metadata-only.

### Added

- `sync_session_id` on the exceptions a failed run raises — `OptimizationError`
  (and therefore the subclasses raised from inside a run, e.g.
  `CloudBrainUnavailableError`) and the typed `ResolutionError` (#2029). This
  closes the gap #2020 recorded and left open: a run that raises mid-flight
  never returns an `OptimizationResult`, so #2020's field could not reach the
  caller — yet the trials that already completed are durably on disk, because
  trials are persisted per-trial rather than batched at finalize. Since
  `traigent sync --all` **skips failed sessions**, those trials were strandable
  with no supported way to name them.

  ```python
  try:
      result = agent.optimize_sync()
  except (OptimizationError, ResolutionError) as exc:
      if exc.sync_session_id:
          subprocess.run(["traigent", "sync", exc.sync_session_id], check=True)
      raise
  ```

  The id comes from the same locality + durability predicate the success path
  uses, so it means exactly what it means on `OptimizationResult` — including
  the store-relative caveat — and is `None` for the same reasons, plus one
  more: a failure before any session exists has nothing to name. The attribute
  defaults to `None` on every one of these types, so it is safe to read
  unconditionally without `getattr`.

  `ResolutionError` is **not** wrapped to carry the id. Its exact type is the
  RFC 0001 §3.4 public contract that lets callers tell a governance rejection
  (stale certificate, evidence leak) from a generic failure, so the id is
  attached in place and the type identity is preserved.

  Scope, precisely. The class-level `None` default — and therefore the promise
  that a bare `exc.sync_session_id` is safe — covers `OptimizationError`,
  its subclasses and `ResolutionError`. Non-`Exception` exits (`SystemExit`,
  `GeneratorExit`, a test framework's outcome exceptions) also get the id
  attached before they propagate, and are never swallowed, but have no class
  default: read those with `getattr(exc, "sync_session_id", None)`.
  `KeyboardInterrupt` / `asyncio.CancelledError` are not affected — they still
  return a partial `OptimizationResult` carrying #2020's field rather than
  raising. `CostLimitExceeded` inherits the attribute but always reads `None`:
  the only thing that raises it is the pre-run cost-approval gate, which fires
  before a session exists, and a mid-run cost limit stops gracefully instead of
  raising.

- `OptimizationResult.sync_session_id` — the id of the run you just finished in
  **that run's local session store**, and the argument
  `traigent sync <SESSION_ID>` accepts against that store (#2020).
  Populated for offline / no-egress runs, no-API-key local-fallback runs,
  explicit-local runs with backend tracking disabled, runs that degraded to
  local-only mid-flight, runs whose trials the backend did not all acknowledge,
  and runs the backend closed early (#1938); `None` when the backend tracked the
  run end-to-end (nothing to upload) or when no local record can be read for it.
  This is the supported surface — read it, not `metadata`. The id is
  store-relative: `traigent sync` is a separate process that resolves its store
  from the environment, so a run that passed the programmatic
  `local_storage_path` option must point the CLI at the same root
  (`TRAIGENT_RESULTS_FOLDER="<that path>" traigent sync <id>`) — the SDK logs a
  warning naming that root when the two differ. For a backend-early-complete
  (#1938) or partially-acknowledged run (which includes a run that degraded to
  local-only mid-flight, since degradation means at least one trial went
  unacknowledged), the trials already submitted are on the
  portal, so syncing re-imports them as a separate experiment — a deliberate
  trade-off, since the SDK's own message for those shapes tells the user to
  sync.

  `result.metadata["local_session_id"]` is also touched, but it is **not** a
  clean mirror and never was: on connected runs `attach_session_metadata` sets
  it to the **backend** session id (so it was never a "this run was offline"
  marker — use `result.source` for provenance), while #1939 set it to the local
  id on offline runs. This release additionally populates the local id for
  no-key, tracking-disabled, degraded, unacknowledged-trial, and #1938 runs —
  and, deliberately, **stops** writing it when no local record can be read for
  the id, so a value `traigent sync` would reject is never surfaced. That last
  part is a narrowing relative to #1939, not a superset.

  Known gaps — two shapes deliberately left uncovered:

  - Backend persistence failed / finalize was not acknowledged. An exception
    out of finalize cannot be distinguished from a committed write whose
    response was lost, so that failure **does not by itself** produce an id —
    the persistence-failure handler deliberately does not add one, because
    handing one out could duplicate a run the backend already holds. It does
    not take one away either: the id is assigned before the finalize block, so
    a run that already qualified under another clause (e.g. partial
    acknowledgement) keeps the id it was given even if finalize then fails.
    The log no longer advises whole-store `traigent local sync` (#2028); it
    points at the portal session id. A targeted repair path is follow-up work.
  - ~~A run that raises mid-flight re-raises instead of returning a result (the
    orchestrator's FAILED path), so no `sync_session_id` is delivered even
    though the local record holds the trials that completed.~~ **Closed by
    #2029** (below): the id now rides on the raised exception.

### Changed

- **BREAKING (default behaviour):** `DatasetConverter.sdk_dataset_to_backend_examples`
  and `upload_sdk_dataset_to_backend` now **fail closed** on a dataset row that
  cannot be converted, raising `DatasetValidationError` instead of dropping the
  row (#1722, `g6:dataset-row-drop`). Previously each failure was logged once and
  counted into a local `error_count` that was discarded before any caller could
  see it, while `ExampleSetMetadata.total_examples` reported the POST-drop count
  — so an upload that silently lost rows reported `success=True, errors=[]`. A
  shorter dataset changes what the optimization is measured against, and a
  dropped row was indistinguishable from a row that was never there.

  Escape hatch: pass `strict=False` (on either method) to accept the drop, and
  `conversion_errors=[]` to the converter for the structured per-row records.
  `upload_sdk_dataset_to_backend` now reports drops in `ConversionResult.errors`
  and no longer returns `success=True` when the backend rejected examples as
  `invalid`. `metadata.total_examples` is now the INPUT row count in both modes.

  A dataset in which every row converts is unaffected.
- **Breaking (narrow):** `TraigentClient(algorithm=..., no_egress=True)` now
  raises `ConfigurationError` at **construction** for the managed algorithms —
  `bayesian`, `optuna`, `tpe`, `cmaes`, `nsga2`. Previously the client
  constructed and the run failed later, at `optimize()`. The combination can
  never work: those algorithms require managed optimization, which requires
  egress. Failing at construction reports it where it can be acted on, with the
  algorithm named. Local algorithms (`grid`, `random`) are unaffected and still
  construct with `no_egress=True`. Fallout of wiring `no_egress` into execution
  intent (#1776); pinned by
  `TestManagedAlgorithmsWithNoEgressIsABreakingChange`.
- No behaviour change for `no_egress=True` combined with an explicit deprecated
  `execution_mode="hybrid"`/`"cloud"`: the run still fails closed with
  `CloudEgressBlockedError` and sends nothing. The `execution_mode` attribute
  does read `hybrid` while the resolved policy reads `local_only, offline=True`,
  which is confusing, but the transport guard — not that attribute — is what
  stops the request, so the contradiction is cosmetic and is reported loudly
  rather than silently resolved in either direction.
- `traigent auth whoami` no longer guesses at a 403 it cannot attribute. It now
  reports `forbidden_indeterminate` and says both causes are possible, instead
  of defaulting to "the key lacks a required scope — grant the scope rather than
  rotating the key", which was stated with full confidence for AWS WAF, Akamai
  and API Gateway blocks that never reached Traigent (#1775).

### Fixed

- `ProductionMCPClient` no longer launders programming errors into benign
  degradation (#1774, #1777). Two related, user-visible behavior changes ship
  together:
  - `MCP_AVAILABLE` now reflects whether the `mcp` package is actually
    importable, rather than a nonexistent `mcp.client.stdio.StdioClientTransport`
    symbol the import block reached for — so it flips `False` -> `True`
    wherever `mcp` is installed. The stdio-transport path this newly exposes
    is not yet ported: `connect()` now raises `NotImplementedError` naming the
    real API (`stdio_client`) instead of quietly returning `False`.
  - `call_tool()` now re-raises `AttributeError` / `TypeError` / `NameError` /
    `ImportError` / `NotImplementedError` instead of routing them into the
    local fallback path and returning a synthetic `success=True,
    is_fallback=True` response with a fake `fallback_*` id. Genuine
    transport/backend failures still fall back exactly as before; only
    programming-error signatures are excluded from fallback.
  Any caller relying on the previous silent-`False` / silent-fallback
  contract for these specific failure modes should catch the new exceptions
  explicitly.

- `traigent sync` now explains what it wants when an id is not found: the old
  `Session <id> not found` gave no path forward for the common case of passing
  `optimization_id`, which `sync` does not accept (#2020).

- `ObservabilityClient` fails fast when no credential resolves. With
  `offline_mode` off and neither an API key, a JWT, nor a non-blank
  `Authorization`/`X-API-Key` in `extra_headers` (gateway/proxy auth counts;
  whitespace-only values do not), the client logs one actionable warning
  naming `TRAIGENT_API_KEY` (or `traigent auth login`) and disables its own
  network lanes, instead of emitting unauthenticated ingest requests that
  401-retry-storm behind an opaque "ingest rejected with status 401".
  Suppression is lane-aware, not a global offline flip: a custom `sender`
  keeps owning ingest delivery, a custom `request_sender` keeps owning
  control-plane calls, and blocked control-plane requests raise a clear
  "no credential resolved" error. `config.offline_mode` is left untouched.
  Blank explicit credentials are also treated as absent by
  `build_headers()`, so they can never overwrite working auth supplied via
  `extra_headers`. Telemetry never raises into the host app.

## [0.23.0] - 2026-07-14

Multi-objective metric-contract release.

### Changed

- **CONTRACT (owner-approved):** `metrics["cost"]` is the per-trial TOTAL on
  every lane (local, hybrid, pruned), reconciling with `total_cost`; the
  per-example mean moved to `metrics["cost_per_example_mean"]` (#1853, #1869).
- **CONTRACT (owner-approved):** the bare `metrics["latency"]` metric is
  MILLISECONDS on every lane; the local builtin previously recorded seconds
  (1000x cross-lane disagreement). `execution_time` stays seconds (#1855, #1872).
- Declared objective orientation is authoritative everywhere: the schema-less
  name-guess fallback now matches whole tokens (no more `uptime` treated as
  minimize), warns whenever an orientation is guessed, and plain-list unknown
  names warn before defaulting to maximize (#1852, #1870).

### Fixed

- `best_metrics` returns the winner's replicate MEAN in aggregated/hybrid runs,
  matching `best_score`; result selection stamps `winning_trial_ids` and
  persistence round-trips them (save/load keeps the aggregated winner identity)
  (#1854, #1871, #1874).

### Added

- One-shot run-level warning when a config's outputs are empty/whitespace at a
  high rate — truncation/parsing/refusal signal (#1851, #1867).
- CONTEXT-mode phantom `best_config` warning covers the no-overlap case: a tuned
  knob that is neither a function parameter nor read via `get_config()` warns
  at decoration time (#1856, #1868).

### Dependencies

- json-repair 0.61.4 (#1875).

## [0.22.0] - 2026-07-14

Weighted-selection contract release (backfilled entry; tag `v0.22.0`).

### Changed

- **CONTRACT (owner-approved):** default/uniform-weight multi-objective runs —
  strategy presets included — return the weighted-aggregate winner as
  `best_config`, not the accuracy-argmax; terminal selection and the post-hoc
  weighted artifact agree (#1846, #1850).
- Trial "score" single-meaning: `Trial.score` is populated with the primary
  objective's value; when a custom `scoring_function` owns "accuracy", the
  builtin exact-match scorer is recorded as `metrics["exact_match_default"]`
  and `metrics["score"]` carries the optimization signal (#1845, #1849).
- `ObjectiveSchema` is threaded into every post-hoc `calculate_weighted_scores`
  call site, so declared minimize orientations are honored instead of
  name-pattern autodetection (#1846 follow-up, #1857).


### Changed (observability transport)

- `ObservabilityClient.flush(timeout=0)` and `close(timeout=0)` now perform an
  immediate poll rather than treating zero as the configured 30-second flush
  timeout. No deadline-exceeded warning is emitted for this intentional
  poll-style use. No-argument flush and close retain synchronous delivery.
- `items_pending` and transport `pending_items` now count both buffered and
  in-flight payloads, so a timed-out sender remains visible until it completes.
  `get_stats()` also exposes `oldest_inflight_age_seconds` to diagnose custom
  senders that do not return.

### Fixed

- Observability delivery keeps exactly one sender active, including after a
  bounded flush returns before its sender completes. A later flush waits for
  that sender within its own deadline before beginning another batch.
- Health callbacks run after transport locks are released, and batch-delivery
  failure events explicitly identify their batch (`item_count` and capped
  `trace_ids`) instead of implying one callback per payload.
- The atexit observability close is explicitly bounded by the default
  `flush_timeout` (30 seconds). This intentionally favors a bounded interpreter
  shutdown over guaranteed delivery of a very slow final backlog.

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

## [0.21.2] - 2026-07-09

### Fixed

- **Backend session now finalizes on every failed exit path in the
  orchestrator, not just on success.** Previously a run that raised
  mid-optimization could leave its backend-tracked session open; finalize is
  now reached on every exit path (#1765, #1819).

## [0.21.1] - 2026-07-08

Session-aggregation and egress-hardening release: closes the finalize-path
gap left by 0.21.0's `@observe` metadata-only default and folds session
result submission into a single, sanitized path.

### Added

- Live-contract CI lane: schedule/dispatch runs are gated until prerequisites
  (a minted key, read from a private file rather than stdout) are met, and
  the lane validates `session_aggregation` payloads against a real backend
  before merge (#1815, #1817).

### Changed

- **`session_aggregation` rollup folded into `finalize`, replacing the dead
  `submit_result` shim.** Session objectives and cost now flow through one
  path instead of two (#1811).
- Budget cost backfill is now limited to completed trials, matching typed
  session objective/cost metrics (#1814, #1816).

### Fixed

- **SDK egress boundary hardened for `session_aggregation`.** `sdk_version`
  is bounded against a malformed override, scalar label fields and
  content-free map keys in the aggregation builder are bounded, label
  matching uses `fullmatch` (rejecting a trailing newline), and
  `statistical_significance` is sanitized before it leaves the SDK.

## [0.21.0] - 2026-07-08

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

- **Onboarding CLI** — `traigent onboard`, `traigent auth device-login`, `traigent quickstart`,
  and `traigent first-prompt --agent claude|cursor|codex`. (Backfilled 2026-08-14; previously
  recorded only as a "New in 0.12.0" note in the getting-started guide.)
- **Stdio MCP server** — `traigent mcp serve` exposes the SDK's MCP server to coding agents over
  stdio. (Backfilled 2026-08-14, same provenance.)
- **Agent workflow spans** — `add_agent_span()` records user-defined agent spans; see
  `docs/api-reference/telemetry.md`. (Backfilled 2026-08-14, same provenance.)

### Removed
- **Breaking in 0.12.0:** removed Python-orchestrated JavaScript optimization through the temporary JS bridge. `ExecutionOptions.runtime`, all `ExecutionOptions.js_*` fields, `traigent.bridges.*`, and `traigent.evaluators.JSEvaluator` are no longer available. JavaScript/TypeScript users should migrate to native `@traigent/sdk` optimization with `optimize(spec)(agentFn)` and `await wrapped.optimize(...)`; see the JS SDK's `docs/getting-started/minimal-integration.md` and `docs/MIGRATION_FROM_PYTHON.md`.

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
