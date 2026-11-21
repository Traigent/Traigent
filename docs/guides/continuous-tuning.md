# Continuous Tuning

Continuous Tuning keeps a deployed agent inside a quality bound and re-optimizes it when
something changes: a breach is observed or a schedule ticks. This release is a pilot: it runs
schedule triggers with human approval of each run only (a monitor-breach trigger compiles but the controller refuses it). The local controller
runs the **spend step** (it invokes the optimizer once a person has approved); releasing a
candidate is planned, not built (a platform-signed release does not exist yet), and this controller never does it (see
[Supported in this release](#supported-in-this-release)).

TVL 1.7 is the source of truth. `traigent.tuning` is sugar that emits a TVL 1.7 module
(requirements, monitors, responses); the TVL reference toolchain validates it. The sugar can
express nothing TVL cannot, and `Tuning(tvl="policy.tvl.yml")` accepts any valid TVL fragment.

A support-ticket router that re-tunes itself on a schedule, serves its in-code default until a signed release exists, and proves a candidate on a held-aside set before anyone is asked:

```python
import traigent
from traigent.tuning import Tuning, FailureRate, RandomAudit, Every, ProposeFix

def ticket_route_check(example, output):          # answers with a label declared below
    return "pass" if output == example["output"] else "fail"

@traigent.observe(
    agent="ticket-router",
    follow="head",                                # serves the function's in-code default until a platform-signed release exists
    tune=Tuning(
        keep=[FailureRate("customer:ticket_route_check", at_most="0.08", window="24h", name="quality_floor")],
        audit=RandomAudit(rate="0.02"),
        on=[Every(30, "2026-12-01T00:00:00Z")],
        fix=ProposeFix(approval="human", deploy="if_promoted"),
        recover="since_head_move",
        # promotion evidence: customer-side checker, its full label set, and a holdout
        checkers={"customer:ticket_route_check": ticket_route_check},
        checker_labels={"customer:ticket_route_check": ["pass", "fail"]},
        holdout="holdout.jsonl",                  # {"input": {...}, "output": expected} per line
    ),
)
@traigent.optimize(
    configuration_space={"model": ["anthropic/claude-haiku-4-5", "anthropic/claude-sonnet-4-6",
                                    "anthropic/claude-haiku-5-5"]},
    objectives=["accuracy"],
    eval_dataset="eval.jsonl",                    # the optimizer's own data, never the holdout
    default_config={"model": "anthropic/claude-haiku-4-5"},
)
def classify(ticket: str) -> str: ...
```

Decoration compiles and validates the policy locally (no network) and attaches the result as
`classify.tuning` (`.tvl`, `.diagnostics`, `.validated`). Calls are unchanged. `.validated` is
true only when the TVL linter ran and reported no error-severity diagnostic; otherwise every
executing path refuses with `E_TVL_VALIDATION_FAILED`, carrying the TVL codes unchanged (for
example `release_rollback_deferred`).

## Supported in this release

The SDK compiles and validates every TVL 1.7 policy, but the local controller runs a pilot
subset. A policy outside it still compiles, checks and simulates; the controller refuses to run
it with `E_UNSUPPORTED_IN_RELEASE`, naming the feature.

| Supported in this release | Notes |
| --- | --- |
| Schedule triggers (`Every`) | a schedule scope lasts until the next tick; the end is carried out of the history replay and checked against a fresh clock before the optimizer starts |
| Human approval of each run (**Approve run**) | by an approver on the configured authority list, checked again at the effect |
| Holdout promotion evidence, reading the head, local store | as described below |

| Specified in TVL, not supported by the SDK yet | What the SDK does |
| --- | --- |
| Monitor-breach triggers (`OnBreach`) | compile and check, but a policy with a breach (episode) trigger is refused with `E_UNSUPPORTED_IN_RELEASE` when the controller is built: the pilot trigger is a schedule |
| Standing approval (`approval: standing`) and standing grants | a policy with a standing rule is refused when the controller is built; recording a grant refuses; nothing reads a grants file |
| External events (`ModelReleased`, `record_event`, `traigent tune event`) | a policy with an external trigger is refused when the controller is built; recording an event refuses |
| RFC 0011 Part B (`when`, cooldown, `silent_for`, `fired_count`) | not implemented by the reference relation |
| Releasing a candidate (`submit_acceptance`, `accept`, `traigent tune accept`) | is planned, not built (a platform-signed release does not exist yet); every release entry point refuses with `E_UNSUPPORTED_IN_RELEASE` before it touches the store, and this controller never moves `head.json` |
| Release rollback (`target: release_rollback_target`) | deferred out of TVL 1.7; the linter reports `release_rollback_deferred` |

| | Python SDK | JavaScript SDK |
| --- | --- | --- |
| Compile policy to a TVL 1.7 module | yes | yes |
| Validate with the TVL linter (`validated` only on a clean lint) | yes | yes |
| Run the loop locally (the pilot subset above) | yes | no |

### What the release witness checks

A release witness (an end-to-end run against the real CLI) should observe:

* a schedule policy runs from `step` to a human approval and an optimizer run that leaves a
  candidate pending, and the head does not move;
* `traigent tune accept ...` exits non-zero with `E_UNSUPPORTED_IN_RELEASE` and creates or
  changes nothing in the store;
* a policy with a `ModelReleased` trigger compiles and checks, and `traigent tune step` on it
  exits non-zero with `E_UNSUPPORTED_IN_RELEASE`;
* `traigent tune event FILE` exits non-zero with `E_UNSUPPORTED_IN_RELEASE` and writes nothing;
* a rule with `approval: standing` is refused with `E_UNSUPPORTED_IN_RELEASE` when the
  controller is built, and a grants file placed in the store authorizes nothing.

RFC 0011 and RFC 0012 are TVL drafts; this guide follows the reference relation as it stands.

## Rates are exact decimals

Rates are decimal strings or `decimal.Decimal`. Floats are rejected with `E_FLOAT_RATE`.

## What each builder emits

| Builder | TVL |
| --- | --- |
| `FailureRate(checker, at_most, window)` | a `rate_bound` requirement, one `phase: runtime` monitor (customer side, `clopper_pearson_rational_v2`) and one `phase: promotion` monitor on the evaluation set. `window` is `6h`, `12h` or `24h`. |
| `RandomAudit(rate)` | `hash_bernoulli_committed_v1` sampling on the runtime monitors (rate strictly between 0 and 1). |
| `ModelReleased(family, tier, admit)` | rule `on_new_model`: external trigger `model_released`, `admit: {tvar}`, `protects` all kept requirements. Compiles and checks, but the controller does not run it in this release. |
| `OnBreach()` | rule `on_breach` (or `on_breach_<name>` with several kept rates): monitor trigger `episode_opened`. Compiles and checks, but the controller does not run it in this release. |
| `Every(days, anchor)` | schedule trigger with `protects`. With no explicit `module_context` snapshot, the TVL `environment.snapshot_id` is the earliest `Every` anchor (TVL requires every anchor within one year after the snapshot); set `module_context={"environment": {"snapshot_id": ...}}` to choose it. |
| `ProposeFix(approval, deploy)` | `approval` and `deploy` on every rule; `approval="none"` is rejected. |

The module context (tvars, objectives, evaluation set, environment) comes from the
`@traigent.optimize` function. For tests pass `Tuning(..., module_context={...})`.

## The human decision this controller takes

The local loop asks a person once, and `traigent tune status` (or `status --text`) shows what
the decision is about.

**Approve run** allows one optimization to spend. Before you approve, `status` prints, per
attempt awaiting approval (`approve_run`): the trigger that fired (the rule's TVL trigger and
its approval mode) and the **spend limit** the optimizer will run under (`max_trials`,
`cost_limit`, `timeout` of the optimized function; `none configured` when there is none).

**Release** is not decided here. A measured candidate waits for a release mechanism that is planned, not built;
`status` prints, per pending deploy (`release_evidence`, read-only): the **incumbent** (the head
the gate measured: fingerprint and generation), the **candidate** fingerprint, the **holdout
counts** per promotion monitor (`n`, `k` violations, `u` unresolved), the rows left out of the
paired comparison, how often this holdout was used before, the **gate result** (TVL's
comparison, each monitor verdict, the joined decision, and any closed warning such as an
unknown cost), the **code version** the candidate and incumbent were measured on, and the
**observed generation** of the stored baseline. The code version is the
`TRAIGENT_AGENT_CODE_VERSION` environment variable if set, else `Tuning(code_version=...)`, else
`src:` plus a sha256 of the decorated function's module source; if none can be determined, no
candidate gets promotion evidence (`E_NO_CODE_VERSION`), there is no default. The local head file is never applied to a serving call (see below), so traces carry no head code version.

```bash
traigent tune status --text   --store .traigent/tuning/production/ticket-router --target agent.py:classify
traigent tune approve <occurrence>  --approver alice --authority authority.yml ...
```

`traigent tune accept` still exists so existing scripts get a clear answer: it exits non-zero
with `E_UNSUPPORTED_IN_RELEASE` before it reads or creates anything.

## Who may decide

Nothing is allowed by default.

* **Authority.** `Controller(..., authority=ApproverAuthority(...))` (or `authorized_approvers=`)
  names who may approve a run. Without one, `approve_optimization` refuses with
  `E_NO_APPROVER_AUTHORITY`, and so does `traigent tune approve` unless `--authority FILE` (or
  `TRAIGENT_TUNING_AUTHORITY`) names a config. An approver not on the list is refused
  (`E_APPROVER_NOT_AUTHORIZED`). The list is checked again at the effect, just before the
  optimizer runs.
* **Local mode is `local-trusted`.** The config file lists approvers:

  ```yaml
  mode: local-trusted
  approvers: [alice, bob]
  ```

  `local-trusted` means: the operator running the command is trusted to be a listed approver. No
  credential is checked, so it proves nothing about who they are. Every approval
  made this way is labelled `local-trusted` in the store, and the CLI prints it.
  **Production needs the platform (portal) to authenticate the person**: pass
  `ApproverAuthority(approvers, mode="platform", authenticator=callable)`, where the callable
  asks the platform. A `local-trusted` authority is refused unless the process environment is
  explicitly non-production (`dev`, `development`, `local`, `test`, `testing`, `stage`,
  `staging`; an unset or unrecognised `ENVIRONMENT` counts as production) AND the store's
  recorded environment is a recognised non-production name (a missing or unknown name is
  refused): a production store never trusts a local
  operator, whatever the process says (`E_LOCAL_AUTHORITY_IN_PRODUCTION`). The check is made
  again at every use of a saved approval. The CLI and `observe` open the
  store of the declared environment, else the process environment, else `production`. An authenticator that raises never authenticates.
* **Every authority decision uses the same authenticator.** Approve run is checked against the
  list and, in `platform` mode, against your authenticator; one that answers False (or raises)
  refuses, at the decision and again just before the effect.

## Time

Every recorded instant is a whole number of seconds after the store's origin (UTC midnight of
the day it was created), exactly the TVL relation's clock. Nothing is rounded to an hour and
nothing is recorded in the future: a window report for a window that has not ended is refused.
Monitor
windows must be aligned to `3600 * tumbling_hours` seconds after the origin. A clock set back
never moves recorded time backwards.

## Promotion evidence: checkers and a holdout

A candidate deploys only on real evidence, measured locally on **every** path: the controller
ignores any `legacy_decision` or `monitor_verdicts` an optimizer supplies (recorded as
`external_verdicts_ignored`).

After the optimizer returns, the controller freezes each candidate config, runs your function
with that config and with the current head config on every holdout example (the two are
interleaved per row, and the order alternates, so provider drift hits both alike), applies each
checker, and records counts `{n, k, u, pl, complete}` (`k` = examples labelled in the
requirement's `label_in`, `u` = unresolved). TVL computes the verdicts
(`evaluate_promotion_monitor`) and the comparison with the head (`epsilon_pareto_gate` on the
`accuracy` objective, exact match against `output`); the SDK computes neither.

Soundness rules:

* A call that raised is **unresolved**, never a score of 0.0. It counts toward
  `max_unresolved_fraction` (the monitor's, else 5 percent); a row either side could not resolve
  leaves the pair, and too many leave no evidence (`too_many_unresolved`).
* A checker answer must be in the checker's declared label set (`checker_labels`, the TVL
  checker registry) and that set must contain the requirement's `label_in`. An answer outside it
  is unresolved and reported as `E_CHECKER_LABEL_OUT_OF_SET`, so a typo cannot pass as "no
  violation".
* Every holdout row needs an expected `output`; a row without one refuses the holdout
  (`holdout_missing_expected_output`).
* A holdout whose bytes or content equal the optimized function's `eval_dataset` is refused
  (`holdout_is_optimization_eval_set`).
* The gate result names the incumbent it measured (fingerprint and generation), and carries the
  gate digest and the holdout digest of **that deploy's own attempt**, never one found by
  scanning for a fingerprint.

**Holdout reuse risk.** The digest check refuses a holdout whose file bytes or content are
identical to the optimizer's `eval_dataset`. It does not prove the two sets are disjoint, nor
that the holdout is independent of anything else; rows that overlap partially, or near
duplicates, pass. Reuse is counted, not prevented: every scheduled or breach-triggered run that
selects against the same holdout makes a lucky pass more likely, and the gate has no
multiple-testing correction. The
holdout digest ties a decision to a file, and `status` shows how many earlier gates used the
same holdout (`prior_uses`). Refresh the holdout with fresh cases on a cadence you can defend,
and treat a high `prior_uses` as a reason to refuse.

No holdout, no checker or label set for a promotion monitor, an unreadable holdout, or an
objective other than `accuracy` records an `E_NO_PROMOTION_EVIDENCE` controller fact
(`controller_facts()`, with a closed `reason`) and the candidate stays `NoDecision`: absence
never promotes.

### A model with no price yet

The cost preflight refuses unpriced models. Unknown is never zero: if cost is an objective
the candidate cannot be promoted (`cost_unknown_unpriced_model`), and otherwise the gate carries
`COST_UNKNOWN_UNPRICED_MODEL` for the person accepting.

## Validate, compile, simulate

```python
policy.to_tvl(module_context)     # the TVL 1.7 module as a dict
policy.validate(module_context)   # TVL diagnostics, unchanged (code, severity, path, message)
Tuning.from_tvl(doc)              # inverse of to_tvl for documents in builder shape
```

```bash
traigent tune compile agent.py:classify -o policy.tvl.yml
traigent tune check policy.tvl.yml          # exit 1 on errors
traigent tune simulate policy.tvl.yml --history history.yml [--format json]
```

`simulate` runs the TVL reference relation over a recorded history and prints the derived facts.
`--tvl FILE` on the controller commands lints the file first and shows TVL's error codes.

## When the TVL toolchain is missing

Validation uses the `tvl` package (`tvl.lints.lint_module`, `tvl.runtime_conditions`) and needs a
release that supports TVL 1.7. If it is not importable or too old, decoration still succeeds
and the policy is marked `validated=False`; `fn.tuning.require_validated()`, `traigent tune check`
and `traigent tune simulate` refuse with `E_TVL_VALIDATOR_UNAVAILABLE`. There is no
schema-only fallback.

### CI: do not lose the TVL tests silently

`tests/unit/tuning` needs the TVL 1.7 toolchain. Collection never imports TVL; without it the
tests skip. A CI job that must keep this coverage sets `TRAIGENT_TVL_REQUIRED=1`
(`1`, `true` or `yes`): a missing or too-old TVL then **fails** every test in the directory
instead of skipping it.

```bash
TRAIGENT_TVL_REQUIRED=1 python -m pytest -q tests/unit/tuning
```

## Local controller guarantees

* **Every optimization needs a recorded approval** by a listed approver, and it is re-checked
  immediately before the optimizer runs. No record, no optimize.
* **Deadlines are re-checked at the effect, against a fresh clock.** The scope is replayed once
  (the TVL relation); its own end (a schedule scope ends at the next tick) is carried out of that
  replay. Before authentication, under the store lock, the controller repairs the journal tail and
  prepares the start-fact append. The very last step before the optimizer is invoked is then the
  authenticator, one clock sample and an append that reads no journal: nothing scans or repairs a
  journal after the clock is sampled, and a replay that takes longer than a second never defers a
  valid run. An approved attempt the final gate refuses (the scope ended, the approver was
  revoked, or the journal is ahead of the clock) finishes `cancelled`, so a human grant that
  never spent does not hold a later schedule; the refusal is recorded as a controller fact
  (`optimize_refused`).
* **Only the SDK's own adapter is handed the final check.** `traigent_optimize_fn` calls it
  inside its artifact lock, immediately before the optimization starts. Every other injected
  optimizer, including one that advertises `accepts_candidate_dir`, is gated by the controller
  immediately before it is invoked and never receives the check.
* **The loaded dataset must be named well.** An optimizer reports the digests of the datasets it
  loaded as a list of `sha256:<64 hex>` strings; anything else (a bare string, nothing, a
  malformed entry) means the identity is unknown and the candidate has no promotion evidence
  (`eval_set_identity_unknown`).
* **The head never moves here.** `head.json` is written once, when the store is created
  (generation 0), and only read afterwards. Release acceptance is planned, not built (a platform-signed release does not exist yet); `submit_acceptance`, `accept`, `build_acceptance` and `traigent tune accept` refuse with
  `E_UNSUPPORTED_IN_RELEASE` before any I/O. `step` raises `E_HEAD_DIVERGED` if `head.json` names
  a generation the relation does not.
* **The optimization runs on an isolated copy** of the `@traigent.optimize` function, so the live
  function keeps serving its own configuration. The copy is pinned to the SDK's offline (local-only)
  execution: a tuning run sends no configuration values (prompts, models, search-space values) to the
  Traigent backend, and a function configured for a connected run (a smart algorithm such as
  `bayesian`, or a hybrid-API evaluator) is refused with `E_CONNECTED_TUNING_REFUSED`. Use `grid`
  or `random`.
* **The tuning store lock needs a POSIX platform.** `fcntl` is imported only when the store is
  locked; the CLI starts without it, and tuning on a platform without it fails with
  `E_LOCK_UNSUPPORTED_PLATFORM`.
* **Store durability.** `meta.json` is written last (an interrupted init is redone on the next
  start). History and controller facts are checksummed JSONL: a torn
  final line is truncated and logged on load; a damaged earlier line raises `E_STORE_CORRUPT`.
* **The CLI finds the optimizer and seeds the head.** With the order above, `observe` returns a plain
  function; the inner `@traigent.optimize` function is exposed as `fn.tuning.optimized`, and
  `traigent tune ... --target` uses it (walking `__wrapped__` as a fallback). When a store is created
  from `--target`, head generation 0 holds the optimized function's `default_config` (else the first
  value of each configuration-space dimension); if none can be determined the head starts `{}` and the
  CLI prints a warning.
* **`follow="head"` serves the in-code default.** No configuration in `head.json` is servable until a
  platform-signed release exists: every call runs on the function's own in-code configuration, whatever
  `head.json` holds. `read_head` and `HeadLoader` remain for status and inspection only. A damaged,
  missing or unreadable head never reaches the agent call (it is logged once and the call runs the
  in-code default).
* **Followed-head traces** (`observe(follow="head")`) carry only `agent` and `head_state`
  (`unreleased`, or `unreadable` when the head file could not be read); no generation, fingerprint or
  configuration is stamped, and the configuration is never copied into trace metadata.

## Known limits

* The controller runs the optimizer with `asyncio.run`, so it must not be called from inside a
  running event loop (an async framework, a notebook cell): run it from a script or a worker.
* The store lock uses `fcntl`, so the local controller is POSIX only.
* The optimizer's artifact folders are routed through process-wide environment variables for the
  duration of one candidate run (guarded by a lock): two controllers in one process, or other
  code reading those variables meanwhile, can interfere.
* Local evidence measures `accuracy` (exact match) only. Cost, latency and other objectives make
  the candidate `NoDecision` (`objective_not_measurable`).
* Spend per run is bounded by the optimized function's own `max_trials` and `cost_limit`.
* The window aggregation is yours to run: the controller records what you hand it
  (`record_window`) and acts on `step`.
* Release rollback and RFC 0011 Part B are not implemented (see the support table).

## Budget

Per-occurrence spend belongs to TVL `run_budget` (RFC 0010). Until the TVL schema accepts it, a
fix budget stays execution metadata (`max_fix_cost_usd` on `@traigent.requirement`) and is not part of the
emitted module.
