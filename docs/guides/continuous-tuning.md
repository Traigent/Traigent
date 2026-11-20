# Continuous Tuning

Continuous Tuning keeps a deployed agent inside a quality bound and re-optimizes it when
something changes: a new model is released, a breach is observed, or a schedule ticks.

TVL 1.7 is the source of truth. `traigent.tuning` is sugar that emits a TVL 1.7 module
(requirements, monitors, responses); the TVL reference toolchain validates it. The sugar can
express nothing TVL cannot, and `Tuning(tvl="policy.tvl.yml")` accepts any valid TVL fragment.

A support-ticket router that re-tunes itself when a new model appears, serves whatever head
a person last accepted, and proves a candidate on a held-aside set before anyone is asked:

```python
import traigent
from traigent.tuning import Tuning, FailureRate, RandomAudit, ModelReleased, OnBreach, ProposeFix

def ticket_route_check(example, output):          # answers with a label declared below
    return "pass" if output == example["output"] else "fail"

@traigent.observe(
    agent="ticket-router",
    follow="head",                                # every call runs on the accepted head config
    tune=Tuning(
        keep=[FailureRate("customer:ticket_route_check", at_most="0.08", window="24h", name="quality_floor")],
        audit=RandomAudit(rate="0.02"),
        on=[ModelReleased(family=["anthropic"], tier=["small", "medium"], admit="model"), OnBreach()],
        fix=ProposeFix(approval="human", deploy="if_promoted"),
        recover="since_head_move",
        # promotion evidence: customer-side checker, its full label set, and a holdout
        checkers={"customer:ticket_route_check": ticket_route_check},
        checker_labels={"customer:ticket_route_check": ["pass", "fail"]},
        holdout="holdout.jsonl",                  # {"input": {...}, "output": expected} per line
    ),
)
@traigent.optimize(
    configuration_space={"model": ["anthropic/claude-haiku-4-5", "anthropic/claude-sonnet-4-6"]},
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

## What is supported where

| | Python SDK | JavaScript SDK |
| --- | --- | --- |
| Compile policy to a TVL 1.7 module | yes | yes |
| Validate with the TVL linter (`validated` only on a clean lint) | yes | yes |
| Run the loop locally (events, schedule, approvals, holdout evidence, head, followed calls) | yes: RFC 0011 Part A | no |
| RFC 0011 Part B (`when`, cooldown, `silent_for`, `fired_count`) | no: specified in TVL, not implemented by the reference relation | no |
| Release rollback (`target: release_rollback_target`) | no: deferred out of TVL 1.7; the linter reports `release_rollback_deferred` | no |

RFC 0011 and RFC 0012 are TVL drafts; this guide follows the reference relation as it stands.

## Rates are exact decimals

Rates are decimal strings or `decimal.Decimal`. Floats are rejected with `E_FLOAT_RATE`.

## What each builder emits

| Builder | TVL |
| --- | --- |
| `FailureRate(checker, at_most, window)` | a `rate_bound` requirement, one `phase: runtime` monitor (customer side, `clopper_pearson_rational_v2`) and one `phase: promotion` monitor on the evaluation set. `window` is `6h`, `12h` or `24h`. |
| `RandomAudit(rate)` | `hash_bernoulli_committed_v1` sampling on the runtime monitors (rate strictly between 0 and 1). |
| `ModelReleased(family, tier, admit)` | rule `on_new_model`: external trigger `model_released`, `admit: {tvar}`, `protects` all kept requirements. |
| `OnBreach()` | rule `on_breach` (or `on_breach_<name>` with several kept rates): monitor trigger `episode_opened`. |
| `Every(days, anchor)` | schedule trigger with `protects`. With no explicit `module_context` snapshot, the TVL `environment.snapshot_id` is the earliest `Every` anchor (TVL requires every anchor within one year after the snapshot); set `module_context={"environment": {"snapshot_id": ...}}` to choose it. |
| `ProposeFix(approval, deploy)` | `approval` and `deploy` on every rule; `approval="none"` is rejected. |

The module context (tvars, objectives, evaluation set, environment) comes from the
`@traigent.optimize` function. For tests pass `Tuning(..., module_context={...})`.

## The two human decisions

The loop asks a person exactly twice, and `traigent tune status` (or `status --text`) shows
what each decision is about.

**Approve run** allows one optimization to spend. Before you approve, `status` prints, per
attempt awaiting approval (`approve_run`): the trigger that fired (the rule's TVL trigger and
its approval mode), the **spend limit** the optimizer will run under (`max_trials`,
`cost_limit`, `timeout` of the optimized function; `none configured` when there is none), and
the model the event admits into the search.

**Accept release** lets a measured candidate into production. Before you accept, `status`
prints, per pending deploy (`accept_release`): the **incumbent** (the head the gate measured:
fingerprint and generation), the **candidate** fingerprint, the **holdout counts** per
promotion monitor (`n`, `k` violations, `u` unresolved), the rows left out of the paired
comparison, how often this holdout was used before, the **gate result** (TVL's comparison, each
monitor verdict, the joined decision, and any closed warning such as an unknown cost), and the
**observed generation** of the serving head. You accept at that generation; if the head moved
since, the acceptance is refused as stale.

An acceptance also binds the **agent code version** (RFC 0012): the code the candidate and the
incumbent were measured on, shown by `status` under Accept release as `code version`. It is the
`TRAIGENT_AGENT_CODE_VERSION` environment variable if set, else `Tuning(code_version=...)`, else
`src:` plus a sha256 of the decorated function's module source; if none can be determined, no
candidate gets promotion evidence (`E_NO_CODE_VERSION`), there is no default. If the code serving
at the moment of the write differs from the accepted one, nothing is written and the occurrence
finishes `deploy_refused` with reason `code_version_mismatch`. `head.json` records the code
version the head was accepted for, and followed-head traces carry it as `head_code_version`.

```bash
traigent tune status --text   --store .traigent/tuning/ticket-router --target agent.py:classify
traigent tune approve <occurrence>  --approver alice --authority authority.yml ...
traigent tune accept  <deploy>      --generation 0 --approver alice --authority authority.yml ...
```

## Who may decide, and how events are trusted

Nothing is allowed by default.

* **Authority.** `Controller(..., authority=ApproverAuthority(...))` (or `authorized_approvers=`)
  names who may give both decisions. Without one, `approve_optimization`, `submit_acceptance`
  and standing grants refuse with `E_NO_APPROVER_AUTHORITY`, and so do `traigent tune approve`
  and `accept` unless `--authority FILE` (or `TRAIGENT_TUNING_AUTHORITY`) names a config. An
  approver not on the list is refused (`E_APPROVER_NOT_AUTHORIZED` for Approve run; recorded as
  `unauthenticated`/`unauthorized` for Accept release). The list is checked again at the effect:
  just before the optimizer runs, and at the instant the head is written.
* **Local mode is `local-trusted`.** The config file lists approvers and trusted event issuers:

  ```yaml
  mode: local-trusted
  approvers: [alice, bob]
  event_issuers:
    platform: {secret_env: TRAIGENT_TUNING_ISSUER_PLATFORM}   # the secret is read from this variable
  ```

  `local-trusted` means: the operator running the command is trusted to be a listed approver. No
  credential is checked, so it proves nothing about who they are. Every approval, grant and
  acceptance made this way is labelled `local-trusted` in the store, and the CLI prints it.
  **Production needs the platform (portal) to authenticate the person**: pass
  `ApproverAuthority(approvers, mode="platform", authenticator=callable)`, where the callable
  asks the platform. A `local-trusted` authority is refused when `ENVIRONMENT` is a production
  name (`E_LOCAL_AUTHORITY_IN_PRODUCTION`). An authenticator that raises never authenticates.
* **Events are verified, not declared.** An external event counts only if its `signature` is the
  HMAC-SHA256 of the envelope under its issuer's secret (`traigent.tuning.sign_event`). The
  envelope's own `authenticated` field is ignored. Without a configured issuer, or with a bad
  signature, the event is still recorded, as TVL's `rejected(unauthenticated)`, and fires
  nothing (`traigent tune event` prints `authenticated: false` and warns). A shared secret is the
  minimum a local setup can do; it does not replace the platform's signing.
* **Every authority decision uses the same authenticator.** Approve run, a standing grant's
  issuer and Accept release are each checked against the list and, in `platform` mode, against
  your authenticator; one that answers False (or raises) refuses, at the decision and again
  just before the effect.
* **Standing grants** (`approval: standing`) cover the optimization only: the grant names the
  rule, a maximum count and an expiry of at most 30 days (RFC 0003 section 7.3), and its issuer
  must be on the authority list when it is recorded and at every use. Each occurrence consumes
  one use. The grant has no cost limit of its own (see known limits).
  **Standing is not supported for a deploy.** A rule with `approval: standing` and
  `deploy: if_promoted` is refused when the controller starts, with
  `E_STANDING_DEPLOY_UNSUPPORTED`; use `deploy: never` with a standing grant, or human
  approval with a deploy. (RFC 0012 section 2.3 says a standing deploy grant would carry the
  acceptance's binding fields; neither the reference relation nor this SDK implements that,
  so there is no partial version.)
  **Same meaning, two layers.** TVL's reference relation models a standing grant as platform
  state: the attempt is auto-approved. The SDK is that platform here, so it checks that a
  grant record exists, is unexpired, has uses left and has a listed issuer. `tune status`
  words the result the same way for both: "approved by standing grant G1".

## Time

Every recorded instant is a whole number of seconds after the store's origin (UTC midnight of
the day it was created), exactly the TVL relation's clock. Nothing is rounded to an hour and
nothing is recorded in the future: a window report for a window that has not ended is refused.
Event `issued_at` is an integer in the same unit (or an RFC 3339 instant, converted). Monitor
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
* The gate result names the incumbent it measured (fingerprint and generation). The acceptance
  binds the gate digest and the holdout digest of **that deploy's own attempt**, never one found
  by scanning for a fingerprint, and the write is refused if the head is no longer that incumbent.

**Holdout reuse risk.** The digest check refuses a holdout whose file bytes or content are
identical to the optimizer's `eval_dataset`. It does not prove the two sets are disjoint, nor
that the holdout is independent of anything else; rows that overlap partially, or near
duplicates, pass. Reuse is counted, not prevented: every scheduled or event-triggered run that
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

A model released after the price table shipped is unpriced. The cost preflight refuses unpriced
models, which would abort the very run an event started to try that model. For the model the
event **admitted** (and only that value), the optimization runs with its cost unknown; every
other unpriced model still fails the preflight. Unknown is never zero: if cost is an objective
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

* **Every optimization needs a recorded authorization** by a listed approver, or a standing
  grant, and it is re-checked immediately before the optimizer runs. No record, no optimize.
* **Scope is re-checked** with the TVL relation immediately before the optimizer runs; an expired
  scope is refused and recorded as a controller fact (`optimize_refused`).
* **The head moves in one commit, under the store lock** (`submit_acceptance`). The relation is
  dry-run first; only a derived `written` goes on to the physical write, which re-checks, at that
  instant: the acceptance is usable (`decided_at <= t < expires_at`), the approver is on the
  list, the serving head is at the expected generation and fingerprint, the gate measured that
  incumbent, and the candidate's configuration matches its fingerprint. It then journals an
  intent (`commits.jsonl`), installs `head.json`, records the acceptance, and journals done. The
  CLI prints the **committed** result, not a replayed verdict.
* **A refused write leaves no trace in the relation.** If the head refuses (`head_write_refused`
  in `controller_facts()`), the acceptance is not recorded as consumed, so the relation and
  `head.json` cannot disagree and the deploy keeps waiting. `step` raises `E_HEAD_DIVERGED` if
  they ever do.
* **A crash is repaired, never completed late.** After a crash the next start finishes a move
  whose head was installed (recording the acceptance at its original instant) and abandons one
  whose head was not, so an acceptance that expired meanwhile is never installed.
  Every call that appends history (events, windows, acceptances, `step`) recovers an open
  commit first, under the store lock, and the effect time of an acceptance is read after that
  lock is held, so one that expired while waiting is refused. The journal keys each attempt
  by its own commit id: an identical resubmission after an abandoned attempt is recovered on
  its own. If history already moved past the crashed acceptance's instant, the acceptance is
  placed at that original instant (the effect happened then) and later lines stay after it; if
  the relation still cannot accept that, the store refuses with `E_HEAD_DIVERGED` and the
  journal stays open.
* **Acceptance ids are per record.** `acceptance_id` covers the approver, the decision instant
  and a nonce, so a retry after an expiry or an unauthorized attempt is a new record. Only an
  exact redelivery of a recorded record is deduplicated; the same id with other content is
  `E_ACCEPTANCE_CONFLICT`.
* **The optimization runs on an isolated copy** of the `@traigent.optimize` function, so the live
  function keeps serving its head.
* **Store durability.** `meta.json` is written last (an interrupted init is redone on the next
  start). History, heads, controller facts and the commit journal are checksummed JSONL: a torn
  final line is truncated and logged on load; a damaged earlier line raises `E_STORE_CORRUPT`.
* **The CLI finds the optimizer and seeds the head.** With the order above, `observe` returns a plain
  function; the inner `@traigent.optimize` function is exposed as `fn.tuning.optimized`, and
  `traigent tune ... --target` uses it (walking `__wrapped__` as a fallback). When a store is created
  from `--target`, head generation 0 holds the optimized function's `default_config` (else the first
  value of each configuration-space dimension); if none can be determined the head starts `{}` and the
  CLI prints a warning.
* **Followed-head traces** (`observe(follow="head")`) carry only `agent`, `head_generation`,
  `head_fingerprint` and `head_code_version`; the head configuration is never copied into trace metadata.

## Known limits

* The controller runs the optimizer with `asyncio.run`, so it must not be called from inside a
  running event loop (an async framework, a notebook cell): run it from a script or a worker.
* The store lock uses `fcntl`, so the local controller is POSIX only.
* The optimizer's artifact folders are routed through process-wide environment variables for the
  duration of one candidate run (guarded by a lock): two controllers in one process, or other
  code reading those variables meanwhile, can interfere.
* Local evidence measures `accuracy` (exact match) only. Cost, latency and other objectives make
  the candidate `NoDecision` (`objective_not_measurable`).
* A standing grant has a count and an expiry but no cost limit; spend is bounded by the optimized
  function's own `max_trials` and `cost_limit`.
* The window aggregation and event ingestion are yours to run: the controller records what you
  hand it (`record_event`, `record_window`) and acts on `step`.
* Release rollback and RFC 0011 Part B are not implemented (see the support table).

## Budget

Per-occurrence spend belongs to TVL `run_budget` (RFC 0010). Until the TVL schema accepts it, a
fix budget stays execution metadata (`max_fix_cost_usd` on `@traigent.requirement`) and is not part of the
emitted module.
