# Continuous Tuning

Continuous Tuning keeps a deployed agent inside a quality bound and re-optimizes it when
something changes: a new model is released, a breach is observed, or a schedule ticks.

TVL 1.7 is the source of truth. `traigent.tuning` is sugar that emits a TVL 1.7 module
(requirements, monitors, responses); the TVL reference toolchain validates it. The sugar can
express nothing TVL cannot, and `Tuning(tvl="policy.tvl.yml")` accepts any valid TVL fragment.

```python
import traigent
from traigent.tuning import Tuning, FailureRate, RandomAudit, ModelReleased, OnBreach, ProposeFix

@traigent.observe(
    agent="doc-classifier",
    tune=Tuning(
        keep=[FailureRate("customer:doc_label_check", at_most="0.08", window="24h", name="quality_floor")],
        audit=RandomAudit(rate="0.02"),
        on=[ModelReleased(family=["anthropic"], tier=["small", "medium"], admit="model"), OnBreach()],
        fix=ProposeFix(approval="human", deploy="if_promoted"),
        recover="since_head_move",
    ),
)
@traigent.optimize(configuration_space={"model": ["anthropic/claude-haiku-4-5", "anthropic/claude-sonnet-4-6"]}, ...)
def classify(doc): ...
```

Decoration compiles and validates the policy locally (no network) and attaches the result as
`classify.tuning` (`.tvl`, `.diagnostics`, `.validated`). Calls are unchanged.

## Rates are exact decimals

Rates are decimal strings or `decimal.Decimal`. Floats are rejected with `E_FLOAT_RATE`.

## What each builder emits

| Builder | TVL |
| --- | --- |
| `FailureRate(checker, at_most, window)` | a `rate_bound` requirement, one `phase: runtime` monitor (customer side, `clopper_pearson_rational_v2`) and one `phase: promotion` monitor on the evaluation set. `window` is `6h`, `12h` or `24h`. |
| `RandomAudit(rate)` | `hash_bernoulli_committed_v1` sampling on the runtime monitors (rate strictly between 0 and 1). |
| `ModelReleased(family, tier, admit)` | rule `on_new_model`: external trigger `model_released`, `admit: {tvar}`, `protects` all kept requirements. |
| `OnBreach()` | rule `on_breach` (or `on_breach_<name>` with several kept rates): monitor trigger `episode_opened`. |
| `Every(days, anchor)` | schedule trigger with `protects`. |
| `ProposeFix(approval, deploy)` | `approval` and `deploy` on every rule; `approval="none"` is rejected. |

The module context (tvars, objectives, evaluation set, environment) comes from the
`@traigent.optimize` function. For tests pass `Tuning(..., module_context={...})`.

## Promotion evidence: checkers and a holdout

A candidate deploys only on real evidence. Give the policy local checkers and a holdout:

```python
def doc_label_check(example, output):      # a label from the requirement's closed list
    return "fail" if not output else "pass"

Tuning(keep=[FailureRate("customer:doc_label_check", at_most="0.08")], ...,
       checkers={"customer:doc_label_check": doc_label_check},
       holdout="holdout.jsonl")            # {"input": {...}, "output": expected} per line
```

After the optimizer returns, the controller freezes one candidate (the optimizer's best
config), runs your function with that config and with the current head config on every
holdout example (paired), applies each checker, and records counts `{n, k, u, pl, complete}`
(`k` = examples labelled in the requirement's `label_in`, `u` = checker or agent errors).
TVL computes the verdicts (`evaluate_promotion_monitor`) and the comparison with the head
(`epsilon_pareto_gate` on the `accuracy` objective, exact match against `output`); the
SDK computes neither. The candidate record carries the verdicts, its fingerprint, the gate
digest and `holdout_digest` (`sha256:` of the holdout file), and an acceptance binds that
digest. Checkers, holdout content and agent outputs never leave the process.

No holdout, no checker for a promotion monitor, an unreadable holdout, or an objective other
than `accuracy` records an `E_NO_PROMOTION_EVIDENCE` controller fact (`controller_facts()`,
with a closed `reason`) and the candidate stays `NoDecision`: absence never promotes.
Checkers that raise count as unresolved, and unresolved counts never yield `MET`.

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

* **Every optimization needs a recorded authorization.** `approval: human` rules wait for
  `approve_optimization`; `approval: standing` rules wait for a recorded standing grant
  (`Controller.record_standing_grant`: grant id, `response/rule`, max occurrences, expiry at most
  30 days, RFC 0003 section 7.3). Each occurrence consumes one use; a retry of the same occurrence
  does not. No record, no optimize.
* **Scope is re-checked** with the TVL relation immediately before the optimizer runs; an expired
  scope is refused and recorded as a controller fact (`optimize_refused`).
* **The head moves only on a TVL-derived acceptance write**, bound to the exact acceptance record
  that produced it. A second record with the same `acceptance_id` and other content is rejected
  (`E_ACCEPTANCE_CONFLICT`). The approver is checked against the controller's current
  `authorized_approvers` when the head is written, including recovery replays; a refusal is
  recorded (`head_write_refused`) and the head file does not move.
* **The optimization runs on an isolated copy** of the `@traigent.optimize` function, so the live
  function keeps serving its head.
* **Store durability.** `meta.json` is written last (an interrupted init is redone on the next
  start). History, heads and controller facts are checksummed JSONL: a torn final line is
  truncated and logged on load; a damaged earlier line raises `E_STORE_CORRUPT`.
* **The CLI finds the optimizer and seeds the head.** With the order above, `observe` returns a plain
  function; the inner `@traigent.optimize` function is exposed as `fn.tuning.optimized`, and
  `traigent tune ... --target` uses it (walking `__wrapped__` as a fallback). When a store is created
  from `--target`, head generation 0 holds the optimized function's `default_config` (else the first
  value of each configuration-space dimension); if none can be determined the head starts `{}` and the
  CLI prints a warning.
* **Followed-head traces** (`observe(follow="head")`) carry only `agent`, `head_generation` and
  `head_fingerprint`; the head configuration is never copied into trace metadata.

## Budget

Per-occurrence spend belongs to TVL `run_budget` (RFC 0010). Until the TVL schema accepts it, a
fix budget stays execution metadata (`max_fix_cost_usd` on `@traigent.requirement`) and is not part of the
emitted module.
