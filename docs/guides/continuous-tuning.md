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

## Budget

Per-occurrence spend belongs to TVL `run_budget` (RFC 0010). Until the TVL schema accepts it, a
fix budget stays execution metadata (`max_fix_cost_usd` on `@traigent.requirement`) and is not part of the
emitted module.
