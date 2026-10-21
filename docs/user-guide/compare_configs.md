# Comparing Fixed Configurations

`compare()` evaluates a list of fully specified configurations head-to-head and
records them as **one experiment**. It is pure evaluation: there is no search.
Use it for "winner vs. current default on a held-out set", or an A/B/N of pinned
agents.

```python
import traigent

@traigent.optimize(
    eval_dataset="dev.jsonl",
    objectives=["accuracy"],
    configuration_space={"model": ["gpt-4o-mini", "gpt-4o"], "temperature": (0.0, 1.0)},
)
def answer(question: str) -> str: ...

result = answer.compare(
    [
        {"model": "gpt-4o", "temperature": 0.0},        # winner
        {"model": "gpt-4o-mini", "temperature": 0.5},   # current default
    ],
    eval_dataset="holdout.jsonl",   # optional; default is the decorated dataset
    repeats=3,
)

result.best_config
for row in result.configs:
    print(row.config, row.mean_score, row.score_stdev, row.total_cost, row.cost_lower_bound)
print(result.holdout.status, result.holdout.holdout_valid)
```

## API

`OptimizedFunction.compare(configs, *, eval_dataset=None, repeats=1)` (sync) and
`await compare_async(...)`. It lives on the decorated function, mirroring
`optimize_sync()` / `optimize()`, so it reuses that function's evaluator,
metrics, objectives and execution policy. Past results:
`answer.comparison_history`.

* **`configs`**: each config must name every declared parameter with a value
  inside the declared configuration space, otherwise `ValueError`. Errors name
  the config's position and the declared parameter names, never client values.
* **`eval_dataset`**: a `Dataset`, JSONL path, list of paths, or inline
  examples, used for this comparison only. `optimize()` rejects a call-time
  `eval_dataset` because it would be ignored; `compare()` honors it.
* **`repeats`**: evaluations per config. Every config runs on the same examples.

## What the result contains

`ComparisonResult` (`traigent.core.comparison`): `configs` (one row per config),
`best_config`, `example_ids`, `same_examples_across_configs`, `holdout`,
`total_cost`, `cost_lower_bound`, `to_dict()`.

Each `ConfigComparison` row has `scores` (one per repeat), `mean_score`,
`score_stdev`, `score_min`/`score_max`, `metrics_mean`, `total_cost`,
`cost_lower_bound`, `cost_unmeasured_repeats`, `failed_repeats`.

### Cost is never a silent zero

If any repeat's cost could not be measured (for example no LLM usage was
captured), `total_cost` is `None` and `cost_lower_bound` holds the spend that
was actually seen (a floor). `0.0` only appears when zero spend was measured.
See [Token and Cost Capture](cost_capture.md).

### Holdout check

`result.holdout` compares the evaluation examples with the decorated (tuning)
dataset by content fingerprint:

| `status` | Meaning | `holdout_valid` |
|----------|---------|-----------------|
| `tuning_dataset` | no `eval_dataset` given; scored on the tuning data | `False` |
| `holdout` | `eval_dataset` shares no example with the tuning data | `True` |
| `overlaps_tuning_dataset` | some examples are also in the tuning data (a warning is logged) | `False` |

## Behavior notes

* `compare()` is **local fixed-config evaluation**. It calls the evaluator that
  `optimize()` builds, through the same trial lifecycle, once per
  (config, repeat) on one fixed example list. It does not call `optimize()`:
  no backend session, no trial submission, no `save_to` persistence, and
  `default_config` is never substituted for a requested config. The comparison
  lives only in the returned object and `comparison_history`; nothing is
  synced to the portal. This is **not** a no-network guarantee: the function's
  own LLM calls, or a configured hybrid API evaluator, still reach their
  providers exactly as in a local `optimize()` run.
* Cost, CI-approval and run-limit controls apply exactly as in `optimize()`,
  over the whole comparison:
  * the local CI approval gate runs before any evaluation;
  * a pre-run estimate for configs x repeats x examples is checked against the
    cost limit (`cost_limit`, `TRAIGENT_RUN_COST_LIMIT`) and needs approval
    (`cost_approved` / `TRAIGENT_COST_APPROVED`) when it exceeds it;
  * one cumulative cost enforcer covers every repeat (permit before each
    evaluation, spend accounted afterwards, failed attempts included);
  * `TRAIGENT_REQUIRE_COST_TRACKING`, strict accounting for cost objectives,
    refusal of unmeasured successful trials under an explicit cap and the
    no-usage guard raise and stop the comparison;
  * the decorator's `timeout`, `max_examples` and `max_total_examples` apply,
    and `compare(..., budget=ExecutionBudget(...))` shares a cumulative budget
    with other runs.
  When a limit is reached the result is partial: `stop_reason` names it
  (`cost_limit`, `timeout`, `max_samples_reached`, `execution_budget`),
  `completed` is `False`, and `n_runs_completed` / `n_runs_requested` say how
  far it got.
* The wrapper's exclusive slot is held once for the whole comparison, so no
  `optimize()` run or promotion can overlap it. `compare()` applies no config
  and leaves the applied config and optimization history unchanged, including
  on an exception or cancellation.
* `n_source_examples`, `n_examples` and `max_examples_applied` report any
  `max_examples` cap. `example_ids` are the ids the evaluator actually
  returned; `same_examples_across_configs` and `coverage_complete` are `False`
  if any evaluation returned a different or partial set.
* An ordinary evaluation failure is a failed repeat (score `None`, cost
  unknown), as in `optimize()`. Policy errors are not converted into failed
  repeats.
* `best_config` follows the primary objective's declared orientation
  (maximize, minimize, or target band).
* Config values are validated with the SDK's own domain semantics: integer
  ranges need integers and include the stepped high endpoint, `step` is
  enforced, categorical/fixed dicts match their values.
* `mock_llm_run` / `mock_llm_mixed` record mock-LLM provenance as for
  optimization runs.
