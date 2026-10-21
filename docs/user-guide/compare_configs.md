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

* `compare()` calls the evaluator that `optimize()` builds (same factory, so the
  same scoring, metric functions and cost capture) directly, once per
  (config, repeat), on one fixed example list. It does **not** go through the
  orchestrator: no backend session, no trial submission, no `save_to`
  persistence, and `default_config` is never substituted for a requested config.
  The comparison lives only in the returned object and `comparison_history`;
  nothing is synced to the portal.
* The wrapper's exclusive slot is held once for the whole comparison, so no
  `optimize()` run or promotion can overlap it. `compare()` applies no config
  and leaves the applied config and optimization history unchanged, including
  on an exception or cancellation.
* The decorator's `max_examples` cap is applied; `n_source_examples`,
  `n_examples` and `max_examples_applied` report it. `example_ids` are the ids
  the evaluator actually returned; `same_examples_across_configs` and
  `coverage_complete` are `False` if any evaluation returned a different or
  partial set (for example after a failed repeat).
* A failed evaluation is recorded as a failed repeat (score `None`, cost
  unknown); its message is reduced to the exception type.
* `best_config` follows the primary objective's declared orientation
  (maximize, minimize, or target band).
* Config values are validated with the SDK's own domain semantics: integer
  ranges need integers, `step` is enforced, categorical/fixed dicts match their
  values.
