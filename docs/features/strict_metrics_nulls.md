# TRAIGENT_STRICT_METRICS_NULLS Feature Flag

## Overview

Set `TRAIGENT_STRICT_METRICS_NULLS` to let some metric paths report a missing
value as `None` in Python (`null` in JSON). For example, some numeric conversions
return `None` when their input is missing or cannot be converted. Score
aggregation can still substitute `0.0` for a missing score.

When usage capture is enabled and available for a custom evaluator, the SDK
aggregates each of the following metrics across examples:

| Metrics | With strict nulls enabled |
| --- | --- |
| `input_tokens`, `output_tokens`, `total_tokens` | Each total is `None` if no example reports that count. |
| `input_cost`, `output_cost`, `total_cost` | Each total is `None` if no example reports that amount. Explicit zero-cost evidence can produce a zero total. When the trial is flagged `cost_unmeasured`, all three are cleared even if some examples reported spend. When some examples report zero provider attempts and the rest report a `cost`, `total_cost` is the sum of those per-example costs, even if no example reported a `total_cost` key (one example with `cost` 0.25 and one failed example with zero attempts give 0.25). |
| Requested cost objectives (`cost`, `input_cost`, `output_cost`, `total_cost`) | Set to `None` when the trial is flagged `cost_unmeasured`. |

With strict nulls off, the custom-evaluator aggregation omits these unknown
values.

## Usage

Set the environment variable before running Traigent:

```bash
export TRAIGENT_STRICT_METRICS_NULLS=true
```

Strict nulls are off by default. The environment variable accepts `true`,
`1`, `yes`, and `on`, without regard to case. Set the variable to enable
strict nulls for local evaluation metrics and custom-evaluator aggregation.

Passing a `TraigentConfig` with `strict_metrics_nulls=True` to
`traigent.initialize()` also enables strict nulls for custom-evaluator
aggregation and other metric readers that honor the configuration. Those
readers stay enabled when the variable is `false` or unset. The local
evaluation metric formatter and its empty metric defaults read the
environment variable directly.

## Best Practices

Use `.get()` to read a metric that can be absent. Check for `None` before
doing arithmetic on an optional metric.

```python
values = [None, 0.0, 1.0]
present = [value for value in values if value is not None]
average = sum(present) / len(present) if present else None
print(average)
```

For trial cost, check `cost_unmeasured` as well as `cost`. A pruned trial
can carry both the flag and a numeric cost. This example displays a cost
total when available.

```python
def cost_label(metrics):
    cost = metrics.get("cost")
    if metrics.get("cost_unmeasured") or cost is None:
        return "unmeasured"
    return f"${cost:.4f}"

for metrics in ({"cost": 1.0}, {"cost": 0.0}, {}, {"cost": 1.0, "cost_unmeasured": 1.0}):
    print(cost_label(metrics))
```

See [Token and Cost Capture](../user-guide/cost_capture.md) for details on
unmeasured cost.
