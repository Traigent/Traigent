# TRAIGENT_STRICT_METRICS_NULLS Feature Flag

## Overview

Set `TRAIGENT_STRICT_METRICS_NULLS` to let some metric paths report a missing
value as `None` in Python (`null` in JSON). For example, some numeric conversions
return `None` when their input is missing or cannot be converted. Score
aggregation can still substitute `0.0` for a missing score.

## Usage

Set the environment variable before running Traigent:

```bash
export TRAIGENT_STRICT_METRICS_NULLS=true
```

The flag is off by default. The environment variable accepts `true`, `1`,
`yes`, and `on`, without regard to case. Set it to `false` or leave it unset
to use the default. You can also enable `strict_metrics_nulls` through a
`TraigentConfig` passed to `traigent.initialize()`.

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
