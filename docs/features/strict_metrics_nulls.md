# TRAIGENT_STRICT_METRICS_NULLS Feature Flag

## Overview

The `TRAIGENT_STRICT_METRICS_NULLS` feature flag controls how Traigent handles missing or invalid metrics. When enabled, it returns `null` (None in Python) for missing values instead of defaulting to `0.0`, preventing misleading calculations and preserving data integrity. On local trials, an unmeasured `cost` is the exception: it is left out in both modes.

## Motivation

Previously, Traigent defaulted missing metrics to `0.0`, which could lead to:
- **Misleading averages**: Missing data counted as zero skewed calculations
- **False precision**: Showing `0.0` implied measurement when none existed
- **Incorrect aggregations**: Sum/average operations included non-existent data
- **Hidden failures**: Real issues masked by artificial zeros

## Usage

### Enabling the Feature

Set the environment variable before running Traigent:

```bash
export TRAIGENT_STRICT_METRICS_NULLS=true
```

Accepted truthy values: `true`, `1`, `yes`, `True`, `TRUE`, `YES`

### Disabling the Feature (Default)

The feature is disabled by default for backward compatibility:

```bash
export TRAIGENT_STRICT_METRICS_NULLS=false
# Or simply don't set the variable
```

## Behavior Comparison

With the flag off, many missing metrics are reported as `0.0`. With it on, many are reported as `null` (`None` in Python). Some empty aggregates, such as an empty mean, stay `0.0` in both modes.

Cost is the exception on a local trial. When its cost could not be measured (an unpriced model, or no captured usage), the `cost` key is left out in both modes and `cost_unmeasured` is `1.0`, so `metrics.get("cost")` returns `None`.

## Affected Components

### Core Functions
- **`safe_float_convert()`** in `traigent/core/utils.py`
  - Returns `None` instead of `0.0` for invalid/missing values when flag enabled

### Metrics Tracking
- **`MetricsTracker`** in `traigent/evaluators/metrics_tracker.py`
  - Applies the flag to some missing metrics and aggregated statistics

### Evaluators
- **`BaseEvaluator`** in `traigent/evaluators/base.py`
  - LLM metrics handling respects the flag for missing values

### Orchestrator
- **`TraigentOrchestrator`** in `traigent/core/orchestrator.py`
  - Aggregation functions filter `None` values before calculations
  - Comparison operations check for `None` before numeric comparisons

## Migration Guide

### For New Projects
Enable the flag from the start to ensure data integrity:
```bash
export TRAIGENT_STRICT_METRICS_NULLS=true
```

### For Existing Projects

1. **Test with the flag enabled** in a development environment:
   ```bash
   TRAIGENT_STRICT_METRICS_NULLS=true python your_script.py
   ```

2. **Update code that expects numeric values.** An unmeasured local `cost` is left out of the metrics, so `metrics["cost"]` raises `KeyError`. Read it with `.get()`:
   ```python
   cost = metrics.get("cost")
   if cost is not None and cost > 0:
       process_cost(cost)
   ```

3. **Update aggregation logic**:
   ```python
   # Before (includes zeros)
   avg = sum(values) / len(values)

   # After (excludes None)
   non_none = [v for v in values if v is not None]
   avg = sum(non_none) / len(non_none) if non_none else None
   ```

4. **Gradually roll out** the change:
   - Enable in development/staging first
   - Monitor for any issues
   - Deploy to production when confident

## Testing

Run the comprehensive test suite:
```bash
TRAIGENT_MOCK_LLM=true pytest tests/unit/config/test_strict_metrics_nulls.py -v
```

The test suite covers:
- Configuration parsing from environment
- `safe_float_convert()` behavior
- MetricsTracker empty format generation
- JSON serialization with null values
- Backward compatibility
- Runtime flag toggling

## Best Practices

1. **Always check for None** before numeric operations:
   ```python
   if value is not None and value > threshold:
       process(value)
   ```

2. **Keep unknown distinct from zero** when you display a value:
   ```python
   cost = metrics.get("cost")  # None when cost is unmeasured
   label = "unknown" if cost is None else f"${cost:.4f}"
   ```

3. **Filter None values** in aggregations:
   ```python
   valid_values = [v for v in values if v is not None]
   if valid_values:
       average = sum(valid_values) / len(valid_values)
   ```

4. **Document expected behavior** in your code:
   ```python
   def process_metrics(metrics: dict) -> float:
       """
       Process metrics dictionary.

       Note: With TRAIGENT_STRICT_METRICS_NULLS=true,
       missing metrics will be None instead of 0.0 (an unmeasured
       local cost is left out instead)
       """
   ```

## Backward Compatibility

The feature is **disabled by default** to maintain backward compatibility. An unmeasured local `cost` is left out in both modes.

To ensure your code works with both modes:
```python
def handle_metric(value):
    # A measured 0.0 is a value; only None means no data
    if value is None:
        return "No data"
    return f"Value: {value}"
```

## Performance Impact

Minimal performance impact:
- Additional environment variable check per conversion
- None filtering in aggregations has negligible overhead
- No impact when flag is disabled

## Troubleshooting

### TypeError with None comparisons
**Error**: `TypeError: '>' not supported between instances of 'NoneType' and 'int'`
**Solution**: Add None checks before comparisons:
```python
if value is not None and value > 0:
    process(value)
```

### JSON serialization issues
**Error**: Some JSON libraries may not handle None
**Solution**: Python's `json` module handles None → null correctly by default

### Unexpected None values
**Check**: Ensure the flag is set correctly:
```python
import os
print(os.environ.get("TRAIGENT_STRICT_METRICS_NULLS"))
```

## Related Configuration

Other Traigent configuration options that work well with strict nulls:
- `TRAIGENT_MOCK_LLM`: Testing without real API calls
- `TRAIGENT_DEBUG`: Enhanced logging for troubleshooting
