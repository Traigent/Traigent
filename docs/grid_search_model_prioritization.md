# Grid Search Model Prioritization Change

## Summary

Changed the GridSearchOptimizer to place the `model` parameter **last** in the iteration order instead of first, enabling faster model comparison with limited trial budgets.

## Motivation

Previously, when running grid search with a limited number of trials (e.g., 30 trials out of 729 total combinations), all trials would use the same model because `model` was placed first in the parameter order. This meant users couldn't compare different models without running the complete grid.

**Old Behavior (model first):**
- Trials 1-243: All `gpt-4o-mini`
- Trials 244-486: All `gpt-4.1-nano`
- Trials 487-729: All `gpt-4.1-mini`

**New Behavior (model last):**
- Every 3 consecutive trials cycle through all 3 models
- With 30 trials: 10 trials per model (evenly distributed)

## Technical Details

### How itertools.product Works

In Python's `itertools.product`, the **rightmost parameter varies fastest**:

```python
list(itertools.product([1, 2], ['a', 'b']))
# [(1, 'a'), (1, 'b'), (2, 'a'), (2, 'b')]
#  ^^^^^^^  ^^^^^^^  ^^^^^^^  ^^^^^^^
#  right param varies: a → b → a → b
```

By placing `model` last, it cycles through all model values before changing other parameters.

### Code Changes

**File:** `traigent/optimizers/grid.py`

Changed line 163 in `_resolve_parameter_names()`:

```python
# OLD: model first
if "model" in sorted_names:
    remaining = [name for name in sorted_names if name != "model"]
    return ["model", *remaining]

# NEW: model last
if "model" in sorted_names:
    remaining = [name for name in sorted_names if name != "model"]
    return [*remaining, "model"]
```

**Test Update:** `tests/unit/optimizers/test_grid.py`

Updated `test_model_prioritized_when_no_order` to expect model last instead of first.

**Spider Custom Optimizer:** `paper_experiments/case_study_spider/pipeline.py`

**Completely removed** the `_OrderedGridSearchOptimizer` class and its global registration. This custom optimizer was:
1. **Registered globally** at module import time, overriding the standard grid optimizer for ALL case studies (not just Spider)
2. **Using insertion order** which preserved the config_space dictionary order (model first)
3. **Redundant** after fixing the base GridSearchOptimizer - it would have just inherited everything without adding value

This was an anti-pattern because:
- Import-time side effects affect unrelated code
- Spider's implementation details leaked into other case studies
- Made debugging difficult (showed "Overriding existing optimizer registration" warning)

Now all case studies (including Spider) use the standard `GridSearchOptimizer` with model prioritization.

## Example: FEVER Case Study

**Configuration Space:**
- 3 models × 3 temperatures × 3 retriever_k × 3 evidence_selector × 3 consistency_checker × 3 verdict_threshold
- **Total:** 729 combinations

**First 15 Trials (new behavior):**
```
 1. model=gpt-4o-mini     temp=0.1 k=1 selector=tfidf
 2. model=gpt-4.1-nano    temp=0.1 k=1 selector=tfidf
 3. model=gpt-4.1-mini    temp=0.1 k=1 selector=tfidf
 4. model=gpt-4o-mini     temp=0.1 k=1 selector=tfidf
 5. model=gpt-4.1-nano    temp=0.1 k=1 selector=tfidf
 6. model=gpt-4.1-mini    temp=0.1 k=1 selector=tfidf
 7. model=gpt-4o-mini     temp=0.1 k=1 selector=tfidf
 8. model=gpt-4.1-nano    temp=0.1 k=1 selector=tfidf
 9. model=gpt-4.1-mini    temp=0.1 k=1 selector=tfidf
10. model=gpt-4o-mini     temp=0.3 k=1 selector=tfidf
...
```

**Model Distribution in First 30 Trials:**
- `gpt-4o-mini`: 10 trials (33%)
- `gpt-4.1-nano`: 10 trials (33%)
- `gpt-4.1-mini`: 10 trials (33%)

## Benefits

1. **Early Model Comparison:** See all models within the first few trials
2. **Budget-Friendly:** Get useful model comparisons even with limited trial budgets
3. **Better Sampling:** More representative sampling of the configuration space
4. **Flexibility:** Users can still override with explicit `parameter_order` if needed

## Backward Compatibility

- **Breaking Change:** Yes, but only affects the **order** of trials, not the overall results
- **Explicit Ordering:** Users with explicit `parameter_order` specifications are unaffected
- **Test Updates:** One test updated to reflect the new behavior

## Usage

No changes needed in user code. Grid search automatically prioritizes model cycling:

```python
@traigent.optimize(
    configuration_space={
        "model": ["gpt-4o-mini", "gpt-4.1-nano", "gpt-4.1-mini"],
        "temperature": [0.1, 0.3, 0.5],
        # ... other parameters
    },
    objectives=["quality", "cost"],
)
def my_function(input):
    return result
```

With `--trials 30`, you'll now see all 3 models tested instead of just one.

## Date

2025-10-18
