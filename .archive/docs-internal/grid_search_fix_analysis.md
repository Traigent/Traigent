# Grid Search Model Cycling Fix - Root Cause Analysis

## Problem Statement

When running FEVER case study with grid search and limited trials (e.g., 30 out of 729), only `gpt-4o-mini` configurations were being tested. Other models (`gpt-4.1-nano`, `gpt-4.1-mini`) were not seen until trial 244+.

**Note:** Spider case-study references below are historical; only `paper_experiments/case_study_fever/` exists in this repo today.

## Root Cause Analysis

### Issue #1: Base GridSearchOptimizer Parameter Ordering

**Location:** `traigent/optimizers/grid.py:161-163`

**Problem:**
```python
if "model" in sorted_names:
    remaining = [name for name in sorted_names if name != "model"]
    return ["model", *remaining]  # ❌ model FIRST
```

**Why this was wrong:**
- In `itertools.product`, the **leftmost** parameter varies **slowest**
- Placing `model` first meant iterating through all 243 combinations with `gpt-4o-mini` before moving to the next model
- With 30 trials, you'd never see other models

**Fix:**
```python
if "model" in sorted_names:
    remaining = [name for name in sorted_names if name != "model"]
    return [*remaining, "model"]  # ✅ model LAST
```

**Why this works:**
- In `itertools.product`, the **rightmost** parameter varies **fastest**
- Placing `model` last makes it cycle through all models quickly: trial 1→gpt-4o-mini, trial 2→gpt-4.1-nano, trial 3→gpt-4.1-mini, trial 4→gpt-4o-mini, etc.

### Issue #2: Spider's Global Optimizer Override

**Location:** `paper_experiments/case_study_spider/pipeline.py:49-61` (now removed)

**Problem:**
```python
class _OrderedGridSearchOptimizer(GridSearchOptimizer):
    def _generate_grid(self):
        param_names = list(self.config_space.keys())  # ❌ Insertion order
        # ... rest of implementation

register_optimizer("grid", _OrderedGridSearchOptimizer)  # ❌ GLOBAL registration
```

**Why this was a critical anti-pattern:**

1. **Import-time side effect:** The `register_optimizer()` call happens when the module is imported, not when Spider is used
2. **Global override:** Affects ALL case studies that use grid search, not just Spider
3. **Silent replacement:** No clear indication that a different optimizer is being used
4. **Insertion order dependency:** Used `list(self.config_space.keys())` which preserves FEVER's definition order (model first)

**How it affected FEVER:**

```python
# FEVER defines config_space with model FIRST
FEVER_CASE_STUDY_SPEC = ScenarioSpec(
    configuration_space={
        "model": ["gpt-4o-mini", "gpt-4.1-nano", "gpt-4.1-mini"],  # ← First!
        "temperature": [0.1, 0.3, 0.5],
        # ... other params
    }
)
```

When running FEVER:
1. `scenario_registry.py` imports Spider module to build registry
2. Spider's `register_optimizer("grid", ...)` executes, replacing standard grid optimizer
3. FEVER uses "grid" optimizer, gets Spider's custom version
4. Custom version uses insertion order → model comes first → same problem as Issue #1

**Fix:**
Completely removed the custom optimizer class and registration. Spider now uses the standard `GridSearchOptimizer` like all other case studies.

## Timeline of Events

1. **Original Design:** Base GridSearchOptimizer prioritized `model` first (intended to scan models before other params)
2. **Spider Customization:** Spider needed specific ordering, created custom optimizer with insertion order
3. **Global Registration:** Spider's optimizer registered globally, affecting all case studies
4. **Hidden Bug:** FEVER (and other case studies) silently used Spider's optimizer, getting insertion order instead of intended model prioritization
5. **Fix Attempt #1:** Changed base optimizer to put model last - but Spider override still active
6. **Discovery:** Realized Spider's global registration was preventing the fix
7. **Final Fix:** Removed Spider custom optimizer entirely

## Verification

### Before Fix
```
First 30 trials: All gpt-4o-mini ❌
Warning: "Overriding existing optimizer registration for 'grid'" ⚠️
```

### After Fix
```
First 30 trials:
  gpt-4o-mini   : 10 trials (33%) ✅
  gpt-4.1-nano  : 10 trials (33%) ✅
  gpt-4.1-mini  : 10 trials (33%) ✅

No override warnings ✅
```

## Lessons Learned

### Anti-patterns Identified

1. **Global Registration at Import Time**
   - ❌ `register_optimizer()` at module level
   - ✅ Register only when actually needed, or use scoped registration

2. **Hidden Dependencies**
   - ❌ Spider's implementation affecting unrelated case studies
   - ✅ Each case study should be independent

3. **Insertion Order Assumptions**
   - ❌ Relying on dictionary insertion order for algorithmic behavior
   - ✅ Use explicit parameter ordering

4. **Silent Overrides**
   - ❌ Replacing global singletons without clear indication
   - ✅ Use composition or explicit configuration

### Best Practices for Similar Situations

1. **Avoid Module-Level Side Effects:** Registration should happen explicitly in application startup, not at import time
2. **Scope Customizations:** Case-study-specific logic should stay within that case study
3. **Make Intentions Clear:** If you need specific ordering, use explicit `parameter_order` configuration
4. **Test Isolation:** Ensure tests don't depend on import order or global state

## Files Changed

1. `traigent/optimizers/grid.py` - Fixed parameter ordering (model last)
2. `tests/unit/optimizers/test_grid.py` - Updated test expectations
3. `paper_experiments/case_study_spider/pipeline.py` - Removed custom optimizer (historical; file not in repo)
4. `docs/grid_search_model_prioritization.md` - Documented the change

## Impact

- ✅ All case studies now benefit from fast model cycling
- ✅ No more hidden optimizer overrides
- ✅ Cleaner architecture with less coupling
- ✅ FEVER (and other case studies) work as intended with limited trial budgets

## Date

2025-10-18
