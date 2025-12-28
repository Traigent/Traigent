# Global Registrations Audit

## Summary

Audit of all global registration systems in the Traigent codebase to identify potential issues similar to the Spider grid optimizer problem.

**Date:** 2025-10-18

## Global Registries Found

### 1. Optimizer Registry ✅ (Core SDK - Acceptable)

**Location:** `traigent/optimizers/registry.py`

**Global State:**
```python
_OPTIMIZER_REGISTRY: dict[str, type[BaseOptimizer]] = {}
```

**Auto-Registration:**
```python
# Lines 180-183 - Runs on module import
_register_builtin_optimizers()  # Registers: grid, random, bayesian
register_optuna_optimizers()     # Registers: optuna_tpe, optuna_random, etc.
```

**Status:** ✅ **SAFE**

**Why this is acceptable:**
- Part of core SDK, not case-study-specific
- Registers built-in optimizers that are part of the Traigent API
- Clear documentation and logging
- Has `clear_registry()` for testing
- Warning when overriding: `logger.warning(f"Overriding existing optimizer registration for '{name}'")`

**Built-in Optimizers Registered:**
- `grid` → `GridSearchOptimizer`
- `random` → `RandomSearchOptimizer`
- `bayesian` → `BayesianOptimizer` (if scikit-learn available)
- `optuna_tpe` → `OptunaTPEOptimizer` (if optuna available)
- `optuna_random` → `OptunaRandomOptimizer`
- `optuna_cmaes` → `OptunaCMAESOptimizer`
- `optuna_nsga2` → `OptunaNSGAIIOptimizer`
- `optuna_grid` → `OptunaGridOptimizer`

### 2. Plugin Registry ✅ (Core SDK - Acceptable)

**Location:** `traigent/plugins/registry.py`

**Global State:**
```python
_global_registry: PluginRegistry  # Singleton instance
```

**API:**
```python
def register_plugin(plugin: TraigentPlugin) -> None:
    _global_registry.register_plugin(plugin)
```

**Status:** ✅ **SAFE**

**Why this is acceptable:**
- Part of core SDK plugin system
- Explicit API for plugin registration
- Designed for user-facing plugin development
- Has discovery mechanism for automatic loading

### 3. Metric Registry ✅ (Core SDK - Acceptable)

**Location:** `traigent/core/metric_registry.py`

**Mechanism:** Instance-based, not truly global singleton

**Status:** ✅ **SAFE**

**Why this is acceptable:**
- Not a global singleton - each context can have its own registry
- Used for metric specifications within optimization runs
- Properly scoped to optimization context

## Case Study Registrations

### ❌ Spider Custom Grid Optimizer (REMOVED)

**Previous Location:** `paper_experiments/case_study_spider/pipeline.py:49-61`

**Status:** ❌ **REMOVED** (was anti-pattern)

**What it did:**
```python
class _OrderedGridSearchOptimizer(GridSearchOptimizer):
    # Custom implementation using insertion order
    ...

register_optimizer("grid", _OrderedGridSearchOptimizer)  # ← Module-level!
```

**Why it was problematic:**
1. **Import-time side effect** - Registered during module import
2. **Global scope leak** - Affected ALL case studies, not just Spider
3. **Silent override** - Replaced "grid" optimizer globally
4. **Hard to debug** - Only showed warning in logs, not obvious to users

**Resolution:** Completely removed. Spider now uses standard `GridSearchOptimizer`.

### ✅ All Other Case Studies

**Checked:**
- `case_study_fever/`
- `case_study_kilt/`
- `case_study_rag/`
- `case_study_spider/` (after fix)
- `case_study_squad/`
- `case_study_summarization/`
- `case_study_triviaqa/`

**Result:** ✅ **NONE** found

No other case studies perform global registrations at module import time.

## Potential Risks

### Low Risk Areas ✅

1. **Optimizer Registry** - Core SDK feature, well-documented
2. **Plugin Registry** - Designed for extensibility, has clear API
3. **Metric Registry** - Properly scoped, not global singleton

### Previous High Risk (Now Fixed) ✅

1. ~~**Spider Grid Optimizer**~~ - REMOVED

## Best Practices for Future Development

### ✅ DO

1. **Use Core Registries Properly**
   ```python
   from traigent.optimizers import register_optimizer

   # Only in your application initialization code, NOT at module level
   def setup_custom_optimizers():
       register_optimizer("my_custom", MyCustomOptimizer)
   ```

2. **Document Custom Registrations**
   - Clearly document any custom registrations
   - Explain why they're needed
   - Note the scope of impact

3. **Prefer Explicit Configuration**
   ```python
   # Instead of global registration
   @traigent.optimize(
       optimizer="grid",
       optimizer_class=MyCustomGridOptimizer,  # Explicit
       ...
   )
   ```

### ❌ DON'T

1. **Never Register at Module Import Time**
   ```python
   # ❌ BAD - runs when module is imported
   register_optimizer("grid", MyCustomOptimizer)
   ```

2. **Never Override Built-in Names Without Good Reason**
   ```python
   # ❌ BAD - overrides standard optimizer
   register_optimizer("grid", MyGridVariant)

   # ✅ GOOD - use unique name
   register_optimizer("my_grid_variant", MyGridVariant)
   ```

3. **Never Leak Case-Study Logic Globally**
   ```python
   # ❌ BAD - in case_study_X/pipeline.py
   register_optimizer("some_algorithm", CaseStudySpecificImpl)

   # ✅ GOOD - keep it local
   def build_optimized_function():
       # Use standard optimizers or configure explicitly
       pass
   ```

## Recommendations

### Immediate Actions ✅ (Completed)

- [x] Remove Spider's global grid optimizer registration
- [x] Verify no other case studies have similar patterns
- [x] Document the anti-pattern

### Future Improvements 💡

1. **Add Lint Rule:** Create a linter to detect `register_*()` calls at module level in case_study_* directories

2. **Registry Scoping:** Consider adding scoped registries for case-study-specific customizations:
   ```python
   with case_study_registry("spider"):
       register_optimizer("grid", SpiderGridOptimizer)
       # Only affects Spider context
   ```

3. **Documentation:** Add section to developer docs about when global registration is appropriate

4. **Testing:** Add integration test that checks no case study modules modify global state during import

## Conclusion

**Current Status:** ✅ **CLEAN**

After removing Spider's custom grid optimizer registration:
- No case studies perform global registrations
- Core SDK registries are properly designed and scoped
- No other anti-patterns detected

**Risk Level:** 🟢 **LOW**

The codebase is now free from improper global registration patterns. Core SDK registries are acceptable because they're:
- Part of the documented API
- Well-tested
- Have clear scoping
- Include safeguards (warnings, test helpers)

---

**Audited by:** Claude Code
**Date:** 2025-10-18
**Next Review:** When adding new case studies or extending plugin system
