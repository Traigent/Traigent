# Agent Instructions: Call Signature Checker Bug Fixes

## Context

You are tasked with fixing bugs in the call signature mismatch detector located at `scripts/check_call_signatures.py`. This is a static analysis tool that finds mismatches between function calls and their definitions in Python code.

## Files to Modify

- **Main script:** `scripts/check_call_signatures.py` (~850 lines)
- **Test fixtures:** `tests/fixtures/signature_check/` (12 files)
- **Review doc:** `docs/reviews/call_signature_checker_review_request.md`

## Bug #1: Imported Class Method Resolution (HIGH)

### Problem
When a class with `@staticmethod` or `@classmethod` is imported from another module, calls to those methods are not resolved.

**Example that fails to detect errors:**
```python
# In imports_absolute.py
from tests.fixtures.signature_check.staticmethod_classmethod import Calculator

Calculator.static_add(1, 2, 3)  # Should ERROR: takes 2 args, got 3
Calculator.static_add(1)        # Should ERROR: missing arg 'b'
```

### Root Cause
The `SymbolTable.resolve()` method in `check_call_signatures.py` (around line 253) has a module path mismatch:

1. Definitions are stored with **relative** paths: `("staticmethod_classmethod", "Calculator.static_add")`
2. Import map resolves to **full** paths: `"tests.fixtures.signature_check.staticmethod_classmethod.Calculator.static_add"`
3. When we split at the last dot and look for `(module="...staticmethod_classmethod.Calculator", name="static_add")`, it doesn't match

### Fix Location
`scripts/check_call_signatures.py`, `SymbolTable.resolve()` method and add a helper `_lookup_resolved()`.

### Fix Strategy
1. Extract the lookup logic into a `_lookup_resolved(resolved: str)` helper method
2. In that helper, after trying exact match `(module, name)`:
   - Try Class.method pattern: split module at last dot to get `(parent_module, class_name)`, then look for `(parent_module, "class_name.name")`
   - **Key fix:** Also try with just the **last component** of parent_module: `(last_module, "class_name.name")`
   - Also try with just the last component of module for functions: `(last_module, name)`

### Test Case to Add
Add to `tests/fixtures/signature_check/imports_absolute.py`:
```python
from tests.fixtures.signature_check.staticmethod_classmethod import Calculator

# Correct calls
ok_static = Calculator.static_add(1, 2)
ok_classmethod = Calculator.from_string("42")

# ERROR cases
err_static1 = Calculator.static_add(1, 2, 3)  # too many args
err_static2 = Calculator.static_add(1)        # missing arg
err_classmethod = Calculator.from_string()    # missing arg
```

---

## Bug #2: Positional-Only Params Passed by Keyword (HIGH)

### Problem
When a function has positional-only parameters (using `/`), passing them as keyword arguments should be an error, but the checker doesn't detect it.

**Example that should error but doesn't:**
```python
def f(a, /, b):
    pass

f(1, a=2, b=3)  # Should ERROR: 'a' is positional-only, cannot be passed as keyword
```

### Root Cause
In `check_call(call, defn)` function (around line 590), the `valid_kwargs` set includes ALL parameter names, but it should exclude positional-only names:

```python
# Current (wrong):
valid_kwargs = set(defn.params.names)

# Should be:
valid_kwargs = set(defn.params.names) - posonly_names
```

Also, there's no explicit check that catches posonly args passed by keyword - we rely on the "unknown kwargs" check which fails because posonly names ARE in `valid_kwargs`.

### Fix Location
`scripts/check_call_signatures.py`, `check_call()` function, around lines 513-595.

### Fix Strategy
1. Move the posonly-as-keyword check to happen **before** the missing args check (around line 525)
2. Compute `posonly_names = set(all_positional_names[:defn.params.posonly_count])`
3. Check `posonly_as_kwarg = call.keyword_names & posonly_names`
4. If non-empty, return an ERROR with message like: `"func() got positional-only arg(s) as keyword: {'a'}"`
5. Also update the unknown kwargs check to exclude posonly names from valid_kwargs

### Test Cases to Add
Add to `tests/fixtures/signature_check/posonly_kwonly_defaults.py`:
```python
# ERROR: posonly_only() got positional-only arg(s) as keyword: {'a'}
err7 = posonly_only(1, a=2)  # 'a' is posonly

# ERROR: posonly_with_regular() got positional-only arg(s) as keyword: {'a'}
err8 = posonly_with_regular(a=1, b=2, c=3)  # 'a' is posonly

# ERROR: full_signature() got positional-only arg(s) as keyword: {'a', 'b'}
err9 = full_signature(a=1, b=2, c=3, d=4, e=5)  # 'a' and 'b' are posonly
```

---

## Bug #3: Missing Error/Warning/Info Counts in --stats (LOW)

### Problem
The review document claims `--stats` prints error/warning/info counts, but the CLI output only shows scan totals.

### Fix Location
`scripts/check_call_signatures.py`, `main()` function, around line 825.

### Fix Strategy
Add three lines after "Calls checked":
```python
print(f"Errors: {stats.errors}")
print(f"Warnings: {stats.warnings}")
print(f"Infos: {stats.infos}")
```

---

## Verification Commands

After fixing, run these to verify:

```bash
# Should show 0 errors on main codebase
python scripts/check_call_signatures.py traigent/ --min-severity error --stats

# Should detect all intentional errors in fixtures (50+ errors)
python scripts/check_call_signatures.py tests/fixtures/signature_check/ --min-severity error --stats

# Specific checks for the bugs:

# Bug #1 - Should show errors for Calculator.static_add in imports_absolute.py
python scripts/check_call_signatures.py tests/fixtures/signature_check/ --min-severity error | grep "Calculator"

# Bug #2 - Should show "positional-only arg(s) as keyword" errors
python scripts/check_call_signatures.py tests/fixtures/signature_check/ --min-severity error | grep "posonly.*keyword"

# Bug #3 - Should show Errors/Warnings/Infos counts
python scripts/check_call_signatures.py tests/fixtures/signature_check/ --stats | grep -E "^(Errors|Warnings|Infos):"
```

---

## Code Quality

After fixing, run:
```bash
.venv/bin/ruff check scripts/check_call_signatures.py --fix
.venv/bin/black scripts/check_call_signatures.py
.venv/bin/isort scripts/check_call_signatures.py
```

---

## Summary Checklist

- [ ] Fix Bug #1: Add `_lookup_resolved()` helper with fallback to last module component
- [ ] Fix Bug #2: Add posonly-as-keyword check before missing args check
- [ ] Fix Bug #3: Add error/warning/info counts to `--stats` output
- [ ] Add test fixtures for Bug #1 (imported class methods)
- [ ] Add test fixtures for Bug #2 (posonly as keyword)
- [ ] Verify 0 false positives on `traigent/`
- [ ] Verify all intentional errors detected in fixtures
- [ ] Run formatters
