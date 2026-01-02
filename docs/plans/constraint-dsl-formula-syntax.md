# Plan: Formula-Style Constraint DSL for Traigent

**Status**: Steps 1-4 Complete
**Created**: 2026-01-01

## Summary

Add operator-based constraint syntax (`>>`, `&`, `|`, `~`) by reconciling existing partial implementations in `constraints.py`, fixing file corruption, and ensuring backward compatibility.

## Motivation

The current `implies(model.equals("gpt-4"), temp.lte(0.7))` syntax is verbose. A formula-like style is more intuitive:

```python
# Operator-based (concise)
model.equals("gpt-4") >> temp.lte(0.7)

# Fluent (readable)
when(model.equals("gpt-4")).then(temp.lte(0.7))

# Canonical (explicit, unchanged)
implies(model.equals("gpt-4"), temp.lte(0.7))
```

## Design Decisions

| Decision | Rationale |
|----------|-----------|
| `>>` for implication | Common in Python DSLs, avoids reserved words |
| `&` / `|` / `~` for AND/OR/NOT | Python's bitwise operators are overloadable |
| `__bool__` raises error | Prevents `if constraint:` bugs |
| Keep `implies()` as canonical | Backward compatibility, explicit API |
| `BoolExpr` base class | Unified interface for all expression types |

## Operator Precedence Warning

Python precedence: `~` > `<<` `>>` > `&` > `^` > `|`

This means `a & b >> c` evaluates as `a & (b >> c)`, NOT `(a & b) >> c`.

**Always use parentheses for clarity:**
```python
(model.equals("gpt-4") & temp.lte(0.7)) >> max_tokens.gte(1000)
```

## Implementation Steps

### Step 1: Fix File Corruption ✅
- Remove malformed line at `elif self.operator == ">=":s: dict[int, str]) -> str:`
- Fix `NotCondition` missing closing paren
- Remove duplicate `evaluate()` methods and spliced code
- Restore valid Python syntax

### Step 2: Reconcile Existing Partial Implementations
- File already has partial `BoolExpr`, `AndCondition`, `OrCondition`, `NotCondition`
- Deduplicate and complete these rather than adding new classes
- Ensure single coherent definition for each

### Step 3: Unify `to_expression()` Signature
- Change `Condition.to_expression(var_name: str)` to `to_expression(var_names: dict[int, str])`
- Extract `var_name = var_names.get(id(self.tvar), ...)` inside the method
- Consistent signature across all `BoolExpr` subclasses

### Step 4: Update `Constraint` to Use `evaluate_config()`
- Change `Constraint.evaluate()` to call `self.when.evaluate_config()` / `self.then.evaluate_config()`
- Required for compound expressions to work correctly

### Step 5: Handle Bare `BoolExpr` in `normalize_constraints()`
- When `BoolExpr` (not `Constraint`) passed in `constraints=[temp.lte(0.7)]`
- Wrap automatically with `Constraint(expr=...)`
- Update type hint to `list[Constraint | BoolExpr | Callable]`

### Step 6: Add `when().then()` Fluent Builder
- Create `WhenBuilder` class and `when()` function
- Add to `__all__` and export from `traigent/__init__.py`

## API Surface

### New Operators on `Condition` / `BoolExpr`

```python
# Implication
model.equals("gpt-4") >> temp.lte(0.7)

# Conjunction
model.equals("gpt-4") & temp.lte(0.7)

# Disjunction
model.equals("gpt-4") | model.equals("gpt-3.5")

# Negation
~model.equals("gpt-4")
```

### Fluent Builder

```python
from traigent import when

when(model.equals("gpt-4")).then(temp.lte(0.7))
```

### Backward Compatible

```python
from traigent import implies, require

implies(model.equals("gpt-4"), temp.lte(0.7))  # unchanged
require(temp.lte(1.5))  # unchanged
```

## Testing Requirements

1. Unit tests for DSL operators: `(a & b) >> c`, `a >> (b | c)`, `~a >> b`
2. Bare `BoolExpr` wrapping in `normalize_constraints()`
3. Regression test for `to_structural_constraint()` with compound expressions
4. Validate TVL export with `spec_loader.py` parser

## Open Questions

1. Should `spec_loader.py` be updated to parse compound boolean expressions?
2. Should we add `.implies()` method on `Condition` for fluent chaining?

## Files to Modify

- `traigent/api/constraints.py` - Main implementation
- `traigent/__init__.py` - Export `when`, `BoolExpr`, `NotCondition`
- `tests/unit/api/test_constraints.py` - Add DSL operator tests
