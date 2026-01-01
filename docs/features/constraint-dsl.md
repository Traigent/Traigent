# Constraint DSL Developer Guide

> **Module**: `traigent.api.constraints`
> **Status**: Implemented
> **Last Updated**: 2026-01-01

> ⚠️ **Don't confuse** this module with `traigent.utils.constraints`, which contains
> legacy constraint classes (`ParameterRangeConstraint`, `ConditionalConstraint`, etc.)
> used for runtime validation. This DSL module (`traigent.api.constraints`) is for
> **declarative structural constraints** on configuration spaces.

## Overview

Traigent's Constraint DSL provides a declarative way to express structural constraints on hyperparameter configuration spaces. Constraints define valid parameter combinations and are enforced during optimization to prevent invalid configurations from being sampled.

The DSL supports **three equivalent syntax styles**, allowing developers to choose their preferred notation:

```python
from traigent import Range, Choices, implies, when

temperature = Range(0.0, 2.0)
model = Choices(["gpt-4", "gpt-3.5-turbo"])

# 1. Functional style (canonical, explicit)
implies(model.equals("gpt-4"), temperature.lte(0.7))

# 2. Operator style (concise, formula-like)
model.equals("gpt-4") >> temperature.lte(0.7)

# 3. Fluent style (readable)
when(model.equals("gpt-4")).then(temperature.lte(0.7))
```

All three produce semantically identical `Constraint` objects.

---

## Table of Contents

1. [Motivation](#motivation)
2. [Quick Reference](#quick-reference)
3. [Core Concepts](#core-concepts)
4. [Syntax Styles](#syntax-styles)
5. [Operators & Combinators](#operators--combinators)
6. [Operator Precedence](#operator-precedence)
7. [API Reference](#api-reference)
8. [Design Decisions](#design-decisions)
9. [Architecture](#architecture)
10. [Examples](#examples)
11. [Testing](#testing)
12. [Next Steps](#next-steps)

---

## Motivation

### Problem

The original constraint API used a functional style:

```python
implies(model.equals("gpt-4"), temp.lte(0.7))
```

While explicit, this becomes cumbersome for complex constraints:

```python
implies(
    and_(model.equals("gpt-4"), temp.lte(0.5)),
    or_(max_tokens.gte(1000), top_p.equals(1.0))
)
```

### Solution

A formula-style DSL using Python operators:

```python
(model.equals("gpt-4") & temp.lte(0.5)) >> (max_tokens.gte(1000) | top_p.equals(1.0))
```

This is:
- **More readable**: Resembles mathematical/logical notation
- **More concise**: No nested function calls
- **IDE-friendly**: Operators provide better auto-completion hints
- **Backward compatible**: `implies()` still works

---

## Quick Reference

### Operators

| Operator | Meaning | Example |
|----------|---------|---------|
| `>>` | Implication (if...then) | `A >> B` |
| `&` | Conjunction (and) | `A & B` |
| `\|` | Disjunction (or) | `A \| B` |
| `~` | Negation (not) | `~A` |

### Condition Builders

| Method | Meaning | Example |
|--------|---------|---------|
| `.equals(v)` | Equal to | `model.equals("gpt-4")` |
| `.not_equals(v)` | Not equal to | `model.not_equals("gpt-4")` |
| `.gt(v)` | Greater than | `temp.gt(0.5)` |
| `.gte(v)` | Greater than or equal | `temp.gte(0.5)` |
| `.lt(v)` | Less than | `temp.lt(1.0)` |
| `.lte(v)` | Less than or equal | `temp.lte(1.0)` |
| `.is_in([...])` | In set of values (Choices only) | `model.is_in(["gpt-4", "gpt-4o"])` |
| `.not_in([...])` | Not in set (Choices only) | `model.not_in(["gpt-3.5"])` |
| `.in_range(lo, hi)` | Within range (numeric ranges) | `temp.in_range(0.3, 0.7)` |

### Constraint Builders

| Function/Method | Purpose | Example |
|-----------------|---------|---------|
| `implies(A, B)` | Create implication constraint | `implies(a, b)` |
| `require(A)` | Create unconditional constraint | `require(temp.lte(1.0))` |
| `when(A).then(B)` | Fluent implication builder | `when(a).then(b)` |
| `A >> B` | Operator implication | `a >> b` |
| `A.implies(B)` | Method implication | `a.implies(b)` |

---

## Core Concepts

### BoolExpr

`BoolExpr` is the abstract base class for all boolean expressions. It defines:

- `to_expression(var_names)` - Convert to TVL string representation
- `evaluate_config(config, var_names)` - Evaluate against a configuration
- Operator overloads: `>>`, `&`, `|`, `~`

```python
from abc import ABC, abstractmethod

class BoolExpr(ABC):
    @abstractmethod
    def to_expression(self, var_names: dict[int, str]) -> str: ...

    @abstractmethod
    def evaluate_config(self, config: dict[str, Any], var_names: dict[int, str]) -> bool: ...

    def __rshift__(self, other: BoolExpr) -> Constraint:
        """A >> B creates Constraint(when=A, then=B)"""
        return Constraint(when=self, then=other)

    def __and__(self, other: BoolExpr) -> AndCondition:
        """A & B creates AndCondition((A, B))"""
        return AndCondition((self, other))

    def __or__(self, other: BoolExpr) -> OrCondition:
        """A | B creates OrCondition((A, B))"""
        return OrCondition((self, other))

    def __invert__(self) -> NotCondition:
        """~A creates NotCondition(A)"""
        return NotCondition(self)
```

### Condition

`Condition` is an atomic boolean expression comparing a parameter to a value:

```python
@dataclass
class Condition(BoolExpr):
    tvar: ParameterRange    # The parameter being constrained
    operator: OperatorType  # Comparison operator (==, <=, etc.)
    value: Any              # Value to compare against
```

### Compound Expressions

- `AndCondition((left, right))` - Conjunction of two expressions
- `OrCondition((left, right))` - Disjunction of two expressions
- `NotCondition(inner)` - Negation of an expression

### Constraint

A `Constraint` wraps boolean expressions to create enforceable rules:

```python
@dataclass
class Constraint:
    when: BoolExpr | None = None   # Antecedent (if...)
    then: BoolExpr | None = None   # Consequent (...then)
    expr: BoolExpr | None = None   # Unconditional requirement
```

- **Implication**: `Constraint(when=A, then=B)` means "if A then B"
- **Requirement**: `Constraint(expr=A)` means "A must always hold"

---

## Syntax Styles

### 1. Functional Style (Canonical)

The original, explicit API:

```python
from traigent import implies, require

# Implication: if model is gpt-4, then temperature must be <= 0.7
implies(model.equals("gpt-4"), temperature.lte(0.7))

# Requirement: temperature must always be <= 1.5
require(temperature.lte(1.5))
```

**When to use**:
- Teaching/documentation (most explicit)
- Complex logic that benefits from function call structure
- Backward compatibility requirements

### 2. Operator Style (Formula)

Concise notation using Python operators:

```python
# Implication using >>
model.equals("gpt-4") >> temperature.lte(0.7)

# Combined conditions
(model.equals("gpt-4") & temperature.lte(0.5)) >> max_tokens.gte(1000)

# Negation
~model.equals("gpt-4") >> temperature.gte(0.8)

# Disjunction
model.equals("gpt-4") >> (top_p.equals(1.0) | temperature.lte(0.5))
```

**When to use**:
- Complex constraints with multiple conditions
- Formula-like logic (reads like math)
- Concise code

### 3. Fluent Style (Builder)

Natural-language-like API using `when().then()`:

```python
from traigent import when

when(model.equals("gpt-4")).then(temperature.lte(0.7))

# Also supports .implies() method
model.equals("gpt-4").implies(temperature.lte(0.7))
```

**When to use**:
- Maximum readability
- Self-documenting code
- Users unfamiliar with operator syntax

---

## Operators & Combinators

### Implication (`>>`)

"If A then B" - when A is true, B must also be true:

```python
model.equals("gpt-4") >> temperature.lte(0.7)
# Semantics: not(model == "gpt-4") or (temperature <= 0.7)
```

### Conjunction (`&`)

"A and B" - both must be true:

```python
model.equals("gpt-4") & temperature.lte(0.7)
# Creates AndCondition, can be used in larger expressions
```

### Disjunction (`|`)

"A or B" - at least one must be true:

```python
model.equals("gpt-4") | model.equals("gpt-4o")
# Creates OrCondition
```

### Negation (`~`)

"not A" - A must be false:

```python
~model.equals("gpt-4")
# Creates NotCondition
```

### Combining Operators

Build complex constraints by combining operators:

```python
# If using gpt-4 AND low temperature, then require high max_tokens
(model.equals("gpt-4") & temperature.lte(0.5)) >> max_tokens.gte(1000)

# If NOT using gpt-4, allow any temperature
~model.equals("gpt-4") >> temperature.in_range(0.0, 2.0)

# Model must be either gpt-4 or gpt-4o when temperature is high
temperature.gt(1.0) >> (model.equals("gpt-4") | model.equals("gpt-4o"))
```

---

## Operator Precedence

⚠️ **Critical**: Python's operator precedence affects how expressions are parsed.

**Python precedence (highest to lowest):**
```
~       (unary negation)
<< >>   (shift operators, including our >>)
&       (bitwise AND)
^       (bitwise XOR)
|       (bitwise OR)
```

### The Gotcha

This means `a & b >> c` parses as `a & (b >> c)`, **NOT** `(a & b) >> c`!

Worse: since `b >> c` returns a `Constraint` (not a `BoolExpr`), the expression `a & <Constraint>` will raise a **TypeError**.

```python
# WRONG: This raises TypeError because (temp.lte(0.5) >> max_tokens.gte(1000))
# returns a Constraint, and Condition & Constraint is not supported.
model.equals("gpt-4") & temp.lte(0.5) >> max_tokens.gte(1000)

# CORRECT: Use parentheses
(model.equals("gpt-4") & temp.lte(0.5)) >> max_tokens.gte(1000)
```

### Rule of Thumb

**Always use parentheses around compound conditions:**

```python
# Antecedent is compound → parenthesize it
(A & B) >> C
(A | B) >> C

# Consequent is compound → parenthesize it
A >> (B & C)
A >> (B | C)

# Both compound → parenthesize both
(A & B) >> (C | D)
```

---

## API Reference

### Functions

#### `implies(when: BoolExpr, then: BoolExpr) -> Constraint`

Create an implication constraint.

```python
implies(model.equals("gpt-4"), temp.lte(0.7))
```

#### `require(expr: BoolExpr) -> Constraint`

Create an unconditional requirement.

```python
require(temp.lte(1.5))  # Always enforced
```

#### `when(condition: BoolExpr) -> WhenBuilder`

Start a fluent constraint builder.

```python
when(model.equals("gpt-4")).then(temp.lte(0.7))
```

#### `normalize_constraints(constraints: list, var_names: dict[int, str] | None = None) -> list[Callable]`

Normalize a mixed list of constraints, callables, and bare `BoolExpr` into callable constraint functions.

```python
constraints = normalize_constraints([
    implies(a, b),        # Already a Constraint -> to_callable()
    temp.lte(0.7),        # Bare BoolExpr → wrapped in Constraint
    lambda cfg: cfg["x"], # Callable → passed through unchanged
])
```

Note: For Constraint/BoolExpr inputs, either set the `name` on your Range/Choices/IntRange
objects or pass an explicit `var_names` mapping so config keys can be resolved.

#### `constraints_to_callables(constraints: list[Constraint], var_names: dict[int, str] | None = None) -> list[Callable[[dict[str, Any]], bool]]`

Convert a list of `Constraint` objects into callables for the decorator.

```python
from traigent import Choices, Range, constraints_to_callables, implies

model = Choices(["gpt-4", "gpt-3.5"], name="model")
temp = Range(0.0, 2.0, name="temperature")

constraint_fns = constraints_to_callables([
    implies(model.equals("gpt-4"), temp.lte(0.7)),
])
```

Note: Like `normalize_constraints`, this relies on parameter names (or `var_names`).

### Classes

#### `Condition(BoolExpr)`

Atomic boolean expression. Created via `ParameterRange` builder methods:

```python
temp.lte(0.7)       # Condition(tvar=temp, operator="<=", value=0.7)
model.equals("x")   # Condition(tvar=model, operator="==", value="x")
```

#### `AndCondition(BoolExpr)`

Conjunction of multiple expressions (stored as tuple):

```python
AndCondition((left, right))  # tuple of BoolExpr
# or via operator:
left & right
```

#### `OrCondition(BoolExpr)`

Disjunction of multiple expressions (stored as tuple):

```python
OrCondition((left, right))  # tuple of BoolExpr
# or via operator:
left | right
```

#### `NotCondition(BoolExpr)`

Negation of an expression:

```python
NotCondition(inner)
# or via operator:
~inner
```

#### `Constraint`

Enforceable constraint with optional implication structure:

```python
Constraint(when=A, then=B)  # Implication
Constraint(expr=A)          # Requirement
```

#### `WhenBuilder`

Fluent builder for `when().then()` syntax:

```python
class WhenBuilder:
    def __init__(self, condition: BoolExpr): ...
    def then(self, consequence: BoolExpr) -> Constraint: ...
```

---

## Design Decisions

| Decision | Rationale |
|----------|-----------|
| **`>>` for implication** | Common in Python DSLs (e.g., Airflow), visually suggests "leads to", avoids reserved words |
| **`&` / `|` / `~` for logic** | Python's overloadable bitwise operators, familiar from pandas/numpy |
| **`__bool__` raises TypeError** | Prevents silent bugs from `if constraint:` usage |
| **Keep `implies()` canonical** | Backward compatibility, explicit for documentation |
| **`BoolExpr` base class** | Unified interface enables operator composition |
| **`to_expression()` returns string** | Decouples constraint definition from solver (TVL, ConfigSpace, etc.) |
| **`var_names` dict by id()** | Allows same Condition class to work with any naming scheme |

### Why Not `=>` or `->`?

Python doesn't allow overloading `=>` (not an operator) or `->` (used for type hints). The `>>` operator is:
- Overloadable via `__rshift__`
- Visually similar to logical implication
- Precedent in other Python DSLs

### Why `BoolExpr` Instead of Just `Condition`?

Compound expressions (`AndCondition`, `OrCondition`, `NotCondition`) need to share the operator interface. A base class provides:
- Type safety for operator return types
- Consistent `to_expression()` / `evaluate_config()` interface
- Extensibility for future expression types

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      User Code                              │
│  model.equals("gpt-4") >> temp.lte(0.7)                    │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                     BoolExpr (ABC)                          │
│  - to_expression(var_names) -> str                         │
│  - evaluate_config(config, var_names) -> bool              │
│  - __rshift__, __and__, __or__, __invert__                 │
└───────────────────────────┬─────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        ▼                   ▼                   ▼
┌───────────────┐   ┌───────────────┐   ┌───────────────┐
│   Condition   │   │ AndCondition  │   │ NotCondition  │
│  (atomic)     │   │   (A & B)     │   │     (~A)      │
└───────────────┘   └───────────────┘   └───────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                      Constraint                             │
│  when: BoolExpr | None                                     │
│  then: BoolExpr | None                                     │
│  expr: BoolExpr | None                                     │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              to_structural_constraint()                     │
│  → TVL StructuralConstraint model                          │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              Optimization Backend                           │
│  (Optuna, ConfigSpace, etc.)                               │
└─────────────────────────────────────────────────────────────┘
```

---

## Examples

### Basic Usage

```python
from traigent import Range, IntRange, Choices, implies, require, when

# Define parameter ranges
temperature = Range(0.0, 2.0)
max_tokens = IntRange(100, 4096)
model = Choices(["gpt-4", "gpt-4o", "gpt-3.5-turbo"])
top_p = Range(0.0, 1.0)

# Define constraints
constraints = [
    # GPT-4 requires lower temperature
    model.equals("gpt-4") >> temperature.lte(0.7),

    # High temperature requires top_p = 1.0
    temperature.gt(1.0) >> top_p.equals(1.0),

    # GPT-3.5 with low temp needs more tokens
    (model.equals("gpt-3.5-turbo") & temperature.lte(0.3)) >> max_tokens.gte(2000),

    # Global requirement
    require(max_tokens.gte(100)),
]
```

### Complex Compound Constraints

```python
# Premium models with conservative settings need high token limits
(
    (model.equals("gpt-4") | model.equals("gpt-4o"))
    & temperature.lte(0.5)
    & top_p.lte(0.9)
) >> max_tokens.gte(2000)

# Either use high temperature OR high top_p, not both
~(temperature.gt(1.0) & top_p.gt(0.9))

# If NOT using gpt-4, then allow creative settings
~model.equals("gpt-4") >> (temperature.gt(1.0) | top_p.gt(0.9))
```

### In Decorator

```python
from traigent import optimize, Range, Choices

temperature = Range(0.0, 2.0)
model = Choices(["gpt-4", "gpt-3.5-turbo"])

@optimize(
    configuration_space={
        "temperature": temperature,
        "model": model,
    },
    constraints=[
        model.equals("gpt-4") >> temperature.lte(0.7),
    ],
)
def my_llm_call(temperature: float, model: str):
    ...
```

---

## Testing

### Run Constraint Tests

```bash
TRAIGENT_MOCK_MODE=true pytest tests/unit/api/test_constraints.py -v
```

### Test Coverage Areas

1. **Operator construction**: `>>`, `&`, `|`, `~` create correct types
2. **Expression generation**: `to_expression()` produces valid TVL strings
3. **Config evaluation**: `evaluate_config()` correctly validates configurations
4. **Normalization**: `normalize_constraints()` handles mixed inputs
5. **Edge cases**: Empty constraints, nested negations, self-referential

### Example Test Cases

```python
import pytest
from traigent import AndCondition, Constraint, Range

def test_operator_creates_constraint():
    temp = Range(0.0, 2.0)
    cond = temp.lte(0.7)
    constraint = cond >> temp.gte(0.1)
    assert isinstance(constraint, Constraint)
    assert constraint.when == cond

def test_conjunction_creates_and_condition():
    temp = Range(0.0, 2.0)
    a = temp.lte(0.7)
    b = temp.gte(0.1)
    result = a & b
    assert isinstance(result, AndCondition)

def test_precedence_requires_parentheses():
    temp = Range(0.0, 2.0)
    a, b, c = temp.lte(0.7), temp.gte(0.1), temp.equals(0.5)

    # Without parens: a & (b >> c)
    with pytest.raises(TypeError):
        _ = a & b >> c

    # With parens: (a & b) >> c
    with_parens = (a & b) >> c
    assert isinstance(with_parens, Constraint)
    assert isinstance(with_parens.when, AndCondition)
```

---

## Next Steps

### Short Term

1. **Add DSL-specific unit tests** in `tests/unit/api/test_constraints.py`
   - Operator precedence edge cases
   - Deep nesting (3+ levels)
   - Mixed styles in same constraint list

2. **Documentation examples** - Add to user guide with real-world patterns

3. **IDE hints** - Improve docstrings for better autocomplete

### Medium Term

1. **TVL parser support** - Update `spec_loader.py` to parse compound expressions from YAML/JSON

2. **Visualization** - Constraint graph visualization for debugging

### Long Term

1. **Constraint inference** - Suggest constraints based on optimization history

2. **Constraint relaxation** - Automatic loosening when too restrictive

3. **Cross-parameter constraints** - `temp + top_p <= 1.5` style arithmetic

---

## Debugging & Diagnostics

### Plain English Explanations

Every `BoolExpr` and `Constraint` has an `explain()` method that returns a human-readable description:

```python
from traigent import Range, Choices, implies, require

model = Choices(["gpt-4", "gpt-3.5"], name="model")
temp = Range(0.0, 2.0, name="temperature")

# Explain a condition
cond = temp.lte(0.7)
print(cond.explain())
# Output: "temperature is at most 0.7"

# Explain a compound condition
compound = model.equals("gpt-4") & temp.lte(0.5)
print(compound.explain())
# Output: "model equals 'gpt-4' AND temperature is at most 0.5"

# Explain a full constraint
c = implies(model.equals("gpt-4"), temp.lte(0.7))
print(c.explain())
# Output: "IF model equals 'gpt-4' THEN temperature is at most 0.7"

# Explain a requirement
r = require(temp.lte(1.5))
print(r.explain())
# Output: "REQUIRE: temperature is at most 1.5"
```

### Constraint Conflict Detection

When constraints cannot be satisfied together, use `check_constraints_conflict()` and `explain_constraint_violation()` to diagnose issues:

Note: `check_constraints_conflict()` is heuristic; it only reports conflicts when
a provided sample config violates every constraint in the set. Provide
representative sample configs and ensure parameter names line up with config keys.

```python
from traigent import Range, require
from traigent.api.constraints import (
    check_constraints_conflict,
    explain_constraint_violation,
)

temp = Range(0.0, 2.0, name="temperature")

# These constraints conflict - temperature can't be both <= 0.5 AND >= 0.8
c1 = require(temp.lte(0.5))
c2 = require(temp.gte(0.8))

# Check for conflicts with sample configs
conflict = check_constraints_conflict(
    [c1, c2],
    sample_configs=[
        {"temperature": 0.3},
        {"temperature": 0.6},
        {"temperature": 0.9},
    ]
)

if conflict:
    print(conflict)
    # Output:
    # Constraint conflict detected:
    #   [1] REQUIRE: temperature is at most 0.5
    #       Violated: REQUIRE: temperature is at most 0.5
    #   [2] REQUIRE: temperature is at least 0.8
    #       Violated: REQUIRE: temperature is at least 0.8
    #   Sample config: {'temperature': 0.6}
```

### Explaining Individual Violations

```python
# Check why a specific config violates a constraint
msg = explain_constraint_violation(c1, {"temperature": 0.9})
print(msg)
# Output:
# Constraint violated: REQUIRE: temperature is at most 0.5
#   Config has: temperature=0.9
```

---

## Changelog

### 2026-01-01

- Added `BoolExpr` base class with operator overloads
- Implemented `AndCondition`, `OrCondition`, `NotCondition`
- Added `WhenBuilder` and `when()` fluent builder
- Updated `normalize_constraints()` to handle bare `BoolExpr`
- Fixed `to_expression()` backward compatibility (accepts str or dict)
- Exported new classes from `traigent/__init__.py`
- **NEW**: Added `explain()` method to all `BoolExpr` and `Constraint` classes
- **NEW**: Added `check_constraints_conflict()` for conflict detection
- **NEW**: Added `explain_constraint_violation()` for debugging

---

## See Also

- [TVL Specification](../tvl/README.md) - Underlying constraint language
- [Configuration Spaces](../user-guide/configuration-spaces.md) - Defining parameter ranges
- [Optimization Guide](../user-guide/optimization.md) - Using constraints in optimization
