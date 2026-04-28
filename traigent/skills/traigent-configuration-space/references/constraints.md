# Constraint System Reference

Complete reference for the Traigent constraint system. Constraints define valid parameter combinations and are enforced during optimization.

## Imports

```python
from traigent import Range, IntRange, Choices, LogRange
from traigent import implies, require, when
from traigent.api.constraints import (
    BoolExpr,
    Condition,
    Constraint,
    AndCondition,
    OrCondition,
    NotCondition,
)
from traigent.api.config_space import ConfigSpace
```

---

## Lambda Constraints

The simplest constraint form. Pass callables that accept a config dict and return `True` for valid configurations.

### Config-only lambda

```python
constraints=[
    lambda config: config["temperature"] < 0.8,
    lambda config: config["max_tokens"] >= 256,
    lambda config: config["temperature"] < 0.8 if config["model"] == "gpt-4o" else True,
]
```

### Config + metrics lambda

Lambdas can optionally accept a second `metrics` argument containing results from previous trials:

```python
constraints=[
    lambda config, metrics: metrics.get("cost", 0) <= 0.10,
    lambda config, metrics: metrics.get("latency_ms", 0) <= 5000,
]
```

---

## Builder Methods on Parameter Ranges

### Numeric Types (Range, IntRange, LogRange)

These methods create `Condition` objects:

```python
temp = Range(0.0, 2.0)

temp.equals(0.5)          # temp == 0.5
temp.not_equals(0.5)      # temp != 0.5
temp.gt(0.5)              # temp > 0.5
temp.gte(0.5)             # temp >= 0.5
temp.lt(0.5)              # temp < 0.5
temp.lte(0.5)             # temp <= 0.5
temp.in_range(0.3, 0.7)   # 0.3 <= temp <= 0.7
temp.is_in([0.0, 0.5])    # temp in [0.0, 0.5]
temp.not_in([0.0, 1.0])   # temp not in [0.0, 1.0]
```

### Categorical Type (Choices)

```python
model = Choices(["gpt-4o-mini", "gpt-4o", "claude-3-5-sonnet-20241022"])

model.equals("gpt-4o")                          # model == "gpt-4o"
model.not_equals("gpt-4o")                      # model != "gpt-4o"
model.is_in(["gpt-4o", "gpt-4o-mini"])          # model in ["gpt-4o", "gpt-4o-mini"]
model.not_in(["claude-3-5-sonnet-20241022"])     # model not in [...]
```

---

## implies(when, then)

Create an implication constraint: "if `when` is true, then `then` must also be true."

The constraint is satisfied when either:
- The `when` condition is false (constraint does not apply), OR
- The `then` condition is true (constraint is met)

```python
from traigent import implies

model = Choices(["gpt-4o-mini", "gpt-4o"])
temp = Range(0.0, 1.0)
max_tokens = IntRange(100, 4096)

# If model is gpt-4o, temperature must be <= 0.7
c1 = implies(model.equals("gpt-4o"), temp.lte(0.7))

# With description and id
c2 = implies(
    model.equals("gpt-4o"),
    max_tokens.gte(1000),
    description="GPT-4o requires at least 1000 tokens",
    id="gpt4o-min-tokens",
)
```

**Signature:**

```python
def implies(
    when: BoolExpr,
    then: BoolExpr,
    description: str | None = None,
    id: str | None = None,
) -> Constraint
```

---

## require(condition)

Create a standalone constraint that must always hold, regardless of other parameter values.

```python
from traigent import require

temp = Range(0.0, 2.0)

# Temperature must always be <= 1.5
c = require(temp.lte(1.5))

# With description
c = require(
    temp.lte(1.5),
    description="Temperature must not exceed 1.5",
)
```

**Signature:**

```python
def require(
    condition: BoolExpr,
    description: str | None = None,
    id: str | None = None,
) -> Constraint
```

---

## when(condition).then(consequence)

Fluent builder for implication constraints. Equivalent to `implies()` but reads more naturally.

```python
from traigent import when

model = Choices(["gpt-4o-mini", "gpt-4o"])
temp = Range(0.0, 1.0)

c = when(model.equals("gpt-4o")).then(temp.lte(0.7))

# With description
c = when(model.equals("gpt-4o")).then(
    temp.lte(0.7),
    description="Limit temperature for GPT-4o",
)
```

---

## BoolExpr Operators

`Condition` objects (and all `BoolExpr` subclasses) support these operators for combining conditions:

### AND: `&`

Both conditions must be true.

```python
model.equals("gpt-4o") & temp.lte(0.7)
```

Returns an `AndCondition`.

### OR: `|`

At least one condition must be true.

```python
model.equals("gpt-4o") | model.equals("gpt-4o-mini")
```

Returns an `OrCondition`.

### NOT: `~`

Negates the condition.

```python
~model.equals("gpt-4o")   # model is NOT gpt-4o
```

Returns a `NotCondition`.

### Implication: `>>`

Shorthand for `implies()`.

```python
model.equals("gpt-4o") >> temp.lte(0.7)
```

Returns a `Constraint`.

### Operator Precedence

Python operator precedence for these operators:

```
~ (highest)  >  >>  >  &  >  |  (lowest)
```

This means `a & b >> c` evaluates as `a & (b >> c)`, NOT `(a & b) >> c`. Always use parentheses:

```python
# Correct
(model.equals("gpt-4o") & temp.gte(0.5)) >> max_tokens.gte(1000)

# Wrong -- unexpected grouping
model.equals("gpt-4o") & temp.gte(0.5) >> max_tokens.gte(1000)
```

### Boolean Context Warning

`BoolExpr` objects cannot be used with Python's `and`, `or`, `not` keywords. Use `&`, `|`, `~` instead:

```python
# Wrong -- raises TypeError
if model.equals("gpt-4o") and temp.lte(0.7):
    ...

# Correct
combined = model.equals("gpt-4o") & temp.lte(0.7)
```

---

## Constraint Class

The `Constraint` dataclass represents a structural constraint.

```python
from traigent.api.constraints import Constraint

# Implication constraint
c = Constraint(
    when=model.equals("gpt-4o"),
    then=temp.lte(0.7),
    description="GPT-4o requires low temperature",
    id="gpt4o-temp",
)

# Standalone expression constraint
c = Constraint(
    expr=temp.lte(1.5),
    description="Temperature cap",
)
```

**Attributes:**

| Attribute     | Type              | Description                                      |
| ------------- | ----------------- | ------------------------------------------------ |
| `when`        | `BoolExpr` or None| Guard condition (for implications)               |
| `then`        | `BoolExpr` or None| Consequent condition (for implications)          |
| `expr`        | `BoolExpr` or None| Standalone condition (mutually exclusive with when/then) |
| `description` | `str` or None     | Human-readable description                       |
| `id`          | `str` or None     | Identifier for the constraint                    |

A `Constraint` must have either `(when, then)` or `expr`, not both.

**Key Methods:**

| Method                           | Returns    | Description                                        |
| -------------------------------- | ---------- | -------------------------------------------------- |
| `.evaluate(config, var_names)`   | `bool`     | Test if a config satisfies the constraint          |
| `.explain(var_names)`            | `str`      | Human-readable explanation                         |
| `.to_callable(var_names)`        | `Callable` | Convert to `config -> bool` function               |
| `.is_implication`                | `bool`     | True if this is a when/then constraint             |

---

## ConfigSpace with Constraints

Bundle parameters and constraints together for validation and reuse:

```python
from traigent import Range, IntRange, Choices, implies
from traigent.api.config_space import ConfigSpace

temp = Range(0.0, 1.0, name="temperature")
tokens = IntRange(100, 4096, name="max_tokens")
model = Choices(["gpt-4o-mini", "gpt-4o"], name="model")

space = ConfigSpace(
    tvars={"temperature": temp, "max_tokens": tokens, "model": model},
    constraints=[
        implies(model.equals("gpt-4o"), temp.lte(0.7)),
        implies(model.equals("gpt-4o"), tokens.gte(256)),
    ],
)

# Validate a config
result = space.validate({"temperature": 0.5, "max_tokens": 500, "model": "gpt-4o"})
print(result.is_valid)        # True
print(result.violations)      # [] (empty if valid)

# Check if the space has any valid configurations
sat = space.check_satisfiability()
```

---

## Mixing Constraint Styles

All constraint styles can be mixed in a single `constraints` list:

```python
from traigent import Range, Choices, implies, require

model = Choices(["gpt-4o-mini", "gpt-4o"], name="model")
temp = Range(0.0, 1.5, name="temperature")

constraints = [
    # Functional style
    implies(model.equals("gpt-4o"), temp.lte(0.7)),

    # Operator style
    model.equals("gpt-4o-mini") >> temp.lte(1.0),

    # Standalone requirement
    require(temp.lte(1.5)),

    # Bare BoolExpr (auto-wrapped with require())
    temp.gte(0.0),

    # Legacy lambda
    lambda config: config.get("max_tokens", 0) <= 4096,
]
```

The `normalize_constraints()` utility handles converting this mixed list to a uniform format.

---

## Constraint Scope Validation

When using builder-style constraints with the decorator, Traigent validates that all parameter ranges referenced by constraints are in scope (i.e., registered in the decorator):

```python
model = Choices(["a", "b"])
budget = Range(1.0, 100.0)  # Not passed to decorator

@traigent.optimize(
    model=model,  # Only model is in scope
    constraints=[
        implies(budget.lte(10), model.is_in(["a"])),  # Raises ConstraintScopeError
    ],
)
def my_func(query):
    ...
```

This raises `ConstraintScopeError` with a message identifying the out-of-scope parameter and listing available ones.

---

## Utility Functions

### constraints_to_callables

Convert a list of `Constraint` objects to callable functions:

```python
from traigent import constraints_to_callables

fns = constraints_to_callables(constraints, var_names=space.var_names)
# Each fn: config dict -> bool
```

### normalize_constraints

Convert a mixed list of `Constraint`, `BoolExpr`, and callable objects to a uniform list of callables:

```python
from traigent import normalize_constraints

normalized = normalize_constraints(mixed_constraints_list, var_names=space.var_names)
```

### explain_constraint_violation

Get a human-readable explanation of why a constraint is violated:

```python
from traigent.api.constraints import explain_constraint_violation

msg = explain_constraint_violation(constraint, config={"temperature": 0.9})
if msg:
    print(msg)  # "Constraint violated: IF ... THEN ...\n  Config has: temperature=0.9"
```

### check_constraints_conflict

Check if constraints conflict with each other given sample configs:

```python
from traigent.api.constraints import check_constraints_conflict

conflict = check_constraints_conflict(
    constraints,
    sample_configs=[{"temperature": 0.3}, {"temperature": 0.9}],
    samples_exhaustive=True,
)
if conflict:
    print(conflict)
```
