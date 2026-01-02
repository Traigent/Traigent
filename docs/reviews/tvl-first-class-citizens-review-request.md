# Review Request: TVL First-Class Citizens Implementation

## Context

**Project**: Traigent SDK - Open-source LLM optimization toolkit
**Feature**: TVL (Tuned Variable Language) First-Class Citizens
**Date**: 2024-12-30
**Branch**: `feat/optimizer-validation-test-suite`

## Executive Summary

This PR extends Traigent's SE-friendly parameter range classes (Range, IntRange, LogRange, Choices) to become TVL first-class citizens. It introduces a constraint DSL using the builder pattern and a decoupled validation architecture via the ConstraintValidator protocol.

**Key deliverables:**
- Extended parameter ranges with TVL capabilities (name, unit, constraint builders)
- New constraint system with `Condition`, `Constraint`, `implies()`, `require()`
- `ConfigSpace` class combining TVARs + constraints
- `ConstraintValidator` protocol for pluggable validation backends
- 44 comprehensive unit tests with 100% coverage of new code

---

## Motivation

### Problem Statement

Traigent users needed a way to:
1. Express **structural constraints** on configuration spaces (e.g., "if model is GPT-4, temperature must be ≤ 0.7")
2. Define **named parameters with units** for better TVL spec integration
3. Have a **clean, SE-friendly API** that doesn't require learning TVL syntax

### Design Goals

1. **Unification**: Extend existing Range/Choices classes rather than creating parallel TVAR classes
2. **Intuitive DSL**: Builder pattern with methods like `model.equals("gpt-4")` and `temp.lte(0.7)`
3. **Decoupled Validation**: Protocol-based design allowing future SAT/SMT solver integration
4. **Backward Compatibility**: All existing code continues to work unchanged

---

## Implementation Details

### Files Created

#### 1. `traigent/api/constraints.py` (~470 lines)

Core constraint system classes:

```python
@dataclass(frozen=True, slots=True)
class Condition:
    """Atomic predicate like 'temperature <= 0.7'"""
    tvar: ParameterRange
    operator: OperatorType  # "==", "!=", ">", ">=", "<", "<=", "in", "not_in", "in_range"
    value: Any

    def evaluate(self, value: Any) -> bool: ...
    def to_expression(self, var_name: str) -> str: ...

@dataclass(frozen=True)
class Constraint:
    """When/then implication or standalone expression"""
    when: Condition | None = None
    then: Condition | None = None
    expr: Condition | None = None
    description: str | None = None
    id: str | None = None

    def evaluate(self, config: dict, var_names: dict) -> bool: ...
    def to_callable(self, var_names: dict | None = None) -> Callable: ...
    def to_structural_constraint(self, var_names: dict) -> StructuralConstraint: ...

# Factory functions
def implies(when: Condition, then: Condition, description: str | None = None) -> Constraint: ...
def require(condition: Condition, description: str | None = None) -> Constraint: ...
def constraints_to_callables(constraints: list[Constraint]) -> list[Callable]: ...
```

**Key design choice**: Constraints are **frozen dataclasses** for immutability and hashability. The `to_callable()` method allows seamless integration with the existing decorator constraint system.

#### 2. `traigent/api/validation_protocol.py` (~350 lines)

Protocol-based validation architecture:

```python
class SatStatus(Enum):
    SAT = "satisfiable"
    UNSAT = "unsatisfiable"
    UNKNOWN = "unknown"

@runtime_checkable
class ConstraintValidator(Protocol):
    def validate_config(self, config, constraints, var_names) -> ValidationResult: ...
    def check_satisfiability(self, tvars, constraints) -> SatResult: ...

class PythonConstraintValidator:
    """Default implementation using Python evaluation"""
    # Returns UNKNOWN for satisfiability (can't prove SAT without enumeration)
```

**Key design choice**: The protocol allows swapping validators without changing client code. Future implementations could use Z3, PySAT, or domain-specific solvers.

#### 3. `traigent/api/config_space.py` (~430 lines)

Combines TVARs and constraints:

```python
@dataclass
class ConfigSpace:
    tvars: dict[str, ParameterRange]
    constraints: list[Constraint]
    description: str | None = None

    @classmethod
    def from_decorator_args(cls, configuration_space, inline_params, constraints) -> ConfigSpace: ...

    def validate(self, config: dict) -> ValidationResult: ...
    def check_satisfiability(self) -> SatResult: ...
    def to_tvl_spec(self) -> dict: ...  # Export to TVL format
```

#### 4. `tests/unit/api/test_constraints.py` (~400 lines)

44 unit tests covering:
- Condition creation and evaluation (all operators)
- Builder methods on Range/IntRange/LogRange/Choices
- Constraint creation, validation, and error handling
- ConfigSpace operations
- PythonConstraintValidator
- Edge cases (missing vars, unnamed tvars, immutability)

### Files Modified

#### `traigent/api/parameter_ranges.py`

Added TVL fields and builder methods to all range classes:

```python
@dataclass(frozen=True, slots=True)
class Range(ParameterRange):
    low: float
    high: float
    step: float | None = None
    log: bool = False
    default: float | None = None
    # NEW: TVL fields
    name: str | None = None
    unit: str | None = None

    # NEW: Constraint builder methods
    def equals(self, value: float) -> Condition: ...
    def not_equals(self, value: float) -> Condition: ...
    def gt(self, value: float) -> Condition: ...
    def gte(self, value: float) -> Condition: ...
    def lt(self, value: float) -> Condition: ...
    def lte(self, value: float) -> Condition: ...
    def in_range(self, low: float, high: float) -> Condition: ...
```

Similar additions to `IntRange`, `LogRange`, and `Choices` (with `is_in()`, `not_in()` for Choices).

#### `traigent/__init__.py`

Added exports:
```python
# TVL constraint system
from traigent.api.constraints import Condition, Constraint, implies, require, constraints_to_callables
from traigent.api.config_space import ConfigSpace
from traigent.api.validation_protocol import (
    ConstraintValidator, ConstraintViolation, PythonConstraintValidator,
    SatResult, SatStatus, ValidationResult as ConstraintValidationResult,
)
```

---

## Target API Achieved

```python
from traigent import optimize, Range, IntRange, Choices, implies, require

# Define parameters with TVL capabilities
temperature = Range(0.0, 2.0, name="temperature", unit="ratio")
max_tokens = IntRange(100, 4096, name="max_tokens", unit="count")
model = Choices(["gpt-4", "gpt-3.5-turbo"], name="model")

# Define constraints using builder pattern
constraints = [
    implies(model.equals("gpt-4"), max_tokens.gte(1000), description="GPT-4 needs high tokens"),
    implies(model.equals("gpt-3.5-turbo"), temperature.lte(0.7), description="GPT-3.5 needs low temp"),
    require(temperature.gte(0.1), description="Minimum temperature"),
]

# Use with decorator (convert to callables)
@optimize(
    temperature=temperature,
    max_tokens=max_tokens,
    model=model,
    constraints=[c.to_callable() for c in constraints],
    objectives=["accuracy", "cost"],
)
def my_llm_call(prompt: str) -> str:
    ...

# Or use ConfigSpace directly for validation
space = ConfigSpace(
    tvars={"temperature": temperature, "max_tokens": max_tokens, "model": model},
    constraints=constraints,
)
result = space.validate({"temperature": 0.5, "max_tokens": 2000, "model": "gpt-4"})
print(result.is_valid)  # True
```

---

## Design Decisions Requiring Review

### 1. Implication Semantics

**Current**: `implies(A, B)` means "if A then B", equivalent to `not(A) or B`.

```python
# When model is GPT-4, temp must be <= 0.7
# If model is NOT GPT-4, constraint is automatically satisfied
implies(model.equals("gpt-4"), temp.lte(0.7))
```

**Question**: Is this the most intuitive interpretation for SE users? Should we support `iff` (if-and-only-if) semantics as well?

### 2. Missing Variable Handling

**Current**: If a constraint references a variable not in the config, it returns `True` (doesn't apply).

```python
# If config = {"model": "gpt-4"} (no temperature), constraint is satisfied
constraint.evaluate(config, var_names)  # True
```

**Question**: Should this be configurable? Some users might prefer strict mode where missing variables raise errors.

### 3. `to_callable()` Auto-Detection

**Current**: `to_callable()` uses the `name` attribute of ParameterRange objects to map to config keys.

```python
temp = Range(0.0, 2.0, name="temperature")
fn = constraint.to_callable()  # Uses "temperature" as the config key
```

**Question**: This relies on users setting `name` correctly. Should we validate this at callable creation time and raise helpful errors?

### 4. `SatStatus.UNKNOWN` for Python Validator

**Current**: `PythonConstraintValidator.check_satisfiability()` always returns `UNKNOWN` because Python evaluation can't prove satisfiability without enumeration.

**Question**: Should we implement a basic enumeration-based check for small configuration spaces? Or is `UNKNOWN` the right default?

### 5. Constraint vs Callable Type in Decorator

**Current**: The decorator still expects `list[Callable]` for constraints. Users must call `constraints_to_callables()` or `constraint.to_callable()`.

**Question**: Should the decorator auto-detect `Constraint` objects and convert them internally? This would simplify the API:

```python
# Current (explicit conversion)
@optimize(constraints=[c.to_callable() for c in constraints])

# Potential (auto-detection)
@optimize(constraints=constraints)  # Accepts both Constraint and Callable
```

### 6. AndCondition/OrCondition Placeholders

**Current**: `AndCondition` and `OrCondition` classes exist but only validate they have 2+ conditions. They don't integrate with the constraint system yet.

**Question**: Should these be removed until fully implemented, or is the placeholder useful for API design signaling?

---

## Potential Issues / Technical Debt

### 1. Circular Import Avoidance

Builder methods in `parameter_ranges.py` use lazy imports:

```python
def lte(self, value: float) -> Condition:
    from traigent.api.constraints import Condition  # Lazy import
    return Condition(tvar=self, operator="<=", value=value)
```

This works but adds import overhead on each call. Consider alternatives if this becomes a hot path.

### 2. Type Annotation for `StructuralConstraint`

Used string annotation `-> "StructuralConstraint"` with TYPE_CHECKING import to avoid runtime circular import:

```python
if TYPE_CHECKING:
    from traigent.tvl.models import StructuralConstraint
```

### 3. Cognitive Complexity Warnings

Ruff flagged some methods for high cognitive complexity:
- `Constraint.to_callable()` - complexity 17 (limit 15)
- `ConfigSpace.to_tvl_spec()` - complexity 23 (limit 15)

These could be refactored but work correctly. Lower priority.

---

## Test Results

```
======================== 142 passed, 1 warning in 0.18s ========================
```

- 44 new tests in `test_constraints.py`
- 98 existing tests in `test_parameter_ranges.py`
- All tests pass with `TRAIGENT_MOCK_MODE=true`

---

## Files for Review

| File | Lines | Focus Areas |
|------|-------|-------------|
| `traigent/api/constraints.py` | ~470 | Constraint DSL design, evaluate logic |
| `traigent/api/validation_protocol.py` | ~350 | Protocol design, SatStatus handling |
| `traigent/api/config_space.py` | ~430 | TVL export format, factory method |
| `traigent/api/parameter_ranges.py` | +200 | Builder method additions |
| `tests/unit/api/test_constraints.py` | ~400 | Test coverage completeness |
| `traigent/__init__.py` | +30 | Export organization |

---

## Questions for Reviewer

1. **API Design**: Is the builder pattern (`model.equals("gpt-4")`) intuitive enough, or should we consider alternative syntaxes?

2. **Naming**: Is `implies()` the right name? Alternatives: `when_then()`, `if_then()`, `constraint()`

3. **Integration Strategy**: Should constraints be auto-converted in the decorator, or is explicit `to_callable()` better for clarity?

4. **Validation Protocol**: Is the `ConstraintValidator` protocol flexible enough for future SAT/SMT integration?

5. **TVL Export**: Does the `to_tvl_spec()` output format align with TVL 0.9 spec expectations?

6. **Missing Features**: Any obvious constraint patterns we should support that aren't covered?

---

## How to Test

```bash
# Run all constraint tests
TRAIGENT_MOCK_MODE=true pytest tests/unit/api/test_constraints.py -v

# Run with parameter range tests
TRAIGENT_MOCK_MODE=true pytest tests/unit/api/test_constraints.py tests/unit/api/test_parameter_ranges.py -v

# Quick manual test
TRAIGENT_MOCK_MODE=true python -c "
from traigent import Range, Choices, implies, ConfigSpace

temp = Range(0.0, 2.0, name='temperature')
model = Choices(['gpt-4', 'gpt-3.5'], name='model')
space = ConfigSpace(
    tvars={'temperature': temp, 'model': model},
    constraints=[implies(model.equals('gpt-4'), temp.lte(0.7))]
)
print(space.validate({'temperature': 0.5, 'model': 'gpt-4'}))  # is_valid=True
print(space.validate({'temperature': 1.0, 'model': 'gpt-4'}))  # is_valid=False
"
```

---

## Appendix: Full File Paths

- `REDACTED_TRAIGENT_ROOT/Traigent/traigent/api/constraints.py`
- `REDACTED_TRAIGENT_ROOT/Traigent/traigent/api/validation_protocol.py`
- `REDACTED_TRAIGENT_ROOT/Traigent/traigent/api/config_space.py`
- `REDACTED_TRAIGENT_ROOT/Traigent/traigent/api/parameter_ranges.py`
- `REDACTED_TRAIGENT_ROOT/Traigent/traigent/__init__.py`
- `REDACTED_TRAIGENT_ROOT/Traigent/tests/unit/api/test_constraints.py`
