# Story 2.4: Support Conditional Variables

Status: done

## Story

As an **ML Engineer**,
I want to define conditional relationships between parameters,
so that certain parameters only apply when others have specific values.

## Acceptance Criteria

1. **AC1: Conditional Range Based on Categorical Value**
   - **Given** a conditional like "if model=gpt-4o, then max_tokens range is [100, 8192]"
   - **When** model=gpt-4o is sampled
   - **Then** max_tokens is sampled from [100, 8192]

2. **AC2: Different Ranges for Different Categorical Values**
   - **Given** a conditional like "if model=gpt-4o-mini, then max_tokens range is [100, 4096]"
   - **When** model=gpt-4o-mini is sampled
   - **Then** max_tokens is constrained to [100, 4096]

3. **AC3: Validation of Conditional Configurations**
   - **Given** an invalid conditional configuration
   - **When** validation runs
   - **Then** the system reports the conflict clearly

4. **AC4: Conditional Sampling Interface**
   - **Given** an ExplorationSpace with conditional TVARs
   - **When** I call `space.sample()`
   - **Then** dependent TVARs are sampled using the constraint appropriate for the parent's value

## Tasks / Subtasks

- [x] Task 1: Design ConditionalConstraint data model (AC: #1, #2)
  - [x] 1.1 Define `ConditionalConstraint` dataclass with parent TVAR reference
  - [x] 1.2 Define mapping of parent values to child constraints
  - [x] 1.3 Add type alias: `TVARConstraint = CategoricalConstraint | NumericalConstraint | ConditionalConstraint`
  - [x] 1.4 Add `validate(value, parent_value)` method

- [x] Task 2: Implement `set_conditional()` method on ExplorationSpace (AC: #1, #2)
  - [x] 2.1 Add `set_conditional(child_name, parent_name, conditions: dict)` method
  - [x] 2.2 Validate that both child and parent TVARs exist
  - [x] 2.3 Validate that parent TVAR has discrete choices (categorical or conditional with categorical outputs)
  - [x] 2.4 Store the conditional constraint on the child TVAR

- [x] Task 3: Update sample() to respect conditionals (AC: #4)
  - [x] 3.1 Implement topological ordering of TVARs (parents before children)
  - [x] 3.2 When sampling a conditional TVAR, look up parent's sampled value
  - [x] 3.3 Apply the constraint corresponding to the parent's value
  - [x] 3.4 Handle missing conditions gracefully (use default or error)

- [x] Task 4: Update validate_config() for conditionals (AC: #3)
  - [x] 4.1 Validate child values against parent-specific constraints
  - [x] 4.2 Provide clear error messages including parent-child relationship
  - [x] 4.3 Error format: "Invalid value for {child}: {value} when {parent}={parent_value}"

- [x] Task 5: Add comprehensive tests for conditional variables (AC: #1, #2, #3, #4)
  - [x] 5.1 Test `set_conditional()` creates proper constraint
  - [x] 5.2 Test sampling respects conditional ranges
  - [x] 5.3 Test validation with valid conditional config
  - [x] 5.4 Test validation rejects invalid conditional config
  - [x] 5.5 Test error handling for missing parent values
  - [x] 5.6 Test circular dependency detection

## Dev Notes

### Architecture Context

This is **Story 2.4** in Epic 2 (Configuration Space & TVL). It adds inter-TVAR dependencies (structural constraints C^str) to the ExplorationSpace.

**Current State (from Story 2.1-2.3):**
- `CategoricalConstraint` and `NumericalConstraint` are per-TVAR domain constraints
- `sample()` samples TVARs independently (no ordering)
- `validate_config()` validates each TVAR independently

**Story 2.4 adds:**
- `ConditionalConstraint` for parent-dependent child domains
- Dependency graph for TVAR sampling order
- Parent-aware validation

### TVL Glossary v2.0 Alignment

Conditional constraints implement **Structural Constraints (C^str)** from TVL Glossary v2.0:
- C^str defines valid relationships between TVARs
- ExplorationSpace 𝒳 = Θ ∩ C^str (parameter space with constraints)
- Conditionals are the most common form of C^str

### Proposed Data Model

```python
from dataclasses import dataclass, field
from typing import Any

@dataclass
class ConditionalConstraint:
    """Constraint dependent on another TVAR's value.

    Implements structural constraints (C^str) where the domain of one TVAR
    depends on the value of another TVAR.

    Attributes:
        parent_qualified_name: The TVAR this constraint depends on
        conditions: Mapping of parent values to child constraints
        default_constraint: Fallback constraint if parent value not in conditions

    Example:
        ConditionalConstraint(
            parent_qualified_name="generator.model",
            conditions={
                "gpt-4o": NumericalConstraint(min=100, max=8192),
                "gpt-4o-mini": NumericalConstraint(min=100, max=4096),
            }
        )
    """
    parent_qualified_name: str
    conditions: dict[Any, TVARConstraint]
    default_constraint: TVARConstraint | None = None

    def get_constraint_for(self, parent_value: Any) -> TVARConstraint | None:
        """Get the constraint for a specific parent value."""
        return self.conditions.get(parent_value, self.default_constraint)

    def validate(self, value: Any, parent_value: Any) -> bool:
        """Validate value given the parent's value."""
        constraint = self.get_constraint_for(parent_value)
        if constraint is None:
            return True  # No constraint for this parent value
        return constraint.validate(value)
```

### Sampling with Dependencies

The `sample()` method must be updated to handle dependencies:

```python
def sample(self, seed: int | None = None) -> Configuration:
    """Sample respecting TVAR dependencies."""
    rng = random.Random(seed)
    config: Configuration = {}

    # Get topological order (parents before children)
    ordered_names = self._topological_sort()

    for qualified_name in ordered_names:
        tvar = self.tvars[qualified_name]

        if not tvar.is_tunable or tvar.constraint is None:
            config[qualified_name] = tvar.default_value
        elif isinstance(tvar.constraint, ConditionalConstraint):
            parent_value = config[tvar.constraint.parent_qualified_name]
            effective_constraint = tvar.constraint.get_constraint_for(parent_value)
            config[qualified_name] = self._sample_with_constraint(
                effective_constraint, tvar.python_type, rng
            )
        else:
            # Existing sampling logic for non-conditional
            ...

    return config

def _topological_sort(self) -> list[str]:
    """Return TVAR names in dependency order (parents before children)."""
    # Build dependency graph
    # Detect cycles (raise error)
    # Return topologically sorted list
    ...
```

### API Design

```python
# Define conditional relationship
space.set_conditional(
    child="generator.max_tokens",
    parent="generator.model",
    conditions={
        "gpt-4o": {"min": 100, "max": 8192},
        "gpt-4o-mini": {"min": 100, "max": 4096},
        "gpt-4-turbo": {"min": 100, "max": 4096},
    }
)

# Sampling respects the conditional
config = space.sample()
# If config["generator.model"] == "gpt-4o"
# Then config["generator.max_tokens"] is in [100, 8192]
```

### Testing Strategy

```python
def test_set_conditional_creates_constraint():
    """Test that set_conditional creates a ConditionalConstraint."""
    space = _create_space_with_model_tvar()
    space.set_conditional(
        child="generator.max_tokens",
        parent="generator.model",
        conditions={
            "gpt-4o": {"min": 100, "max": 8192},
            "gpt-4o-mini": {"min": 100, "max": 4096},
        }
    )
    tvar = space.get_tvar("generator", "max_tokens")
    assert isinstance(tvar.constraint, ConditionalConstraint)

def test_sample_respects_conditional_range():
    """Test sampling applies correct range based on parent value."""
    space = _create_space_with_conditional()

    # Sample many times and verify ranges are respected
    for _ in range(100):
        config = space.sample()
        model = config["generator.model"]
        max_tokens = config["generator.max_tokens"]

        if model == "gpt-4o":
            assert 100 <= max_tokens <= 8192
        elif model == "gpt-4o-mini":
            assert 100 <= max_tokens <= 4096

def test_validate_invalid_conditional_config():
    """Test validation rejects invalid conditional config."""
    space = _create_space_with_conditional()

    # gpt-4o-mini has max_tokens limit of 4096
    is_valid, errors = space.validate_config({
        "generator.model": "gpt-4o-mini",
        "generator.max_tokens": 5000  # Invalid! Over limit
    })
    assert not is_valid
    assert "generator.max_tokens" in errors[0]
    assert "gpt-4o-mini" in errors[0]

def test_circular_dependency_raises_error():
    """Test that circular dependencies are detected."""
    space = _create_space_with_two_tvars()
    space.set_conditional(child="a", parent="b", conditions={...})
    with pytest.raises(ValueError, match="circular"):
        space.set_conditional(child="b", parent="a", conditions={...})
```

### File Structure

```
traigent/integrations/haystack/
├── configspace.py       # MODIFY: Add ConditionalConstraint, update sample()/validate()
└── __init__.py          # MODIFY: Export ConditionalConstraint

tests/integrations/
└── test_configspace.py  # MODIFY: Add TestConditionalConstraint tests
```

### Dependencies

- **Story 2.1 (done)**: Provides base ExplorationSpace, TVAR, constraint types
- **Story 2.2 (done)**: Provides set_choices(), sample() foundation
- **Story 2.3 (done)**: Provides set_range(), numerical sampling
- **Story 2.6 (later)**: Will add validation for configuration space consistency

### NFR Considerations

- **NFR-4 (≤10 lines integration)**: Conditionals should be easy to define:
  ```python
  space.set_conditional("generator.max_tokens", "generator.model", {
      "gpt-4o": {"min": 100, "max": 8192},
      "gpt-4o-mini": {"min": 100, "max": 4096},
  })
  ```

### Edge Cases to Handle

1. **Missing parent value in conditions**: Use `default_constraint` if provided, otherwise use `default_value`
   - **Implemented behavior:** When `get_constraint_for(parent_value)` returns `None`, sampling uses `tvar.default_value`
   - **Validation:** `validate_config()` returns `True` for values when no constraint is defined for the parent value
2. **Parent TVAR not found**: Raise KeyError
3. **Child TVAR not found**: Raise KeyError
4. **Parent is not categorical**: Raise ValueError (conditionals require discrete parent)
5. **Circular dependencies**: Detect and raise ValueError (checked BEFORE type validation)
6. **Multi-level conditionals (A→B→C)**: Supported via `_has_discrete_choices()` helper
7. **Parent value is None (optional parameter)**: Include None in conditions dict if needed

### References

- [Source: _bmad-output/epics.md - Epic 2, Story 2.4 - FR-204]
- [Pattern: traigent/integrations/haystack/configspace.py - Existing constraint types]
- [Dependency: Stories 2.1, 2.2, 2.3 - Base constraint and sampling infrastructure]

## Dev Agent Record

### Agent Model Used

Claude Opus 4.5 (claude-opus-4-5-20251101)

### Debug Log References

N/A - No issues encountered

### Completion Notes List

- Implemented `ConditionalConstraint` dataclass with parent reference, conditions mapping, and validation
- Added `set_conditional()` method supporting categorical and multi-level conditional parents
- Implemented topological sort via `_get_dependency_order()` for correct sampling order
- Added circular dependency detection via `_would_create_cycle()`
- Added `_has_discrete_choices()` helper to support multi-level conditionals (A→B→C chains)
- Updated `sample()` to process TVARs in dependency order
- Updated `validate_config()` with parent-aware validation and clear error messages
- Exported `ConditionalConstraint` in `__init__.py`
- All 100 tests pass including 21 new conditional-specific tests

### File List

- `traigent/integrations/haystack/configspace.py` - Added ConditionalConstraint, set_conditional(), dependency ordering
- `traigent/integrations/haystack/__init__.py` - Added ConditionalConstraint export
- `tests/integrations/test_configspace.py` - Added 21 tests for conditional variables

## Change Log

| Date       | Change                                    | Author                                   |
|------------|-------------------------------------------|------------------------------------------|
| 2025-12-20 | Story created for conditional variables  | Claude Opus 4.5 (create-story workflow)  |
| 2025-12-20 | Story implemented with all tests passing | Claude Opus 4.5                          |
