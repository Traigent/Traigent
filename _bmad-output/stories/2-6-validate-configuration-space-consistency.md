# Story 2.6: Validate Configuration Space Consistency

Status: done

## Story

As an **ML Engineer**,
I want the system to validate my ExplorationSpace for consistency,
so that I catch configuration errors before running expensive experiments.

## Acceptance Criteria

1. **AC1: Valid Configuration Space Returns True**
   - **Given** an ExplorationSpace with valid parameters and conditionals
   - **When** I call `space.validate()`
   - **Then** the system returns True

2. **AC2: Conflicting Conditionals Raise ConfigurationSpaceError**
   - **Given** an ExplorationSpace with conflicting conditionals
   - **When** I call `space.validate()`
   - **Then** the system raises `ConfigurationSpaceError` with clear description of the conflict
   - **Note:** "Conflicting conditionals" means: (a) parent TVAR doesn't exist, (b) parent
     has condition values not in its choices, (c) circular dependencies exist

3. **AC3: Invalid Range Raises ConfigurationSpaceError**
   - **Given** an ExplorationSpace with a parameter range where min > max
   - **When** I call `space.validate()`
   - **Then** the system raises `ConfigurationSpaceError` identifying the invalid range

4. **AC4: Zero Tunable Parameters Raises ValueError**
   - **Given** an ExplorationSpace with zero tunable parameters
   - **When** I call `space.validate()`
   - **Then** the system raises ValueError indicating no tunable components were found

## Tasks / Subtasks

- [x] Task 1: Implement `validate()` method on ExplorationSpace (AC: #1, #2, #3, #4)
  - [x] 1.1 Add `validate() -> bool` method that raises on error
  - [x] 1.2 Check that at least one TVAR has `is_tunable=True`
  - [x] 1.3 Check numerical constraints have valid ranges (min < max)
  - [x] 1.4 Check categorical constraints have non-empty choices
  - [x] 1.5 Check conditional constraints reference valid parent TVARs

- [x] Task 2: Validate conditional consistency (AC: #2)
  - [x] 2.1 Check that parent TVAR exists for each conditional
  - [x] 2.2 Check that parent has categorical constraint (conditionals require discrete parent)
  - [x] 2.3 Check that all parent values in conditions are valid choices
  - [x] 2.4 Detect circular conditional dependencies

- [x] Task 3: Use existing ConfigurationSpaceError exception (AC: #2, #3)
  - [x] 3.1 Import `ConfigurationSpaceError` from `traigent.utils.exceptions`
  - [x] 3.2 Use it for all validation errors (already has structured error support)
  - [x] 3.3 **DO NOT** create a new ValidationError - use existing infrastructure

- [x] Task 4: Add comprehensive tests (AC: #1, #2, #3, #4)
  - [x] 4.1 Test valid space returns True
  - [x] 4.2 Test invalid range raises ConfigurationSpaceError
  - [x] 4.3 Test empty choices raises ConfigurationSpaceError
  - [x] 4.4 Test zero tunable TVARs raises ValueError
  - [x] 4.5 Test invalid conditional parent raises ConfigurationSpaceError
  - [x] 4.6 Test circular dependency raises ConfigurationSpaceError
  - [x] 4.7 Test condition values not in parent choices raises ConfigurationSpaceError

## Dev Notes

### Architecture Context

This is **Story 2.6** in Epic 2 (Configuration Space & TVL). It provides pre-optimization validation to catch configuration errors early.

**Current State (from Story 2.1-2.5):**
- `validate_config(config)` validates a specific configuration against constraints
- No validation of the ExplorationSpace itself

**Story 2.6 adds:**
- `validate()` method to check ExplorationSpace consistency
- `ValidationError` exception for configuration errors
- Pre-optimization checks for constraint validity

### Implementation Pattern

```python
# Use the existing exception - DO NOT create a new one
from traigent.utils.exceptions import ConfigurationSpaceError


class ExplorationSpace:
    def validate(self) -> bool:
        """Validate the exploration space for consistency.

        Checks:
        - At least one TVAR is tunable
        - Numerical constraints have valid ranges (min < max)
        - Categorical constraints have non-empty choices
        - Conditional constraints reference valid parent TVARs
        - No circular conditional dependencies

        Returns:
            True if validation passes

        Raises:
            ValueError: If no tunable TVARs exist
            ConfigurationSpaceError: If any constraint is invalid
        """
        # Check at least one tunable TVAR
        if len(self.tunable_tvars) == 0:
            raise ValueError("No tunable parameters found in ExplorationSpace")

        # Validate each TVAR's constraint
        for qualified_name, tvar in self.tvars.items():
            self._validate_tvar_constraint(qualified_name, tvar)

        # Validate conditional dependencies
        self._validate_conditionals()

        return True

    def _validate_tvar_constraint(self, name: str, tvar: TVAR) -> None:
        """Validate a single TVAR's constraint."""
        if tvar.constraint is None:
            return

        if isinstance(tvar.constraint, NumericalConstraint):
            if tvar.constraint.min >= tvar.constraint.max:
                raise ConfigurationSpaceError(
                    f"Invalid range for {name}: min ({tvar.constraint.min}) "
                    f">= max ({tvar.constraint.max})"
                )
            if tvar.constraint.step is not None and tvar.constraint.step <= 0:
                raise ConfigurationSpaceError(
                    f"Invalid step for {name}: step must be positive"
                )

        elif isinstance(tvar.constraint, CategoricalConstraint):
            if len(tvar.constraint.choices) == 0:
                raise ConfigurationSpaceError(
                    f"Empty choices for {name}"
                )

    def _validate_conditionals(self) -> None:
        """Validate conditional constraint dependencies."""
        # Build dependency graph
        deps: dict[str, str] = {}  # child -> parent

        for name, tvar in self.tvars.items():
            if isinstance(tvar.constraint, ConditionalConstraint):
                parent_name = tvar.constraint.parent_qualified_name

                # Check parent exists
                if parent_name not in self.tvars:
                    raise ConfigurationSpaceError(
                        f"Conditional {name} references non-existent parent {parent_name}"
                    )

                # Check parent has discrete choices (categorical or conditional with categorical)
                parent = self.tvars[parent_name]
                if not self._has_discrete_choices(parent):
                    raise ConfigurationSpaceError(
                        f"Conditional {name} requires parent with discrete choices, "
                        f"but {parent_name} has {type(parent.constraint).__name__}"
                    )

                # Check all condition values are valid parent choices
                parent_choices = self._get_all_parent_choices(parent)
                for parent_value in tvar.constraint.conditions.keys():
                    if parent_value not in parent_choices:
                        raise ConfigurationSpaceError(
                            f"Conditional {name} references invalid parent value "
                            f"{parent_value!r} (not in {parent_name} choices)"
                        )

                deps[name] = parent_name

        # Check for cycles
        self._check_circular_dependencies(deps)

    def _check_circular_dependencies(self, deps: dict[str, str]) -> None:
        """Detect circular dependencies in conditional graph."""
        for start in deps:
            visited = set()
            current = start
            while current in deps:
                if current in visited:
                    raise ConfigurationSpaceError(
                        f"Circular dependency detected involving {current}"
                    )
                visited.add(current)
                current = deps[current]
```

### Testing Strategy

```python
def test_validate_valid_space_returns_true():
    """Test that validate() returns True for valid space."""
    space = _create_valid_space()
    assert space.validate() is True

def test_validate_no_tunable_raises_valueerror():
    """Test that validate() raises ValueError when no tunable TVARs."""
    space = ExplorationSpace(tvars={
        "generator.model": TVAR(
            name="model", scope="generator",
            python_type="str", default_value="gpt-4o",
            is_tunable=False  # Not tunable
        )
    })
    with pytest.raises(ValueError, match="No tunable parameters"):
        space.validate()

def test_validate_invalid_range_raises_validationerror():
    """Test that validate() raises ValidationError for min >= max."""
    space = ExplorationSpace(tvars={
        "generator.temperature": TVAR(
            name="temperature", scope="generator",
            python_type="float", default_value=0.7,
            constraint=NumericalConstraint(min=2.0, max=1.0),  # Invalid!
            is_tunable=True
        )
    })
    with pytest.raises(ValidationError, match="Invalid range"):
        space.validate()

def test_validate_empty_choices_raises_validationerror():
    """Test that validate() raises ValidationError for empty choices."""
    space = ExplorationSpace(tvars={
        "generator.model": TVAR(
            name="model", scope="generator",
            python_type="str", default_value="gpt-4o",
            constraint=CategoricalConstraint(choices=[]),  # Invalid!
            is_tunable=True
        )
    })
    with pytest.raises(ValidationError, match="Empty choices"):
        space.validate()

def test_validate_circular_dependency_raises_validationerror():
    """Test that validate() raises ValidationError for circular deps."""
    # Setup circular conditional: A depends on B, B depends on A
    # ...
    with pytest.raises(ValidationError, match="Circular dependency"):
        space.validate()
```

### File Structure

```
traigent/integrations/haystack/
├── configspace.py       # MODIFY: Add validate(), ValidationError
└── __init__.py          # MODIFY: Export ValidationError

tests/integrations/
└── test_configspace.py  # MODIFY: Add TestValidate class
```

### Dependencies

- **Story 2.1 (done)**: Provides ExplorationSpace, TVAR, constraint types
- **Story 2.4 (ready)**: Provides ConditionalConstraint (validate depends on this)

### NFR Considerations

- **NFR-7 (reduce time-to-optimize)**: Validation catches errors early:
  ```python
  space = from_pipeline(pipeline, as_exploration_space=True)
  space.set_range("generator.temperature", 2.0, 0.5)  # Mistake: min > max
  space.validate()  # Catches error before expensive optimization
  ```

### References

- [Source: _bmad-output/epics.md - Epic 2, Story 2.6 - FR-206]
- [Pattern: traigent/integrations/haystack/configspace.py - Existing validation]
- [Dependency: Stories 2.1-2.5]

## Dev Agent Record

### Agent Model Used

Claude Opus 4.5 (claude-opus-4-5-20251101)

### Debug Log References

N/A - No issues encountered

### Completion Notes List

- Implemented `validate()` method on ExplorationSpace that validates before optimization
- Validates: at least one tunable TVAR, valid numerical ranges (min < max), non-empty categorical choices
- Validates conditional constraints: parent exists, parent has discrete choices, all condition values valid
- Detects circular dependencies in conditional graph
- Uses existing `ConfigurationSpaceError` from `traigent.utils.exceptions`
- Refactored helper methods to reduce cognitive complexity (extracted `_validate_numerical_constraint`, `_validate_categorical_constraint`, etc.)
- Added 22 comprehensive tests across 7 test classes
- All 137 configspace tests pass

### File List

- `traigent/integrations/haystack/configspace.py` - Added validate() and helper methods
- `tests/integrations/test_configspace.py` - Added 22 tests for validate() functionality

## Change Log

| Date       | Change                                       | Author                                   |
|------------|----------------------------------------------|------------------------------------------|
| 2025-12-20 | Story created for space validation           | Claude Opus 4.5 (create-story workflow)  |
| 2025-12-20 | Story implemented with all tests passing     | Claude Opus 4.5                          |
