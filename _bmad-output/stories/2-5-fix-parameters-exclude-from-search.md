# Story 2.5: Fix Parameters (Exclude from Search)

Status: done

## Story

As an **ML Engineer**,
I want to explicitly fix certain parameters so they're excluded from optimization,
so that I can lock down values I don't want changed.

## Acceptance Criteria

1. **AC1: Fix Parameter to Specific Value**
   - **Given** an ExplorationSpace with parameter `generator.model`
   - **When** I call `space.fix("generator.model", "gpt-4o")`
   - **Then** that parameter is removed from the search space (is_tunable=False)
   - **And** all sampled configs use the fixed value "gpt-4o"

2. **AC2: Unfix Parameter to Restore to Search**
   - **Given** a fixed parameter
   - **When** I call `space.unfix("generator.model")`
   - **Then** the parameter is restored to the search space (is_tunable=True)
   - **And** sampling uses the original constraint

3. **AC3: Fix Preserves Constraint for Later Unfix**
   - **Given** a parameter with `CategoricalConstraint(choices=["gpt-4o", "gpt-4o-mini"])`
   - **When** I call `fix("generator.model", "gpt-4o")` then later `unfix("generator.model")`
   - **Then** the original constraint is preserved and restored

4. **AC4: Fix with Value Not in Constraint**
   - **Given** a categorical parameter with choices `["a", "b", "c"]`
   - **When** I call `fix("param", "invalid_value")`
   - **Then** the system raises ValueError (fixed value must be valid)

## Tasks / Subtasks

- [x] Task 1: Implement `fix()` method on ExplorationSpace (AC: #1, #3, #4)
  - [x] 1.1 Add `fix(qualified_name: str, value: Any) -> None` method
  - [x] 1.2 Validate that the TVAR exists (raise KeyError if not)
  - [x] 1.3 Validate that the value is valid per existing constraint (raise ValueError if not)
  - [x] 1.4 Store the original constraint for later restoration
  - [x] 1.5 Set `is_tunable=False` and update `default_value` to the fixed value

- [x] Task 2: Implement `unfix()` method on ExplorationSpace (AC: #2, #3)
  - [x] 2.1 Add `unfix(qualified_name: str) -> None` method
  - [x] 2.2 Validate that the TVAR exists (raise KeyError if not)
  - [x] 2.3 Restore the original constraint from storage
  - [x] 2.4 Set `is_tunable=True`
  - [x] 2.5 Optionally restore original default_value

- [x] Task 3: Update sample() behavior for fixed TVARs (AC: #1)
  - [x] 3.1 Verify that sample() uses default_value for is_tunable=False (already implemented)
  - [x] 3.2 Add tests to confirm fixed TVARs are not sampled

- [x] Task 4: Add comprehensive tests (AC: #1, #2, #3, #4)
  - [x] 4.1 Test `fix()` sets is_tunable=False
  - [x] 4.2 Test `fix()` updates default_value to fixed value
  - [x] 4.3 Test `sample()` returns fixed value for fixed TVARs
  - [x] 4.4 Test `unfix()` restores is_tunable=True
  - [x] 4.5 Test `unfix()` restores original constraint
  - [x] 4.6 Test `fix()` raises ValueError for invalid value
  - [x] 4.7 Test `fix()` raises KeyError for non-existent TVAR
  - [x] 4.8 Test `unfix()` on never-fixed TVAR (should be no-op or error)

## Dev Notes

### Architecture Context

This is **Story 2.5** in Epic 2 (Configuration Space & TVL). It provides explicit control over which TVARs are included in optimization.

**Current State (from Story 2.1-2.3):**
- TVARs have `is_tunable` flag (auto-discovered based on type)
- `sample()` uses `default_value` for non-tunable TVARs
- `tunable_tvars` and `fixed_tvars` properties exist

**Story 2.5 adds:**
- `fix()` method to explicitly exclude a TVAR from search
- `unfix()` method to restore a TVAR to the search space
- Constraint preservation for later unfix

### Implementation Pattern

```python
@dataclass
class TVAR:
    # Existing fields...
    _original_constraint: TVARConstraint | None = field(default=None, repr=False)
    _original_default: Any = field(default=None, repr=False)

class ExplorationSpace:
    def fix(self, qualified_name: str, value: Any) -> None:
        """Fix a TVAR to a specific value, excluding it from optimization.

        The fixed value is used in all sampled configurations. The original
        constraint is preserved so the TVAR can be unfixed later.

        Args:
            qualified_name: Full path like 'generator.model'
            value: The value to fix the parameter to

        Raises:
            KeyError: If the TVAR is not found
            ValueError: If the value is not valid per the TVAR's constraint
        """
        tvar = self.tvars.get(qualified_name)
        if tvar is None:
            raise KeyError(f"TVAR not found: {qualified_name}")

        # Validate value against constraint
        if tvar.constraint is not None and not tvar.constraint.validate(value):
            if isinstance(tvar.constraint, CategoricalConstraint):
                raise ValueError(
                    f"Cannot fix {qualified_name} to {value!r}: "
                    f"not in choices {tvar.constraint.choices}"
                )
            elif isinstance(tvar.constraint, NumericalConstraint):
                raise ValueError(
                    f"Cannot fix {qualified_name} to {value!r}: "
                    f"not in range [{tvar.constraint.min}, {tvar.constraint.max}]"
                )

        # Store original state for unfix
        if tvar._original_constraint is None:  # Only store if not already fixed
            tvar._original_constraint = tvar.constraint
            tvar._original_default = tvar.default_value

        # Fix the TVAR
        tvar.is_tunable = False
        tvar.default_value = value

    def unfix(self, qualified_name: str) -> None:
        """Restore a fixed TVAR to the optimization search space.

        Args:
            qualified_name: Full path like 'generator.model'

        Raises:
            KeyError: If the TVAR is not found
        """
        tvar = self.tvars.get(qualified_name)
        if tvar is None:
            raise KeyError(f"TVAR not found: {qualified_name}")

        # Restore original state
        if tvar._original_constraint is not None:
            tvar.constraint = tvar._original_constraint
            tvar._original_constraint = None
        if tvar._original_default is not None:
            tvar.default_value = tvar._original_default
            tvar._original_default = None

        tvar.is_tunable = True
```

### Testing Strategy

```python
def test_fix_sets_is_tunable_false():
    """Test that fix() sets is_tunable=False."""
    space = ExplorationSpace(tvars={
        "generator.model": TVAR(
            name="model", scope="generator",
            python_type="str", default_value="gpt-4o",
            constraint=CategoricalConstraint(choices=["gpt-4o", "gpt-4o-mini"]),
            is_tunable=True
        )
    })
    assert space.get_tvar("generator", "model").is_tunable is True
    space.fix("generator.model", "gpt-4o")
    assert space.get_tvar("generator", "model").is_tunable is False

def test_fix_updates_default_value():
    """Test that fix() updates the default value."""
    space = _create_space_with_model_tvar()
    space.fix("generator.model", "gpt-4o-mini")
    assert space.get_tvar("generator", "model").default_value == "gpt-4o-mini"

def test_sample_returns_fixed_value():
    """Test that sample() returns the fixed value for fixed TVARs."""
    space = _create_space_with_model_tvar()
    space.fix("generator.model", "gpt-4o")

    # Sample multiple times - should always get fixed value
    for _ in range(10):
        config = space.sample()
        assert config["generator.model"] == "gpt-4o"

def test_unfix_restores_tunable():
    """Test that unfix() restores is_tunable=True."""
    space = _create_space_with_model_tvar()
    space.fix("generator.model", "gpt-4o")
    space.unfix("generator.model")
    assert space.get_tvar("generator", "model").is_tunable is True

def test_unfix_restores_constraint():
    """Test that unfix() restores the original constraint."""
    space = _create_space_with_model_tvar()
    original_choices = ["gpt-4o", "gpt-4o-mini"]

    space.fix("generator.model", "gpt-4o")
    space.unfix("generator.model")

    tvar = space.get_tvar("generator", "model")
    assert isinstance(tvar.constraint, CategoricalConstraint)
    assert tvar.constraint.choices == original_choices

def test_fix_raises_valueerror_for_invalid_value():
    """Test that fix() raises ValueError for invalid value."""
    space = _create_space_with_model_tvar()
    with pytest.raises(ValueError, match="not in choices"):
        space.fix("generator.model", "invalid-model")
```

### File Structure

```
traigent/integrations/haystack/
├── configspace.py       # MODIFY: Add fix(), unfix() methods, add _original_* fields to TVAR
└── __init__.py          # No changes needed

tests/integrations/
└── test_configspace.py  # MODIFY: Add TestFix class with tests
```

### Dependencies

- **Story 2.1 (done)**: Provides TVAR with is_tunable flag
- **Story 2.2 (done)**: Provides sample() that respects is_tunable
- **Story 2.6 (later)**: Will validate configuration space consistency

### NFR Considerations

- **NFR-4 (≤10 lines integration)**: Fixing parameters should be simple:
  ```python
  space = from_pipeline(pipeline, as_exploration_space=True)
  space.fix("generator.model", "gpt-4o")  # Lock to specific model
  config = space.sample()  # model is always "gpt-4o"
  ```

### Edge Cases to Handle

1. **Fix a TVAR that's already fixed**: Should update to new value (not error)
2. **Unfix a TVAR that was never fixed**: Should be no-op (already tunable)
3. **Fix with None for optional parameter**: Should work if constraint allows None
4. **Fix then modify constraint then unfix**: Should restore original constraint
5. **Multiple fix/unfix cycles**: Should work correctly

### References

- [Source: _bmad-output/epics.md - Epic 2, Story 2.5 - FR-205]
- [Pattern: traigent/integrations/haystack/configspace.py - Existing TVAR, is_tunable]
- [Dependency: Stories 2.1-2.3 - Base TVAR infrastructure]

## Dev Agent Record

### Agent Model Used

Claude Opus 4.5 (claude-opus-4-5-20251101)

### Debug Log References

N/A - No issues encountered

### Completion Notes List

- Added `_original_constraint`, `_original_default`, and `_is_fixed` private fields to TVAR dataclass
- Implemented `fix()` method with value validation against constraints
- Implemented `unfix()` method to restore original state
- Added `fixed_tvars` property to get list of fixed TVAR names
- Added 15 comprehensive tests across 3 test classes (TestFix, TestUnfix, TestFixUnfixIntegration)
- All 115 configspace tests pass

### File List

- `traigent/integrations/haystack/configspace.py` - Added TVAR fields and fix/unfix methods
- `tests/integrations/test_configspace.py` - Added 15 tests for fix/unfix functionality

## Change Log

| Date       | Change                                    | Author                                   |
|------------|-------------------------------------------|------------------------------------------|
| 2025-12-20 | Story created for fix/unfix parameters   | Claude Opus 4.5 (create-story workflow)  |
| 2025-12-20 | Story implemented with all tests passing | Claude Opus 4.5                          |
