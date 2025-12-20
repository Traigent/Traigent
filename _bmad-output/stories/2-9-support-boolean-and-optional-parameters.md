# Story 2.9: Support Boolean and Optional Parameters

Status: done

## Story

As an **ML Engineer**,
I want boolean and Optional parameters to be handled correctly during auto-discovery,
so that all common parameter types are supported without manual intervention.

## Acceptance Criteria

1. **AC1: Boolean Parameters as Categorical**
   - **Given** a component with a `bool` parameter (e.g., `scale_score: bool`)
   - **When** auto-discovery runs
   - **Then** the parameter is typed as categorical with choices `[True, False]`

2. **AC2: Optional Parameters Include None**
   - **Given** a component with an `Optional[T]` parameter (e.g., `Optional[int]`)
   - **When** auto-discovery runs
   - **Then** the parameter includes `None` as a valid choice alongside the T-type range

3. **AC3: Optional String with Literal Choices**
   - **Given** a parameter with `Optional[str]` and no default
   - **When** auto-discovery runs
   - **Then** the parameter is typed as categorical with `[None]` plus any Literal choices if present

4. **AC4: Optional with Numerical Range**
   - **Given** a parameter with `Optional[float]` and a known range
   - **When** auto-discovery runs
   - **Then** the ExplorationSpace allows sampling from range OR None

## Tasks / Subtasks

- [x] Task 1: Verify boolean handling in ExplorationSpace conversion (AC: #1)
  - [x] 1.1 Confirm `from_pipeline_spec()` converts bool TVARs to CategoricalConstraint([True, False])
  - [x] 1.2 Add tests for boolean parameter discovery and sampling
  - [x] 1.3 Ensure sample() returns actual bool values (not strings)

- [x] Task 2: Enhance Optional parameter handling (AC: #2, #3, #4)
  - [x] 2.1 Update `_convert_to_constraint()` to handle Optional types
  - [x] 2.2 For Optional[Categorical]: add None to choices (already implemented in 2.1)
  - [x] 2.3 For Optional[Numerical]: added `is_optional` field to TVAR and `OPTIONAL_NONE_PROBABILITY` class constant
  - [x] 2.4 For Optional without type info: create CategoricalConstraint([None, current_value]) (already implemented)

- [x] Task 3: Handle sampling with Optional constraints (AC: #2, #4)
  - [x] 3.1 Update sample() to handle None as valid choice via `_sample_optional_numerical()`
  - [x] 3.2 Define probability of sampling None vs. actual value: 10% via `OPTIONAL_NONE_PROBABILITY`
  - [x] 3.3 Refactored sample() into helper methods to reduce cognitive complexity

- [x] Task 4: Add comprehensive tests (AC: #1, #2, #3, #4)
  - [x] 4.1 Test bool parameter sampled as actual bool values
  - [x] 4.2 Test Optional[bool] includes None in choices
  - [x] 4.3 Test Optional[int] can sample None or integer
  - [x] 4.4 Test validation accepts None for optional parameters
  - [x] 4.5 Test validation rejects None for non-optional parameters
  - [x] 4.6 Test OPTIONAL_NONE_PROBABILITY affects sampling frequency

## Dev Notes

### Architecture Context

This is **Story 2.9** in Epic 2 (Configuration Space & TVL). It completes type support for auto-discovery.

**Current State (from Story 2.1):**
- Boolean types are converted to `CategoricalConstraint([True, False])` ✓
- Optional types have `is_optional=True` and add None to choices ✓

**Story 2.9 focuses on:**
- Verifying existing boolean handling works correctly
- Ensuring Optional parameters are handled consistently
- Adding tests for edge cases

### Existing Implementation Check

From `configspace.py` line 382-394:
```python
# Handle boolean types as categorical
if discovered.python_type == "bool":
    choices: list[Any] = [True, False]
    if discovered.is_optional:
        choices.append(None)
    return CategoricalConstraint(choices=choices)

# Handle Literal types with choices
if discovered.literal_choices is not None:
    choices = list(discovered.literal_choices)
    if discovered.is_optional and None not in choices:
        choices.append(None)
    return CategoricalConstraint(choices=choices)
```

**This means AC1-AC3 may already be partially implemented!**

Let's verify and add any missing pieces:

### Optional Numerical Handling

> **⚠️ DESIGN DECISION REQUIRED:** The current implementation doesn't fully handle
> `Optional[float]` with ranges. The TVAR dataclass needs an `is_optional` flag
> to track this at sampling time.

**Option A: Treat None as categorical choice (RECOMMENDED)**

```python
# In TVAR dataclass, add:
is_optional: bool = False  # Track if None is valid

# Optional[float] with range becomes:
# Sample either None OR a value from the range
# Probability: 10% for None, 90% for range sampling (configurable)
```

**Option B: Add NoneOr constraint type**

```python
@dataclass
class NoneOrConstraint:
    """Constraint that allows None or values from inner constraint."""
    inner: TVARConstraint
    none_probability: float = 0.1  # 10% chance of None
```

**Implementation Decision: Option A (simpler)**

- Add `is_optional: bool = False` field to TVAR dataclass
- During `from_pipeline_spec()`, set `is_optional=True` when `DiscoveredTVAR.is_optional=True`
- When sampling Optional[Numerical], first flip a coin for None (10% default)
- If not None, sample from numerical range
- Default None probability could be configurable via class constant

### Implementation Pattern

```python
# In sample() method:
def sample(self, seed: int | None = None) -> Configuration:
    # ... existing code ...
    for qualified_name, tvar in self.tvars.items():
        if not tvar.is_tunable or tvar.constraint is None:
            config[qualified_name] = tvar.default_value
        elif isinstance(tvar.constraint, CategoricalConstraint):
            config[qualified_name] = rng.choice(tvar.constraint.choices)
        elif isinstance(tvar.constraint, NumericalConstraint):
            # Check if this is an optional numerical
            if tvar.is_optional:  # Need to add this flag or check
                # 10% chance of None for optional parameters
                if rng.random() < 0.1:
                    config[qualified_name] = None
                    continue
            config[qualified_name] = self._sample_numerical(...)
```

### Testing Strategy

```python
def test_bool_parameter_becomes_categorical():
    """Test bool parameter is converted to CategoricalConstraint([True, False])."""
    spec = PipelineSpec(scopes=[
        TVARScope(
            name="ranker",
            class_name="CohereRanker",
            class_type="haystack.components.rankers.CohereRanker",
            tvars={
                "scale_score": DiscoveredTVAR(
                    name="scale_score",
                    value=True,
                    python_type="bool",
                    is_tunable=True
                )
            }
        )
    ])

    space = ExplorationSpace.from_pipeline_spec(spec)
    tvar = space.get_tvar("ranker", "scale_score")

    assert isinstance(tvar.constraint, CategoricalConstraint)
    assert tvar.constraint.choices == [True, False]

def test_optional_str_includes_none():
    """Test Optional[str] parameter includes None in choices."""
    spec = _create_spec_with_optional_str()
    space = ExplorationSpace.from_pipeline_spec(spec)
    tvar = space.get_tvar("component", "optional_param")

    assert None in tvar.constraint.choices

def test_sample_bool_returns_bool():
    """Test that sampling boolean returns actual bool, not string."""
    space = _create_space_with_bool_tvar()
    config = space.sample()
    assert isinstance(config["ranker.scale_score"], bool)

def test_sample_optional_can_return_none():
    """Test that sampling optional parameter can return None."""
    space = _create_space_with_optional_tvar()
    # Sample many times
    found_none = False
    for _ in range(100):
        config = space.sample()
        if config["component.optional_param"] is None:
            found_none = True
            break
    assert found_none, "None should be sampled for optional parameters"

def test_validate_accepts_none_for_optional():
    """Test that validation accepts None for optional parameters."""
    space = _create_space_with_optional_tvar()
    is_valid, errors = space.validate_config({
        "component.optional_param": None
    })
    assert is_valid
```

### File Structure

```
traigent/integrations/haystack/
├── configspace.py       # MODIFY: Enhance Optional handling if needed
└── __init__.py          # No changes needed

tests/integrations/
└── test_configspace.py  # MODIFY: Add TestBooleanAndOptional class
```

### Dependencies

- **Story 2.1 (done)**: Provides base conversion logic
- **Story 2.2 (done)**: Provides sample() and set_choices()

### NFR Considerations

- **NFR-5 (≥90% component coverage)**: Boolean and Optional support is critical for high coverage

### Edge Cases to Handle

1. **Optional[bool]**: Should have choices `[True, False, None]`
2. **Optional[Literal["a", "b"]]**: Should have choices `["a", "b", None]`
3. **Optional without default**: Should default to None
4. **Deeply nested Optional**: `Optional[Optional[T]]` - flatten to just include None once

### References

- [Source: _bmad-output/epics.md - Epic 2, Story 2.9 - FR-202, FR-203]
- [Pattern: traigent/integrations/haystack/configspace.py - Existing conversion]
- [Dependency: Stories 2.1-2.3]

## Dev Agent Record

### Agent Model Used

Claude Opus 4.5 (claude-opus-4-5-20251101)

### Debug Log References

N/A - No issues encountered

### Completion Notes List

- Verified existing boolean handling converts to `CategoricalConstraint([True, False])` correctly
- Added `is_optional: bool = False` field to TVAR dataclass for tracking Optional[Numerical] types
- Added `OPTIONAL_NONE_PROBABILITY: ClassVar[float] = 0.1` class constant to ExplorationSpace
- Updated `from_pipeline_spec()` to set `is_optional=True` for Optional numerical TVARs
- Refactored `sample()` method by extracting `_sample_tvar()`, `_sample_conditional_tvar()`, `_sample_optional_numerical()` helpers to reduce cognitive complexity
- Updated `validate_config()` to accept None for optional TVARs
- Fixed `NumericalConstraint.validate()` to handle None values (returns False instead of TypeError)
- Added 13 comprehensive tests in `TestBooleanAndOptionalParameters` class
- All 177 configspace tests pass

**Code Review Fixes (2025-12-20):**
- Fixed `fix(None)` to work for optional numerical TVARs (was failing due to NumericalConstraint.validate())
- Fixed optional numericals under ConditionalConstraint to sample None (updated `_sample_with_constraint()`)
- Fixed `from_pipeline_spec()` to preserve `is_optional=True` for optional numericals without inferred ranges
- Updated docstring for `from_pipeline_spec()` to accurately describe optional numerical handling
- Added 4 additional tests for fix(None), conditional optional, and is_optional preservation

### File List

- `traigent/integrations/haystack/configspace.py` - Added is_optional field, OPTIONAL_NONE_PROBABILITY, refactored sample(), fixed optional handling
- `tests/integrations/test_configspace.py` - Added TestBooleanAndOptionalParameters with 13 tests

## Change Log

| Date       | Change                                       | Author                                   |
|------------|----------------------------------------------|------------------------------------------|
| 2025-12-20 | Story created for bool/optional support      | Claude Opus 4.5 (create-story workflow)  |
| 2025-12-20 | Story implemented with all tests passing     | Claude Opus 4.5                          |
| 2025-12-20 | Fixed code review issues for optional handling | Claude Opus 4.5                          |
