# Story 2.2: Support Categorical Variables

Status: done

## Story

As an **ML Engineer**,
I want to define categorical parameters with specific choices (e.g., model selection),
so that the optimizer samples only from valid options.

## Acceptance Criteria

1. **AC1: Define Categorical Parameter with Choices**
   - **Given** a parameter like `generator.model`
   - **When** I define it as categorical with choices `["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"]`
   - **Then** the ExplorationSpace only samples from those exact values

2. **AC2: Auto-Discovery from Literal Type Hints**
   - **Given** a Literal type hint in the original component (e.g., `Literal["a", "b", "c"]`)
   - **When** auto-discovery runs
   - **Then** the parameter is automatically typed as categorical with the Literal values as choices

3. **AC3: Manual Override of Choices**
   - **Given** an ExplorationSpace with auto-discovered categorical parameter
   - **When** I call `space.set_choices("generator.model", ["gpt-4o", "gpt-4o-mini"])`
   - **Then** the parameter's choices are updated to the new list
   - **And** the original choices are replaced

4. **AC4: Categorical Sampling Interface**
   - **Given** an ExplorationSpace with categorical TVARs
   - **When** I call `space.sample()` or an optimizer samples a configuration
   - **Then** each categorical TVAR value is uniformly sampled from its choices

5. **AC5: Validation of Categorical Values**
   - **Given** a configuration with categorical parameters
   - **When** I call `space.validate_config(config)`
   - **Then** the system returns False if any categorical value is not in its choices
   - **And** the error message identifies which parameter has an invalid value

## Tasks / Subtasks

- [x] Task 1: Implement `set_choices()` method on ExplorationSpace (AC: #1, #3)
  - [x] 1.1 Add `set_choices(qualified_name: str, choices: list[Any])` method
  - [x] 1.2 Validate that the TVAR exists before modifying
  - [x] 1.3 Replace existing CategoricalConstraint or convert NumericalConstraint to Categorical
  - [x] 1.4 Raise KeyError if TVAR not found, ValueError if choices list is empty

- [x] Task 2: Implement `sample()` method for ExplorationSpace (AC: #4)
  - [x] 2.1 Add `sample(seed: int | None = None) -> Configuration` method
  - [x] 2.2 For categorical TVARs: uniform random selection from choices
  - [x] 2.3 For numerical TVARs: uniform random within range (or log-uniform if log_scale)
  - [x] 2.4 Only sample from `tunable_tvars`, use default values for fixed TVARs
  - [x] 2.5 Return a Configuration dict mapping qualified names to sampled values

- [x] Task 3: Enhance validation error messages (AC: #5)
  - [x] 3.1 Update `validate_config()` to provide specific error for invalid categorical values
  - [x] 3.2 Error message format: "Invalid value for {qualified_name}: {value!r} not in choices {choices}"
  - [x] 3.3 Test validation with valid and invalid categorical configurations

- [x] Task 4: Add tests for categorical variable support (AC: #1, #2, #3, #4, #5)
  - [x] 4.1 Test `set_choices()` on existing categorical TVAR
  - [x] 4.2 Test `set_choices()` converting numerical to categorical
  - [x] 4.3 Test `set_choices()` error handling (missing TVAR, empty choices)
  - [x] 4.4 Test `sample()` returns valid categorical values
  - [x] 4.5 Test `sample()` with seed for reproducibility
  - [x] 4.6 Test validation rejects invalid categorical values with clear error

## Dev Notes

### Architecture Context

This is **Story 2.2** in Epic 2 (Configuration Space & TVL). It builds on Story 2.1's foundation (`ExplorationSpace`, `TVAR`, `CategoricalConstraint`) to add sampling and modification capabilities for categorical variables.

**Story 2.1 provided:**
- `CategoricalConstraint` with `choices: list[Any]` and `validate()` method
- `TVAR` with `constraint` field and `validate()` method
- `ExplorationSpace` with `validate_config()` method
- Auto-discovery from `DiscoveredTVAR.literal_choices`

**Story 2.2 adds:**
- `set_choices()` to modify/override categorical constraints
- `sample()` to generate random configurations
- Enhanced validation error messages

### Implementation Pattern (from Story 2.1)

Follow the existing pattern in `configspace.py`:

```python
# In ExplorationSpace class:

def set_choices(self, qualified_name: str, choices: list[Any]) -> None:
    """Set or override choices for a categorical TVAR.

    Args:
        qualified_name: Full path like 'generator.model'
        choices: List of valid values

    Raises:
        KeyError: If TVAR not found
        ValueError: If choices list is empty
    """
    if not choices:
        raise ValueError(f"Choices list cannot be empty for {qualified_name}")

    tvar = self.tvars.get(qualified_name)
    if tvar is None:
        raise KeyError(f"TVAR not found: {qualified_name}")

    # Replace constraint with CategoricalConstraint
    tvar.constraint = CategoricalConstraint(choices=list(choices))

def sample(self, seed: int | None = None) -> Configuration:
    """Sample a random configuration from the exploration space.

    Args:
        seed: Random seed for reproducibility

    Returns:
        Configuration dict mapping qualified names to sampled values
    """
    import random
    if seed is not None:
        random.seed(seed)

    config: Configuration = {}
    for qualified_name, tvar in self.tvars.items():
        if not tvar.is_tunable:
            config[qualified_name] = tvar.default_value
            continue

        if tvar.constraint is None:
            config[qualified_name] = tvar.default_value
        elif isinstance(tvar.constraint, CategoricalConstraint):
            config[qualified_name] = random.choice(tvar.constraint.choices)
        elif isinstance(tvar.constraint, NumericalConstraint):
            # For numerical, sample uniformly (log-uniform if log_scale)
            if tvar.constraint.log_scale:
                import math
                log_min = math.log(tvar.constraint.min)
                log_max = math.log(tvar.constraint.max)
                config[qualified_name] = math.exp(random.uniform(log_min, log_max))
            else:
                config[qualified_name] = random.uniform(
                    tvar.constraint.min, tvar.constraint.max
                )
            # Handle discrete/integer
            if tvar.constraint.step is not None or tvar.python_type == "int":
                config[qualified_name] = int(round(config[qualified_name]))

    return config
```

### Existing Code to Modify

**File: `traigent/integrations/haystack/configspace.py`**
- Add `set_choices()` method to `ExplorationSpace`
- Add `sample()` method to `ExplorationSpace`
- Enhance `validate_config()` error messages for categorical violations

### Testing Strategy

```python
def test_set_choices_on_categorical_tvar():
    """Test modifying choices on existing categorical TVAR."""
    space = ExplorationSpace(tvars={
        "generator.model": TVAR(
            name="model",
            scope="generator",
            python_type="str",
            default_value="gpt-4o",
            constraint=CategoricalConstraint(choices=["gpt-4o", "gpt-4o-mini"])
        )
    })

    space.set_choices("generator.model", ["gpt-4-turbo", "gpt-4o"])
    tvar = space.get_tvar("generator", "model")
    assert tvar.constraint.choices == ["gpt-4-turbo", "gpt-4o"]

def test_sample_returns_valid_categorical():
    """Test that sample() returns values from choices."""
    space = ExplorationSpace(tvars={
        "generator.model": TVAR(
            name="model",
            scope="generator",
            python_type="str",
            default_value="gpt-4o",
            constraint=CategoricalConstraint(choices=["gpt-4o", "gpt-4o-mini"]),
            is_tunable=True
        )
    })

    config = space.sample(seed=42)
    assert config["generator.model"] in ["gpt-4o", "gpt-4o-mini"]

def test_sample_reproducible_with_seed():
    """Test that sample() is reproducible with same seed."""
    space = ExplorationSpace(tvars={...})
    config1 = space.sample(seed=42)
    config2 = space.sample(seed=42)
    assert config1 == config2

def test_validate_config_rejects_invalid_categorical():
    """Test validation error for invalid categorical value."""
    space = ExplorationSpace(tvars={
        "generator.model": TVAR(
            name="model",
            scope="generator",
            python_type="str",
            default_value="gpt-4o",
            constraint=CategoricalConstraint(choices=["gpt-4o", "gpt-4o-mini"])
        )
    })

    is_valid, errors = space.validate_config({"generator.model": "invalid-model"})
    assert not is_valid
    assert "generator.model" in errors[0]
    assert "invalid-model" in errors[0]
```

### File Structure

```
traigent/integrations/haystack/
├── configspace.py       # MODIFY: Add set_choices(), sample() to ExplorationSpace
└── __init__.py          # No changes needed (Configuration already exported)

tests/integrations/
└── test_configspace.py  # MODIFY: Add tests for categorical sampling
```

### Dependencies

- **Story 2.1 (done)**: Provides base ExplorationSpace, TVAR, CategoricalConstraint
- **Story 2.3 (next)**: Will add numerical variable sampling (builds on sample() foundation)

### NFR Considerations

- **NFR-4 (≤10 lines integration)**: Sampling should be simple:
  ```python
  space = from_pipeline(pipeline, as_exploration_space=True)
  config = space.sample()  # Get random config
  ```

### TVL Glossary Alignment (v2.0)

- `CategoricalConstraint.choices` = Domain (Dᵢ) for discrete TVAR
- `sample()` generates Configuration (θ) by selecting from each Dᵢ
- `set_choices()` modifies the Domain for a TVAR

### References

- [Source: _bmad-output/epics.md - Epic 2, Story 2.2 - FR-202]
- [Pattern: traigent/integrations/haystack/configspace.py - Existing ExplorationSpace]
- [Dependency: Story 2.1 - CategoricalConstraint, TVAR types]

## Dev Agent Record

### Agent Model Used

Claude Opus 4.5 (claude-opus-4-5-20251101)

### Debug Log References

- All 187 integration tests pass (71 configspace + 116 introspection)
- `make format` and `make lint` pass with no errors

### Completion Notes List

1. **Task 1 Complete**: Added `set_choices()` method to ExplorationSpace with proper error handling
2. **Task 2 Complete**: Added `sample()` method with support for categorical (uniform) and numerical (uniform/log-uniform) sampling
3. **Task 3 Complete**: Enhanced `validate_config()` with specific error messages for categorical vs numerical violations
4. **Task 4 Complete**: Added 24 new tests covering all acceptance criteria

### Implementation Notes

- Added bonus `set_range()` method as foundation for Story 2.3 (same pattern as `set_choices()`)
- `sample()` handles integer types correctly by rounding to int
- `sample()` respects `is_tunable=False` by using default values for fixed TVARs
- `sample()` supports log-uniform sampling when `log_scale=True`
- Enhanced validation provides context-specific error messages (choices list for categorical, range for numerical)

### Code Review Fixes (Post-Implementation)

**Round 1:**
- **Fixed global RNG mutation**: Changed `sample()` to use `random.Random(seed)` local instance instead of `random.seed()`
- **Fixed log_scale edge case**: Added fallback to linear sampling when `log_scale=True` but `min <= 0`
- **Clarified set_choices() behavior**: Updated docstring to document that order is preserved, duplicates bias sampling, and None must be explicitly included for optional parameters
- **Fixed terminology drift**: Changed "ConfigSpace" to "ExplorationSpace" in AC1 to align with TVL Glossary v2.0

**Round 2:**
- **Fixed step alignment in NumericalConstraint.validate()**: Now enforces that values are on the step grid (with floating point tolerance)
- **Fixed discrete sampling in sample()**: Now uses proper uniform grid sampling via `randint(0, num_steps)` instead of `round()` which had midpoint bias
- **Added step validation in set_range()**: Raises `ValueError` if `step <= 0`
- **Refactored sample()**: Extracted `_sample_numerical()` helper to reduce cognitive complexity

### File List

**Modified Files:**

- `traigent/integrations/haystack/configspace.py` - Added set_choices(), set_range(), sample(), _sample_numerical() methods; enhanced validate_config() and NumericalConstraint.validate()
- `tests/integrations/test_configspace.py` - Added 32 new tests (TestSetChoices, TestSetRange, TestSample, TestValidateConfigCategorical, step alignment tests)

## Change Log

| Date       | Change                                                                    | Author                                  |
|------------|---------------------------------------------------------------------------|----------------------------------------|
| 2025-12-20 | Story created for categorical variable support                            | Claude Opus 4.5 (create-story workflow) |
| 2025-12-20 | Implementation complete - all tasks done, 24 new tests pass               | Claude Opus 4.5 (dev-story workflow)    |
| 2025-12-20 | Code review fixes: local RNG, log_scale fallback, terminology, docstrings | Claude Opus 4.5 (code-review workflow)  |
| 2025-12-20 | Code review fixes: step alignment, discrete sampling, step validation     | Claude Opus 4.5 (code-review workflow)  |
