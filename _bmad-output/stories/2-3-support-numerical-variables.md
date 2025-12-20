# Story 2.3: Support Numerical Variables

Status: done

## Story

As an **ML Engineer**,
I want to define numerical parameters with ranges (continuous floats, discrete integers),
so that the optimizer searches within meaningful bounds.

## Acceptance Criteria

1. **AC1: Define Float Parameter with Range**
   - **Given** a parameter like `generator.temperature`
   - **When** I define it as float with range [0.0, 2.0]
   - **Then** the ExplorationSpace samples continuous values within that range

2. **AC2: Define Integer Parameter with Range**
   - **Given** a parameter like `retriever.top_k`
   - **When** I define it as integer with range [1, 50]
   - **Then** the ExplorationSpace samples only integer values within that range

3. **AC3: Log-Scale Sampling**
   - **Given** a parameter with `log_scale=True`
   - **When** sampling occurs
   - **Then** values are sampled uniformly in log space (useful for learning rates, etc.)

## Tasks / Subtasks

- [x] Task 1: Implement numerical range support in ExplorationSpace (AC: #1, #2)
  - [x] 1.1 Add `set_range(qualified_name, min, max, log_scale, step)` method
  - [x] 1.2 Validate that min < max and step > 0 if provided
  - [x] 1.3 Support continuous float ranges via `random.uniform()`
  - [x] 1.4 Support discrete integer ranges via `random.randint()`

- [x] Task 2: Implement log-scale sampling (AC: #3)
  - [x] 2.1 Add log-uniform sampling when `log_scale=True` and `min > 0`
  - [x] 2.2 Fallback to linear sampling when `min <= 0` (log undefined)
  - [x] 2.3 Document the fallback behavior in docstrings

- [x] Task 3: Implement discrete numerical sampling with step (AC: #2)
  - [x] 3.1 Add grid-based sampling for numerical ranges with `step` parameter
  - [x] 3.2 Use `randint(0, num_steps)` for uniform grid selection (avoiding midpoint bias)
  - [x] 3.3 Preserve integer types when both min and step are integers

- [x] Task 4: Add validation for numerical constraints (AC: #1, #2, #3)
  - [x] 4.1 Implement step alignment validation in `NumericalConstraint.validate()`
  - [x] 4.2 Use floating point tolerance for step grid checking
  - [x] 4.3 Provide clear error messages for out-of-range values

- [x] Task 5: Add comprehensive tests for numerical variable support
  - [x] 5.1 Test `set_range()` with float ranges
  - [x] 5.2 Test `set_range()` with integer ranges
  - [x] 5.3 Test `set_range()` with log_scale
  - [x] 5.4 Test `set_range()` with step parameter
  - [x] 5.5 Test sampling returns values within range
  - [x] 5.6 Test sampling respects step grid alignment
  - [x] 5.7 Test log_scale fallback when min <= 0
  - [x] 5.8 Test validation rejects values outside range

## Dev Notes

### Architecture Context

This is **Story 2.3** in Epic 2 (Configuration Space & TVL). It completes the numerical variable support that was partially implemented in Story 2.2.

**Note:** Most of Story 2.3 functionality was actually implemented during Story 2.2 as a natural extension of the categorical variable work. The `set_range()` method, log-scale sampling, step-based discrete sampling, and associated tests were all added in Story 2.2.

**Story 2.2 already provided:**
- `set_range()` method with min, max, log_scale, step parameters
- `_sample_numerical()` helper for sampling numerical TVARs
- Log-uniform sampling when `log_scale=True` and `min > 0`
- Linear fallback when `min <= 0`
- Discrete grid sampling using `randint(0, num_steps)`
- Step alignment validation in `NumericalConstraint.validate()`
- Comprehensive test coverage for all numerical scenarios

### Implementation Pattern

The implementation follows the existing pattern in `configspace.py`:

```python
def set_range(
    self,
    qualified_name: str,
    min_val: float | int,
    max_val: float | int,
    log_scale: bool = False,
    step: float | int | None = None,
) -> None:
    """Set or override range for a numerical TVAR."""
    if min_val >= max_val:
        raise ValueError(f"min_val ({min_val}) must be less than max_val ({max_val})")
    if step is not None and step <= 0:
        raise ValueError(f"step must be positive, got {step}")

    tvar = self.tvars.get(qualified_name)
    if tvar is None:
        raise KeyError(f"TVAR not found: {qualified_name}")

    tvar.constraint = NumericalConstraint(
        min=min_val, max=max_val, log_scale=log_scale, step=step
    )

def _sample_numerical(
    self,
    constraint: NumericalConstraint,
    python_type: str,
    rng: random.Random,
) -> float | int:
    """Sample from numerical constraint with proper handling."""
    # Discrete sampling: uniform over grid points
    if constraint.step is not None and constraint.step > 0:
        num_steps = int((constraint.max - constraint.min) / constraint.step)
        step_idx = rng.randint(0, num_steps)
        value = constraint.min + step_idx * constraint.step
        if isinstance(constraint.step, int) and isinstance(constraint.min, int):
            return int(value)
        return value

    # Integer type without step: use randint
    if python_type == "int":
        return rng.randint(int(constraint.min), int(constraint.max))

    # Log-uniform sampling (only valid for positive ranges)
    if constraint.log_scale and constraint.min > 0:
        log_min = math.log(constraint.min)
        log_max = math.log(constraint.max)
        return math.exp(rng.uniform(log_min, log_max))

    # Uniform sampling (fallback for log_scale with min <= 0)
    return rng.uniform(constraint.min, constraint.max)
```

### Testing Strategy

Tests verify all acceptance criteria:

```python
def test_sample_numerical_returns_value_in_range():
    """Test that sample() returns values within numerical range."""
    space = ExplorationSpace(tvars={
        "generator.temperature": TVAR(
            name="temperature", scope="generator",
            python_type="float", default_value=0.7,
            constraint=NumericalConstraint(min=0.0, max=2.0),
            is_tunable=True
        )
    })
    for _ in range(100):
        config = space.sample()
        assert 0.0 <= config["generator.temperature"] <= 2.0

def test_sample_integer_returns_int():
    """Test that integer TVARs return integer values."""
    space = ExplorationSpace(tvars={
        "retriever.top_k": TVAR(
            name="top_k", scope="retriever",
            python_type="int", default_value=10,
            constraint=NumericalConstraint(min=1, max=50),
            is_tunable=True
        )
    })
    config = space.sample()
    assert isinstance(config["retriever.top_k"], int)

def test_sample_log_scale():
    """Test log-uniform sampling."""
    space = ExplorationSpace(tvars={
        "optimizer.lr": TVAR(
            name="lr", scope="optimizer",
            python_type="float", default_value=0.01,
            constraint=NumericalConstraint(min=0.0001, max=0.1, log_scale=True),
            is_tunable=True
        )
    })
    config = space.sample()
    assert 0.0001 <= config["optimizer.lr"] <= 0.1
```

### File Structure

```
traigent/integrations/haystack/
├── configspace.py       # Contains set_range(), _sample_numerical() (no changes needed)
└── __init__.py          # No changes needed

tests/integrations/
└── test_configspace.py  # Contains 79 tests including numerical tests (no changes needed)
```

### Dependencies

- **Story 2.1 (done)**: Provides base ExplorationSpace, TVAR, NumericalConstraint
- **Story 2.2 (done)**: Provided set_range(), sample(), _sample_numerical()
- **Story 2.4 (next)**: Will add conditional variables

### NFR Considerations

- **NFR-4 (≤10 lines integration)**: Setting ranges should be simple:
  ```python
  space = from_pipeline(pipeline, as_exploration_space=True)
  space.set_range("generator.temperature", 0.0, 1.5)
  space.set_range("optimizer.lr", 0.0001, 0.1, log_scale=True)
  config = space.sample()  # Get random config within ranges
  ```

### TVL Glossary Alignment (v2.0)

- `NumericalConstraint(min, max)` = Domain (Dᵢ) for continuous TVAR
- `NumericalConstraint(step=N)` = Domain (Dᵢ) for discrete TVAR on grid
- `log_scale=True` = Sampling uniformly in log space
- `sample()` generates Configuration (θ) by selecting from each Dᵢ

### References

- [Source: _bmad-output/epics.md - Epic 2, Story 2.3 - FR-203]
- [Pattern: traigent/integrations/haystack/configspace.py - Existing ExplorationSpace]
- [Dependency: Story 2.2 - set_range(), sample(), _sample_numerical() methods]

## Dev Agent Record

### Agent Model Used

Claude Opus 4.5 (claude-opus-4-5-20251101)

### Debug Log References

- All 79 integration tests pass (47 from Story 2.1 + 32 from Story 2.2)
- `make format` and `make lint` pass with no errors

### Completion Notes List

1. **Already Complete**: Story 2.3 functionality was implemented during Story 2.2
2. **Task 1 Complete**: `set_range()` method exists with full validation
3. **Task 2 Complete**: Log-scale sampling with fallback implemented in `_sample_numerical()`
4. **Task 3 Complete**: Discrete grid sampling using proper `randint(0, num_steps)` approach
5. **Task 4 Complete**: `NumericalConstraint.validate()` enforces step alignment with floating point tolerance
6. **Task 5 Complete**: Tests exist in TestSetRange and TestSample classes

### Implementation Notes

This story was retroactively marked as done because all acceptance criteria are satisfied by the existing implementation from Story 2.2. The numerical variable support was a natural extension of the categorical variable work.

Key features already working:
- Continuous float sampling: `rng.uniform(min, max)`
- Discrete integer sampling: `rng.randint(min, max)`
- Step-based grid sampling: `randint(0, num_steps)` then `min + step_idx * step`
- Log-uniform sampling: `exp(uniform(log(min), log(max)))`
- Fallback to linear when `min <= 0`
- Step alignment validation with floating point tolerance

### File List

**No files modified** - Story 2.3 is satisfied by existing implementation from Story 2.2.

**Files containing the implementation:**

- `traigent/integrations/haystack/configspace.py` - Contains set_range(), _sample_numerical(), NumericalConstraint
- `tests/integrations/test_configspace.py` - Contains TestSetRange, TestSample test classes

## Change Log

| Date       | Change                                                      | Author                                   |
|------------|-------------------------------------------------------------|------------------------------------------|
| 2025-12-20 | Story created - marked as done (already implemented in 2.2) | Claude Opus 4.5 (create-story workflow) |
