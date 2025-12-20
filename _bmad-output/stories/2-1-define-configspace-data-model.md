# Story 2.1: Define ConfigSpace Data Model

Status: done

## Story

As an **ML Engineer**,
I want a ConfigSpace object that holds all tunable parameters with their types and ranges,
so that I have a structured representation of the optimization search space.

## Acceptance Criteria

1. **AC1: ConfigSpace Parameter Structure**
   - **Given** a ConfigSpace object
   - **When** I inspect its parameters
   - **Then** each parameter has: name, component, type, default value, and constraints (choices/range)

2. **AC2: ConfigSpace from Pipeline Introspection**
   - **Given** an introspected pipeline
   - **When** I call `from_pipeline(pipeline)`
   - **Then** the return value is a valid ConfigSpace object with all discovered parameters
   - **And** the ConfigSpace can be used for optimization search space definition

3. **AC3: Parameter Type Representation**
   - **Given** a parameter with a specific Python type
   - **When** represented in ConfigSpace
   - **Then** categorical types have `choices` constraint (list of valid values)
   - **And** numerical types have `range` constraint (min, max) with optional `log_scale` flag
   - **And** boolean types are represented as categorical with `[True, False]` choices

4. **AC4: Tunable vs Fixed Parameters**
   - **Given** a ConfigSpace with multiple parameters
   - **When** I query tunable parameters
   - **Then** only parameters marked as `is_tunable=True` are included in the search space
   - **And** fixed parameters are preserved but excluded from optimization sampling

5. **AC5: Scoped Parameter Access**
   - **Given** a ConfigSpace with parameters from multiple components
   - **When** I access parameters by scope (e.g., "generator.temperature")
   - **Then** parameters are uniquely addressable via `scope_name.parameter_name` path

## Tasks / Subtasks

- [x] Task 1: Design the TVAR (Tuned Variable) data model for optimization (AC: #1, #3)
  - [x] 1.1 Create `traigent/integrations/haystack/configspace.py` module
  - [x] 1.2 Define `TVAR` dataclass with: name, scope, python_type, default_value, constraints
  - [x] 1.3 Define `TVARConstraint` union type: `CategoricalConstraint | NumericalConstraint`
  - [x] 1.4 Implement `CategoricalConstraint` with `choices: list[Any]`
  - [x] 1.5 Implement `NumericalConstraint` with `min`, `max`, `log_scale`, `step` (for discrete)

- [x] Task 2: Define ExplorationSpace class for optimization (AC: #1, #2, #4, #5)
  - [x] 2.1 Create `ExplorationSpace` dataclass (separate from discovery `PipelineSpec`)
  - [x] 2.2 Implement `tvars: dict[str, TVAR]` mapping qualified names to TVARs
  - [x] 2.3 Implement `get_tvar(scope_name, tvar_name)` for scoped access
  - [x] 2.4 Implement `tunable_tvars` property returning only is_tunable=True TVARs
  - [x] 2.5 Implement `fixed_tvars` property returning only is_tunable=False TVARs

- [x] Task 3: Create ExplorationSpace factory from PipelineSpec (AC: #2, #3)
  - [x] 3.1 Implement `ExplorationSpace.from_pipeline_spec(spec: PipelineSpec)` factory method
  - [x] 3.2 Convert `DiscoveredTVAR` objects to optimization `TVAR` objects
  - [x] 3.3 Map `literal_choices` to `CategoricalConstraint`
  - [x] 3.4 Map `default_range` to `NumericalConstraint`
  - [x] 3.5 Handle boolean types as categorical with `[True, False]` choices

- [x] Task 4: Integrate with from_pipeline() (AC: #2)
  - [x] 4.1 Update `from_pipeline()` to optionally return ExplorationSpace
  - [x] 4.2 Add `as_exploration_space: bool = False` parameter to `from_pipeline()`
  - [x] 4.3 When `as_exploration_space=True`, return `ExplorationSpace.from_pipeline_spec(spec)`

- [x] Task 5: Add comprehensive tests (AC: #1, #2, #3, #4, #5)
  - [x] 5.1 Create `tests/integrations/test_configspace.py`
  - [x] 5.2 Test TVAR creation with categorical constraints
  - [x] 5.3 Test TVAR creation with numerical constraints
  - [x] 5.4 Test ExplorationSpace creation from PipelineSpec
  - [x] 5.5 Test scoped parameter access
  - [x] 5.6 Test tunable vs fixed parameter filtering

## Dev Notes

### Architecture Context

This is **Story 2.1** in the Configuration Space & TVL epic. It establishes the data model that will be used for all optimization operations in subsequent stories.

**Key Design Decisions:**

1. **Separation of Discovery vs Optimization Types:**
   - `PipelineSpec` + `DiscoveredTVAR` = Discovery output (Epic 1)
   - `ConfigSpace` + `TVAR` = Optimization input (Epic 2+)
   - This separation allows:
     - Discovery to capture raw pipeline state without optimization constraints
     - Optimization to add ranges, conditionals, and sampling logic
     - Users to modify ConfigSpace after discovery but before optimization

2. **Qualified Parameter Names:**
   - Parameters are addressed as `"scope_name.tvar_name"` (e.g., `"generator.temperature"`)
   - This prevents naming conflicts between components
   - Aligns with TVL syntax in Story 2.7

3. **Constraint Types:**
   - `CategoricalConstraint`: For Literal types, enums, booleans, model selection
   - `NumericalConstraint`: For int/float with min/max bounds, optional log scale
   - Conditionals (Story 2.4) will be added later as a separate constraint type

### Existing Type Mapping (from Epic 1)

The following DiscoveredTVAR fields map to TVAR constraints:

| DiscoveredTVAR Field | TVAR Constraint |
|---------------------|-----------------|
| `literal_choices` | `CategoricalConstraint(choices=...)` |
| `default_range` | `NumericalConstraint(min=..., max=...)` |
| `range_type="log"` | `NumericalConstraint(log_scale=True)` |
| `range_type="discrete"` | `NumericalConstraint(step=1)` |
| `python_type="bool"` | `CategoricalConstraint(choices=[True, False])` |
| `is_optional=True` | Include `None` in choices |

### Type Definitions

```python
from dataclasses import dataclass, field
from typing import Any, Literal

@dataclass
class CategoricalConstraint:
    """Constraint for categorical/discrete choice parameters."""
    choices: list[Any]

    def validate(self, value: Any) -> bool:
        return value in self.choices

@dataclass
class NumericalConstraint:
    """Constraint for numerical parameters with range bounds."""
    min: float | int
    max: float | int
    log_scale: bool = False
    step: float | int | None = None  # For discrete numerical

    def validate(self, value: float | int) -> bool:
        return self.min <= value <= self.max

TVARConstraint = CategoricalConstraint | NumericalConstraint

@dataclass
class TVAR:
    """A Tuned Variable for optimization search space."""
    name: str  # Parameter name (e.g., "temperature")
    scope: str  # Component scope (e.g., "generator")
    python_type: str  # "int", "float", "str", "bool", "Literal"
    default_value: Any
    constraint: TVARConstraint | None = None
    is_tunable: bool = True

    @property
    def qualified_name(self) -> str:
        """Full path like 'generator.temperature'."""
        return f"{self.scope}.{self.name}"

@dataclass
class ConfigSpace:
    """Optimization search space with tunable parameters."""
    tvars: dict[str, TVAR] = field(default_factory=dict)

    def get_tvar(self, scope: str, name: str) -> TVAR | None:
        """Get TVAR by scope and name."""
        return self.tvars.get(f"{scope}.{name}")

    @property
    def tunable_tvars(self) -> dict[str, TVAR]:
        """Return only tunable TVARs."""
        return {k: v for k, v in self.tvars.items() if v.is_tunable}

    @property
    def fixed_tvars(self) -> dict[str, TVAR]:
        """Return only fixed TVARs."""
        return {k: v for k, v in self.tvars.items() if not v.is_tunable}

    @classmethod
    def from_pipeline_spec(cls, spec: "PipelineSpec") -> "ConfigSpace":
        """Create ConfigSpace from discovered PipelineSpec."""
        # Implementation in Task 3
        ...
```

### Relationship to Epic 1 Types

```
Epic 1 (Discovery)              Epic 2 (Optimization)
─────────────────               ────────────────────
PipelineSpec                    ConfigSpace
  ├─ TVARScope[]      ─────►      ├─ TVAR{}
  │    └─ DiscoveredTVAR{}        │    └─ TVARConstraint
  ├─ Connection[]                 └─ (metadata)
  └─ loops[]
```

- `PipelineSpec` captures the raw pipeline structure (scopes, connections, loops)
- `ConfigSpace` captures only the optimization-relevant TVARs with their constraints
- `from_pipeline()` returns `PipelineSpec` by default (backwards compatible)
- `from_pipeline(..., as_config_space=True)` returns `ConfigSpace` for optimization

### Testing Strategy

```python
def test_tvar_categorical_constraint():
    """Test TVAR with categorical choices."""
    constraint = CategoricalConstraint(choices=["gpt-4o", "gpt-4o-mini"])
    tvar = TVAR(
        name="model",
        scope="generator",
        python_type="str",
        default_value="gpt-4o",
        constraint=constraint
    )
    assert tvar.qualified_name == "generator.model"
    assert constraint.validate("gpt-4o") is True
    assert constraint.validate("invalid") is False

def test_tvar_numerical_constraint():
    """Test TVAR with numerical range."""
    constraint = NumericalConstraint(min=0.0, max=2.0)
    tvar = TVAR(
        name="temperature",
        scope="generator",
        python_type="float",
        default_value=0.7,
        constraint=constraint
    )
    assert constraint.validate(1.0) is True
    assert constraint.validate(3.0) is False

def test_configspace_from_pipeline_spec():
    """Test ConfigSpace creation from PipelineSpec."""
    # Create PipelineSpec with DiscoveredTVARs
    spec = PipelineSpec(scopes=[
        TVARScope(
            name="generator",
            class_name="OpenAIGenerator",
            class_type="haystack.components.generators.OpenAIGenerator",
            category="Generator",
            tvars={
                "temperature": DiscoveredTVAR(
                    name="temperature",
                    value=0.7,
                    python_type="float",
                    default_range=(0.0, 2.0),
                    is_tunable=True
                )
            }
        )
    ])

    config_space = ConfigSpace.from_pipeline_spec(spec)
    tvar = config_space.get_tvar("generator", "temperature")
    assert tvar is not None
    assert isinstance(tvar.constraint, NumericalConstraint)
```

### File Structure

```
traigent/integrations/haystack/
├── __init__.py          # Add ConfigSpace export
├── models.py            # Existing: PipelineSpec, DiscoveredTVAR, TVARScope
├── configspace.py       # NEW: TVAR, TVARConstraint, ConfigSpace
└── introspection.py     # Update: from_pipeline() with as_config_space param
```

### NFR Considerations

- **NFR-4 (≤10 lines integration):** ConfigSpace creation should be a single call:
  ```python
  config_space = from_pipeline(pipeline, as_config_space=True)
  ```
- Lightweight data structures - no heavy computation during ConfigSpace creation

### References

- [Source: docs/PRD_Agentic_Workflow_Tuning_Haystack.docx - FR-201]
- [Source: _bmad-output/epics.md - Epic 2, Story 2.1]
- [Pattern: traigent/integrations/haystack/models.py - Existing data models]
- [Dependency: Story 1.1-1.7 - PipelineSpec and DiscoveredTVAR types]

## Dev Agent Record

### Agent Model Used

Claude Opus 4.5 (claude-opus-4-5-20251101)

### Debug Log References

- All 163 integration tests pass (116 existing + 47 new for Story 2.1)
- `make format` and `make lint` pass with no errors

### Completion Notes List

1. **Task 1 Complete**: Created `configspace.py` with TVAR, CategoricalConstraint, NumericalConstraint dataclasses
2. **Task 2 Complete**: Implemented `ExplorationSpace` class with tunable/fixed filtering and scoped access
3. **Task 3 Complete**: Implemented `ExplorationSpace.from_pipeline_spec()` factory with proper type mapping
4. **Task 4 Complete**: Updated `from_pipeline()` with `as_exploration_space` parameter and type overloads
5. **Task 5 Complete**: Added 47 comprehensive tests covering all acceptance criteria

### Implementation Notes

- Named the optimization class `ExplorationSpace` instead of `ConfigSpace` to avoid conflict with the backwards-compatible alias `ConfigSpace` (which maps to `PipelineSpec`)
- Added `validate_config()` method to `ExplorationSpace` for configuration validation
- Included `validate()` methods on both constraint types and TVAR for runtime validation
- Added proper type overloads for `from_pipeline()` to provide correct type inference

### File List

**New Files Created:**

- `traigent/integrations/haystack/configspace.py` - TVAR, TVARConstraint, ExplorationSpace types
- `tests/integrations/test_configspace.py` - 47 test cases covering all ACs

**Modified Files:**

- `traigent/integrations/haystack/__init__.py` - Added exports for new types
- `traigent/integrations/haystack/introspection.py` - Added `as_exploration_space` parameter to `from_pipeline()`

## Change Log

| Date | Change | Author |
|------|--------|--------|
| 2025-12-20 | Story created for ConfigSpace data model | Claude Opus 4.5 (create-story workflow) |
| 2025-12-20 | Implementation complete - all tasks done, 47 new tests pass | Claude Opus 4.5 (dev-story workflow) |
