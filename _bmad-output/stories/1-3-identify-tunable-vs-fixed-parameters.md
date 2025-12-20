# Story 1.3: Identify Tunable vs Fixed Parameters

Status: done

## Story

As an **ML Engineer**,
I want the system to automatically identify which parameters are tunable vs fixed,
so that I don't waste time trying to optimize non-configurable settings.

## Acceptance Criteria

1. **AC1: Tunable Parameter Detection**
   - **Given** a component with various parameter types
   - **When** I introspect the pipeline
   - **Then** parameters with type hints (int, float, str, Literal) are marked as tunable
   - **And** each tunable parameter has `is_tunable=True`

2. **AC2: Fixed Parameter Detection**
   - **Given** a component with complex parameter types
   - **When** I introspect the pipeline
   - **Then** parameters that are callables, objects, or document_store are marked as fixed
   - **And** each fixed parameter has `is_tunable=False` with a reason

3. **AC3: Default Ranges for Tunable Parameters**
   - **Given** a tunable numeric parameter (int, float)
   - **When** I introspect the pipeline
   - **Then** the system provides sensible default ranges based on parameter semantics
   - **And** ranges are stored in a new `default_range` field on Parameter

4. **AC4: Categorical Parameter Detection**
   - **Given** a parameter with `Literal["gpt-4o", "gpt-4o-mini"]` type hint
   - **When** I introspect the pipeline
   - **Then** the parameter is marked as categorical with those specific choices
   - **And** `literal_choices` contains the exact allowed values

## Tasks / Subtasks

- [x] Task 1: Add default range field to Parameter dataclass (AC: #3)
  - [x] 1.1 Add `default_range: tuple[Any, Any] | None` field to Parameter in models.py
  - [x] 1.2 Add `range_type: str | None` field for range semantics ("continuous", "discrete", "log")
  - [x] 1.3 Update Parameter `__repr__` to display range when present

- [x] Task 2: Implement parameter semantics detection (AC: #3)
  - [x] 2.1 Create `_infer_parameter_semantics(param_name: str, python_type: str) -> dict` in introspection.py
  - [x] 2.2 Define common parameter patterns and their default ranges:
    - `temperature`: (0.0, 2.0) continuous
    - `top_p`: (0.0, 1.0) continuous
    - `top_k`: (1, 100) discrete
    - `max_tokens`: (1, 4096) discrete
    - `presence_penalty`: (-2.0, 2.0) continuous
    - `frequency_penalty`: (-2.0, 2.0) continuous
  - [x] 2.3 Return `None` for parameters without known semantics

- [x] Task 3: Integrate range inference into parameter extraction (AC: #3)
  - [x] 3.1 Call `_infer_parameter_semantics()` in `_extract_parameters()`
  - [x] 3.2 Populate `default_range` and `range_type` fields on Parameter
  - [x] 3.3 Only set ranges for tunable numeric parameters (int, float)

- [x] Task 4: Enhance tunability validation tests (AC: #1, #2)
  - [x] 4.1 Add test verifying int/float/str/bool/Literal parameters are tunable
  - [x] 4.2 Add test verifying callable parameters are not tunable
  - [x] 4.3 Add test verifying document_store parameters are not tunable
  - [x] 4.4 Add test verifying object-type parameters are not tunable

- [x] Task 5: Add default range tests (AC: #3)
  - [x] 5.1 Test temperature parameter gets (0.0, 2.0) range
  - [x] 5.2 Test top_k parameter gets (1, 100) discrete range
  - [x] 5.3 Test unknown parameter name gets None range
  - [x] 5.4 Test range is only set for tunable numeric parameters

- [x] Task 6: Add categorical parameter tests (AC: #4)
  - [x] 6.1 Test Literal type parameter has is_tunable=True
  - [x] 6.2 Test Literal choices are correctly extracted
  - [x] 6.3 Test model parameter with Literal["gpt-4o", "gpt-4o-mini"] has both choices

## Dev Notes

### Architecture Context

This is **Story 1.3** in the Haystack Integration epic. It builds on Stories 1.1 and 1.2:
- Story 1.1: Established component extraction with `from_pipeline()`
- Story 1.2: Added parameter extraction with type detection and basic tunability

**Key Insight: Most Tunability Logic Already Exists**

Story 1.2 already implemented:
- `TUNABLE_TYPES = {"int", "float", "str", "bool", "Literal"}`
- `NON_TUNABLE_PARAM_NAMES` for document_store, callbacks, etc.
- `_is_tunable_parameter()` function
- `Parameter` dataclass with `is_tunable`, `literal_choices`, `non_tunable_reason`

**What's NEW in Story 1.3:**
- **Default ranges for tunable parameters** - the primary new functionality
- Parameter semantics inference based on common naming patterns
- Additional validation tests to ensure coverage

### Previous Story Learnings (from Story 1.2)

**Implementation Patterns:**
- Used `typing.get_type_hints()` for PEP 563 compatibility
- Refactored extraction logic into small helper functions to reduce complexity
- 24 tests pass for Stories 1.1 + 1.2

**Existing Tunability Detection (introspection.py:382-414):**
```python
def _is_tunable_parameter(
    param_name: str, python_type: str, value: Any
) -> tuple[bool, str | None]:
    """Determine if a parameter is tunable."""
    # Check if parameter name indicates non-tunable
    param_name_lower = param_name.lower()
    for non_tunable_name in NON_TUNABLE_PARAM_NAMES:
        if non_tunable_name in param_name_lower:
            return (False, f"Parameter '{param_name}' is a complex object")

    # Check if it's a callable
    if callable(value) and not isinstance(value, type):
        return (False, "Parameter is a callable")

    # Check if the type is tunable
    if python_type in TUNABLE_TYPES:
        return (True, None)

    # Unknown or object types are not tunable by default
    if python_type in ("unknown", "object", "Union"):
        return (False, f"Type '{python_type}' is not automatically tunable")

    return (False, f"Type '{python_type}' requires manual specification")
```

### Default Range Semantics

**Common LLM Parameter Ranges:**

| Parameter | Type | Range | Scale |
|-----------|------|-------|-------|
| temperature | float | (0.0, 2.0) | continuous |
| top_p | float | (0.0, 1.0) | continuous |
| top_k | int | (1, 100) | discrete |
| max_tokens | int | (1, 4096) | discrete |
| presence_penalty | float | (-2.0, 2.0) | continuous |
| frequency_penalty | float | (-2.0, 2.0) | continuous |

**Retriever Parameter Ranges:**

| Parameter | Type | Range | Scale |
|-----------|------|-------|-------|
| top_k | int | (1, 100) | discrete |
| score_threshold | float | (0.0, 1.0) | continuous |
| similarity_threshold | float | (0.0, 1.0) | continuous |

**Implementation Pattern:**
```python
# Suggested structure for parameter semantics
PARAMETER_SEMANTICS = {
    "temperature": {"range": (0.0, 2.0), "scale": "continuous"},
    "top_p": {"range": (0.0, 1.0), "scale": "continuous"},
    "top_k": {"range": (1, 100), "scale": "discrete"},
    "max_tokens": {"range": (1, 4096), "scale": "discrete"},
    "presence_penalty": {"range": (-2.0, 2.0), "scale": "continuous"},
    "frequency_penalty": {"range": (-2.0, 2.0), "scale": "continuous"},
    "score_threshold": {"range": (0.0, 1.0), "scale": "continuous"},
    "similarity_threshold": {"range": (0.0, 1.0), "scale": "continuous"},
}

def _infer_parameter_semantics(param_name: str, python_type: str) -> dict | None:
    """Infer default range and scale for a parameter based on its name."""
    if python_type not in ("int", "float"):
        return None

    param_lower = param_name.lower()
    for pattern, semantics in PARAMETER_SEMANTICS.items():
        if pattern in param_lower:
            return semantics
    return None
```

### File Modifications Required

**models.py - Extend Parameter dataclass:**
```python
@dataclass
class Parameter:
    # Existing fields...
    name: str
    value: Any
    python_type: str
    type_hint: str | None = None
    is_tunable: bool = True
    literal_choices: list[Any] | None = None
    is_optional: bool = False
    non_tunable_reason: str | None = None
    # NEW fields for Story 1.3
    default_range: tuple[Any, Any] | None = None
    range_type: str | None = None  # "continuous", "discrete", or "log"
```

**introspection.py - Add semantics inference:**
```python
# Add after TUNABLE_TYPES
PARAMETER_SEMANTICS = {
    "temperature": {"range": (0.0, 2.0), "scale": "continuous"},
    # ... other patterns
}

def _infer_parameter_semantics(param_name: str, python_type: str) -> dict | None:
    """Infer default range for common parameter patterns."""
    ...

# Update _extract_parameters() to call _infer_parameter_semantics()
```

### Testing Strategy

**Mock Components for Range Testing:**
```python
class MockGeneratorWithRanges:
    """Mock generator with parameters that should get default ranges."""
    __module__ = "haystack.components.generators"

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.7,
        max_tokens: int = 100,
        top_p: float = 0.9,
        custom_param: float = 0.5,  # No known semantics
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.custom_param = custom_param
```

**Test Assertions:**
```python
def test_temperature_gets_default_range():
    config_space = from_pipeline(pipeline_with_generator)
    temp_param = config_space.get_parameter("generator", "temperature")
    assert temp_param.default_range == (0.0, 2.0)
    assert temp_param.range_type == "continuous"

def test_unknown_param_no_range():
    config_space = from_pipeline(pipeline_with_generator)
    custom_param = config_space.get_parameter("generator", "custom_param")
    assert custom_param.default_range is None
    assert custom_param.range_type is None
```

### References

- [Source: docs/PRD_Agentic_Workflow_Tuning_Haystack.docx - FR-104]
- [Source: _bmad-output/epics.md - Epic 1, Story 1.3]
- [Pattern: Story 1.2 implementation - traigent/integrations/haystack/introspection.py]
- [OpenAI API docs - parameter ranges reference]
- [Haystack component parameter patterns]

## Dev Agent Record

### Agent Model Used

Claude Opus 4.5 (claude-opus-4-5-20251101)

### Debug Log References

- No issues encountered during implementation
- All 49 tests pass (28 from Stories 1.1/1.2 + 21 new tests for Story 1.3)

### Completion Notes List

- Added `default_range` and `range_type` fields to Parameter dataclass
- Implemented `PARAMETER_SEMANTICS` dictionary with 8 common parameter patterns
- Created `_infer_parameter_semantics()` function for range inference
- Integrated range inference into `_extract_parameters()`
- Added TestTunabilityDetection class with 8 tests for AC1/AC2
- Added TestDefaultRanges class with 12 tests for AC3 (including all 8 PARAMETER_SEMANTICS patterns)
- Extended TestParameter with range display test
- All code passes `make format` and `make lint`

### File List

**Modified Files:**

- traigent/integrations/haystack/models.py - Added default_range, range_type fields
- traigent/integrations/haystack/introspection.py - Added PARAMETER_SEMANTICS and _infer_parameter_semantics()
- traigent/integrations/haystack/\_\_init\_\_.py - Exported PARAMETER_SEMANTICS for public API
- tests/integrations/test_haystack_introspection.py - Added 21 new tests (8 tunability + 12 ranges + 1 repr)

## Change Log

| Date | Change | Author |
|------|--------|--------|
| 2025-12-20 | Story created from Epic 1.3 with focus on default ranges | Claude Opus 4.5 (create-story workflow) |
| 2025-12-20 | Implementation complete - all tasks done, ready for review | Claude Opus 4.5 (dev-story workflow) |
| 2025-12-20 | Code review fixes: Added 4 missing range tests, exported PARAMETER_SEMANTICS, fixed test counts | Claude Opus 4.5 (code-review workflow) |
