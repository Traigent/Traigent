# Story 1.2: Extract Component Parameters

Status: done

## Story

As an **ML Engineer**,
I want to see all init parameters for each component in my pipeline,
so that I know what values can potentially be tuned.

## Acceptance Criteria

1. **AC1: Parameter Extraction from Generator Components**
   - **Given** a Pipeline with an OpenAIGenerator component
   - **When** I introspect the pipeline
   - **Then** the system extracts parameters: model, temperature, max_tokens, top_p, etc.
   - **And** each parameter includes its current value and Python type

2. **AC2: Multiple Component Parameter Extraction**
   - **Given** a Pipeline with multiple component types (Generator, Retriever, Embedder)
   - **When** I introspect the pipeline
   - **Then** parameters are extracted for ALL components
   - **And** each parameter is namespaced by component (e.g., `generator.temperature`)

3. **AC3: Complex Nested Parameter Handling**
   - **Given** a component with complex nested parameters (objects, callables)
   - **When** I introspect the pipeline
   - **Then** the system extracts top-level parameters only (not nested objects)
   - **And** nested objects are recorded as non-tunable with their type

4. **AC4: Parameter Type Detection**
   - **Given** components with various parameter types (int, float, str, bool, Literal)
   - **When** I introspect the pipeline
   - **Then** each parameter includes accurate Python type information
   - **And** Literal types preserve their allowed values

## Tasks / Subtasks

- [x] Task 1: Define Parameter data model (AC: #1, #4)
  - [x] 1.1 Create `Parameter` dataclass in `models.py` with fields: name, value, python_type, type_hint, is_tunable
  - [x] 1.2 Add `Literal` support to store allowed choices when type hint is `Literal[...]`
  - [x] 1.3 Update `Component.parameters` type from `dict[str, Any]` to `dict[str, Parameter]`

- [x] Task 2: Implement parameter extraction from component __init__ signature (AC: #1, #2)
  - [x] 2.1 Create `_extract_parameters(component) -> dict[str, Parameter]` function in introspection.py
  - [x] 2.2 Use `inspect.signature()` to get component's `__init__` parameters
  - [x] 2.3 Get current parameter values from component instance using `getattr()`
  - [x] 2.4 Extract type hints using `typing.get_type_hints()` for PEP 563 compatibility

- [x] Task 3: Implement type hint parsing (AC: #4)
  - [x] 3.1 Create `_parse_type_hint(hint) -> tuple[str, list | None]` to extract type name and Literal choices
  - [x] 3.2 Handle common types: int, float, str, bool, Optional[T], Literal[...], Union[...]
  - [x] 3.3 Use `typing.get_origin()` and `typing.get_args()` for generic type handling
  - [x] 3.4 Return "unknown" for types that cannot be parsed

- [x] Task 4: Handle nested/complex parameters (AC: #3)
  - [x] 4.1 Identify parameters with callable or object types (not int/float/str/bool/Literal)
  - [x] 4.2 Mark complex parameters as `is_tunable=False` with reason stored
  - [x] 4.3 Skip parameters that are: document_store, callback, callable, or complex objects

- [x] Task 5: Update from_pipeline() to populate parameters (AC: #1, #2, #3, #4)
  - [x] 5.1 Call `_extract_parameters()` in `_create_component()`
  - [x] 5.2 Store extracted parameters in Component.parameters field
  - [x] 5.3 Ensure ConfigSpace provides access to all parameters across all components

- [x] Task 6: Add comprehensive tests (AC: #1, #2, #3, #4)
  - [x] 6.1 Test parameter extraction from mock Generator with model, temperature, max_tokens
  - [x] 6.2 Test parameter extraction from multiple component types
  - [x] 6.3 Test complex nested parameter handling (document_store, callbacks)
  - [x] 6.4 Test Literal type hint parsing preserves choices
  - [x] 6.5 Test Optional[T] type hint parsing
  - [x] 6.6 Test component with no type hints (parameters still extracted with "unknown" type)

## Dev Notes

### Architecture Context

This is **Story 1.2** in the Haystack Integration epic. It builds directly on Story 1.1's foundation:
- Extends `Component` dataclass with parameter extraction
- Adds `Parameter` dataclass to model individual parameters
- Uses Python's `inspect` module for signature introspection

**Key Architectural Decisions:**
- Parameters are extracted from `__init__` signature, NOT runtime state
- Current values come from `getattr()` on the component instance
- Type hints are best-effort - components without hints still work
- This prepares for Epic 2 where parameters get ranges and constraints

### Previous Story Learnings (from Story 1.1)

**Patterns Established:**
- Duck typing for Pipeline validation (`hasattr(obj, "walk")`)
- Component extraction via `pipeline.walk()` with `_components` fallback
- Category detection from class name and module path
- Mock-based testing without actual Haystack dependency

**File Structure Established:**
```
traigent/integrations/haystack/
├── __init__.py          # Exports: from_pipeline, ConfigSpace, Component
├── introspection.py     # from_pipeline(), _extract_components(), _create_component()
└── models.py            # ConfigSpace, Component dataclasses
```

**Code Quality from Story 1.1:**
- All tests pass (13 tests in test_haystack_introspection.py)
- Type hints on all public functions
- Docstrings following Google style

### Haystack Component Parameter Patterns

**OpenAIGenerator Parameters (typical):**
```python
class OpenAIGenerator:
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        api_key: Secret = Secret.from_env_var("OPENAI_API_KEY"),
        ...
    ):
```

**InMemoryBM25Retriever Parameters:**
```python
class InMemoryBM25Retriever:
    def __init__(
        self,
        document_store: InMemoryDocumentStore,  # Complex - not tunable
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 10,
        scale_score: bool = False,
    ):
```

**Parameter Categories:**
1. **Tunable Primitives**: int, float, str, bool
2. **Tunable Categoricals**: Literal["a", "b", "c"], Enum
3. **Non-Tunable Objects**: document_store, callback functions, API clients
4. **Secrets**: API keys (never exposed or tuned)

### Python Inspection Patterns

**Getting __init__ Signature:**
```python
import inspect
from typing import get_origin, get_args, Literal

sig = inspect.signature(component.__init__)
for param_name, param in sig.parameters.items():
    if param_name == 'self':
        continue

    # Get type hint
    type_hint = param.annotation
    if type_hint is inspect.Parameter.empty:
        type_hint = None

    # Get current value from instance
    current_value = getattr(component, param_name, param.default)

    # Parse Literal types
    if get_origin(type_hint) is Literal:
        choices = get_args(type_hint)  # ('gpt-4o', 'gpt-4o-mini')
```

**Handling Optional[T]:**
```python
from typing import Union, get_origin, get_args

def unwrap_optional(hint):
    """Extract T from Optional[T] (Union[T, None])."""
    if get_origin(hint) is Union:
        args = get_args(hint)
        # Optional[T] is Union[T, None]
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            return non_none[0], True  # (inner_type, is_optional)
    return hint, False
```

### Testing Strategy

**Mock Component Classes:**
```python
class MockOpenAIGenerator:
    """Mock generator with typical LLM parameters."""
    __module__ = "haystack.components.generators"

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: Optional[float] = 0.7,
        max_tokens: Optional[int] = 100,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

class MockRetrieverWithDocStore:
    """Mock retriever with non-tunable document_store."""
    __module__ = "haystack.components.retrievers"

    def __init__(
        self,
        document_store: Any,  # Complex object
        top_k: int = 10,
    ):
        self.document_store = document_store
        self.top_k = top_k
```

### File Modifications Required

**models.py - Add Parameter dataclass:**
```python
@dataclass
class Parameter:
    """Represents a parameter discovered in a component."""
    name: str
    value: Any
    python_type: str  # "int", "float", "str", "bool", "Literal", "unknown"
    type_hint: str | None  # Original type hint as string
    is_tunable: bool = True
    literal_choices: list[Any] | None = None  # For Literal types
    is_optional: bool = False
    non_tunable_reason: str | None = None  # Why it's not tunable
```

**introspection.py - Add extraction functions:**
```python
def _extract_parameters(component: object) -> dict[str, Parameter]:
    """Extract all __init__ parameters from a component."""
    ...

def _parse_type_hint(hint: Any) -> tuple[str, list | None, bool]:
    """Parse a type hint into (type_name, literal_choices, is_optional)."""
    ...

def _is_tunable_type(python_type: str, value: Any) -> tuple[bool, str | None]:
    """Determine if a parameter type is tunable, with reason if not."""
    ...
```

### References

- [Source: docs/PRD_Agentic_Workflow_Tuning_Haystack.docx - FR-103]
- [Source: _bmad-output/epics.md - Epic 1, Story 1.2]
- [Pattern: Story 1.1 implementation - traigent/integrations/haystack/]
- [Python docs: inspect.signature](https://docs.python.org/3/library/inspect.html#inspect.signature)
- [Python docs: typing.get_origin, typing.get_args](https://docs.python.org/3/library/typing.html)

## Dev Agent Record

### Agent Model Used

Claude Opus 4.5 (claude-opus-4-5-20251101)

### Debug Log References

- Fixed type hint extraction issue with `from __future__ import annotations` (PEP 563)
- Used `typing.get_type_hints()` instead of direct annotation access
- Refactored `_extract_parameters()` to reduce cognitive complexity

### Completion Notes List

- All 24 tests pass (13 from Story 1.1 + 11 new tests for Story 1.2)
- All acceptance criteria verified via tests
- Code passes `make format` and `make lint`

### File List

- traigent/integrations/haystack/models.py - Added Parameter dataclass
- traigent/integrations/haystack/introspection.py - Added parameter extraction functions
- traigent/integrations/haystack/__init__.py - Added Parameter to exports
- traigent/integrations/__init__.py - Added Haystack integration exports (from_pipeline, ConfigSpace, Component, Parameter)
- tests/integrations/test_haystack_introspection.py - Added parameter extraction tests

## Change Log

| Date | Change | Author |
|------|--------|--------|
| 2025-12-19 | Story created from Epic 1.2 with comprehensive dev context | Claude Opus 4.5 (create-story workflow) |
| 2025-12-19 | Implementation complete - all tasks done, ready for review | Claude Opus 4.5 (dev-story workflow) |
| 2025-12-19 | Code review: Fixed 5 issues, added get_parameter() for AC2, added edge case tests | Claude Opus 4.5 (code-review workflow) |
