# Story 1.6: Support Custom @component Decorated Components

Status: done

## Story

As an **ML Engineer**,
I want Traigent to recognize my custom components decorated with @component,
so that my custom logic is included in optimization.

## Acceptance Criteria

1. **AC1: Custom @component Class Introspection**
   - **Given** a Pipeline with a custom `@component` decorated class
   - **When** I introspect the pipeline
   - **Then** the system extracts the component and its __init__ parameters
   - **And** marks appropriate parameters as tunable based on type hints

2. **AC2: Custom Component Without Type Hints**
   - **Given** a custom component without type hints
   - **When** I introspect the pipeline
   - **Then** the system includes the component but marks parameters as requiring manual specification
   - **And** is_tunable=False with appropriate non_tunable_reason

3. **AC3: Custom Component Category Detection**
   - **Given** a custom @component decorated class
   - **When** I introspect the pipeline
   - **Then** the system attempts to detect category from class name or module
   - **And** falls back to "Component" for unrecognized patterns

4. **AC4: Mixed Pipeline Support**
   - **Given** a Pipeline with both built-in Haystack and custom @component classes
   - **When** I introspect the pipeline
   - **Then** both types are correctly extracted and processed
   - **And** the ConfigSpace contains all components

## Tasks / Subtasks

- [x] Task 1: Verify @component decorator detection (AC: #1)
  - [x] 1.1 Confirm current implementation handles @component decorated classes
  - [x] 1.2 Test that `__init__` parameters are extracted correctly
  - [x] 1.3 Verify type hints are parsed for tunable parameters

- [x] Task 2: Handle custom components without type hints (AC: #2)
  - [x] 2.1 Verify parameters without type hints get python_type="unknown"
  - [x] 2.2 Verify is_tunable=False for unknown types
  - [x] 2.3 Add appropriate non_tunable_reason message

- [x] Task 3: Enhance category detection for custom components (AC: #3)
  - [x] 3.1 Review _detect_component_category() for custom patterns
  - [x] 3.2 Ensure fallback to "Component" works correctly
  - [x] 3.3 Category detected from class name and module name

- [x] Task 4: Add comprehensive tests (AC: #1, #2, #3, #4)
  - [x] 4.1 Test @component decorated class extraction
  - [x] 4.2 Test custom component without type hints
  - [x] 4.3 Test mixed pipeline with built-in and custom components
  - [x] 4.4 Test category detection for custom components

## Dev Notes

### Architecture Context

This is **Story 1.6** in the Haystack Integration epic. It ensures custom user-defined components work with introspection.

**Builds on Previous Stories:**
- Story 1.1-1.3: Established parameter extraction with type detection
- The existing implementation should already handle @component decorated classes

**Key Insight: Haystack @component Decorator**

In Haystack 2.x, users can create custom components using the `@component` decorator:

```python
from haystack import component

@component
class MyCustomRetriever:
    """Custom retriever implementation."""

    def __init__(self, threshold: float = 0.5, max_results: int = 10):
        self.threshold = threshold
        self.max_results = max_results

    @component.output_types(documents=list)
    def run(self, query: str) -> dict:
        # Custom retrieval logic
        return {"documents": [...]}
```

**What @component Does:**
- Marks the class as a valid Haystack component
- Allows it to be added to pipelines
- The class still has a normal `__init__` method with parameters

**Current Implementation Status:**

The existing `from_pipeline()` function uses duck typing and should already work with @component decorated classes because:
1. `pipeline.walk()` yields all components regardless of decoration
2. `_extract_parameters()` inspects `__init__` signatures
3. `_parse_type_hint()` handles standard Python type hints

### Testing Strategy

**Mock Custom Component:**
```python
# Simulating @component decorator behavior
class MockCustomComponent:
    """Mock custom component decorated with @component."""

    __module__ = "user_project.components"

    def __init__(
        self,
        threshold: float = 0.5,
        max_results: int = 10,
        custom_option: str = "default",
    ):
        self.threshold = threshold
        self.max_results = max_results
        self.custom_option = custom_option

class MockCustomComponentNoHints:
    """Mock custom component without type hints."""

    __module__ = "user_project.custom"

    def __init__(self, param1, param2="value"):
        self.param1 = param1
        self.param2 = param2
```

### Expected Behavior Verification

For Story 1.6, we mainly need to verify and document that:
1. Custom components ARE extracted (they should be already)
2. Parameters ARE detected with correct types
3. Category detection falls back gracefully
4. Mixed pipelines work correctly

This may be more of a verification/testing story than new implementation.

### References

- [Source: docs/PRD_Agentic_Workflow_Tuning_Haystack.docx - FR-107]
- [Source: _bmad-output/epics.md - Epic 1, Story 1.6]
- [Haystack custom components docs](https://docs.haystack.deepset.ai/docs/custom-components)

## Dev Agent Record

### Agent Model Used

Claude Opus 4.5 (claude-opus-4-5-20251101)

### Debug Log References

- No issues encountered - existing implementation already handles custom components
- All 87 tests pass (80 from Stories 1.1-1.5 + 7 new tests for Story 1.6)

### Completion Notes List

- Verified existing implementation handles @component decorated classes correctly
- Verified parameter extraction works with type hints
- Verified category detection works from class name and module name
- Added 7 comprehensive tests for custom component scenarios
- All code passes `make format` and `make lint`

### File List

**Modified Files:**

- tests/integrations/test_haystack_introspection.py - Added 7 new tests (TestCustomComponentIntrospection class)

## Change Log

| Date | Change | Author |
|------|--------|--------|
| 2025-12-20 | Story created for custom @component support | Claude Opus 4.5 (create-story workflow) |
| 2025-12-20 | Implementation verified, tests added, marked as done | Claude Opus 4.5 (dev-story workflow) |
