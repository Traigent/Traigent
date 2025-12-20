# Story 1.1: Extract Components from Pipeline

Status: done

## Story

As an **ML Engineer**,
I want to pass my Haystack Pipeline object to Traigent and get a list of all components,
so that I can see what's in my pipeline without manually inspecting the code.

## Acceptance Criteria

1. **AC1: Basic Component Extraction**
   - **Given** a valid Haystack Pipeline object with multiple components
   - **When** I call `from_pipeline(pipeline)`
   - **Then** the system returns a ConfigSpace object
   - **And** `config_space.components` contains all component names in the pipeline
   - **And** each component includes its class type (e.g., OpenAIGenerator, InMemoryBM25Retriever)

2. **AC2: Empty Pipeline Handling**
   - **Given** an empty Pipeline object
   - **When** I call `from_pipeline(pipeline)`
   - **Then** the system returns a ConfigSpace with empty components list without error

3. **AC3: Component Type Identification**
   - **Given** a Pipeline with various component types (Generator, Retriever, Router, etc.)
   - **When** I call `from_pipeline(pipeline)`
   - **Then** each component in `config_space.components` includes the fully qualified class name

## Tasks / Subtasks

- [x] Task 1: Create Haystack integration module structure (AC: #1, #2, #3)
  - [x] 1.1 Create `traigent/integrations/haystack/` directory
  - [x] 1.2 Create `traigent/integrations/haystack/__init__.py` with exports
  - [x] 1.3 Create `traigent/integrations/haystack/introspection.py` for pipeline analysis

- [x] Task 2: Implement ConfigSpace data model (AC: #1, #2)
  - [x] 2.1 Create `traigent/integrations/haystack/models.py` with ConfigSpace class
  - [x] 2.2 Define Component dataclass with name, class_type, and parameters fields
  - [x] 2.3 Implement ConfigSpace.components property returning list of Component objects

- [x] Task 3: Implement from_pipeline() function (AC: #1, #2, #3)
  - [x] 3.1 Create `from_pipeline(pipeline: Pipeline) -> ConfigSpace` function in introspection.py
  - [x] 3.2 Use `pipeline.walk()` to extract components (with fallback to _components)
  - [x] 3.3 For each component, extract: name (key), class type (type(component).__name__), module path
  - [x] 3.4 Handle empty pipeline case - return ConfigSpace with empty components list

- [x] Task 4: Implement component type detection (AC: #3)
  - [x] 4.1 Extract fully qualified class name using `type(component).__module__ + '.' + type(component).__name__`
  - [x] 4.2 Identify component category (Generator, Retriever, Router, etc.) from class hierarchy
  - [x] 4.3 Store both short name and full module path in Component dataclass

- [x] Task 5: Add Haystack dependency and integration tests (AC: #1, #2, #3)
  - [x] 5.1 Add `haystack-ai>=2.0.0` to optional dependencies in pyproject.toml under `[haystack]` group
  - [x] 5.2 Create `tests/integrations/test_haystack_introspection.py`
  - [x] 5.3 Write test for basic component extraction with multi-component pipeline
  - [x] 5.4 Write test for empty pipeline handling
  - [x] 5.5 Write test for component type identification

## Dev Notes

### Architecture Context

This is the **first story** in the Haystack Integration epic. It establishes the foundation for pipeline introspection that all subsequent stories will build upon.

**Key Architectural Decisions:**
- The `ConfigSpace` class will be expanded in Epic 2 to include parameter types, ranges, and conditionals
- `from_pipeline()` is the primary entry point for users - keep the API simple
- Follow existing Traigent integration patterns from `traigent/integrations/` (see LangChain, OpenAI integrations)

### Existing Codebase Patterns

**Integration Module Pattern:** (from `traigent/integrations/__init__.py`)
```python
# Pattern: Lazy loading with availability flags
try:
    from .haystack import from_pipeline, ConfigSpace
    HAYSTACK_INTEGRATION_AVAILABLE = True
except ImportError:
    HAYSTACK_INTEGRATION_AVAILABLE = False
```

**Base Integration Class:** (from `traigent/integrations/base.py`)
- Use `BaseOverrideManager` pattern if parameter injection is needed later
- Follow thread-safe state management using `ActivationState`

**Dependency Pattern:** (from `pyproject.toml`)
```toml
[project.optional-dependencies]
haystack = [
    "haystack-ai>=2.0.0",
    "networkx>=3.0",  # For graph analysis in later stories
]
```

### Haystack API Reference

**Pipeline Component Access:**
```python
from haystack import Pipeline

pipeline = Pipeline()
# Add components...

# Get all components as dict[str, Component]
components = pipeline.walk()  # or pipeline._components (internal)

# Each component has:
# - Name (the key in the dict)
# - Type (class instance)
# - Init parameters accessible via component.__init__.__signature__
```

**v1 Supported Components (from PRD):**
- Generators: OpenAIGenerator, AnthropicGenerator, HuggingFaceLocalGenerator
- Retrievers: InMemoryBM25Retriever, InMemoryEmbeddingRetriever
- Embedders: OpenAIDocumentEmbedder, OpenAITextEmbedder
- Routers: MetadataRouter, FileTypeRouter, ConditionalRouter
- Converters: HTMLToDocument, MarkdownToDocument, TextFileToDocument
- Builders: PromptBuilder, AnswerBuilder
- Writers: DocumentWriter

### File Structure Requirements

```
traigent/integrations/
├── haystack/
│   ├── __init__.py          # Public exports: from_pipeline, ConfigSpace
│   ├── introspection.py     # from_pipeline() implementation
│   └── models.py            # ConfigSpace, Component dataclasses
tests/integrations/
└── test_haystack_introspection.py
```

### Testing Standards

- Use `pytest` with markers: `@pytest.mark.integration`
- Mock Haystack Pipeline for unit tests to avoid dependency
- Create fixtures for common pipeline configurations
- Test both happy path and edge cases (empty pipeline, unknown component types)

### Code Quality Requirements

**Before committing, run:**
```bash
make format && make lint
```

**Type hints required:**
```python
def from_pipeline(pipeline: "Pipeline") -> ConfigSpace:
    """Extract components from a Haystack Pipeline.

    Args:
        pipeline: A Haystack Pipeline instance

    Returns:
        ConfigSpace with discovered components

    Raises:
        TypeError: If pipeline is not a valid Haystack Pipeline
    """
```

### NFR Considerations (for Story 1.7)

This story lays groundwork for NFR-1 (<100ms introspection) and NFR-5 (≥90% component coverage):
- Keep introspection logic lightweight - no heavy processing in from_pipeline()
- Use lazy loading patterns - don't import haystack until needed
- Prepare for component type registry to track coverage

### References

- [Source: docs/PRD_Agentic_Workflow_Tuning_Haystack.docx - FR-101, FR-102]
- [Source: _bmad-output/epics.md - Epic 1, Story 1.1]
- [Pattern: traigent/integrations/llms/langchain/base.py - Integration structure]
- [Pattern: traigent/integrations/__init__.py - Lazy loading pattern]

## Dev Agent Record

### Agent Model Used

Claude Opus 4.5 (claude-opus-4-5-20251101)

### Debug Log References

- All Python files verified with `python3 -m py_compile`
- No runtime errors during syntax verification

### Completion Notes List

1. **Task 1 Complete**: Created `traigent/integrations/haystack/` module with `__init__.py`, `introspection.py`, and `models.py`
2. **Task 2 Complete**: Implemented `Component` and `ConfigSpace` dataclasses with all required properties and methods
3. **Task 3 Complete**: Implemented `from_pipeline()` function with `pipeline.walk()` and `_components` fallback
4. **Task 4 Complete**: Implemented `_detect_component_category()` for Generator, Retriever, Router, Writer, Embedder, etc.
5. **Task 5 Complete**: Added `haystack-ai>=2.0.0` dependency and comprehensive test suite with 17 test cases

### File List

**New Files Created:**
- `traigent/integrations/haystack/__init__.py` - Public exports
- `traigent/integrations/haystack/introspection.py` - from_pipeline() implementation
- `traigent/integrations/haystack/models.py` - ConfigSpace and Component dataclasses
- `tests/integrations/__init__.py` - Test module init
- `tests/integrations/test_haystack_introspection.py` - 17 test cases for all ACs (TestFromPipeline: 7, TestConfigSpace: 8, TestComponent: 2)

**Modified Files:**
- `pyproject.toml` - Added `[haystack]` optional dependency group
- `traigent/integrations/__init__.py` - Added Haystack exports with availability flag

## Change Log

| Date | Change | Author |
|------|--------|--------|
| 2025-12-19 | Story created from Epic 1.1 | Claude (create-story workflow) |
| 2025-12-19 | Implementation complete - all tasks done, ready for review | Claude Opus 4.5 (dev-story workflow) |
| 2025-12-20 | Code review: Fixed test count (10→17), marked as done | Claude Opus 4.5 (code-review workflow) |
