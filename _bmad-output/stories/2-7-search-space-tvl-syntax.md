# Story 2.7: Search Space TVL Syntax

Status: done

## Story

As an **ML Engineer**,
I want to specify parameter ranges and choices using Search Space TVL syntax,
so that I can customize and share search space definitions in a portable format.

> **⚠️ REVIEW NOTE:** This story MUST reuse the existing `traigent/tvl/spec_loader.py`
> infrastructure rather than creating a parallel schema. The `configuration_space`
> section in existing TVL specs should be the basis for this feature.

## Acceptance Criteria

1. **AC1: Load Search Space from TVL File**
   - **Given** a Search Space TVL file with parameter range specifications
   - **When** I load it with `ExplorationSpace.from_search_tvl("search_space.tvl")`
   - **Then** the ExplorationSpace is configured according to the TVL definitions

2. **AC2: Export Search Space to TVL File**
   - **Given** an ExplorationSpace object
   - **When** I call `space.to_search_tvl("search_space.tvl")`
   - **Then** the file contains a valid Search Space TVL representation (ranges, choices, conditionals)

3. **AC3: Parse Error with Line Number**
   - **Given** a Search Space TVL file with syntax errors
   - **When** I try to load it
   - **Then** the system raises a clear parsing error with line number

4. **AC4: Round-Trip Preservation**
   - **Given** an ExplorationSpace with various constraints
   - **When** I export to TVL and reload
   - **Then** the constraints are preserved (choices, ranges, log_scale, conditionals)

## Tasks / Subtasks

- [x] Task 1: Reuse existing TVL infrastructure (AC: #1, #2)
  - [x] 1.1 Reuse `traigent/tvl/spec_loader.py` for loading TVL specs
  - [x] 1.2 Support categorical via existing `configuration_space` section
  - [x] 1.3 Support numerical: continuous and integer types with ranges
  - [x] 1.4 Support conditionals via ConditionalConstraint serialization
  - [x] 1.5 Support fixed values via constraint=None serialization

- [x] Task 2: Implement `from_tvl_spec()` class method (AC: #1, #3)
  - [x] 2.1 Add `ExplorationSpace.from_tvl_spec(path, environment)` method
  - [x] 2.2 Leverage existing `load_tvl_spec()` for YAML parsing
  - [x] 2.3 Convert TVL domain to TVAR constraints via `_tvar_from_tvl_domain()`
  - [x] 2.4 Errors propagated from `load_tvl_spec()` include file context

- [x] Task 3: Implement `to_tvl()` method (AC: #2)
  - [x] 3.1 Add `space.to_tvl(path, include_metadata, description)` method
  - [x] 3.2 Convert TVARs to TVL YAML format via helper methods
  - [x] 3.3 Include optional metadata section with export info

- [x] Task 4: Add comprehensive tests (AC: #1, #2, #3, #4)
  - [x] 4.1 Test loading simple TVL with ranges and choices (5 tests)
  - [x] 4.2 Test loading TVL with boolean parameters
  - [x] 4.3 Test exporting space to TVL (4 tests)
  - [x] 4.4 Test round-trip preservation
  - [x] 4.5 Test invalid/missing file raises TVLValidationError
  - [x] 4.6 Test conditional constraint export

## Dev Notes

### Architecture Context

This is **Story 2.7** in Epic 2 (Configuration Space & TVL). It provides a portable, human-readable format for search space definitions.

**Key Distinction:**
- **Search Space TVL (Epic 2)**: Defines parameter *ranges* for optimization
- **Tuned Config TVL (Epic 7)**: Captures *specific values* + metrics as artifacts

### Implementation Approach (REVISED)

Instead of creating a new "search-space-tvl" schema, this story should:

1. **Reuse `traigent/tvl/spec_loader.py`** - The existing `load_tvl_spec()` function
   already parses `configuration_space` with types, ranges, and values
2. **Add `ExplorationSpace.from_tvl_spec(path)`** - Wrapper that calls `load_tvl_spec()`
   and converts the result to an ExplorationSpace
3. **Add `to_tvl(path)`** - Export method using the existing TVL schema format

### Existing TVL Schema (from spec_loader.py)

The existing `configuration_space` section supports:

```yaml
configuration_space:
  # Categorical parameter
  generator.model:
    type: categorical
    values: ["gpt-4o", "gpt-4o-mini"]
    default: "gpt-4o"

  # Continuous parameter
  generator.temperature:
    type: continuous
    range: [0.0, 2.0]
    default: 0.7

  # Integer parameter
  retriever.top_k:
    type: integer
    range: [1, 50]
    default: 10

  # Boolean parameter
  generator.stream:
    type: boolean
```

### Conditional Constraints (via existing constraints section)

For conditional constraints, use the existing `constraints.structural` section:

```yaml
constraints:
  structural:
    - when: "params.generator.model == 'gpt-4o'"
      then: "params.generator.max_tokens <= 8192"
    - when: "params.generator.model == 'gpt-4o-mini'"
      then: "params.generator.max_tokens <= 4096"
```

### DEPRECATED - Original Proposal (DO NOT USE)

The following schema was originally proposed but should NOT be implemented
as it duplicates existing infrastructure:

```yaml
# DEPRECATED - DO NOT IMPLEMENT
# Search Space TVL Schema v1.0
# Defines the optimization search space

version: "1.0"
schema: "search-space-tvl"

parameters:
  # Categorical parameter with explicit choices
  generator.model:
    type: categorical
    choices: ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"]
    default: "gpt-4o"

  # Numerical parameter with range
  generator.temperature:
    type: float
    range:
      min: 0.0
      max: 2.0
    default: 0.7

  # Integer parameter with step
  retriever.top_k:
    type: int
    range:
      min: 1
      max: 50
      step: 1
    default: 10

  # Log-scale parameter
  optimizer.learning_rate:
    type: float
    range:
      min: 0.0001
      max: 0.1
      log_scale: true
    default: 0.01

  # Conditional parameter
  generator.max_tokens:
    type: int
    conditional:
      parent: generator.model
      conditions:
        gpt-4o:
          range: {min: 100, max: 8192}
        gpt-4o-mini:
          range: {min: 100, max: 4096}
    default: 1024

  # Fixed parameter (excluded from search)
  generator.seed:
    fixed: 42

# Optional metadata
metadata:
  created: "2025-12-20"
  description: "Search space for RAG pipeline optimization"
```

### Implementation Pattern

```python
class TVLParseError(Exception):
    """Exception raised when parsing TVL fails."""

    def __init__(self, message: str, line: int | None = None, file: str | None = None):
        self.line = line
        self.file = file
        if line:
            message = f"Line {line}: {message}"
        if file:
            message = f"{file}: {message}"
        super().__init__(message)


class ExplorationSpace:
    @classmethod
    def from_search_tvl(cls, path: str | Path) -> ExplorationSpace:
        """Load an ExplorationSpace from a Search Space TVL file.

        Args:
            path: Path to the TVL file (YAML format)

        Returns:
            ExplorationSpace configured per the TVL definitions

        Raises:
            FileNotFoundError: If the file doesn't exist
            TVLParseError: If the TVL syntax is invalid
        """
        import yaml
        from pathlib import Path

        path = Path(path)
        try:
            with open(path) as f:
                data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            line = getattr(e, 'problem_mark', None)
            line_num = line.line + 1 if line else None
            raise TVLParseError(str(e), line=line_num, file=str(path)) from e

        # Validate schema
        if data.get("schema") != "search-space-tvl":
            raise TVLParseError("Invalid schema: expected 'search-space-tvl'", file=str(path))

        # Parse parameters
        tvars: dict[str, TVAR] = {}
        for qualified_name, spec in data.get("parameters", {}).items():
            tvar = cls._parse_tvar_spec(qualified_name, spec)
            tvars[qualified_name] = tvar

        return cls(tvars=tvars)

    def to_search_tvl(self, path: str | Path) -> None:
        """Export this ExplorationSpace to a Search Space TVL file.

        Args:
            path: Path to write the TVL file
        """
        import yaml
        from pathlib import Path

        data = {
            "version": "1.0",
            "schema": "search-space-tvl",
            "parameters": {}
        }

        for qualified_name, tvar in self.tvars.items():
            spec = self._tvar_to_spec(tvar)
            data["parameters"][qualified_name] = spec

        path = Path(path)
        with open(path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
```

### Testing Strategy

```python
def test_from_search_tvl_loads_categorical():
    """Test loading categorical parameter from TVL."""
    tvl_content = """
version: "1.0"
schema: "search-space-tvl"
parameters:
  generator.model:
    type: categorical
    choices: ["gpt-4o", "gpt-4o-mini"]
    default: "gpt-4o"
"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.tvl', delete=False) as f:
        f.write(tvl_content)
        f.flush()
        space = ExplorationSpace.from_search_tvl(f.name)

    tvar = space.get_tvar("generator", "model")
    assert isinstance(tvar.constraint, CategoricalConstraint)
    assert tvar.constraint.choices == ["gpt-4o", "gpt-4o-mini"]

def test_to_search_tvl_round_trip():
    """Test exporting and reloading preserves constraints."""
    original = _create_complex_space()

    with tempfile.TemporaryDirectory() as tmpdir:
        tvl_path = Path(tmpdir) / "space.tvl"
        original.to_search_tvl(tvl_path)
        reloaded = ExplorationSpace.from_search_tvl(tvl_path)

    # Compare constraints
    for name, tvar in original.tvars.items():
        reloaded_tvar = reloaded.tvars[name]
        assert type(tvar.constraint) == type(reloaded_tvar.constraint)
        # ... detailed constraint comparison

def test_from_search_tvl_parse_error_includes_line():
    """Test that parse errors include line numbers."""
    tvl_content = """
version: "1.0"
schema: "search-space-tvl"
parameters:
  generator.model:
    type: [invalid yaml
"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.tvl', delete=False) as f:
        f.write(tvl_content)
        f.flush()

        with pytest.raises(TVLParseError) as exc_info:
            ExplorationSpace.from_search_tvl(f.name)

        assert exc_info.value.line is not None
```

### File Structure

```
traigent/integrations/haystack/
├── configspace.py       # MODIFY: Add from_search_tvl(), to_search_tvl()
├── tvl.py               # NEW: TVL parsing utilities, TVLParseError
└── __init__.py          # MODIFY: Export TVL functions

tests/integrations/
├── test_configspace.py  # MODIFY: Add TestTVL class
└── fixtures/            # NEW: Test TVL files
    ├── valid_space.tvl
    ├── invalid_syntax.tvl
    └── complex_space.tvl
```

### Dependencies

- **Story 2.1-2.5 (done/ready)**: Provides constraint types
- **Story 2.4 (ready)**: Provides ConditionalConstraint for conditional TVL
- **External**: PyYAML for parsing (already a dependency via Haystack)

### NFR Considerations

- **NFR-4 (≤10 lines integration)**: TVL should simplify configuration:
  ```python
  # Load pre-defined search space
  space = ExplorationSpace.from_search_tvl("my_search_space.tvl")
  # Or export current space for sharing
  space.to_search_tvl("exported_space.tvl")
  ```

### References

- [Source: _bmad-output/epics.md - Epic 2, Story 2.7 - FR-201]
- [Pattern: traigent core TVL parser (future integration)]
- [Dependency: Stories 2.1-2.6]

## Dev Agent Record

### Agent Model Used

Claude Opus 4.5 (claude-opus-4-5-20251101)

### Debug Log References

N/A - No issues encountered

### Completion Notes List

- Reused existing `traigent/tvl/spec_loader.py` infrastructure instead of creating parallel schema
- Added `from_tvl_spec(path, environment)` class method that wraps `load_tvl_spec()`
- Added `_tvar_from_tvl_domain()` helper to convert TVL domains to TVAR constraints
- Added `to_tvl(path, include_metadata, description)` method for export
- Refactored TVL export into helper methods to reduce cognitive complexity:
  - `_constraint_to_tvl_spec()`, `_categorical_to_tvl_spec()`, `_numerical_to_tvl_spec()`, `_conditional_to_tvl_spec()`
- Added `_infer_type_from_choices()` module-level helper
- Added 15 comprehensive tests in `TestTVLIntegration` class
- All 173 configspace tests pass

### File List

- `traigent/integrations/haystack/configspace.py` - Added from_tvl_spec(), to_tvl(), helper methods
- `tests/integrations/test_configspace.py` - Added TestTVLIntegration with 15 tests

## Change Log

| Date       | Change                                    | Author                                   |
|------------|-------------------------------------------|------------------------------------------|
| 2025-12-20 | Story created for TVL syntax support     | Claude Opus 4.5 (create-story workflow)  |
| 2025-12-20 | Story implemented with all tests passing | Claude Opus 4.5                          |
