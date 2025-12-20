# Story 1.5: Detect Loops and Max Runs

Status: done

## Story

As an **ML Engineer**,
I want the system to detect loops in my pipeline and extract max_runs_per_component,
so that optimization can account for bounded iteration.

## Acceptance Criteria

1. **AC1: Loop Detection in Pipeline**
   - **Given** a Pipeline with a loop (e.g., agent retry logic)
   - **When** I introspect the pipeline
   - **Then** the system identifies the loop structure
   - **And** `config_space.has_loops` returns True

2. **AC2: Max Runs Extraction**
   - **Given** a Pipeline with max_runs_per_component settings
   - **When** I introspect the pipeline
   - **Then** the system extracts max_runs_per_component for each component
   - **And** this info is stored in component metadata

3. **AC3: No Loops Detection**
   - **Given** a Pipeline without loops (DAG structure)
   - **When** I introspect the pipeline
   - **Then** `config_space.has_loops` returns False
   - **And** no loop information is reported

4. **AC4: Unbounded Loop Warning**
   - **Given** a cyclic Pipeline without max_runs_per_component set
   - **When** I introspect the pipeline
   - **Then** the system logs a warning about unbounded loops
   - **And** marks the loop as potentially unbounded in metadata

## Tasks / Subtasks

- [x] Task 1: Add loop detection fields to ConfigSpace (AC: #1, #3)
  - [x] 1.1 Add `has_loops: bool` property to ConfigSpace
  - [x] 1.2 Add `loops: list[list[str]]` field to store detected cycles
  - [x] 1.3 Implement cycle detection using NetworkX `simple_cycles()`

- [x] Task 2: Add max_runs extraction to Component (AC: #2)
  - [x] 2.1 Add `max_runs: int | None` field to Component dataclass
  - [x] 2.2 Extract max_runs from pipeline metadata if available
  - [x] 2.3 Handle pipelines without max_runs metadata

- [x] Task 3: Implement loop detection in introspection (AC: #1, #3)
  - [x] 3.1 Create `_detect_loops(config_space) -> list[list[str]]` function
  - [x] 3.2 Use `nx.simple_cycles()` for cycle detection
  - [x] 3.3 Integrate into `from_pipeline()` to populate loops field

- [x] Task 4: Handle unbounded loops (AC: #4)
  - [x] 4.1 Check if looped components have max_runs set
  - [x] 4.2 Log warning for unbounded loops via `_warn_unbounded_loops()`
  - [x] 4.3 Add `unbounded_loops` property to ConfigSpace

- [x] Task 5: Add comprehensive tests (AC: #1, #2, #3, #4)
  - [x] 5.1 Test DAG pipeline has_loops=False
  - [x] 5.2 Test cyclic pipeline has_loops=True
  - [x] 5.3 Test max_runs extraction from component
  - [x] 5.4 Test unbounded loop detection
  - [x] 5.5 Test multiple cycles detected

## Dev Notes

### Architecture Context

This is **Story 1.5** in the Haystack Integration epic. It extends graph analysis with cycle detection.

**Builds on Previous Stories:**
- Story 1.4: Added `edges` and `to_networkx()` to ConfigSpace
- Now we use the NetworkX DiGraph to detect cycles

**Key Insight: NetworkX Cycle Detection**

NetworkX provides `simple_cycles()` for detecting all simple cycles in a directed graph:

```python
import networkx as nx

G = nx.DiGraph()
G.add_edges_from([("a", "b"), ("b", "c"), ("c", "a")])  # Creates a cycle

cycles = list(nx.simple_cycles(G))
# [['a', 'b', 'c']]
```

### Haystack Loop Patterns

**Agent Retry Loop:**
```python
# Haystack 2.x uses max_runs_per_component to limit iterations
pipeline = Pipeline(max_runs_per_component=5)
pipeline.add_component("router", ConditionalRouter(...))
pipeline.add_component("generator", OpenAIGenerator(...))
pipeline.connect("router.retry", "generator")  # Loop back
pipeline.connect("generator.output", "router")
```

**Extracting max_runs:**
```python
# Haystack Pipeline stores max_runs_per_component
max_runs = pipeline.max_runs_per_component  # e.g., 5 or None
```

### Data Model Extensions

```python
@dataclass
class Component:
    # Existing fields...
    name: str
    class_name: str
    class_type: str
    category: str
    parameters: dict[str, Parameter]
    # NEW: max_runs for loop-aware components
    max_runs: int | None = None

@dataclass
class ConfigSpace:
    components: list[Component]
    edges: list[Edge]
    # NEW: Loop detection fields
    loops: list[list[str]] = field(default_factory=list)

    @property
    def has_loops(self) -> bool:
        """Check if pipeline contains cycles."""
        return len(self.loops) > 0

    @property
    def unbounded_loops(self) -> list[list[str]]:
        """Return loops where no component has max_runs set."""
        ...
```

### Loop Detection Implementation

```python
def _detect_loops(graph: "nx.DiGraph") -> list[list[str]]:
    """Detect all simple cycles in the pipeline graph.

    Args:
        graph: NetworkX DiGraph of the pipeline

    Returns:
        List of cycles, where each cycle is a list of component names
    """
    try:
        import networkx as nx
        return list(nx.simple_cycles(graph))
    except Exception:
        return []
```

### Testing Strategy

**Mock Pipeline with Loop:**
```python
class MockPipelineWithLoop:
    def __init__(self):
        self._components = {
            "router": MockRouter(),
            "generator": MockGenerator(),
        }
        self._graph = nx.DiGraph()
        self._graph.add_edges_from([
            ("router", "generator"),
            ("generator", "router"),  # Creates cycle
        ])
        self.max_runs_per_component = 5

    @property
    def graph(self):
        return self._graph

    def walk(self):
        for name, comp in self._components.items():
            yield name, comp
```

### References

- [Source: docs/PRD_Agentic_Workflow_Tuning_Haystack.docx - FR-106]
- [Source: _bmad-output/epics.md - Epic 1, Story 1.5]
- [NetworkX simple_cycles documentation](https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.cycles.simple_cycles.html)
- [Haystack Pipeline loops documentation](https://docs.haystack.deepset.ai/docs/pipelines#loops)

## Dev Agent Record

### Agent Model Used

Claude Opus 4.5 (claude-opus-4-5-20251101)

### Debug Log References

- No issues encountered during implementation
- All 80 tests pass (66 from Stories 1.1-1.4 + 14 new tests for Story 1.5)

### Completion Notes List

- Added `loops: list[list[str]]` field to ConfigSpace dataclass
- Added `has_loops` property to ConfigSpace
- Added `unbounded_loops` property to ConfigSpace
- Added `max_runs: int | None` field to Component dataclass
- Implemented `_extract_max_runs()` function for pipeline max_runs extraction
- Implemented `_detect_loops()` function using nx.simple_cycles()
- Implemented `_warn_unbounded_loops()` for logging unbounded loop warnings
- Updated ConfigSpace `__repr__` to show loop count when loops present
- All code passes `make format` and `make lint`

### File List

**Modified Files:**

- traigent/integrations/haystack/models.py - Added loops field, has_loops/unbounded_loops properties
- traigent/integrations/haystack/introspection.py - Added _extract_max_runs(), _detect_loops(), _warn_unbounded_loops()
- tests/integrations/test_haystack_introspection.py - Added 14 new tests (TestLoopDetection: 4, TestMaxRunsExtraction: 3, TestUnboundedLoops: 3, TestConfigSpaceLoopProperties: 4)

## Change Log

| Date | Change | Author |
|------|--------|--------|
| 2025-12-20 | Story created with loop detection context | Claude Opus 4.5 (create-story workflow) |
| 2025-12-20 | Implementation complete - all tasks done, ready for review | Claude Opus 4.5 (dev-story workflow) |
| 2025-12-20 | Code review passed - all ACs verified, marked as done | Claude Opus 4.5 (code-review workflow) |
