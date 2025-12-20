# Story 1.4: Extract Pipeline Graph Structure

Status: done

## Story

As an **ML Engineer**,
I want to see how components are connected in my pipeline,
so that I understand the data flow and can reason about optimization impact.

## Acceptance Criteria

1. **AC1: Linear Pipeline Graph Extraction**
   - **Given** a Pipeline with connected components (A → B → C)
   - **When** I introspect the pipeline
   - **Then** the system returns edge information showing connections
   - **And** edges are stored in `config_space.edges` as a list of (source, target, connection_info) tuples

2. **AC2: NetworkX DiGraph Access**
   - **Given** a Pipeline with connected components
   - **When** I call `config_space.to_networkx()`
   - **Then** the graph structure is returned as a NetworkX DiGraph
   - **And** nodes correspond to component names
   - **And** edges correspond to data flow connections

3. **AC3: Branching Pipeline Support**
   - **Given** a Pipeline with branching (A → B, A → C)
   - **When** I introspect the pipeline
   - **Then** the system correctly captures both branches
   - **And** `config_space.edges` contains both (A, B) and (A, C) connections

4. **AC4: Connection Metadata**
   - **Given** a Pipeline with socket-based connections (output_x → input_y)
   - **When** I introspect the pipeline
   - **Then** each edge includes sender_socket and receiver_socket information
   - **And** this metadata is accessible from the edge tuple

## Tasks / Subtasks

- [x] Task 1: Add Edge data model to models.py (AC: #1, #4)
  - [x] 1.1 Create `Edge` dataclass with fields: source, target, sender_socket, receiver_socket
  - [x] 1.2 Add `edges: list[Edge]` field to ConfigSpace dataclass
  - [x] 1.3 Add `Edge` to module exports in `__init__.py`

- [x] Task 2: Implement graph extraction from Pipeline (AC: #1, #3)
  - [x] 2.1 Create `_extract_graph(pipeline) -> list[Edge]` function in introspection.py
  - [x] 2.2 Use `pipeline.graph.edges` to get NetworkX edges (Haystack uses NetworkX internally)
  - [x] 2.3 Handle pipelines without graph attribute (empty edges list)
  - [x] 2.4 Extract socket information from edge data

- [x] Task 3: Integrate graph extraction into from_pipeline() (AC: #1, #3, #4)
  - [x] 3.1 Call `_extract_graph()` in `from_pipeline()`
  - [x] 3.2 Populate ConfigSpace.edges with extracted edges
  - [x] 3.3 Handle empty pipelines (return empty edges list)

- [x] Task 4: Implement to_networkx() method (AC: #2)
  - [x] 4.1 Add `to_networkx() -> nx.DiGraph` method to ConfigSpace class
  - [x] 4.2 Create DiGraph with component names as nodes
  - [x] 4.3 Add edges with socket metadata as edge attributes
  - [x] 4.4 Add networkx to optional dependencies if not already present

- [x] Task 5: Add comprehensive tests (AC: #1, #2, #3, #4)
  - [x] 5.1 Test linear pipeline graph extraction (A → B → C)
  - [x] 5.2 Test branching pipeline graph extraction (A → B, A → C)
  - [x] 5.3 Test to_networkx() returns valid DiGraph
  - [x] 5.4 Test edge metadata (socket information)
  - [x] 5.5 Test empty pipeline returns empty edges
  - [x] 5.6 Test pipeline without graph attribute (graceful handling)

## Dev Notes

### Architecture Context

This is **Story 1.4** in the Haystack Integration epic. It extends the introspection with graph structure extraction.

**Builds on Previous Stories:**
- Story 1.1: Established `from_pipeline()`, `ConfigSpace`, `Component` dataclasses
- Story 1.2: Added `Parameter` extraction with type detection
- Story 1.3: Added tunability detection and default ranges

**Key Insight: Haystack Uses NetworkX Internally**

Haystack Pipelines already use NetworkX for their internal graph representation. The `Pipeline.graph` attribute is a `networkx.DiGraph` containing:
- Nodes: Component instances (keyed by name)
- Edges: Data flow connections with socket metadata

### Previous Story Learnings

**From Story 1.1-1.3:**
- Duck typing for Pipeline validation works well
- Mock-based testing avoids Haystack dependency in tests
- Helper functions with clear responsibilities keep code maintainable
- 49 tests now pass covering Stories 1.1-1.3

**File Structure Established:**
```
traigent/integrations/haystack/
├── __init__.py          # Exports: from_pipeline, ConfigSpace, Component, Parameter, PARAMETER_SEMANTICS
├── introspection.py     # from_pipeline(), extraction functions
└── models.py            # ConfigSpace, Component, Parameter dataclasses
```

### Haystack Pipeline Graph API

**Accessing the Graph:**
```python
from haystack import Pipeline

pipeline = Pipeline()
pipeline.add_component("retriever", retriever)
pipeline.add_component("generator", generator)
pipeline.connect("retriever.documents", "generator.documents")

# Haystack's internal graph (NetworkX DiGraph)
graph = pipeline.graph

# Nodes are component names
print(list(graph.nodes()))  # ['retriever', 'generator']

# Edges contain connection info
for sender, receiver, data in graph.edges(data=True):
    print(f"{sender} -> {receiver}")
    print(f"  sender_socket: {data.get('conn_type', 'default')}")
    # Edge data structure varies by Haystack version
```

**Connection Syntax:**
```python
# Haystack 2.x connection format
pipeline.connect("component_a.output_socket", "component_b.input_socket")

# The graph stores edges as:
# - Source node: "component_a"
# - Target node: "component_b"
# - Edge data: Contains socket mapping information
```

### Edge Data Model Design

```python
@dataclass
class Edge:
    """Represents a connection between two components in the pipeline.

    Attributes:
        source: Name of the source component
        target: Name of the target component
        sender_socket: Output socket name on source (e.g., "documents", "replies")
        receiver_socket: Input socket name on target (e.g., "documents", "prompt")
    """
    source: str
    target: str
    sender_socket: str | None = None
    receiver_socket: str | None = None
```

### ConfigSpace Extension

```python
@dataclass
class ConfigSpace:
    components: list[Component] = field(default_factory=list)
    edges: list[Edge] = field(default_factory=list)  # NEW

    def to_networkx(self) -> "nx.DiGraph":
        """Convert to NetworkX DiGraph for graph algorithms."""
        import networkx as nx

        G = nx.DiGraph()

        # Add nodes (components)
        for component in self.components:
            G.add_node(component.name, **{"class_name": component.class_name})

        # Add edges with metadata
        for edge in self.edges:
            G.add_edge(
                edge.source,
                edge.target,
                sender_socket=edge.sender_socket,
                receiver_socket=edge.receiver_socket
            )

        return G
```

### Graph Extraction Implementation

```python
def _extract_graph(pipeline: "Pipeline") -> list[Edge]:
    """Extract graph structure from a Haystack Pipeline.

    Args:
        pipeline: A valid Haystack Pipeline instance

    Returns:
        List of Edge objects representing component connections
    """
    edges: list[Edge] = []

    # Check if pipeline has graph attribute
    graph = getattr(pipeline, "graph", None)
    if graph is None:
        return edges

    # Extract edges from NetworkX graph
    for sender, receiver, data in graph.edges(data=True):
        edge = Edge(
            source=sender,
            target=receiver,
            sender_socket=data.get("sender_socket"),
            receiver_socket=data.get("receiver_socket"),
        )
        edges.append(edge)

    return edges
```

### Testing Strategy

**Mock Pipeline with Graph:**
```python
class MockPipelineWithGraph:
    """Mock Pipeline with NetworkX graph for testing."""

    def __init__(self, components: dict, connections: list[tuple]):
        self._components = components
        self._graph = nx.DiGraph()

        # Add nodes
        for name in components:
            self._graph.add_node(name)

        # Add edges: [(sender, receiver, sender_socket, receiver_socket), ...]
        for conn in connections:
            sender, receiver = conn[0], conn[1]
            sender_socket = conn[2] if len(conn) > 2 else None
            receiver_socket = conn[3] if len(conn) > 3 else None
            self._graph.add_edge(
                sender, receiver,
                sender_socket=sender_socket,
                receiver_socket=receiver_socket
            )

    @property
    def graph(self):
        return self._graph

    def walk(self):
        for name, component in self._components.items():
            yield name, component
```

**Test Cases:**
```python
def test_linear_pipeline_graph():
    """AC1: Linear pipeline A → B → C."""
    pipeline = MockPipelineWithGraph(
        components={"a": MockComponent(), "b": MockComponent(), "c": MockComponent()},
        connections=[("a", "b", "output", "input"), ("b", "c", "output", "input")]
    )
    config_space = from_pipeline(pipeline)

    assert len(config_space.edges) == 2
    assert config_space.edges[0].source == "a"
    assert config_space.edges[0].target == "b"

def test_branching_pipeline_graph():
    """AC3: Branching pipeline A → B, A → C."""
    pipeline = MockPipelineWithGraph(
        components={"a": MockComponent(), "b": MockComponent(), "c": MockComponent()},
        connections=[("a", "b"), ("a", "c")]
    )
    config_space = from_pipeline(pipeline)

    assert len(config_space.edges) == 2
    sources = {e.source for e in config_space.edges}
    targets = {e.target for e in config_space.edges}
    assert sources == {"a"}
    assert targets == {"b", "c"}

def test_to_networkx():
    """AC2: to_networkx() returns valid DiGraph."""
    pipeline = MockPipelineWithGraph(
        components={"a": MockComponent(), "b": MockComponent()},
        connections=[("a", "b")]
    )
    config_space = from_pipeline(pipeline)

    G = config_space.to_networkx()
    assert isinstance(G, nx.DiGraph)
    assert list(G.nodes()) == ["a", "b"]
    assert list(G.edges()) == [("a", "b")]
```

### Dependencies

**NetworkX is already a dependency** (from pyproject.toml):
```toml
[project.optional-dependencies]
haystack = [
    "haystack-ai>=2.0.0",
    "networkx>=3.0",  # Already present for graph analysis
]
```

### NFR Considerations

This story prepares for:
- **Story 1.5**: Loop detection using graph analysis (detecting cycles in DiGraph)
- **NFR-1**: Graph extraction should be O(E) where E is number of edges, maintaining <100ms target

### References

- [Source: docs/PRD_Agentic_Workflow_Tuning_Haystack.docx - FR-105]
- [Source: _bmad-output/epics.md - Epic 1, Story 1.4]
- [Haystack Pipeline.connect() API](https://docs.haystack.deepset.ai/docs/pipelines)
- [NetworkX DiGraph documentation](https://networkx.org/documentation/stable/reference/classes/digraph.html)
- [Pattern: Story 1.1-1.3 implementation - traigent/integrations/haystack/]

## Dev Agent Record

### Agent Model Used

Claude Opus 4.5 (claude-opus-4-5-20251101)

### Debug Log References

- No issues encountered during implementation
- All 66 tests pass (49 from Stories 1.1-1.3 + 17 new tests for Story 1.4)

### Completion Notes List

- Added `Edge` dataclass with source, target, sender_socket, receiver_socket fields
- Added `edges: list[Edge]` field to ConfigSpace
- Implemented `_extract_graph()` function for NetworkX edge extraction
- Implemented `to_networkx()` method with node/edge attributes
- Updated ConfigSpace `__repr__` to show edge count
- Added `Edge` to module exports
- All code passes `make format` and `make lint`

### File List

**Modified Files:**

- traigent/integrations/haystack/models.py - Added Edge dataclass, edges field, to_networkx() method
- traigent/integrations/haystack/introspection.py - Added _extract_graph() function
- traigent/integrations/haystack/\_\_init\_\_.py - Added Edge to exports
- tests/integrations/test_haystack_introspection.py - Added 17 new tests (TestGraphExtraction: 7, TestToNetworkx: 6, TestEdge: 4)

## Change Log

| Date | Change | Author |
|------|--------|--------|
| 2025-12-20 | Story created with comprehensive graph extraction context | Claude Opus 4.5 (create-story workflow) |
| 2025-12-20 | Implementation complete - all tasks done, ready for review | Claude Opus 4.5 (dev-story workflow) |
| 2025-12-20 | Code review passed - all ACs verified, marked as done | Claude Opus 4.5 (code-review workflow) |
