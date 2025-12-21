#!/usr/bin/env python3
"""Example: Pipeline Introspection with Traigent.

This example demonstrates how to use Traigent's introspection capabilities
to discover tunable parameters in a Haystack pipeline.

Coverage: Epic 1 (Pipeline Discovery & Analysis)
"""

from __future__ import annotations

from typing import Literal
from unittest.mock import MagicMock


# Create proper mock component classes with typed __init__ signatures
# The introspector uses inspect.signature() to discover parameters
class MockRetriever:
    """Mock retriever component with tunable parameters."""

    def __init__(
        self,
        top_k: int = 10,
        score_threshold: float = 0.5,
    ):
        self.top_k = top_k
        self.score_threshold = score_threshold


class MockGenerator:
    """Mock generator component with tunable parameters."""

    def __init__(
        self,
        model: Literal["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"] = "gpt-4o",
        temperature: float = 0.7,
        max_tokens: int = 1024,
        top_p: float = 1.0,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p


# Rename classes to look like Haystack components
MockRetriever.__name__ = "InMemoryBM25Retriever"
MockRetriever.__module__ = "haystack.components.retrievers"
MockGenerator.__name__ = "OpenAIGenerator"
MockGenerator.__module__ = "haystack.components.generators"


def create_mock_pipeline():
    """Create a mock Haystack pipeline for demonstration.

    In production, you would use a real Haystack Pipeline:

        from haystack import Pipeline
        from haystack.components.generators import OpenAIGenerator
        from haystack.components.retrievers import InMemoryBM25Retriever

        pipeline = Pipeline()
        pipeline.add_component("retriever", InMemoryBM25Retriever(...))
        pipeline.add_component("generator", OpenAIGenerator(model="gpt-4o"))
        pipeline.connect("retriever", "generator")
    """
    pipeline = MagicMock()

    # Create component instances with proper __init__ signatures
    retriever = MockRetriever(top_k=10, score_threshold=0.5)
    generator = MockGenerator(model="gpt-4o", temperature=0.7, max_tokens=1024)

    # Configure pipeline methods
    def get_component(name):
        if name == "retriever":
            return retriever
        elif name == "generator":
            return generator
        return None

    pipeline.get_component.side_effect = get_component
    pipeline.walk.return_value = [
        ("retriever", retriever),
        ("generator", generator),
    ]
    pipeline.graph = MagicMock()
    pipeline.graph.edges.return_value = [("retriever", "generator", {})]

    return pipeline


def example_basic_introspection():
    """Demonstrate basic pipeline introspection."""
    print("=" * 60)
    print("Example 1: Basic Pipeline Introspection")
    print("=" * 60)

    from traigent.integrations.haystack import ExplorationSpace, from_pipeline

    pipeline = create_mock_pipeline()

    # First, introspect the pipeline to get a PipelineSpec
    pipeline_spec = from_pipeline(pipeline)

    print("\nDiscovered pipeline structure:")
    print(f"  Scopes (components): {len(pipeline_spec.scopes)}")
    for scope in pipeline_spec.scopes:
        print(f"    - {scope.name} ({scope.category})")
        for tvar_name, tvar in scope.tvars.items():
            val = tvar.value
            tunable = "tunable" if tvar.is_tunable else "fixed"
            print(f"      - {tvar.name}: {tvar.python_type} = {val} ({tunable})")

    # Create exploration space from pipeline spec
    space = ExplorationSpace.from_pipeline_spec(pipeline_spec)

    print(f"\nCreated ExplorationSpace with {len(space.tvars)} parameters:")
    for qualified_name, tvar in space.tvars.items():
        print(f"  - {qualified_name}: {tvar.python_type}")
        if tvar.constraint:
            print(f"    Constraint: {tvar.constraint}")

    print("\n")


def example_component_info():
    """Demonstrate component information extraction."""
    print("=" * 60)
    print("Example 2: Component Information")
    print("=" * 60)

    from traigent.integrations.haystack import ExplorationSpace, from_pipeline

    pipeline = create_mock_pipeline()
    pipeline_spec = from_pipeline(pipeline)
    space = ExplorationSpace.from_pipeline_spec(pipeline_spec)

    print("\nComponents in pipeline:")
    for scope_name in space.scope_names:
        print(f"\n  Component: {scope_name}")
        tvars = space.get_tvars_by_scope(scope_name)
        print(f"    Parameters: {len(tvars)}")
        for name, tvar in tvars.items():
            print(f"      - {tvar.name} ({tvar.python_type})")

    print("\n")


def example_filtering_parameters():
    """Demonstrate parameter filtering capabilities."""
    print("=" * 60)
    print("Example 3: Filtering Parameters")
    print("=" * 60)

    from traigent.integrations.haystack import ExplorationSpace, from_pipeline

    pipeline = create_mock_pipeline()
    pipeline_spec = from_pipeline(pipeline)
    space = ExplorationSpace.from_pipeline_spec(pipeline_spec)

    # Get only tunable parameters
    tunable = space.tunable_tvars
    print(f"\nTunable parameters: {len(tunable)}")
    for name, tvar in tunable.items():
        print(f"  - {tvar.qualified_name}")

    # Get parameters by type
    float_params = [t for t in space.tvars.values() if t.python_type == "float"]
    print(f"\nFloat parameters: {len(float_params)}")
    for tvar in float_params:
        print(f"  - {tvar.qualified_name}: {tvar.default_value}")

    print("\n")


def example_graph_structure():
    """Demonstrate pipeline graph extraction."""
    print("=" * 60)
    print("Example 4: Pipeline Graph Structure")
    print("=" * 60)

    from traigent.integrations.haystack import from_pipeline

    pipeline = create_mock_pipeline()
    pipeline_spec = from_pipeline(pipeline)

    print("\nPipeline connections:")
    for connection in pipeline_spec.connections:
        print(f"  {connection.source} -> {connection.target}")

    print("\n")


if __name__ == "__main__":
    example_basic_introspection()
    example_component_info()
    example_filtering_parameters()
    example_graph_structure()

    print("All introspection examples completed successfully!")
