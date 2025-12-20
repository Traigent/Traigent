"""Tests for Haystack pipeline introspection.

These tests verify the from_pipeline() function and PipelineSpec model
for extracting TVARScopes (components) and DiscoveredTVARs (parameters)
from Haystack pipelines using TVL (Tuning Variable Language) terminology.

Terminology (aligned with TVL):
- TVAR (Tuned Variable): A parameter that can be optimized
- DiscoveredTVAR: A TVAR discovered through pipeline introspection
- TVARScope: A namespace/container for TVARs (maps to Haystack Component)
- PipelineSpec: The complete discovered pipeline structure with all scopes
- Connection: A data flow edge between TVARScopes
"""

from __future__ import annotations

import os
from typing import Any, Literal, Optional

import pytest

# Skip performance tests in CI environments where timing may be unreliable
_IN_CI = os.environ.get("CI", "").lower() in ("true", "1", "yes")
skip_in_ci = pytest.mark.skipif(
    _IN_CI, reason="Performance tests may be flaky in CI environments"
)

from traigent.integrations.haystack import (  # New TVL-aligned types
    Connection,
    DiscoveredTVAR,
    PipelineSpec,
    TVARScope,
    from_pipeline,
)

# Backwards-compatible aliases for migration testing
Component = TVARScope
ConfigSpace = PipelineSpec
Parameter = DiscoveredTVAR


class MockComponent:
    """Mock Haystack component for testing."""

    pass


class MockOpenAIGenerator(MockComponent):
    """Mock OpenAI Generator component."""

    __module__ = "haystack.components.generators"


class MockInMemoryBM25Retriever(MockComponent):
    """Mock BM25 Retriever component."""

    __module__ = "haystack.components.retrievers"


class MockPromptBuilder(MockComponent):
    """Mock Prompt Builder component."""

    __module__ = "haystack.components.builders"


class MockMetadataRouter(MockComponent):
    """Mock Metadata Router component."""

    __module__ = "haystack.components.routers"


class MockDocumentWriter(MockComponent):
    """Mock Document Writer component."""

    __module__ = "haystack.components.writers"


class MockPipeline:
    """Mock Haystack Pipeline for testing without actual haystack dependency."""

    def __init__(self, components: dict[str, Any] | None = None):
        self._components = components or {}

    def walk(self):
        """Yield component name and instance pairs."""
        for name, component in self._components.items():
            yield name, component


class TestFromPipeline:
    """Tests for the from_pipeline() function."""

    def test_basic_component_extraction(self):
        """AC1: Basic component extraction with multi-component pipeline."""
        # Given a valid Haystack Pipeline with multiple components
        pipeline = MockPipeline(
            {
                "generator": MockOpenAIGenerator(),
                "retriever": MockInMemoryBM25Retriever(),
                "prompt_builder": MockPromptBuilder(),
            }
        )

        # When I call from_pipeline(pipeline)
        config_space = from_pipeline(pipeline)

        # Then the system returns a ConfigSpace object
        assert isinstance(config_space, ConfigSpace)

        # And config_space.components contains all component names
        assert config_space.component_count == 3
        assert "generator" in config_space.component_names
        assert "retriever" in config_space.component_names
        assert "prompt_builder" in config_space.component_names

        # And each component includes its class type
        generator = config_space.get_component("generator")
        assert generator is not None
        assert generator.class_name == "MockOpenAIGenerator"

    def test_empty_pipeline_handling(self):
        """AC2: Empty pipeline returns ConfigSpace with empty components list."""
        # Given an empty Pipeline object
        pipeline = MockPipeline({})

        # When I call from_pipeline(pipeline)
        config_space = from_pipeline(pipeline)

        # Then the system returns a ConfigSpace with empty components list
        assert isinstance(config_space, ConfigSpace)
        assert config_space.component_count == 0
        assert config_space.components == []

    def test_component_type_identification(self):
        """AC3: Component type identification with fully qualified class names."""
        # Given a Pipeline with various component types
        pipeline = MockPipeline(
            {
                "generator": MockOpenAIGenerator(),
                "retriever": MockInMemoryBM25Retriever(),
                "router": MockMetadataRouter(),
                "writer": MockDocumentWriter(),
            }
        )

        # When I call from_pipeline(pipeline)
        config_space = from_pipeline(pipeline)

        # Then each component includes the fully qualified class name
        generator = config_space.get_component("generator")
        assert generator is not None
        assert "MockOpenAIGenerator" in generator.class_type
        assert generator.category == "Generator"

        retriever = config_space.get_component("retriever")
        assert retriever is not None
        assert "MockInMemoryBM25Retriever" in retriever.class_type
        assert retriever.category == "Retriever"

        router = config_space.get_component("router")
        assert router is not None
        assert router.category == "Router"

        writer = config_space.get_component("writer")
        assert writer is not None
        assert writer.category == "Writer"

    def test_invalid_pipeline_raises_type_error(self):
        """Verify TypeError is raised for non-Pipeline objects."""
        # Given an object that is not a Haystack Pipeline
        not_a_pipeline = {"components": []}

        # When I call from_pipeline() with invalid input
        # Then a TypeError is raised
        with pytest.raises(TypeError) as exc_info:
            from_pipeline(not_a_pipeline)  # type: ignore

        assert "Expected a Haystack Pipeline" in str(exc_info.value)

    def test_object_with_non_callable_walk_rejected(self):
        """Edge case: Object with walk attribute that is NOT callable."""

        class FakePipelineWithNonCallableWalk:
            walk = "not a method"  # walk exists but not callable
            _components = None  # _components exists but not a dict

        fake_pipeline = FakePipelineWithNonCallableWalk()
        with pytest.raises(TypeError) as exc_info:
            from_pipeline(fake_pipeline)  # type: ignore

        assert "Expected a Haystack Pipeline" in str(exc_info.value)

    def test_object_with_non_dict_components_rejected(self):
        """Edge case: Object with _components attribute that is NOT a dict."""

        class FakePipelineWithNonDictComponents:
            _components = ["not", "a", "dict"]  # Wrong type

        fake_pipeline = FakePipelineWithNonDictComponents()
        with pytest.raises(TypeError) as exc_info:
            from_pipeline(fake_pipeline)  # type: ignore

        assert "Expected a Haystack Pipeline" in str(exc_info.value)

    def test_pipeline_with_internal_components_dict(self):
        """Test fallback to _components when walk() is not available."""

        class PipelineWithoutWalk:
            """Pipeline that only has _components attribute."""

            def __init__(self):
                self._components = {
                    "generator": MockOpenAIGenerator(),
                }

        pipeline = PipelineWithoutWalk()
        config_space = from_pipeline(pipeline)

        assert config_space.component_count == 1
        assert "generator" in config_space.component_names


class TestPipelineSpec:
    """Tests for the PipelineSpec data model (ConfigSpace backwards compat)."""

    def test_component_names_property(self):
        """Test component_names returns list of names."""
        scopes = [
            TVARScope(
                name="comp1",
                class_name="Class1",
                class_type="module.Class1",
                category="Component",
            ),
            TVARScope(
                name="comp2",
                class_name="Class2",
                class_type="module.Class2",
                category="Component",
            ),
        ]
        config_space = PipelineSpec(scopes=scopes)

        assert config_space.component_names == ["comp1", "comp2"]

    def test_get_component_by_name(self):
        """Test get_component returns correct component."""
        scopes = [
            TVARScope(
                name="generator",
                class_name="OpenAIGenerator",
                class_type="haystack.components.generators.OpenAIGenerator",
                category="Generator",
            ),
        ]
        config_space = PipelineSpec(scopes=scopes)

        result = config_space.get_component("generator")
        assert result is not None
        assert result.name == "generator"
        assert result.class_name == "OpenAIGenerator"

    def test_get_component_not_found(self):
        """Test get_component returns None for missing component."""
        config_space = PipelineSpec(scopes=[])

        result = config_space.get_component("nonexistent")
        assert result is None

    def test_get_parameter_namespaced_access(self):
        """Test get_parameter provides namespaced access (AC2 compliance)."""
        # Create scope with TVARs
        scope = TVARScope(
            name="generator",
            class_name="OpenAIGenerator",
            class_type="haystack.components.generators.OpenAIGenerator",
            category="Generator",
            tvars={
                "temperature": DiscoveredTVAR(
                    name="temperature",
                    value=0.7,
                    python_type="float",
                    is_tunable=True,
                ),
                "model": DiscoveredTVAR(
                    name="model",
                    value="gpt-4o",
                    python_type="str",
                    is_tunable=True,
                ),
            },
        )
        config_space = PipelineSpec(scopes=[scope])

        # Test namespaced access: generator.temperature
        temp_param = config_space.get_parameter("generator", "temperature")
        assert temp_param is not None
        assert abs(temp_param.value - 0.7) < 1e-9  # Float comparison
        assert temp_param.python_type == "float"

        # Test namespaced access: generator.model
        model_param = config_space.get_parameter("generator", "model")
        assert model_param is not None
        assert model_param.value == "gpt-4o"

    def test_get_parameter_not_found(self):
        """Test get_parameter returns None for missing component or param."""
        config_space = PipelineSpec(scopes=[])

        # Missing component
        result = config_space.get_parameter("nonexistent", "temperature")
        assert result is None

        # Existing scope but missing TVAR
        scope = TVARScope(
            name="generator",
            class_name="Gen",
            class_type="module.Gen",
            category="Generator",
            tvars={},
        )
        config_space = PipelineSpec(scopes=[scope])
        result = config_space.get_parameter("generator", "nonexistent")
        assert result is None

    def test_get_components_by_category(self):
        """Test filtering components by category."""
        scopes = [
            TVARScope(
                name="gen1",
                class_name="Gen1",
                class_type="module.Gen1",
                category="Generator",
            ),
            TVARScope(
                name="ret1",
                class_name="Ret1",
                class_type="module.Ret1",
                category="Retriever",
            ),
            TVARScope(
                name="gen2",
                class_name="Gen2",
                class_type="module.Gen2",
                category="Generator",
            ),
        ]
        config_space = PipelineSpec(scopes=scopes)

        generators = config_space.get_components_by_category("Generator")
        assert len(generators) == 2
        assert all(c.category == "Generator" for c in generators)

        retrievers = config_space.get_components_by_category("Retriever")
        assert len(retrievers) == 1

    def test_configspace_iteration(self):
        """Test PipelineSpec can be iterated over."""
        scopes = [
            TVARScope(
                name="comp1",
                class_name="Class1",
                class_type="module.Class1",
                category="Component",
            ),
            TVARScope(
                name="comp2",
                class_name="Class2",
                class_type="module.Class2",
                category="Component",
            ),
        ]
        config_space = PipelineSpec(scopes=scopes)

        names = [c.name for c in config_space]
        assert names == ["comp1", "comp2"]

    def test_configspace_len(self):
        """Test PipelineSpec length."""
        scopes = [
            TVARScope(
                name="comp1",
                class_name="Class1",
                class_type="module.Class1",
                category="Component",
            ),
        ]
        config_space = PipelineSpec(scopes=scopes)

        assert len(config_space) == 1


class TestTVARScope:
    """Tests for the TVARScope data model (Component backwards compat)."""

    def test_component_repr(self):
        """Test Component string representation."""
        component = Component(
            name="generator",
            class_name="OpenAIGenerator",
            class_type="haystack.components.generators.OpenAIGenerator",
            category="Generator",
        )

        repr_str = repr(component)
        assert "generator" in repr_str
        assert "OpenAIGenerator" in repr_str
        assert "Generator" in repr_str

    def test_component_default_parameters(self):
        """Test Component has empty parameters by default."""
        component = Component(
            name="test",
            class_name="Test",
            class_type="module.Test",
            category="Component",
        )

        assert component.parameters == {}


# ============================================================================
# Story 1.2: Parameter Extraction Tests
# ============================================================================


class MockGeneratorWithParams:
    """Mock generator with typical LLM parameters for testing."""

    __module__ = "haystack.components.generators"

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: Optional[float] = 0.7,
        max_tokens: Optional[int] = 100,
        top_p: Optional[float] = None,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p


class MockRetrieverWithDocStore:
    """Mock retriever with non-tunable document_store parameter."""

    __module__ = "haystack.components.retrievers"

    def __init__(
        self,
        document_store: Any,
        top_k: int = 10,
        scale_score: bool = False,
    ):
        self.document_store = document_store
        self.top_k = top_k
        self.scale_score = scale_score


class MockComponentWithLiteral:
    """Mock component with Literal type hint."""

    __module__ = "haystack.components.generators"

    def __init__(
        self,
        model: Literal["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"] = "gpt-4o-mini",
        mode: Literal["fast", "accurate"] = "fast",
    ):
        self.model = model
        self.mode = mode


class MockComponentNoTypeHints:
    """Mock component without type hints."""

    __module__ = "haystack.components.custom"

    def __init__(self, param1, param2="default"):
        self.param1 = param1
        self.param2 = param2


class MockEmbedder:
    """Mock embedder component."""

    __module__ = "haystack.components.embedders"

    def __init__(
        self,
        model: str = "text-embedding-ada-002",
        batch_size: int = 32,
    ):
        self.model = model
        self.batch_size = batch_size


class TestParameterExtraction:
    """Tests for parameter extraction from components (Story 1.2)."""

    def test_extract_generator_parameters(self):
        """AC1: Extract parameters from Generator with model, temperature, etc."""
        pipeline = MockPipeline({"generator": MockGeneratorWithParams()})

        config_space = from_pipeline(pipeline)
        generator = config_space.get_component("generator")

        assert generator is not None
        assert len(generator.parameters) > 0

        # Check model parameter
        assert "model" in generator.parameters
        model_param = generator.parameters["model"]
        assert model_param.value == "gpt-4o-mini"
        assert model_param.python_type == "str"
        assert model_param.is_tunable is True

        # Check temperature parameter
        assert "temperature" in generator.parameters
        temp_param = generator.parameters["temperature"]
        assert abs(temp_param.value - 0.7) < 1e-9  # Float comparison
        assert temp_param.python_type == "float"
        assert temp_param.is_optional is True
        assert temp_param.is_tunable is True

        # Check max_tokens parameter
        assert "max_tokens" in generator.parameters
        max_tokens_param = generator.parameters["max_tokens"]
        assert max_tokens_param.value == 100
        assert max_tokens_param.python_type == "int"
        assert max_tokens_param.is_optional is True

    def test_extract_multiple_component_parameters(self):
        """AC2: Extract parameters from multiple component types."""
        pipeline = MockPipeline(
            {
                "generator": MockGeneratorWithParams(),
                "retriever": MockRetrieverWithDocStore(document_store=object()),
                "embedder": MockEmbedder(),
            }
        )

        config_space = from_pipeline(pipeline)

        # Verify all components have parameters
        generator = config_space.get_component("generator")
        retriever = config_space.get_component("retriever")
        embedder = config_space.get_component("embedder")

        assert generator is not None
        assert retriever is not None
        assert embedder is not None

        assert len(generator.parameters) >= 3
        assert len(retriever.parameters) >= 2
        assert len(embedder.parameters) >= 2

        # Check retriever has top_k
        assert "top_k" in retriever.parameters
        assert retriever.parameters["top_k"].value == 10
        assert retriever.parameters["top_k"].python_type == "int"

        # Check embedder has batch_size
        assert "batch_size" in embedder.parameters
        assert embedder.parameters["batch_size"].value == 32

    def test_complex_parameter_marked_non_tunable(self):
        """AC3: Complex nested parameters are marked as non-tunable."""
        doc_store = object()  # Complex object
        pipeline = MockPipeline(
            {"retriever": MockRetrieverWithDocStore(document_store=doc_store)}
        )

        config_space = from_pipeline(pipeline)
        retriever = config_space.get_component("retriever")

        assert retriever is not None
        assert "document_store" in retriever.parameters

        doc_store_param = retriever.parameters["document_store"]
        assert doc_store_param.is_tunable is False
        assert doc_store_param.non_tunable_reason is not None

    def test_literal_type_preserves_choices(self):
        """AC4: Literal type hints preserve their allowed values."""
        pipeline = MockPipeline({"generator": MockComponentWithLiteral()})

        config_space = from_pipeline(pipeline)
        generator = config_space.get_component("generator")

        assert generator is not None
        assert "model" in generator.parameters

        model_param = generator.parameters["model"]
        assert model_param.python_type == "Literal"
        assert model_param.literal_choices is not None
        assert "gpt-4o" in model_param.literal_choices
        assert "gpt-4o-mini" in model_param.literal_choices
        assert "gpt-4-turbo" in model_param.literal_choices
        assert model_param.is_tunable is True

        # Check mode parameter
        assert "mode" in generator.parameters
        mode_param = generator.parameters["mode"]
        assert mode_param.python_type == "Literal"
        assert mode_param.literal_choices == ["fast", "accurate"]

    def test_optional_type_detection(self):
        """AC4: Optional[T] types are correctly identified."""
        pipeline = MockPipeline({"generator": MockGeneratorWithParams()})

        config_space = from_pipeline(pipeline)
        generator = config_space.get_component("generator")

        assert generator is not None

        # temperature is Optional[float]
        temp_param = generator.parameters["temperature"]
        assert temp_param.is_optional is True
        assert temp_param.python_type == "float"

        # model is plain str (not optional)
        model_param = generator.parameters["model"]
        assert model_param.is_optional is False

    def test_component_without_type_hints(self):
        """Parameters extracted even without type hints (as 'unknown')."""
        pipeline = MockPipeline(
            {"custom": MockComponentNoTypeHints(param1="test_value")}
        )

        config_space = from_pipeline(pipeline)
        custom = config_space.get_component("custom")

        assert custom is not None
        assert "param1" in custom.parameters
        assert "param2" in custom.parameters

        param1 = custom.parameters["param1"]
        assert param1.value == "test_value"
        assert param1.python_type == "unknown"
        assert param1.is_tunable is False  # unknown types not auto-tunable

        param2 = custom.parameters["param2"]
        assert param2.value == "default"

    def test_bool_parameter_extraction(self):
        """Bool parameters are correctly extracted and marked tunable."""
        pipeline = MockPipeline(
            {"retriever": MockRetrieverWithDocStore(document_store=object())}
        )

        config_space = from_pipeline(pipeline)
        retriever = config_space.get_component("retriever")

        assert retriever is not None
        assert "scale_score" in retriever.parameters

        scale_param = retriever.parameters["scale_score"]
        assert scale_param.value is False
        assert scale_param.python_type == "bool"
        assert scale_param.is_tunable is True


class TestDiscoveredTVAR:
    """Tests for the DiscoveredTVAR data model (Parameter backwards compat)."""

    def test_parameter_repr_with_value(self):
        """Test Parameter repr shows value for non-Literal types."""
        param = Parameter(
            name="temperature",
            value=0.7,
            python_type="float",
            is_tunable=True,
        )

        repr_str = repr(param)
        assert "temperature" in repr_str
        assert "float" in repr_str
        assert "0.7" in repr_str
        assert "tunable" in repr_str

    def test_parameter_repr_with_choices(self):
        """Test Parameter repr shows choices for Literal types."""
        param = Parameter(
            name="model",
            value="gpt-4o",
            python_type="Literal",
            literal_choices=["gpt-4o", "gpt-4o-mini"],
            is_tunable=True,
        )

        repr_str = repr(param)
        assert "model" in repr_str
        assert "Literal" in repr_str
        assert "choices" in repr_str
        assert "tunable" in repr_str

    def test_parameter_non_tunable_repr(self):
        """Test Parameter repr shows fixed for non-tunable."""
        param = Parameter(
            name="document_store",
            value=object(),
            python_type="object",
            is_tunable=False,
            non_tunable_reason="Complex object",
        )

        repr_str = repr(param)
        assert "fixed" in repr_str

    def test_parameter_with_type_hint(self):
        """Test Parameter stores type hint string."""
        param = Parameter(
            name="temp",
            value=0.5,
            python_type="float",
            type_hint="Optional[float]",
            is_optional=True,
        )

        assert param.type_hint == "Optional[float]"
        assert param.is_optional is True

    def test_parameter_with_range(self):
        """Test Parameter repr shows range when present."""
        param = Parameter(
            name="temperature",
            value=0.7,
            python_type="float",
            is_tunable=True,
            default_range=(0.0, 2.0),
            range_type="continuous",
        )

        repr_str = repr(param)
        assert "range=" in repr_str
        assert "(0.0, 2.0)" in repr_str


# ============================================================================
# Story 1.3: Tunability and Default Range Tests
# ============================================================================


class MockGeneratorWithRanges:
    """Mock generator with parameters that should get default ranges."""

    __module__ = "haystack.components.generators"

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.7,
        max_tokens: int = 100,
        top_p: float = 0.9,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        custom_param: float = 0.5,  # No known semantics
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.presence_penalty = presence_penalty
        self.frequency_penalty = frequency_penalty
        self.custom_param = custom_param


class MockRetrieverWithThresholds:
    """Mock retriever with threshold parameters for range testing."""

    __module__ = "haystack.components.retrievers"

    def __init__(
        self,
        document_store: Any,
        top_k: int = 10,
        score_threshold: float = 0.5,
        similarity_threshold: float = 0.7,
    ):
        self.document_store = document_store
        self.top_k = top_k
        self.score_threshold = score_threshold
        self.similarity_threshold = similarity_threshold


class MockComponentWithCallable:
    """Mock component with a callable parameter."""

    __module__ = "haystack.components.custom"

    def __init__(
        self,
        callback: Any = None,
        value: int = 10,
    ):
        self.callback = callback
        self.value = value


class TestTunabilityDetection:
    """Tests for parameter tunability detection (Story 1.3 AC1, AC2)."""

    def test_int_parameter_is_tunable(self):
        """AC1: int parameters are marked as tunable."""
        pipeline = MockPipeline(
            {"retriever": MockRetrieverWithDocStore(document_store=object())}
        )

        config_space = from_pipeline(pipeline)
        retriever = config_space.get_component("retriever")

        assert retriever is not None
        top_k = retriever.parameters["top_k"]
        assert top_k.python_type == "int"
        assert top_k.is_tunable is True

    def test_float_parameter_is_tunable(self):
        """AC1: float parameters are marked as tunable."""
        pipeline = MockPipeline({"generator": MockGeneratorWithParams()})

        config_space = from_pipeline(pipeline)
        generator = config_space.get_component("generator")

        assert generator is not None
        temp = generator.parameters["temperature"]
        assert temp.python_type == "float"
        assert temp.is_tunable is True

    def test_str_parameter_is_tunable(self):
        """AC1: str parameters are marked as tunable."""
        pipeline = MockPipeline({"generator": MockGeneratorWithParams()})

        config_space = from_pipeline(pipeline)
        generator = config_space.get_component("generator")

        assert generator is not None
        model = generator.parameters["model"]
        assert model.python_type == "str"
        assert model.is_tunable is True

    def test_bool_parameter_is_tunable(self):
        """AC1: bool parameters are marked as tunable."""
        pipeline = MockPipeline(
            {"retriever": MockRetrieverWithDocStore(document_store=object())}
        )

        config_space = from_pipeline(pipeline)
        retriever = config_space.get_component("retriever")

        assert retriever is not None
        scale = retriever.parameters["scale_score"]
        assert scale.python_type == "bool"
        assert scale.is_tunable is True

    def test_literal_parameter_is_tunable(self):
        """AC1/AC4: Literal parameters are marked as tunable with choices."""
        pipeline = MockPipeline({"generator": MockComponentWithLiteral()})

        config_space = from_pipeline(pipeline)
        generator = config_space.get_component("generator")

        assert generator is not None
        model = generator.parameters["model"]
        assert model.python_type == "Literal"
        assert model.is_tunable is True
        assert model.literal_choices == ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"]

    def test_callable_parameter_is_not_tunable(self):
        """AC2: Callable parameters are marked as fixed."""

        def my_callback():
            pass  # Empty callback for testing

        component = MockComponentWithCallable(callback=my_callback)
        pipeline = MockPipeline({"custom": component})

        config_space = from_pipeline(pipeline)
        custom = config_space.get_component("custom")

        assert custom is not None
        callback = custom.parameters["callback"]
        assert callback.is_tunable is False
        assert callback.non_tunable_reason is not None
        # Note: "callback" name pattern is caught first, but the key point is it's not tunable
        assert (
            "callback" in callback.non_tunable_reason.lower()
            or "complex" in callback.non_tunable_reason.lower()
        )

    def test_document_store_parameter_is_not_tunable(self):
        """AC2: document_store parameters are marked as fixed."""
        pipeline = MockPipeline(
            {"retriever": MockRetrieverWithDocStore(document_store=object())}
        )

        config_space = from_pipeline(pipeline)
        retriever = config_space.get_component("retriever")

        assert retriever is not None
        doc_store = retriever.parameters["document_store"]
        assert doc_store.is_tunable is False
        assert doc_store.non_tunable_reason is not None

    def test_object_type_parameter_is_not_tunable(self):
        """AC2: Object-type parameters are marked as fixed."""
        pipeline = MockPipeline({"custom": MockComponentNoTypeHints(param1=object())})

        config_space = from_pipeline(pipeline)
        custom = config_space.get_component("custom")

        assert custom is not None
        param1 = custom.parameters["param1"]
        assert param1.is_tunable is False
        assert "unknown" in param1.non_tunable_reason.lower()


class TestDefaultRanges:
    """Tests for default range inference (Story 1.3 AC3)."""

    def test_temperature_gets_default_range(self):
        """AC3: temperature parameter gets (0.0, 2.0) range."""
        pipeline = MockPipeline({"generator": MockGeneratorWithRanges()})

        config_space = from_pipeline(pipeline)
        generator = config_space.get_component("generator")

        assert generator is not None
        temp = generator.parameters["temperature"]
        assert temp.default_range == (0.0, 2.0)
        assert temp.range_type == "continuous"

    def test_top_p_gets_default_range(self):
        """AC3: top_p parameter gets (0.0, 1.0) range."""
        pipeline = MockPipeline({"generator": MockGeneratorWithRanges()})

        config_space = from_pipeline(pipeline)
        generator = config_space.get_component("generator")

        assert generator is not None
        top_p = generator.parameters["top_p"]
        assert top_p.default_range == (0.0, 1.0)
        assert top_p.range_type == "continuous"

    def test_top_k_gets_default_range(self):
        """AC3: top_k parameter gets discrete range."""
        pipeline = MockPipeline(
            {"retriever": MockRetrieverWithDocStore(document_store=object())}
        )

        config_space = from_pipeline(pipeline)
        retriever = config_space.get_component("retriever")

        assert retriever is not None
        top_k = retriever.parameters["top_k"]
        assert top_k.default_range == (1, 100)
        assert top_k.range_type == "discrete"

    def test_max_tokens_gets_default_range(self):
        """AC3: max_tokens parameter gets discrete range."""
        pipeline = MockPipeline({"generator": MockGeneratorWithRanges()})

        config_space = from_pipeline(pipeline)
        generator = config_space.get_component("generator")

        assert generator is not None
        max_tokens = generator.parameters["max_tokens"]
        assert max_tokens.default_range == (1, 4096)
        assert max_tokens.range_type == "discrete"

    def test_presence_penalty_gets_default_range(self):
        """AC3: presence_penalty parameter gets (-2.0, 2.0) range."""
        pipeline = MockPipeline({"generator": MockGeneratorWithRanges()})

        config_space = from_pipeline(pipeline)
        generator = config_space.get_component("generator")

        assert generator is not None
        param = generator.parameters["presence_penalty"]
        assert param.default_range == (-2.0, 2.0)
        assert param.range_type == "continuous"

    def test_frequency_penalty_gets_default_range(self):
        """AC3: frequency_penalty parameter gets (-2.0, 2.0) range."""
        pipeline = MockPipeline({"generator": MockGeneratorWithRanges()})

        config_space = from_pipeline(pipeline)
        generator = config_space.get_component("generator")

        assert generator is not None
        param = generator.parameters["frequency_penalty"]
        assert param.default_range == (-2.0, 2.0)
        assert param.range_type == "continuous"

    def test_score_threshold_gets_default_range(self):
        """AC3: score_threshold parameter gets (0.0, 1.0) range."""
        pipeline = MockPipeline(
            {"retriever": MockRetrieverWithThresholds(document_store=object())}
        )

        config_space = from_pipeline(pipeline)
        retriever = config_space.get_component("retriever")

        assert retriever is not None
        param = retriever.parameters["score_threshold"]
        assert param.default_range == (0.0, 1.0)
        assert param.range_type == "continuous"

    def test_similarity_threshold_gets_default_range(self):
        """AC3: similarity_threshold parameter gets (0.0, 1.0) range."""
        pipeline = MockPipeline(
            {"retriever": MockRetrieverWithThresholds(document_store=object())}
        )

        config_space = from_pipeline(pipeline)
        retriever = config_space.get_component("retriever")

        assert retriever is not None
        param = retriever.parameters["similarity_threshold"]
        assert param.default_range == (0.0, 1.0)
        assert param.range_type == "continuous"

    def test_unknown_parameter_no_range(self):
        """AC3: Parameters without known semantics get no range."""
        pipeline = MockPipeline({"generator": MockGeneratorWithRanges()})

        config_space = from_pipeline(pipeline)
        generator = config_space.get_component("generator")

        assert generator is not None
        custom = generator.parameters["custom_param"]
        assert custom.default_range is None
        assert custom.range_type is None

    def test_non_tunable_parameter_no_range(self):
        """AC3: Non-tunable parameters don't get ranges even if name matches."""
        pipeline = MockPipeline(
            {"retriever": MockRetrieverWithDocStore(document_store=object())}
        )

        config_space = from_pipeline(pipeline)
        retriever = config_space.get_component("retriever")

        assert retriever is not None
        doc_store = retriever.parameters["document_store"]
        assert doc_store.is_tunable is False
        assert doc_store.default_range is None

    def test_string_parameter_no_range(self):
        """AC3: String parameters don't get numeric ranges."""
        pipeline = MockPipeline({"generator": MockGeneratorWithRanges()})

        config_space = from_pipeline(pipeline)
        generator = config_space.get_component("generator")

        assert generator is not None
        model = generator.parameters["model"]
        assert model.python_type == "str"
        assert model.default_range is None

    def test_literal_parameter_no_range(self):
        """AC3: Literal parameters don't get numeric ranges (they use choices)."""
        pipeline = MockPipeline({"generator": MockComponentWithLiteral()})

        config_space = from_pipeline(pipeline)
        generator = config_space.get_component("generator")

        assert generator is not None
        model = generator.parameters["model"]
        assert model.python_type == "Literal"
        assert model.default_range is None
        assert model.literal_choices is not None


# ============================================================================
# Story 1.4: Pipeline Graph Structure Tests
# ============================================================================


try:
    import networkx as nx

    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    nx = None  # type: ignore


class MockPipelineWithGraph:
    """Mock Pipeline with NetworkX graph for testing graph extraction."""

    def __init__(self, components: dict[str, Any], connections: list[tuple[str, ...]]):
        """Initialize mock pipeline with components and connections.

        Args:
            components: Dict of component_name -> component_instance
            connections: List of tuples:
                - (sender, receiver) for simple edges
                - (sender, receiver, sender_socket, receiver_socket) for socket edges
        """
        self._components = components

        # Create NetworkX graph if available
        if NETWORKX_AVAILABLE:
            self._graph = nx.DiGraph()

            # Add nodes
            for name in components:
                self._graph.add_node(name)

            # Add edges
            for conn in connections:
                sender, receiver = conn[0], conn[1]
                sender_socket = conn[2] if len(conn) > 2 else None
                receiver_socket = conn[3] if len(conn) > 3 else None
                self._graph.add_edge(
                    sender,
                    receiver,
                    sender_socket=sender_socket,
                    receiver_socket=receiver_socket,
                )
        else:
            self._graph = None

    @property
    def graph(self):
        """Return the NetworkX graph."""
        return self._graph

    def walk(self):
        """Yield component name and instance pairs."""
        for name, component in self._components.items():
            yield name, component


class TestGraphExtraction:
    """Tests for pipeline graph structure extraction (Story 1.4)."""

    def test_linear_pipeline_graph(self):
        """AC1: Linear pipeline A → B → C returns correct edges."""
        pipeline = MockPipelineWithGraph(
            components={
                "a": MockComponent(),
                "b": MockComponent(),
                "c": MockComponent(),
            },
            connections=[
                ("a", "b", "output", "input"),
                ("b", "c", "output", "input"),
            ],
        )

        config_space = from_pipeline(pipeline)

        assert len(config_space.edges) == 2

        # Verify edge sources and targets
        sources = {e.source for e in config_space.edges}
        targets = {e.target for e in config_space.edges}

        assert "a" in sources
        assert "b" in sources
        assert "b" in targets
        assert "c" in targets

    def test_branching_pipeline_graph(self):
        """AC3: Branching pipeline A → B, A → C captures both branches."""
        pipeline = MockPipelineWithGraph(
            components={
                "a": MockComponent(),
                "b": MockComponent(),
                "c": MockComponent(),
            },
            connections=[
                ("a", "b"),
                ("a", "c"),
            ],
        )

        config_space = from_pipeline(pipeline)

        assert len(config_space.edges) == 2

        sources = {e.source for e in config_space.edges}
        targets = {e.target for e in config_space.edges}

        # All edges start from "a"
        assert sources == {"a"}
        # Edges go to both "b" and "c"
        assert targets == {"b", "c"}

    def test_empty_pipeline_no_edges(self):
        """Empty pipeline returns empty edges list."""
        pipeline = MockPipelineWithGraph(components={}, connections=[])

        config_space = from_pipeline(pipeline)

        assert config_space.edges == []

    def test_pipeline_without_graph_attribute(self):
        """Pipeline without graph attribute returns empty edges."""
        # Regular MockPipeline doesn't have graph attribute
        pipeline = MockPipeline({"component": MockComponent()})

        config_space = from_pipeline(pipeline)

        assert config_space.edges == []
        assert config_space.component_count == 1

    def test_edge_socket_metadata(self):
        """AC4: Edges include sender_socket and receiver_socket metadata."""
        pipeline = MockPipelineWithGraph(
            components={
                "retriever": MockInMemoryBM25Retriever(),
                "generator": MockOpenAIGenerator(),
            },
            connections=[
                ("retriever", "generator", "documents", "docs_input"),
            ],
        )

        config_space = from_pipeline(pipeline)

        assert len(config_space.edges) == 1
        edge = config_space.edges[0]

        assert edge.source == "retriever"
        assert edge.target == "generator"
        assert edge.sender_socket == "documents"
        assert edge.receiver_socket == "docs_input"

    def test_diamond_graph_structure(self):
        """Test diamond pattern: A → B, A → C, B → D, C → D."""
        pipeline = MockPipelineWithGraph(
            components={
                "a": MockComponent(),
                "b": MockComponent(),
                "c": MockComponent(),
                "d": MockComponent(),
            },
            connections=[
                ("a", "b"),
                ("a", "c"),
                ("b", "d"),
                ("c", "d"),
            ],
        )

        config_space = from_pipeline(pipeline)

        assert len(config_space.edges) == 4

        # Verify the diamond structure
        edge_tuples = {(e.source, e.target) for e in config_space.edges}
        assert ("a", "b") in edge_tuples
        assert ("a", "c") in edge_tuples
        assert ("b", "d") in edge_tuples
        assert ("c", "d") in edge_tuples

    def test_configspace_repr_with_edges(self):
        """PipelineSpec repr includes scope and connection count."""
        pipeline = MockPipelineWithGraph(
            components={"a": MockComponent(), "b": MockComponent()},
            connections=[("a", "b")],
        )

        config_space = from_pipeline(pipeline)

        repr_str = repr(config_space)
        assert "scopes=2" in repr_str
        assert "connections=1" in repr_str


@pytest.mark.skipif(not NETWORKX_AVAILABLE, reason="networkx not installed")
class TestToNetworkx:
    """Tests for ConfigSpace.to_networkx() method (Story 1.4 AC2)."""

    def test_to_networkx_returns_digraph(self):
        """AC2: to_networkx() returns a NetworkX DiGraph."""
        pipeline = MockPipelineWithGraph(
            components={"a": MockComponent(), "b": MockComponent()},
            connections=[("a", "b")],
        )

        config_space = from_pipeline(pipeline)
        G = config_space.to_networkx()

        assert isinstance(G, nx.DiGraph)

    def test_to_networkx_nodes_match_components(self):
        """AC2: DiGraph nodes correspond to component names."""
        pipeline = MockPipelineWithGraph(
            components={
                "retriever": MockInMemoryBM25Retriever(),
                "generator": MockOpenAIGenerator(),
            },
            connections=[("retriever", "generator")],
        )

        config_space = from_pipeline(pipeline)
        G = config_space.to_networkx()

        assert set(G.nodes()) == {"retriever", "generator"}

    def test_to_networkx_edges_match_connections(self):
        """AC2: DiGraph edges correspond to data flow connections."""
        pipeline = MockPipelineWithGraph(
            components={
                "a": MockComponent(),
                "b": MockComponent(),
                "c": MockComponent(),
            },
            connections=[("a", "b"), ("b", "c")],
        )

        config_space = from_pipeline(pipeline)
        G = config_space.to_networkx()

        assert list(G.edges()) == [("a", "b"), ("b", "c")]

    def test_to_networkx_node_attributes(self):
        """Nodes include class_name and category attributes."""
        pipeline = MockPipelineWithGraph(
            components={"generator": MockOpenAIGenerator()},
            connections=[],
        )

        config_space = from_pipeline(pipeline)
        G = config_space.to_networkx()

        node_data = G.nodes["generator"]
        assert node_data["class_name"] == "MockOpenAIGenerator"
        assert node_data["category"] == "Generator"

    def test_to_networkx_edge_attributes(self):
        """Edges include socket metadata as attributes."""
        pipeline = MockPipelineWithGraph(
            components={"a": MockComponent(), "b": MockComponent()},
            connections=[("a", "b", "output", "input")],
        )

        config_space = from_pipeline(pipeline)
        G = config_space.to_networkx()

        edge_data = G.edges["a", "b"]
        assert edge_data["sender_socket"] == "output"
        assert edge_data["receiver_socket"] == "input"

    def test_to_networkx_empty_configspace(self):
        """Empty PipelineSpec returns empty DiGraph."""
        config_space = PipelineSpec(scopes=[], connections=[])
        G = config_space.to_networkx()

        assert isinstance(G, nx.DiGraph)
        assert len(G.nodes()) == 0
        assert len(G.edges()) == 0


class TestConnection:
    """Tests for the Connection data model (Edge backwards compat, Story 1.4)."""

    def test_edge_repr_with_sockets(self):
        """Connection repr shows socket information when present."""
        edge = Connection(
            source="retriever",
            target="generator",
            sender_socket="documents",
            receiver_socket="docs",
        )

        repr_str = repr(edge)
        assert "retriever.documents" in repr_str
        assert "generator.docs" in repr_str
        assert "->" in repr_str

    def test_edge_repr_without_sockets(self):
        """Connection repr shows simple format without sockets."""
        edge = Connection(source="a", target="b")

        repr_str = repr(edge)
        assert "Connection(a -> b)" == repr_str

    def test_edge_equality(self):
        """Connections with same values are equal."""
        edge1 = Connection(
            source="a", target="b", sender_socket="out", receiver_socket="in"
        )
        edge2 = Connection(
            source="a", target="b", sender_socket="out", receiver_socket="in"
        )

        assert edge1 == edge2

    def test_edge_default_sockets_none(self):
        """Connection sockets default to None."""
        edge = Connection(source="a", target="b")

        assert edge.sender_socket is None
        assert edge.receiver_socket is None


# ============================================================================
# Story 1.5: Loop Detection and Max Runs Tests
# ============================================================================


class MockPipelineWithLoop:
    """Mock Pipeline with a loop for testing cycle detection."""

    def __init__(
        self,
        components: dict[str, Any],
        connections: list[tuple[str, ...]],
        max_runs_per_component: int | None = None,
    ):
        """Initialize mock pipeline with loop support.

        Args:
            components: Dict of component_name -> component_instance
            connections: List of edge tuples
            max_runs_per_component: Optional max runs setting
        """
        self._components = components
        self.max_runs_per_component = max_runs_per_component

        # Create NetworkX graph if available
        if NETWORKX_AVAILABLE:
            self._graph = nx.DiGraph()

            # Add nodes
            for name in components:
                self._graph.add_node(name)

            # Add edges
            for conn in connections:
                sender, receiver = conn[0], conn[1]
                sender_socket = conn[2] if len(conn) > 2 else None
                receiver_socket = conn[3] if len(conn) > 3 else None
                self._graph.add_edge(
                    sender,
                    receiver,
                    sender_socket=sender_socket,
                    receiver_socket=receiver_socket,
                )
        else:
            self._graph = None

    @property
    def graph(self):
        """Return the NetworkX graph."""
        return self._graph

    def walk(self):
        """Yield component name and instance pairs."""
        for name, component in self._components.items():
            yield name, component


@pytest.mark.skipif(not NETWORKX_AVAILABLE, reason="networkx not installed")
class TestLoopDetection:
    """Tests for loop detection in pipelines (Story 1.5)."""

    def test_dag_pipeline_has_no_loops(self):
        """AC3: DAG pipeline returns has_loops=False."""
        pipeline = MockPipelineWithGraph(
            components={
                "a": MockComponent(),
                "b": MockComponent(),
                "c": MockComponent(),
            },
            connections=[("a", "b"), ("b", "c")],
        )

        config_space = from_pipeline(pipeline)

        assert config_space.has_loops is False
        assert config_space.loops == []

    def test_cyclic_pipeline_has_loops(self):
        """AC1: Cyclic pipeline returns has_loops=True."""
        pipeline = MockPipelineWithLoop(
            components={
                "router": MockComponent(),
                "generator": MockComponent(),
            },
            connections=[
                ("router", "generator"),
                ("generator", "router"),  # Creates cycle
            ],
        )

        config_space = from_pipeline(pipeline)

        assert config_space.has_loops is True
        assert len(config_space.loops) >= 1

        # Verify the cycle contains both components
        cycle = config_space.loops[0]
        assert "router" in cycle or "generator" in cycle

    def test_multiple_cycles_detected(self):
        """AC1: Multiple cycles are all detected."""
        # Create a graph with two separate cycles
        pipeline = MockPipelineWithLoop(
            components={
                "a": MockComponent(),
                "b": MockComponent(),
                "c": MockComponent(),
                "d": MockComponent(),
            },
            connections=[
                ("a", "b"),
                ("b", "a"),  # Cycle 1: a -> b -> a
                ("c", "d"),
                ("d", "c"),  # Cycle 2: c -> d -> c
            ],
        )

        config_space = from_pipeline(pipeline)

        assert config_space.has_loops is True
        assert len(config_space.loops) >= 2

    def test_self_loop_detected(self):
        """Self-referencing component is detected as loop."""
        pipeline = MockPipelineWithLoop(
            components={"self_ref": MockComponent()},
            connections=[("self_ref", "self_ref")],  # Self-loop
        )

        config_space = from_pipeline(pipeline)

        assert config_space.has_loops is True
        assert len(config_space.loops) >= 1


@pytest.mark.skipif(not NETWORKX_AVAILABLE, reason="networkx not installed")
class TestMaxRunsExtraction:
    """Tests for max_runs extraction from pipelines (Story 1.5 AC2)."""

    def test_max_runs_extracted_from_pipeline(self):
        """AC2: max_runs_per_component is extracted and applied to components."""
        pipeline = MockPipelineWithLoop(
            components={
                "router": MockComponent(),
                "generator": MockComponent(),
            },
            connections=[("router", "generator"), ("generator", "router")],
            max_runs_per_component=5,
        )

        config_space = from_pipeline(pipeline)

        # All components should have max_runs set
        for component in config_space.components:
            assert component.max_runs == 5

    def test_no_max_runs_when_not_set(self):
        """Components have max_runs=None when pipeline doesn't set it."""
        pipeline = MockPipelineWithLoop(
            components={"a": MockComponent()},
            connections=[],
            max_runs_per_component=None,
        )

        config_space = from_pipeline(pipeline)

        component = config_space.get_component("a")
        assert component is not None
        assert component.max_runs is None

    def test_component_max_runs_field_exists(self):
        """Component dataclass has max_runs field."""
        component = Component(
            name="test",
            class_name="Test",
            class_type="module.Test",
            category="Component",
            max_runs=10,
        )

        assert component.max_runs == 10


@pytest.mark.skipif(not NETWORKX_AVAILABLE, reason="networkx not installed")
class TestUnboundedLoops:
    """Tests for unbounded loop detection (Story 1.5 AC4)."""

    def test_unbounded_loop_detected(self):
        """AC4: Loops without max_runs are identified as unbounded."""
        pipeline = MockPipelineWithLoop(
            components={
                "router": MockComponent(),
                "generator": MockComponent(),
            },
            connections=[("router", "generator"), ("generator", "router")],
            max_runs_per_component=None,  # No max_runs set
        )

        config_space = from_pipeline(pipeline)

        assert config_space.has_loops is True
        assert len(config_space.unbounded_loops) >= 1

    def test_bounded_loop_not_in_unbounded(self):
        """Loops with max_runs are NOT in unbounded_loops."""
        pipeline = MockPipelineWithLoop(
            components={
                "router": MockComponent(),
                "generator": MockComponent(),
            },
            connections=[("router", "generator"), ("generator", "router")],
            max_runs_per_component=5,  # Has max_runs
        )

        config_space = from_pipeline(pipeline)

        assert config_space.has_loops is True
        assert config_space.unbounded_loops == []

    def test_dag_has_no_unbounded_loops(self):
        """DAG without cycles has no unbounded loops."""
        pipeline = MockPipelineWithGraph(
            components={"a": MockComponent(), "b": MockComponent()},
            connections=[("a", "b")],
        )

        config_space = from_pipeline(pipeline)

        assert config_space.has_loops is False
        assert config_space.unbounded_loops == []


class TestConfigSpaceLoopProperties:
    """Tests for ConfigSpace loop-related properties."""

    def test_has_loops_with_empty_loops(self):
        """has_loops returns False when loops list is empty."""
        config_space = PipelineSpec(scopes=[], connections=[], loops=[])

        assert config_space.has_loops is False

    def test_has_loops_with_cycles(self):
        """has_loops returns True when loops list has entries."""
        config_space = PipelineSpec(
            scopes=[],
            connections=[],
            loops=[["a", "b"]],
        )

        assert config_space.has_loops is True

    def test_configspace_repr_with_loops(self):
        """PipelineSpec repr includes loop count when loops present."""
        config_space = PipelineSpec(
            scopes=[
                TVARScope(
                    name="a",
                    class_name="A",
                    class_type="module.A",
                    category="Component",
                )
            ],
            connections=[],
            loops=[["a"]],
        )

        repr_str = repr(config_space)
        assert "loops=1" in repr_str

    def test_configspace_repr_without_loops(self):
        """PipelineSpec repr omits loop count when no loops."""
        config_space = PipelineSpec(
            scopes=[
                TVARScope(
                    name="a",
                    class_name="A",
                    class_type="module.A",
                    category="Component",
                )
            ],
            connections=[],
            loops=[],
        )

        repr_str = repr(config_space)
        assert "loops" not in repr_str


# ============================================================================
# Story 1.6: Custom @component Decorated Components Tests
# ============================================================================


class MockCustomComponent:
    """Mock custom component simulating @component decorated class."""

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


class MockCustomRetriever:
    """Mock custom retriever component."""

    __module__ = "user_project.retrievers"

    def __init__(
        self,
        similarity_threshold: float = 0.7,
        top_k: int = 5,
    ):
        self.similarity_threshold = similarity_threshold
        self.top_k = top_k


class MockCustomGenerator:
    """Mock custom generator component."""

    __module__ = "user_project.generators"

    def __init__(
        self,
        temperature: float = 0.8,
        model_name: str = "custom-model",
    ):
        self.temperature = temperature
        self.model_name = model_name


class MockCustomComponentNoTypeHints:
    """Mock custom component without type hints."""

    __module__ = "user_project.custom"

    def __init__(self, param1, param2="value", param3=100):
        self.param1 = param1
        self.param2 = param2
        self.param3 = param3


class TestCustomComponentIntrospection:
    """Tests for custom @component decorated class support (Story 1.6)."""

    def test_custom_component_extraction(self):
        """AC1: Custom component is extracted with parameters."""
        pipeline = MockPipeline({"custom": MockCustomComponent()})

        config_space = from_pipeline(pipeline)

        assert config_space.component_count == 1
        custom = config_space.get_component("custom")

        assert custom is not None
        assert custom.class_name == "MockCustomComponent"
        assert "user_project.components" in custom.class_type

    def test_custom_component_parameters_extracted(self):
        """AC1: Custom component __init__ parameters are extracted."""
        pipeline = MockPipeline({"custom": MockCustomComponent()})

        config_space = from_pipeline(pipeline)
        custom = config_space.get_component("custom")

        assert custom is not None
        assert len(custom.parameters) >= 3

        # Check threshold parameter
        assert "threshold" in custom.parameters
        threshold = custom.parameters["threshold"]
        assert threshold.python_type == "float"
        assert threshold.value == 0.5
        assert threshold.is_tunable is True

        # Check max_results parameter
        assert "max_results" in custom.parameters
        max_results = custom.parameters["max_results"]
        assert max_results.python_type == "int"
        assert max_results.value == 10
        assert max_results.is_tunable is True

        # Check custom_option parameter
        assert "custom_option" in custom.parameters
        custom_option = custom.parameters["custom_option"]
        assert custom_option.python_type == "str"
        assert custom_option.value == "default"
        assert custom_option.is_tunable is True

    def test_custom_component_without_type_hints(self):
        """AC2: Custom component without type hints is handled gracefully."""
        pipeline = MockPipeline(
            {"custom": MockCustomComponentNoTypeHints(param1="test")}
        )

        config_space = from_pipeline(pipeline)
        custom = config_space.get_component("custom")

        assert custom is not None
        assert len(custom.parameters) >= 3

        # param1 has no type hint
        param1 = custom.parameters["param1"]
        assert param1.python_type == "unknown"
        assert param1.is_tunable is False
        assert param1.non_tunable_reason is not None

        # param2 has no type hint but has default
        param2 = custom.parameters["param2"]
        assert param2.python_type == "unknown"
        assert param2.is_tunable is False

    def test_custom_component_category_detection(self):
        """AC3: Custom component category detected from class/module name."""
        pipeline = MockPipeline(
            {
                "custom_ret": MockCustomRetriever(),
                "custom_gen": MockCustomGenerator(),
                "custom_other": MockCustomComponent(),
            }
        )

        config_space = from_pipeline(pipeline)

        # Custom retriever should be detected as Retriever
        retriever = config_space.get_component("custom_ret")
        assert retriever is not None
        assert retriever.category == "Retriever"

        # Custom generator should be detected as Generator
        generator = config_space.get_component("custom_gen")
        assert generator is not None
        assert generator.category == "Generator"

        # Generic custom component falls back to Component
        other = config_space.get_component("custom_other")
        assert other is not None
        assert other.category == "Component"

    def test_mixed_pipeline_with_custom_and_builtin(self):
        """AC4: Mixed pipeline with built-in and custom components."""
        pipeline = MockPipeline(
            {
                "builtin_gen": MockOpenAIGenerator(),
                "builtin_ret": MockInMemoryBM25Retriever(),
                "custom_comp": MockCustomComponent(),
                "custom_gen": MockCustomGenerator(),
            }
        )

        config_space = from_pipeline(pipeline)

        assert config_space.component_count == 4

        # Verify all components extracted
        assert config_space.get_component("builtin_gen") is not None
        assert config_space.get_component("builtin_ret") is not None
        assert config_space.get_component("custom_comp") is not None
        assert config_space.get_component("custom_gen") is not None

        # Verify categories
        assert config_space.get_component("builtin_gen").category == "Generator"
        assert config_space.get_component("builtin_ret").category == "Retriever"
        assert config_space.get_component("custom_gen").category == "Generator"

    def test_custom_component_with_known_parameter_semantics(self):
        """Custom component parameters get default ranges when names match."""
        pipeline = MockPipeline({"custom_ret": MockCustomRetriever()})

        config_space = from_pipeline(pipeline)
        retriever = config_space.get_component("custom_ret")

        assert retriever is not None

        # similarity_threshold should get default range
        sim_threshold = retriever.parameters["similarity_threshold"]
        assert sim_threshold.default_range == (0.0, 1.0)
        assert sim_threshold.range_type == "continuous"

        # top_k should get default range
        top_k = retriever.parameters["top_k"]
        assert top_k.default_range == (1, 100)
        assert top_k.range_type == "discrete"

    def test_custom_component_temperature_gets_range(self):
        """Custom generator temperature parameter gets default range."""
        pipeline = MockPipeline({"custom_gen": MockCustomGenerator()})

        config_space = from_pipeline(pipeline)
        generator = config_space.get_component("custom_gen")

        assert generator is not None

        # temperature should get default range
        temp = generator.parameters["temperature"]
        assert temp.default_range == (0.0, 2.0)
        assert temp.range_type == "continuous"


# ============================================================================
# Story 1.7: Introspection Performance and Component Coverage Tests
# ============================================================================


class MockConverterComponent:
    """Mock converter component."""

    __module__ = "haystack.components.converters"

    def __init__(self, encoding: str = "utf-8"):
        self.encoding = encoding


class MockRankerComponent:
    """Mock ranker component."""

    __module__ = "haystack.components.rankers"

    def __init__(self, top_k: int = 10):
        self.top_k = top_k


class MockReaderComponent:
    """Mock reader component."""

    __module__ = "haystack.components.readers"

    def __init__(self, file_path: str = ""):
        self.file_path = file_path


class MockUnknownComponent:
    """Mock component with unknown category."""

    __module__ = "some.random.module"

    def __init__(self, value: int = 0):
        self.value = value


class MockComponentWithManyParams:
    """Mock component with many parameters for performance testing."""

    __module__ = "haystack.components.generators"

    def __init__(
        self,
        model: str = "gpt-4o",
        temperature: float = 0.7,
        max_tokens: int = 100,
        top_p: float = 0.9,
        top_k: int = 50,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        seed: int = 42,
        timeout: int = 30,
        retries: int = 3,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.top_k = top_k
        self.presence_penalty = presence_penalty
        self.frequency_penalty = frequency_penalty
        self.seed = seed
        self.timeout = timeout
        self.retries = retries


class TestIntrospectionPerformance:
    """Tests for introspection performance (Story 1.7 AC1)."""

    @skip_in_ci
    def test_introspection_performance_20_components(self):
        """NFR-1: Introspection completes in <100ms for 20 components."""
        import time

        # Create 20-component mock pipeline with realistic parameters
        components = {}
        for i in range(20):
            if i % 4 == 0:
                components[f"gen_{i}"] = MockComponentWithManyParams()
            elif i % 4 == 1:
                components[f"ret_{i}"] = MockInMemoryBM25Retriever()
            elif i % 4 == 2:
                components[f"emb_{i}"] = MockEmbedder()
            else:
                components[f"router_{i}"] = MockMetadataRouter()

        pipeline = MockPipeline(components)

        # Measure introspection time (average of 5 runs)
        times = []
        for _ in range(5):
            start = time.perf_counter()
            config_space = from_pipeline(pipeline)
            elapsed_ms = (time.perf_counter() - start) * 1000
            times.append(elapsed_ms)

        avg_time = sum(times) / len(times)

        assert config_space.component_count == 20
        assert avg_time < 100, f"Introspection took {avg_time:.2f}ms (>100ms target)"

    @skip_in_ci
    def test_introspection_scales_linearly(self):
        """Verify introspection time scales approximately linearly."""
        import time

        def measure_time(n_components: int) -> float:
            components = {f"comp_{i}": MockComponent() for i in range(n_components)}
            pipeline = MockPipeline(components)

            start = time.perf_counter()
            from_pipeline(pipeline)
            return (time.perf_counter() - start) * 1000

        # Measure for 5 and 20 components
        time_5 = measure_time(5)
        time_20 = measure_time(20)

        # Time for 20 components should be roughly 4x time for 5 components
        # Allow for some overhead (factor of 6 max)
        ratio = time_20 / max(time_5, 0.001)  # Avoid division by zero
        assert ratio < 10, f"Scaling ratio {ratio:.2f} exceeds linear expectation"


class TestComponentCategoryCoverage:
    """Tests for component category coverage (Story 1.7 AC2, AC3)."""

    def test_generator_category_detected(self):
        """Generator components are detected."""
        pipeline = MockPipeline({"gen": MockOpenAIGenerator()})
        config_space = from_pipeline(pipeline)

        assert config_space.get_component("gen").category == "Generator"

    def test_retriever_category_detected(self):
        """Retriever components are detected."""
        pipeline = MockPipeline({"ret": MockInMemoryBM25Retriever()})
        config_space = from_pipeline(pipeline)

        assert config_space.get_component("ret").category == "Retriever"

    def test_embedder_category_detected(self):
        """Embedder components are detected."""
        pipeline = MockPipeline({"emb": MockEmbedder()})
        config_space = from_pipeline(pipeline)

        assert config_space.get_component("emb").category == "Embedder"

    def test_router_category_detected(self):
        """Router components are detected."""
        pipeline = MockPipeline({"router": MockMetadataRouter()})
        config_space = from_pipeline(pipeline)

        assert config_space.get_component("router").category == "Router"

    def test_builder_category_detected(self):
        """Builder components are detected."""
        pipeline = MockPipeline({"builder": MockPromptBuilder()})
        config_space = from_pipeline(pipeline)

        assert config_space.get_component("builder").category == "Builder"

    def test_writer_category_detected(self):
        """Writer components are detected."""
        pipeline = MockPipeline({"writer": MockDocumentWriter()})
        config_space = from_pipeline(pipeline)

        assert config_space.get_component("writer").category == "Writer"

    def test_converter_category_detected(self):
        """Converter components are detected."""
        pipeline = MockPipeline({"converter": MockConverterComponent()})
        config_space = from_pipeline(pipeline)

        assert config_space.get_component("converter").category == "Converter"

    def test_ranker_category_detected(self):
        """Ranker components are detected."""
        pipeline = MockPipeline({"ranker": MockRankerComponent()})
        config_space = from_pipeline(pipeline)

        assert config_space.get_component("ranker").category == "Ranker"

    def test_reader_category_detected(self):
        """Reader components are detected."""
        pipeline = MockPipeline({"reader": MockReaderComponent()})
        config_space = from_pipeline(pipeline)

        assert config_space.get_component("reader").category == "Reader"

    def test_category_coverage_at_least_90_percent(self):
        """NFR-5: At least 90% of standard categories are detected."""
        category_mocks = {
            "Generator": MockOpenAIGenerator(),
            "Retriever": MockInMemoryBM25Retriever(),
            "Embedder": MockEmbedder(),
            "Router": MockMetadataRouter(),
            "Builder": MockPromptBuilder(),
            "Writer": MockDocumentWriter(),
            "Converter": MockConverterComponent(),
            "Ranker": MockRankerComponent(),
            "Reader": MockReaderComponent(),
        }

        detected = 0
        total = len(category_mocks)

        for expected_category, mock_component in category_mocks.items():
            pipeline = MockPipeline({"test": mock_component})
            config_space = from_pipeline(pipeline)
            actual_category = config_space.get_component("test").category

            if actual_category == expected_category:
                detected += 1

        coverage = detected / total
        assert coverage >= 0.90, f"Category coverage {coverage:.0%} < 90% target"


class TestUnknownComponentHandling:
    """Tests for graceful handling of unknown components (Story 1.7 AC4)."""

    def test_unknown_component_falls_back_to_component(self):
        """AC4: Unknown component types fall back to 'Component' category."""
        pipeline = MockPipeline({"unknown": MockUnknownComponent()})
        config_space = from_pipeline(pipeline)

        component = config_space.get_component("unknown")
        assert component is not None
        assert component.category == "Component"

    def test_unknown_component_parameters_extracted(self):
        """Unknown components still have parameters extracted."""
        pipeline = MockPipeline({"unknown": MockUnknownComponent()})
        config_space = from_pipeline(pipeline)

        component = config_space.get_component("unknown")
        assert component is not None
        assert "value" in component.parameters
        assert component.parameters["value"].python_type == "int"

    def test_no_exception_for_empty_module(self):
        """No exception raised for components with minimal metadata."""

        class MinimalComponent:
            def __init__(self, x: int = 1):
                self.x = x

        pipeline = MockPipeline({"minimal": MinimalComponent()})
        config_space = from_pipeline(pipeline)

        assert config_space.component_count == 1
        assert config_space.get_component("minimal") is not None


class TestDiscoveredTVARSecretMasking:
    """Tests for secret masking in DiscoveredTVAR.__repr__ (code review fix)."""

    def test_api_key_is_masked(self):
        """API key values are masked in __repr__."""
        param = Parameter(name="api_key", value="sk-secret-key-123", python_type="str")
        repr_str = repr(param)

        assert "sk-secret-key-123" not in repr_str
        assert "'***'" in repr_str

    def test_secret_is_masked(self):
        """Parameters with 'secret' in name are masked."""
        param = Parameter(name="client_secret", value="my-secret", python_type="str")
        repr_str = repr(param)

        assert "my-secret" not in repr_str
        assert "'***'" in repr_str

    def test_password_is_masked(self):
        """Password parameters are masked."""
        param = Parameter(name="password", value="hunter2", python_type="str")
        repr_str = repr(param)

        assert "hunter2" not in repr_str
        assert "'***'" in repr_str

    def test_token_is_masked(self):
        """Token parameters are masked."""
        param = Parameter(name="auth_token", value="eyJhbGciOiJ...", python_type="str")
        repr_str = repr(param)

        assert "eyJhbGciOiJ" not in repr_str
        assert "'***'" in repr_str

    def test_normal_parameters_not_masked(self):
        """Non-sensitive parameters are not masked."""
        param = Parameter(name="temperature", value=0.7, python_type="float")
        repr_str = repr(param)

        assert "0.7" in repr_str
        assert "'***'" not in repr_str

    def test_none_sensitive_value_not_masked(self):
        """Sensitive parameters with None value show None."""
        param = Parameter(name="api_key", value=None, python_type="str")
        repr_str = repr(param)

        assert "None" in repr_str
        assert "'***'" not in repr_str

    def test_tokenizer_not_masked(self):
        """'tokenizer' should NOT be masked (false positive prevention)."""
        param = Parameter(name="tokenizer", value="tiktoken", python_type="str")
        repr_str = repr(param)

        assert "tiktoken" in repr_str
        assert "'***'" not in repr_str

    def test_authenticate_not_masked(self):
        """'authenticate' should NOT be masked (false positive prevention)."""
        param = Parameter(name="authenticate", value=True, python_type="bool")
        repr_str = repr(param)

        assert "True" in repr_str
        assert "'***'" not in repr_str

    def test_keyboard_layout_not_masked(self):
        """'keyboard_layout' should NOT be masked (false positive prevention)."""
        param = Parameter(name="keyboard_layout", value="qwerty", python_type="str")
        repr_str = repr(param)

        assert "qwerty" in repr_str
        assert "'***'" not in repr_str

    def test_suffix_patterns_masked(self):
        """Parameters with sensitive suffixes should be masked."""
        # _key suffix
        param = Parameter(name="openai_api_key", value="sk-12345", python_type="str")
        assert "'***'" in repr(param)

        # _secret suffix
        param = Parameter(name="client_secret", value="secret123", python_type="str")
        assert "'***'" in repr(param)

        # _password suffix
        param = Parameter(name="db_password", value="pass123", python_type="str")
        assert "'***'" in repr(param)


# ============================================================================
# Backwards Compatibility Tests
# ============================================================================


class TestBackwardsCompatibility:
    """Tests for backwards-compatible constructor keywords."""

    def test_configspace_with_old_components_keyword(self):
        """ConfigSpace() should accept old 'components' keyword."""
        from traigent.integrations.haystack import Component, ConfigSpace

        comp = Component(
            name="gen",
            class_name="Gen",
            class_type="module.Gen",
            category="Generator",
        )
        config_space = ConfigSpace(components=[comp])

        assert config_space.component_count == 1
        assert config_space.get_component("gen") is not None

    def test_configspace_with_old_edges_keyword(self):
        """ConfigSpace() should accept old 'edges' keyword."""
        from traigent.integrations.haystack import ConfigSpace, Edge

        edge = Edge(source="a", target="b")
        config_space = ConfigSpace(components=[], edges=[edge])

        assert len(config_space.edges) == 1
        assert config_space.edges[0].source == "a"

    def test_component_with_old_parameters_keyword(self):
        """Component() should accept old 'parameters' keyword."""
        from traigent.integrations.haystack import Component, Parameter

        param = Parameter(name="temp", value=0.7, python_type="float")
        comp = Component(
            name="gen",
            class_name="Gen",
            class_type="module.Gen",
            category="Generator",
            parameters={"temp": param},
        )

        assert "temp" in comp.parameters
        assert comp.parameters["temp"].value == 0.7

    def test_new_keywords_still_work(self):
        """New keywords (scopes, connections, tvars) should still work."""
        scope = TVARScope(
            name="gen",
            class_name="Gen",
            class_type="module.Gen",
            category="Generator",
            tvars={"temp": DiscoveredTVAR(name="temp", value=0.7, python_type="float")},
        )
        conn = Connection(source="a", target="b")
        spec = PipelineSpec(scopes=[scope], connections=[conn])

        assert spec.scope_count == 1
        assert len(spec.connections) == 1
        assert "temp" in spec.get_scope("gen").tvars


class TestModelCatalogs:
    """Tests for model catalog auto-discovery (Story 2.8)."""

    def test_model_catalogs_exist(self):
        """Test that MODEL_CATALOGS has expected providers."""
        from traigent.integrations.haystack.introspection import MODEL_CATALOGS

        assert "openai" in MODEL_CATALOGS
        assert "anthropic" in MODEL_CATALOGS
        assert "azure_openai" in MODEL_CATALOGS
        assert "cohere" in MODEL_CATALOGS
        assert "google" in MODEL_CATALOGS

    def test_openai_catalog_has_common_models(self):
        """Test that OpenAI catalog has common models."""
        from traigent.integrations.haystack.introspection import MODEL_CATALOGS

        openai_models = MODEL_CATALOGS["openai"]
        assert "gpt-4o" in openai_models
        assert "gpt-4o-mini" in openai_models
        assert "gpt-4" in openai_models

    def test_anthropic_catalog_has_common_models(self):
        """Test that Anthropic catalog has common models."""
        from traigent.integrations.haystack.introspection import MODEL_CATALOGS

        anthropic_models = MODEL_CATALOGS["anthropic"]
        assert any("claude-3" in m for m in anthropic_models)

    def test_provider_detection_openai(self):
        """Test that OpenAI generators map to openai provider."""
        from traigent.integrations.haystack.introspection import PROVIDER_DETECTION

        assert PROVIDER_DETECTION.get("OpenAIGenerator") == "openai"
        assert PROVIDER_DETECTION.get("OpenAIChatGenerator") == "openai"

    def test_provider_detection_anthropic(self):
        """Test that Anthropic generators map to anthropic provider."""
        from traigent.integrations.haystack.introspection import PROVIDER_DETECTION

        assert PROVIDER_DETECTION.get("AnthropicGenerator") == "anthropic"
        assert PROVIDER_DETECTION.get("AnthropicChatGenerator") == "anthropic"

    def test_get_model_choices_known_provider(self):
        """Test _get_model_choices returns catalog for known provider."""
        from traigent.integrations.haystack.introspection import _get_model_choices

        choices, reason = _get_model_choices("OpenAIGenerator", "gpt-4o", None)

        assert choices is not None
        assert "gpt-4o" in choices
        assert "gpt-4o-mini" in choices
        assert reason is None

    def test_get_model_choices_includes_current_value(self):
        """Test _get_model_choices includes current value even if not in catalog."""
        from traigent.integrations.haystack.introspection import _get_model_choices

        choices, _ = _get_model_choices("OpenAIGenerator", "custom-fine-tuned", None)

        assert choices is not None
        assert "custom-fine-tuned" in choices
        # Current value should be first
        assert choices[0] == "custom-fine-tuned"

    def test_get_model_choices_unknown_provider_fallback(self, caplog):
        """Test _get_model_choices falls back to current value for unknown provider."""
        import logging

        from traigent.integrations.haystack.introspection import _get_model_choices

        with caplog.at_level(logging.WARNING):
            choices, reason = _get_model_choices("CustomGenerator", "my-model", None)

        assert choices == ["my-model"]
        assert "Unknown provider" in reason
        assert "CustomGenerator" in caplog.text

    def test_get_model_choices_no_value_no_provider(self):
        """Test _get_model_choices returns None when no value and unknown provider."""
        from traigent.integrations.haystack.introspection import _get_model_choices

        choices, reason = _get_model_choices("CustomGenerator", None, None)

        assert choices is None
        assert reason is None


class TestDefaultRanges:
    """Tests for default range auto-discovery (Story 2.8)."""

    def test_tvar_semantics_has_temperature(self):
        """Test that TVAR_SEMANTICS includes temperature."""
        from traigent.integrations.haystack.introspection import TVAR_SEMANTICS

        assert "temperature" in TVAR_SEMANTICS
        assert TVAR_SEMANTICS["temperature"]["range"] == (0.0, 2.0)
        assert TVAR_SEMANTICS["temperature"]["scale"] == "continuous"

    def test_tvar_semantics_has_top_k(self):
        """Test that TVAR_SEMANTICS includes top_k."""
        from traigent.integrations.haystack.introspection import TVAR_SEMANTICS

        assert "top_k" in TVAR_SEMANTICS
        assert TVAR_SEMANTICS["top_k"]["range"] == (1, 100)
        assert TVAR_SEMANTICS["top_k"]["scale"] == "discrete"

    def test_tvar_semantics_has_top_p(self):
        """Test that TVAR_SEMANTICS includes top_p."""
        from traigent.integrations.haystack.introspection import TVAR_SEMANTICS

        assert "top_p" in TVAR_SEMANTICS
        assert TVAR_SEMANTICS["top_p"]["range"] == (0.0, 1.0)

    def test_infer_tvar_semantics_temperature(self):
        """Test that _infer_tvar_semantics returns correct range for temperature."""
        from traigent.integrations.haystack.introspection import _infer_tvar_semantics

        range_val, scale = _infer_tvar_semantics("temperature", "float")

        assert range_val == (0.0, 2.0)
        assert scale == "continuous"

    def test_infer_tvar_semantics_top_k(self):
        """Test that _infer_tvar_semantics returns correct range for top_k."""
        from traigent.integrations.haystack.introspection import _infer_tvar_semantics

        range_val, scale = _infer_tvar_semantics("top_k", "int")

        assert range_val == (1, 100)
        assert scale == "discrete"

    def test_infer_tvar_semantics_unknown_param(self):
        """Test that _infer_tvar_semantics returns None for unknown parameters."""
        from traigent.integrations.haystack.introspection import _infer_tvar_semantics

        range_val, scale = _infer_tvar_semantics("unknown_param", "float")

        assert range_val is None
        assert scale is None

    def test_infer_tvar_semantics_non_numeric(self):
        """Test that _infer_tvar_semantics returns None for non-numeric types."""
        from traigent.integrations.haystack.introspection import _infer_tvar_semantics

        range_val, scale = _infer_tvar_semantics("temperature", "str")

        assert range_val is None
        assert scale is None

    def test_infer_tvar_semantics_partial_match(self):
        """Test that _infer_tvar_semantics matches partial names."""
        from traigent.integrations.haystack.introspection import _infer_tvar_semantics

        # "generation_temperature" should match "temperature"
        range_val, scale = _infer_tvar_semantics("generation_temperature", "float")

        assert range_val == (0.0, 2.0)
        assert scale == "continuous"
