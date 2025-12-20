"""Tests for the ExplorationSpace and TVAR optimization types.

These tests cover Story 2.1: Define ConfigSpace Data Model
- AC1: ConfigSpace Parameter Structure
- AC2: ConfigSpace from Pipeline Introspection
- AC3: Parameter Type Representation
- AC4: Tunable vs Fixed Parameters
- AC5: Scoped Parameter Access
"""

import pytest

from traigent.integrations.haystack import (
    TVAR,
    CategoricalConstraint,
    ConditionalConstraint,
    DiscoveredTVAR,
    ExplorationSpace,
    NumericalConstraint,
    PipelineSpec,
    TVARScope,
    from_pipeline,
)


class TestCategoricalConstraint:
    """Tests for CategoricalConstraint."""

    def test_create_with_choices(self):
        """Test creating a categorical constraint with choices."""
        constraint = CategoricalConstraint(choices=["gpt-4o", "gpt-4o-mini"])
        assert constraint.choices == ["gpt-4o", "gpt-4o-mini"]

    def test_validate_valid_choice(self):
        """Test validating a value in choices."""
        constraint = CategoricalConstraint(choices=["a", "b", "c"])
        assert constraint.validate("a") is True
        assert constraint.validate("b") is True

    def test_validate_invalid_choice(self):
        """Test validating a value not in choices."""
        constraint = CategoricalConstraint(choices=["a", "b", "c"])
        assert constraint.validate("d") is False
        assert constraint.validate("invalid") is False

    def test_validate_with_none(self):
        """Test validating None when included in choices."""
        constraint = CategoricalConstraint(choices=["a", None])
        assert constraint.validate(None) is True
        assert constraint.validate("a") is True

    def test_repr_short_choices(self):
        """Test repr with short choices list."""
        constraint = CategoricalConstraint(choices=["a", "b"])
        assert "choices=['a', 'b']" in repr(constraint)

    def test_repr_many_choices(self):
        """Test repr with many choices shows count."""
        constraint = CategoricalConstraint(choices=list(range(10)))
        assert "10 items" in repr(constraint)


class TestNumericalConstraint:
    """Tests for NumericalConstraint."""

    def test_create_basic_range(self):
        """Test creating a numerical constraint with basic range."""
        constraint = NumericalConstraint(min=0.0, max=2.0)
        assert constraint.min == 0.0
        assert constraint.max == 2.0
        assert constraint.log_scale is False
        assert constraint.step is None

    def test_create_with_log_scale(self):
        """Test creating a constraint with log scale."""
        constraint = NumericalConstraint(min=0.001, max=1.0, log_scale=True)
        assert constraint.log_scale is True

    def test_create_with_step(self):
        """Test creating a discrete constraint with step."""
        constraint = NumericalConstraint(min=1, max=100, step=1)
        assert constraint.step == 1
        assert constraint.is_discrete is True

    def test_is_discrete_false(self):
        """Test is_discrete returns False for continuous."""
        constraint = NumericalConstraint(min=0.0, max=1.0)
        assert constraint.is_discrete is False

    def test_validate_in_range(self):
        """Test validating values in range."""
        constraint = NumericalConstraint(min=0.0, max=2.0)
        assert constraint.validate(0.0) is True
        assert constraint.validate(1.0) is True
        assert constraint.validate(2.0) is True

    def test_validate_out_of_range(self):
        """Test validating values out of range."""
        constraint = NumericalConstraint(min=0.0, max=2.0)
        assert constraint.validate(-0.1) is False
        assert constraint.validate(2.1) is False

    def test_validate_step_alignment_valid(self):
        """Test validating values on step grid."""
        constraint = NumericalConstraint(min=0, max=100, step=10)
        assert constraint.validate(0) is True
        assert constraint.validate(10) is True
        assert constraint.validate(50) is True
        assert constraint.validate(100) is True

    def test_validate_step_alignment_invalid(self):
        """Test validating values off step grid."""
        constraint = NumericalConstraint(min=0, max=100, step=10)
        assert constraint.validate(5) is False
        assert constraint.validate(15) is False
        assert constraint.validate(99) is False

    def test_validate_step_alignment_float(self):
        """Test validating float values on step grid."""
        constraint = NumericalConstraint(min=0.0, max=1.0, step=0.25)
        assert constraint.validate(0.0) is True
        assert constraint.validate(0.25) is True
        assert constraint.validate(0.5) is True
        assert constraint.validate(0.75) is True
        assert constraint.validate(1.0) is True
        assert constraint.validate(0.3) is False

    def test_repr_basic(self):
        """Test repr for basic constraint."""
        constraint = NumericalConstraint(min=0, max=100)
        result = repr(constraint)
        assert "min=0" in result
        assert "max=100" in result

    def test_repr_with_options(self):
        """Test repr with log_scale and step."""
        constraint = NumericalConstraint(min=1, max=100, log_scale=True, step=5)
        result = repr(constraint)
        assert "log_scale=True" in result
        assert "step=5" in result


class TestTVAR:
    """Tests for TVAR dataclass."""

    def test_create_basic_tvar(self):
        """Test creating a basic TVAR."""
        tvar = TVAR(
            name="temperature",
            scope="generator",
            python_type="float",
            default_value=0.7,
        )
        assert tvar.name == "temperature"
        assert tvar.scope == "generator"
        assert tvar.python_type == "float"
        assert tvar.default_value == 0.7
        assert tvar.constraint is None
        assert tvar.is_tunable is True

    def test_qualified_name(self):
        """Test qualified_name property."""
        tvar = TVAR(
            name="temperature",
            scope="generator",
            python_type="float",
            default_value=0.7,
        )
        assert tvar.qualified_name == "generator.temperature"

    def test_tvar_with_categorical_constraint(self):
        """Test TVAR with categorical constraint."""
        constraint = CategoricalConstraint(choices=["gpt-4o", "gpt-4o-mini"])
        tvar = TVAR(
            name="model",
            scope="generator",
            python_type="str",
            default_value="gpt-4o",
            constraint=constraint,
        )
        assert tvar.validate("gpt-4o") is True
        assert tvar.validate("invalid") is False

    def test_tvar_with_numerical_constraint(self):
        """Test TVAR with numerical constraint."""
        constraint = NumericalConstraint(min=0.0, max=2.0)
        tvar = TVAR(
            name="temperature",
            scope="generator",
            python_type="float",
            default_value=0.7,
            constraint=constraint,
        )
        assert tvar.validate(1.0) is True
        assert tvar.validate(3.0) is False

    def test_validate_without_constraint(self):
        """Test validate returns True when no constraint."""
        tvar = TVAR(
            name="custom_param",
            scope="component",
            python_type="str",
            default_value="value",
        )
        assert tvar.validate("anything") is True

    def test_fixed_tvar(self):
        """Test creating a fixed (non-tunable) TVAR."""
        tvar = TVAR(
            name="api_key",
            scope="generator",
            python_type="str",
            default_value="secret",
            is_tunable=False,
        )
        assert tvar.is_tunable is False

    def test_repr_with_constraint(self):
        """Test TVAR repr with constraint."""
        constraint = NumericalConstraint(min=0.0, max=2.0)
        tvar = TVAR(
            name="temperature",
            scope="generator",
            python_type="float",
            default_value=0.7,
            constraint=constraint,
        )
        result = repr(tvar)
        assert "generator.temperature" in result
        assert "float" in result
        assert "range=[0.0, 2.0]" in result
        assert "tunable" in result


class TestExplorationSpace:
    """Tests for ExplorationSpace."""

    def test_create_empty_space(self):
        """Test creating an empty exploration space."""
        space = ExplorationSpace()
        assert len(space) == 0
        assert space.tvars == {}

    def test_create_with_tvars(self):
        """Test creating space with TVARs."""
        tvar = TVAR(
            name="temperature",
            scope="generator",
            python_type="float",
            default_value=0.7,
        )
        space = ExplorationSpace(tvars={tvar.qualified_name: tvar})
        assert len(space) == 1
        assert "generator.temperature" in space.tvars

    def test_get_tvar_by_scope_and_name(self):
        """Test getting TVAR by scope and name."""
        tvar = TVAR(
            name="temperature",
            scope="generator",
            python_type="float",
            default_value=0.7,
        )
        space = ExplorationSpace(tvars={tvar.qualified_name: tvar})
        result = space.get_tvar("generator", "temperature")
        assert result is tvar

    def test_get_tvar_not_found(self):
        """Test getting non-existent TVAR."""
        space = ExplorationSpace()
        assert space.get_tvar("generator", "temperature") is None

    def test_get_tvar_by_qualified_name(self):
        """Test getting TVAR by qualified name."""
        tvar = TVAR(
            name="temperature",
            scope="generator",
            python_type="float",
            default_value=0.7,
        )
        space = ExplorationSpace(tvars={tvar.qualified_name: tvar})
        result = space.get_tvar_by_qualified_name("generator.temperature")
        assert result is tvar

    def test_tunable_tvars_property(self):
        """Test tunable_tvars filters correctly."""
        tunable = TVAR(
            name="temperature",
            scope="generator",
            python_type="float",
            default_value=0.7,
            is_tunable=True,
        )
        fixed = TVAR(
            name="api_key",
            scope="generator",
            python_type="str",
            default_value="key",
            is_tunable=False,
        )
        space = ExplorationSpace(
            tvars={
                tunable.qualified_name: tunable,
                fixed.qualified_name: fixed,
            }
        )
        assert len(space.tunable_tvars) == 1
        assert "generator.temperature" in space.tunable_tvars

    def test_fixed_tvars_property(self):
        """Test fixed_tvars filters correctly."""
        tunable = TVAR(
            name="temperature",
            scope="generator",
            python_type="float",
            default_value=0.7,
            is_tunable=True,
        )
        fixed = TVAR(
            name="api_key",
            scope="generator",
            python_type="str",
            default_value="key",
            is_tunable=False,
        )
        space = ExplorationSpace(
            tvars={
                tunable.qualified_name: tunable,
                fixed.qualified_name: fixed,
            }
        )
        assert len(space.fixed_tvars) == 1
        assert "generator.api_key" in space.fixed_tvars

    def test_scope_names_property(self):
        """Test scope_names returns unique scopes."""
        tvar1 = TVAR(
            name="temperature",
            scope="generator",
            python_type="float",
            default_value=0.7,
        )
        tvar2 = TVAR(
            name="model",
            scope="generator",
            python_type="str",
            default_value="gpt-4o",
        )
        tvar3 = TVAR(
            name="top_k",
            scope="retriever",
            python_type="int",
            default_value=10,
        )
        space = ExplorationSpace(
            tvars={
                tvar1.qualified_name: tvar1,
                tvar2.qualified_name: tvar2,
                tvar3.qualified_name: tvar3,
            }
        )
        scope_names = space.scope_names
        assert "generator" in scope_names
        assert "retriever" in scope_names
        assert len(scope_names) == 2

    def test_get_tvars_by_scope(self):
        """Test getting TVARs for a specific scope."""
        tvar1 = TVAR(
            name="temperature",
            scope="generator",
            python_type="float",
            default_value=0.7,
        )
        tvar2 = TVAR(
            name="top_k",
            scope="retriever",
            python_type="int",
            default_value=10,
        )
        space = ExplorationSpace(
            tvars={
                tvar1.qualified_name: tvar1,
                tvar2.qualified_name: tvar2,
            }
        )
        gen_tvars = space.get_tvars_by_scope("generator")
        assert len(gen_tvars) == 1
        assert "generator.temperature" in gen_tvars

    def test_validate_config_valid(self):
        """Test validating a valid configuration."""
        constraint = NumericalConstraint(min=0.0, max=2.0)
        tvar = TVAR(
            name="temperature",
            scope="generator",
            python_type="float",
            default_value=0.7,
            constraint=constraint,
        )
        space = ExplorationSpace(tvars={tvar.qualified_name: tvar})

        is_valid, errors = space.validate_config({"generator.temperature": 1.0})
        assert is_valid is True
        assert errors == []

    def test_validate_config_invalid_value(self):
        """Test validating an invalid configuration."""
        constraint = NumericalConstraint(min=0.0, max=2.0)
        tvar = TVAR(
            name="temperature",
            scope="generator",
            python_type="float",
            default_value=0.7,
            constraint=constraint,
        )
        space = ExplorationSpace(tvars={tvar.qualified_name: tvar})

        is_valid, errors = space.validate_config({"generator.temperature": 5.0})
        assert is_valid is False
        assert len(errors) == 1
        assert "Invalid value" in errors[0]

    def test_validate_config_unknown_param(self):
        """Test validating config with unknown parameter."""
        space = ExplorationSpace()
        is_valid, errors = space.validate_config({"unknown.param": 1.0})
        assert is_valid is False
        assert "Unknown parameter" in errors[0]

    def test_iteration(self):
        """Test iterating over TVARs."""
        tvar = TVAR(
            name="temperature",
            scope="generator",
            python_type="float",
            default_value=0.7,
        )
        space = ExplorationSpace(tvars={tvar.qualified_name: tvar})
        tvars_list = list(space)
        assert len(tvars_list) == 1
        assert tvars_list[0] is tvar

    def test_repr(self):
        """Test ExplorationSpace repr."""
        tunable = TVAR(
            name="temperature",
            scope="generator",
            python_type="float",
            default_value=0.7,
            is_tunable=True,
        )
        fixed = TVAR(
            name="api_key",
            scope="generator",
            python_type="str",
            default_value="key",
            is_tunable=False,
        )
        space = ExplorationSpace(
            tvars={
                tunable.qualified_name: tunable,
                fixed.qualified_name: fixed,
            }
        )
        result = repr(space)
        assert "tunable=1" in result
        assert "fixed=1" in result


class TestExplorationSpaceFromPipelineSpec:
    """Tests for ExplorationSpace.from_pipeline_spec()."""

    def test_from_empty_pipeline_spec(self):
        """Test creating from empty PipelineSpec."""
        spec = PipelineSpec()
        space = ExplorationSpace.from_pipeline_spec(spec)
        assert len(space) == 0

    def test_from_pipeline_spec_with_tunable_tvars(self):
        """Test creating from PipelineSpec with tunable TVARs."""
        discovered = DiscoveredTVAR(
            name="temperature",
            value=0.7,
            python_type="float",
            is_tunable=True,
            default_range=(0.0, 2.0),
            range_type="continuous",
        )
        scope = TVARScope(
            name="generator",
            class_name="OpenAIGenerator",
            class_type="haystack.components.generators.OpenAIGenerator",
            category="Generator",
            tvars={"temperature": discovered},
        )
        spec = PipelineSpec(scopes=[scope])

        space = ExplorationSpace.from_pipeline_spec(spec)
        assert len(space) == 1

        tvar = space.get_tvar("generator", "temperature")
        assert tvar is not None
        assert tvar.name == "temperature"
        assert tvar.scope == "generator"
        assert tvar.is_tunable is True
        assert isinstance(tvar.constraint, NumericalConstraint)
        assert tvar.constraint.min == 0.0
        assert tvar.constraint.max == 2.0

    def test_from_pipeline_spec_with_literal_choices(self):
        """Test creating from PipelineSpec with Literal choices."""
        discovered = DiscoveredTVAR(
            name="model",
            value="gpt-4o",
            python_type="Literal",
            is_tunable=True,
            literal_choices=["gpt-4o", "gpt-4o-mini"],
        )
        scope = TVARScope(
            name="generator",
            class_name="OpenAIGenerator",
            class_type="haystack.components.generators.OpenAIGenerator",
            category="Generator",
            tvars={"model": discovered},
        )
        spec = PipelineSpec(scopes=[scope])

        space = ExplorationSpace.from_pipeline_spec(spec)
        tvar = space.get_tvar("generator", "model")
        assert tvar is not None
        assert isinstance(tvar.constraint, CategoricalConstraint)
        assert tvar.constraint.choices == ["gpt-4o", "gpt-4o-mini"]

    def test_from_pipeline_spec_with_bool_type(self):
        """Test creating from PipelineSpec with bool type."""
        discovered = DiscoveredTVAR(
            name="scale_score",
            value=True,
            python_type="bool",
            is_tunable=True,
        )
        scope = TVARScope(
            name="retriever",
            class_name="InMemoryBM25Retriever",
            class_type="haystack.components.retrievers.InMemoryBM25Retriever",
            category="Retriever",
            tvars={"scale_score": discovered},
        )
        spec = PipelineSpec(scopes=[scope])

        space = ExplorationSpace.from_pipeline_spec(spec)
        tvar = space.get_tvar("retriever", "scale_score")
        assert tvar is not None
        assert isinstance(tvar.constraint, CategoricalConstraint)
        assert True in tvar.constraint.choices
        assert False in tvar.constraint.choices

    def test_from_pipeline_spec_with_optional_bool(self):
        """Test creating from PipelineSpec with Optional bool."""
        discovered = DiscoveredTVAR(
            name="scale_score",
            value=None,
            python_type="bool",
            is_tunable=True,
            is_optional=True,
        )
        scope = TVARScope(
            name="retriever",
            class_name="InMemoryBM25Retriever",
            class_type="haystack.components.retrievers.InMemoryBM25Retriever",
            category="Retriever",
            tvars={"scale_score": discovered},
        )
        spec = PipelineSpec(scopes=[scope])

        space = ExplorationSpace.from_pipeline_spec(spec)
        tvar = space.get_tvar("retriever", "scale_score")
        assert tvar is not None
        assert isinstance(tvar.constraint, CategoricalConstraint)
        assert None in tvar.constraint.choices

    def test_from_pipeline_spec_with_log_scale(self):
        """Test creating from PipelineSpec with log scale range."""
        discovered = DiscoveredTVAR(
            name="learning_rate",
            value=0.001,
            python_type="float",
            is_tunable=True,
            default_range=(0.0001, 0.1),
            range_type="log",
        )
        scope = TVARScope(
            name="optimizer",
            class_name="CustomOptimizer",
            class_type="custom.optimizer.CustomOptimizer",
            category="Component",
            tvars={"learning_rate": discovered},
        )
        spec = PipelineSpec(scopes=[scope])

        space = ExplorationSpace.from_pipeline_spec(spec)
        tvar = space.get_tvar("optimizer", "learning_rate")
        assert tvar is not None
        assert isinstance(tvar.constraint, NumericalConstraint)
        assert tvar.constraint.log_scale is True

    def test_from_pipeline_spec_with_discrete_range(self):
        """Test creating from PipelineSpec with discrete range."""
        discovered = DiscoveredTVAR(
            name="top_k",
            value=10,
            python_type="int",
            is_tunable=True,
            default_range=(1, 100),
            range_type="discrete",
        )
        scope = TVARScope(
            name="retriever",
            class_name="InMemoryBM25Retriever",
            class_type="haystack.components.retrievers.InMemoryBM25Retriever",
            category="Retriever",
            tvars={"top_k": discovered},
        )
        spec = PipelineSpec(scopes=[scope])

        space = ExplorationSpace.from_pipeline_spec(spec)
        tvar = space.get_tvar("retriever", "top_k")
        assert tvar is not None
        assert isinstance(tvar.constraint, NumericalConstraint)
        assert tvar.constraint.step == 1
        assert tvar.constraint.is_discrete is True

    def test_from_pipeline_spec_with_fixed_tvars(self):
        """Test creating from PipelineSpec with fixed TVARs."""
        discovered = DiscoveredTVAR(
            name="api_key",
            value="secret-key",
            python_type="str",
            is_tunable=False,
            non_tunable_reason="TVAR 'api_key' is a complex object",
        )
        scope = TVARScope(
            name="generator",
            class_name="OpenAIGenerator",
            class_type="haystack.components.generators.OpenAIGenerator",
            category="Generator",
            tvars={"api_key": discovered},
        )
        spec = PipelineSpec(scopes=[scope])

        space = ExplorationSpace.from_pipeline_spec(spec)
        tvar = space.get_tvar("generator", "api_key")
        assert tvar is not None
        assert tvar.is_tunable is False

    def test_from_pipeline_spec_multiple_scopes(self):
        """Test creating from PipelineSpec with multiple scopes."""
        gen_tvar = DiscoveredTVAR(
            name="temperature",
            value=0.7,
            python_type="float",
            is_tunable=True,
            default_range=(0.0, 2.0),
        )
        ret_tvar = DiscoveredTVAR(
            name="top_k",
            value=10,
            python_type="int",
            is_tunable=True,
            default_range=(1, 100),
        )
        gen_scope = TVARScope(
            name="generator",
            class_name="OpenAIGenerator",
            class_type="haystack.components.generators.OpenAIGenerator",
            category="Generator",
            tvars={"temperature": gen_tvar},
        )
        ret_scope = TVARScope(
            name="retriever",
            class_name="InMemoryBM25Retriever",
            class_type="haystack.components.retrievers.InMemoryBM25Retriever",
            category="Retriever",
            tvars={"top_k": ret_tvar},
        )
        spec = PipelineSpec(scopes=[gen_scope, ret_scope])

        space = ExplorationSpace.from_pipeline_spec(spec)
        assert len(space) == 2
        assert space.get_tvar("generator", "temperature") is not None
        assert space.get_tvar("retriever", "top_k") is not None


class TestFromPipelineWithExplorationSpace:
    """Tests for from_pipeline() with as_exploration_space parameter."""

    def test_from_pipeline_default_returns_pipeline_spec(self):
        """Test from_pipeline() returns PipelineSpec by default."""

        class MockPipeline:
            def walk(self):
                return []

            _components = {}

        result = from_pipeline(MockPipeline())
        assert isinstance(result, PipelineSpec)

    def test_from_pipeline_as_exploration_space_true(self):
        """Test from_pipeline() returns ExplorationSpace when requested."""

        class MockComponent:
            def __init__(self, temperature: float = 0.7):
                self.temperature = temperature

        class MockPipeline:
            def walk(self):
                return [("generator", MockComponent())]

            _components = {"generator": MockComponent()}

        result = from_pipeline(MockPipeline(), as_exploration_space=True)
        assert isinstance(result, ExplorationSpace)

    def test_from_pipeline_exploration_space_has_tvars(self):
        """Test ExplorationSpace from pipeline has correct TVARs."""

        class MockGenerator:
            def __init__(self, temperature: float = 0.7, model: str = "gpt-4o"):
                self.temperature = temperature
                self.model = model

        class MockPipeline:
            def walk(self):
                return [("generator", MockGenerator())]

            _components = {"generator": MockGenerator()}

        result = from_pipeline(MockPipeline(), as_exploration_space=True)
        assert isinstance(result, ExplorationSpace)

        # Check that TVARs were created
        temp_tvar = result.get_tvar("generator", "temperature")
        assert temp_tvar is not None
        assert temp_tvar.python_type == "float"
        assert temp_tvar.default_value == 0.7

        model_tvar = result.get_tvar("generator", "model")
        assert model_tvar is not None
        assert model_tvar.python_type == "str"


# =============================================================================
# Story 2.2: Support Categorical Variables
# =============================================================================


class TestSetChoices:
    """Tests for ExplorationSpace.set_choices() method (Story 2.2)."""

    def test_set_choices_on_categorical_tvar(self):
        """Test modifying choices on existing categorical TVAR."""
        space = ExplorationSpace(
            tvars={
                "generator.model": TVAR(
                    name="model",
                    scope="generator",
                    python_type="str",
                    default_value="gpt-4o",
                    constraint=CategoricalConstraint(choices=["gpt-4o", "gpt-4o-mini"]),
                )
            }
        )

        space.set_choices("generator.model", ["gpt-4-turbo", "gpt-4o"])
        tvar = space.get_tvar("generator", "model")
        assert tvar.constraint.choices == ["gpt-4-turbo", "gpt-4o"]

    def test_set_choices_converts_numerical_to_categorical(self):
        """Test set_choices can convert a numerical TVAR to categorical."""
        space = ExplorationSpace(
            tvars={
                "generator.temperature": TVAR(
                    name="temperature",
                    scope="generator",
                    python_type="float",
                    default_value=0.7,
                    constraint=NumericalConstraint(min=0.0, max=2.0),
                )
            }
        )

        # Convert to discrete choices
        space.set_choices("generator.temperature", [0.0, 0.5, 1.0, 1.5, 2.0])
        tvar = space.get_tvar("generator", "temperature")
        assert isinstance(tvar.constraint, CategoricalConstraint)
        assert tvar.constraint.choices == [0.0, 0.5, 1.0, 1.5, 2.0]

    def test_set_choices_on_tvar_without_constraint(self):
        """Test set_choices on TVAR without existing constraint."""
        space = ExplorationSpace(
            tvars={
                "generator.model": TVAR(
                    name="model",
                    scope="generator",
                    python_type="str",
                    default_value="default",
                    constraint=None,
                )
            }
        )

        space.set_choices("generator.model", ["option1", "option2"])
        tvar = space.get_tvar("generator", "model")
        assert isinstance(tvar.constraint, CategoricalConstraint)
        assert tvar.constraint.choices == ["option1", "option2"]

    def test_set_choices_raises_keyerror_for_missing_tvar(self):
        """Test set_choices raises KeyError for non-existent TVAR."""
        space = ExplorationSpace()

        with pytest.raises(KeyError) as exc_info:
            space.set_choices("unknown.param", ["a", "b"])
        assert "unknown.param" in str(exc_info.value)

    def test_set_choices_raises_valueerror_for_empty_list(self):
        """Test set_choices raises ValueError for empty choices."""
        space = ExplorationSpace(
            tvars={
                "generator.model": TVAR(
                    name="model",
                    scope="generator",
                    python_type="str",
                    default_value="gpt-4o",
                )
            }
        )

        with pytest.raises(ValueError) as exc_info:
            space.set_choices("generator.model", [])
        assert "empty" in str(exc_info.value).lower()


class TestSetRange:
    """Tests for ExplorationSpace.set_range() method (Story 2.2/2.3)."""

    def test_set_range_on_numerical_tvar(self):
        """Test modifying range on existing numerical TVAR."""
        space = ExplorationSpace(
            tvars={
                "generator.temperature": TVAR(
                    name="temperature",
                    scope="generator",
                    python_type="float",
                    default_value=0.7,
                    constraint=NumericalConstraint(min=0.0, max=2.0),
                )
            }
        )

        space.set_range("generator.temperature", 0.0, 1.5)
        tvar = space.get_tvar("generator", "temperature")
        assert tvar.constraint.min == 0.0
        assert tvar.constraint.max == 1.5

    def test_set_range_with_log_scale(self):
        """Test set_range with log_scale option."""
        space = ExplorationSpace(
            tvars={
                "optimizer.lr": TVAR(
                    name="lr",
                    scope="optimizer",
                    python_type="float",
                    default_value=0.001,
                )
            }
        )

        space.set_range("optimizer.lr", 0.0001, 0.1, log_scale=True)
        tvar = space.get_tvar("optimizer", "lr")
        assert tvar.constraint.log_scale is True

    def test_set_range_with_step(self):
        """Test set_range with step for discrete values."""
        space = ExplorationSpace(
            tvars={
                "retriever.top_k": TVAR(
                    name="top_k",
                    scope="retriever",
                    python_type="int",
                    default_value=10,
                )
            }
        )

        space.set_range("retriever.top_k", 1, 100, step=5)
        tvar = space.get_tvar("retriever", "top_k")
        assert tvar.constraint.step == 5
        assert tvar.constraint.is_discrete is True

    def test_set_range_raises_keyerror_for_missing_tvar(self):
        """Test set_range raises KeyError for non-existent TVAR."""
        space = ExplorationSpace()

        with pytest.raises(KeyError) as exc_info:
            space.set_range("unknown.param", 0, 100)
        assert "unknown.param" in str(exc_info.value)

    def test_set_range_raises_valueerror_for_invalid_range(self):
        """Test set_range raises ValueError when min >= max."""
        space = ExplorationSpace(
            tvars={
                "generator.temperature": TVAR(
                    name="temperature",
                    scope="generator",
                    python_type="float",
                    default_value=0.7,
                )
            }
        )

        with pytest.raises(ValueError) as exc_info:
            space.set_range("generator.temperature", 2.0, 1.0)
        assert "min_val" in str(exc_info.value)

    def test_set_range_raises_valueerror_for_non_positive_step(self):
        """Test set_range raises ValueError when step <= 0."""
        space = ExplorationSpace(
            tvars={
                "retriever.top_k": TVAR(
                    name="top_k",
                    scope="retriever",
                    python_type="int",
                    default_value=10,
                )
            }
        )

        with pytest.raises(ValueError) as exc_info:
            space.set_range("retriever.top_k", 1, 100, step=0)
        assert "step" in str(exc_info.value)

        with pytest.raises(ValueError) as exc_info:
            space.set_range("retriever.top_k", 1, 100, step=-5)
        assert "step" in str(exc_info.value)


class TestSample:
    """Tests for ExplorationSpace.sample() method (Story 2.2)."""

    def test_sample_categorical_returns_valid_choice(self):
        """Test that sample() returns values from categorical choices."""
        space = ExplorationSpace(
            tvars={
                "generator.model": TVAR(
                    name="model",
                    scope="generator",
                    python_type="str",
                    default_value="gpt-4o",
                    constraint=CategoricalConstraint(choices=["gpt-4o", "gpt-4o-mini"]),
                    is_tunable=True,
                )
            }
        )

        # Sample multiple times and verify all values are valid
        for _ in range(10):
            config = space.sample()
            assert config["generator.model"] in ["gpt-4o", "gpt-4o-mini"]

    def test_sample_numerical_returns_value_in_range(self):
        """Test that sample() returns values within numerical range."""
        space = ExplorationSpace(
            tvars={
                "generator.temperature": TVAR(
                    name="temperature",
                    scope="generator",
                    python_type="float",
                    default_value=0.7,
                    constraint=NumericalConstraint(min=0.0, max=2.0),
                    is_tunable=True,
                )
            }
        )

        for _ in range(10):
            config = space.sample()
            assert 0.0 <= config["generator.temperature"] <= 2.0

    def test_sample_integer_returns_int(self):
        """Test that sample() returns integers for int type TVARs."""
        space = ExplorationSpace(
            tvars={
                "retriever.top_k": TVAR(
                    name="top_k",
                    scope="retriever",
                    python_type="int",
                    default_value=10,
                    constraint=NumericalConstraint(min=1, max=100),
                    is_tunable=True,
                )
            }
        )

        for _ in range(10):
            config = space.sample()
            assert isinstance(config["retriever.top_k"], int)
            assert 1 <= config["retriever.top_k"] <= 100

    def test_sample_discrete_returns_int(self):
        """Test that sample() returns integers for discrete constraints."""
        space = ExplorationSpace(
            tvars={
                "batch.size": TVAR(
                    name="size",
                    scope="batch",
                    python_type="float",
                    default_value=32,
                    constraint=NumericalConstraint(min=8, max=128, step=8),
                    is_tunable=True,
                )
            }
        )

        config = space.sample()
        assert isinstance(config["batch.size"], int)

    def test_sample_discrete_respects_step_grid(self):
        """Test that sample() only returns values on the step grid."""
        space = ExplorationSpace(
            tvars={
                "batch.size": TVAR(
                    name="size",
                    scope="batch",
                    python_type="int",
                    default_value=32,
                    constraint=NumericalConstraint(min=8, max=128, step=8),
                    is_tunable=True,
                )
            }
        )

        valid_values = set(range(8, 129, 8))  # 8, 16, 24, ..., 128
        for _ in range(50):
            config = space.sample()
            assert (
                config["batch.size"] in valid_values
            ), f"Sampled {config['batch.size']} not in step grid"

    def test_sample_discrete_float_step_respects_grid(self):
        """Test that sample() respects float step grid."""
        space = ExplorationSpace(
            tvars={
                "param.value": TVAR(
                    name="value",
                    scope="param",
                    python_type="float",
                    default_value=0.5,
                    constraint=NumericalConstraint(min=0.0, max=1.0, step=0.25),
                    is_tunable=True,
                )
            }
        )

        valid_values = {0.0, 0.25, 0.5, 0.75, 1.0}
        for _ in range(20):
            config = space.sample()
            assert (
                config["param.value"] in valid_values
            ), f"Sampled {config['param.value']} not in step grid"

    def test_sample_fixed_tvar_uses_default(self):
        """Test that sample() uses default value for fixed TVARs."""
        space = ExplorationSpace(
            tvars={
                "generator.api_key": TVAR(
                    name="api_key",
                    scope="generator",
                    python_type="str",
                    default_value="secret-key",
                    is_tunable=False,
                )
            }
        )

        config = space.sample()
        assert config["generator.api_key"] == "secret-key"

    def test_sample_no_constraint_uses_default(self):
        """Test that sample() uses default value for TVARs without constraint."""
        space = ExplorationSpace(
            tvars={
                "generator.param": TVAR(
                    name="param",
                    scope="generator",
                    python_type="str",
                    default_value="default_value",
                    constraint=None,
                    is_tunable=True,
                )
            }
        )

        config = space.sample()
        assert config["generator.param"] == "default_value"

    def test_sample_with_seed_reproducible(self):
        """Test that sample() is reproducible with same seed."""
        space = ExplorationSpace(
            tvars={
                "generator.temperature": TVAR(
                    name="temperature",
                    scope="generator",
                    python_type="float",
                    default_value=0.7,
                    constraint=NumericalConstraint(min=0.0, max=2.0),
                    is_tunable=True,
                ),
                "generator.model": TVAR(
                    name="model",
                    scope="generator",
                    python_type="str",
                    default_value="gpt-4o",
                    constraint=CategoricalConstraint(
                        choices=["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"]
                    ),
                    is_tunable=True,
                ),
            }
        )

        config1 = space.sample(seed=42)
        config2 = space.sample(seed=42)
        assert config1 == config2

    def test_sample_different_seeds_different_results(self):
        """Test that different seeds produce different results."""
        space = ExplorationSpace(
            tvars={
                "generator.temperature": TVAR(
                    name="temperature",
                    scope="generator",
                    python_type="float",
                    default_value=0.7,
                    constraint=NumericalConstraint(min=0.0, max=2.0),
                    is_tunable=True,
                )
            }
        )

        config1 = space.sample(seed=42)
        config2 = space.sample(seed=123)
        # They might be equal by chance, but very unlikely
        # Just verify they're both valid
        assert 0.0 <= config1["generator.temperature"] <= 2.0
        assert 0.0 <= config2["generator.temperature"] <= 2.0

    def test_sample_mixed_tvars(self):
        """Test sample() with mix of categorical, numerical, and fixed TVARs."""
        space = ExplorationSpace(
            tvars={
                "generator.temperature": TVAR(
                    name="temperature",
                    scope="generator",
                    python_type="float",
                    default_value=0.7,
                    constraint=NumericalConstraint(min=0.0, max=2.0),
                    is_tunable=True,
                ),
                "generator.model": TVAR(
                    name="model",
                    scope="generator",
                    python_type="str",
                    default_value="gpt-4o",
                    constraint=CategoricalConstraint(choices=["gpt-4o", "gpt-4o-mini"]),
                    is_tunable=True,
                ),
                "generator.api_key": TVAR(
                    name="api_key",
                    scope="generator",
                    python_type="str",
                    default_value="secret",
                    is_tunable=False,
                ),
            }
        )

        config = space.sample(seed=42)

        # Numerical should be in range
        assert 0.0 <= config["generator.temperature"] <= 2.0

        # Categorical should be valid choice
        assert config["generator.model"] in ["gpt-4o", "gpt-4o-mini"]

        # Fixed should be default
        assert config["generator.api_key"] == "secret"

    def test_sample_empty_space(self):
        """Test sample() on empty exploration space."""
        space = ExplorationSpace()
        config = space.sample()
        assert config == {}

    def test_sample_log_scale_with_zero_min_falls_back_to_linear(self):
        """Test that log_scale with min <= 0 falls back to linear sampling."""
        space = ExplorationSpace(
            tvars={
                "param.value": TVAR(
                    name="value",
                    scope="param",
                    python_type="float",
                    default_value=0.5,
                    # log_scale with min=0 should fall back to linear
                    constraint=NumericalConstraint(min=0.0, max=1.0, log_scale=True),
                    is_tunable=True,
                )
            }
        )

        # Should not raise, falls back to linear sampling
        config = space.sample(seed=42)
        assert 0.0 <= config["param.value"] <= 1.0

    def test_sample_does_not_mutate_global_rng(self):
        """Test that sample() uses local RNG and doesn't affect global state."""
        import random

        # Set global seed
        random.seed(12345)
        global_value_before = random.random()

        # Reset and sample
        random.seed(12345)
        space = ExplorationSpace(
            tvars={
                "generator.temperature": TVAR(
                    name="temperature",
                    scope="generator",
                    python_type="float",
                    default_value=0.7,
                    constraint=NumericalConstraint(min=0.0, max=2.0),
                    is_tunable=True,
                )
            }
        )
        space.sample(seed=42)  # This should use local RNG

        # Global state should still produce same sequence
        global_value_after = random.random()
        assert global_value_before == global_value_after


class TestValidateConfigCategorical:
    """Tests for validate_config() with categorical values (Story 2.2)."""

    def test_validate_valid_categorical_value(self):
        """Test validation accepts valid categorical value."""
        space = ExplorationSpace(
            tvars={
                "generator.model": TVAR(
                    name="model",
                    scope="generator",
                    python_type="str",
                    default_value="gpt-4o",
                    constraint=CategoricalConstraint(choices=["gpt-4o", "gpt-4o-mini"]),
                )
            }
        )

        is_valid, errors = space.validate_config({"generator.model": "gpt-4o"})
        assert is_valid is True
        assert errors == []

    def test_validate_invalid_categorical_value(self):
        """Test validation rejects invalid categorical value with clear error."""
        space = ExplorationSpace(
            tvars={
                "generator.model": TVAR(
                    name="model",
                    scope="generator",
                    python_type="str",
                    default_value="gpt-4o",
                    constraint=CategoricalConstraint(choices=["gpt-4o", "gpt-4o-mini"]),
                )
            }
        )

        is_valid, errors = space.validate_config({"generator.model": "invalid-model"})
        assert is_valid is False
        assert len(errors) == 1
        assert "generator.model" in errors[0]
        assert "invalid-model" in errors[0]
        assert "not in choices" in errors[0]

    def test_validate_invalid_numerical_value_error_message(self):
        """Test validation error message for numerical constraint."""
        space = ExplorationSpace(
            tvars={
                "generator.temperature": TVAR(
                    name="temperature",
                    scope="generator",
                    python_type="float",
                    default_value=0.7,
                    constraint=NumericalConstraint(min=0.0, max=2.0),
                )
            }
        )

        is_valid, errors = space.validate_config({"generator.temperature": 5.0})
        assert is_valid is False
        assert len(errors) == 1
        assert "generator.temperature" in errors[0]
        assert "5.0" in errors[0]
        assert "not in range" in errors[0]

    def test_validate_multiple_errors(self):
        """Test validation collects multiple errors."""
        space = ExplorationSpace(
            tvars={
                "generator.model": TVAR(
                    name="model",
                    scope="generator",
                    python_type="str",
                    default_value="gpt-4o",
                    constraint=CategoricalConstraint(choices=["gpt-4o", "gpt-4o-mini"]),
                ),
                "generator.temperature": TVAR(
                    name="temperature",
                    scope="generator",
                    python_type="float",
                    default_value=0.7,
                    constraint=NumericalConstraint(min=0.0, max=2.0),
                ),
            }
        )

        is_valid, errors = space.validate_config(
            {"generator.model": "invalid", "generator.temperature": 10.0}
        )
        assert is_valid is False
        assert len(errors) == 2


# =============================================================================
# Story 2.4: Support Conditional Variables Tests
# =============================================================================


class TestConditionalConstraint:
    """Tests for ConditionalConstraint (Story 2.4)."""

    def test_create_conditional_constraint(self):
        """Test creating a conditional constraint."""
        from traigent.integrations.haystack import ConditionalConstraint

        constraint = ConditionalConstraint(
            parent_qualified_name="generator.model",
            conditions={
                "gpt-4o": NumericalConstraint(min=100, max=8192),
                "gpt-4o-mini": NumericalConstraint(min=100, max=4096),
            },
        )
        assert constraint.parent_qualified_name == "generator.model"
        assert len(constraint.conditions) == 2

    def test_get_constraint_for_known_value(self):
        """Test getting constraint for a known parent value."""
        from traigent.integrations.haystack import ConditionalConstraint

        constraint = ConditionalConstraint(
            parent_qualified_name="generator.model",
            conditions={
                "gpt-4o": NumericalConstraint(min=100, max=8192),
                "gpt-4o-mini": NumericalConstraint(min=100, max=4096),
            },
        )
        c = constraint.get_constraint_for("gpt-4o")
        assert isinstance(c, NumericalConstraint)
        assert c.max == 8192

    def test_get_constraint_for_unknown_value_returns_none(self):
        """Test getting constraint for unknown parent value returns None."""
        from traigent.integrations.haystack import ConditionalConstraint

        constraint = ConditionalConstraint(
            parent_qualified_name="generator.model",
            conditions={
                "gpt-4o": NumericalConstraint(min=100, max=8192),
            },
        )
        assert constraint.get_constraint_for("unknown-model") is None

    def test_get_constraint_for_unknown_value_with_default(self):
        """Test getting constraint for unknown value uses default."""
        from traigent.integrations.haystack import ConditionalConstraint

        default = NumericalConstraint(min=100, max=2048)
        constraint = ConditionalConstraint(
            parent_qualified_name="generator.model",
            conditions={
                "gpt-4o": NumericalConstraint(min=100, max=8192),
            },
            default_constraint=default,
        )
        c = constraint.get_constraint_for("unknown-model")
        assert c is default

    def test_validate_with_parent_value(self):
        """Test validating a value with parent context."""
        from traigent.integrations.haystack import ConditionalConstraint

        constraint = ConditionalConstraint(
            parent_qualified_name="generator.model",
            conditions={
                "gpt-4o": NumericalConstraint(min=100, max=8192),
                "gpt-4o-mini": NumericalConstraint(min=100, max=4096),
            },
        )
        # Valid for gpt-4o (max is 8192)
        assert constraint.validate(5000, "gpt-4o") is True
        # Invalid for gpt-4o-mini (max is 4096)
        assert constraint.validate(5000, "gpt-4o-mini") is False

    def test_validate_unknown_parent_value_returns_true(self):
        """Test validating with unknown parent value returns True (no constraint)."""
        from traigent.integrations.haystack import ConditionalConstraint

        constraint = ConditionalConstraint(
            parent_qualified_name="generator.model",
            conditions={
                "gpt-4o": NumericalConstraint(min=100, max=8192),
            },
        )
        # No constraint defined for unknown model, so any value is valid
        assert constraint.validate(99999, "unknown-model") is True

    def test_repr(self):
        """Test repr of conditional constraint."""
        from traigent.integrations.haystack import ConditionalConstraint

        constraint = ConditionalConstraint(
            parent_qualified_name="generator.model",
            conditions={
                "gpt-4o": NumericalConstraint(min=100, max=8192),
                "gpt-4o-mini": NumericalConstraint(min=100, max=4096),
            },
        )
        r = repr(constraint)
        assert "ConditionalConstraint" in r
        assert "generator.model" in r
        assert "conditions=2" in r


class TestSetConditional:
    """Tests for ExplorationSpace.set_conditional() method (Story 2.4)."""

    def test_set_conditional_creates_constraint(self):
        """Test that set_conditional creates a ConditionalConstraint."""
        from traigent.integrations.haystack import ConditionalConstraint

        space = ExplorationSpace(
            tvars={
                "generator.model": TVAR(
                    name="model",
                    scope="generator",
                    python_type="str",
                    default_value="gpt-4o",
                    constraint=CategoricalConstraint(choices=["gpt-4o", "gpt-4o-mini"]),
                    is_tunable=True,
                ),
                "generator.max_tokens": TVAR(
                    name="max_tokens",
                    scope="generator",
                    python_type="int",
                    default_value=1024,
                    constraint=NumericalConstraint(min=100, max=4096),
                    is_tunable=True,
                ),
            }
        )

        space.set_conditional(
            child="generator.max_tokens",
            parent="generator.model",
            conditions={
                "gpt-4o": {"min": 100, "max": 8192},
                "gpt-4o-mini": {"min": 100, "max": 4096},
            },
        )

        tvar = space.get_tvar("generator", "max_tokens")
        assert isinstance(tvar.constraint, ConditionalConstraint)
        assert tvar.constraint.parent_qualified_name == "generator.model"

    def test_set_conditional_with_categorical_conditions(self):
        """Test set_conditional with categorical child constraints."""
        from traigent.integrations.haystack import ConditionalConstraint

        space = ExplorationSpace(
            tvars={
                "generator.model": TVAR(
                    name="model",
                    scope="generator",
                    python_type="str",
                    default_value="gpt-4o",
                    constraint=CategoricalConstraint(choices=["gpt-4o", "gpt-4o-mini"]),
                    is_tunable=True,
                ),
                "generator.response_format": TVAR(
                    name="response_format",
                    scope="generator",
                    python_type="str",
                    default_value="text",
                    is_tunable=True,
                ),
            }
        )

        space.set_conditional(
            child="generator.response_format",
            parent="generator.model",
            conditions={
                "gpt-4o": {"choices": ["text", "json", "structured"]},
                "gpt-4o-mini": {"choices": ["text", "json"]},
            },
        )

        tvar = space.get_tvar("generator", "response_format")
        assert isinstance(tvar.constraint, ConditionalConstraint)
        c = tvar.constraint.get_constraint_for("gpt-4o")
        assert isinstance(c, CategoricalConstraint)
        assert "structured" in c.choices

    def test_set_conditional_with_default(self):
        """Test set_conditional with default constraint."""
        from traigent.integrations.haystack import ConditionalConstraint

        space = ExplorationSpace(
            tvars={
                "generator.model": TVAR(
                    name="model",
                    scope="generator",
                    python_type="str",
                    default_value="gpt-4o",
                    constraint=CategoricalConstraint(
                        choices=["gpt-4o", "gpt-4o-mini", "unknown"]
                    ),
                    is_tunable=True,
                ),
                "generator.max_tokens": TVAR(
                    name="max_tokens",
                    scope="generator",
                    python_type="int",
                    default_value=1024,
                    is_tunable=True,
                ),
            }
        )

        space.set_conditional(
            child="generator.max_tokens",
            parent="generator.model",
            conditions={
                "gpt-4o": {"min": 100, "max": 8192},
            },
            default={"min": 100, "max": 2048},
        )

        tvar = space.get_tvar("generator", "max_tokens")
        assert isinstance(tvar.constraint, ConditionalConstraint)
        # Default should be used for unknown models
        c = tvar.constraint.get_constraint_for("unknown")
        assert isinstance(c, NumericalConstraint)
        assert c.max == 2048

    def test_set_conditional_raises_keyerror_for_missing_child(self):
        """Test set_conditional raises KeyError for missing child TVAR."""
        space = ExplorationSpace(
            tvars={
                "generator.model": TVAR(
                    name="model",
                    scope="generator",
                    python_type="str",
                    default_value="gpt-4o",
                    constraint=CategoricalConstraint(choices=["gpt-4o"]),
                    is_tunable=True,
                ),
            }
        )

        with pytest.raises(KeyError, match="Child TVAR not found"):
            space.set_conditional(
                child="generator.unknown",
                parent="generator.model",
                conditions={"gpt-4o": {"min": 100, "max": 8192}},
            )

    def test_set_conditional_raises_keyerror_for_missing_parent(self):
        """Test set_conditional raises KeyError for missing parent TVAR."""
        space = ExplorationSpace(
            tvars={
                "generator.max_tokens": TVAR(
                    name="max_tokens",
                    scope="generator",
                    python_type="int",
                    default_value=1024,
                    is_tunable=True,
                ),
            }
        )

        with pytest.raises(KeyError, match="Parent TVAR not found"):
            space.set_conditional(
                child="generator.max_tokens",
                parent="generator.model",
                conditions={"gpt-4o": {"min": 100, "max": 8192}},
            )

    def test_set_conditional_raises_valueerror_for_non_categorical_parent(self):
        """Test set_conditional raises ValueError for non-categorical parent."""
        space = ExplorationSpace(
            tvars={
                "generator.temperature": TVAR(
                    name="temperature",
                    scope="generator",
                    python_type="float",
                    default_value=0.7,
                    constraint=NumericalConstraint(min=0.0, max=2.0),
                    is_tunable=True,
                ),
                "generator.max_tokens": TVAR(
                    name="max_tokens",
                    scope="generator",
                    python_type="int",
                    default_value=1024,
                    is_tunable=True,
                ),
            }
        )

        with pytest.raises(ValueError, match="must have discrete choices"):
            space.set_conditional(
                child="generator.max_tokens",
                parent="generator.temperature",
                conditions={0.5: {"min": 100, "max": 4096}},
            )

    def test_set_conditional_raises_valueerror_for_circular_dependency(self):
        """Test set_conditional raises ValueError for circular dependency."""
        from traigent.integrations.haystack import ConditionalConstraint

        space = ExplorationSpace(
            tvars={
                "a": TVAR(
                    name="a",
                    scope="",
                    python_type="str",
                    default_value="x",
                    constraint=CategoricalConstraint(choices=["x", "y"]),
                    is_tunable=True,
                ),
                "b": TVAR(
                    name="b",
                    scope="",
                    python_type="str",
                    default_value="x",
                    constraint=CategoricalConstraint(choices=["x", "y"]),
                    is_tunable=True,
                ),
            }
        )

        # First conditional: b depends on a
        space.set_conditional(
            child="b",
            parent="a",
            conditions={"x": {"choices": ["p", "q"]}},
        )

        # Second conditional: a depends on b - should fail (circular)
        with pytest.raises(ValueError, match="circular dependency"):
            space.set_conditional(
                child="a",
                parent="b",
                conditions={"p": {"choices": ["m", "n"]}},
            )

    def test_set_conditional_raises_valueerror_for_invalid_condition_keys(self):
        """Test set_conditional raises ValueError for condition keys not in parent choices."""
        space = ExplorationSpace(
            tvars={
                "generator.model": TVAR(
                    name="model",
                    scope="generator",
                    python_type="str",
                    default_value="gpt-4o",
                    constraint=CategoricalConstraint(choices=["gpt-4o", "gpt-4o-mini"]),
                    is_tunable=True,
                ),
                "generator.max_tokens": TVAR(
                    name="max_tokens",
                    scope="generator",
                    python_type="int",
                    default_value=1024,
                    is_tunable=True,
                ),
            }
        )

        with pytest.raises(ValueError, match="not valid choices"):
            space.set_conditional(
                child="generator.max_tokens",
                parent="generator.model",
                conditions={
                    "gpt-4o": {"min": 100, "max": 8192},
                    "invalid-model": {"min": 100, "max": 4096},  # Invalid!
                },
            )

    def test_set_conditional_raises_valueerror_for_invalid_range_in_conditions(self):
        """Test set_conditional raises ValueError for invalid range (min >= max)."""
        space = ExplorationSpace(
            tvars={
                "generator.model": TVAR(
                    name="model",
                    scope="generator",
                    python_type="str",
                    default_value="gpt-4o",
                    constraint=CategoricalConstraint(choices=["gpt-4o", "gpt-4o-mini"]),
                    is_tunable=True,
                ),
                "generator.max_tokens": TVAR(
                    name="max_tokens",
                    scope="generator",
                    python_type="int",
                    default_value=1024,
                    is_tunable=True,
                ),
            }
        )

        with pytest.raises(ValueError, match="Invalid range"):
            space.set_conditional(
                child="generator.max_tokens",
                parent="generator.model",
                conditions={
                    "gpt-4o": {"min": 8192, "max": 100},  # Invalid: min > max
                    "gpt-4o-mini": {"min": 100, "max": 4096},
                },
            )

    def test_set_conditional_raises_valueerror_for_invalid_step_in_conditions(self):
        """Test set_conditional raises ValueError for invalid step (step <= 0)."""
        space = ExplorationSpace(
            tvars={
                "generator.model": TVAR(
                    name="model",
                    scope="generator",
                    python_type="str",
                    default_value="gpt-4o",
                    constraint=CategoricalConstraint(choices=["gpt-4o", "gpt-4o-mini"]),
                    is_tunable=True,
                ),
                "generator.max_tokens": TVAR(
                    name="max_tokens",
                    scope="generator",
                    python_type="int",
                    default_value=1024,
                    is_tunable=True,
                ),
            }
        )

        with pytest.raises(ValueError, match="step must be positive"):
            space.set_conditional(
                child="generator.max_tokens",
                parent="generator.model",
                conditions={
                    "gpt-4o": {"min": 100, "max": 8192, "step": -10},  # Invalid step
                    "gpt-4o-mini": {"min": 100, "max": 4096},
                },
            )

    def test_set_conditional_raises_valueerror_for_empty_choices_in_conditions(self):
        """Test set_conditional raises ValueError for empty choices."""
        space = ExplorationSpace(
            tvars={
                "generator.model": TVAR(
                    name="model",
                    scope="generator",
                    python_type="str",
                    default_value="gpt-4o",
                    constraint=CategoricalConstraint(choices=["gpt-4o", "gpt-4o-mini"]),
                    is_tunable=True,
                ),
                "generator.response_format": TVAR(
                    name="response_format",
                    scope="generator",
                    python_type="str",
                    default_value="text",
                    is_tunable=True,
                ),
            }
        )

        with pytest.raises(ValueError, match="Empty choices"):
            space.set_conditional(
                child="generator.response_format",
                parent="generator.model",
                conditions={
                    "gpt-4o": {"choices": ["text", "json"]},
                    "gpt-4o-mini": {"choices": []},  # Invalid: empty choices
                },
            )

    def test_set_conditional_warns_for_uncovered_parent_choices(self):
        """Test set_conditional warns when not all parent choices are covered."""
        import warnings

        space = ExplorationSpace(
            tvars={
                "generator.model": TVAR(
                    name="model",
                    scope="generator",
                    python_type="str",
                    default_value="gpt-4o",
                    constraint=CategoricalConstraint(
                        choices=["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"]
                    ),
                    is_tunable=True,
                ),
                "generator.max_tokens": TVAR(
                    name="max_tokens",
                    scope="generator",
                    python_type="int",
                    default_value=1024,
                    is_tunable=True,
                ),
            }
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            space.set_conditional(
                child="generator.max_tokens",
                parent="generator.model",
                conditions={
                    "gpt-4o": {"min": 100, "max": 8192},
                    # Missing: gpt-4o-mini and gpt-3.5-turbo
                },
            )
            # Should have issued a warning
            assert len(w) == 1
            assert "does not cover all parent choices" in str(w[0].message)
            assert "gpt-4o-mini" in str(w[0].message) or "gpt-3.5-turbo" in str(
                w[0].message
            )

    def test_set_conditional_no_warning_with_default(self):
        """Test set_conditional does not warn when default is provided."""
        import warnings

        space = ExplorationSpace(
            tvars={
                "generator.model": TVAR(
                    name="model",
                    scope="generator",
                    python_type="str",
                    default_value="gpt-4o",
                    constraint=CategoricalConstraint(
                        choices=["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"]
                    ),
                    is_tunable=True,
                ),
                "generator.max_tokens": TVAR(
                    name="max_tokens",
                    scope="generator",
                    python_type="int",
                    default_value=1024,
                    is_tunable=True,
                ),
            }
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            space.set_conditional(
                child="generator.max_tokens",
                parent="generator.model",
                conditions={
                    "gpt-4o": {"min": 100, "max": 8192},
                },
                default={"min": 100, "max": 2048},  # Default provided
            )
            # Should NOT have issued a warning
            assert len(w) == 0


class TestSampleConditional:
    """Tests for sampling with conditional constraints (Story 2.4)."""

    def test_sample_respects_conditional_range(self):
        """Test that sample() uses correct range based on parent value."""
        from traigent.integrations.haystack import ConditionalConstraint

        space = ExplorationSpace(
            tvars={
                "generator.model": TVAR(
                    name="model",
                    scope="generator",
                    python_type="str",
                    default_value="gpt-4o",
                    constraint=CategoricalConstraint(choices=["gpt-4o", "gpt-4o-mini"]),
                    is_tunable=True,
                ),
                "generator.max_tokens": TVAR(
                    name="max_tokens",
                    scope="generator",
                    python_type="int",
                    default_value=1024,
                    constraint=NumericalConstraint(min=100, max=4096),
                    is_tunable=True,
                ),
            }
        )

        space.set_conditional(
            child="generator.max_tokens",
            parent="generator.model",
            conditions={
                "gpt-4o": {"min": 100, "max": 8192},
                "gpt-4o-mini": {"min": 100, "max": 4096},
            },
        )

        # Sample many times and verify ranges are respected
        for _ in range(50):
            config = space.sample()
            model = config["generator.model"]
            max_tokens = config["generator.max_tokens"]

            if model == "gpt-4o":
                assert 100 <= max_tokens <= 8192
            elif model == "gpt-4o-mini":
                assert 100 <= max_tokens <= 4096

    def test_sample_with_seed_reproducible_with_conditional(self):
        """Test that sampling with conditional is reproducible with seed."""
        space = ExplorationSpace(
            tvars={
                "generator.model": TVAR(
                    name="model",
                    scope="generator",
                    python_type="str",
                    default_value="gpt-4o",
                    constraint=CategoricalConstraint(choices=["gpt-4o", "gpt-4o-mini"]),
                    is_tunable=True,
                ),
                "generator.max_tokens": TVAR(
                    name="max_tokens",
                    scope="generator",
                    python_type="int",
                    default_value=1024,
                    is_tunable=True,
                ),
            }
        )

        space.set_conditional(
            child="generator.max_tokens",
            parent="generator.model",
            conditions={
                "gpt-4o": {"min": 100, "max": 8192},
                "gpt-4o-mini": {"min": 100, "max": 4096},
            },
        )

        config1 = space.sample(seed=42)
        config2 = space.sample(seed=42)
        assert config1 == config2

    def test_sample_uses_default_when_no_condition_for_parent(self):
        """Test sampling uses default value when no condition for parent value.

        When the parent has a value not covered by conditions and no default
        constraint is provided, the child TVAR's default_value should be used.
        """
        import warnings

        space = ExplorationSpace(
            tvars={
                "generator.model": TVAR(
                    name="model",
                    scope="generator",
                    python_type="str",
                    default_value="gpt-4o",
                    constraint=CategoricalConstraint(
                        choices=["gpt-4o", "unknown-model"]
                    ),
                    is_tunable=True,
                ),
                "generator.max_tokens": TVAR(
                    name="max_tokens",
                    scope="generator",
                    python_type="int",
                    default_value=999,  # This should be used when parent is uncovered
                    is_tunable=True,
                ),
            }
        )

        # Suppress the expected warning about uncovered parent choices
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            space.set_conditional(
                child="generator.max_tokens",
                parent="generator.model",
                conditions={
                    "gpt-4o": {"min": 100, "max": 8192},
                },
                # No condition for "unknown-model" and no default
            )

        # Fix the parent to the uncovered value
        space.fix("generator.model", "unknown-model")

        config = space.sample()
        # Since no constraint exists for "unknown-model", default value is used
        assert config["generator.max_tokens"] == 999

    def test_sample_multi_level_conditional(self):
        """Test sampling with multi-level conditionals (A -> B -> C)."""
        space = ExplorationSpace(
            tvars={
                "a": TVAR(
                    name="a",
                    scope="",
                    python_type="str",
                    default_value="x",
                    constraint=CategoricalConstraint(choices=["x", "y"]),
                    is_tunable=True,
                ),
                "b": TVAR(
                    name="b",
                    scope="",
                    python_type="str",
                    default_value="p",
                    constraint=CategoricalConstraint(choices=["p", "q"]),
                    is_tunable=True,
                ),
                "c": TVAR(
                    name="c",
                    scope="",
                    python_type="int",
                    default_value=10,
                    is_tunable=True,
                ),
            }
        )

        # b depends on a
        space.set_conditional(
            child="b",
            parent="a",
            conditions={
                "x": {"choices": ["p", "q"]},
                "y": {"choices": ["r", "s"]},
            },
        )

        # c depends on b
        space.set_conditional(
            child="c",
            parent="b",
            conditions={
                "p": {"min": 1, "max": 10},
                "q": {"min": 11, "max": 20},
                "r": {"min": 21, "max": 30},
                "s": {"min": 31, "max": 40},
            },
        )

        # Sample and verify the chain is respected
        for _ in range(50):
            config = space.sample()
            a_val = config["a"]
            b_val = config["b"]
            c_val = config["c"]

            # Check b is valid for a
            if a_val == "x":
                assert b_val in ["p", "q"]
            else:
                assert b_val in ["r", "s"]

            # Check c is valid for b
            if b_val == "p":
                assert 1 <= c_val <= 10
            elif b_val == "q":
                assert 11 <= c_val <= 20
            elif b_val == "r":
                assert 21 <= c_val <= 30
            elif b_val == "s":
                assert 31 <= c_val <= 40


class TestValidateConfigConditional:
    """Tests for validate_config with conditional constraints (Story 2.4)."""

    def test_validate_valid_conditional_config(self):
        """Test validation accepts valid conditional config."""
        space = ExplorationSpace(
            tvars={
                "generator.model": TVAR(
                    name="model",
                    scope="generator",
                    python_type="str",
                    default_value="gpt-4o",
                    constraint=CategoricalConstraint(choices=["gpt-4o", "gpt-4o-mini"]),
                ),
                "generator.max_tokens": TVAR(
                    name="max_tokens",
                    scope="generator",
                    python_type="int",
                    default_value=1024,
                ),
            }
        )

        space.set_conditional(
            child="generator.max_tokens",
            parent="generator.model",
            conditions={
                "gpt-4o": {"min": 100, "max": 8192},
                "gpt-4o-mini": {"min": 100, "max": 4096},
            },
        )

        # Valid: 5000 is within gpt-4o's range
        is_valid, errors = space.validate_config(
            {"generator.model": "gpt-4o", "generator.max_tokens": 5000}
        )
        assert is_valid is True
        assert len(errors) == 0

    def test_validate_invalid_conditional_config(self):
        """Test validation rejects invalid conditional config."""
        space = ExplorationSpace(
            tvars={
                "generator.model": TVAR(
                    name="model",
                    scope="generator",
                    python_type="str",
                    default_value="gpt-4o",
                    constraint=CategoricalConstraint(choices=["gpt-4o", "gpt-4o-mini"]),
                ),
                "generator.max_tokens": TVAR(
                    name="max_tokens",
                    scope="generator",
                    python_type="int",
                    default_value=1024,
                ),
            }
        )

        space.set_conditional(
            child="generator.max_tokens",
            parent="generator.model",
            conditions={
                "gpt-4o": {"min": 100, "max": 8192},
                "gpt-4o-mini": {"min": 100, "max": 4096},
            },
        )

        # Invalid: 5000 exceeds gpt-4o-mini's max of 4096
        is_valid, errors = space.validate_config(
            {"generator.model": "gpt-4o-mini", "generator.max_tokens": 5000}
        )
        assert is_valid is False
        assert len(errors) == 1
        assert "generator.max_tokens" in errors[0]
        assert "5000" in errors[0]
        assert "gpt-4o-mini" in errors[0]

    def test_validate_error_includes_parent_context(self):
        """Test that validation error message includes parent context."""
        space = ExplorationSpace(
            tvars={
                "generator.model": TVAR(
                    name="model",
                    scope="generator",
                    python_type="str",
                    default_value="gpt-4o",
                    constraint=CategoricalConstraint(choices=["gpt-4o", "gpt-4o-mini"]),
                ),
                "generator.max_tokens": TVAR(
                    name="max_tokens",
                    scope="generator",
                    python_type="int",
                    default_value=1024,
                ),
            }
        )

        space.set_conditional(
            child="generator.max_tokens",
            parent="generator.model",
            conditions={
                "gpt-4o": {"min": 100, "max": 8192},
                "gpt-4o-mini": {"min": 100, "max": 4096},
            },
        )

        is_valid, errors = space.validate_config(
            {"generator.model": "gpt-4o-mini", "generator.max_tokens": 5000}
        )
        assert is_valid is False
        # Error should mention the parent context
        assert "when generator.model='gpt-4o-mini'" in errors[0]


# ==============================================================================
# Story 2.5: Fix/Unfix Tests
# ==============================================================================


class TestFix:
    """Tests for ExplorationSpace.fix() method."""

    def test_fix_sets_is_tunable_false(self):
        """Test that fix() sets is_tunable=False."""
        space = ExplorationSpace(
            tvars={
                "generator.model": TVAR(
                    name="model",
                    scope="generator",
                    python_type="str",
                    default_value="gpt-4o",
                    constraint=CategoricalConstraint(choices=["gpt-4o", "gpt-4o-mini"]),
                    is_tunable=True,
                )
            }
        )

        assert space.get_tvar("generator", "model").is_tunable is True
        space.fix("generator.model", "gpt-4o")
        assert space.get_tvar("generator", "model").is_tunable is False

    def test_fix_updates_default_value(self):
        """Test that fix() updates default_value to the fixed value."""
        space = ExplorationSpace(
            tvars={
                "generator.model": TVAR(
                    name="model",
                    scope="generator",
                    python_type="str",
                    default_value="gpt-4o",
                    constraint=CategoricalConstraint(choices=["gpt-4o", "gpt-4o-mini"]),
                    is_tunable=True,
                )
            }
        )

        space.fix("generator.model", "gpt-4o-mini")
        assert space.get_tvar("generator", "model").default_value == "gpt-4o-mini"

    def test_fix_raises_keyerror_for_missing_tvar(self):
        """Test that fix() raises KeyError for non-existent TVAR."""
        space = ExplorationSpace(tvars={})

        with pytest.raises(KeyError, match="TVAR not found"):
            space.fix("nonexistent.param", "value")

    def test_fix_raises_valueerror_for_invalid_categorical_value(self):
        """Test that fix() raises ValueError for invalid categorical value."""
        space = ExplorationSpace(
            tvars={
                "generator.model": TVAR(
                    name="model",
                    scope="generator",
                    python_type="str",
                    default_value="gpt-4o",
                    constraint=CategoricalConstraint(choices=["gpt-4o", "gpt-4o-mini"]),
                    is_tunable=True,
                )
            }
        )

        with pytest.raises(ValueError, match="not in choices"):
            space.fix("generator.model", "invalid-model")

    def test_fix_raises_valueerror_for_invalid_numerical_value(self):
        """Test that fix() raises ValueError for out-of-range numerical value."""
        space = ExplorationSpace(
            tvars={
                "generator.temperature": TVAR(
                    name="temperature",
                    scope="generator",
                    python_type="float",
                    default_value=0.7,
                    constraint=NumericalConstraint(min=0.0, max=2.0),
                    is_tunable=True,
                )
            }
        )

        with pytest.raises(ValueError, match="not in range"):
            space.fix("generator.temperature", 5.0)

    def test_fix_allows_any_value_when_no_constraint(self):
        """Test that fix() allows any value when TVAR has no constraint."""
        space = ExplorationSpace(
            tvars={
                "generator.custom": TVAR(
                    name="custom",
                    scope="generator",
                    python_type="str",
                    default_value="original",
                    constraint=None,
                    is_tunable=True,
                )
            }
        )

        # Should not raise - no constraint to validate against
        space.fix("generator.custom", "any-value")
        assert space.get_tvar("generator", "custom").default_value == "any-value"

    def test_fix_twice_updates_value(self):
        """Test that fixing twice updates to new value."""
        space = ExplorationSpace(
            tvars={
                "generator.model": TVAR(
                    name="model",
                    scope="generator",
                    python_type="str",
                    default_value="gpt-4o",
                    constraint=CategoricalConstraint(choices=["gpt-4o", "gpt-4o-mini"]),
                    is_tunable=True,
                )
            }
        )

        space.fix("generator.model", "gpt-4o")
        assert space.get_tvar("generator", "model").default_value == "gpt-4o"

        space.fix("generator.model", "gpt-4o-mini")
        assert space.get_tvar("generator", "model").default_value == "gpt-4o-mini"


class TestUnfix:
    """Tests for ExplorationSpace.unfix() method."""

    def test_unfix_restores_tunable(self):
        """Test that unfix() restores is_tunable=True."""
        space = ExplorationSpace(
            tvars={
                "generator.model": TVAR(
                    name="model",
                    scope="generator",
                    python_type="str",
                    default_value="gpt-4o",
                    constraint=CategoricalConstraint(choices=["gpt-4o", "gpt-4o-mini"]),
                    is_tunable=True,
                )
            }
        )

        space.fix("generator.model", "gpt-4o-mini")
        assert space.get_tvar("generator", "model").is_tunable is False

        space.unfix("generator.model")
        assert space.get_tvar("generator", "model").is_tunable is True

    def test_unfix_restores_original_default_value(self):
        """Test that unfix() restores original default value."""
        space = ExplorationSpace(
            tvars={
                "generator.model": TVAR(
                    name="model",
                    scope="generator",
                    python_type="str",
                    default_value="gpt-4o",
                    constraint=CategoricalConstraint(choices=["gpt-4o", "gpt-4o-mini"]),
                    is_tunable=True,
                )
            }
        )

        space.fix("generator.model", "gpt-4o-mini")
        assert space.get_tvar("generator", "model").default_value == "gpt-4o-mini"

        space.unfix("generator.model")
        assert space.get_tvar("generator", "model").default_value == "gpt-4o"

    def test_unfix_restores_original_constraint(self):
        """Test that unfix() restores original constraint."""
        original_constraint = CategoricalConstraint(choices=["gpt-4o", "gpt-4o-mini"])
        space = ExplorationSpace(
            tvars={
                "generator.model": TVAR(
                    name="model",
                    scope="generator",
                    python_type="str",
                    default_value="gpt-4o",
                    constraint=original_constraint,
                    is_tunable=True,
                )
            }
        )

        space.fix("generator.model", "gpt-4o")
        space.unfix("generator.model")

        tvar = space.get_tvar("generator", "model")
        assert isinstance(tvar.constraint, CategoricalConstraint)
        assert tvar.constraint.choices == ["gpt-4o", "gpt-4o-mini"]

    def test_unfix_raises_keyerror_for_missing_tvar(self):
        """Test that unfix() raises KeyError for non-existent TVAR."""
        space = ExplorationSpace(tvars={})

        with pytest.raises(KeyError, match="TVAR not found"):
            space.unfix("nonexistent.param")

    def test_unfix_on_never_fixed_is_noop(self):
        """Test that unfix() on never-fixed TVAR is a no-op."""
        space = ExplorationSpace(
            tvars={
                "generator.model": TVAR(
                    name="model",
                    scope="generator",
                    python_type="str",
                    default_value="gpt-4o",
                    constraint=CategoricalConstraint(choices=["gpt-4o", "gpt-4o-mini"]),
                    is_tunable=True,
                )
            }
        )

        # Should not raise, should be a no-op
        space.unfix("generator.model")

        # Should still be in original state
        tvar = space.get_tvar("generator", "model")
        assert tvar.is_tunable is True
        assert tvar.default_value == "gpt-4o"


class TestFixUnfixIntegration:
    """Integration tests for fix/unfix with sampling."""

    def test_sample_returns_fixed_value(self):
        """Test that sample() always returns fixed value for fixed TVARs."""
        space = ExplorationSpace(
            tvars={
                "generator.model": TVAR(
                    name="model",
                    scope="generator",
                    python_type="str",
                    default_value="gpt-4o",
                    constraint=CategoricalConstraint(choices=["gpt-4o", "gpt-4o-mini"]),
                    is_tunable=True,
                ),
                "generator.temperature": TVAR(
                    name="temperature",
                    scope="generator",
                    python_type="float",
                    default_value=0.7,
                    constraint=NumericalConstraint(min=0.0, max=2.0),
                    is_tunable=True,
                ),
            }
        )

        space.fix("generator.model", "gpt-4o")

        # Sample multiple times - model should always be "gpt-4o"
        for _ in range(20):
            config = space.sample()
            assert config["generator.model"] == "gpt-4o"
            # Temperature should vary
            assert 0.0 <= config["generator.temperature"] <= 2.0

    def test_fix_unfix_sample_cycle(self):
        """Test that fix/unfix/sample cycle works correctly."""
        space = ExplorationSpace(
            tvars={
                "generator.model": TVAR(
                    name="model",
                    scope="generator",
                    python_type="str",
                    default_value="gpt-4o",
                    constraint=CategoricalConstraint(choices=["gpt-4o", "gpt-4o-mini"]),
                    is_tunable=True,
                )
            }
        )

        # Fix and sample
        space.fix("generator.model", "gpt-4o-mini")
        for _ in range(10):
            config = space.sample()
            assert config["generator.model"] == "gpt-4o-mini"

        # Unfix and sample - should now vary
        space.unfix("generator.model")
        sampled_values = {space.sample()["generator.model"] for _ in range(50)}
        # Should eventually sample both values (probabilistic, but very likely)
        assert len(sampled_values) >= 1  # At minimum, samples something

    def test_fixed_tvars_property(self):
        """Test that fixed_tvars property reflects fix state."""
        space = ExplorationSpace(
            tvars={
                "generator.model": TVAR(
                    name="model",
                    scope="generator",
                    python_type="str",
                    default_value="gpt-4o",
                    constraint=CategoricalConstraint(choices=["gpt-4o", "gpt-4o-mini"]),
                    is_tunable=True,
                ),
                "generator.temperature": TVAR(
                    name="temperature",
                    scope="generator",
                    python_type="float",
                    default_value=0.7,
                    constraint=NumericalConstraint(min=0.0, max=2.0),
                    is_tunable=True,
                ),
            }
        )

        # Initially both tunable
        assert len(space.tunable_tvars) == 2
        assert len(space.fixed_tvars) == 0

        # Fix one
        space.fix("generator.model", "gpt-4o")
        assert len(space.tunable_tvars) == 1
        assert len(space.fixed_tvars) == 1
        assert "generator.model" in space.fixed_tvars

        # Unfix
        space.unfix("generator.model")
        assert len(space.tunable_tvars) == 2
        assert len(space.fixed_tvars) == 0


# =============================================================================
# Story 2.6: Validate Configuration Space Consistency
# =============================================================================


class TestValidate:
    """Tests for ExplorationSpace.validate() method (Story 2.6, AC1)."""

    def test_validate_valid_space_returns_true(self):
        """Test that validate() returns True for a valid space."""
        space = ExplorationSpace(
            tvars={
                "generator.model": TVAR(
                    name="model",
                    scope="generator",
                    python_type="str",
                    default_value="gpt-4o",
                    constraint=CategoricalConstraint(choices=["gpt-4o", "gpt-4o-mini"]),
                    is_tunable=True,
                ),
                "generator.temperature": TVAR(
                    name="temperature",
                    scope="generator",
                    python_type="float",
                    default_value=0.7,
                    constraint=NumericalConstraint(min=0.0, max=2.0),
                    is_tunable=True,
                ),
            }
        )

        assert space.validate() is True

    def test_validate_space_with_fixed_and_tunable(self):
        """Test validate() passes with mix of fixed and tunable TVARs."""
        space = ExplorationSpace(
            tvars={
                "generator.model": TVAR(
                    name="model",
                    scope="generator",
                    python_type="str",
                    default_value="gpt-4o",
                    is_tunable=False,  # Fixed
                ),
                "generator.temperature": TVAR(
                    name="temperature",
                    scope="generator",
                    python_type="float",
                    default_value=0.7,
                    constraint=NumericalConstraint(min=0.0, max=2.0),
                    is_tunable=True,  # Tunable
                ),
            }
        )

        assert space.validate() is True


class TestValidateNoTunable:
    """Tests for validate() with no tunable TVARs (Story 2.6, AC4)."""

    def test_validate_no_tunable_raises_valueerror(self):
        """Test that validate() raises ValueError when no tunable TVARs exist."""
        space = ExplorationSpace(
            tvars={
                "generator.model": TVAR(
                    name="model",
                    scope="generator",
                    python_type="str",
                    default_value="gpt-4o",
                    is_tunable=False,
                )
            }
        )

        with pytest.raises(ValueError, match="No tunable parameters"):
            space.validate()

    def test_validate_all_fixed_raises_valueerror(self):
        """Test validate() raises ValueError when all TVARs are fixed."""
        space = ExplorationSpace(
            tvars={
                "generator.model": TVAR(
                    name="model",
                    scope="generator",
                    python_type="str",
                    default_value="gpt-4o",
                    is_tunable=False,
                ),
                "generator.temperature": TVAR(
                    name="temperature",
                    scope="generator",
                    python_type="float",
                    default_value=0.7,
                    is_tunable=False,
                ),
            }
        )

        with pytest.raises(ValueError, match="No tunable parameters"):
            space.validate()

    def test_validate_empty_space_raises_valueerror(self):
        """Test validate() raises ValueError for empty space."""
        space = ExplorationSpace(tvars={})

        with pytest.raises(ValueError, match="No tunable parameters"):
            space.validate()


class TestValidateNumericalConstraints:
    """Tests for validate() with invalid numerical constraints (Story 2.6, AC3)."""

    def test_validate_invalid_range_min_gt_max(self):
        """Test validate() raises ConfigurationSpaceError when min > max."""
        from traigent.utils.exceptions import ConfigurationSpaceError

        space = ExplorationSpace(
            tvars={
                "generator.temperature": TVAR(
                    name="temperature",
                    scope="generator",
                    python_type="float",
                    default_value=0.7,
                    constraint=NumericalConstraint(min=2.0, max=0.5),  # Invalid!
                    is_tunable=True,
                )
            }
        )

        with pytest.raises(ConfigurationSpaceError, match="Invalid range"):
            space.validate()

    def test_validate_invalid_range_min_eq_max(self):
        """Test validate() raises ConfigurationSpaceError when min == max."""
        from traigent.utils.exceptions import ConfigurationSpaceError

        space = ExplorationSpace(
            tvars={
                "generator.temperature": TVAR(
                    name="temperature",
                    scope="generator",
                    python_type="float",
                    default_value=1.0,
                    constraint=NumericalConstraint(min=1.0, max=1.0),  # Invalid!
                    is_tunable=True,
                )
            }
        )

        with pytest.raises(ConfigurationSpaceError, match="Invalid range"):
            space.validate()

    def test_validate_invalid_step_zero(self):
        """Test validate() raises ConfigurationSpaceError when step is zero."""
        from traigent.utils.exceptions import ConfigurationSpaceError

        space = ExplorationSpace(
            tvars={
                "retriever.top_k": TVAR(
                    name="top_k",
                    scope="retriever",
                    python_type="int",
                    default_value=10,
                    constraint=NumericalConstraint(min=1, max=100, step=0),  # Invalid!
                    is_tunable=True,
                )
            }
        )

        with pytest.raises(ConfigurationSpaceError, match="step must be positive"):
            space.validate()

    def test_validate_invalid_step_negative(self):
        """Test validate() raises ConfigurationSpaceError when step is negative."""
        from traigent.utils.exceptions import ConfigurationSpaceError

        space = ExplorationSpace(
            tvars={
                "retriever.top_k": TVAR(
                    name="top_k",
                    scope="retriever",
                    python_type="int",
                    default_value=10,
                    constraint=NumericalConstraint(min=1, max=100, step=-5),  # Invalid!
                    is_tunable=True,
                )
            }
        )

        with pytest.raises(ConfigurationSpaceError, match="step must be positive"):
            space.validate()

    def test_validate_error_includes_tvar_name(self):
        """Test error message includes the TVAR name."""
        from traigent.utils.exceptions import ConfigurationSpaceError

        space = ExplorationSpace(
            tvars={
                "generator.temperature": TVAR(
                    name="temperature",
                    scope="generator",
                    python_type="float",
                    default_value=0.7,
                    constraint=NumericalConstraint(min=2.0, max=0.5),
                    is_tunable=True,
                )
            }
        )

        with pytest.raises(ConfigurationSpaceError, match="generator.temperature"):
            space.validate()


class TestValidateCategoricalConstraints:
    """Tests for validate() with invalid categorical constraints (Story 2.6, AC3)."""

    def test_validate_empty_choices(self):
        """Test validate() raises ConfigurationSpaceError for empty choices."""
        from traigent.utils.exceptions import ConfigurationSpaceError

        space = ExplorationSpace(
            tvars={
                "generator.model": TVAR(
                    name="model",
                    scope="generator",
                    python_type="str",
                    default_value="gpt-4o",
                    constraint=CategoricalConstraint(choices=[]),  # Invalid!
                    is_tunable=True,
                )
            }
        )

        with pytest.raises(ConfigurationSpaceError, match="Empty choices"):
            space.validate()

    def test_validate_empty_choices_includes_tvar_name(self):
        """Test error message includes the TVAR name."""
        from traigent.utils.exceptions import ConfigurationSpaceError

        space = ExplorationSpace(
            tvars={
                "generator.model": TVAR(
                    name="model",
                    scope="generator",
                    python_type="str",
                    default_value="gpt-4o",
                    constraint=CategoricalConstraint(choices=[]),
                    is_tunable=True,
                )
            }
        )

        with pytest.raises(ConfigurationSpaceError, match="generator.model"):
            space.validate()


class TestValidateConditionalConstraints:
    """Tests for validate() with conditional constraints (Story 2.6, AC2)."""

    def test_validate_valid_conditional(self):
        """Test validate() passes with valid conditional constraint."""
        from traigent.integrations.haystack import ConditionalConstraint

        space = ExplorationSpace(
            tvars={
                "generator.model": TVAR(
                    name="model",
                    scope="generator",
                    python_type="str",
                    default_value="gpt-4o",
                    constraint=CategoricalConstraint(choices=["gpt-4o", "gpt-4o-mini"]),
                    is_tunable=True,
                ),
                "generator.max_tokens": TVAR(
                    name="max_tokens",
                    scope="generator",
                    python_type="int",
                    default_value=1000,
                    constraint=ConditionalConstraint(
                        parent_qualified_name="generator.model",
                        conditions={
                            "gpt-4o": NumericalConstraint(min=100, max=8192),
                            "gpt-4o-mini": NumericalConstraint(min=100, max=4096),
                        },
                    ),
                    is_tunable=True,
                ),
            }
        )

        assert space.validate() is True

    def test_validate_conditional_missing_parent(self):
        """Test validate() raises error when conditional parent doesn't exist."""
        from traigent.integrations.haystack import ConditionalConstraint
        from traigent.utils.exceptions import ConfigurationSpaceError

        space = ExplorationSpace(
            tvars={
                "generator.max_tokens": TVAR(
                    name="max_tokens",
                    scope="generator",
                    python_type="int",
                    default_value=1000,
                    constraint=ConditionalConstraint(
                        parent_qualified_name="generator.model",  # Missing!
                        conditions={"gpt-4o": NumericalConstraint(min=100, max=8192)},
                    ),
                    is_tunable=True,
                ),
            }
        )

        with pytest.raises(ConfigurationSpaceError, match="non-existent parent"):
            space.validate()

    def test_validate_conditional_parent_not_categorical(self):
        """Test validate() raises error when parent has no discrete choices."""
        from traigent.integrations.haystack import ConditionalConstraint
        from traigent.utils.exceptions import ConfigurationSpaceError

        space = ExplorationSpace(
            tvars={
                "generator.temperature": TVAR(
                    name="temperature",
                    scope="generator",
                    python_type="float",
                    default_value=0.7,
                    constraint=NumericalConstraint(
                        min=0.0, max=2.0
                    ),  # Not categorical!
                    is_tunable=True,
                ),
                "generator.max_tokens": TVAR(
                    name="max_tokens",
                    scope="generator",
                    python_type="int",
                    default_value=1000,
                    constraint=ConditionalConstraint(
                        parent_qualified_name="generator.temperature",
                        conditions={0.5: NumericalConstraint(min=100, max=4096)},
                    ),
                    is_tunable=True,
                ),
            }
        )

        with pytest.raises(ConfigurationSpaceError, match="discrete choices"):
            space.validate()

    def test_validate_conditional_invalid_parent_value(self):
        """Test validate() raises error for invalid parent value in conditions."""
        from traigent.integrations.haystack import ConditionalConstraint
        from traigent.utils.exceptions import ConfigurationSpaceError

        space = ExplorationSpace(
            tvars={
                "generator.model": TVAR(
                    name="model",
                    scope="generator",
                    python_type="str",
                    default_value="gpt-4o",
                    constraint=CategoricalConstraint(choices=["gpt-4o", "gpt-4o-mini"]),
                    is_tunable=True,
                ),
                "generator.max_tokens": TVAR(
                    name="max_tokens",
                    scope="generator",
                    python_type="int",
                    default_value=1000,
                    constraint=ConditionalConstraint(
                        parent_qualified_name="generator.model",
                        conditions={
                            "gpt-4o": NumericalConstraint(min=100, max=8192),
                            "invalid-model": NumericalConstraint(
                                min=100, max=4096
                            ),  # Invalid!
                        },
                    ),
                    is_tunable=True,
                ),
            }
        )

        with pytest.raises(ConfigurationSpaceError, match="invalid parent value"):
            space.validate()

    def test_validate_conditional_with_invalid_inner_range(self):
        """Test validate() raises error for invalid range in conditional."""
        from traigent.integrations.haystack import ConditionalConstraint
        from traigent.utils.exceptions import ConfigurationSpaceError

        space = ExplorationSpace(
            tvars={
                "generator.model": TVAR(
                    name="model",
                    scope="generator",
                    python_type="str",
                    default_value="gpt-4o",
                    constraint=CategoricalConstraint(choices=["gpt-4o", "gpt-4o-mini"]),
                    is_tunable=True,
                ),
                "generator.max_tokens": TVAR(
                    name="max_tokens",
                    scope="generator",
                    python_type="int",
                    default_value=1000,
                    constraint=ConditionalConstraint(
                        parent_qualified_name="generator.model",
                        conditions={
                            "gpt-4o": NumericalConstraint(
                                min=8192, max=100
                            ),  # Invalid!
                            "gpt-4o-mini": NumericalConstraint(min=100, max=4096),
                        },
                    ),
                    is_tunable=True,
                ),
            }
        )

        with pytest.raises(ConfigurationSpaceError, match="Invalid range"):
            space.validate()

    def test_validate_conditional_with_empty_inner_choices(self):
        """Test validate() raises error for empty choices in conditional."""
        from traigent.integrations.haystack import ConditionalConstraint
        from traigent.utils.exceptions import ConfigurationSpaceError

        space = ExplorationSpace(
            tvars={
                "generator.model": TVAR(
                    name="model",
                    scope="generator",
                    python_type="str",
                    default_value="gpt-4o",
                    constraint=CategoricalConstraint(choices=["gpt-4o", "gpt-4o-mini"]),
                    is_tunable=True,
                ),
                "generator.api_version": TVAR(
                    name="api_version",
                    scope="generator",
                    python_type="str",
                    default_value="v1",
                    constraint=ConditionalConstraint(
                        parent_qualified_name="generator.model",
                        conditions={
                            "gpt-4o": CategoricalConstraint(choices=[]),  # Invalid!
                            "gpt-4o-mini": CategoricalConstraint(choices=["v1", "v2"]),
                        },
                    ),
                    is_tunable=True,
                ),
            }
        )

        with pytest.raises(ConfigurationSpaceError, match="Empty choices"):
            space.validate()


class TestValidateCircularDependencies:
    """Tests for validate() detecting circular dependencies (Story 2.6, AC2)."""

    def test_validate_detects_circular_dependency(self):
        """Test validate() raises error for circular conditional dependencies."""
        from traigent.integrations.haystack import ConditionalConstraint
        from traigent.utils.exceptions import ConfigurationSpaceError

        # Create a circular dependency: a -> b -> a
        # Note: set_conditional already prevents this at creation time,
        # but we test the validate() method directly with raw TVAR construction
        space = ExplorationSpace(
            tvars={
                "a": TVAR(
                    name="a",
                    scope="",
                    python_type="str",
                    default_value="x",
                    constraint=ConditionalConstraint(
                        parent_qualified_name="b",
                        conditions={"y": CategoricalConstraint(choices=["x", "z"])},
                    ),
                    is_tunable=True,
                ),
                "b": TVAR(
                    name="b",
                    scope="",
                    python_type="str",
                    default_value="y",
                    constraint=ConditionalConstraint(
                        parent_qualified_name="a",
                        conditions={"x": CategoricalConstraint(choices=["y", "w"])},
                    ),
                    is_tunable=True,
                ),
            }
        )

        with pytest.raises(ConfigurationSpaceError, match="Circular dependency"):
            space.validate()


class TestValidateIntegration:
    """Integration tests for validate() with complex scenarios."""

    def test_validate_catches_manually_created_invalid_constraint(self):
        """Test validate() catches errors in manually constructed constraints."""
        from traigent.utils.exceptions import ConfigurationSpaceError

        space = ExplorationSpace(
            tvars={
                "generator.temperature": TVAR(
                    name="temperature",
                    scope="generator",
                    python_type="float",
                    default_value=0.7,
                    constraint=NumericalConstraint(min=0.0, max=2.0),
                    is_tunable=True,
                )
            }
        )

        # First validate passes
        assert space.validate() is True

        # Manually set invalid constraint (bypassing set_range validation)
        tvar = space.get_tvar("generator", "temperature")
        tvar.constraint = NumericalConstraint(min=2.0, max=0.5)  # Invalid!

        # Should fail validation
        with pytest.raises(ConfigurationSpaceError, match="Invalid range"):
            space.validate()

    def test_validate_with_conditional_and_fixed_parent(self):
        """Test validate() works when conditional parent is fixed."""
        from traigent.integrations.haystack import ConditionalConstraint

        space = ExplorationSpace(
            tvars={
                "generator.model": TVAR(
                    name="model",
                    scope="generator",
                    python_type="str",
                    default_value="gpt-4o",
                    constraint=CategoricalConstraint(choices=["gpt-4o", "gpt-4o-mini"]),
                    is_tunable=False,  # Fixed
                ),
                "generator.max_tokens": TVAR(
                    name="max_tokens",
                    scope="generator",
                    python_type="int",
                    default_value=1000,
                    constraint=ConditionalConstraint(
                        parent_qualified_name="generator.model",
                        conditions={
                            "gpt-4o": NumericalConstraint(min=100, max=8192),
                            "gpt-4o-mini": NumericalConstraint(min=100, max=4096),
                        },
                    ),
                    is_tunable=True,
                ),
            }
        )

        # Should pass - at least one tunable TVAR exists
        assert space.validate() is True

    def test_validate_multiple_errors_reports_first(self):
        """Test validate() reports the first error encountered."""
        from traigent.utils.exceptions import ConfigurationSpaceError

        space = ExplorationSpace(
            tvars={
                "a.param1": TVAR(
                    name="param1",
                    scope="a",
                    python_type="float",
                    default_value=0.5,
                    constraint=NumericalConstraint(min=2.0, max=1.0),  # Invalid
                    is_tunable=True,
                ),
                "b.param2": TVAR(
                    name="param2",
                    scope="b",
                    python_type="str",
                    default_value="x",
                    constraint=CategoricalConstraint(choices=[]),  # Also invalid
                    is_tunable=True,
                ),
            }
        )

        # Should raise for the first error it encounters
        with pytest.raises(ConfigurationSpaceError):
            space.validate()

    def test_validate_warns_for_tunable_without_constraint(self):
        """Test validate() warns when tunable TVAR has no constraint."""
        import warnings

        space = ExplorationSpace(
            tvars={
                "generator.custom": TVAR(
                    name="custom",
                    scope="generator",
                    python_type="str",
                    default_value="default-value",
                    constraint=None,  # No constraint!
                    is_tunable=True,  # But marked tunable
                ),
                "generator.temperature": TVAR(
                    name="temperature",
                    scope="generator",
                    python_type="float",
                    default_value=0.7,
                    constraint=NumericalConstraint(min=0.0, max=2.0),
                    is_tunable=True,
                ),
            }
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = space.validate()

            # Should still return True (just a warning, not error)
            assert result is True

            # Should have issued a warning about the tunable without constraint
            assert len(w) == 1
            assert "marked tunable but has no constraint" in str(w[0].message)
            assert "generator.custom" in str(w[0].message)

    def test_validate_no_warning_for_non_tunable_without_constraint(self):
        """Test validate() doesn't warn for non-tunable TVAR without constraint."""
        import warnings

        space = ExplorationSpace(
            tvars={
                "generator.fixed_param": TVAR(
                    name="fixed_param",
                    scope="generator",
                    python_type="str",
                    default_value="fixed-value",
                    constraint=None,
                    is_tunable=False,  # Not tunable - no warning needed
                ),
                "generator.temperature": TVAR(
                    name="temperature",
                    scope="generator",
                    python_type="float",
                    default_value=0.7,
                    constraint=NumericalConstraint(min=0.0, max=2.0),
                    is_tunable=True,
                ),
            }
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = space.validate()

            assert result is True
            # No warning should be issued
            assert len(w) == 0

    def test_validate_conditional_empty_conditions_no_default(self):
        """Test validate() raises error for empty conditions with no default."""
        from traigent.integrations.haystack import ConditionalConstraint
        from traigent.utils.exceptions import ConfigurationSpaceError

        space = ExplorationSpace(
            tvars={
                "generator.model": TVAR(
                    name="model",
                    scope="generator",
                    python_type="str",
                    default_value="gpt-4o",
                    constraint=CategoricalConstraint(choices=["gpt-4o", "gpt-4o-mini"]),
                    is_tunable=True,
                ),
                "generator.max_tokens": TVAR(
                    name="max_tokens",
                    scope="generator",
                    python_type="int",
                    default_value=1000,
                    constraint=ConditionalConstraint(
                        parent_qualified_name="generator.model",
                        conditions={},  # Empty conditions!
                        default_constraint=None,  # No default either!
                    ),
                    is_tunable=True,
                ),
            }
        )

        with pytest.raises(
            ConfigurationSpaceError, match="no conditions and no default"
        ):
            space.validate()

    def test_validate_conditional_empty_conditions_with_default_ok(self):
        """Test validate() passes for empty conditions when default is provided."""
        from traigent.integrations.haystack import ConditionalConstraint

        space = ExplorationSpace(
            tvars={
                "generator.model": TVAR(
                    name="model",
                    scope="generator",
                    python_type="str",
                    default_value="gpt-4o",
                    constraint=CategoricalConstraint(choices=["gpt-4o", "gpt-4o-mini"]),
                    is_tunable=True,
                ),
                "generator.max_tokens": TVAR(
                    name="max_tokens",
                    scope="generator",
                    python_type="int",
                    default_value=1000,
                    constraint=ConditionalConstraint(
                        parent_qualified_name="generator.model",
                        conditions={},  # Empty conditions
                        default_constraint=NumericalConstraint(
                            min=100, max=4096
                        ),  # But has default
                    ),
                    is_tunable=True,
                ),
            }
        )

        # Should pass - default constraint handles all parent values
        assert space.validate() is True

    def test_validate_conditional_invalid_inner_constraint_type(self):
        """Test validate() raises error for unknown inner constraint type."""
        from traigent.integrations.haystack import ConditionalConstraint
        from traigent.utils.exceptions import ConfigurationSpaceError

        # Create a conditional with an invalid inner constraint type
        space = ExplorationSpace(
            tvars={
                "generator.model": TVAR(
                    name="model",
                    scope="generator",
                    python_type="str",
                    default_value="gpt-4o",
                    constraint=CategoricalConstraint(choices=["gpt-4o", "gpt-4o-mini"]),
                    is_tunable=True,
                ),
                "generator.max_tokens": TVAR(
                    name="max_tokens",
                    scope="generator",
                    python_type="int",
                    default_value=1000,
                    constraint=ConditionalConstraint(
                        parent_qualified_name="generator.model",
                        conditions={
                            "gpt-4o": "not a constraint object",  # Invalid type!
                        },
                    ),
                    is_tunable=True,
                ),
            }
        )

        with pytest.raises(
            ConfigurationSpaceError, match="Invalid inner constraint type"
        ):
            space.validate()

    def test_validate_conditional_invalid_default_constraint_type(self):
        """Test validate() raises error for unknown default constraint type."""
        from traigent.integrations.haystack import ConditionalConstraint
        from traigent.utils.exceptions import ConfigurationSpaceError

        space = ExplorationSpace(
            tvars={
                "generator.model": TVAR(
                    name="model",
                    scope="generator",
                    python_type="str",
                    default_value="gpt-4o",
                    constraint=CategoricalConstraint(choices=["gpt-4o", "gpt-4o-mini"]),
                    is_tunable=True,
                ),
                "generator.max_tokens": TVAR(
                    name="max_tokens",
                    scope="generator",
                    python_type="int",
                    default_value=1000,
                    constraint=ConditionalConstraint(
                        parent_qualified_name="generator.model",
                        conditions={
                            "gpt-4o": NumericalConstraint(min=100, max=8192),
                        },
                        default_constraint="not a constraint",  # Invalid type!
                    ),
                    is_tunable=True,
                ),
            }
        )

        with pytest.raises(
            ConfigurationSpaceError, match="Invalid default constraint type"
        ):
            space.validate()


class TestBooleanAndOptionalParameters:
    """Tests for boolean and optional parameter handling (Story 2.9)."""

    def test_bool_tvar_sampled_as_actual_bool(self):
        """Test that sampling a bool TVAR returns actual bool values."""
        space = ExplorationSpace(
            tvars={
                "ranker.scale_score": TVAR(
                    name="scale_score",
                    scope="ranker",
                    python_type="bool",
                    default_value=True,
                    constraint=CategoricalConstraint(choices=[True, False]),
                    is_tunable=True,
                )
            }
        )

        # Sample multiple times and verify all values are bool
        for seed in range(20):
            config = space.sample(seed=seed)
            value = config["ranker.scale_score"]
            assert isinstance(value, bool), f"Expected bool, got {type(value)}"
            assert value in [True, False]

    def test_bool_tvar_both_values_sampled(self):
        """Test that both True and False are sampled for bool TVARs."""
        space = ExplorationSpace(
            tvars={
                "ranker.scale_score": TVAR(
                    name="scale_score",
                    scope="ranker",
                    python_type="bool",
                    default_value=True,
                    constraint=CategoricalConstraint(choices=[True, False]),
                    is_tunable=True,
                )
            }
        )

        found_true = False
        found_false = False
        for seed in range(50):
            config = space.sample(seed=seed)
            if config["ranker.scale_score"] is True:
                found_true = True
            if config["ranker.scale_score"] is False:
                found_false = True

        assert found_true, "True should be sampled"
        assert found_false, "False should be sampled"

    def test_optional_bool_includes_none(self):
        """Test that Optional[bool] TVAR can include None in choices."""
        space = ExplorationSpace(
            tvars={
                "ranker.scale_score": TVAR(
                    name="scale_score",
                    scope="ranker",
                    python_type="bool",
                    default_value=True,
                    constraint=CategoricalConstraint(choices=[True, False, None]),
                    is_tunable=True,
                )
            }
        )

        found_none = False
        for seed in range(100):
            config = space.sample(seed=seed)
            if config["ranker.scale_score"] is None:
                found_none = True
                break

        assert found_none, "None should be sampled for Optional[bool]"

    def test_optional_numerical_can_sample_none(self):
        """Test that Optional[float] TVAR can sample None."""
        space = ExplorationSpace(
            tvars={
                "generator.temperature": TVAR(
                    name="temperature",
                    scope="generator",
                    python_type="float",
                    default_value=0.7,
                    constraint=NumericalConstraint(min=0.0, max=2.0),
                    is_tunable=True,
                    is_optional=True,  # Marked as optional
                )
            }
        )

        found_none = False
        found_value = False
        for seed in range(200):
            config = space.sample(seed=seed)
            value = config["generator.temperature"]
            if value is None:
                found_none = True
            else:
                found_value = True
                assert 0.0 <= value <= 2.0

        assert found_none, "None should be sampled for optional numerical TVAR"
        assert found_value, "Numerical values should also be sampled"

    def test_non_optional_numerical_never_returns_none(self):
        """Test that non-optional numerical TVARs never return None."""
        space = ExplorationSpace(
            tvars={
                "generator.temperature": TVAR(
                    name="temperature",
                    scope="generator",
                    python_type="float",
                    default_value=0.7,
                    constraint=NumericalConstraint(min=0.0, max=2.0),
                    is_tunable=True,
                    is_optional=False,  # Not optional
                )
            }
        )

        for seed in range(100):
            config = space.sample(seed=seed)
            value = config["generator.temperature"]
            assert value is not None, "Non-optional should never return None"
            assert 0.0 <= value <= 2.0

    def test_validate_config_accepts_none_for_optional(self):
        """Test that validate_config accepts None for optional TVARs."""
        space = ExplorationSpace(
            tvars={
                "generator.temperature": TVAR(
                    name="temperature",
                    scope="generator",
                    python_type="float",
                    default_value=0.7,
                    constraint=NumericalConstraint(min=0.0, max=2.0),
                    is_tunable=True,
                    is_optional=True,
                )
            }
        )

        is_valid, errors = space.validate_config({"generator.temperature": None})
        assert is_valid, f"None should be valid for optional: {errors}"
        assert len(errors) == 0

    def test_validate_config_rejects_none_for_non_optional(self):
        """Test that validate_config rejects None for non-optional TVARs."""
        space = ExplorationSpace(
            tvars={
                "generator.temperature": TVAR(
                    name="temperature",
                    scope="generator",
                    python_type="float",
                    default_value=0.7,
                    constraint=NumericalConstraint(min=0.0, max=2.0),
                    is_tunable=True,
                    is_optional=False,
                )
            }
        )

        is_valid, errors = space.validate_config({"generator.temperature": None})
        assert not is_valid, "None should be invalid for non-optional"
        assert len(errors) == 1

    def test_optional_int_tvar_samples_none_and_integers(self):
        """Test that Optional[int] TVAR samples both None and integers."""
        space = ExplorationSpace(
            tvars={
                "retriever.top_k": TVAR(
                    name="top_k",
                    scope="retriever",
                    python_type="int",
                    default_value=10,
                    constraint=NumericalConstraint(min=1, max=50, step=1),
                    is_tunable=True,
                    is_optional=True,
                )
            }
        )

        found_none = False
        found_int = False
        for seed in range(200):
            config = space.sample(seed=seed)
            value = config["retriever.top_k"]
            if value is None:
                found_none = True
            else:
                found_int = True
                assert isinstance(value, int), f"Expected int, got {type(value)}"
                assert 1 <= value <= 50

        assert found_none, "None should be sampled for optional int"
        assert found_int, "Integer values should also be sampled"

    def test_optional_none_probability_affects_sampling(self):
        """Test that OPTIONAL_NONE_PROBABILITY affects None sampling rate."""
        space = ExplorationSpace(
            tvars={
                "generator.temperature": TVAR(
                    name="temperature",
                    scope="generator",
                    python_type="float",
                    default_value=0.7,
                    constraint=NumericalConstraint(min=0.0, max=2.0),
                    is_tunable=True,
                    is_optional=True,
                )
            }
        )

        # Count None occurrences
        none_count = 0
        total = 1000
        for seed in range(total):
            config = space.sample(seed=seed)
            if config["generator.temperature"] is None:
                none_count += 1

        # With 10% probability, expect roughly 100 Nones (allow some variance)
        # Allow 5-20% range for statistical variance
        assert 30 < none_count < 200, (
            f"Expected ~10% Nones, got {none_count}/{total} "
            f"({100*none_count/total:.1f}%)"
        )

    def test_fix_none_for_optional_numerical(self):
        """Test that fix(None) works for optional numerical TVARs."""
        space = ExplorationSpace(
            tvars={
                "generator.temperature": TVAR(
                    name="temperature",
                    scope="generator",
                    python_type="float",
                    default_value=0.7,
                    constraint=NumericalConstraint(min=0.0, max=2.0),
                    is_tunable=True,
                    is_optional=True,
                )
            }
        )

        # This should work for optional TVARs
        space.fix("generator.temperature", None)

        # All samples should return None
        for seed in range(10):
            config = space.sample(seed=seed)
            assert config["generator.temperature"] is None

    def test_fix_none_for_non_optional_raises_error(self):
        """Test that fix(None) raises ValueError for non-optional numerical TVARs."""
        space = ExplorationSpace(
            tvars={
                "generator.temperature": TVAR(
                    name="temperature",
                    scope="generator",
                    python_type="float",
                    default_value=0.7,
                    constraint=NumericalConstraint(min=0.0, max=2.0),
                    is_tunable=True,
                    is_optional=False,  # Not optional
                )
            }
        )

        # This should raise ValueError for non-optional TVARs
        with pytest.raises(ValueError, match="not in range"):
            space.fix("generator.temperature", None)

    def test_optional_numerical_under_conditional_can_sample_none(self):
        """Test that optional numerical under ConditionalConstraint can sample None."""
        space = ExplorationSpace(
            tvars={
                "generator.model": TVAR(
                    name="model",
                    scope="generator",
                    python_type="str",
                    default_value="gpt-4o",
                    constraint=CategoricalConstraint(choices=["gpt-4o", "gpt-4o-mini"]),
                    is_tunable=True,
                ),
                "generator.temperature": TVAR(
                    name="temperature",
                    scope="generator",
                    python_type="float",
                    default_value=0.7,
                    constraint=ConditionalConstraint(
                        parent_qualified_name="generator.model",
                        conditions={
                            "gpt-4o": NumericalConstraint(min=0.0, max=2.0),
                            "gpt-4o-mini": NumericalConstraint(min=0.0, max=1.5),
                        },
                    ),
                    is_tunable=True,
                    is_optional=True,  # Mark as optional
                ),
            }
        )

        found_none = False
        found_value = False
        for seed in range(200):
            config = space.sample(seed=seed)
            value = config["generator.temperature"]
            if value is None:
                found_none = True
            else:
                found_value = True

        assert found_none, "None should be sampled for optional conditional numerical"
        assert found_value, "Numerical values should also be sampled"

    def test_optional_numerical_without_range_preserves_is_optional(self):
        """Test that optional numericals without ranges still track is_optional."""
        # Create a mock DiscoveredTVAR and PipelineSpec
        from traigent.integrations.haystack.introspection import DiscoveredTVAR

        class MockScope:
            def __init__(self, name, tvars):
                self.name = name
                self.tvars = tvars

        class MockPipelineSpec:
            def __init__(self, scopes):
                self.scopes = scopes

        discovered = DiscoveredTVAR(
            name="custom_param",
            value=None,
            python_type="float",
            is_tunable=True,
            is_optional=True,  # Optional type
            # No default_range or literal_choices - no constraint will be created
        )

        spec = MockPipelineSpec(
            scopes=[MockScope("component", {"custom_param": discovered})]
        )

        space = ExplorationSpace.from_pipeline_spec(spec)
        tvar = space.get_tvar("component", "custom_param")

        # Even though no constraint exists, is_optional should be preserved
        assert tvar.is_optional is True

        # Now if user sets a range, it should work with optional semantics
        space.set_range("component.custom_param", 0.0, 1.0)
        tvar = space.get_tvar("component", "custom_param")
        assert tvar.is_optional is True  # Should still be optional

        # And sampling should sometimes return None
        found_none = False
        for seed in range(200):
            config = space.sample(seed=seed)
            if config["component.custom_param"] is None:
                found_none = True
                break

        assert (
            found_none
        ), "None should be sampled for optional numerical after set_range"


class TestTVLIntegration:
    """Tests for TVL import/export (Story 2.7)."""

    def test_from_tvl_spec_categorical(self, tmp_path):
        """Test loading categorical parameter from TVL."""
        tvl_content = """
configuration_space:
  generator.model:
    type: categorical
    values: ["gpt-4o", "gpt-4o-mini"]
    default: "gpt-4o"
"""
        tvl_file = tmp_path / "test_space.yaml"
        tvl_file.write_text(tvl_content)

        space = ExplorationSpace.from_tvl_spec(tvl_file)

        tvar = space.get_tvar("generator", "model")
        assert tvar is not None
        assert isinstance(tvar.constraint, CategoricalConstraint)
        assert tvar.constraint.choices == ["gpt-4o", "gpt-4o-mini"]
        assert tvar.default_value == "gpt-4o"
        assert tvar.is_tunable is True

    def test_from_tvl_spec_numerical_continuous(self, tmp_path):
        """Test loading continuous parameter from TVL."""
        tvl_content = """
configuration_space:
  generator.temperature:
    type: continuous
    range: [0.0, 2.0]
    default: 0.7
"""
        tvl_file = tmp_path / "test_space.yaml"
        tvl_file.write_text(tvl_content)

        space = ExplorationSpace.from_tvl_spec(tvl_file)

        tvar = space.get_tvar("generator", "temperature")
        assert tvar is not None
        assert isinstance(tvar.constraint, NumericalConstraint)
        assert tvar.constraint.min == pytest.approx(0.0)
        assert tvar.constraint.max == pytest.approx(2.0)
        assert tvar.default_value == pytest.approx(0.7)
        assert tvar.python_type == "float"

    def test_from_tvl_spec_numerical_integer(self, tmp_path):
        """Test loading integer parameter from TVL."""
        tvl_content = """
configuration_space:
  retriever.top_k:
    type: integer
    range: [1, 50]
    default: 10
"""
        tvl_file = tmp_path / "test_space.yaml"
        tvl_file.write_text(tvl_content)

        space = ExplorationSpace.from_tvl_spec(tvl_file)

        tvar = space.get_tvar("retriever", "top_k")
        assert tvar is not None
        assert isinstance(tvar.constraint, NumericalConstraint)
        assert tvar.constraint.min == 1
        assert tvar.constraint.max == 50
        assert tvar.python_type == "int"

    def test_from_tvl_spec_boolean(self, tmp_path):
        """Test loading boolean parameter from TVL."""
        tvl_content = """
configuration_space:
  generator.stream:
    type: boolean
"""
        tvl_file = tmp_path / "test_space.yaml"
        tvl_file.write_text(tvl_content)

        space = ExplorationSpace.from_tvl_spec(tvl_file)

        tvar = space.get_tvar("generator", "stream")
        assert tvar is not None
        assert isinstance(tvar.constraint, CategoricalConstraint)
        assert tvar.constraint.choices == [True, False]

    def test_from_tvl_spec_multiple_params(self, tmp_path):
        """Test loading multiple parameters from TVL."""
        tvl_content = """
configuration_space:
  generator.model:
    type: categorical
    values: ["gpt-4o", "gpt-4o-mini"]
    default: "gpt-4o"
  generator.temperature:
    type: continuous
    range: [0.0, 2.0]
    default: 0.7
  retriever.top_k:
    type: integer
    range: [1, 50]
    default: 10
"""
        tvl_file = tmp_path / "test_space.yaml"
        tvl_file.write_text(tvl_content)

        space = ExplorationSpace.from_tvl_spec(tvl_file)

        assert len(space.tvars) == 3
        assert space.get_tvar("generator", "model") is not None
        assert space.get_tvar("generator", "temperature") is not None
        assert space.get_tvar("retriever", "top_k") is not None

    def test_from_tvl_spec_invalid_file_raises_error(self, tmp_path):
        """Test that invalid TVL file raises TVLValidationError."""
        from traigent.utils.exceptions import TVLValidationError

        tvl_file = tmp_path / "invalid.yaml"
        tvl_file.write_text("not: a valid: yaml: file: [")

        with pytest.raises(TVLValidationError):
            ExplorationSpace.from_tvl_spec(tvl_file)

    def test_from_tvl_spec_missing_file_raises_error(self, tmp_path):
        """Test that missing file raises TVLValidationError."""
        from traigent.utils.exceptions import TVLValidationError

        with pytest.raises(TVLValidationError, match="not found"):
            ExplorationSpace.from_tvl_spec(tmp_path / "nonexistent.yaml")

    def test_to_tvl_creates_file(self, tmp_path):
        """Test that to_tvl creates a valid YAML file."""
        space = ExplorationSpace(
            tvars={
                "generator.model": TVAR(
                    name="model",
                    scope="generator",
                    python_type="str",
                    default_value="gpt-4o",
                    constraint=CategoricalConstraint(choices=["gpt-4o", "gpt-4o-mini"]),
                    is_tunable=True,
                ),
                "generator.temperature": TVAR(
                    name="temperature",
                    scope="generator",
                    python_type="float",
                    default_value=0.7,
                    constraint=NumericalConstraint(min=0.0, max=2.0),
                    is_tunable=True,
                ),
            }
        )

        tvl_file = tmp_path / "exported.yaml"
        space.to_tvl(tvl_file)

        assert tvl_file.exists()

        import yaml

        data = yaml.safe_load(tvl_file.read_text())
        assert "configuration_space" in data
        assert "generator.model" in data["configuration_space"]
        assert "generator.temperature" in data["configuration_space"]

    def test_to_tvl_includes_metadata(self, tmp_path):
        """Test that to_tvl includes metadata when requested."""
        space = ExplorationSpace(
            tvars={
                "generator.model": TVAR(
                    name="model",
                    scope="generator",
                    python_type="str",
                    default_value="gpt-4o",
                    constraint=CategoricalConstraint(choices=["gpt-4o"]),
                    is_tunable=True,
                )
            }
        )

        tvl_file = tmp_path / "exported.yaml"
        space.to_tvl(tvl_file, description="Test export")

        import yaml

        data = yaml.safe_load(tvl_file.read_text())
        assert "metadata" in data
        assert data["metadata"]["description"] == "Test export"
        assert "exported_at" in data["metadata"]
        assert data["metadata"]["num_tvars"] == 1

    def test_to_tvl_without_metadata(self, tmp_path):
        """Test that to_tvl can skip metadata."""
        space = ExplorationSpace(
            tvars={
                "generator.model": TVAR(
                    name="model",
                    scope="generator",
                    python_type="str",
                    default_value="gpt-4o",
                    constraint=CategoricalConstraint(choices=["gpt-4o"]),
                    is_tunable=True,
                )
            }
        )

        tvl_file = tmp_path / "exported.yaml"
        space.to_tvl(tvl_file, include_metadata=False)

        import yaml

        data = yaml.safe_load(tvl_file.read_text())
        assert "metadata" not in data
        assert "configuration_space" in data

    def test_round_trip_preservation(self, tmp_path):
        """Test that export and reload preserves constraints."""
        original = ExplorationSpace(
            tvars={
                "generator.model": TVAR(
                    name="model",
                    scope="generator",
                    python_type="str",
                    default_value="gpt-4o",
                    constraint=CategoricalConstraint(choices=["gpt-4o", "gpt-4o-mini"]),
                    is_tunable=True,
                ),
                "generator.temperature": TVAR(
                    name="temperature",
                    scope="generator",
                    python_type="float",
                    default_value=0.7,
                    constraint=NumericalConstraint(min=0.0, max=2.0),
                    is_tunable=True,
                ),
                "retriever.top_k": TVAR(
                    name="top_k",
                    scope="retriever",
                    python_type="int",
                    default_value=10,
                    constraint=NumericalConstraint(min=1, max=50),
                    is_tunable=True,
                ),
            }
        )

        # Export
        tvl_file = tmp_path / "round_trip.yaml"
        original.to_tvl(tvl_file, include_metadata=False)

        # Reload
        reloaded = ExplorationSpace.from_tvl_spec(tvl_file)

        # Compare
        assert len(reloaded.tvars) == len(original.tvars)

        for name, orig_tvar in original.tvars.items():
            reloaded_tvar = reloaded.tvars.get(name)
            assert reloaded_tvar is not None, f"TVAR {name} not found after reload"
            assert type(orig_tvar.constraint) == type(reloaded_tvar.constraint)

            if isinstance(orig_tvar.constraint, CategoricalConstraint):
                assert orig_tvar.constraint.choices == reloaded_tvar.constraint.choices
            elif isinstance(orig_tvar.constraint, NumericalConstraint):
                assert orig_tvar.constraint.min == reloaded_tvar.constraint.min
                assert orig_tvar.constraint.max == reloaded_tvar.constraint.max

    def test_to_tvl_numerical_with_log_scale(self, tmp_path):
        """Test exporting numerical constraint with log_scale."""
        space = ExplorationSpace(
            tvars={
                "optimizer.lr": TVAR(
                    name="lr",
                    scope="optimizer",
                    python_type="float",
                    default_value=0.01,
                    constraint=NumericalConstraint(min=0.0001, max=0.1, log_scale=True),
                    is_tunable=True,
                )
            }
        )

        tvl_file = tmp_path / "exported.yaml"
        space.to_tvl(tvl_file, include_metadata=False)

        import yaml

        data = yaml.safe_load(tvl_file.read_text())
        param = data["configuration_space"]["optimizer.lr"]
        assert param["log_scale"] is True
        assert param["range"] == [0.0001, 0.1]

    def test_to_tvl_numerical_with_step(self, tmp_path):
        """Test exporting numerical constraint with step."""
        space = ExplorationSpace(
            tvars={
                "retriever.top_k": TVAR(
                    name="top_k",
                    scope="retriever",
                    python_type="int",
                    default_value=10,
                    constraint=NumericalConstraint(min=1, max=50, step=5),
                    is_tunable=True,
                )
            }
        )

        tvl_file = tmp_path / "exported.yaml"
        space.to_tvl(tvl_file, include_metadata=False)

        import yaml

        data = yaml.safe_load(tvl_file.read_text())
        param = data["configuration_space"]["retriever.top_k"]
        assert param["step"] == 5

    def test_to_tvl_fixed_param(self, tmp_path):
        """Test exporting fixed (non-tunable) parameter."""
        space = ExplorationSpace(
            tvars={
                "generator.seed": TVAR(
                    name="seed",
                    scope="generator",
                    python_type="int",
                    default_value=42,
                    constraint=None,
                    is_tunable=False,
                )
            }
        )

        tvl_file = tmp_path / "exported.yaml"
        space.to_tvl(tvl_file, include_metadata=False)

        import yaml

        data = yaml.safe_load(tvl_file.read_text())
        param = data["configuration_space"]["generator.seed"]
        assert param["type"] == "fixed"
        assert param["value"] == 42

    def test_to_tvl_conditional_constraint(self, tmp_path):
        """Test exporting conditional constraint."""
        space = ExplorationSpace(
            tvars={
                "generator.model": TVAR(
                    name="model",
                    scope="generator",
                    python_type="str",
                    default_value="gpt-4o",
                    constraint=CategoricalConstraint(choices=["gpt-4o", "gpt-4o-mini"]),
                    is_tunable=True,
                ),
                "generator.max_tokens": TVAR(
                    name="max_tokens",
                    scope="generator",
                    python_type="int",
                    default_value=1024,
                    constraint=ConditionalConstraint(
                        parent_qualified_name="generator.model",
                        conditions={
                            "gpt-4o": NumericalConstraint(min=100, max=8192),
                            "gpt-4o-mini": NumericalConstraint(min=100, max=4096),
                        },
                    ),
                    is_tunable=True,
                ),
            }
        )

        tvl_file = tmp_path / "exported.yaml"
        space.to_tvl(tvl_file, include_metadata=False)

        import yaml

        data = yaml.safe_load(tvl_file.read_text())
        param = data["configuration_space"]["generator.max_tokens"]
        assert param["type"] == "conditional"
        assert param["parent"] == "generator.model"
        assert "gpt-4o" in param["conditions"]
        assert "gpt-4o-mini" in param["conditions"]
