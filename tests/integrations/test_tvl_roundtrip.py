"""Tests for TVL round-trip (Story 3.6).

This module tests that ExplorationSpace.to_tvl() and from_tvl_spec()
correctly preserve all parameter attributes including:
- log_scale
- step
- conditional parameters
- fixed parameters
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import yaml

from traigent.integrations.haystack.configspace import (
    TVAR,
    CategoricalConstraint,
    ConditionalConstraint,
    ExplorationSpace,
    NumericalConstraint,
)
from traigent.tvl.spec_loader import (
    ConditionalDomain,
    NumericalDomain,
    load_tvl_spec,
)


class TestNumericalDomainParsing:
    """Tests for parsing numerical domains with log_scale and step."""

    def test_continuous_with_log_scale(self):
        """Test parsing continuous parameter with log_scale."""
        tvl_content = """
configuration_space:
  learning_rate:
    type: continuous
    range: [0.0001, 0.1]
    log_scale: true
    default: 0.01
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(tvl_content)
            f.flush()

            spec = load_tvl_spec(spec_path=f.name)

            domain = spec.configuration_space["learning_rate"]
            assert isinstance(domain, NumericalDomain)
            assert domain.min == 0.0001
            assert domain.max == 0.1
            assert domain.log_scale is True
            assert domain.step is None

    def test_continuous_with_step(self):
        """Test parsing continuous parameter with step."""
        tvl_content = """
configuration_space:
  dropout:
    type: continuous
    range: [0.0, 1.0]
    step: 0.1
    default: 0.5
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(tvl_content)
            f.flush()

            spec = load_tvl_spec(spec_path=f.name)

            domain = spec.configuration_space["dropout"]
            assert isinstance(domain, NumericalDomain)
            assert domain.min == 0.0
            assert domain.max == 1.0
            assert domain.step == 0.1
            assert domain.log_scale is False

    def test_integer_with_step(self):
        """Test parsing integer parameter with step."""
        tvl_content = """
configuration_space:
  batch_size:
    type: integer
    range: [16, 128]
    step: 16
    default: 32
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(tvl_content)
            f.flush()

            spec = load_tvl_spec(spec_path=f.name)

            domain = spec.configuration_space["batch_size"]
            assert isinstance(domain, NumericalDomain)
            assert domain.min == 16
            assert domain.max == 128
            assert domain.step == 16
            assert domain.is_integer is True

    def test_simple_continuous_returns_tuple(self):
        """Test that simple continuous without log_scale/step returns tuple."""
        tvl_content = """
configuration_space:
  temperature:
    type: continuous
    range: [0.0, 2.0]
    default: 0.7
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(tvl_content)
            f.flush()

            spec = load_tvl_spec(spec_path=f.name)

            domain = spec.configuration_space["temperature"]
            assert isinstance(domain, tuple)
            assert domain == (0.0, 2.0)


class TestConditionalDomainParsing:
    """Tests for parsing conditional parameters."""

    def test_conditional_with_numerical_ranges(self):
        """Test parsing conditional with numerical child ranges."""
        tvl_content = """
configuration_space:
  generator.model:
    type: categorical
    values: ["gpt-4o", "gpt-4o-mini"]
    default: "gpt-4o"
  generator.max_tokens:
    type: conditional
    parent: "generator.model"
    conditions:
      gpt-4o:
        range: [100, 8192]
      gpt-4o-mini:
        range: [100, 4096]
    default:
      range: [100, 2048]
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(tvl_content)
            f.flush()

            spec = load_tvl_spec(spec_path=f.name)

            domain = spec.configuration_space["generator.max_tokens"]
            assert isinstance(domain, ConditionalDomain)
            assert domain.parent == "generator.model"
            assert "gpt-4o" in domain.conditions
            assert "gpt-4o-mini" in domain.conditions
            assert domain.default is not None

            # Check condition values
            gpt4o_range = domain.conditions["gpt-4o"]
            assert gpt4o_range == (100.0, 8192.0)

    def test_conditional_with_categorical_values(self):
        """Test parsing conditional with categorical child values."""
        tvl_content = """
configuration_space:
  framework:
    type: categorical
    values: ["pytorch", "tensorflow"]
    default: "pytorch"
  optimizer:
    type: conditional
    parent: "framework"
    conditions:
      pytorch:
        values: ["adam", "sgd", "adamw"]
      tensorflow:
        values: ["adam", "sgd", "rmsprop"]
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(tvl_content)
            f.flush()

            spec = load_tvl_spec(spec_path=f.name)

            domain = spec.configuration_space["optimizer"]
            assert isinstance(domain, ConditionalDomain)
            assert domain.parent == "framework"
            assert domain.conditions["pytorch"] == ["adam", "sgd", "adamw"]
            assert domain.conditions["tensorflow"] == ["adam", "sgd", "rmsprop"]


class TestFixedParameterParsing:
    """Tests for parsing fixed parameters."""

    def test_fixed_parameter(self):
        """Test parsing fixed parameter type."""
        tvl_content = """
configuration_space:
  model:
    type: fixed
    value: "gpt-4o"
  temperature:
    type: continuous
    range: [0.0, 2.0]
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(tvl_content)
            f.flush()

            spec = load_tvl_spec(spec_path=f.name)

            from traigent.tvl.spec_loader import FixedDomain

            domain = spec.configuration_space["model"]
            # Fixed parameters now return FixedDomain for proper TVL round-trip
            assert isinstance(domain, FixedDomain)
            assert domain.value == "gpt-4o"


class TestExplorationSpaceFromTVL:
    """Tests for ExplorationSpace.from_tvl_spec() with new domain types."""

    def test_from_tvl_with_log_scale(self):
        """Test ExplorationSpace handles NumericalDomain with log_scale."""
        tvl_content = """
configuration_space:
  learning_rate:
    type: continuous
    range: [0.0001, 0.1]
    log_scale: true
    default: 0.01
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(tvl_content)
            f.flush()

            space = ExplorationSpace.from_tvl_spec(f.name)

            tvar = space.get_tvar_by_qualified_name("learning_rate")
            assert tvar is not None
            assert isinstance(tvar.constraint, NumericalConstraint)
            assert tvar.constraint.log_scale is True
            assert tvar.constraint.min == 0.0001
            assert tvar.constraint.max == 0.1

    def test_from_tvl_with_step(self):
        """Test ExplorationSpace handles NumericalDomain with step."""
        tvl_content = """
configuration_space:
  batch_size:
    type: integer
    range: [16, 128]
    step: 16
    default: 32
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(tvl_content)
            f.flush()

            space = ExplorationSpace.from_tvl_spec(f.name)

            tvar = space.get_tvar_by_qualified_name("batch_size")
            assert tvar is not None
            assert isinstance(tvar.constraint, NumericalConstraint)
            assert tvar.constraint.step == 16
            assert tvar.python_type == "int"

    def test_from_tvl_with_conditional(self):
        """Test ExplorationSpace handles ConditionalDomain."""
        tvl_content = """
configuration_space:
  generator.model:
    type: categorical
    values: ["gpt-4o", "gpt-4o-mini"]
    default: "gpt-4o"
  generator.max_tokens:
    type: conditional
    parent: "generator.model"
    conditions:
      gpt-4o:
        range: [100, 8192]
      gpt-4o-mini:
        range: [100, 4096]
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(tvl_content)
            f.flush()

            space = ExplorationSpace.from_tvl_spec(f.name)

            tvar = space.get_tvar_by_qualified_name("generator.max_tokens")
            assert tvar is not None
            assert isinstance(tvar.constraint, ConditionalConstraint)
            assert tvar.constraint.parent_qualified_name == "generator.model"

            # Check conditions
            gpt4o_constraint = tvar.constraint.get_constraint_for("gpt-4o")
            assert isinstance(gpt4o_constraint, NumericalConstraint)
            assert gpt4o_constraint.max == 8192


class TestTVLRoundTrip:
    """Tests for full round-trip: ExplorationSpace -> TVL -> ExplorationSpace."""

    def test_roundtrip_simple_parameters(self):
        """Test round-trip with simple categorical and numerical parameters."""
        # Create original space
        original = ExplorationSpace(
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

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            # Export
            original.to_tvl(f.name)

            # Import
            reloaded = ExplorationSpace.from_tvl_spec(f.name)

            # Verify
            assert len(reloaded.tvars) == 2

            model_tvar = reloaded.get_tvar_by_qualified_name("generator.model")
            assert model_tvar is not None
            assert isinstance(model_tvar.constraint, CategoricalConstraint)
            assert set(model_tvar.constraint.choices) == {"gpt-4o", "gpt-4o-mini"}

            temp_tvar = reloaded.get_tvar_by_qualified_name("generator.temperature")
            assert temp_tvar is not None
            assert isinstance(temp_tvar.constraint, NumericalConstraint)
            assert temp_tvar.constraint.min == 0.0
            assert temp_tvar.constraint.max == 2.0

    def test_roundtrip_with_log_scale(self):
        """Test round-trip preserves log_scale."""
        original = ExplorationSpace(
            tvars={
                "learning_rate": TVAR(
                    name="learning_rate",
                    scope="default",
                    python_type="float",
                    default_value=0.01,
                    constraint=NumericalConstraint(min=0.0001, max=0.1, log_scale=True),
                ),
            }
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            original.to_tvl(f.name)
            reloaded = ExplorationSpace.from_tvl_spec(f.name)

            tvar = reloaded.get_tvar_by_qualified_name("learning_rate")
            assert tvar is not None
            assert isinstance(tvar.constraint, NumericalConstraint)
            assert tvar.constraint.log_scale is True

    def test_roundtrip_with_step(self):
        """Test round-trip preserves step."""
        original = ExplorationSpace(
            tvars={
                "batch_size": TVAR(
                    name="batch_size",
                    scope="default",
                    python_type="int",
                    default_value=32,
                    constraint=NumericalConstraint(min=16, max=128, step=16),
                ),
            }
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            original.to_tvl(f.name)
            reloaded = ExplorationSpace.from_tvl_spec(f.name)

            tvar = reloaded.get_tvar_by_qualified_name("batch_size")
            assert tvar is not None
            assert isinstance(tvar.constraint, NumericalConstraint)
            assert tvar.constraint.step == 16

    def test_roundtrip_with_conditional(self):
        """Test round-trip preserves conditional constraints."""
        original = ExplorationSpace(
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
                    constraint=ConditionalConstraint(
                        parent_qualified_name="generator.model",
                        conditions={
                            "gpt-4o": NumericalConstraint(min=100, max=8192),
                            "gpt-4o-mini": NumericalConstraint(min=100, max=4096),
                        },
                    ),
                ),
            }
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            original.to_tvl(f.name)
            reloaded = ExplorationSpace.from_tvl_spec(f.name)

            tvar = reloaded.get_tvar_by_qualified_name("generator.max_tokens")
            assert tvar is not None
            assert isinstance(tvar.constraint, ConditionalConstraint)
            assert tvar.constraint.parent_qualified_name == "generator.model"

            gpt4o = tvar.constraint.get_constraint_for("gpt-4o")
            assert isinstance(gpt4o, NumericalConstraint)
            assert gpt4o.max == 8192


class TestTVLExportFormat:
    """Tests for the exported TVL format."""

    def test_export_includes_metadata(self):
        """Test that export includes metadata section."""
        space = ExplorationSpace(
            tvars={
                "temperature": TVAR(
                    name="temperature",
                    scope="default",
                    python_type="float",
                    default_value=0.7,
                    constraint=NumericalConstraint(min=0.0, max=2.0),
                ),
            }
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            space.to_tvl(f.name, description="Test export")

            with open(f.name) as fp:
                data = yaml.safe_load(fp)

            assert "metadata" in data
            assert data["metadata"]["description"] == "Test export"
            assert data["metadata"]["num_tvars"] == 1
            assert data["metadata"]["num_tunable"] == 1

    def test_export_conditional_format(self):
        """Test that conditional parameters export correctly."""
        space = ExplorationSpace(
            tvars={
                "model": TVAR(
                    name="model",
                    scope="default",
                    python_type="str",
                    default_value="gpt-4o",
                    constraint=CategoricalConstraint(choices=["gpt-4o", "gpt-4o-mini"]),
                ),
                "max_tokens": TVAR(
                    name="max_tokens",
                    scope="default",
                    python_type="int",
                    default_value=1024,
                    constraint=ConditionalConstraint(
                        parent_qualified_name="model",
                        conditions={
                            "gpt-4o": NumericalConstraint(min=100, max=8192),
                        },
                        default_constraint=NumericalConstraint(min=100, max=2048),
                    ),
                ),
            }
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            space.to_tvl(f.name, include_metadata=False)

            with open(f.name) as fp:
                data = yaml.safe_load(fp)

            max_tokens_spec = data["configuration_space"]["max_tokens"]
            assert max_tokens_spec["type"] == "conditional"
            assert max_tokens_spec["parent"] == "model"
            assert "gpt-4o" in max_tokens_spec["conditions"]
            assert "default" in max_tokens_spec
