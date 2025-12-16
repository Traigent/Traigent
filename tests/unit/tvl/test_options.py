"""Unit tests for TVL options module."""

from pathlib import Path

import pytest

from traigent.tvl.options import TVLOptions


class TestTVLOptions:
    """Tests for TVLOptions class."""

    def test_basic_creation(self) -> None:
        """TVLOptions can be created with minimal args."""
        opts = TVLOptions(spec_path="test.tvl")
        assert opts.spec_path == "test.tvl"
        assert opts.environment is None
        assert opts.validate_constraints is True
        assert opts.registry_resolver is None

    def test_all_options(self) -> None:
        """TVLOptions with all options specified."""
        opts = TVLOptions(
            spec_path="specs/my_spec.tvl",
            environment="production",
            validate_constraints=False,
            apply_evaluation_set=False,
            apply_configuration_space=False,
            apply_objectives=False,
            apply_constraints=False,
            apply_budget=False,
        )

        assert opts.spec_path == "specs/my_spec.tvl"
        assert opts.environment == "production"
        assert opts.validate_constraints is False
        assert opts.apply_evaluation_set is False
        assert opts.apply_configuration_space is False
        assert opts.apply_objectives is False
        assert opts.apply_constraints is False
        assert opts.apply_budget is False

    def test_path_coercion_from_string(self) -> None:
        """String path is preserved as string."""
        opts = TVLOptions(spec_path="path/to/spec.tvl")
        assert opts.spec_path == "path/to/spec.tvl"
        assert isinstance(opts.spec_path, str)

    def test_path_coercion_from_pathlib(self) -> None:
        """Pathlib Path requires string conversion before passing to TVLOptions."""
        # Pydantic requires string input; Path objects must be converted
        opts = TVLOptions(spec_path=str(Path("path/to/spec.tvl")))
        # The validator preserves string format
        assert isinstance(opts.spec_path, str)
        assert "spec.tvl" in opts.spec_path

    def test_extra_fields_forbidden(self) -> None:
        """Extra fields raise ValidationError."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            TVLOptions(spec_path="test.tvl", unknown_field="value")  # type: ignore


class TestTVLOptionsMergedWith:
    """Tests for TVLOptions.merged_with method."""

    def test_merged_with_new_environment(self) -> None:
        """merged_with creates a copy with new environment."""
        original = TVLOptions(spec_path="test.tvl", environment="dev")
        merged = original.merged_with(environment="prod")

        # Original unchanged
        assert original.environment == "dev"
        # New instance has updated environment
        assert merged.environment == "prod"
        # Other fields preserved
        assert merged.spec_path == "test.tvl"

    def test_merged_with_same_environment(self) -> None:
        """merged_with returns self when environment unchanged."""
        original = TVLOptions(spec_path="test.tvl", environment="dev")
        merged = original.merged_with(environment="dev")

        assert merged is original

    def test_merged_with_none_environment(self) -> None:
        """merged_with with None environment returns self."""
        original = TVLOptions(spec_path="test.tvl", environment="dev")
        merged = original.merged_with(environment=None)

        assert merged is original

    def test_merged_with_preserves_all_fields(self) -> None:
        """merged_with preserves all other fields."""
        original = TVLOptions(
            spec_path="test.tvl",
            environment="dev",
            validate_constraints=False,
            apply_evaluation_set=False,
            apply_configuration_space=True,
            apply_objectives=True,
            apply_constraints=False,
            apply_budget=False,
        )
        merged = original.merged_with(environment="prod")

        assert merged.spec_path == original.spec_path
        assert merged.validate_constraints == original.validate_constraints
        assert merged.apply_evaluation_set == original.apply_evaluation_set
        assert merged.apply_configuration_space == original.apply_configuration_space
        assert merged.apply_objectives == original.apply_objectives
        assert merged.apply_constraints == original.apply_constraints
        assert merged.apply_budget == original.apply_budget


class TestTVLOptionsToKwargs:
    """Tests for TVLOptions.to_kwargs method."""

    def test_to_kwargs_basic(self) -> None:
        """to_kwargs returns loader kwargs."""
        opts = TVLOptions(spec_path="test.tvl")
        kwargs = opts.to_kwargs()

        assert kwargs["spec_path"] == "test.tvl"
        assert kwargs["environment"] is None
        assert kwargs["validate_constraints"] is True
        assert kwargs["registry_resolver"] is None

    def test_to_kwargs_with_environment(self) -> None:
        """to_kwargs includes environment when set."""
        opts = TVLOptions(spec_path="test.tvl", environment="production")
        kwargs = opts.to_kwargs()

        assert kwargs["environment"] == "production"

    def test_to_kwargs_with_validate_constraints_false(self) -> None:
        """to_kwargs includes validate_constraints setting."""
        opts = TVLOptions(spec_path="test.tvl", validate_constraints=False)
        kwargs = opts.to_kwargs()

        assert kwargs["validate_constraints"] is False

    def test_to_kwargs_does_not_include_apply_flags(self) -> None:
        """to_kwargs only includes loader kwargs, not apply flags."""
        opts = TVLOptions(
            spec_path="test.tvl",
            apply_evaluation_set=False,
            apply_configuration_space=False,
        )
        kwargs = opts.to_kwargs()

        # Only loader-relevant kwargs
        assert set(kwargs.keys()) == {
            "spec_path",
            "environment",
            "validate_constraints",
            "registry_resolver",
        }
