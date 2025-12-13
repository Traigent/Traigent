"""Unit tests for traigent.core.samplers.factory.

Tests for SamplerFactory registration and construction helpers.
"""

# Traceability: CONC-Layer-Core CONC-Quality-Usability
# FUNC-OPT-ALGORITHMS REQ-OPT-ALG-004 SYNC-OptimizationFlow

from __future__ import annotations

from collections.abc import Generator

import pytest

from traigent.core.samplers.base import BaseSampler
from traigent.core.samplers.factory import SamplerFactory
from traigent.core.samplers.random_sampler import RandomSampler


class TestSamplerFactoryRegistry:
    """Tests for SamplerFactory registration system."""

    @pytest.fixture(autouse=True)
    def reset_registry(self) -> Generator[None, None, None]:
        """Reset registry state before each test."""
        # Store original registry
        original = SamplerFactory._registry.copy()
        yield
        # Restore original registry
        SamplerFactory._registry = original

    def test_default_registry_contains_random_sampler(self) -> None:
        """Test that factory has random sampler registered by default."""
        assert "random" in SamplerFactory._registry
        assert SamplerFactory._registry["random"] == RandomSampler

    def test_register_new_sampler_with_valid_name(self) -> None:
        """Test registering a new sampler with valid name."""

        class CustomSampler(BaseSampler):
            """Mock custom sampler."""

            def sample(self, **kwargs) -> None:
                """Sample implementation."""
                return None

            def clone(self) -> BaseSampler:
                """Clone implementation."""
                return CustomSampler()

        SamplerFactory.register("custom", CustomSampler)

        assert "custom" in SamplerFactory._registry
        assert SamplerFactory._registry["custom"] == CustomSampler

    def test_register_converts_name_to_lowercase(self) -> None:
        """Test that registration converts sampler name to lowercase."""

        class CustomSampler(BaseSampler):
            """Mock custom sampler."""

            def sample(self, **kwargs) -> None:
                """Sample implementation."""
                return None

            def clone(self) -> BaseSampler:
                """Clone implementation."""
                return CustomSampler()

        SamplerFactory.register("MyCustomSampler", CustomSampler)

        assert "mycustomsampler" in SamplerFactory._registry
        assert SamplerFactory._registry["mycustomsampler"] == CustomSampler

    def test_register_overwrites_existing_sampler(self) -> None:
        """Test that registering same name overwrites previous registration."""

        class FirstSampler(BaseSampler):
            """First mock sampler."""

            def sample(self, **kwargs) -> None:
                """Sample implementation."""
                return None

            def clone(self) -> BaseSampler:
                """Clone implementation."""
                return FirstSampler()

        class SecondSampler(BaseSampler):
            """Second mock sampler."""

            def sample(self, **kwargs) -> None:
                """Sample implementation."""
                return None

            def clone(self) -> BaseSampler:
                """Clone implementation."""
                return SecondSampler()

        SamplerFactory.register("custom", FirstSampler)
        SamplerFactory.register("custom", SecondSampler)

        assert SamplerFactory._registry["custom"] == SecondSampler

    def test_register_raises_on_empty_name(self) -> None:
        """Test that register raises ValueError for empty name."""

        class CustomSampler(BaseSampler):
            """Mock custom sampler."""

            def sample(self, **kwargs) -> None:
                """Sample implementation."""
                return None

            def clone(self) -> BaseSampler:
                """Clone implementation."""
                return CustomSampler()

        with pytest.raises(ValueError, match="Sampler name must be non-empty"):
            SamplerFactory.register("", CustomSampler)

    def test_register_raises_on_non_basesampler_class(self) -> None:
        """Test that register raises TypeError for non-BaseSampler class."""

        class NotASampler:
            """Not a sampler class."""

            pass

        with pytest.raises(TypeError, match="must inherit from BaseSampler"):
            SamplerFactory.register("invalid", NotASampler)  # type: ignore

    def test_register_accepts_basesampler_subclass(self) -> None:
        """Test that register accepts any BaseSampler subclass."""

        class ValidSampler(BaseSampler):
            """Valid sampler subclass."""

            def sample(self, **kwargs) -> None:
                """Sample implementation."""
                return None

            def clone(self) -> BaseSampler:
                """Clone implementation."""
                return ValidSampler()

        # Should not raise
        SamplerFactory.register("valid", ValidSampler)
        assert SamplerFactory._registry["valid"] == ValidSampler


class TestSamplerFactoryCreate:
    """Tests for SamplerFactory.create() method."""

    @pytest.fixture(autouse=True)
    def reset_registry(self) -> Generator[None, None, None]:
        """Reset registry state before each test."""
        # Store original registry
        original = SamplerFactory._registry.copy()
        yield
        # Restore original registry
        SamplerFactory._registry = original

    def test_create_with_none_config_raises_without_params(self) -> None:
        """Test that create with None config raises if sampler needs params."""
        # RandomSampler requires population, so this should raise
        with pytest.raises(TypeError):
            SamplerFactory.create(None)

    def test_create_with_empty_config_raises_without_params(self) -> None:
        """Test create with empty config raises if sampler needs params."""
        # RandomSampler requires population, so this should raise
        with pytest.raises(TypeError):
            SamplerFactory.create({})

    def test_create_with_explicit_random_type(self) -> None:
        """Test create with explicit random type."""
        config = {"type": "random", "params": {"population": [1, 2, 3]}}
        sampler = SamplerFactory.create(config)

        assert isinstance(sampler, RandomSampler)

    def test_create_with_uppercase_type_name(self) -> None:
        """Test that create handles uppercase type names."""
        config = {"type": "RANDOM", "params": {"population": [1, 2, 3]}}
        sampler = SamplerFactory.create(config)

        assert isinstance(sampler, RandomSampler)

    def test_create_with_mixed_case_type_name(self) -> None:
        """Test that create handles mixed case type names."""
        config = {"type": "RaNdOm", "params": {"population": [1, 2, 3]}}
        sampler = SamplerFactory.create(config)

        assert isinstance(sampler, RandomSampler)

    def test_create_passes_params_to_sampler_constructor(self) -> None:
        """Test that create passes params to sampler constructor."""
        config = {
            "type": "random",
            "params": {
                "population": [10, 20, 30],
                "sample_limit": 5,
                "replace": True,
                "seed": 42,
            },
        }
        sampler = SamplerFactory.create(config)

        assert isinstance(sampler, RandomSampler)
        # Verify params were passed (sample should use the population)
        sample = sampler.sample()
        assert sample in [10, 20, 30]

    def test_create_with_empty_params(self) -> None:
        """Test create with empty params dict."""
        config = {"type": "random", "params": {}}

        # Should raise because RandomSampler requires population
        with pytest.raises(TypeError):
            SamplerFactory.create(config)

    def test_create_with_no_params_key(self) -> None:
        """Test create when config has no params key."""
        config = {"type": "random"}

        # Should raise because RandomSampler requires population
        with pytest.raises(TypeError):
            SamplerFactory.create(config)

    def test_create_raises_on_unknown_sampler_type(self) -> None:
        """Test that create raises ValueError for unknown sampler type."""
        config = {"type": "unknown_sampler", "params": {}}

        with pytest.raises(ValueError, match="Unknown sampler 'unknown_sampler'"):
            SamplerFactory.create(config)

    def test_create_error_message_lists_registered_samplers(
        self,
    ) -> None:
        """Test error message includes list of registered samplers."""
        config = {"type": "nonexistent", "params": {}}

        with pytest.raises(ValueError) as exc_info:
            SamplerFactory.create(config)

        error_msg = str(exc_info.value)
        assert "Registered:" in error_msg
        assert "random" in error_msg

    def test_create_with_custom_registered_sampler(self) -> None:
        """Test create with a custom registered sampler."""

        class CustomSampler(BaseSampler):
            """Mock custom sampler."""

            def __init__(self, value: int = 0) -> None:
                """Initialize with custom value."""
                super().__init__()
                self.value = value

            def sample(self, **kwargs) -> int:
                """Return the stored value."""
                return self.value

            def clone(self) -> BaseSampler:
                """Clone implementation."""
                return CustomSampler(self.value)

        SamplerFactory.register("custom", CustomSampler)

        config = {"type": "custom", "params": {"value": 42}}
        sampler = SamplerFactory.create(config)

        assert isinstance(sampler, CustomSampler)
        assert sampler.sample() == 42

    def test_create_with_no_type_defaults_to_random(self) -> None:
        """Test that create defaults to random when type is missing."""
        config = {"params": {"population": [1, 2, 3]}}
        sampler = SamplerFactory.create(config)

        assert isinstance(sampler, RandomSampler)

    def test_create_handles_non_string_type_gracefully(
        self,
    ) -> None:
        """Test that create converts non-string type to string."""
        config = {"type": 123, "params": {}}

        # Should attempt to convert to string
        with pytest.raises(ValueError, match="Unknown sampler '123'"):
            SamplerFactory.create(config)

    def test_create_with_mapping_config(self) -> None:
        """Test create accepts any Mapping type, not just dict."""
        from collections import OrderedDict

        config = OrderedDict([("type", "random"), ("params", {"population": [1, 2]})])
        sampler = SamplerFactory.create(config)

        assert isinstance(sampler, RandomSampler)


class TestSamplerFactoryIntegration:
    """Integration tests for SamplerFactory."""

    @pytest.fixture(autouse=True)
    def reset_registry(self) -> Generator[None, None, None]:
        """Reset registry state before each test."""
        # Store original registry
        original = SamplerFactory._registry.copy()
        yield
        # Restore original registry
        SamplerFactory._registry = original

    def test_register_and_create_workflow(self) -> None:
        """Test complete workflow of registering and creating a sampler."""

        class CountingSampler(BaseSampler):
            """Sampler that counts samples."""

            def __init__(self, max_count: int = 10) -> None:
                """Initialize with max count."""
                super().__init__()
                self.max_count = max_count
                self.count = 0

            def sample(self, **kwargs) -> int | None:
                """Return incrementing count."""
                if self.count >= self.max_count:
                    self._mark_exhausted()
                    return None
                self.count += 1
                return self.count

            def clone(self) -> BaseSampler:
                """Clone implementation."""
                return CountingSampler(self.max_count)

        # Register the sampler
        SamplerFactory.register("counter", CountingSampler)

        # Create instance via factory
        config = {"type": "counter", "params": {"max_count": 3}}
        sampler = SamplerFactory.create(config)

        # Verify it works
        assert sampler.sample() == 1
        assert sampler.sample() == 2
        assert sampler.sample() == 3
        assert sampler.sample() is None
        assert sampler.exhausted

    def test_multiple_samplers_registered(self) -> None:
        """Test that multiple samplers can be registered and created."""

        class SamplerA(BaseSampler):
            """First sampler type."""

            def sample(self, **kwargs) -> str:
                """Return A."""
                return "A"

            def clone(self) -> BaseSampler:
                """Clone implementation."""
                return SamplerA()

        class SamplerB(BaseSampler):
            """Second sampler type."""

            def sample(self, **kwargs) -> str:
                """Return B."""
                return "B"

            def clone(self) -> BaseSampler:
                """Clone implementation."""
                return SamplerB()

        SamplerFactory.register("type_a", SamplerA)
        SamplerFactory.register("type_b", SamplerB)

        sampler_a = SamplerFactory.create({"type": "type_a"})
        sampler_b = SamplerFactory.create({"type": "type_b"})

        assert sampler_a.sample() == "A"
        assert sampler_b.sample() == "B"

    def test_factory_create_with_complex_params(self) -> None:
        """Test factory create with complex nested parameters."""
        config = {
            "type": "random",
            "params": {
                "population": [{"id": 1}, {"id": 2}, {"id": 3}],
                "sample_limit": 2,
                "replace": False,
            },
        }

        sampler = SamplerFactory.create(config)
        assert isinstance(sampler, RandomSampler)

        sample1 = sampler.sample()
        sample2 = sampler.sample()
        sample3 = sampler.sample()

        assert sample1 in [{"id": 1}, {"id": 2}, {"id": 3}]
        assert sample2 in [{"id": 1}, {"id": 2}, {"id": 3}]
        assert sample3 is None  # Exhausted after 2 samples


class TestSamplerFactoryEdgeCases:
    """Edge case tests for SamplerFactory."""

    @pytest.fixture(autouse=True)
    def reset_registry(self) -> Generator[None, None, None]:
        """Reset registry state before each test."""
        # Store original registry
        original = SamplerFactory._registry.copy()
        yield
        # Restore original registry
        SamplerFactory._registry = original

    def test_register_with_whitespace_in_name(self) -> None:
        """Test register with whitespace in name."""

        class CustomSampler(BaseSampler):
            """Mock sampler."""

            def sample(self, **kwargs) -> None:
                """Sample implementation."""
                return None

            def clone(self) -> BaseSampler:
                """Clone implementation."""
                return CustomSampler()

        # Name with spaces should be converted to lowercase
        SamplerFactory.register("  spaced name  ", CustomSampler)

        # Should be stored with spaces intact but lowercased
        assert "  spaced name  " in SamplerFactory._registry

    def test_create_with_none_params_raises_type_error(
        self,
    ) -> None:
        """Test create when params is None raises TypeError."""

        class NoParamSampler(BaseSampler):
            """Sampler that requires no params."""

            def __init__(self) -> None:
                """Initialize with no params."""
                super().__init__()

            def sample(self, **kwargs) -> str:
                """Return fixed value."""
                return "value"

            def clone(self) -> BaseSampler:
                """Clone implementation."""
                return NoParamSampler()

        SamplerFactory.register("noparam", NoParamSampler)

        config = {"type": "noparam", "params": None}  # type: ignore

        # dict(None) raises TypeError
        with pytest.raises(TypeError, match="'NoneType' object is not iterable"):
            SamplerFactory.create(config)

    def test_create_preserves_original_config(self) -> None:
        """Test that create does not modify the original config."""
        original_config = {
            "type": "random",
            "params": {"population": [1, 2, 3]},
        }
        config_copy = original_config.copy()

        SamplerFactory.create(config_copy)

        # Config should be unchanged
        assert config_copy == original_config

    def test_registry_is_class_level_shared(self) -> None:
        """Test that registry is shared across all instances."""

        class CustomSampler(BaseSampler):
            """Mock sampler."""

            def sample(self, **kwargs) -> None:
                """Sample implementation."""
                return None

            def clone(self) -> BaseSampler:
                """Clone implementation."""
                return CustomSampler()

        SamplerFactory.register("shared", CustomSampler)

        # Registry should be accessible via class
        assert "shared" in SamplerFactory._registry

    def test_create_with_invalid_params_propagates_error(
        self,
    ) -> None:
        """Test that invalid params cause appropriate error from sampler."""
        config = {
            "type": "random",
            "params": {
                "population": [1, 2, 3],
                "sample_limit": -5,  # Invalid
            },
        }

        with pytest.raises(ValueError, match="sample_limit must be a positive integer"):
            SamplerFactory.create(config)

    def test_register_same_class_with_different_names(
        self,
    ) -> None:
        """Test registering same sampler class with multiple names."""

        class SharedSampler(BaseSampler):
            """Sampler registered under multiple names."""

            def sample(self, **kwargs) -> str:
                """Return value."""
                return "shared"

            def clone(self) -> BaseSampler:
                """Clone implementation."""
                return SharedSampler()

        SamplerFactory.register("alias1", SharedSampler)
        SamplerFactory.register("alias2", SharedSampler)

        sampler1 = SamplerFactory.create({"type": "alias1"})
        sampler2 = SamplerFactory.create({"type": "alias2"})

        assert isinstance(sampler1, type(sampler2))
        assert sampler1.sample() == sampler2.sample()
