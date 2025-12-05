"""Comprehensive tests for traigent.core.metric_registry module.

Tests cover MetricSpec dataclass and MetricRegistry for metric aggregation.
"""

from __future__ import annotations

import pytest

from traigent.core.metric_registry import (
    MetricRegistry,
    MetricSpec,
)


class TestMetricSpec:
    """Test MetricSpec dataclass."""

    def test_basic_creation(self):
        """Test basic MetricSpec creation."""
        spec = MetricSpec(name="accuracy", aggregator="mean")

        assert spec.name == "accuracy"
        assert spec.aggregator == "mean"
        assert spec.mandatory is False
        assert spec.description is None

    def test_with_mandatory_flag(self):
        """Test MetricSpec with mandatory flag."""
        spec = MetricSpec(
            name="total_cost",
            aggregator="sum",
            mandatory=True,
        )

        assert spec.mandatory is True

    def test_with_description(self):
        """Test MetricSpec with description."""
        spec = MetricSpec(
            name="latency",
            aggregator="mean",
            description="Average response time in seconds",
        )

        assert spec.description == "Average response time in seconds"

    def test_all_aggregator_types(self):
        """Test all valid aggregator types."""
        spec_mean = MetricSpec(name="metric1", aggregator="mean")
        spec_sum = MetricSpec(name="metric2", aggregator="sum")
        spec_last = MetricSpec(name="metric3", aggregator="last")

        assert spec_mean.aggregator == "mean"
        assert spec_sum.aggregator == "sum"
        assert spec_last.aggregator == "last"

    def test_dataclass_equality(self):
        """Test MetricSpec equality."""
        spec1 = MetricSpec(name="accuracy", aggregator="mean", mandatory=True)
        spec2 = MetricSpec(name="accuracy", aggregator="mean", mandatory=True)

        assert spec1 == spec2

    def test_dataclass_inequality(self):
        """Test MetricSpec inequality."""
        spec1 = MetricSpec(name="accuracy", aggregator="mean")
        spec2 = MetricSpec(name="accuracy", aggregator="sum")

        assert spec1 != spec2

    def test_slots_optimization(self):
        """Test that MetricSpec uses slots for memory optimization."""
        spec = MetricSpec(name="test", aggregator="mean")

        # Slots prevent dynamic attribute addition
        with pytest.raises(AttributeError):
            spec.new_attribute = "value"  # type: ignore


class TestMetricRegistryBasics:
    """Test basic MetricRegistry functionality."""

    def test_basic_creation(self):
        """Test basic registry creation."""
        registry = MetricRegistry()

        assert registry.default_aggregator == "mean"
        assert len(list(registry.specs())) == 0

    def test_custom_default_aggregator(self):
        """Test registry with custom default aggregator."""
        registry = MetricRegistry(default_aggregator="sum")

        assert registry.default_aggregator == "sum"

    def test_register_single_spec(self):
        """Test registering a single metric spec."""
        registry = MetricRegistry()
        spec = MetricSpec(name="accuracy", aggregator="mean")

        registry.register(spec)

        assert registry.get("accuracy") == spec

    def test_register_many_specs(self):
        """Test registering multiple metric specs."""
        registry = MetricRegistry()
        specs = [
            MetricSpec(name="accuracy", aggregator="mean"),
            MetricSpec(name="cost", aggregator="sum"),
            MetricSpec(name="latency", aggregator="mean"),
        ]

        registry.register_many(*specs)

        assert len(list(registry.specs())) == 3

    def test_get_existing_spec(self):
        """Test getting an existing metric spec."""
        registry = MetricRegistry()
        spec = MetricSpec(name="accuracy", aggregator="mean")
        registry.register(spec)

        retrieved = registry.get("accuracy")

        assert retrieved is not None
        assert retrieved.name == "accuracy"

    def test_get_nonexistent_spec(self):
        """Test getting a non-existent metric spec."""
        registry = MetricRegistry()

        retrieved = registry.get("nonexistent")

        assert retrieved is None


class TestMetricRegistryDefault:
    """Test default MetricRegistry factory method."""

    def test_default_factory(self):
        """Test default() factory creates pre-populated registry."""
        registry = MetricRegistry.default()

        assert len(list(registry.specs())) >= 4

    def test_default_has_cost_metric(self):
        """Test default registry has total_cost metric."""
        registry = MetricRegistry.default()

        cost_spec = registry.get("total_cost")

        assert cost_spec is not None
        assert cost_spec.aggregator == "sum"
        assert cost_spec.mandatory is True

    def test_default_has_tokens_metric(self):
        """Test default registry has total_tokens metric."""
        registry = MetricRegistry.default()

        tokens_spec = registry.get("total_tokens")

        assert tokens_spec is not None
        assert tokens_spec.aggregator == "sum"
        assert tokens_spec.mandatory is True

    def test_default_has_duration_metric(self):
        """Test default registry has total_duration metric."""
        registry = MetricRegistry.default()

        duration_spec = registry.get("total_duration")

        assert duration_spec is not None
        assert duration_spec.aggregator == "sum"

    def test_default_has_examples_metric(self):
        """Test default registry has examples_attempted_total metric."""
        registry = MetricRegistry.default()

        examples_spec = registry.get("examples_attempted_total")

        assert examples_spec is not None
        assert examples_spec.aggregator == "sum"


class TestMetricRegistryAggregation:
    """Test aggregation logic in MetricRegistry."""

    def test_aggregator_for_registered_metric(self):
        """Test getting aggregator for registered metric."""
        registry = MetricRegistry()
        registry.register(MetricSpec(name="accuracy", aggregator="mean"))

        aggregator = registry.aggregator_for("accuracy")

        assert aggregator == "mean"

    def test_aggregator_for_unregistered_metric(self):
        """Test getting aggregator for unregistered metric uses default."""
        registry = MetricRegistry(default_aggregator="sum")

        aggregator = registry.aggregator_for("unknown_metric")

        assert aggregator == "sum"

    def test_different_aggregators(self):
        """Test different aggregator types."""
        registry = MetricRegistry()
        registry.register_many(
            MetricSpec(name="metric1", aggregator="mean"),
            MetricSpec(name="metric2", aggregator="sum"),
            MetricSpec(name="metric3", aggregator="last"),
        )

        assert registry.aggregator_for("metric1") == "mean"
        assert registry.aggregator_for("metric2") == "sum"
        assert registry.aggregator_for("metric3") == "last"


class TestMetricRegistryMandatory:
    """Test mandatory metric functionality."""

    def test_is_mandatory_for_mandatory_metric(self):
        """Test is_mandatory returns True for mandatory metrics."""
        registry = MetricRegistry()
        registry.register(MetricSpec(name="cost", aggregator="sum", mandatory=True))

        assert registry.is_mandatory("cost") is True

    def test_is_mandatory_for_optional_metric(self):
        """Test is_mandatory returns False for optional metrics."""
        registry = MetricRegistry()
        registry.register(
            MetricSpec(name="accuracy", aggregator="mean", mandatory=False)
        )

        assert registry.is_mandatory("accuracy") is False

    def test_is_mandatory_for_unregistered_metric(self):
        """Test is_mandatory returns False for unregistered metrics."""
        registry = MetricRegistry()

        assert registry.is_mandatory("unknown") is False

    def test_mandatory_metric_names_property(self):
        """Test mandatory_metric_names property."""
        registry = MetricRegistry()
        registry.register_many(
            MetricSpec(name="cost", aggregator="sum", mandatory=True),
            MetricSpec(name="accuracy", aggregator="mean", mandatory=False),
            MetricSpec(name="tokens", aggregator="sum", mandatory=True),
        )

        mandatory_names = list(registry.mandatory_metric_names)

        assert len(mandatory_names) == 2
        assert "cost" in mandatory_names
        assert "tokens" in mandatory_names
        assert "accuracy" not in mandatory_names

    def test_no_mandatory_metrics(self):
        """Test registry with no mandatory metrics."""
        registry = MetricRegistry()
        registry.register_many(
            MetricSpec(name="metric1", aggregator="mean"),
            MetricSpec(name="metric2", aggregator="sum"),
        )

        mandatory_names = list(registry.mandatory_metric_names)

        assert len(mandatory_names) == 0


class TestMetricRegistryClone:
    """Test registry cloning functionality."""

    def test_clone_basic(self):
        """Test basic clone operation."""
        registry = MetricRegistry()
        registry.register(MetricSpec(name="accuracy", aggregator="mean"))

        cloned = registry.clone()

        assert cloned is not registry
        assert cloned.get("accuracy") is not None

    def test_clone_preserves_specs(self):
        """Test clone preserves all specs."""
        registry = MetricRegistry()
        registry.register_many(
            MetricSpec(name="metric1", aggregator="mean"),
            MetricSpec(name="metric2", aggregator="sum"),
            MetricSpec(name="metric3", aggregator="last"),
        )

        cloned = registry.clone()

        assert len(list(cloned.specs())) == 3
        assert cloned.get("metric1") is not None
        assert cloned.get("metric2") is not None
        assert cloned.get("metric3") is not None

    def test_clone_preserves_default_aggregator(self):
        """Test clone preserves default aggregator."""
        registry = MetricRegistry(default_aggregator="sum")

        cloned = registry.clone()

        assert cloned.default_aggregator == "sum"

    def test_clone_is_independent(self):
        """Test cloned registry is independent of original."""
        registry = MetricRegistry()
        registry.register(MetricSpec(name="metric1", aggregator="mean"))

        cloned = registry.clone()
        cloned.register(MetricSpec(name="metric2", aggregator="sum"))

        # Original should not have metric2
        assert registry.get("metric2") is None
        # Clone should have both
        assert cloned.get("metric1") is not None
        assert cloned.get("metric2") is not None

    def test_clone_empty_registry(self):
        """Test cloning an empty registry."""
        registry = MetricRegistry()

        cloned = registry.clone()

        assert len(list(cloned.specs())) == 0

    def test_clone_with_descriptions(self):
        """Test clone preserves descriptions."""
        registry = MetricRegistry()
        registry.register(
            MetricSpec(
                name="metric1",
                aggregator="mean",
                description="Test description",
            )
        )

        cloned = registry.clone()
        cloned_spec = cloned.get("metric1")

        assert cloned_spec is not None
        assert cloned_spec.description == "Test description"


class TestMetricRegistrySpecs:
    """Test specs() method."""

    def test_specs_returns_tuple(self):
        """Test specs() returns tuple."""
        registry = MetricRegistry()
        registry.register(MetricSpec(name="metric1", aggregator="mean"))

        specs = registry.specs()

        assert isinstance(specs, tuple)

    def test_specs_returns_all_registered(self):
        """Test specs() returns all registered specs."""
        registry = MetricRegistry()
        registry.register_many(
            MetricSpec(name="metric1", aggregator="mean"),
            MetricSpec(name="metric2", aggregator="sum"),
        )

        specs = registry.specs()

        assert len(specs) == 2

    def test_specs_empty_registry(self):
        """Test specs() on empty registry."""
        registry = MetricRegistry()

        specs = registry.specs()

        assert len(specs) == 0


class TestMetricRegistryOverwrite:
    """Test overwriting existing metrics."""

    def test_register_overwrites_existing(self):
        """Test registering same metric name overwrites."""
        registry = MetricRegistry()
        registry.register(MetricSpec(name="metric1", aggregator="mean"))
        registry.register(MetricSpec(name="metric1", aggregator="sum"))

        spec = registry.get("metric1")

        assert spec is not None
        assert spec.aggregator == "sum"

    def test_last_registration_wins(self):
        """Test last registration wins when registering many."""
        registry = MetricRegistry()
        registry.register_many(
            MetricSpec(name="metric1", aggregator="mean", mandatory=False),
            MetricSpec(name="metric1", aggregator="sum", mandatory=True),
        )

        spec = registry.get("metric1")

        assert spec is not None
        assert spec.aggregator == "sum"
        assert spec.mandatory is True


class TestMetricRegistryIntegration:
    """Test integration scenarios."""

    def test_complete_workflow(self):
        """Test complete workflow: create, register, query, clone."""
        # Create and populate
        registry = MetricRegistry(default_aggregator="mean")
        registry.register_many(
            MetricSpec(name="cost", aggregator="sum", mandatory=True),
            MetricSpec(name="accuracy", aggregator="mean"),
            MetricSpec(name="latency", aggregator="mean"),
        )

        # Query
        assert registry.aggregator_for("cost") == "sum"
        assert registry.is_mandatory("cost") is True
        assert registry.aggregator_for("unknown") == "mean"

        # Clone and modify
        cloned = registry.clone()
        cloned.register(MetricSpec(name="new_metric", aggregator="last"))

        assert registry.get("new_metric") is None
        assert cloned.get("new_metric") is not None

    def test_default_registry_workflow(self):
        """Test workflow with default registry."""
        registry = MetricRegistry.default()

        # Should have mandatory metrics
        mandatory = list(registry.mandatory_metric_names)
        assert len(mandatory) >= 4

        # Can add custom metrics
        registry.register(MetricSpec(name="custom", aggregator="mean", mandatory=False))

        assert registry.get("custom") is not None
        assert not registry.is_mandatory("custom")
