"""Unit tests for the safety constraint system.

Tests cover:
- SafetyThreshold creation and validation
- SafetyMetric and its .above()/.below()/.between() factories
- SafetyConstraint evaluation against metrics
- CompoundSafetyConstraint (AND/OR combinations)
- SafetyValidator statistical validation with Clopper-Pearson
- RAGAS metric presets (when available)
- Non-RAGAS metric factories
"""

from __future__ import annotations

from typing import Any

import pytest

from traigent.api.safety import (
    CompoundSafetyConstraint,
    MetricKeyMetric,
    SafetyConstraint,
    SafetyThreshold,
    SafetyValidationResult,
    SafetyValidator,
    bias_score,
    custom_safety,
    get_available_safety_presets,
    hallucination_rate,
    safety_score,
    toxicity_score,
)


class TestSafetyThreshold:
    """Tests for SafetyThreshold dataclass."""

    def test_threshold_creation(self) -> None:
        """Test basic threshold creation."""
        threshold = SafetyThreshold(
            metric_name="faithfulness",
            operator=">=",
            value=0.9,
        )
        assert threshold.metric_name == "faithfulness"
        assert threshold.operator == ">="
        assert threshold.value == 0.9
        assert threshold.confidence is None
        assert threshold.min_samples == 30

    def test_threshold_with_confidence(self) -> None:
        """Test threshold with statistical confidence."""
        threshold = SafetyThreshold(
            metric_name="accuracy",
            operator=">=",
            value=0.85,
            confidence=0.95,
            min_samples=50,
        )
        assert threshold.confidence == 0.95
        assert threshold.min_samples == 50

    def test_threshold_invalid_confidence(self) -> None:
        """Test that invalid confidence raises error."""
        with pytest.raises(ValueError, match="confidence must be in"):
            SafetyThreshold(
                metric_name="test",
                operator=">=",
                value=0.9,
                confidence=1.5,  # Invalid: > 1
            )

        with pytest.raises(ValueError, match="confidence must be in"):
            SafetyThreshold(
                metric_name="test",
                operator=">=",
                value=0.9,
                confidence=0.0,  # Invalid: must be > 0
            )

    def test_threshold_invalid_operator(self) -> None:
        """Test that invalid operator raises error."""
        with pytest.raises(ValueError, match="Unknown operator"):
            SafetyThreshold(
                metric_name="test",
                operator="~=",  # Invalid
                value=0.9,
            )

    def test_threshold_invalid_min_samples(self) -> None:
        """Test that invalid min_samples raises error."""
        with pytest.raises(ValueError, match="min_samples must be >= 1"):
            SafetyThreshold(
                metric_name="test",
                operator=">=",
                value=0.9,
                min_samples=0,
            )

    def test_threshold_immutability(self) -> None:
        """Test that threshold is frozen/immutable."""
        threshold = SafetyThreshold(
            metric_name="test",
            operator=">=",
            value=0.9,
        )
        with pytest.raises(AttributeError):
            threshold.value = 0.8  # type: ignore[misc]


class TestMetricKeyMetric:
    """Tests for MetricKeyMetric (reads from metrics dict)."""

    def test_metric_key_above(self) -> None:
        """Test .above() constraint creation."""
        metric = MetricKeyMetric(
            name="accuracy",
            metric_key="accuracy",
            description="Accuracy score",
        )
        constraint = metric.above(0.9)

        assert isinstance(constraint, SafetyConstraint)
        assert constraint.threshold.operator == ">="
        assert constraint.threshold.value == 0.9
        assert constraint.threshold.metric_name == "accuracy"

    def test_metric_key_below(self) -> None:
        """Test .below() constraint creation."""
        metric = MetricKeyMetric(
            name="error_rate",
            metric_key="error_rate",
            description="Error rate",
        )
        constraint = metric.below(0.1)

        assert isinstance(constraint, SafetyConstraint)
        assert constraint.threshold.operator == "<="
        assert constraint.threshold.value == 0.1

    def test_metric_key_between(self) -> None:
        """Test .between() constraint creation."""
        metric = MetricKeyMetric(
            name="latency",
            metric_key="latency",
            description="Response latency",
        )
        constraint = metric.between(10, 100)

        assert isinstance(constraint, CompoundSafetyConstraint)
        assert constraint.combinator == "and"

    def test_metric_key_with_confidence(self) -> None:
        """Test constraint with statistical confidence."""
        metric = MetricKeyMetric(
            name="accuracy",
            metric_key="accuracy",
            description="Accuracy",
        )
        constraint = metric.above(0.9, confidence=0.95)

        assert constraint.threshold.confidence == 0.95


class TestSafetyConstraint:
    """Tests for SafetyConstraint evaluation."""

    def test_constraint_evaluation_pass(self) -> None:
        """Test constraint that passes."""
        metric = MetricKeyMetric(
            name="accuracy",
            metric_key="accuracy",
            description="Accuracy metric",
        )
        constraint = metric.above(0.9)

        config: dict[str, Any] = {"model": "gpt-4"}
        metrics = {"accuracy": 0.95}

        result = constraint(config, metrics)
        assert result is True

    def test_constraint_evaluation_fail(self) -> None:
        """Test constraint that fails."""
        metric = MetricKeyMetric(
            name="accuracy",
            metric_key="accuracy",
            description="Accuracy metric",
        )
        constraint = metric.above(0.9)

        config: dict[str, Any] = {"model": "gpt-4"}
        metrics = {"accuracy": 0.85}

        result = constraint(config, metrics)
        assert result is False

    def test_constraint_missing_metric(self) -> None:
        """Test constraint with missing metric uses default (0.0)."""
        metric = MetricKeyMetric(
            name="accuracy",
            metric_key="accuracy",
            description="Accuracy metric",
            default=0.0,
        )
        constraint = metric.above(0.9)

        config: dict[str, Any] = {"model": "gpt-4"}
        metrics = {"latency": 100}  # No accuracy metric

        # Default is 0.0, which fails the >= 0.9 threshold
        result = constraint(config, metrics)
        assert result is False

    def test_constraint_less_than(self) -> None:
        """Test less-than constraint."""
        metric = MetricKeyMetric(
            name="error_rate",
            metric_key="error_rate",
            description="Error rate",
        )
        constraint = metric.below(0.1)

        assert constraint({}, {"error_rate": 0.05}) is True
        assert constraint({}, {"error_rate": 0.15}) is False

    def test_constraint_and_combination(self) -> None:
        """Test AND combination of constraints."""
        m1 = MetricKeyMetric(name="accuracy", metric_key="accuracy")
        m2 = MetricKeyMetric(name="latency", metric_key="latency")

        c1 = m1.above(0.9)
        c2 = m2.below(100)

        combined = c1 & c2
        assert isinstance(combined, CompoundSafetyConstraint)
        assert combined.combinator == "and"

        # Both pass
        assert combined({}, {"accuracy": 0.95, "latency": 50}) is True
        # First fails
        assert combined({}, {"accuracy": 0.85, "latency": 50}) is False
        # Second fails
        assert combined({}, {"accuracy": 0.95, "latency": 150}) is False

    def test_constraint_or_combination(self) -> None:
        """Test OR combination of constraints."""
        m1 = MetricKeyMetric(name="accuracy", metric_key="accuracy")
        m2 = MetricKeyMetric(name="latency", metric_key="latency")

        c1 = m1.above(0.9)
        c2 = m2.below(100)

        combined = c1 | c2
        assert isinstance(combined, CompoundSafetyConstraint)
        assert combined.combinator == "or"

        # Both pass
        assert combined({}, {"accuracy": 0.95, "latency": 50}) is True
        # First passes
        assert combined({}, {"accuracy": 0.95, "latency": 150}) is True
        # Second passes
        assert combined({}, {"accuracy": 0.85, "latency": 50}) is True
        # Both fail
        assert combined({}, {"accuracy": 0.85, "latency": 150}) is False


class TestCompoundSafetyConstraint:
    """Tests for CompoundSafetyConstraint."""

    def test_compound_all_mode(self) -> None:
        """Test compound constraint in 'and' mode."""
        m1 = MetricKeyMetric(name="a", metric_key="a")
        m2 = MetricKeyMetric(name="b", metric_key="b")

        compound = CompoundSafetyConstraint(
            constraints=[m1.above(0.5), m2.above(0.5)],
            combinator="and",
        )

        assert compound({}, {"a": 0.6, "b": 0.6}) is True
        assert compound({}, {"a": 0.6, "b": 0.4}) is False

    def test_compound_any_mode(self) -> None:
        """Test compound constraint in 'or' mode."""
        m1 = MetricKeyMetric(name="a", metric_key="a")
        m2 = MetricKeyMetric(name="b", metric_key="b")

        compound = CompoundSafetyConstraint(
            constraints=[m1.above(0.5), m2.above(0.5)],
            combinator="or",
        )

        assert compound({}, {"a": 0.6, "b": 0.6}) is True
        assert compound({}, {"a": 0.6, "b": 0.4}) is True
        assert compound({}, {"a": 0.4, "b": 0.4}) is False

    def test_compound_chaining(self) -> None:
        """Test chaining compound constraints."""
        m1 = MetricKeyMetric(name="a", metric_key="a")
        m2 = MetricKeyMetric(name="b", metric_key="b")
        m3 = MetricKeyMetric(name="c", metric_key="c")

        c1 = m1.above(0.5)
        c2 = m2.above(0.5)
        c3 = m3.above(0.5)

        # (a AND b) OR c
        combined = (c1 & c2) | c3

        assert combined({}, {"a": 0.6, "b": 0.6, "c": 0.4}) is True  # a AND b pass
        assert combined({}, {"a": 0.4, "b": 0.4, "c": 0.6}) is True  # c passes
        assert combined({}, {"a": 0.4, "b": 0.4, "c": 0.4}) is False  # all fail

    def test_compound_chaining_preserves_tree_structure(self) -> None:
        """Test that chaining preserves boolean expression tree structure.

        This test specifically catches the bug where (c1 & c2) | c3 would
        flatten to [c1, c2, c3] with combinator="or", incorrectly evaluating
        as "c1 OR c2 OR c3" instead of "(c1 AND c2) OR c3".
        """
        m1 = MetricKeyMetric(name="a", metric_key="a")
        m2 = MetricKeyMetric(name="b", metric_key="b")
        m3 = MetricKeyMetric(name="c", metric_key="c")

        c1 = m1.above(0.5)
        c2 = m2.above(0.5)
        c3 = m3.above(0.5)

        # (a AND b) OR c
        combined = (c1 & c2) | c3

        # This is the key test case that catches the flattening bug:
        # a=False, b=True, c=False
        # Correct: (False AND True) OR False = False OR False = False
        # Buggy (flattened): False OR True OR False = True
        assert combined({}, {"a": 0.4, "b": 0.6, "c": 0.4}) is False

        # Also test the inverse: a OR (b AND c)
        combined2 = c1 | (c2 & c3)

        # a=False, b=True, c=False
        # Correct: False OR (True AND False) = False OR False = False
        # Buggy (flattened): False OR True OR False = True
        assert combined2({}, {"a": 0.4, "b": 0.6, "c": 0.4}) is False

        # Verify correct behavior when inner compound passes
        # a=True, b=True, c=False → (True AND True) OR False = True
        assert combined({}, {"a": 0.6, "b": 0.6, "c": 0.4}) is True


class TestSafetyValidator:
    """Tests for SafetyValidator with Clopper-Pearson bounds."""

    def test_validator_simple_threshold(self) -> None:
        """Test validation without statistical confidence."""
        metric = MetricKeyMetric(name="accuracy", metric_key="accuracy")
        constraint = metric.above(0.9)
        validator = SafetyValidator()

        # Record 95 passing trials out of 100
        for _ in range(95):
            validator.record_result(constraint, {}, {"accuracy": 0.95})
        for _ in range(5):
            validator.record_result(constraint, {}, {"accuracy": 0.85})

        result = validator.validate(constraint)

        assert isinstance(result, SafetyValidationResult)
        assert result.satisfied is True
        assert result.observed_rate == 0.95
        assert result.threshold == 0.9

    def test_validator_with_confidence(self) -> None:
        """Test validation with Clopper-Pearson confidence bounds."""
        metric = MetricKeyMetric(name="accuracy", metric_key="accuracy")
        constraint = metric.above(0.85, confidence=0.95)
        validator = SafetyValidator()

        # Record 90 passing trials out of 100
        for _ in range(90):
            validator.record_result(constraint, {}, {"accuracy": 0.9})
        for _ in range(10):
            validator.record_result(constraint, {}, {"accuracy": 0.8})

        result = validator.validate(constraint)

        assert result.observed_rate == 0.90
        # Lower bound should be less than observed rate
        assert result.lower_bound < result.observed_rate
        # Decision should be based on lower bound vs threshold
        assert result.satisfied == (result.lower_bound >= 0.85)

    def test_validator_edge_cases(self) -> None:
        """Test validator edge cases."""
        metric = MetricKeyMetric(name="accuracy", metric_key="accuracy")
        constraint = metric.above(0.9)
        validator = SafetyValidator()

        # Zero trials - validate without recording any results
        result = validator.validate(constraint)
        assert result.satisfied is False
        assert result.sample_count == 0

        # Now record all successes
        validator2 = SafetyValidator()
        for _ in range(100):
            validator2.record_result(constraint, {}, {"accuracy": 0.95})
        result = validator2.validate(constraint)
        assert result.satisfied is True
        assert result.observed_rate == 1.0

        # All failures
        validator3 = SafetyValidator()
        for _ in range(100):
            validator3.record_result(constraint, {}, {"accuracy": 0.8})
        result = validator3.validate(constraint)
        assert result.satisfied is False
        assert result.observed_rate == 0.0


class TestNonRAGASPresets:
    """Tests for non-RAGAS metric factory functions."""

    def test_hallucination_rate_factory(self) -> None:
        """Test hallucination_rate() factory."""
        metric = hallucination_rate()

        assert isinstance(metric, MetricKeyMetric)
        assert metric.metric_key == "hallucination_rate"

        constraint = metric.below(0.1)
        assert constraint.threshold.operator == "<="
        assert constraint.threshold.value == 0.1

    def test_hallucination_rate_evaluation(self) -> None:
        """Test hallucination_rate constraint evaluation."""
        constraint = hallucination_rate().below(0.1)

        # Low hallucination rate passes
        assert constraint({}, {"hallucination_rate": 0.05}) is True
        # High hallucination rate fails
        assert constraint({}, {"hallucination_rate": 0.15}) is False
        # Boundary case
        assert constraint({}, {"hallucination_rate": 0.1}) is True

    def test_hallucination_rate_default_value(self) -> None:
        """Test hallucination_rate defaults to 1.0 (fail-safe) when missing."""
        constraint = hallucination_rate().below(0.1)

        # When metric is missing, default is 1.0 (worst case)
        # 1.0 > 0.1 so constraint should fail
        assert constraint({}, {}) is False
        assert constraint({}, {"other_metric": 0.5}) is False

    def test_toxicity_score_factory(self) -> None:
        """Test toxicity_score() factory."""
        metric = toxicity_score()

        assert isinstance(metric, MetricKeyMetric)
        assert metric.metric_key == "toxicity"

        constraint = metric.below(0.05)
        assert constraint.threshold.operator == "<="

    def test_toxicity_score_evaluation(self) -> None:
        """Test toxicity_score constraint evaluation."""
        constraint = toxicity_score().below(0.05)

        assert constraint({}, {"toxicity": 0.02}) is True
        assert constraint({}, {"toxicity": 0.08}) is False
        # Default is 1.0 when missing
        assert constraint({}, {}) is False

    def test_bias_score_factory(self) -> None:
        """Test bias_score() factory."""
        metric = bias_score()

        assert isinstance(metric, MetricKeyMetric)
        assert metric.metric_key == "bias"

    def test_bias_score_evaluation(self) -> None:
        """Test bias_score constraint evaluation."""
        constraint = bias_score().below(0.1)

        assert constraint({}, {"bias": 0.05}) is True
        assert constraint({}, {"bias": 0.15}) is False

    def test_safety_score_factory(self) -> None:
        """Test safety_score() factory."""
        metric = safety_score()

        assert isinstance(metric, MetricKeyMetric)
        assert metric.metric_key == "safety_score"

    def test_safety_score_evaluation(self) -> None:
        """Test safety_score constraint evaluation (higher is better)."""
        constraint = safety_score().above(0.9)

        assert constraint({}, {"safety_score": 0.95}) is True
        assert constraint({}, {"safety_score": 0.85}) is False
        # Default is 0.0 when missing (fail-safe for higher-is-better)
        assert constraint({}, {}) is False

    def test_custom_safety_factory(self) -> None:
        """Test custom_safety() factory with custom evaluator."""

        def my_evaluator(config: dict[str, Any], metrics: dict[str, Any]) -> float:
            return 1.0 if metrics.get("match", False) else 0.0

        metric = custom_safety(
            name="exact_match",
            evaluator=my_evaluator,
            description="Exact string match",
        )

        assert metric.name == "exact_match"
        assert metric.description == "Exact string match"

    def test_custom_safety_evaluation(self) -> None:
        """Test custom_safety constraint evaluation."""

        def my_evaluator(config: dict[str, Any], metrics: dict[str, Any]) -> float:
            return float(metrics.get("custom_score", 0.0))

        metric = custom_safety(
            name="custom",
            evaluator=my_evaluator,
            description="Custom metric",
        )
        constraint = metric.above(0.8)

        assert constraint({}, {"custom_score": 0.9}) is True
        assert constraint({}, {"custom_score": 0.7}) is False
        assert constraint({}, {}) is False  # Default 0.0

    def test_custom_safety_with_config(self) -> None:
        """Test custom_safety evaluator can access config."""

        def config_aware_evaluator(
            config: dict[str, Any], metrics: dict[str, Any]
        ) -> float:
            # Score based on config
            score = float(metrics.get("score", 0.0))
            if config.get("model") == "gpt-4":
                return score * 1.1  # Boost for gpt-4
            return score

        metric = custom_safety(
            name="config_aware",
            evaluator=config_aware_evaluator,
            description="Config-aware metric",
        )
        constraint = metric.above(1.0)

        # With gpt-4, 0.95 * 1.1 = 1.045 >= 1.0
        assert constraint({"model": "gpt-4"}, {"score": 0.95}) is True
        # Without gpt-4 boost, 0.95 < 1.0
        assert constraint({"model": "gpt-3"}, {"score": 0.95}) is False

    def test_custom_metric_key(self) -> None:
        """Test factory with custom metric key."""
        metric = hallucination_rate(metric_key="custom_hallucination")

        assert metric.metric_key == "custom_hallucination"

        # Verify it reads from custom key
        constraint = metric.below(0.1)
        assert constraint({}, {"custom_hallucination": 0.05}) is True
        assert constraint({}, {"hallucination_rate": 0.05}) is False  # Wrong key


class TestRAGASPresets:
    """Tests for RAGAS metric presets."""

    def test_faithfulness_preset(self) -> None:
        """Test faithfulness RAGAS preset."""
        from traigent.api.safety import faithfulness

        assert faithfulness.name == "faithfulness"
        assert "factual" in faithfulness.description.lower()

        # Test constraint creation
        constraint = faithfulness.above(0.9)
        assert constraint.threshold.value == 0.9
        assert constraint.threshold.operator == ">="

    def test_answer_relevancy_preset(self) -> None:
        """Test answer_relevancy RAGAS preset."""
        from traigent.api.safety import answer_relevancy

        assert answer_relevancy.name == "answer_relevancy"
        assert "relevant" in answer_relevancy.description.lower()

        constraint = answer_relevancy.above(0.8)
        assert constraint.threshold.value == 0.8

    def test_context_precision_preset(self) -> None:
        """Test context_precision RAGAS preset."""
        from traigent.api.safety import context_precision

        assert context_precision.name == "context_precision"
        constraint = context_precision.above(0.7)
        assert constraint.threshold.value == 0.7

    def test_context_recall_preset(self) -> None:
        """Test context_recall RAGAS preset."""
        from traigent.api.safety import context_recall

        assert context_recall.name == "context_recall"
        constraint = context_recall.above(0.85)
        assert constraint.threshold.value == 0.85

    def test_answer_similarity_preset(self) -> None:
        """Test answer_similarity RAGAS preset."""
        from traigent.api.safety import answer_similarity

        assert answer_similarity.name == "answer_similarity"
        constraint = answer_similarity.above(0.9)
        assert constraint.threshold.value == 0.9

    def test_ragas_presets_support_between(self) -> None:
        """Test that RAGAS presets support .between() method."""
        from traigent.api.safety import faithfulness

        constraint = faithfulness.between(0.7, 0.95)
        assert isinstance(constraint, CompoundSafetyConstraint)
        assert constraint.combinator == "and"

    def test_ragas_presets_support_confidence(self) -> None:
        """Test that RAGAS presets support confidence parameter."""
        from traigent.api.safety import faithfulness

        constraint = faithfulness.above(0.9, confidence=0.95)
        assert constraint.threshold.confidence == 0.95


class TestAvailablePresets:
    """Tests for get_available_safety_presets()."""

    def test_returns_dict(self) -> None:
        """Test that function returns a dictionary."""
        presets = get_available_safety_presets()
        assert isinstance(presets, dict)

    def test_includes_non_ragas(self) -> None:
        """Test that non-RAGAS presets are always included."""
        presets = get_available_safety_presets()

        # Non-RAGAS presets should always be available
        non_ragas_keys = {
            "hallucination_rate",
            "toxicity_score",
            "bias_score",
            "safety_score",
        }
        for key in non_ragas_keys:
            assert (
                key in presets
            ), f"Expected '{key}' in presets, got {list(presets.keys())}"


class TestDecoratorIntegration:
    """Tests for integration with @optimize decorator."""

    def test_safety_constraints_accepted(self) -> None:
        """Test that safety_constraints parameter is accepted."""
        from traigent.api.decorators import optimize

        # This should not raise - just testing parameter acceptance
        metric = hallucination_rate()
        constraint = metric.below(0.1)

        @optimize(
            objectives=["accuracy"],
            configuration_space={"model": ["gpt-4"]},
            safety_constraints=[constraint],
        )
        def my_func(x: str) -> str:
            return x

        assert hasattr(my_func, "safety_constraints")
        assert my_func.safety_constraints == [constraint]

    def test_multiple_safety_constraints(self) -> None:
        """Test multiple safety constraints."""
        from traigent.api.decorators import optimize

        c1 = hallucination_rate().below(0.1)
        c2 = toxicity_score().below(0.05)

        @optimize(
            objectives=["accuracy"],
            configuration_space={"model": ["gpt-4"]},
            safety_constraints=[c1, c2],
        )
        def my_func(x: str) -> str:
            return x

        assert len(my_func.safety_constraints) == 2

    def test_compound_safety_constraint(self) -> None:
        """Test compound safety constraint in decorator."""
        from traigent.api.decorators import optimize

        c1 = hallucination_rate().below(0.1)
        c2 = toxicity_score().below(0.05)
        combined = c1 & c2

        @optimize(
            objectives=["accuracy"],
            configuration_space={"model": ["gpt-4"]},
            safety_constraints=[combined],
        )
        def my_func(x: str) -> str:
            return x

        assert len(my_func.safety_constraints) == 1
        assert isinstance(my_func.safety_constraints[0], CompoundSafetyConstraint)


class TestAPIExports:
    """Tests for public API exports."""

    def test_all_presets_exported(self) -> None:
        """Test that all presets are exported from traigent.api."""
        from traigent.api import (
            CompoundSafetyConstraint,
            SafetyConstraint,
            SafetyThreshold,
            SafetyValidator,
            bias_score,
            custom_safety,
            get_available_safety_presets,
            hallucination_rate,
            safety_score,
            toxicity_score,
        )

        # Just verify imports work
        assert SafetyConstraint is not None
        assert CompoundSafetyConstraint is not None
        assert SafetyThreshold is not None
        assert SafetyValidator is not None
        assert hallucination_rate is not None
        assert toxicity_score is not None
        assert bias_score is not None
        assert safety_score is not None
        assert custom_safety is not None
        assert get_available_safety_presets is not None

    def test_ragas_presets_exported(self) -> None:
        """Test that RAGAS presets are exported from traigent.api."""
        from traigent.api import (
            answer_relevancy,
            answer_similarity,
            context_precision,
            context_recall,
            faithfulness,
        )

        # Verify imports work (they should be lazy-loaded metric objects)
        assert faithfulness is not None
        assert answer_relevancy is not None
        assert context_precision is not None
        assert context_recall is not None
        assert answer_similarity is not None


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_exact_boundary_values(self) -> None:
        """Test constraint evaluation at exact boundary values."""
        metric = MetricKeyMetric(name="score", metric_key="score")

        # >= operator at exact boundary
        constraint_gte = metric.above(0.9)
        assert constraint_gte({}, {"score": 0.9}) is True
        assert constraint_gte({}, {"score": 0.8999999}) is False

        # <= operator at exact boundary
        constraint_lte = metric.below(0.1)
        assert constraint_lte({}, {"score": 0.1}) is True
        assert constraint_lte({}, {"score": 0.1000001}) is False

    def test_zero_and_one_values(self) -> None:
        """Test constraint evaluation with 0 and 1 values."""
        metric = MetricKeyMetric(name="score", metric_key="score")

        constraint_above = metric.above(0.5)
        assert constraint_above({}, {"score": 0.0}) is False
        assert constraint_above({}, {"score": 1.0}) is True

        constraint_below = metric.below(0.5)
        assert constraint_below({}, {"score": 0.0}) is True
        assert constraint_below({}, {"score": 1.0}) is False

    def test_between_constraint_evaluation(self) -> None:
        """Test between() constraint evaluation."""
        metric = MetricKeyMetric(name="latency", metric_key="latency")

        constraint = metric.between(10, 100)

        # Within range
        assert constraint({}, {"latency": 50}) is True
        assert constraint({}, {"latency": 10}) is True  # Lower boundary
        assert constraint({}, {"latency": 100}) is True  # Upper boundary

        # Outside range
        assert constraint({}, {"latency": 5}) is False
        assert constraint({}, {"latency": 150}) is False

    def test_nan_handling(self) -> None:
        """Test that NaN values fall back to default."""
        import math

        metric = MetricKeyMetric(name="score", metric_key="score", default=0.0)
        constraint = metric.above(0.5)

        # NaN should use default (0.0), which fails >= 0.5
        assert constraint({}, {"score": float("nan")}) is False

        # Verify explicit NaN via math.nan
        assert constraint({}, {"score": math.nan}) is False

    def test_integer_metric_values(self) -> None:
        """Test that integer metric values are handled correctly."""
        metric = MetricKeyMetric(name="count", metric_key="count")

        constraint = metric.above(5)
        assert constraint({}, {"count": 10}) is True
        assert constraint({}, {"count": 3}) is False
        assert constraint({}, {"count": 5}) is True  # Boundary

    def test_deeply_nested_compound_constraints(self) -> None:
        """Test deeply nested compound constraint structures."""
        m1 = MetricKeyMetric(name="a", metric_key="a")
        m2 = MetricKeyMetric(name="b", metric_key="b")
        m3 = MetricKeyMetric(name="c", metric_key="c")
        m4 = MetricKeyMetric(name="d", metric_key="d")

        c1 = m1.above(0.5)
        c2 = m2.above(0.5)
        c3 = m3.above(0.5)
        c4 = m4.above(0.5)

        # ((a AND b) OR c) AND d
        combined = ((c1 & c2) | c3) & c4

        # d must pass, plus either (a AND b) or c
        assert combined({}, {"a": 0.6, "b": 0.6, "c": 0.4, "d": 0.6}) is True
        assert combined({}, {"a": 0.4, "b": 0.4, "c": 0.6, "d": 0.6}) is True
        assert (
            combined({}, {"a": 0.6, "b": 0.6, "c": 0.4, "d": 0.4}) is False
        )  # d fails
        assert (
            combined({}, {"a": 0.4, "b": 0.6, "c": 0.4, "d": 0.6}) is False
        )  # a&b fails, c fails


class TestSafetyValidatorAdvanced:
    """Advanced tests for SafetyValidator."""

    def test_validator_reset(self) -> None:
        """Test validator reset() clears all results."""
        metric = MetricKeyMetric(name="accuracy", metric_key="accuracy")
        constraint = metric.above(0.9)
        validator = SafetyValidator()

        # Record some results
        for _ in range(10):
            validator.record_result(constraint, {}, {"accuracy": 0.95})

        result = validator.validate(constraint)
        assert result.sample_count == 10

        # Reset and verify
        validator.reset()
        result = validator.validate(constraint)
        assert result.sample_count == 0
        assert result.satisfied is False

    def test_validator_multiple_constraints(self) -> None:
        """Test validator can track multiple constraints separately."""
        m1 = MetricKeyMetric(name="accuracy", metric_key="accuracy")
        m2 = MetricKeyMetric(name="latency", metric_key="latency")

        c1 = m1.above(0.9)
        c2 = m2.below(100)

        validator = SafetyValidator()

        # Record results for c1
        for _ in range(10):
            validator.record_result(c1, {}, {"accuracy": 0.95})

        # Record results for c2
        for _ in range(5):
            validator.record_result(c2, {}, {"latency": 50})

        # Validate each separately
        result1 = validator.validate(c1)
        result2 = validator.validate(c2)

        assert result1.sample_count == 10
        assert result2.sample_count == 5
        assert result1.satisfied is True
        assert result2.satisfied is True

    def test_validator_record_returns_result(self) -> None:
        """Test that record_result returns the constraint result."""
        metric = MetricKeyMetric(name="accuracy", metric_key="accuracy")
        constraint = metric.above(0.9)
        validator = SafetyValidator()

        # Passing trial
        result = validator.record_result(constraint, {}, {"accuracy": 0.95})
        assert result is True

        # Failing trial
        result = validator.record_result(constraint, {}, {"accuracy": 0.85})
        assert result is False

    def test_validator_clopper_pearson_bounds(self) -> None:
        """Test Clopper-Pearson confidence interval calculation."""
        metric = MetricKeyMetric(name="accuracy", metric_key="accuracy")
        constraint = metric.above(0.8, confidence=0.95)
        validator = SafetyValidator()

        # Record 80 passing, 20 failing (80% success rate)
        for _ in range(80):
            validator.record_result(constraint, {}, {"accuracy": 0.85})
        for _ in range(20):
            validator.record_result(constraint, {}, {"accuracy": 0.75})

        result = validator.validate(constraint)

        assert result.observed_rate == 0.8
        assert result.sample_count == 100
        # Lower bound should be below observed rate for 95% CI
        assert result.lower_bound < result.observed_rate
        # Lower bound should be above 0.7 for 100 samples at 80% success
        assert result.lower_bound > 0.7

    def test_validator_validation_message(self) -> None:
        """Test that validation result includes descriptive message."""
        metric = MetricKeyMetric(name="accuracy", metric_key="accuracy")
        constraint = metric.above(0.9)
        validator = SafetyValidator()

        validator.record_result(constraint, {}, {"accuracy": 0.95})
        result = validator.validate(constraint)

        assert "accuracy" in result.message
        assert "100" in result.message or "meets" in result.message

    def test_validator_with_above_constraint(self) -> None:
        """Test validator correctly tracks pass rate for above() constraints.

        The validator tracks what percentage of trials PASS the constraint,
        then compares that pass rate against the threshold value.
        """
        metric = MetricKeyMetric(name="accuracy", metric_key="accuracy")
        # Constraint: accuracy >= 0.9
        constraint = metric.above(0.9)
        validator = SafetyValidator()

        # Record 8 passing trials (accuracy >= 0.9)
        for _ in range(8):
            validator.record_result(constraint, {}, {"accuracy": 0.95})
        # Record 2 failing trials (accuracy < 0.9)
        for _ in range(2):
            validator.record_result(constraint, {}, {"accuracy": 0.85})

        result = validator.validate(constraint)
        # 80% of trials passed the constraint
        assert result.observed_rate == 0.8
        assert result.sample_count == 10
        # For >= constraint, validator checks: 0.8 >= 0.9 → False
        assert result.satisfied is False

    def test_validator_mixed_results(self) -> None:
        """Test validator with mixed pass/fail results."""
        metric = MetricKeyMetric(name="accuracy", metric_key="accuracy")
        constraint = metric.above(0.9)
        validator = SafetyValidator()

        # 70% pass rate
        for _ in range(70):
            validator.record_result(constraint, {}, {"accuracy": 0.95})
        for _ in range(30):
            validator.record_result(constraint, {}, {"accuracy": 0.85})

        result = validator.validate(constraint)
        assert result.observed_rate == 0.7
        # Should fail because 70% < 90% threshold
        assert result.satisfied is False


class TestMetricKeyMetricAdvanced:
    """Advanced tests for MetricKeyMetric."""

    def test_invert_parameter(self) -> None:
        """Test MetricKeyMetric with invert=True."""
        # For metrics where lower is better, invert flips the value
        metric = MetricKeyMetric(
            name="error_rate",
            metric_key="error_rate",
            invert=True,
        )

        # With invert=True, value becomes (1 - value)
        # So error_rate=0.1 becomes 0.9
        constraint = metric.above(0.9)
        assert constraint({}, {"error_rate": 0.1}) is True  # 1 - 0.1 = 0.9 >= 0.9
        assert constraint({}, {"error_rate": 0.2}) is False  # 1 - 0.2 = 0.8 < 0.9

    def test_description_attribute(self) -> None:
        """Test that description is properly stored."""
        metric = MetricKeyMetric(
            name="test",
            metric_key="test",
            description="A test metric",
        )
        assert metric.description == "A test metric"

    def test_name_attribute(self) -> None:
        """Test that name is properly stored."""
        metric = MetricKeyMetric(
            name="test_metric",
            metric_key="test_key",
        )
        assert metric.name == "test_metric"
        assert metric.metric_key == "test_key"


class TestCompoundConstraintEdgeCases:
    """Edge case tests for compound constraints."""

    def test_single_constraint_in_compound(self) -> None:
        """Test compound with single constraint."""
        metric = MetricKeyMetric(name="a", metric_key="a")
        c1 = metric.above(0.5)

        compound = CompoundSafetyConstraint(
            constraints=[c1],
            combinator="and",
        )

        assert compound({}, {"a": 0.6}) is True
        assert compound({}, {"a": 0.4}) is False

    def test_empty_metrics_dict(self) -> None:
        """Test constraint with empty metrics dict."""
        metric = MetricKeyMetric(name="a", metric_key="a", default=0.0)
        constraint = metric.above(0.5)

        # Empty metrics should use default (0.0), failing >= 0.5
        assert constraint({}, {}) is False

    def test_constraint_requires_metrics_property(self) -> None:
        """Test requires_metrics property."""
        metric = MetricKeyMetric(name="a", metric_key="a")
        constraint = metric.above(0.5)

        assert constraint.requires_metrics is True

        compound = constraint & metric.below(0.9)
        assert compound.requires_metrics is True
