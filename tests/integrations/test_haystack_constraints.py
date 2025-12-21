"""Tests for Haystack metric constraints module.

Coverage: Epic 4, Story 4.3 (Define Cost and Latency Constraints)
"""

from __future__ import annotations

import pytest

from traigent.integrations.haystack.metric_constraints import (
    ConstraintCheckResult,
    ConstraintViolation,
    MetricConstraint,
    check_constraints,
    cost_constraint,
    latency_constraint,
    quality_constraint,
)


class TestMetricConstraint:
    """Tests for MetricConstraint class."""

    def test_less_than_or_equal(self):
        """Test <= operator."""
        constraint = MetricConstraint("value", "<=", 100)
        assert constraint.check({"value": 50})
        assert constraint.check({"value": 100})
        assert not constraint.check({"value": 101})

    def test_less_than(self):
        """Test < operator."""
        constraint = MetricConstraint("value", "<", 100)
        assert constraint.check({"value": 50})
        assert not constraint.check({"value": 100})
        assert not constraint.check({"value": 101})

    def test_greater_than_or_equal(self):
        """Test >= operator."""
        constraint = MetricConstraint("accuracy", ">=", 0.8)
        assert constraint.check({"accuracy": 0.9})
        assert constraint.check({"accuracy": 0.8})
        assert not constraint.check({"accuracy": 0.7})

    def test_greater_than(self):
        """Test > operator."""
        constraint = MetricConstraint("value", ">", 100)
        assert constraint.check({"value": 101})
        assert not constraint.check({"value": 100})
        assert not constraint.check({"value": 50})

    def test_equal(self):
        """Test == operator."""
        constraint = MetricConstraint("value", "==", 100)
        assert constraint.check({"value": 100})
        assert not constraint.check({"value": 101})
        assert not constraint.check({"value": 99})

    def test_not_equal(self):
        """Test != operator."""
        constraint = MetricConstraint("value", "!=", 100)
        assert constraint.check({"value": 101})
        assert constraint.check({"value": 99})
        assert not constraint.check({"value": 100})

    def test_missing_metric(self):
        """Test handling of missing metric."""
        constraint = MetricConstraint("missing", "<=", 100)
        assert not constraint.check({"other": 50})

    def test_none_value(self):
        """Test handling of None metric value."""
        constraint = MetricConstraint("value", "<=", 100)
        assert not constraint.check({"value": None})

    def test_invalid_operator(self):
        """Test that invalid operators raise error."""
        with pytest.raises(ValueError, match="Invalid operator"):
            MetricConstraint("value", "??", 100)

    def test_default_name(self):
        """Test default name generation."""
        constraint = MetricConstraint("latency_p95_ms", "<=", 500)
        assert constraint.name == "latency_p95_ms <= 500"

    def test_custom_name(self):
        """Test custom name."""
        constraint = MetricConstraint("latency_p95_ms", "<=", 500, name="my_constraint")
        assert constraint.name == "my_constraint"

    def test_violation_message(self):
        """Test violation message generation."""
        constraint = MetricConstraint("cost", "<=", 0.05)
        message = constraint.get_violation_message({"cost": 0.10})
        assert "cost" in message
        assert "0.10" in message or "0.1" in message
        assert "<=" in message
        assert "0.05" in message

    def test_violation_message_missing_metric(self):
        """Test violation message when metric is missing."""
        constraint = MetricConstraint("cost", "<=", 0.05)
        message = constraint.get_violation_message({})
        assert "missing" in message


class TestCheckConstraints:
    """Tests for check_constraints function."""

    def test_empty_constraints(self):
        """Test with empty constraints list."""
        result = check_constraints([], {"value": 100})
        assert result.all_satisfied
        assert result.satisfied_count == 0
        assert result.total_count == 0
        assert len(result.violations) == 0

    def test_all_satisfied(self):
        """Test when all constraints are satisfied."""
        constraints = [
            MetricConstraint("cost", "<=", 0.10),
            MetricConstraint("latency", "<=", 1000),
            MetricConstraint("accuracy", ">=", 0.8),
        ]
        metrics = {"cost": 0.05, "latency": 500, "accuracy": 0.9}

        result = check_constraints(constraints, metrics)

        assert result.all_satisfied
        assert result.satisfied_count == 3
        assert result.total_count == 3
        assert len(result.violations) == 0

    def test_some_violated(self):
        """Test when some constraints are violated."""
        constraints = [
            MetricConstraint("cost", "<=", 0.05),  # Violated
            MetricConstraint("latency", "<=", 1000),  # Satisfied
            MetricConstraint("accuracy", ">=", 0.9),  # Violated
        ]
        metrics = {"cost": 0.10, "latency": 500, "accuracy": 0.8}

        result = check_constraints(constraints, metrics)

        assert not result.all_satisfied
        assert result.satisfied_count == 1
        assert result.total_count == 3
        assert len(result.violations) == 2

    def test_all_violated(self):
        """Test when all constraints are violated."""
        constraints = [
            MetricConstraint("cost", "<=", 0.01),
            MetricConstraint("accuracy", ">=", 0.99),
        ]
        metrics = {"cost": 0.10, "accuracy": 0.5}

        result = check_constraints(constraints, metrics)

        assert not result.all_satisfied
        assert result.satisfied_count == 0
        assert result.total_count == 2
        assert len(result.violations) == 2

    def test_violation_details(self):
        """Test violation details are captured."""
        constraints = [MetricConstraint("cost", "<=", 0.05)]
        metrics = {"cost": 0.10}

        result = check_constraints(constraints, metrics)

        assert len(result.violations) == 1
        violation = result.violations[0]
        assert violation.constraint == constraints[0]
        assert violation.actual_value == 0.10
        assert "cost" in violation.message

    def test_violation_messages_property(self):
        """Test violation_messages property."""
        constraints = [
            MetricConstraint("cost", "<=", 0.05),
            MetricConstraint("latency", "<=", 100),
        ]
        metrics = {"cost": 0.10, "latency": 500}

        result = check_constraints(constraints, metrics)

        messages = result.violation_messages
        assert len(messages) == 2
        assert any("cost" in m for m in messages)
        assert any("latency" in m for m in messages)


class TestCostConstraint:
    """Tests for cost_constraint helper."""

    def test_default_metric_name(self):
        """Test default total_cost metric name."""
        constraint = cost_constraint(max_cost=0.05)
        assert constraint.metric_name == "total_cost"
        assert constraint.op == "<="
        assert constraint.threshold == 0.05

    def test_custom_metric_name(self):
        """Test custom metric name."""
        constraint = cost_constraint(max_cost=0.05, metric_name="api_cost")
        assert constraint.metric_name == "api_cost"

    def test_constraint_check(self):
        """Test constraint behavior."""
        constraint = cost_constraint(max_cost=0.05)
        assert constraint.check({"total_cost": 0.03})
        assert constraint.check({"total_cost": 0.05})
        assert not constraint.check({"total_cost": 0.06})


class TestLatencyConstraint:
    """Tests for latency_constraint helper."""

    def test_p50_only(self):
        """Test with only p50 specified."""
        constraints = latency_constraint(p50_ms=100)
        assert len(constraints) == 1
        assert constraints[0].metric_name == "latency_p50_ms"
        assert constraints[0].threshold == 100

    def test_p95_only(self):
        """Test with only p95 specified."""
        constraints = latency_constraint(p95_ms=500)
        assert len(constraints) == 1
        assert constraints[0].metric_name == "latency_p95_ms"

    def test_p99_only(self):
        """Test with only p99 specified."""
        constraints = latency_constraint(p99_ms=1000)
        assert len(constraints) == 1
        assert constraints[0].metric_name == "latency_p99_ms"

    def test_multiple_percentiles(self):
        """Test with multiple percentiles."""
        constraints = latency_constraint(p50_ms=100, p95_ms=500, p99_ms=1000)
        assert len(constraints) == 3
        names = [c.metric_name for c in constraints]
        assert "latency_p50_ms" in names
        assert "latency_p95_ms" in names
        assert "latency_p99_ms" in names

    def test_mean_and_max(self):
        """Test mean and max latency constraints."""
        constraints = latency_constraint(mean_ms=200, max_ms=1000)
        assert len(constraints) == 2
        names = [c.metric_name for c in constraints]
        assert "latency_mean_ms" in names
        assert "latency_max_ms" in names

    def test_no_thresholds_raises(self):
        """Test that no thresholds raises error."""
        with pytest.raises(ValueError, match="At least one latency threshold"):
            latency_constraint()

    def test_constraint_check(self):
        """Test constraint behavior."""
        constraints = latency_constraint(p95_ms=500)
        assert constraints[0].check({"latency_p95_ms": 300})
        assert constraints[0].check({"latency_p95_ms": 500})
        assert not constraints[0].check({"latency_p95_ms": 600})


class TestQualityConstraint:
    """Tests for quality_constraint helper."""

    def test_min_only(self):
        """Test with only min_value specified."""
        constraints = quality_constraint("accuracy", min_value=0.8)
        assert len(constraints) == 1
        assert constraints[0].metric_name == "accuracy"
        assert constraints[0].op == ">="
        assert constraints[0].threshold == 0.8

    def test_max_only(self):
        """Test with only max_value specified."""
        constraints = quality_constraint("error_rate", max_value=0.1)
        assert len(constraints) == 1
        assert constraints[0].metric_name == "error_rate"
        assert constraints[0].op == "<="
        assert constraints[0].threshold == 0.1

    def test_min_and_max(self):
        """Test with both min and max."""
        constraints = quality_constraint("score", min_value=0.5, max_value=1.0)
        assert len(constraints) == 2
        ops = [c.op for c in constraints]
        assert ">=" in ops
        assert "<=" in ops

    def test_no_values_raises(self):
        """Test that no values raises error."""
        with pytest.raises(ValueError, match="At least one of min_value or max_value"):
            quality_constraint("accuracy")

    def test_constraint_check(self):
        """Test constraint behavior."""
        constraints = quality_constraint("accuracy", min_value=0.8)
        assert constraints[0].check({"accuracy": 0.9})
        assert constraints[0].check({"accuracy": 0.8})
        assert not constraints[0].check({"accuracy": 0.7})


class TestConstraintViolation:
    """Tests for ConstraintViolation dataclass."""

    def test_creation(self):
        """Test ConstraintViolation creation."""
        constraint = MetricConstraint("cost", "<=", 0.05)
        violation = ConstraintViolation(
            constraint=constraint,
            actual_value=0.10,
            message="Cost exceeded",
        )
        assert violation.constraint == constraint
        assert violation.actual_value == 0.10
        assert violation.message == "Cost exceeded"


class TestConstraintCheckResult:
    """Tests for ConstraintCheckResult dataclass."""

    def test_default_values(self):
        """Test default values."""
        result = ConstraintCheckResult(all_satisfied=True)
        assert result.all_satisfied
        assert result.violations == []
        assert result.satisfied_count == 0
        assert result.total_count == 0

    def test_violation_messages_empty(self):
        """Test violation_messages with no violations."""
        result = ConstraintCheckResult(all_satisfied=True)
        assert result.violation_messages == []


class TestIntegration:
    """Integration tests for constraint checking."""

    @pytest.mark.asyncio
    async def test_evaluator_with_constraints(self):
        """Test HaystackEvaluator with constraints."""
        from unittest.mock import MagicMock

        from traigent.integrations.haystack import EvaluationDataset, HaystackEvaluator

        # Create mock pipeline
        pipeline = MagicMock()
        pipeline.run.return_value = {
            "llm": {
                "replies": ["Response"],
                "meta": [
                    {
                        "model": "gpt-4o",
                        "usage": {"prompt_tokens": 100, "completion_tokens": 50},
                    }
                ],
            }
        }

        dataset = EvaluationDataset.from_dicts(
            [{"input": {"query": "test"}, "expected": "result"}]
        )

        # Define constraints - cost constraint will be violated
        constraints = [
            cost_constraint(max_cost=0.0001),  # Very low - will fail
        ]

        evaluator = HaystackEvaluator(
            pipeline=pipeline,
            haystack_dataset=dataset,
            output_key="llm.replies",
            track_costs=True,
            constraints=constraints,
        )

        result = await evaluator.evaluate(
            func=pipeline.run,
            config={},
            dataset=dataset.to_core_dataset(),
        )

        # Check constraint results are in metrics
        assert "constraints_satisfied" in result.aggregated_metrics
        assert result.aggregated_metrics["constraints_satisfied"] is False
        assert result.aggregated_metrics["constraints_checked"] == 1
        assert result.aggregated_metrics["constraints_passed"] == 0

    @pytest.mark.asyncio
    async def test_evaluator_constraints_satisfied(self):
        """Test HaystackEvaluator with satisfied constraints."""
        from unittest.mock import MagicMock

        from traigent.integrations.haystack import EvaluationDataset, HaystackEvaluator

        pipeline = MagicMock()
        pipeline.run.return_value = {
            "llm": {
                "replies": ["Response"],
                "meta": [
                    {
                        "model": "gpt-4o",
                        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
                    }
                ],
            }
        }

        dataset = EvaluationDataset.from_dicts(
            [{"input": {"query": "test"}, "expected": "result"}]
        )

        # Define constraints - very lenient
        constraints = [
            cost_constraint(max_cost=1.0),  # High limit - will pass
        ]

        evaluator = HaystackEvaluator(
            pipeline=pipeline,
            haystack_dataset=dataset,
            output_key="llm.replies",
            track_costs=True,
            constraints=constraints,
        )

        result = await evaluator.evaluate(
            func=pipeline.run,
            config={},
            dataset=dataset.to_core_dataset(),
        )

        # Check constraint results
        assert result.aggregated_metrics["constraints_satisfied"] is True
        assert result.aggregated_metrics["constraints_passed"] == 1

    @pytest.mark.asyncio
    async def test_evaluator_no_constraints(self):
        """Test HaystackEvaluator without constraints."""
        from unittest.mock import MagicMock

        from traigent.integrations.haystack import EvaluationDataset, HaystackEvaluator

        pipeline = MagicMock()
        pipeline.run.return_value = {"llm": {"replies": ["Response"]}}

        dataset = EvaluationDataset.from_dicts(
            [{"input": {"query": "test"}, "expected": "result"}]
        )

        evaluator = HaystackEvaluator(
            pipeline=pipeline,
            haystack_dataset=dataset,
            output_key="llm.replies",
        )

        result = await evaluator.evaluate(
            func=pipeline.run,
            config={},
            dataset=dataset.to_core_dataset(),
        )

        # No constraint keys when no constraints defined
        assert "constraints_satisfied" not in result.aggregated_metrics

    @pytest.mark.asyncio
    async def test_evaluator_with_latency_constraints(self):
        """Test HaystackEvaluator with latency constraints."""
        from unittest.mock import MagicMock

        from traigent.integrations.haystack import EvaluationDataset, HaystackEvaluator

        pipeline = MagicMock()
        pipeline.run.return_value = {"llm": {"replies": ["Response"]}}

        dataset = EvaluationDataset.from_dicts(
            [{"input": {"query": "test"}, "expected": "result"}]
        )

        # Latency constraints - very lenient for mock
        constraints = latency_constraint(p95_ms=10000)

        evaluator = HaystackEvaluator(
            pipeline=pipeline,
            haystack_dataset=dataset,
            output_key="llm.replies",
            track_latency=True,
            constraints=constraints,
        )

        result = await evaluator.evaluate(
            func=pipeline.run,
            config={},
            dataset=dataset.to_core_dataset(),
        )

        # Check constraint and latency results
        assert "constraints_satisfied" in result.aggregated_metrics
        assert "latency_p95_ms" in result.aggregated_metrics


class TestFilterByConstraints:
    """Tests for filter_by_constraints function (Story 4.5)."""

    def test_filter_empty_list(self):
        """Test filtering empty list returns empty list."""
        from traigent.integrations.haystack import filter_by_constraints

        result = filter_by_constraints([])
        assert result == []

    def test_filter_all_satisfying(self):
        """Test filtering when all results satisfy constraints."""
        from dataclasses import dataclass, field

        from traigent.integrations.haystack import filter_by_constraints

        @dataclass
        class MockResult:
            aggregated_metrics: dict = field(default_factory=dict)

        results = [
            MockResult(
                aggregated_metrics={"constraints_satisfied": True, "accuracy": 0.9}
            ),
            MockResult(
                aggregated_metrics={"constraints_satisfied": True, "accuracy": 0.8}
            ),
        ]

        filtered = filter_by_constraints(results)
        assert len(filtered) == 2

    def test_filter_none_satisfying(self):
        """Test filtering when no results satisfy constraints."""
        from dataclasses import dataclass, field

        from traigent.integrations.haystack import filter_by_constraints

        @dataclass
        class MockResult:
            aggregated_metrics: dict = field(default_factory=dict)

        results = [
            MockResult(
                aggregated_metrics={"constraints_satisfied": False, "accuracy": 0.9}
            ),
            MockResult(
                aggregated_metrics={"constraints_satisfied": False, "accuracy": 0.8}
            ),
        ]

        filtered = filter_by_constraints(results)
        assert len(filtered) == 0

    def test_filter_mixed_results(self):
        """Test filtering with mixed constraint satisfaction."""
        from dataclasses import dataclass, field

        from traigent.integrations.haystack import filter_by_constraints

        @dataclass
        class MockResult:
            aggregated_metrics: dict = field(default_factory=dict)

        results = [
            MockResult(
                aggregated_metrics={"constraints_satisfied": True, "accuracy": 0.9}
            ),
            MockResult(
                aggregated_metrics={"constraints_satisfied": False, "accuracy": 0.95}
            ),
            MockResult(
                aggregated_metrics={"constraints_satisfied": True, "accuracy": 0.8}
            ),
        ]

        filtered = filter_by_constraints(results)
        assert len(filtered) == 2
        assert all(r.aggregated_metrics["constraints_satisfied"] for r in filtered)

    def test_filter_missing_constraint_key(self):
        """Test filtering when constraint key is missing."""
        from dataclasses import dataclass, field

        from traigent.integrations.haystack import filter_by_constraints

        @dataclass
        class MockResult:
            aggregated_metrics: dict = field(default_factory=dict)

        results = [
            MockResult(
                aggregated_metrics={"accuracy": 0.9}
            ),  # No constraints_satisfied
            MockResult(aggregated_metrics={"constraints_satisfied": True}),
        ]

        filtered = filter_by_constraints(results)
        assert len(filtered) == 1

    def test_filter_with_legacy_metrics(self):
        """Test filtering with legacy .metrics attribute."""
        from traigent.integrations.haystack import filter_by_constraints

        class LegacyResult:
            def __init__(self, metrics):
                self.metrics = metrics

        results = [
            LegacyResult({"constraints_satisfied": True, "accuracy": 0.9}),
            LegacyResult({"constraints_satisfied": False, "accuracy": 0.8}),
        ]

        filtered = filter_by_constraints(results)
        assert len(filtered) == 1

    def test_filter_custom_constraint_key(self):
        """Test filtering with custom constraint key."""
        from dataclasses import dataclass, field

        from traigent.integrations.haystack import filter_by_constraints

        @dataclass
        class MockResult:
            aggregated_metrics: dict = field(default_factory=dict)

        results = [
            MockResult(aggregated_metrics={"custom_satisfied": True}),
            MockResult(aggregated_metrics={"custom_satisfied": False}),
        ]

        filtered = filter_by_constraints(results, constraints_key="custom_satisfied")
        assert len(filtered) == 1


class TestGetBestSatisfying:
    """Tests for get_best_satisfying function (Story 4.5)."""

    def test_best_from_empty_list(self):
        """Test getting best from empty list returns None."""
        from traigent.integrations.haystack import get_best_satisfying

        result = get_best_satisfying([])
        assert result is None

    def test_best_none_satisfying(self):
        """Test getting best when no results satisfy returns None."""
        from dataclasses import dataclass, field

        from traigent.integrations.haystack import get_best_satisfying

        @dataclass
        class MockResult:
            aggregated_metrics: dict = field(default_factory=dict)

        results = [
            MockResult(
                aggregated_metrics={"constraints_satisfied": False, "accuracy": 0.9}
            ),
            MockResult(
                aggregated_metrics={"constraints_satisfied": False, "accuracy": 0.8}
            ),
        ]

        best = get_best_satisfying(results, metric="accuracy")
        assert best is None

    def test_best_maximize(self):
        """Test getting best with maximize=True."""
        from dataclasses import dataclass, field

        from traigent.integrations.haystack import get_best_satisfying

        @dataclass
        class MockResult:
            aggregated_metrics: dict = field(default_factory=dict)

        results = [
            MockResult(
                aggregated_metrics={"constraints_satisfied": True, "accuracy": 0.8}
            ),
            MockResult(
                aggregated_metrics={"constraints_satisfied": True, "accuracy": 0.95}
            ),
            MockResult(
                aggregated_metrics={"constraints_satisfied": True, "accuracy": 0.85}
            ),
        ]

        best = get_best_satisfying(results, metric="accuracy", maximize=True)
        assert best is not None
        assert best.aggregated_metrics["accuracy"] == pytest.approx(0.95)

    def test_best_minimize(self):
        """Test getting best with maximize=False."""
        from dataclasses import dataclass, field

        from traigent.integrations.haystack import get_best_satisfying

        @dataclass
        class MockResult:
            aggregated_metrics: dict = field(default_factory=dict)

        results = [
            MockResult(
                aggregated_metrics={"constraints_satisfied": True, "total_cost": 0.05}
            ),
            MockResult(
                aggregated_metrics={"constraints_satisfied": True, "total_cost": 0.02}
            ),
            MockResult(
                aggregated_metrics={"constraints_satisfied": True, "total_cost": 0.08}
            ),
        ]

        best = get_best_satisfying(results, metric="total_cost", maximize=False)
        assert best is not None
        assert best.aggregated_metrics["total_cost"] == pytest.approx(0.02)

    def test_best_ignores_unsatisfying(self):
        """Test that best ignores results not satisfying constraints."""
        from dataclasses import dataclass, field

        from traigent.integrations.haystack import get_best_satisfying

        @dataclass
        class MockResult:
            aggregated_metrics: dict = field(default_factory=dict)

        results = [
            MockResult(
                aggregated_metrics={"constraints_satisfied": False, "accuracy": 0.99}
            ),
            MockResult(
                aggregated_metrics={"constraints_satisfied": True, "accuracy": 0.85}
            ),
            MockResult(
                aggregated_metrics={"constraints_satisfied": True, "accuracy": 0.80}
            ),
        ]

        best = get_best_satisfying(results, metric="accuracy")
        assert best is not None
        # Should return 0.85, not 0.99 (which doesn't satisfy constraints)
        assert best.aggregated_metrics["accuracy"] == pytest.approx(0.85)
        assert best.aggregated_metrics["constraints_satisfied"] is True

    def test_best_missing_metric(self):
        """Test handling results missing the metric."""
        from dataclasses import dataclass, field

        from traigent.integrations.haystack import get_best_satisfying

        @dataclass
        class MockResult:
            aggregated_metrics: dict = field(default_factory=dict)

        results = [
            MockResult(
                aggregated_metrics={"constraints_satisfied": True}
            ),  # No accuracy
            MockResult(
                aggregated_metrics={"constraints_satisfied": True, "accuracy": 0.85}
            ),
        ]

        best = get_best_satisfying(results, metric="accuracy")
        assert best is not None
        assert best.aggregated_metrics["accuracy"] == pytest.approx(0.85)

    def test_best_single_result(self):
        """Test getting best with single satisfying result."""
        from dataclasses import dataclass, field

        from traigent.integrations.haystack import get_best_satisfying

        @dataclass
        class MockResult:
            aggregated_metrics: dict = field(default_factory=dict)

        results = [
            MockResult(
                aggregated_metrics={"constraints_satisfied": True, "accuracy": 0.9}
            ),
        ]

        best = get_best_satisfying(results, metric="accuracy")
        assert best is not None
        assert best.aggregated_metrics["accuracy"] == pytest.approx(0.9)

    def test_best_with_evaluation_result(self):
        """Test with actual EvaluationResult type."""
        from traigent.evaluators.base import EvaluationResult
        from traigent.integrations.haystack import get_best_satisfying

        results = [
            EvaluationResult(
                config={"temp": 0.7},
                aggregated_metrics={"constraints_satisfied": True, "accuracy": 0.8},
            ),
            EvaluationResult(
                config={"temp": 0.9},
                aggregated_metrics={"constraints_satisfied": True, "accuracy": 0.9},
            ),
            EvaluationResult(
                config={"temp": 0.5},
                aggregated_metrics={"constraints_satisfied": False, "accuracy": 0.95},
            ),
        ]

        best = get_best_satisfying(results, metric="accuracy")
        assert best is not None
        assert best.config == {"temp": 0.9}
        assert best.aggregated_metrics["accuracy"] == pytest.approx(0.9)
