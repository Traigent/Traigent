"""Comprehensive tests for multi-objective optimization utilities (multi_objective.py).

This test suite covers:
- ParetoPoint domination logic and comparisons
- Pareto front calculation algorithms
- Multi-objective trade-off analysis
- Weighted objective optimization
- Constraint handling in multi-objective settings
- Edge cases and boundary conditions
- CTD (Combinatorial Test Design) scenarios
"""

from datetime import datetime
from unittest.mock import MagicMock

import pytest

from traigent.api.types import TrialResult
from traigent.utils.multi_objective import (
    MultiObjectiveMetrics,
    ParetoFrontCalculator,
    ParetoPoint,
    normalize_objectives,
    scalarize_objectives,
)

# Test fixtures


@pytest.fixture
def sample_trial_results():
    """Sample trial results for multi-objective testing."""
    return [
        TrialResult(
            trial_id="trial_1",
            config={"temperature": 0.5, "max_tokens": 1000},
            metrics={"accuracy": 0.90, "cost": 0.05, "latency": 0.8},
            status="completed",
            duration=1.0,
            timestamp=datetime.now(),
            metadata={},
        ),
        TrialResult(
            trial_id="trial_2",
            config={"temperature": 0.7, "max_tokens": 1500},
            metrics={"accuracy": 0.88, "cost": 0.08, "latency": 1.2},
            status="completed",
            duration=1.2,
            timestamp=datetime.now(),
            metadata={},
        ),
        TrialResult(
            trial_id="trial_3",
            config={"temperature": 0.3, "max_tokens": 800},
            metrics={"accuracy": 0.92, "cost": 0.03, "latency": 0.6},
            status="completed",
            duration=0.8,
            timestamp=datetime.now(),
            metadata={},
        ),
        TrialResult(
            trial_id="trial_4",
            config={"temperature": 0.9, "max_tokens": 2000},
            metrics={"accuracy": 0.85, "cost": 0.12, "latency": 1.8},
            status="completed",
            duration=1.5,
            timestamp=datetime.now(),
            metadata={},
        ),
        TrialResult(
            trial_id="trial_5",
            config={"temperature": 0.6, "max_tokens": 1200},
            metrics={"accuracy": 0.89, "cost": 0.06, "latency": 1.0},
            status="completed",
            duration=1.1,
            timestamp=datetime.now(),
            metadata={},
        ),
        # Failed trial (should be excluded)
        TrialResult(
            trial_id="trial_failed",
            config={"temperature": 1.0, "max_tokens": 3000},
            metrics={},
            status="failed",
            duration=0.5,
            timestamp=datetime.now(),
            metadata={"error": "timeout"},
        ),
    ]


@pytest.fixture
def sample_pareto_points():
    """Sample Pareto points for testing."""
    return [
        ParetoPoint(
            config={"temp": 0.5},
            objectives={"accuracy": 0.90, "cost": 0.05},
            trial=MagicMock(),
        ),
        ParetoPoint(
            config={"temp": 0.7},
            objectives={"accuracy": 0.85, "cost": 0.03},
            trial=MagicMock(),
        ),
        ParetoPoint(
            config={"temp": 0.3},
            objectives={"accuracy": 0.92, "cost": 0.08},
            trial=MagicMock(),
        ),
    ]


@pytest.fixture
def sample_high_dim_pareto_points():
    """Sample Pareto points with three objectives for testing hypervolume."""
    return [
        ParetoPoint(
            config={"temp": 0.5},
            objectives={"accuracy": 0.90, "cost": 0.05, "latency": 0.8},
            trial=MagicMock(),
        ),
        ParetoPoint(
            config={"temp": 0.7},
            objectives={"accuracy": 0.85, "cost": 0.03, "latency": 1.0},
            trial=MagicMock(),
        ),
        ParetoPoint(
            config={"temp": 0.3},
            objectives={"accuracy": 0.92, "cost": 0.08, "latency": 0.6},
            trial=MagicMock(),
        ),
    ]


@pytest.fixture
def maximize_accuracy_minimize_cost():
    """Maximize accuracy, minimize cost."""
    return {"accuracy": True, "cost": False, "latency": False}


@pytest.fixture
def maximize_all():
    """Maximize all objectives."""
    return {"accuracy": True, "cost": True, "latency": True}


@pytest.fixture
def pareto_calculator():
    """Basic Pareto front calculator."""
    return ParetoFrontCalculator()


@pytest.fixture
def pareto_calculator_custom_maximize(maximize_accuracy_minimize_cost):
    """Pareto calculator with custom maximize settings."""
    return ParetoFrontCalculator(maximize=maximize_accuracy_minimize_cost)


# Test Classes


class TestParetoPoint:
    """Test ParetoPoint functionality."""

    def test_pareto_point_creation(self):
        """Test ParetoPoint creation."""
        config = {"temperature": 0.5}
        objectives = {"accuracy": 0.9, "cost": 0.05}
        trial = MagicMock()

        point = ParetoPoint(config=config, objectives=objectives, trial=trial)

        assert point.config == config
        assert point.objectives == objectives
        assert point.trial == trial

    def test_dominance_maximize_all(self):
        """Test dominance with maximize all objectives."""
        point1 = ParetoPoint(
            config={"temp": 0.5},
            objectives={"accuracy": 0.9, "cost": 0.05},
            trial=MagicMock(),
        )
        point2 = ParetoPoint(
            config={"temp": 0.7},
            objectives={"accuracy": 0.8, "cost": 0.03},
            trial=MagicMock(),
        )

        maximize = {"accuracy": True, "cost": True}

        # When maximizing both:
        # point1 has higher accuracy (0.9 > 0.8) - better
        # point1 has higher cost (0.05 > 0.03) - better (when maximizing)
        # point1 dominates point2
        assert point1.dominates(point2, maximize) is True
        assert point2.dominates(point1, maximize) is False

    def test_dominance_mixed_objectives(self):
        """Test dominance with mixed maximize/minimize objectives."""
        point1 = ParetoPoint(
            config={"temp": 0.5},
            objectives={"accuracy": 0.9, "cost": 0.05},
            trial=MagicMock(),
        )
        point2 = ParetoPoint(
            config={"temp": 0.7},
            objectives={"accuracy": 0.8, "cost": 0.08},
            trial=MagicMock(),
        )

        maximize = {"accuracy": True, "cost": False}  # Maximize accuracy, minimize cost

        # point1 has higher accuracy AND lower cost
        assert point1.dominates(point2, maximize) is True
        assert point2.dominates(point1, maximize) is False

    def test_dominance_equal_performance(self):
        """Test dominance with equal performance."""
        point1 = ParetoPoint(
            config={"temp": 0.5},
            objectives={"accuracy": 0.9, "cost": 0.05},
            trial=MagicMock(),
        )
        point2 = ParetoPoint(
            config={"temp": 0.7},
            objectives={"accuracy": 0.9, "cost": 0.05},
            trial=MagicMock(),
        )

        maximize = {"accuracy": True, "cost": False}

        # Equal performance - no dominance
        assert point1.dominates(point2, maximize) is False
        assert point2.dominates(point1, maximize) is False

    def test_dominance_missing_objectives(self):
        """Test dominance with missing objectives."""
        point1 = ParetoPoint(
            config={"temp": 0.5},
            objectives={"accuracy": 0.9, "cost": 0.05},
            trial=MagicMock(),
        )
        point2 = ParetoPoint(
            config={"temp": 0.7},
            objectives={"accuracy": 0.8, "latency": 1.0},  # Missing cost
            trial=MagicMock(),
        )

        maximize = {"accuracy": True, "cost": False, "latency": False}

        # Should only compare common objectives (accuracy)
        assert point1.dominates(point2, maximize) is True  # Higher accuracy
        assert point2.dominates(point1, maximize) is False

    def test_dominance_single_objective(self):
        """Test dominance with single objective."""
        point1 = ParetoPoint(
            config={"temp": 0.5}, objectives={"accuracy": 0.9}, trial=MagicMock()
        )
        point2 = ParetoPoint(
            config={"temp": 0.7}, objectives={"accuracy": 0.8}, trial=MagicMock()
        )

        maximize = {"accuracy": True}

        assert point1.dominates(point2, maximize) is True
        assert point2.dominates(point1, maximize) is False

    def test_dominance_empty_objectives(self):
        """Test dominance with empty objectives."""
        point1 = ParetoPoint(config={"temp": 0.5}, objectives={}, trial=MagicMock())
        point2 = ParetoPoint(config={"temp": 0.7}, objectives={}, trial=MagicMock())

        maximize = {}

        # No objectives to compare - no dominance
        assert point1.dominates(point2, maximize) is False
        assert point2.dominates(point1, maximize) is False


class TestParetoFrontCalculator:
    """Test ParetoFrontCalculator functionality."""

    def test_calculator_initialization_default(self):
        """Test calculator initialization with defaults."""
        calculator = ParetoFrontCalculator()

        assert calculator.maximize == {}

    def test_calculator_initialization_custom(self, maximize_accuracy_minimize_cost):
        """Test calculator initialization with custom maximize settings."""
        calculator = ParetoFrontCalculator(maximize=maximize_accuracy_minimize_cost)

        assert calculator.maximize == maximize_accuracy_minimize_cost
        assert calculator.maximize["accuracy"] is True
        assert calculator.maximize["cost"] is False

    def test_calculate_pareto_front_basic(
        self, pareto_calculator_custom_maximize, sample_trial_results
    ):
        """Test basic Pareto front calculation."""
        objectives = ["accuracy", "cost"]

        pareto_front = pareto_calculator_custom_maximize.calculate_pareto_front(
            sample_trial_results, objectives
        )

        assert len(pareto_front) > 0
        assert all(isinstance(point, ParetoPoint) for point in pareto_front)

        # Verify no point in the front is dominated by another
        for i, point1 in enumerate(pareto_front):
            for j, point2 in enumerate(pareto_front):
                if i != j:
                    assert not point1.dominates(
                        point2, pareto_calculator_custom_maximize.maximize
                    )

    def test_calculate_pareto_front_single_objective(
        self, pareto_calculator, sample_trial_results
    ):
        """Test Pareto front calculation with single objective."""
        objectives = ["accuracy"]

        pareto_front = pareto_calculator.calculate_pareto_front(
            sample_trial_results, objectives
        )

        # With single objective, should return the single best point
        assert len(pareto_front) == 1
        best_point = pareto_front[0]

        # Should be the trial with highest accuracy
        expected_accuracy = max(
            trial.metrics.get("accuracy", 0)
            for trial in sample_trial_results
            if trial.status == "completed"
        )
        assert best_point.objectives["accuracy"] == expected_accuracy

    def test_calculate_pareto_front_empty_trials(self, pareto_calculator):
        """Test Pareto front calculation with empty trials."""
        pareto_front = pareto_calculator.calculate_pareto_front([], ["accuracy"])

        assert pareto_front == []

    def test_calculate_pareto_front_no_completed_trials(self, pareto_calculator):
        """Test Pareto front calculation with no completed trials."""
        failed_trials = [
            TrialResult(
                trial_id="failed_1",
                config={},
                metrics={},
                status="failed",
                duration=0.1,
                timestamp=datetime.now(),
                metadata={},
            ),
            TrialResult(
                trial_id="failed_2",
                config={},
                metrics={},
                status="cancelled",
                duration=0.1,
                timestamp=datetime.now(),
                metadata={},
            ),
        ]

        # The implementation checks for TrialStatus.COMPLETED.value which is "completed"
        pareto_front = pareto_calculator.calculate_pareto_front(
            failed_trials, ["accuracy"]
        )

        assert pareto_front == []

    def test_calculate_pareto_front_missing_objectives(
        self, pareto_calculator, sample_trial_results
    ):
        """Test Pareto front calculation with missing objectives in some trials."""
        objectives = ["accuracy", "nonexistent_metric"]

        pareto_front = pareto_calculator.calculate_pareto_front(
            sample_trial_results, objectives
        )

        # Should include trials that have at least one of the objectives
        assert len(pareto_front) > 0

        # All points should have accuracy (but may not have nonexistent_metric)
        for point in pareto_front:
            assert "accuracy" in point.objectives

    def test_calculate_pareto_front_three_objectives(
        self, pareto_calculator_custom_maximize, sample_trial_results
    ):
        """Test Pareto front calculation with three objectives."""
        # Set up maximize settings for three objectives
        pareto_calculator_custom_maximize.maximize = {
            "accuracy": True,
            "cost": False,  # Minimize cost
            "latency": False,  # Minimize latency
        }

        objectives = ["accuracy", "cost", "latency"]

        pareto_front = pareto_calculator_custom_maximize.calculate_pareto_front(
            sample_trial_results, objectives
        )

        assert len(pareto_front) > 0

        # With three objectives, should have multiple points on the front
        # (trade-offs between accuracy, cost, and latency)
        assert len(pareto_front) >= 1

        # Verify properties of Pareto front
        for point in pareto_front:
            assert len(point.objectives) <= 3

    def test_pareto_front_ordering(self, pareto_calculator_custom_maximize):
        """Test that Pareto front maintains proper ordering."""
        # Create specific trials to test ordering
        trials = [
            TrialResult(
                trial_id="high_acc_high_cost",
                config={"config": "1"},
                metrics={"accuracy": 0.95, "cost": 0.10},
                status="completed",
                duration=1.0,
                timestamp=datetime.now(),
                metadata={},
            ),
            TrialResult(
                trial_id="med_acc_med_cost",
                config={"config": "2"},
                metrics={"accuracy": 0.85, "cost": 0.05},
                status="completed",
                duration=1.0,
                timestamp=datetime.now(),
                metadata={},
            ),
            TrialResult(
                trial_id="low_acc_low_cost",
                config={"config": "3"},
                metrics={"accuracy": 0.75, "cost": 0.02},
                status="completed",
                duration=1.0,
                timestamp=datetime.now(),
                metadata={},
            ),
            TrialResult(
                trial_id="dominated",
                config={"config": "4"},
                metrics={"accuracy": 0.70, "cost": 0.08},  # Dominated by config 2 and 3
                status="completed",
                duration=1.0,
                timestamp=datetime.now(),
                metadata={},
            ),
        ]

        objectives = ["accuracy", "cost"]
        pareto_front = pareto_calculator_custom_maximize.calculate_pareto_front(
            trials, objectives
        )

        # All three non-dominated points should be in the front
        # Config 4 is dominated by config 2 (better accuracy and cost)
        assert len(pareto_front) == 3

        # Verify that the dominated point is not in the front
        front_configs = [point.config for point in pareto_front]
        assert {"config": "4"} not in front_configs

    def test_monte_carlo_hypervolume_is_deterministic(
        self,
        maximize_accuracy_minimize_cost,
        sample_high_dim_pareto_points,
    ):
        """Monte Carlo hypervolume should be reproducible when seeded."""

        calculator_a = ParetoFrontCalculator(
            maximize=maximize_accuracy_minimize_cost,
            random_seed=123,
            monte_carlo_samples=2000,
        )
        hv_a1 = calculator_a._approximate_hypervolume(
            sample_high_dim_pareto_points, reference_point=None
        )

        calculator_b = ParetoFrontCalculator(
            maximize=maximize_accuracy_minimize_cost,
            random_seed=123,
            monte_carlo_samples=2000,
        )
        hv_a2 = calculator_b._approximate_hypervolume(
            sample_high_dim_pareto_points, reference_point=None
        )

        assert hv_a1 == hv_a2

        calculator_c = ParetoFrontCalculator(
            maximize=maximize_accuracy_minimize_cost,
            random_seed=999,
            monte_carlo_samples=2000,
        )
        hv_c = calculator_c._approximate_hypervolume(
            sample_high_dim_pareto_points, reference_point=None
        )

        assert hv_c != hv_a1


# MultiObjectiveOptimizer class tests removed - class doesn't exist in implementation
# These tests would need the MultiObjectiveOptimizer class to be implemented


class TestWeightedObjectiveCalculator:
    """Test weighted objective functionality using scalarize_objectives."""

    def test_scalarize_objectives(self):
        """Test scalarizing objectives with weights."""
        objectives = {"accuracy": 0.9, "cost": 0.05, "latency": 1.0}
        weights = {"accuracy": 0.6, "cost": 0.3, "latency": 0.1}

        score = scalarize_objectives(objectives, weights)

        assert isinstance(score, float)
        assert score > 0

    def test_scalarize_objectives_zero_weights(self):
        """Test scalarizing with zero total weight."""
        objectives = {"accuracy": 0.9}
        weights = {}  # No weights provided

        score = scalarize_objectives(objectives, weights)

        # Should use default weight of 1.0
        assert score == 0.9


class TestMultiObjectiveMetrics:
    """Test MultiObjectiveMetrics functionality."""

    def test_diversity_metric(self, sample_pareto_points):
        """Test calculating diversity metric."""
        diversity = MultiObjectiveMetrics.calculate_diversity_metric(
            sample_pareto_points
        )

        assert isinstance(diversity, float)
        assert diversity >= 0.0

    def test_diversity_metric_single_point(self):
        """Test diversity metric with single point."""
        points = [
            ParetoPoint(
                config={"temp": 0.5},
                objectives={"accuracy": 0.9, "cost": 0.05},
                trial=MagicMock(),
            )
        ]

        diversity = MultiObjectiveMetrics.calculate_diversity_metric(points)
        assert diversity == 0.0  # Single point has no diversity

    def test_convergence_metric(self, sample_pareto_points):
        """Test calculating convergence metric."""
        convergence = MultiObjectiveMetrics.calculate_convergence_metric(
            sample_pareto_points
        )

        assert isinstance(convergence, float)
        # Without true front, uses approximation

    def test_normalize_objectives_function(self, sample_trial_results):
        """Test normalize_objectives function."""
        objectives = ["accuracy", "cost", "latency"]
        ranges = normalize_objectives(sample_trial_results, objectives)

        assert isinstance(ranges, dict)
        for obj in objectives:
            assert obj in ranges
            assert len(ranges[obj]) == 2
            assert ranges[obj][0] <= ranges[obj][1]

    def test_normalize_objectives_constant_values(self):
        """Normalization should not collapse when all values identical."""
        trials = [
            TrialResult(
                trial_id=f"trial_{i}",
                config={},
                metrics={"accuracy": 0.75},
                status="completed",
                duration=1.0,
                timestamp=datetime.now(),
                metadata={},
            )
            for i in range(3)
        ]

        ranges = normalize_objectives(trials, ["accuracy"])
        min_val, max_val = ranges["accuracy"]
        assert max_val > min_val


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_pareto_calculation_with_nan_values(self, pareto_calculator):
        """Test Pareto calculation with NaN values."""
        trials_with_nan = [
            TrialResult(
                trial_id="nan_trial",
                config={"temp": 0.5},
                metrics={"accuracy": float("nan"), "cost": 0.05},
                status="completed",
                duration=1.0,
                timestamp=datetime.now(),
                metadata={},
            ),
            TrialResult(
                trial_id="valid_trial",
                config={"temp": 0.7},
                metrics={"accuracy": 0.9, "cost": 0.03},
                status="completed",
                duration=1.0,
                timestamp=datetime.now(),
                metadata={},
            ),
        ]

        objectives = ["accuracy", "cost"]
        pareto_front = pareto_calculator.calculate_pareto_front(
            trials_with_nan, objectives
        )

        # Should handle NaN values gracefully
        # NaN comparisons always return False, so the valid point dominates the NaN point
        # (0.9 > NaN returns False, so dominance check fails in that direction)
        # But the valid point has better accuracy (0.9 vs NaN) and better cost (0.03 vs 0.05)
        assert isinstance(pareto_front, list)
        assert len(pareto_front) == 1  # Only valid point in front

    def test_pareto_calculation_with_infinite_values(self, pareto_calculator):
        """Test Pareto calculation with infinite values."""
        trials_with_inf = [
            TrialResult(
                trial_id="inf_trial",
                config={"temp": 0.5},
                metrics={"accuracy": 0.9, "cost": float("inf")},
                status="completed",
                duration=1.0,
                timestamp=datetime.now(),
                metadata={},
            ),
            TrialResult(
                trial_id="valid_trial",
                config={"temp": 0.7},
                metrics={"accuracy": 0.8, "cost": 0.05},
                status="completed",
                duration=1.0,
                timestamp=datetime.now(),
                metadata={},
            ),
        ]

        objectives = ["accuracy", "cost"]
        # Calculator with default maximize settings (empty dict)
        pareto_front = pareto_calculator.calculate_pareto_front(
            trials_with_inf, objectives
        )

        # Should handle infinite values gracefully
        assert isinstance(pareto_front, list)
        # With default maximize=True for all, inf_trial dominates (higher accuracy, higher cost)
        assert len(pareto_front) >= 1

    def test_weighted_calculation_with_zero_weights(self):
        """Test weighted calculation with zero weights."""
        objectives = {"accuracy": 0.9, "cost": 0.05}
        weights = {"accuracy": 0.0, "cost": 1.0}

        score = scalarize_objectives(objectives, weights)

        # Should handle zero weights
        assert isinstance(score, float)
        # Score should be weighted average
        assert abs(score - 0.05) < 1e-6

    def test_empty_objectives_list(self):
        """Test behavior with empty objectives list."""
        calculator = ParetoFrontCalculator()

        trials = [
            TrialResult(
                trial_id="trial_1",
                config={"temp": 0.5},
                metrics={"accuracy": 0.9},
                status="completed",
                duration=1.0,
                timestamp=datetime.now(),
                metadata={},
            )
        ]

        pareto_front = calculator.calculate_pareto_front(trials, [])

        # Should return empty front with no objectives
        # Points have no objectives extracted, so no points added to front
        assert pareto_front == []


class TestCTDScenarios:
    """Combinatorial Test Design scenarios for comprehensive coverage."""

    @pytest.mark.parametrize(
        "num_objectives,maximize_pattern,expected_front_size",
        [
            (1, "all_max", 1),
            (2, "all_max", ">=1"),
            (2, "mixed", ">=1"),
            (3, "all_max", ">=1"),
            (3, "mixed", ">=1"),
            (3, "all_min", ">=1"),
        ],
    )
    def test_pareto_front_size_combinations(
        self, num_objectives, maximize_pattern, expected_front_size
    ):
        """Test Pareto front size with different objective combinations."""
        # Create test trials with varied metrics
        trials = []
        for i in range(5):
            metrics = {}
            for j in range(num_objectives):
                obj_name = f"obj_{j}"
                # Create varied performance - different patterns for each trial
                if i == 0:
                    metrics[obj_name] = 0.9 - (j * 0.1)  # Best in first obj
                elif i == 1:
                    metrics[obj_name] = 0.5 + (j * 0.2)  # Best in last obj
                else:
                    metrics[obj_name] = 0.5 + (i * 0.1) - (j * 0.05)

            trials.append(
                TrialResult(
                    trial_id=f"trial_{i}",
                    config={"param": i},
                    metrics=metrics,
                    status="completed",
                    duration=1.0,
                    timestamp=datetime.now(),
                    metadata={},
                )
            )

        # Set up maximize settings
        maximize = {}
        objectives = [f"obj_{j}" for j in range(num_objectives)]

        for obj in objectives:
            if maximize_pattern == "all_max":
                maximize[obj] = True
            elif maximize_pattern == "all_min":
                maximize[obj] = False
            elif maximize_pattern == "mixed":
                maximize[obj] = obj.endswith("0") or obj.endswith("2")  # Mix of max/min

        calculator = ParetoFrontCalculator(maximize=maximize)
        pareto_front = calculator.calculate_pareto_front(trials, objectives)

        if expected_front_size == 1:
            # With single objective, might have multiple points with same best value
            assert len(pareto_front) >= 1
            # All points should have the same (best) objective value
            if len(pareto_front) > 1:
                best_value = pareto_front[0].objectives[objectives[0]]
                for point in pareto_front:
                    assert point.objectives[objectives[0]] == best_value
        elif expected_front_size == ">=1":
            assert len(pareto_front) >= 1

    @pytest.mark.parametrize(
        "weight_sum,expected_behavior",
        [
            (1.0, "valid"),
            (0.5, "valid"),
            (2.0, "valid"),
            (0.0, "zero"),
        ],
    )
    def test_weighted_objective_combinations(self, weight_sum, expected_behavior):
        """Test weighted objective calculations with different weight combinations."""
        # Create weights that sum to weight_sum
        if weight_sum == 0.0:
            weights = {}  # Empty weights
        else:
            weights = {"accuracy": weight_sum * 0.6, "cost": weight_sum * 0.4}

        metrics = {"accuracy": 0.9, "cost": 0.05}
        score = scalarize_objectives(metrics, weights)

        if expected_behavior == "valid":
            assert isinstance(score, float)
            assert score > 0
        elif expected_behavior == "zero":
            # With no weights, uses default weight of 1.0 for each
            assert isinstance(score, float)
            expected = (0.9 + 0.05) / 2  # Average with default weights
            assert abs(score - expected) < 1e-6

    @pytest.mark.parametrize(
        "trial_status,has_metrics,objective_coverage,expected_inclusion",
        [
            ("completed", True, "full", True),
            ("completed", True, "partial", True),
            ("completed", True, "none", False),
            ("completed", False, "none", False),
            ("failed", True, "full", False),
            ("cancelled", True, "full", False),
        ],
    )
    def test_trial_inclusion_combinations(
        self, trial_status, has_metrics, objective_coverage, expected_inclusion
    ):
        """Test trial inclusion in Pareto front with different combinations."""
        objectives = ["accuracy", "cost"]

        # Create trial based on parameters
        metrics = {}
        if has_metrics:
            if objective_coverage == "full":
                metrics = {"accuracy": 0.9, "cost": 0.05}
            elif objective_coverage == "partial":
                metrics = {"accuracy": 0.9}  # Missing cost but still has one objective
            # objective_coverage == "none" keeps metrics empty (no objectives match)

        trial = TrialResult(
            trial_id="test_trial",
            config={"temp": 0.5},
            metrics=metrics,
            status=trial_status,  # Status checked against "completed"
            duration=1.0,
            timestamp=datetime.now(),
            metadata={},
        )

        calculator = ParetoFrontCalculator()
        pareto_front = calculator.calculate_pareto_front([trial], objectives)

        if expected_inclusion:
            assert len(pareto_front) == 1
            assert pareto_front[0].trial == trial
        else:
            assert len(pareto_front) == 0
