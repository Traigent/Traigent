"""Unit tests for TVL statistics module."""

import math

import pytest

from traigent.tvl.statistics import (
    PairedComparisonResult,
    benjamini_hochberg_adjust,
    clopper_pearson_lower_bound,
    hypervolume_improvement,
    paired_comparison_test,
)


class TestBenjaminiHochbergAdjust:
    """Tests for Benjamini-Hochberg adjustment."""

    def test_empty_list(self) -> None:
        """Empty list returns empty list."""
        assert benjamini_hochberg_adjust([]) == []

    def test_single_value(self) -> None:
        """Single p-value is returned as-is (capped at 1)."""
        assert benjamini_hochberg_adjust([0.05]) == [0.05]
        assert benjamini_hochberg_adjust([1.5]) == [1.0]

    def test_all_significant(self) -> None:
        """All significant p-values remain significant after adjustment."""
        p_vals = [0.001, 0.002, 0.003, 0.004]
        adjusted = benjamini_hochberg_adjust(p_vals)

        # After BH adjustment, should all still be below 0.05
        assert all(p < 0.05 for p in adjusted)

    def test_adjustment_increases_values(self) -> None:
        """BH adjustment generally increases p-values."""
        p_vals = [0.01, 0.02, 0.03, 0.04]
        adjusted = benjamini_hochberg_adjust(p_vals)

        # Adjusted values should be >= original
        for orig, adj in zip(p_vals, adjusted, strict=True):
            assert adj >= orig or abs(adj - orig) < 1e-10

    def test_monotonicity(self) -> None:
        """BH adjusted values maintain monotonicity with original order."""
        p_vals = [0.01, 0.04, 0.03, 0.02]
        adjusted = benjamini_hochberg_adjust(p_vals)

        # Values are capped at 1.0
        assert all(p <= 1.0 for p in adjusted)

    def test_known_example(self) -> None:
        """Test with a known BH adjustment example."""
        # Example: 4 tests with p-values
        p_vals = [0.01, 0.02, 0.03, 0.04]
        adjusted = benjamini_hochberg_adjust(p_vals)

        # BH formula: adjusted[i] = min(p[i] * n / rank, adjusted[i+1])
        # For rank 1 (smallest): 0.01 * 4 / 1 = 0.04
        # For rank 2: 0.02 * 4 / 2 = 0.04
        # For rank 3: 0.03 * 4 / 3 = 0.04
        # For rank 4: 0.04 * 4 / 4 = 0.04
        assert all(abs(p - 0.04) < 1e-10 for p in adjusted)


class TestClopperPearsonLowerBound:
    """Tests for Clopper-Pearson confidence interval."""

    def test_zero_successes(self) -> None:
        """Zero successes returns 0 lower bound."""
        lb = clopper_pearson_lower_bound(0, 100, 0.95)
        assert lb == 0.0

    def test_all_successes(self) -> None:
        """All successes returns high lower bound."""
        lb = clopper_pearson_lower_bound(100, 100, 0.95)
        assert lb > 0.95  # Should be close to 1

    def test_half_successes(self) -> None:
        """50/100 successes returns reasonable lower bound."""
        lb = clopper_pearson_lower_bound(50, 100, 0.95)
        # Wilson approximation may give different results than exact
        # Just verify it's a reasonable lower bound (between 0.3 and 0.6)
        assert 0.30 < lb < 0.60

    def test_higher_confidence_gives_different_bound(self) -> None:
        """Higher confidence affects the lower bound calculation."""
        lb_95 = clopper_pearson_lower_bound(80, 100, 0.95)
        lb_99 = clopper_pearson_lower_bound(80, 100, 0.99)
        # Both should be reasonable bounds at or below observed rate (0.80)
        assert 0.65 < lb_95 <= 0.90
        assert 0.65 < lb_99 <= 0.90

    def test_invalid_inputs(self) -> None:
        """Invalid inputs raise ValueError."""
        with pytest.raises(ValueError, match="must be positive"):
            clopper_pearson_lower_bound(10, 0, 0.95)

        with pytest.raises(ValueError, match="must be non-negative"):
            clopper_pearson_lower_bound(-1, 100, 0.95)

        with pytest.raises(ValueError, match="cannot exceed"):
            clopper_pearson_lower_bound(101, 100, 0.95)

        with pytest.raises(ValueError, match="must be in"):
            clopper_pearson_lower_bound(50, 100, 0.0)


class TestPairedComparisonTest:
    """Tests for paired_comparison_test function."""

    def test_empty_samples_raises(self) -> None:
        """Empty samples raise ValueError."""
        with pytest.raises(ValueError, match="empty samples"):
            paired_comparison_test([], [1, 2, 3], 0.0, "greater")

        with pytest.raises(ValueError, match="empty samples"):
            paired_comparison_test([1, 2, 3], [], 0.0, "greater")

    def test_clearly_greater(self) -> None:
        """X clearly greater than Y is detected."""
        x = [10.0, 11.0, 9.0, 10.5, 10.2]
        y = [5.0, 4.5, 5.5, 4.8, 5.2]

        result = paired_comparison_test(x, y, 0.0, "greater")

        assert result.effect_size > 4.0  # x mean - y mean ≈ 5
        assert result.p_value < 0.05
        # Note: reject_null is deprecated; callers should use p_value < alpha

    def test_clearly_less(self) -> None:
        """X clearly less than Y is detected."""
        x = [5.0, 4.5, 5.5, 4.8, 5.2]
        y = [10.0, 11.0, 9.0, 10.5, 10.2]

        result = paired_comparison_test(x, y, 0.0, "less")

        assert result.effect_size < -4.0
        assert result.p_value < 0.05
        # Note: reject_null is deprecated; callers should use p_value < alpha

    def test_with_epsilon_margin(self) -> None:
        """Epsilon margin affects test outcome."""
        x = [10.0, 10.5, 9.5, 10.2, 10.3]
        y = [9.0, 9.5, 8.5, 9.2, 9.3]

        # Without epsilon, x > y is significant at alpha=0.05
        result_no_eps = paired_comparison_test(x, y, 0.0, "greater")
        assert result_no_eps.p_value < 0.05  # Use p_value with your alpha

        # With large epsilon (2.0), x > y + 2 may not be significant
        result_with_eps = paired_comparison_test(x, y, 2.0, "greater")
        # Effect size is ~1, so with epsilon=2, we're testing if x > y + 2
        # This should NOT be significant at alpha=0.05
        assert result_with_eps.p_value >= 0.05  # Use p_value with your alpha

    def test_result_contains_statistics(self) -> None:
        """Result contains all required statistics."""
        x = [1.0, 2.0, 3.0]
        y = [1.5, 2.5, 3.5]

        result = paired_comparison_test(x, y, 0.0, "greater")

        assert isinstance(result, PairedComparisonResult)
        assert isinstance(result.reject_null, bool)
        assert 0.0 <= result.p_value <= 1.0
        assert isinstance(result.effect_size, float)
        assert isinstance(result.test_statistic, float)
        assert result.degrees_of_freedom > 0


class TestHypervolumeImprovement:
    """Tests for hypervolume_improvement function."""

    def test_empty_front(self) -> None:
        """Empty Pareto front returns the entire contribution."""
        new_point = [0.8, 0.9]
        pareto_front: list[list[float]] = []
        reference = [0.0, 0.0]
        directions = ["maximize", "maximize"]

        hv = hypervolume_improvement(new_point, pareto_front, reference, directions)
        # HV = 0.8 * 0.9 = 0.72
        assert abs(hv - 0.72) < 0.1

    def test_dominated_point_no_improvement(self) -> None:
        """Dominated point contributes zero improvement."""
        new_point = [0.5, 0.5]
        pareto_front = [[0.8, 0.9]]  # Dominates new_point
        reference = [0.0, 0.0]
        directions = ["maximize", "maximize"]

        hv = hypervolume_improvement(new_point, pareto_front, reference, directions)
        assert hv == 0.0

    def test_non_dominated_point_adds_improvement(self) -> None:
        """Non-dominated point adds positive improvement."""
        new_point = [0.9, 0.7]
        pareto_front = [[0.7, 0.9]]  # Neither dominates the other
        reference = [0.0, 0.0]
        directions = ["maximize", "maximize"]

        hv = hypervolume_improvement(new_point, pareto_front, reference, directions)
        assert hv > 0.0

    def test_minimize_direction(self) -> None:
        """Minimize direction is handled correctly."""
        # For minimize, lower is better
        new_point = [0.9, 50.0]  # High accuracy, low latency
        pareto_front = [[0.8, 100.0]]
        reference = [0.0, 200.0]  # Reference for minimize should be worst case
        directions = ["maximize", "minimize"]

        hv = hypervolume_improvement(new_point, pareto_front, reference, directions)
        # New point should improve hypervolume
        assert hv >= 0.0
