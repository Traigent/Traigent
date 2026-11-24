"""Unit tests for TVL banded objectives and TOST."""

import pytest

from traigent.tvl.models import BandTarget
from traigent.tvl.objectives import (
    BandedObjectiveSpec,
    TOSTResult,
    band_deviation,
    compare_banded_objectives,
    compare_banded_with_tost,
    is_in_band,
    tost_equivalence_test,
)


class TestIsInBand:
    """Tests for is_in_band function."""

    def test_value_inside_band(self) -> None:
        """Value inside band returns True."""
        band = BandTarget(low=0.8, high=0.95)
        assert is_in_band(0.85, band) is True
        assert is_in_band(0.9, band) is True

    def test_value_at_bounds(self) -> None:
        """Values at bounds are inside the band."""
        band = BandTarget(low=0.8, high=0.95)
        assert is_in_band(0.8, band) is True
        assert is_in_band(0.95, band) is True

    def test_value_outside_band(self) -> None:
        """Value outside band returns False."""
        band = BandTarget(low=0.8, high=0.95)
        assert is_in_band(0.79, band) is False
        assert is_in_band(0.96, band) is False


class TestBandDeviation:
    """Tests for band_deviation function."""

    def test_inside_band_zero_deviation(self) -> None:
        """Values inside band have zero deviation."""
        band = BandTarget(low=0.8, high=0.95)
        assert band_deviation(0.85, band) == 0.0
        assert band_deviation(0.8, band) == 0.0
        assert band_deviation(0.95, band) == 0.0

    def test_below_band(self) -> None:
        """Values below band have positive deviation."""
        band = BandTarget(low=0.8, high=0.95)
        assert abs(band_deviation(0.75, band) - 0.05) < 1e-10
        assert abs(band_deviation(0.7, band) - 0.1) < 1e-10

    def test_above_band(self) -> None:
        """Values above band have positive deviation."""
        band = BandTarget(low=0.8, high=0.95)
        assert abs(band_deviation(1.0, band) - 0.05) < 1e-10
        assert abs(band_deviation(1.05, band) - 0.1) < 1e-10


class TestTOSTEquivalenceTest:
    """Tests for tost_equivalence_test function."""

    def test_empty_samples_raises(self) -> None:
        """TOST raises error on empty samples."""
        band = BandTarget(low=0.8, high=0.95)
        with pytest.raises(ValueError, match="empty samples"):
            tost_equivalence_test([], band)

    def test_single_sample_inside_band(self) -> None:
        """Single sample inside band shows equivalent."""
        band = BandTarget(low=0.8, high=0.95)
        result = tost_equivalence_test([0.87], band)
        assert result.is_equivalent is True
        assert result.sample_size == 1

    def test_single_sample_outside_band(self) -> None:
        """Single sample outside band shows not equivalent."""
        band = BandTarget(low=0.8, high=0.95)
        result = tost_equivalence_test([0.75], band)
        assert result.is_equivalent is False

    def test_n1_edge_case_at_lower_bound(self) -> None:
        """n=1 edge case: sample exactly at lower bound."""
        band = BandTarget(low=0.8, high=0.95)
        result = tost_equivalence_test([0.8], band)
        # At the bound, considered inside
        assert result.is_equivalent is True
        assert result.sample_size == 1
        assert result.sample_std == 0.0  # No variance with n=1
        assert result.confidence_interval == (0.8, 0.8)  # Degenerate CI

    def test_n1_edge_case_at_upper_bound(self) -> None:
        """n=1 edge case: sample exactly at upper bound."""
        band = BandTarget(low=0.8, high=0.95)
        result = tost_equivalence_test([0.95], band)
        assert result.is_equivalent is True
        assert result.sample_size == 1

    def test_n1_edge_case_just_outside_lower(self) -> None:
        """n=1 edge case: sample just below lower bound."""
        band = BandTarget(low=0.8, high=0.95)
        result = tost_equivalence_test([0.79999], band)
        assert result.is_equivalent is False
        assert result.sample_size == 1

    def test_n1_edge_case_just_outside_upper(self) -> None:
        """n=1 edge case: sample just above upper bound."""
        band = BandTarget(low=0.8, high=0.95)
        result = tost_equivalence_test([0.95001], band)
        assert result.is_equivalent is False
        assert result.sample_size == 1

    def test_n1_result_statistics_complete(self) -> None:
        """n=1 edge case: all result statistics are valid."""
        band = BandTarget(low=0.8, high=0.95)
        result = tost_equivalence_test([0.87], band, alpha=0.05)

        # Verify all statistics are computed correctly for n=1
        assert isinstance(result.is_equivalent, bool)
        assert result.sample_size == 1
        assert abs(result.sample_mean - 0.87) < 1e-10
        assert result.sample_std == 0.0  # No variance possible with n=1
        # P-values should be valid (0 <= p <= 1)
        assert 0.0 <= result.p_lower <= 1.0
        assert 0.0 <= result.p_upper <= 1.0
        # Confidence interval should be a degenerate point for n=1
        assert len(result.confidence_interval) == 2
        assert result.confidence_interval[0] == result.confidence_interval[1]

    def test_samples_clearly_inside_band(self) -> None:
        """Samples clearly inside band show equivalence."""
        band = BandTarget(low=0.8, high=0.95)
        samples = [0.88, 0.87, 0.89, 0.86, 0.88, 0.90, 0.87, 0.88, 0.89, 0.87]
        result = tost_equivalence_test(samples, band, alpha=0.05)

        assert result.sample_size == 10
        assert 0.85 < result.sample_mean < 0.92
        assert result.is_equivalent is True

    def test_samples_outside_band(self) -> None:
        """Samples clearly outside band show not equivalent."""
        band = BandTarget(low=0.8, high=0.95)
        samples = [0.70, 0.72, 0.68, 0.71, 0.69, 0.73, 0.70, 0.71, 0.69, 0.70]
        result = tost_equivalence_test(samples, band, alpha=0.05)

        assert result.sample_mean < 0.75
        assert result.is_equivalent is False

    def test_result_contains_statistics(self) -> None:
        """TOST result contains all required statistics."""
        band = BandTarget(low=0.8, high=0.95)
        samples = [0.85, 0.87, 0.89, 0.86, 0.88]
        result = tost_equivalence_test(samples, band)

        assert isinstance(result, TOSTResult)
        assert result.sample_size == 5
        assert result.sample_std >= 0
        assert 0.0 <= result.p_lower <= 1.0
        assert 0.0 <= result.p_upper <= 1.0
        assert len(result.confidence_interval) == 2
        assert result.confidence_interval[0] <= result.confidence_interval[1]


class TestCompareBandedObjectives:
    """Tests for compare_banded_objectives function."""

    def test_a_in_band_b_not(self) -> None:
        """A wins when only A is in band."""
        band = BandTarget(low=0.8, high=0.95)
        result = compare_banded_objectives(0.87, 0.75, band)

        assert result.winner == "a"
        assert result.a_in_band is True
        assert result.b_in_band is False

    def test_b_in_band_a_not(self) -> None:
        """B wins when only B is in band."""
        band = BandTarget(low=0.8, high=0.95)
        result = compare_banded_objectives(0.75, 0.87, band)

        assert result.winner == "b"
        assert result.a_in_band is False
        assert result.b_in_band is True

    def test_both_in_band_smaller_deviation_wins(self) -> None:
        """When both in band, smaller deviation wins."""
        band = BandTarget(low=0.8, high=0.95)
        # Both in band, a closer to center
        result = compare_banded_objectives(0.875, 0.81, band)

        assert result.a_in_band is True
        assert result.b_in_band is True
        # a is closer to center (0.875), b is at edge (0.81)
        # Deviation from band edges: a=0, b=0 (both inside)
        # So tie on deviation
        assert result.winner == "tie"

    def test_neither_in_band_smaller_deviation_wins(self) -> None:
        """When neither in band, smaller deviation wins."""
        band = BandTarget(low=0.8, high=0.95)
        result = compare_banded_objectives(0.78, 0.70, band)

        assert result.a_in_band is False
        assert result.b_in_band is False
        assert result.a_deviation < result.b_deviation
        assert result.winner == "a"

    def test_tie_on_deviation(self) -> None:
        """Tie when deviations are equal."""
        band = BandTarget(low=0.8, high=0.95)
        result = compare_banded_objectives(0.85, 0.85, band)

        assert result.winner == "tie"
        assert result.a_deviation == result.b_deviation


class TestCompareBandedWithTOST:
    """Tests for compare_banded_with_tost function."""

    def test_a_equivalent_b_not(self) -> None:
        """A wins when A passes TOST and B doesn't."""
        band = BandTarget(low=0.8, high=0.95)
        samples_a = [0.87, 0.88, 0.86, 0.89, 0.87, 0.88, 0.86, 0.88, 0.87, 0.88]
        samples_b = [0.70, 0.72, 0.68, 0.71, 0.69, 0.73, 0.70, 0.71, 0.69, 0.70]

        winner, tost_a, tost_b = compare_banded_with_tost(samples_a, samples_b, band)

        assert winner == "a"
        assert tost_a.is_equivalent is True
        assert tost_b.is_equivalent is False

    def test_b_equivalent_a_not(self) -> None:
        """B wins when B passes TOST and A doesn't."""
        band = BandTarget(low=0.8, high=0.95)
        samples_a = [0.70, 0.72, 0.68, 0.71, 0.69, 0.73, 0.70, 0.71, 0.69, 0.70]
        samples_b = [0.87, 0.88, 0.86, 0.89, 0.87, 0.88, 0.86, 0.88, 0.87, 0.88]

        winner, tost_a, tost_b = compare_banded_with_tost(samples_a, samples_b, band)

        assert winner == "b"
        assert tost_a.is_equivalent is False
        assert tost_b.is_equivalent is True


class TestBandedObjectiveSpec:
    """Tests for BandedObjectiveSpec class."""

    def test_creation(self) -> None:
        """BandedObjectiveSpec can be created."""
        band = BandTarget(low=0.8, high=0.95)
        spec = BandedObjectiveSpec(
            name="accuracy", target=band, test="TOST", alpha=0.05
        )

        assert spec.name == "accuracy"
        assert spec.target == band
        assert spec.test == "TOST"
        assert spec.alpha == 0.05

    def test_evaluate(self) -> None:
        """evaluate() runs TOST and returns result."""
        band = BandTarget(low=0.8, high=0.95)
        spec = BandedObjectiveSpec(name="accuracy", target=band)

        samples = [0.87, 0.88, 0.86, 0.89, 0.87]
        result = spec.evaluate(samples)

        assert isinstance(result, TOSTResult)
        assert result.sample_size == 5

    def test_is_satisfied(self) -> None:
        """is_satisfied() returns TOST equivalence."""
        band = BandTarget(low=0.8, high=0.95)
        spec = BandedObjectiveSpec(name="accuracy", target=band)

        # Samples inside band
        good_samples = [0.87, 0.88, 0.86, 0.89, 0.87, 0.88, 0.86, 0.88, 0.87, 0.88]
        assert spec.is_satisfied(good_samples) is True

        # Samples outside band
        bad_samples = [0.70, 0.72, 0.68, 0.71, 0.69, 0.73, 0.70, 0.71, 0.69, 0.70]
        assert spec.is_satisfied(bad_samples) is False

    def test_from_dict(self) -> None:
        """from_dict() creates spec from dict representation."""
        data = {
            "name": "latency",
            "band": {"target": [90, 110], "test": "TOST", "alpha": 0.10},
        }
        spec = BandedObjectiveSpec.from_dict(data)

        assert spec.name == "latency"
        assert spec.target.low == 90
        assert spec.target.high == 110
        assert spec.alpha == 0.10
