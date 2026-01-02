"""Banded objectives and TOST equivalence testing for TVL 0.9.

This module provides functions for evaluating banded objectives using
Two One-Sided Tests (TOST) for equivalence testing, as defined in
TVL 0.9 specification.

Concept: CONC-TVLSpec
Implements: FUNC-TVLSPEC
Sync: SYNC-OptimizationFlow
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from .models import BandTarget
from .statistics import _t_cdf_approx


@dataclass(slots=True)
class TOSTResult:
    """Result of a Two One-Sided Tests (TOST) equivalence test.

    Attributes:
        is_equivalent: Whether the sample is statistically equivalent to the band.
        p_lower: P-value for the lower bound test (H0: mean <= low).
        p_upper: P-value for the upper bound test (H0: mean >= high).
        sample_mean: Mean of the samples.
        sample_std: Standard deviation of the samples.
        sample_size: Number of samples.
        confidence_interval: 90% CI for the mean (corresponding to alpha=0.05 two-sided).
    """

    is_equivalent: bool
    p_lower: float
    p_upper: float
    sample_mean: float
    sample_std: float
    sample_size: int
    confidence_interval: tuple[float, float]


@dataclass(slots=True)
class BandedComparisonResult:
    """Result of comparing two values against a banded target.

    Attributes:
        winner: Which value is better ("a", "b", or "tie").
        a_deviation: Deviation of value A from the band.
        b_deviation: Deviation of value B from the band.
        a_in_band: Whether value A is within the band.
        b_in_band: Whether value B is within the band.
    """

    winner: Literal["a", "b", "tie"]
    a_deviation: float
    b_deviation: float
    a_in_band: bool
    b_in_band: bool


def is_in_band(value: float, target: BandTarget) -> bool:
    """Check if a value falls within the target band.

    Args:
        value: The value to check.
        target: The band target specification.

    Returns:
        True if value is within [low, high], False otherwise.
    """
    return target.contains(value)


def band_deviation(value: float, target: BandTarget) -> float:
    """Calculate the deviation of a value from the band.

    Args:
        value: The value to measure.
        target: The band target specification.

    Returns:
        0 if inside band, positive distance to nearest bound otherwise.
    """
    return target.deviation(value)


def _compute_t_statistic(
    sample_mean: float,
    hypothesized_mean: float,
    sample_std: float,
    sample_size: int,
) -> float:
    """Compute t-statistic for one-sample t-test.

    Args:
        sample_mean: Mean of the sample.
        hypothesized_mean: The hypothesized population mean.
        sample_std: Standard deviation of the sample.
        sample_size: Number of samples.

    Returns:
        t-statistic value.
    """
    if sample_std == 0:
        if sample_mean == hypothesized_mean:
            return 0.0
        return float("inf") if sample_mean > hypothesized_mean else float("-inf")

    standard_error = sample_std / math.sqrt(sample_size)
    return (sample_mean - hypothesized_mean) / standard_error


# _t_cdf_approx is imported from statistics module


def _compute_p_value_one_sided_greater(t_stat: float, df: int) -> float:
    """Compute p-value for one-sided t-test (H1: mean > hypothesized).

    Args:
        t_stat: t-statistic value.
        df: Degrees of freedom.

    Returns:
        P-value for the test.
    """
    return 1.0 - _t_cdf_approx(t_stat, df)


def _compute_p_value_one_sided_less(t_stat: float, df: int) -> float:
    """Compute p-value for one-sided t-test (H1: mean < hypothesized).

    Args:
        t_stat: t-statistic value.
        df: Degrees of freedom.

    Returns:
        P-value for the test.
    """
    return _t_cdf_approx(t_stat, df)


def _compute_confidence_interval(
    sample_mean: float,
    sample_std: float,
    sample_size: int,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Compute confidence interval for the mean.

    For TOST with alpha=0.05, we use a 90% CI (1 - 2*alpha).

    Args:
        sample_mean: Mean of the sample.
        sample_std: Standard deviation of the sample.
        sample_size: Number of samples.
        alpha: Significance level for each one-sided test.

    Returns:
        (lower, upper) bounds of confidence interval.
    """
    if sample_size <= 1 or sample_std == 0:
        return (sample_mean, sample_mean)

    df = sample_size - 1
    standard_error = sample_std / math.sqrt(sample_size)

    # t-critical value for (1 - alpha) one-sided, which corresponds to
    # (1 - 2*alpha) two-sided confidence interval
    # Approximation for t-critical
    if df > 100:
        # Normal approximation
        t_crit = (
            1.645
            if math.isclose(alpha, 0.05, rel_tol=0.0, abs_tol=1e-12)
            else _inverse_normal_cdf(1 - alpha)
        )
    else:
        # Approximation for small df
        t_crit = _approximate_t_critical(1 - alpha, df)

    margin = t_crit * standard_error
    return (sample_mean - margin, sample_mean + margin)


def _inverse_normal_cdf(p: float) -> float:
    """Approximate inverse of standard normal CDF.

    Uses Abramowitz & Stegun approximation 26.2.23.

    Args:
        p: Probability value in (0, 1).

    Returns:
        z such that P(Z <= z) = p.
    """
    # Handle boundary cases
    if p <= 0 or p >= 1:
        return float("-inf") if p <= 0 else float("inf")

    # For p > 0.5, use symmetry
    if p > 0.5:
        return -_inverse_normal_cdf(1 - p)

    # Abramowitz & Stegun approximation
    t = math.sqrt(-2 * math.log(p))
    c0, c1, c2 = 2.515517, 0.802853, 0.010328
    d1, d2, d3 = 1.432788, 0.189269, 0.001308

    return -(
        t - (c0 + c1 * t + c2 * t * t) / (1 + d1 * t + d2 * t * t + d3 * t * t * t)
    )


def _approximate_t_critical(confidence: float, df: int) -> float:
    """Approximate t-critical value.

    Args:
        confidence: Confidence level (e.g., 0.95 for 95%).
        df: Degrees of freedom.

    Returns:
        Approximate t-critical value.
    """
    # Get z-score for normal
    z = _inverse_normal_cdf(confidence)

    # Cornish-Fisher expansion for t correction
    g1 = (z**3 + z) / 4
    g2 = (5 * z**5 + 16 * z**3 + 3 * z) / 96
    g3 = (3 * z**7 + 19 * z**5 + 17 * z**3 - 15 * z) / 384

    return z + g1 / df + g2 / (df * df) + g3 / (df**3)


def tost_equivalence_test(
    samples: Sequence[float],
    target: BandTarget,
    alpha: float = 0.05,
) -> TOSTResult:
    """Perform Two One-Sided Tests (TOST) for equivalence.

    Tests whether the population mean is statistically contained within
    the target band [low, high] at significance level alpha.

    The null hypothesis is that the mean is NOT in the band (either
    mean <= low OR mean >= high). Rejecting both one-sided nulls
    establishes equivalence.

    Args:
        samples: Sequence of observed values.
        target: Band target specification with [low, high] bounds.
        alpha: Significance level for each one-sided test (default 0.05).

    Returns:
        TOSTResult with equivalence decision and statistics.

    Raises:
        ValueError: If samples is empty or target bounds are invalid.
    """
    if not samples:
        raise ValueError("Cannot perform TOST on empty samples")

    if target.low is None or target.high is None:
        raise ValueError("BandTarget must have defined low and high bounds")

    n = len(samples)
    sample_mean = sum(samples) / n

    if n == 1:
        # Single sample: can only check if it's in the band
        in_band = target.contains(sample_mean)
        return TOSTResult(
            is_equivalent=in_band,
            p_lower=0.0 if sample_mean > target.low else 1.0,
            p_upper=0.0 if sample_mean < target.high else 1.0,
            sample_mean=sample_mean,
            sample_std=0.0,
            sample_size=1,
            confidence_interval=(sample_mean, sample_mean),
        )

    # Compute sample standard deviation
    variance = sum((x - sample_mean) ** 2 for x in samples) / (n - 1)
    sample_std = math.sqrt(variance)

    df = n - 1

    # Lower bound test: H0: mean <= low, H1: mean > low
    # We want to reject H0 (show mean is greater than low)
    t_lower = _compute_t_statistic(sample_mean, target.low, sample_std, n)
    p_lower = _compute_p_value_one_sided_greater(t_lower, df)

    # Upper bound test: H0: mean >= high, H1: mean < high
    # We want to reject H0 (show mean is less than high)
    t_upper = _compute_t_statistic(sample_mean, target.high, sample_std, n)
    p_upper = _compute_p_value_one_sided_less(t_upper, df)

    # Equivalence is established if BOTH null hypotheses are rejected
    is_equivalent = p_lower < alpha and p_upper < alpha

    # Compute confidence interval
    ci = _compute_confidence_interval(sample_mean, sample_std, n, alpha)

    return TOSTResult(
        is_equivalent=is_equivalent,
        p_lower=p_lower,
        p_upper=p_upper,
        sample_mean=sample_mean,
        sample_std=sample_std,
        sample_size=n,
        confidence_interval=ci,
    )


def compare_banded_objectives(
    value_a: float,
    value_b: float,
    target: BandTarget,
) -> BandedComparisonResult:
    """Compare two values against a banded target.

    For banded objectives, the goal is to be within the band.
    If both or neither are in the band, compare by deviation.

    Args:
        value_a: First value to compare.
        value_b: Second value to compare.
        target: The band target specification.

    Returns:
        BandedComparisonResult indicating which value is better.
    """
    a_in_band = target.contains(value_a)
    b_in_band = target.contains(value_b)
    a_deviation = target.deviation(value_a)
    b_deviation = target.deviation(value_b)

    # Determine winner
    if a_in_band and not b_in_band:
        winner: Literal["a", "b", "tie"] = "a"
    elif b_in_band and not a_in_band:
        winner = "b"
    else:
        # Both in band or both outside: compare deviations
        # Smaller deviation is better (closer to band center)
        if abs(a_deviation - b_deviation) < 1e-10:
            winner = "tie"
        elif a_deviation < b_deviation:
            winner = "a"
        else:
            winner = "b"

    return BandedComparisonResult(
        winner=winner,
        a_deviation=a_deviation,
        b_deviation=b_deviation,
        a_in_band=a_in_band,
        b_in_band=b_in_band,
    )


def compare_banded_with_tost(
    samples_a: Sequence[float],
    samples_b: Sequence[float],
    target: BandTarget,
    alpha: float = 0.05,
) -> tuple[Literal["a", "b", "tie"], TOSTResult, TOSTResult]:
    """Compare two sample sets against a banded target using TOST.

    This function performs TOST on both sample sets and determines
    which one is more likely to satisfy the banded objective.

    Args:
        samples_a: First set of samples.
        samples_b: Second set of samples.
        target: The band target specification.
        alpha: Significance level for TOST.

    Returns:
        Tuple of (winner, tost_a, tost_b) where winner is "a", "b", or "tie".
    """
    if target.low is None or target.high is None:
        raise ValueError("BandTarget must have defined low and high bounds")

    tost_a = tost_equivalence_test(samples_a, target, alpha)
    tost_b = tost_equivalence_test(samples_b, target, alpha)

    # If one is equivalent and the other isn't, equivalent wins
    if tost_a.is_equivalent and not tost_b.is_equivalent:
        return ("a", tost_a, tost_b)
    if tost_b.is_equivalent and not tost_a.is_equivalent:
        return ("b", tost_a, tost_b)

    # If both are equivalent or both are not, compare by deviation from band center
    band_center = (target.low + target.high) / 2
    a_dist = abs(tost_a.sample_mean - band_center)
    b_dist = abs(tost_b.sample_mean - band_center)

    if abs(a_dist - b_dist) < 1e-10:
        return ("tie", tost_a, tost_b)
    elif a_dist < b_dist:
        return ("a", tost_a, tost_b)
    else:
        return ("b", tost_a, tost_b)


@dataclass(slots=True)
class BandedObjectiveSpec:
    """Specification for a banded objective.

    This combines the objective name with its band target and
    statistical test configuration.

    Attributes:
        name: Name of the objective/metric.
        target: The band target specification.
        test: Statistical test to use (currently only "TOST" supported).
        alpha: Significance level for the test.
    """

    name: str
    target: BandTarget
    test: Literal["TOST"] = "TOST"
    alpha: float = 0.05

    def evaluate(self, samples: Sequence[float]) -> TOSTResult:
        """Evaluate samples against this banded objective.

        Args:
            samples: Sequence of observed values.

        Returns:
            TOSTResult with equivalence decision.
        """
        return tost_equivalence_test(samples, self.target, self.alpha)

    def is_satisfied(self, samples: Sequence[float]) -> bool:
        """Check if samples satisfy this banded objective.

        Args:
            samples: Sequence of observed values.

        Returns:
            True if TOST establishes equivalence.
        """
        return self.evaluate(samples).is_equivalent

    @classmethod
    def from_dict(cls, data: dict) -> BandedObjectiveSpec:
        """Create from dictionary representation.

        Args:
            data: Dictionary with name, band (target, test, alpha).

        Returns:
            BandedObjectiveSpec instance.
        """
        band_data = data.get("band", {})
        target = BandTarget.from_dict(band_data.get("target", band_data))

        return cls(
            name=data["name"],
            target=target,
            test=band_data.get("test", "TOST"),
            alpha=float(band_data.get("alpha", 0.05)),
        )
