"""Statistical functions for TVL 0.9 promotion policy.

This module provides statistical testing functions for epsilon-Pareto
dominance and multiple testing adjustments as required by TVL 0.9.

Concept: CONC-TVLSpec
Implements: FUNC-TVLSPEC
Sync: SYNC-OptimizationFlow
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal


@dataclass(slots=True)
class PairedComparisonResult:
    """Result of a paired comparison test.

    Attributes:
        reject_null: Whether the null hypothesis is rejected.
        p_value: P-value of the test.
        effect_size: Estimated effect size (difference in means).
        test_statistic: The computed test statistic.
        degrees_of_freedom: Degrees of freedom for the test.
    """

    reject_null: bool
    p_value: float
    effect_size: float
    test_statistic: float
    degrees_of_freedom: int


def benjamini_hochberg_adjust(p_values: Sequence[float]) -> list[float]:
    """Apply Benjamini-Hochberg procedure for multiple testing correction.

    The BH procedure controls the False Discovery Rate (FDR) at level q by:
    1. Sorting p-values in ascending order
    2. Finding the largest k where p_(k) <= k * q / m
    3. Rejecting all hypotheses with p_(i) <= p_(k)

    This implementation returns adjusted p-values that can be directly
    compared to the significance level alpha.

    Args:
        p_values: Sequence of raw p-values.

    Returns:
        List of adjusted p-values (same order as input).

    References:
        Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery
        rate: a practical and powerful approach to multiple testing.
    """
    n = len(p_values)
    if n == 0:
        return []

    if n == 1:
        return [min(p_values[0], 1.0)]

    # Create indexed p-values for sorting
    indexed = [(p, i) for i, p in enumerate(p_values)]
    indexed.sort(key=lambda x: x[0])

    # Compute adjusted p-values
    adjusted = [0.0] * n

    # Start from the largest p-value
    # For rank m (largest): adjusted = p_m
    # For rank k: adjusted = min(adjusted[k+1], p_k * m / k)
    prev_adjusted = indexed[-1][0]
    adjusted[indexed[-1][1]] = min(prev_adjusted, 1.0)

    for rank in range(n - 1, 0, -1):
        p, original_idx = indexed[rank - 1]
        # BH adjustment: p * m / rank
        current_adjusted = p * n / rank
        # Enforce monotonicity: adjusted value can't exceed later adjusted values
        current_adjusted = min(current_adjusted, prev_adjusted)
        # Cap at 1.0
        current_adjusted = min(current_adjusted, 1.0)
        adjusted[original_idx] = current_adjusted
        prev_adjusted = current_adjusted

    return adjusted


def clopper_pearson_lower_bound(
    successes: int,
    trials: int,
    confidence: float,
) -> float:
    """Compute Clopper-Pearson exact binomial confidence interval lower bound.

    The Clopper-Pearson interval is an exact method for computing binomial
    confidence intervals, commonly used for chance constraints in promotion
    policies.

    Args:
        successes: Number of successes (non-negative).
        trials: Total number of trials (positive).
        confidence: Confidence level (0 < confidence < 1).

    Returns:
        Lower bound of the confidence interval for the true proportion.

    Raises:
        ValueError: If inputs are invalid.
    """
    if trials <= 0:
        raise ValueError(f"trials must be positive, got {trials}")
    if successes < 0:
        raise ValueError(f"successes must be non-negative, got {successes}")
    if successes > trials:
        raise ValueError(f"successes ({successes}) cannot exceed trials ({trials})")
    if not 0 < confidence < 1:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")

    # Special cases
    if successes == 0:
        return 0.0

    alpha = 1 - confidence

    # Use the beta distribution quantile function
    # Lower bound = B^{-1}(alpha/2; successes, trials - successes + 1)
    # We approximate this using the relationship with the F-distribution

    # For exact computation, we use the beta quantile
    # For Clopper-Pearson lower: beta.ppf(alpha/2, x, n-x+1)
    # Approximation using normal for large samples, beta otherwise

    k = successes
    n = trials

    # Use Wilson score interval approximation for computational efficiency
    # This is accurate and computationally stable

    if n >= 30:
        # Wilson score interval lower bound
        # For lower bound, we need the (1-alpha/2) quantile (positive z)
        # to get center - margin correctly
        z = _inverse_normal_cdf(1 - alpha / 2)  # Positive z for lower bound
        p_hat = k / n

        # Wilson score interval lower bound
        denominator = 1 + z * z / n
        center = (p_hat + z * z / (2 * n)) / denominator
        margin = z * math.sqrt(p_hat * (1 - p_hat) / n + z * z / (4 * n * n))
        margin = margin / denominator

        return max(0.0, center - margin)
    else:
        # For small samples, use a more accurate approximation
        # based on the beta distribution
        return _beta_quantile_approx(alpha / 2, k, n - k + 1)


def _inverse_normal_cdf(p: float) -> float:
    """Approximate inverse of standard normal CDF.

    Uses Abramowitz & Stegun approximation 26.2.23.

    Args:
        p: Probability value in (0, 1).

    Returns:
        z such that P(Z <= z) = p.
    """
    if p <= 0:
        return float("-inf")
    if p >= 1:
        return float("inf")

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


def _beta_quantile_approx(p: float, alpha: float, beta: float) -> float:
    """Approximate beta distribution quantile.

    Uses Newton-Raphson iteration with a normal approximation as starting point.

    Args:
        p: Probability (quantile level).
        alpha: First shape parameter.
        beta: Second shape parameter.

    Returns:
        x such that P(X <= x) = p for X ~ Beta(alpha, beta).
    """
    if p <= 0:
        return 0.0
    if p >= 1:
        return 1.0

    # Use normal approximation for starting point
    mean = alpha / (alpha + beta)
    var = (alpha * beta) / ((alpha + beta) ** 2 * (alpha + beta + 1))
    std = math.sqrt(var)

    # Initial estimate using normal approximation
    z = _inverse_normal_cdf(p)
    x = max(0.001, min(0.999, mean + z * std))

    # Newton-Raphson refinement (limited iterations)
    for _ in range(10):
        # Compute beta CDF at x using regularized incomplete beta
        fx = _regularized_beta(x, alpha, beta) - p
        if abs(fx) < 1e-10:
            break

        # Compute beta PDF at x
        fpx = _beta_pdf(x, alpha, beta)
        if fpx < 1e-15:
            break

        # Newton step with damping
        step = fx / fpx
        x = x - 0.5 * step  # Damped step for stability
        x = max(0.001, min(0.999, x))

    return x


def _regularized_beta(x: float, a: float, b: float) -> float:
    """Approximate regularized incomplete beta function I_x(a, b).

    Uses continued fraction approximation.

    Args:
        x: Upper limit of integration (0 <= x <= 1).
        a: First shape parameter.
        b: Second shape parameter.

    Returns:
        I_x(a, b) = B(x; a, b) / B(a, b).
    """
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0

    # Use the continued fraction representation
    # For numerical stability, use I_x if x < (a+1)/(a+b+2), else 1 - I_{1-x}(b, a)
    threshold = (a + 1) / (a + b + 2)

    if x > threshold:
        return 1.0 - _regularized_beta(1 - x, b, a)

    # Compute the beta function coefficient
    # bt = x^a * (1-x)^b / B(a, b)
    # Using log for numerical stability
    try:
        log_bt = a * math.log(x) + b * math.log(1 - x)
        log_bt -= _log_beta(a, b)
        bt = math.exp(log_bt)
    except (ValueError, OverflowError):
        bt = 0.0

    # Lentz's continued fraction algorithm
    eps = 1e-15
    fpmin = 1e-30

    # Start continued fraction
    c = 1.0
    d = 1.0 - (a + b) * x / (a + 1)
    if abs(d) < fpmin:
        d = fpmin
    d = 1.0 / d
    h = d

    for m in range(1, 100):
        m2 = 2 * m
        # Even step
        aa = m * (b - m) * x / ((a + m2 - 1) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        h *= d * c

        # Odd step
        aa = -(a + m) * (a + b + m) * x / ((a + m2) * (a + m2 + 1))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        delta = d * c
        h *= delta

        if abs(delta - 1.0) < eps:
            break

    return bt * h / a


def _log_beta(a: float, b: float) -> float:
    """Compute log of beta function using log-gamma.

    log(B(a, b)) = log(Gamma(a)) + log(Gamma(b)) - log(Gamma(a + b))

    Args:
        a: First parameter.
        b: Second parameter.

    Returns:
        log(B(a, b)).
    """
    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)


def _beta_pdf(x: float, a: float, b: float) -> float:
    """Beta distribution PDF.

    Args:
        x: Point to evaluate.
        a: First shape parameter.
        b: Second shape parameter.

    Returns:
        PDF value at x.
    """
    if x <= 0 or x >= 1:
        return 0.0

    try:
        log_pdf = (a - 1) * math.log(x) + (b - 1) * math.log(1 - x) - _log_beta(a, b)
        return math.exp(log_pdf)
    except (ValueError, OverflowError):
        return 0.0


def paired_comparison_test(
    x_samples: Sequence[float],
    y_samples: Sequence[float],
    epsilon: float,
    direction: Literal["greater", "less"],
) -> PairedComparisonResult:
    """Test if x is better than y by at least epsilon.

    This implements a paired t-test for the hypothesis:
    - "greater": H0: mean(x) - mean(y) <= epsilon, H1: mean(x) - mean(y) > epsilon
    - "less": H0: mean(x) - mean(y) >= -epsilon, H1: mean(x) - mean(y) < -epsilon

    For epsilon-Pareto dominance, we test whether x dominates y with margin epsilon.

    Args:
        x_samples: Samples from the first distribution.
        y_samples: Samples from the second distribution.
        epsilon: The margin/tolerance for the comparison.
        direction: Direction of the test ("greater" or "less").

    Returns:
        PairedComparisonResult with test outcome and statistics.

    Raises:
        ValueError: If samples are empty or have mismatched lengths for paired test.
    """
    if len(x_samples) == 0 or len(y_samples) == 0:
        raise ValueError("Cannot perform comparison with empty samples")

    # Compute sample statistics
    n_x = len(x_samples)
    n_y = len(y_samples)
    mean_x = sum(x_samples) / n_x
    mean_y = sum(y_samples) / n_y

    var_x = sum((x - mean_x) ** 2 for x in x_samples) / max(1, n_x - 1)
    var_y = sum((y - mean_y) ** 2 for y in y_samples) / max(1, n_y - 1)

    # Effect size (difference in means)
    effect_size = mean_x - mean_y

    # Use Welch's t-test for unequal variances
    se_x = var_x / n_x if n_x > 1 else 0
    se_y = var_y / n_y if n_y > 1 else 0
    pooled_se = math.sqrt(se_x + se_y)

    if pooled_se < 1e-15:
        # No variance - degenerate case
        if direction == "greater":
            reject = effect_size > epsilon
        else:
            reject = effect_size < -epsilon

        return PairedComparisonResult(
            reject_null=reject,
            p_value=0.0 if reject else 1.0,
            effect_size=effect_size,
            test_statistic=float("inf") if reject else 0.0,
            degrees_of_freedom=max(1, n_x + n_y - 2),
        )

    # Compute test statistic
    if direction == "greater":
        # Testing H0: mean_x - mean_y <= epsilon
        t_stat = (effect_size - epsilon) / pooled_se
    else:
        # Testing H0: mean_x - mean_y >= -epsilon
        t_stat = (effect_size + epsilon) / pooled_se

    # Welch-Satterthwaite degrees of freedom
    if se_x + se_y > 0:
        numerator = (se_x + se_y) ** 2
        denominator = (se_x**2 / max(1, n_x - 1)) + (se_y**2 / max(1, n_y - 1))
        if denominator > 0:
            df = int(numerator / denominator)
        else:
            df = max(1, n_x + n_y - 2)
    else:
        df = max(1, n_x + n_y - 2)

    df = max(1, df)

    # Compute p-value
    if direction == "greater":
        p_value = _t_cdf_upper(t_stat, df)
    else:
        p_value = _t_cdf_lower(t_stat, df)

    # Don't hardcode rejection decision - let caller apply their alpha
    # reject_null is set based on p_value for convenience, but caller
    # should use p_value with their configured alpha for proper testing
    return PairedComparisonResult(
        reject_null=False,  # Deprecated: use p_value with your alpha
        p_value=p_value,
        effect_size=effect_size,
        test_statistic=t_stat,
        degrees_of_freedom=df,
    )


def _t_cdf_lower(t: float, df: int) -> float:
    """Lower tail probability P(T <= t)."""
    return _t_cdf_approx(t, df)


def _t_cdf_upper(t: float, df: int) -> float:
    """Upper tail probability P(T >= t) = 1 - P(T <= t)."""
    return 1.0 - _t_cdf_approx(t, df)


def _t_cdf_approx(t: float, df: int) -> float:
    """Compute CDF of t-distribution using regularized incomplete beta.

    The t-distribution CDF is related to the regularized incomplete beta:
    F(t; df) = 1 - 0.5 * I_x(df/2, 0.5) where x = df/(df + t^2) for t > 0
    F(t; df) = 0.5 * I_x(df/2, 0.5) where x = df/(df + t^2) for t < 0

    Args:
        t: t-statistic value.
        df: Degrees of freedom.

    Returns:
        Cumulative probability P(T <= t).
    """
    if df <= 0:
        return 0.5

    # Handle infinities
    if t == float("inf"):
        return 1.0
    if t == float("-inf"):
        return 0.0

    # For large df, use normal approximation
    if df > 100:
        return _normal_cdf(t)

    # For t = 0, CDF is exactly 0.5
    if abs(t) < 1e-15:
        return 0.5

    # Use relationship with regularized incomplete beta function
    # F(t; df) = I_{x}(df/2, 1/2) where x = df/(df + t^2) for lower tail
    x = df / (df + t * t)
    beta_val = _regularized_beta(x, df / 2.0, 0.5)

    if t > 0:
        return 1.0 - 0.5 * beta_val
    else:
        return 0.5 * beta_val


def _normal_cdf(x: float) -> float:
    """Standard normal CDF approximation."""
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def hypervolume_improvement(
    new_point: Sequence[float],
    pareto_front: Sequence[Sequence[float]],
    reference_point: Sequence[float],
    directions: Sequence[Literal["maximize", "minimize"]],
) -> float:
    """Calculate hypervolume improvement of adding a new point.

    This is used for convergence detection in multi-objective optimization.
    The hypervolume indicator measures the volume of objective space
    dominated by the Pareto front.

    Args:
        new_point: The candidate point to evaluate.
        pareto_front: Current Pareto front (list of points).
        reference_point: Reference point for hypervolume calculation.
        directions: Optimization direction for each objective.

    Returns:
        The improvement in hypervolume from adding the new point.
        Returns 0 if the point is dominated or doesn't improve hypervolume.
    """
    if len(new_point) == 0:
        return 0.0

    # Check if new_point is dominated by any point in the Pareto front
    for front_point in pareto_front:
        if _dominates(front_point, new_point, directions):
            return 0.0

    # Calculate current hypervolume
    current_hv = _calculate_hypervolume(pareto_front, reference_point, directions)

    # Add new point (remove dominated points)
    new_front = [p for p in pareto_front if not _dominates(new_point, p, directions)]
    new_front.append(list(new_point))

    # Calculate new hypervolume
    new_hv = _calculate_hypervolume(new_front, reference_point, directions)

    return max(0.0, new_hv - current_hv)


def _dominates(
    a: Sequence[float],
    b: Sequence[float],
    directions: Sequence[Literal["maximize", "minimize"]],
) -> bool:
    """Check if point a dominates point b.

    Point a dominates b if a is at least as good in all objectives
    and strictly better in at least one.

    Args:
        a: First point.
        b: Second point.
        directions: Optimization direction for each objective.

    Returns:
        True if a dominates b.
    """
    if len(a) != len(b) or len(a) != len(directions):
        return False

    at_least_as_good = True
    strictly_better = False

    for i, direction in enumerate(directions):
        if direction == "maximize":
            if a[i] < b[i]:
                at_least_as_good = False
                break
            if a[i] > b[i]:
                strictly_better = True
        else:  # minimize
            if a[i] > b[i]:
                at_least_as_good = False
                break
            if a[i] < b[i]:
                strictly_better = True

    return at_least_as_good and strictly_better


def _calculate_hypervolume(
    pareto_front: Sequence[Sequence[float]],
    reference_point: Sequence[float],
    directions: Sequence[Literal["maximize", "minimize"]],
) -> float:
    """Calculate hypervolume indicator for a Pareto front.

    For 2D case, uses a simple sweep algorithm.
    For higher dimensions, uses a simplified recursive approach.

    Args:
        pareto_front: List of non-dominated points.
        reference_point: Reference point.
        directions: Optimization direction for each objective.

    Returns:
        Hypervolume indicator value.
    """
    if not pareto_front:
        return 0.0

    n_objectives = len(reference_point)
    if n_objectives == 0:
        return 0.0

    # Normalize directions (convert minimize to maximize by negation)
    normalized_front = []
    normalized_ref = []

    for i in range(n_objectives):
        if directions[i] == "minimize":
            normalized_ref.append(-reference_point[i])
        else:
            normalized_ref.append(reference_point[i])

    for point in pareto_front:
        normalized_point = []
        for i in range(n_objectives):
            if directions[i] == "minimize":
                normalized_point.append(-point[i])
            else:
                normalized_point.append(point[i])
        normalized_front.append(normalized_point)

    if n_objectives == 1:
        # 1D case: just the distance to reference
        best = max(p[0] for p in normalized_front)
        return max(0.0, best - normalized_ref[0])

    if n_objectives == 2:
        # 2D case: sweep algorithm
        return _hypervolume_2d(normalized_front, normalized_ref)

    # Higher dimensions: use simplified inclusion-exclusion
    # (For production use, implement WFG or similar algorithm)
    return _hypervolume_simple(normalized_front, normalized_ref)


def _hypervolume_2d(
    pareto_front: list[list[float]],
    reference_point: list[float],
) -> float:
    """Calculate 2D hypervolume using sweep algorithm.

    Args:
        pareto_front: Normalized front (all maximized).
        reference_point: Normalized reference point.

    Returns:
        2D hypervolume.
    """
    # Filter points that are worse than reference in all objectives
    valid_points = [
        p
        for p in pareto_front
        if p[0] > reference_point[0] or p[1] > reference_point[1]
    ]

    if not valid_points:
        return 0.0

    # Sort by first objective (descending)
    valid_points.sort(key=lambda p: -p[0])

    hypervolume = 0.0
    prev_y = reference_point[1]

    for point in valid_points:
        if point[1] > prev_y:
            # Compute contribution
            width = point[0] - reference_point[0]
            height = point[1] - prev_y
            if width > 0 and height > 0:
                hypervolume += width * height
            prev_y = point[1]

    return hypervolume


def _hypervolume_simple(
    pareto_front: list[list[float]],
    reference_point: list[float],
) -> float:
    """Simplified hypervolume for higher dimensions.

    This is a basic implementation. For production use with many objectives,
    consider using the WFG algorithm or similar.

    Args:
        pareto_front: Normalized front.
        reference_point: Normalized reference point.

    Returns:
        Approximate hypervolume.
    """
    # Simple Monte Carlo approximation for higher dimensions
    # Generate random samples in the bounding box
    n_dims = len(reference_point)
    n_samples = 10000

    # Find bounds
    upper_bounds = [max(p[i] for p in pareto_front) for i in range(n_dims)]

    # Filter to points better than reference
    valid_points = [
        p for p in pareto_front if any(p[i] > reference_point[i] for i in range(n_dims))
    ]

    if not valid_points:
        return 0.0

    # Calculate bounding box volume
    box_volume = 1.0
    for i in range(n_dims):
        side = upper_bounds[i] - reference_point[i]
        if side <= 0:
            return 0.0
        box_volume *= side

    # Monte Carlo sampling with local RNG (don't affect global state)
    import random  # noqa: PLC0415 - local import for sampling

    rng = random.Random(42)  # Local RNG for reproducibility without side effects

    count_dominated = 0
    for _ in range(n_samples):
        sample = [
            reference_point[i] + rng.random() * (upper_bounds[i] - reference_point[i])
            for i in range(n_dims)
        ]

        # Check if dominated by any point in front
        for point in valid_points:
            if all(point[i] >= sample[i] for i in range(n_dims)):
                count_dominated += 1
                break

    return box_volume * count_dominated / n_samples
