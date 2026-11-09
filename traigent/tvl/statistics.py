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

try:  # scipy ships only in the optional ``[bayesian]`` extra
    from scipy import stats as _scipy_stats
except ImportError:  # pragma: no cover - exercised by hiding scipy in tests
    _scipy_stats = None  # type: ignore[assignment]


@dataclass(slots=True)
class PairedComparisonResult:
    """Result of a paired comparison test.

    Attributes:
        reject_null: Whether the null hypothesis is rejected.
        p_value: P-value of the test.
        effect_size: Estimated effect size (difference in means).
        test_statistic: The computed test statistic.
        degrees_of_freedom: Degrees of freedom for the test.
        zero_variance_inconclusive: True when the differences had no spread
            and the caller asked for ``zero_variance="inconclusive"``; the
            p-value is then forced to 1.0 because a mean cannot be tested.
    """

    reject_null: bool
    p_value: float
    effect_size: float
    test_statistic: float
    degrees_of_freedom: int
    zero_variance_inconclusive: bool = False


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


def bonferroni_adjust(p_values: Sequence[float]) -> list[float]:
    """Apply Bonferroni multiple-testing correction.

    Args:
        p_values: Sequence of raw p-values.

    Returns:
        List of adjusted p-values (same order as input).
    """
    n = len(p_values)
    if n == 0:
        return []

    return [min(p * n, 1.0) for p in p_values]


def holm_bonferroni_adjust(p_values: Sequence[float]) -> list[float]:
    """Apply step-down Holm-Bonferroni multiple-testing correction.

    Args:
        p_values: Sequence of raw p-values.

    Returns:
        List of adjusted p-values (same order as input).
    """
    n = len(p_values)
    if n == 0:
        return []

    indexed = [(p, i) for i, p in enumerate(p_values)]
    indexed.sort(key=lambda x: x[0])

    adjusted = [0.0] * n
    previous_adjusted = 0.0
    for rank, (p, original_idx) in enumerate(indexed):
        current_adjusted = min(p * (n - rank), 1.0)
        current_adjusted = max(current_adjusted, previous_adjusted)
        adjusted[original_idx] = current_adjusted
        previous_adjusted = current_adjusted

    return adjusted


def clopper_pearson_lower_bound(
    successes: int,
    trials: int,
    confidence: float,
) -> float:
    """Compute the exact Clopper-Pearson lower confidence bound.

    The bound is the ``(1 - confidence) / 2`` quantile of
    ``Beta(successes, trials - successes + 1)`` at every ``trials`` (the
    two-sided convention). With scipy installed (the ``[bayesian]`` extra) the
    quantile comes from :func:`scipy.stats.beta.ppf`; without it a bisection on
    the regularized incomplete beta (:func:`_beta_quantile`) gives the same
    value to about 1e-12 relative accuracy for practical sample sizes. ``successes == 0`` returns 0.0 and
    ``successes == trials`` is the closed form ``(alpha/2) ** (1/trials)``.

    Earlier versions used a Wilson interval for ``trials >= 30`` (looser than
    exact at small counts, e.g. 1/40 at 95%: 0.004425 vs 0.000633) and a
    non-converging Newton iteration for ``trials < 30`` (#3222, #3051).

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
    if confidence <= 0 or confidence >= 1:  # NOSONAR - defensive validation
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")

    if successes == 0:
        return 0.0

    tail = (1 - confidence) / 2
    k = successes
    n = trials
    if k == n:
        # Beta(n, 1) has CDF x**n, so the quantile is exact in closed form.
        return float(tail ** (1.0 / n))

    if _scipy_stats is not None:
        return float(_scipy_stats.beta.ppf(tail, k, n - k + 1))
    return _beta_quantile(tail, k, n - k + 1)


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
    if p >= 1:  # NOSONAR - defensive boundary check
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


def _beta_quantile(p: float, a: float, b: float) -> float:
    """Beta quantile by bisection on the regularized incomplete beta.

    I_x(a, b) is strictly increasing in x, so bisection on (0, 1) converges
    monotonically. Stops when the bracket is below 1e-13 of its upper end (relative, so
    extreme tails stay accurate) or after 200 halvings, so it always terminates.

    Returns:
        x such that P(X <= x) = p for X ~ Beta(a, b).
    """
    if p <= 0:
        return 0.0
    if p >= 1:  # NOSONAR - defensive boundary check
        return 1.0

    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if _regularized_beta(mid, a, b) < p:
            lo = mid
        else:
            hi = mid
        if hi - lo <= 1e-13 * hi:
            break
    return 0.5 * (lo + hi)


def _compute_beta_coefficient(x: float, a: float, b: float) -> float:
    """Compute the beta function coefficient bt = x^a * (1-x)^b / B(a, b)."""
    try:
        log_bt = a * math.log(x) + b * math.log(1 - x) - _log_beta(a, b)
        return math.exp(log_bt)
    except (ValueError, OverflowError):
        return 0.0


def _cf_step(aa: float, c: float, d: float, fpmin: float) -> tuple[float, float, float]:
    """Single continued fraction step in Lentz's algorithm."""
    d = 1.0 + aa * d
    if abs(d) < fpmin:
        d = fpmin
    c = 1.0 + aa / c
    if abs(c) < fpmin:
        c = fpmin
    d = 1.0 / d
    delta = d * c
    return c, d, delta


def _regularized_beta(x: float, a: float, b: float) -> float:
    """Approximate regularized incomplete beta function I_x(a, b).

    Uses continued fraction approximation.

    Args:
        x: Upper limit of integration, strictly in (0, 1). Callers must validate.
        a: First shape parameter (positive).
        b: Second shape parameter (positive).

    Returns:
        I_x(a, b) = B(x; a, b) / B(a, b).

    Note:
        This internal function assumes x is valid. All callers compute x from
        expressions like df/(df + t^2) that are guaranteed to be in (0, 1).
    """
    # For numerical stability, use I_x if x < threshold, else 1 - I_{1-x}(b, a)
    threshold = (a + 1) / (a + b + 2)
    if x > threshold:
        return 1.0 - _regularized_beta(1 - x, b, a)

    bt = _compute_beta_coefficient(x, a, b)

    # Lentz's continued fraction algorithm
    eps = 1e-15
    fpmin = 1e-30

    c = 1.0
    d = 1.0 - (a + b) * x / (a + 1)
    if abs(d) < fpmin:
        d = fpmin
    d = 1.0 / d
    h = d

    for m in range(1, 10000):
        m2 = 2 * m
        # Even step
        aa_even = m * (b - m) * x / ((a + m2 - 1) * (a + m2))
        c, d, delta = _cf_step(aa_even, c, d, fpmin)
        h *= delta

        # Odd step
        aa_odd = -(a + m) * (a + b + m) * x / ((a + m2) * (a + m2 + 1))
        c, d, delta = _cf_step(aa_odd, c, d, fpmin)
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


def paired_comparison_test(
    x_samples: Sequence[float],
    y_samples: Sequence[float],
    epsilon: float,
    direction: Literal["greater", "less"],
    zero_variance: Literal["inconclusive", "sign"] = "inconclusive",
) -> PairedComparisonResult:
    """Test if x is better than y by at least epsilon using a paired t-test.

    This implements a paired t-test on the differences (x_i - y_i):
    - "greater": H0: mean(x-y) <= epsilon, H1: mean(x-y) > epsilon
    - "less": H0: mean(x-y) >= -epsilon, H1: mean(x-y) < -epsilon

    For epsilon-Pareto dominance, we test whether x dominates y with margin epsilon.

    The paired test requires equal-length, aligned samples (e.g., metrics from
    the same evaluation runs for incumbent vs candidate).

    Args:
        x_samples: Samples from the first distribution (must match y_samples length).
        y_samples: Samples from the second distribution (must match x_samples length).
        epsilon: The margin/tolerance for the comparison.
        direction: Direction of the test ("greater" or "less").
        zero_variance: How to decide a zero-variance difference vector, where
            the t statistic is undefined. ``"inconclusive"`` (default, fails
            closed for mean-based policy decisions such as the promotion gate)
            returns ``p_value=1.0`` and sets ``zero_variance_inconclusive``.
            ``"sign"`` decides it with an exact sign test (see
            :func:`zero_variance_sign_test_p_value`), which answers "better on
            more examples than chance", not a mean difference.

    Returns:
        PairedComparisonResult with test outcome and statistics.

    Raises:
        ValueError: If samples are empty or have mismatched lengths.
    """
    _validate_paired_samples(x_samples, y_samples)

    n = len(x_samples)
    differences = [x - y for x, y in zip(x_samples, y_samples, strict=True)]
    mean_diff, var_diff = _compute_difference_stats(differences, n)
    se_diff = math.sqrt(var_diff / n) if n > 1 and var_diff > 0 else 0.0
    df = max(1, n - 1)

    if se_diff < 1e-15:
        if zero_variance == "inconclusive":
            return PairedComparisonResult(
                reject_null=False,
                p_value=1.0,
                effect_size=mean_diff,
                test_statistic=0.0,
                degrees_of_freedom=df,
                zero_variance_inconclusive=True,
            )
        return _handle_degenerate_case(differences, epsilon, direction, df)

    t_stat, p_value = _compute_test_statistics(
        mean_diff, epsilon, se_diff, direction, df
    )

    return PairedComparisonResult(
        reject_null=False,  # Deprecated: use p_value with your alpha
        p_value=p_value,
        effect_size=mean_diff,
        test_statistic=t_stat,
        degrees_of_freedom=df,
    )


def _validate_paired_samples(
    x_samples: Sequence[float], y_samples: Sequence[float]
) -> None:
    """Validate that paired samples are non-empty and equal length."""
    if len(x_samples) == 0 or len(y_samples) == 0:
        raise ValueError("Cannot perform comparison with empty samples")
    if len(x_samples) != len(y_samples):
        raise ValueError(
            f"Paired test requires equal-length samples, got {len(x_samples)} and {len(y_samples)}"
        )


def _compute_difference_stats(differences: list[float], n: int) -> tuple[float, float]:
    """Compute mean and variance of differences."""
    mean_diff = sum(differences) / n
    var_diff = (
        sum((d - mean_diff) ** 2 for d in differences) / (n - 1) if n > 1 else 0.0
    )
    return mean_diff, var_diff


def zero_variance_sign_test_p_value(
    differences: Sequence[float],
    epsilon: float,
    direction: Literal["greater", "less"],
) -> float:
    """One-sided exact sign-test p-value for a (near) zero-variance sample.

    When the paired differences have no spread the t statistic is undefined
    (zero standard error), so a t-test cannot decide the case. The exact sign
    test is used instead. For ``direction == "greater"`` let ``k`` be the number
    of differences strictly above ``epsilon`` (``"less"``: strictly below
    ``-epsilon``). Under the null ``P(diff beyond epsilon) <= 1/2`` each pair is
    on the alternative side with probability at most ``1/2``, so
    ``p = P(Binomial(n, 1/2) >= k)`` is exact and conservative, assuming only
    independent pairs (no normality). For a constant difference beyond epsilon
    ``k = n`` and ``p = 0.5**n``; if ``k = 0`` then ``p = 1.0``.

    Estimand: this tests the *median* (sign) of the differences, not the mean.
    A zero-variance sample carries no information about the spread of the
    population, so a mean-based claim would need extra assumptions (e.g.
    symmetry); the sign test is the assumption-light choice and it is
    deliberately conservative at small ``n`` (``n = 5`` cannot go below
    ``0.03125`` one-sided, ``0.0625`` two-sided).

    The result is never exactly 0: it is floored at the smallest positive float.
    """
    n = len(differences)
    if direction == "greater":
        k = sum(1 for d in differences if d > epsilon)
    else:
        k = sum(1 for d in differences if d < -epsilon)
    if k == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(k, n + 1))
    p = tail / (2**n)  # exact big-int ratio; no 0.5**n underflow
    return float(min(1.0, max(p, math.ulp(0.0))))


def _handle_degenerate_case(
    differences: Sequence[float],
    epsilon: float,
    direction: Literal["greater", "less"],
    df: int,
) -> PairedComparisonResult:
    """Decide the zero-variance case with an exact sign test.

    See :func:`zero_variance_sign_test_p_value`. ``reject_null`` is deprecated
    and left ``False`` like the non-degenerate path, so callers must compare
    ``p_value`` with their own alpha.
    """
    p_value = zero_variance_sign_test_p_value(differences, epsilon, direction)
    effect_size = sum(differences) / len(differences)
    return PairedComparisonResult(
        reject_null=False,
        p_value=p_value,
        effect_size=effect_size,
        test_statistic=float("inf") if p_value < 1.0 else 0.0,
        degrees_of_freedom=df,
    )


def _compute_test_statistics(
    mean_diff: float,
    epsilon: float,
    se_diff: float,
    direction: Literal["greater", "less"],
    df: int,
) -> tuple[float, float]:
    """Compute t-statistic and p-value for the test."""
    if direction == "greater":
        t_stat = (mean_diff - epsilon) / se_diff
        p_value = _t_cdf_upper(t_stat, df)
    else:
        t_stat = (mean_diff + epsilon) / se_diff
        p_value = _t_cdf_lower(t_stat, df)
    return t_stat, p_value


def _t_cdf_lower(t: float, df: int) -> float:
    """Lower tail probability P(T <= t), computed directly (no 1 - upper)."""
    return _t_sf(-t, df)


def _t_cdf_upper(t: float, df: int) -> float:
    """Upper tail probability P(T >= t), computed directly (never as 1 - CDF)."""
    return _t_sf(t, df)


def _t_cdf_approx(t: float, df: int) -> float:
    """CDF of the t-distribution, P(T <= t), exact for every ``df``.

    Equal to the lower tail; use :func:`_t_cdf_upper` for upper-tail p-values,
    since ``1 - _t_cdf_approx`` loses the tail to rounding (#3223).
    """
    return _t_sf(-t, df)


def _t_sf(t: float, df: int) -> float:
    """Survival function P(T >= t) of Student's t with ``df`` degrees of freedom.

    Uses :func:`scipy.stats.t.sf` when scipy is importable. Otherwise it uses
    ``sf(t) = 0.5 * I_x(df / 2, 1 / 2)`` with ``x = df / (df + t**2)`` for
    ``t > 0`` (and ``1`` minus that for ``t < 0``) on the regularized incomplete
    beta, for every ``df`` -- there is no normal-approximation switch, and the
    tail is never obtained as ``1 - CDF`` so it neither loses precision nor
    underflows to 0.0 before the true value does (#3223).
    """
    if df <= 0:
        return 0.5
    if t == float("inf"):
        return 0.0
    if t == float("-inf"):
        return 1.0
    if t == 0:
        return 0.5
    if abs(t) > 1e100:
        # t * t would overflow and scipy returns 0.0 for a representable tail.
        # For x = df / (df + t**2) -> 0, I_x(a, 1/2) ~ x**a / (a * B(a, 1/2)).
        a = df / 2.0
        log_tail = (
            math.log(0.5)
            + a * (math.log(df) - 2.0 * math.log(abs(t)))
            - math.log(a)
            - _log_beta(a, 0.5)
        )
        tail = math.exp(log_tail) if log_tail > -745 else 0.0
        return tail if t > 0 else 1.0 - tail
    if _scipy_stats is not None:
        return float(_scipy_stats.t.sf(t, df))
    tail = 0.5 * _regularized_beta(df / (df + t * t), df / 2.0, 0.5)
    return tail if t > 0 else 1.0 - tail


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


def _compare_objective(
    a_val: float, b_val: float, direction: Literal["maximize", "minimize"]
) -> tuple[bool, bool]:
    """Compare single objective values.

    Returns:
        Tuple of (is_at_least_as_good, is_strictly_better).
    """
    if direction == "maximize":
        return a_val >= b_val, a_val > b_val
    else:  # minimize
        return a_val <= b_val, a_val < b_val


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

    strictly_better = False
    for i, direction in enumerate(directions):
        at_least_as_good, is_better = _compare_objective(a[i], b[i], direction)
        if not at_least_as_good:
            return False
        if is_better:
            strictly_better = True

    return strictly_better


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
    # This PRNG is used for statistical sampling, NOT for security purposes.
    # Using random.Random is appropriate here - we need reproducible sampling
    # with a fixed seed (42) for deterministic hypervolume estimation.
    import random  # noqa: PLC0415,S311 - local import for sampling

    rng = random.Random(42)  # NOSONAR - Monte Carlo sampling, not security

    count_dominated = 0
    for _ in range(n_samples):
        sample = []
        for i in range(n_dims):
            fraction = rng.random()  # NOSONAR - Monte Carlo sampling, not cryptographic
            sample.append(
                reference_point[i] + fraction * (upper_bounds[i] - reference_point[i])
            )

        # Check if dominated by any point in front
        for point in valid_points:
            if all(point[i] >= sample[i] for i in range(n_dims)):
                count_dominated += 1
                break

    return box_volume * count_dominated / n_samples
