"""Statistical significance equivalence groups for trial comparison.

Provides functions to identify groups of trials that are statistically
indistinguishable from the best AND significantly better than the rest,
using paired t-tests on per-example continuous metrics.

Example:
    >>> from traigent.core.stat_significance import find_equivalence_group
    >>> trials = [
    ...     {"values": [0.001, 0.002, 0.001], "metric_value": 0.0013, "trial_idx": 0},
    ...     {"values": [0.010, 0.012, 0.011], "metric_value": 0.011, "trial_idx": 1},
    ... ]
    >>> winners = find_equivalence_group(trials, higher_is_better=False)
    >>> # winners == [0] if trial 0 is significantly cheaper
"""

# Traceability: CONC-Layer-Core CONC-Quality-Performance FUNC-OPT-ALGORITHMS SYNC-OptimizationFlow

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from traigent.tvl.statistics import _inverse_normal_cdf, paired_comparison_test
from traigent.utils.objectives import (
    coerce_finite_objective_score,
    is_minimization_objective,
)

if TYPE_CHECKING:
    from traigent.api.types import TrialResult

__all__ = [
    "BEST_CONFIG_MARGIN_ALPHA",
    "EquivalenceGroupResult",
    "compute_best_config_margin",
    "compute_significance",
    "extract_trial_data_for_metric",
    "find_equivalence_group",
]

BEST_CONFIG_MARGIN_ALPHA = 0.05

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EquivalenceGroupResult:
    """Result of an equivalence group computation.

    Attributes:
        winners: Trial indices that received the badge.
        top_group: Trial indices in the top (statistically equivalent) group.
        rest_group: Trial indices in the rest group.
        badge_name: Name of the badge/metric evaluated.
    """

    winners: list[int]
    top_group: list[int]
    rest_group: list[int]
    badge_name: str


def _is_significantly_worse_than_any(
    trial: dict,
    top_group: list[dict],
    *,
    direction: Literal["greater", "less"],
    higher_is_better: bool,
    alpha: float,
    epsilon: float,
    badge_name: str,
) -> bool:
    """Return True if *trial* is significantly worse than any member of *top_group*."""
    for top_trial in top_group:
        result = paired_comparison_test(
            x_samples=top_trial["values"],
            y_samples=trial["values"],
            epsilon=epsilon,
            direction=direction,
        )
        if higher_is_better:
            top_is_better = top_trial["metric_value"] > trial["metric_value"]
        else:
            top_is_better = top_trial["metric_value"] < trial["metric_value"]

        if result.p_value < alpha and top_is_better:
            logger.debug(
                "%s: trial %d significantly worse than trial %d (p=%.4f, effect=%.6f)",
                badge_name,
                trial["trial_idx"],
                top_trial["trial_idx"],
                result.p_value,
                result.effect_size,
            )
            return True
    return False


def _verify_winners(
    top_group: list[dict],
    rest_group: list[dict],
    *,
    direction: Literal["greater", "less"],
    alpha: float,
    epsilon: float,
) -> list[int]:
    """Return indices of top-group trials that significantly beat at least one rest trial."""
    verified: list[int] = []
    for top_trial in top_group:
        for rest_trial in rest_group:
            result = paired_comparison_test(
                x_samples=top_trial["values"],
                y_samples=rest_trial["values"],
                epsilon=epsilon,
                direction=direction,
            )
            if result.p_value < alpha:
                verified.append(top_trial["trial_idx"])
                break
    return verified


def find_equivalence_group(
    trial_data: list[dict],
    alpha: float = 0.05,
    higher_is_better: bool = True,
    badge_name: str = "",
    epsilon: float = 0.0,
) -> EquivalenceGroupResult:
    """Find the equivalence group of trials for a continuous metric.

    Identifies trials that are:
    1. Not significantly different from each other (form a "top group")
    2. Significantly better than at least one trial outside the group

    Uses a paired t-test to compare per-example values between trials.

    Algorithm:
        1. Sort trials by aggregate metric (best first).
        2. Greedily build a top group: add each trial unless it is
           significantly worse than any current top-group member.
        3. If no rest group exists, return empty (everyone equivalent).
        4. Verify: each top-group trial must significantly beat at least
           one rest-group trial to be awarded the badge.

    Args:
        trial_data: List of dicts, each with:
            - "values": list[float] — per-example metric values
            - "metric_value": float — aggregate metric (mean, total, etc.)
            - "trial_idx": int — original trial index
        alpha: Significance level (default 0.05).
        higher_is_better: If True, higher metric is better (accuracy).
            If False, lower is better (cost, latency).
        badge_name: Label for logging.
        epsilon: Margin for the paired t-test (default 0.0).

    Returns:
        EquivalenceGroupResult with winners, top_group, rest_group.
    """
    if len(trial_data) < 2:
        return EquivalenceGroupResult(
            winners=[], top_group=[], rest_group=[], badge_name=badge_name
        )

    # Sort by metric (best first)
    sorted_trials = sorted(
        trial_data, key=lambda t: t["metric_value"], reverse=higher_is_better
    )

    # Direction for paired t-test
    direction: Literal["greater", "less"] = "greater" if higher_is_better else "less"

    # Build top group greedily
    top_group = [sorted_trials[0]]
    rest_group: list[dict] = []

    for trial in sorted_trials[1:]:
        if _is_significantly_worse_than_any(
            trial,
            top_group,
            direction=direction,
            higher_is_better=higher_is_better,
            alpha=alpha,
            epsilon=epsilon,
            badge_name=badge_name,
        ):
            rest_group.append(trial)
        else:
            top_group.append(trial)

    top_indices = [t["trial_idx"] for t in top_group]
    rest_indices = [t["trial_idx"] for t in rest_group]

    # If no rest group, everyone is equivalent — no badges
    if not rest_group:
        logger.debug(
            "%s: all %d trials equivalent, no badges awarded",
            badge_name,
            len(trial_data),
        )
        return EquivalenceGroupResult(
            winners=[],
            top_group=top_indices,
            rest_group=[],
            badge_name=badge_name,
        )

    # Verify: each top trial must significantly beat at least one rest trial
    verified = _verify_winners(
        top_group,
        rest_group,
        direction=direction,
        alpha=alpha,
        epsilon=epsilon,
    )

    logger.debug(
        "%s: winners=%s (top=%s, rest=%s)",
        badge_name,
        verified,
        top_indices,
        rest_indices,
    )

    return EquivalenceGroupResult(
        winners=verified,
        top_group=top_indices,
        rest_group=rest_indices,
        badge_name=badge_name,
    )


def _extract_example_map(
    example_results: list[Any],
    metric_name: str,
) -> dict[str, float]:
    """Build {example_id: metric_value} from a trial's example_results list."""
    example_map: dict[str, float] = {}
    for er in example_results:
        if isinstance(er, dict):
            eid = er.get("example_id")
            metrics = er.get("metrics", {})
        else:
            eid = getattr(er, "example_id", None)
            metrics = getattr(er, "metrics", {})
        if eid and metric_name in metrics:
            example_map[eid] = metrics[metric_name]
    return example_map


def extract_trial_data_for_metric(
    trials: list[TrialResult],
    metric_name: str,
) -> list[dict[str, Any]]:
    """Extract per-example metric values from trials for paired comparison.

    Builds aligned value vectors by intersecting on ``example_id`` so that
    paired t-tests compare the same examples across trials.

    Args:
        trials: List of completed trial results.
        metric_name: Name of the metric to extract (e.g. ``"accuracy"``).

    Returns:
        List of dicts with keys ``"values"``, ``"metric_value"``, ``"trial_idx"``,
        one per trial that has both per-example data and the requested metric.
    """
    # First pass: collect per-trial {example_id: metric_value} maps
    trial_maps: list[tuple[int, dict[str, float], float]] = []
    for i, trial in enumerate(trials):
        if not trial.is_successful:
            continue
        if metric_name not in trial.metrics:
            continue
        example_results = trial.metadata.get("example_results", [])
        if not example_results:
            continue

        example_map = _extract_example_map(example_results, metric_name)
        if example_map:
            trial_maps.append((i, example_map, trial.metrics[metric_name]))

    if len(trial_maps) < 2:
        return []

    # Intersect example IDs across all trials
    shared_ids = set(trial_maps[0][1].keys())
    for _, emap, _ in trial_maps[1:]:
        shared_ids &= emap.keys()

    _MIN_SHARED_EXAMPLES = 5
    if len(shared_ids) < _MIN_SHARED_EXAMPLES:
        logger.debug(
            "Too few shared examples (%d) across trials for metric '%s'; need >= %d",
            len(shared_ids),
            metric_name,
            _MIN_SHARED_EXAMPLES,
        )
        return []

    # Build ordered value vectors from shared IDs
    # Use mean of shared examples as aggregate so ranking is consistent
    # with the paired comparison (which only sees shared examples).
    ordered_ids = sorted(shared_ids)
    result: list[dict[str, Any]] = []
    for trial_idx, emap, _full_aggregate in trial_maps:
        values = [emap[eid] for eid in ordered_ids]
        shared_aggregate = sum(values) / len(values)
        result.append(
            {
                "values": values,
                "metric_value": shared_aggregate,
                "trial_idx": trial_idx,
            }
        )

    return result


def compute_significance(
    trials: list[TrialResult],
    objectives: list[str],
    objective_orientations: dict[str, str] | None = None,
    alpha: float = 0.05,
) -> dict[str, dict[str, Any]]:
    """Compute statistical significance for each objective after optimization.

    For each objective, extracts per-example metric data, determines direction,
    and calls :func:`find_equivalence_group` to identify winners.

    Args:
        trials: All trial results from the optimization run.
        objectives: List of objective names (e.g. ``["accuracy", "cost"]``).
        objective_orientations: Optional mapping of objective name to orientation
            (``"maximize"``, ``"minimize"``, ``"band"``).  Typically from
            :class:`~traigent.core.objectives.ObjectiveSchema`.
        alpha: Significance level for paired t-tests.

    Returns:
        Dict mapping objective name to significance result dict with keys
        ``"winners"``, ``"top_group"``, ``"rest_group"``, ``"badge_name"``,
        ``"n_shared_examples"``.
        Objectives with insufficient data (< 5 shared examples) or
        ``"band"`` orientation are skipped.
    """
    results: dict[str, dict[str, Any]] = {}

    for obj_name in objectives:
        # Determine direction
        orientation = (objective_orientations or {}).get(obj_name)

        # Skip band objectives — significance is not meaningful
        if orientation == "band":
            logger.debug("Skipping band objective '%s' for significance", obj_name)
            continue

        if orientation == "maximize":
            higher_is_better = True
        elif orientation == "minimize":
            higher_is_better = False
        else:
            # Fallback to heuristic — no explicit orientation provided
            higher_is_better = not is_minimization_objective(obj_name)
            logger.warning(
                "No explicit orientation for objective '%s'; "
                "using heuristic (higher_is_better=%s)",
                obj_name,
                higher_is_better,
            )

        trial_data = extract_trial_data_for_metric(trials, obj_name)
        if len(trial_data) < 2:
            logger.debug(
                "Skipping significance for '%s': fewer than 2 trials with data",
                obj_name,
            )
            continue

        equiv = find_equivalence_group(
            trial_data=trial_data,
            alpha=alpha,
            higher_is_better=higher_is_better,
            badge_name=obj_name,
        )

        n_shared = len(trial_data[0]["values"])
        results[obj_name] = {
            "winners": equiv.winners,
            "top_group": equiv.top_group,
            "rest_group": equiv.rest_group,
            "badge_name": equiv.badge_name,
            "n_shared_examples": n_shared,
        }

    return results


# ---------------------------------------------------------------------------
# best_config winner-vs-runner-up margin significance (issue #1866)
# ---------------------------------------------------------------------------


def _config_identity(config: dict[str, Any] | None) -> tuple[tuple[str, str], ...]:
    """Order-independent identity key for a config dict."""
    return tuple(sorted((str(k), repr(v)) for k, v in (config or {}).items()))


def _is_binary(values: list[float]) -> bool:
    """True when every value is exactly 0 or 1 (a 0/1 binary scorer)."""
    return all(value in (0, 0.0, 1, 1.0) for value in values)


def _mcnemar_exact_p(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value from discordant pair counts.

    Under H0 (winner and runner-up equally likely to be correct on a discordant
    example) the ``b + c`` discordant pairs follow ``Binomial(b + c, 0.5)``. The
    two-sided exact p-value doubles the smaller tail (capped at 1.0). When there
    are no discordant pairs the two configs agree on every example → ``p = 1.0``.
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) * (0.5**n)
    return min(1.0, 2.0 * tail)


def _paired_proportion_ci(
    b: int, c: int, n_shared: int, alpha: float
) -> tuple[float, float]:
    """Wald CI for the paired difference in proportions ``delta = (b - c)/n``.

    Uses the standard large-sample variance for McNemar's marginal-homogeneity
    difference with a normal quantile (:func:`_inverse_normal_cdf`). ``n_shared``
    is always ``>= b + c`` (discordant pairs cannot exceed the shared examples),
    so the variance is non-negative.
    """
    delta = (b - c) / n_shared
    variance = (b + c - (b - c) ** 2 / n_shared) / (n_shared**2)
    se = math.sqrt(max(0.0, variance))
    z = _inverse_normal_cdf(1.0 - alpha / 2.0)
    return (delta - z * se, delta + z * se)


def _mean_diff_ci(diffs: list[float], alpha: float) -> tuple[float, float]:
    """Normal-approximation CI for the mean of paired differences.

    A large-sample approximation (uses a normal, not a t, quantile); at the
    small n typical of eval folds it is slightly narrower than an exact
    t-interval, so the p-value from the paired t-test remains the primary
    tie criterion in :func:`_verdict`.
    """
    n = len(diffs)
    mean = sum(diffs) / n
    var = sum((d - mean) ** 2 for d in diffs) / (n - 1) if n > 1 else 0.0
    se = math.sqrt(var / n) if n > 0 else 0.0
    z = _inverse_normal_cdf(1.0 - alpha / 2.0)
    return (mean - z * se, mean + z * se)


# Effects with |delta| at or below this are treated as directionless (no
# favoring) when deciding whether a significant result is a clear win.
_DIRECTION_TOL = 1e-12

# Reason attached to a significant margin that favors the RUNNER-UP rather than
# the selected winner (the winner won on non-shared examples but loses on the
# comparable ones), collapsed to the conservative ``"statistical_tie"``.
_ADVERSE_DIRECTION_REASON = (
    "winner significantly underperforms the runner-up on the shared examples; "
    "the selection was driven by non-shared examples"
)


def _verdict(
    p_value: float | None,
    ci: tuple[float, float] | None,
    alpha: float,
    *,
    delta: float | None = None,
    minimize: bool = False,
) -> tuple[str, str | None]:
    """Classify a margin as ``"clear"``, ``"statistical_tie"``, or ``"na"``.

    Returns ``(verdict, reason)``; ``reason`` is ``None`` except for the
    direction-guarded tie described below.

    A ``"statistical_tie"`` is declared when the paired test is not significant
    at ``alpha`` OR the margin's confidence interval includes 0 — either
    condition means the winner is statistically interchangeable with the
    runner-up on the primary objective.

    ``"clear"`` additionally requires the significant effect to FAVOR the
    selected winner. ``delta`` is framed as ``mean(winner - runner)`` in raw
    metric units, so the winner is favored when ``delta > 0`` for maximize
    objectives and ``delta < 0`` for minimize objectives. A significant effect
    that favors the RUNNER-UP — the winner was selected on non-shared examples
    yet loses on every comparable one — is NOT a clear win; it collapses to the
    conservative ``"statistical_tie"`` with an explanatory reason.
    """
    if p_value is None:
        return "na", None
    ci_includes_zero = ci is not None and ci[0] <= 0.0 <= ci[1]
    if p_value > alpha or ci_includes_zero:
        return "statistical_tie", None
    if delta is not None:
        favors_winner = delta < -_DIRECTION_TOL if minimize else delta > _DIRECTION_TOL
        if not favors_winner:
            return "statistical_tie", _ADVERSE_DIRECTION_REASON
    return "clear", None


def _resolve_winner(
    scored: list[tuple[TrialResult, float]],
    winner_trial_id: str | None,
    winner_config: dict[str, Any] | None,
    minimize: bool,
) -> TrialResult:
    """Resolve the winning trial from the selection basis.

    Prefers the exact ``winner_trial_id``; falls back to the best-scoring trial
    of the winning config, then to the overall best-scoring trial.
    """
    if winner_trial_id is not None:
        for trial, _value in scored:
            if trial.trial_id == winner_trial_id:
                return trial
    chooser = min if minimize else max
    if winner_config is not None:
        key = _config_identity(winner_config)
        matches = [(t, v) for t, v in scored if _config_identity(t.config) == key]
        if matches:
            return chooser(matches, key=lambda pair: pair[1])[0]
    return chooser(scored, key=lambda pair: pair[1])[0]


def _mcnemar_margin(
    winner_values: list[float],
    runner_values: list[float],
    n_shared: int,
    effective_alpha: float,
    *,
    minimize: bool,
) -> dict[str, Any]:
    """McNemar exact margin payload for 0/1 binary per-example scores.

    ``effective_alpha`` is the multiplicity-corrected significance level (see
    :func:`compute_best_config_margin`). It drives BOTH the CI level — the
    interval reported in ``ci95`` is the ``(1 - effective_alpha)`` interval — and
    the ``p <= effective_alpha`` threshold in :func:`_verdict`, so the CI and the
    p-value test the same corrected bar.
    """
    b = sum(
        1 for w, r in zip(winner_values, runner_values, strict=True) if w >= 0.5 > r
    )
    c = sum(
        1 for w, r in zip(winner_values, runner_values, strict=True) if w < 0.5 <= r
    )
    p_value = _mcnemar_exact_p(b, c)
    ci = _paired_proportion_ci(b, c, n_shared, effective_alpha)
    delta = (b - c) / n_shared
    verdict, reason = _verdict(
        p_value, ci, effective_alpha, delta=delta, minimize=minimize
    )
    payload: dict[str, Any] = {
        "delta": delta,
        "ci95": [ci[0], ci[1]],
        "p_value": p_value,
        "verdict": verdict,
        "test": "mcnemar_exact",
        "n_shared_examples": n_shared,
        "discordant": {"b": b, "c": c},
    }
    if reason is not None:
        payload["reason"] = reason
    return payload


def _paired_t_margin(
    winner_values: list[float],
    runner_values: list[float],
    n_shared: int,
    effective_alpha: float,
    *,
    minimize: bool,
) -> dict[str, Any]:
    """Paired t-test margin payload for continuous per-example scores.

    Reuses :func:`~traigent.tvl.statistics.paired_comparison_test` (one-sided,
    ``epsilon=0``) and converts to a two-sided p-value. The zero-variance case
    is handled directly: a constant non-zero difference is a perfectly
    consistent (significant) win; an all-zero difference is a perfect tie.

    ``effective_alpha`` is the multiplicity-corrected significance level (see
    :func:`compute_best_config_margin`). It drives BOTH the ``ci95`` interval —
    reported at the ``(1 - effective_alpha)`` level — and the
    ``p <= effective_alpha`` threshold in :func:`_verdict`, keeping the CI and
    the p-value on the same corrected bar.
    """
    diffs = [w - r for w, r in zip(winner_values, runner_values, strict=True)]
    mean_diff = sum(diffs) / n_shared
    var_diff = (
        sum((d - mean_diff) ** 2 for d in diffs) / (n_shared - 1)
        if n_shared > 1
        else 0.0
    )
    if var_diff <= 0.0:
        p_value = 1.0 if abs(mean_diff) <= 1e-12 else 0.0
        ci: tuple[float, float] = (mean_diff, mean_diff)
    else:
        result = paired_comparison_test(
            x_samples=winner_values,
            y_samples=runner_values,
            epsilon=0.0,
            direction="greater",
        )
        p_upper = result.p_value
        p_value = min(1.0, 2.0 * min(p_upper, 1.0 - p_upper))
        mean_diff = result.effect_size
        ci = _mean_diff_ci(diffs, effective_alpha)
    verdict, reason = _verdict(
        p_value, ci, effective_alpha, delta=mean_diff, minimize=minimize
    )
    payload: dict[str, Any] = {
        "delta": mean_diff,
        "ci95": [ci[0], ci[1]],
        "p_value": p_value,
        "verdict": verdict,
        "test": "paired_t",
        "n_shared_examples": n_shared,
    }
    if reason is not None:
        payload["reason"] = reason
    return payload


def compute_best_config_margin(
    eligible_trials: list[TrialResult],
    *,
    winner_trial_id: str | None,
    winner_config: dict[str, Any] | None,
    primary_objective: str | None,
    orientation: str | None = None,
    alpha: float = BEST_CONFIG_MARGIN_ALPHA,
) -> dict[str, Any] | None:
    """Winner-vs-runner-up paired margin significance for ``best_config`` (#1866).

    Additive qualification of ``results.best_config`` — it does NOT change which
    config wins. It runs a paired test between the winner and the runner-up (the
    2nd-best *distinct* config by the primary objective) on their shared
    per-example scores: McNemar exact for 0/1 binary scorers, a paired t-test
    (via :func:`~traigent.tvl.statistics.paired_comparison_test`) for continuous
    scorers. It reports the margin, a CI on the margin, the p-value, and a
    verdict.

    The margin is always computed on the ``primary_objective``'s per-example
    scores (the "is this winner real?" question customers act on), even for
    weighted multi-objective runs — per-example data only exists for scorer
    metrics, not for the weighted aggregate. The runner-up is the best distinct
    config on that same primary objective.

    Multiplicity correction (winner's curse). The winner was NOT pre-specified:
    it was selected as the best of ``n_configs`` *distinct* candidate configs,
    and the runner-up is the empirically-closest of the remaining ones. Testing
    that lucky best-vs-second pair at the nominal ``alpha`` — as if it had been
    fixed in advance — is a post-selection multiplicity error: with enough noise
    configs some pair looks "significant" by chance. We correct with a Bonferroni
    adjustment over the ``n_configs - 1`` comparisons of the winner against the
    other configs::

        effective_alpha = alpha / max(1, n_configs - 1)

    ``n_configs`` is the number of DISTINCT configs in the selection pool (the
    scored eligible trials the winner was chosen from), not the trial count. For
    a genuine head-to-head (``n_configs`` of 1 or 2) ``effective_alpha == alpha``
    — no over-correction. ``effective_alpha`` is applied CONSISTENTLY to both the
    p-value threshold AND the CI level: ``"clear"`` requires
    ``p <= effective_alpha`` AND the ``(1 - effective_alpha)`` CI to exclude 0
    AND the effect to favor the winner. Both ``effective_alpha`` and
    ``n_configs`` are recorded in the payload so the verdict is auditable; the
    uncorrected nominal ``alpha`` is also kept under ``alpha``.

    Returns:
        ``None`` when there is no runner-up to compare against — fewer than two
        distinct eligible configs, no primary objective, or no eligible trial
        carrying a finite primary value. When two configs exist but share no
        comparable per-example data, returns a dict with ``verdict="na"`` and
        ``p_value=None`` (the aggregate delta is still reported). Otherwise the
        dict carries ``verdict`` of ``"clear"`` or ``"statistical_tie"`` plus
        ``runner_up``, ``delta``, ``ci95``, ``p_value``, ``effective_alpha``,
        and ``n_configs``.

    Note:
        A ``"statistical_tie"`` winner is statistically interchangeable with its
        runner-up on the primary objective: at typical eval sizes (n=20-80) a
        margin whose CI includes 0 is not a decision. Secondary objectives
        (cost, latency) should break such ties rather than noise. A
        ``"statistical_tie"`` is also returned — carrying a ``reason`` — when the
        paired test IS significant but the effect favors the runner-up (the
        winner was selected on non-shared examples yet loses on the comparable
        ones); ``"clear"`` requires the significant effect to favor the selected
        winner.
    """
    if primary_objective is None:
        return None
    trials = [trial for trial in eligible_trials if trial.is_successful]
    if len(trials) < 2:
        return None

    minimize = is_minimization_objective(primary_objective, orientation=orientation)

    def primary_value(trial: TrialResult) -> float | None:
        score = coerce_finite_objective_score(trial.get_metric(primary_objective))
        return None if score is None else float(score)

    scored = [(t, v) for t in trials if (v := primary_value(t)) is not None]
    if len(scored) < 2:
        return None

    winner = _resolve_winner(scored, winner_trial_id, winner_config, minimize)
    winner_key = _config_identity(winner.config)
    others = [(t, v) for t, v in scored if _config_identity(t.config) != winner_key]
    if not others:
        return None  # only one distinct config — no runner-up to qualify against

    runner_up = (min if minimize else max)(others, key=lambda pair: pair[1])[0]

    # Multiplicity correction: the winner was selected as best-of-N DISTINCT
    # configs, so the winner-vs-runner-up test carries a winner's-curse /
    # post-selection bias. Bonferroni-correct the nominal alpha over the
    # n_configs - 1 comparisons of the winner against the other configs. Count
    # DISTINCT configs in the scored selection pool, not trials.
    n_configs = len({_config_identity(t.config) for t, _ in scored})
    effective_alpha = alpha / max(1, n_configs - 1)

    winner_primary = primary_value(winner)
    runner_primary = primary_value(runner_up)
    aggregate_delta = (
        winner_primary - runner_primary
        if winner_primary is not None and runner_primary is not None
        else None
    )

    base: dict[str, Any] = {
        "runner_up": dict(runner_up.config or {}),
        "runner_up_trial_id": runner_up.trial_id,
        "winner_trial_id": winner.trial_id,
        "primary_objective": primary_objective,
        "alpha": alpha,
        "effective_alpha": effective_alpha,
        "n_configs": n_configs,
    }

    paired = extract_trial_data_for_metric([winner, runner_up], primary_objective)
    by_idx = {entry["trial_idx"]: entry for entry in paired}
    if 0 not in by_idx or 1 not in by_idx:
        return {
            **base,
            "delta": aggregate_delta,
            "ci95": None,
            "p_value": None,
            "verdict": "na",
            "test": "none",
            "n_shared_examples": 0,
            "reason": "insufficient shared per-example data for a paired test",
        }

    winner_values = by_idx[0]["values"]
    runner_values = by_idx[1]["values"]
    n_shared = len(winner_values)

    if _is_binary(winner_values) and _is_binary(runner_values):
        margin = _mcnemar_margin(
            winner_values, runner_values, n_shared, effective_alpha, minimize=minimize
        )
    else:
        margin = _paired_t_margin(
            winner_values, runner_values, n_shared, effective_alpha, minimize=minimize
        )
    return {**base, **margin}
