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
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from traigent.tvl.statistics import paired_comparison_test
from traigent.utils.objectives import is_minimization_objective

if TYPE_CHECKING:
    from traigent.api.types import TrialResult

__all__ = [
    "EquivalenceGroupResult",
    "compute_significance",
    "extract_trial_data_for_metric",
    "find_equivalence_group",
]

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
                "%s: trial %d significantly worse than trial %d "
                "(p=%.4f, effect=%.6f)",
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
