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
from typing import Literal

from traigent.tvl.statistics import paired_comparison_test

__all__ = [
    "EquivalenceGroupResult",
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
        is_significantly_worse = False

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
                is_significantly_worse = True
                logger.debug(
                    "%s: trial %d significantly worse than trial %d "
                    "(p=%.4f, effect=%.6f)",
                    badge_name,
                    trial["trial_idx"],
                    top_trial["trial_idx"],
                    result.p_value,
                    result.effect_size,
                )
                break

        if is_significantly_worse:
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
    verified = []
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
