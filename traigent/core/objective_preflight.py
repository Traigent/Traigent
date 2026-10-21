"""Pre-trial check that each objective can be produced by some metric (#2668).

Without any user scorer, the local evaluator can only emit SDK-owned metric
names, so an objective outside that closed set (typically a typo) can never be
measured and the run would end ``OBJECTIVE_UNMATCHED`` only after spending every
trial. This warns up front. Any user scorer (``metric_functions`` or
``scoring_function``) makes the binding undecidable, because a mapping-returning
scorer may emit arbitrary keys, so the check stays silent then.

It logs a warning rather than raising: the name set is a heuristic and a hard
rejection is a product decision.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from difflib import get_close_matches
from typing import Any

from traigent.core.objective_directions import CANONICAL_OBJECTIVE_ORIENTATIONS
from traigent.utils.logging import get_logger

logger = get_logger(__name__)

# Metric names the built-in local evaluator emits besides the canonical
# SDK-owned objective names.
_EXTRA_BUILTIN_METRICS = frozenset(
    {
        "score",
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "evaluation_cost",
        "cost_unmeasured",
        "cost_lower_bound",
    }
)


def warn_unproducible_objectives(
    metric_functions: Mapping[str, Callable[..., Any]] | None,
    objectives: Sequence[str],
    scoring_function: Callable[..., Any] | None = None,
) -> list[str]:
    """Warn for objectives no built-in metric can produce; return their names."""
    if metric_functions or scoring_function is not None:
        return []
    known = sorted(set(CANONICAL_OBJECTIVE_ORIENTATIONS) | _EXTRA_BUILTIN_METRICS)
    unmatched = [str(o) for o in objectives if str(o) not in known]
    # Objective names are customer-chosen text (paths, labels, newlines): never
    # echo them. Log the ordinal plus, at most, the SDK's own known-name hint.
    for ordinal, name in enumerate(unmatched, start=1):
        close = get_close_matches(name, known, n=1, cutoff=0.6)
        hint = f" Did you mean '{close[0]}'?" if close else ""
        logger.warning(
            "Objective #%d of %d unmatched objective(s) matches no built-in metric "
            "and no metric_functions/scoring_function is configured, so it can "
            "never be measured and the run will end OBJECTIVE_UNMATCHED after all "
            "trials.%s",
            ordinal,
            len(unmatched),
            hint,
        )
    return unmatched
