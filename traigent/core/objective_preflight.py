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

import inspect
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


def _is_near_miss(key: str, objective: str) -> bool:
    """Whether ``key`` looks like a misspelt or abbreviated ``objective``."""
    key, objective = key.lower(), objective.lower()
    if not key or not objective:
        return False
    return (
        objective.startswith(key)
        or key.startswith(objective)
        or bool(get_close_matches(key, [objective], n=1, cutoff=0.6))
    )


def _may_return_mapping(func: Callable[..., Any]) -> bool:
    """Whether ``func`` could emit arbitrary metric keys (undecidable pre-run)."""
    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):
        return True
    # Only a known scalar return annotation proves the scorer cannot emit a
    # mapping; unannotated scorers may return {"accuracy": ...}.
    annotation = signature.return_annotation
    return annotation not in (float, int, bool, "float", "int", "bool")


def _warn_misnamed_metric_keys(
    metric_functions: Mapping[str, Callable[..., Any]] | None,
    objectives: Sequence[str],
) -> list[str]:
    """Warn when a metric_functions key is a near-miss for an uncovered objective.

    ``metric_functions={"acc": f}`` with ``objectives=["accuracy"]`` silently
    leaves the objective to the built-in metric, so the user's scorer is never
    the optimization signal. Only an objective with no key of its own is
    considered, and only keys that are not themselves objectives, so a legitimate
    informational metric next to an objective never warns. Names are
    customer-chosen text and are never echoed; an objective is named only when it
    is an SDK-owned built-in. Returns the offending keys.
    """
    if not metric_functions:
        return []
    # A scorer that may return a mapping can supply any objective key, so the
    # dict keys do not decide coverage: stay silent rather than guess.
    if any(_may_return_mapping(f) for f in metric_functions.values()):
        return []
    keys = [str(k) for k in metric_functions]
    objective_names = {str(o) for o in objectives}
    uncovered = [o for o in objectives if str(o) not in keys]
    known = set(CANONICAL_OBJECTIVE_ORIENTATIONS) | _EXTRA_BUILTIN_METRICS
    flagged: list[str] = []
    for ordinal, key in enumerate(keys, start=1):
        if key in objective_names:
            continue
        match = next((str(o) for o in uncovered if _is_near_miss(key, str(o))), None)
        if match is None:
            continue
        flagged.append(key)
        target = f"built-in objective '{match}'" if match in known else "an objective"
        logger.warning(
            "metric_functions key #%d of %d matches no objective but looks like a "
            "near-miss for %s, which has no metric_functions entry. That objective "
            "may not be scored by your function (the built-in metric or no metric "
            "is used instead). If that is not intended, rename the key to the "
            "objective name.",
            ordinal,
            len(keys),
            target,
        )
    return flagged


def warn_unproducible_objectives(
    metric_functions: Mapping[str, Callable[..., Any]] | None,
    objectives: Sequence[str],
    scoring_function: Callable[..., Any] | None = None,
) -> list[str]:
    """Warn for objectives no built-in metric can produce; return their names."""
    if metric_functions or scoring_function is not None:
        if scoring_function is None:
            _warn_misnamed_metric_keys(metric_functions, objectives)
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
