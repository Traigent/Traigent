"""Cost formatting shared by the multi-objective-tradeoff examples."""

from __future__ import annotations


def format_cost(total_cost: float | None, lower_bound: float | None = None) -> str:
    """Format a run's total cost, never presenting an unmeasured cost as $0.

    ``total_cost`` is ``None`` when any trial's cost could not be measured; the
    known part of the spend is then only a lower bound.
    """
    if total_cost is None:
        if lower_bound:
            return f"unmeasured (known spend is at least ${lower_bound:.6f})"
        return "unmeasured"
    return f"${total_cost:.6f}"
