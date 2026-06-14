"""Edit-budget schedules for skill-document training."""

from __future__ import annotations

import math
from typing import Literal

EditBudgetSchedule = Literal["constant", "cosine"]


def edit_budget_for_step(
    *,
    edit_budget: int,
    edit_budget_floor: int,
    schedule: EditBudgetSchedule,
    step_index: int,
    total_steps: int,
) -> int:
    """Return the edit budget for a zero-based global trainer step."""

    if schedule == "constant":
        return edit_budget
    if schedule != "cosine":
        raise ValueError(f"Unsupported edit budget schedule: {schedule!r}")

    if edit_budget <= 0:
        raise ValueError("edit_budget must be > 0")
    if edit_budget_floor <= 0:
        raise ValueError("edit_budget_floor must be > 0")

    floor = min(edit_budget_floor, edit_budget)
    if total_steps <= 1:
        return edit_budget

    clamped_step = min(max(step_index, 0), total_steps - 1)
    progress = clamped_step / (total_steps - 1)
    value = floor + (edit_budget - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))
    return max(floor, min(edit_budget, int(round(value))))


__all__ = ["EditBudgetSchedule", "edit_budget_for_step"]
