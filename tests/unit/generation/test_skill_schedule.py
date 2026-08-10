from __future__ import annotations

from traigent.generation.skill_train.schedule import edit_budget_for_step


def test_constant_edit_budget_schedule_is_unchanged() -> None:
    values = [
        edit_budget_for_step(
            edit_budget=6,
            edit_budget_floor=2,
            schedule="constant",
            step_index=step,
            total_steps=5,
        )
        for step in range(5)
    ]

    assert values == [6, 6, 6, 6, 6]


def test_cosine_edit_budget_schedule_endpoints() -> None:
    first = edit_budget_for_step(
        edit_budget=6,
        edit_budget_floor=2,
        schedule="cosine",
        step_index=0,
        total_steps=5,
    )
    last = edit_budget_for_step(
        edit_budget=6,
        edit_budget_floor=2,
        schedule="cosine",
        step_index=4,
        total_steps=5,
    )

    assert first == 6
    assert last == 2


def test_cosine_edit_budget_schedule_is_monotone_and_respects_floor() -> None:
    values = [
        edit_budget_for_step(
            edit_budget=8,
            edit_budget_floor=3,
            schedule="cosine",
            step_index=step,
            total_steps=9,
        )
        for step in range(9)
    ]

    assert values == sorted(values, reverse=True)
    assert min(values) == 3
