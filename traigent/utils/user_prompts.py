"""User prompt formatting helpers."""

from __future__ import annotations

from collections.abc import Iterable

PROMPT_WIDTH = 63


def _print_box(lines: Iterable[str]) -> None:
    border = "=" * PROMPT_WIDTH
    print(border)
    for line in lines:
        print(line)
    print(border)


def print_budget_prompt(current_limit: float, spent: float) -> None:
    """Print the budget limit reached prompt with clear labeling."""
    lines: list[str] = [
        "  BUDGET LIMIT REACHED",
        "",
        "  Your optimization cost limit has been reached.",
        "",
        f"  Spent / Limit: ${spent:.2f} / ${current_limit:.2f}",
    ]
    if spent > current_limit:
        overage = spent - current_limit
        lines.extend(
            [
                f"  Over limit by: ${overage:.2f}",
                f"  Minimum add to continue: > ${overage:.2f}",
            ]
        )
    lines.extend(
        [
            "",
            "  [1] Raise limit and CONTINUE optimization",
            "  [2] STOP optimization",
        ]
    )
    _print_box(lines)
    print()


def print_vendor_error_prompt(title: str, explanation: str) -> None:
    """Print the vendor error prompt with a concise explanation."""
    lines: list[str] = [
        "  VENDOR ERROR ENCOUNTERED",
        "",
        f"  Error: {title}",
        "",
    ]
    lines.extend([f"  {line}" for line in explanation.splitlines()])
    lines.extend(
        [
            "",
            "  [1] RESUME optimization",
            "  [2] STOP optimization",
        ]
    )
    _print_box(lines)
    print()
