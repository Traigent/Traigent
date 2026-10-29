"""USD formatting for console and error messages."""

from __future__ import annotations

import math


def format_usd(amount: float | None) -> str:
    """Format a USD amount without hiding sub-cent values.

    Amounts of at least one dollar, and amounts that are exact cents, use two
    decimals (``$14.12``, ``$0.01``, ``$0.50``). Any other amount below one
    dollar keeps three significant digits (``$0.0051``, ``$0.0104``) and never
    switches to scientific notation, so a limit of ``$0.004`` is not printed
    as ``$0.00`` and an estimate above a limit never prints equal to it.
    """
    if amount is None:
        return "unknown"
    value = float(amount)
    if not math.isfinite(value):
        return f"${value}"
    magnitude = abs(value)
    if (
        magnitude >= 1
        or magnitude == 0
        or math.isclose(magnitude, round(magnitude, 2), rel_tol=0, abs_tol=1e-12)
    ):
        return f"${value:.2f}"
    decimals = 2 - math.floor(math.log10(magnitude))
    text = f"{value:.{decimals}f}".rstrip("0")
    whole, _, frac = text.partition(".")
    return f"${whole}.{frac.ljust(2, '0')}"
