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
    if value < 0:
        return "-" + format_usd(-value)
    if not math.isfinite(value):
        return f"${value}"
    magnitude = abs(value)
    if (
        magnitude >= 1
        or magnitude == 0
        or math.isclose(magnitude, round(magnitude, 2), rel_tol=0, abs_tol=1e-12)
    ):
        return f"${value:.2f}"
    base = 2 - math.floor(math.log10(magnitude))
    text = ""
    # Three significant digits; if that collapses to a different exact-cent
    # amount (0.01004 -> "0.01"), keep digits until it no longer does, so an
    # amount above a limit never prints equal to it.
    for decimals in range(base, base + 7):
        text = f"{value:.{decimals}f}"
        shown = float(text)
        if not math.isclose(abs(shown), round(abs(shown), 2), rel_tol=0, abs_tol=1e-12):
            break
        if decimals > base and shown != round(shown, 2):
            break
        if shown != value and decimals >= base + 6:
            break
    text = text.rstrip("0")
    whole, _, frac = text.partition(".")
    return f"${whole}.{frac.ljust(2, '0')}"
