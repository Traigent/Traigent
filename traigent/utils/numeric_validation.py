"""Small shared validators for positive numeric options and TRAIGENT_* env vars.

Rejects ``bool``, NaN, infinity, non-integral values where an integer is
required, and non-positive values, with an error naming the option or the
environment variable (Traigent#2898, Traigent#2899).
"""

from __future__ import annotations

import logging
import math
import os

_logger = logging.getLogger(__name__)


def validate_positive_int(name: str, value: object) -> int:
    """Return ``value`` as a positive ``int`` or raise ``ValueError`` naming ``name``.

    Integral floats such as ``3.0`` are normalised to ``int``. Bools, NaN, inf,
    non-integral floats, strings and values below 1 are rejected.
    """
    candidate: object = value
    if isinstance(candidate, float):
        if not math.isfinite(candidate) or not candidate.is_integer():
            raise ValueError(
                f"{name} must be a positive integer, "
                f"got {type(value).__name__} {value!r}"
            )
        candidate = int(candidate)
    if isinstance(candidate, bool) or not isinstance(candidate, int):
        raise ValueError(
            f"{name} must be a positive integer, got {type(value).__name__} {value!r}"
        )
    if candidate < 1:
        raise ValueError(f"{name} must be a positive integer (got {candidate})")
    return candidate


def parse_positive_finite_float(name: str, raw: object) -> float:
    """Parse ``raw`` as a finite float > 0 or raise ``ValueError`` naming ``name``."""
    if isinstance(raw, bool):
        raise ValueError(f"{name} must be a positive finite number, got {raw!r}")
    try:
        value = float(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{name} must be a positive finite number, got {raw!r}"
        ) from exc
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive finite number, got {raw!r}")
    return value


def positive_float_env(name: str, default: float) -> float:
    """Read env var ``name`` as a positive finite float, else warn and use ``default``."""
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return parse_positive_finite_float(name, raw)
    except ValueError:
        _logger.warning("Invalid %s=%r, using default %s", name, raw, default)
        return default


def positive_int_env(name: str, default: int) -> int:
    """Read env var ``name`` as a positive int, else warn and use ``default``.

    Accepts integer text only (``"3600"``); ``"1.5"``, ``"nan"`` and ``"abc"``
    are rejected.
    """
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        parsed = int(raw.strip())
    except ValueError:
        parsed = 0
    if parsed < 1:
        _logger.warning("Invalid %s=%r, using default %s", name, raw, default)
        return default
    return parsed
