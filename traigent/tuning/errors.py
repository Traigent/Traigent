"""Closed error codes for the Continuous Tuning builders."""

from __future__ import annotations

E_FLOAT_RATE = "E_FLOAT_RATE"
E_RATE_PRECISION = "E_RATE_PRECISION"
E_INVALID_TUNING = "E_INVALID_TUNING"
E_TVL_VALIDATOR_UNAVAILABLE = "E_TVL_VALIDATOR_UNAVAILABLE"
E_NOT_ROUNDTRIPPABLE = "E_NOT_ROUNDTRIPPABLE"


class TuningError(ValueError):
    """A Continuous Tuning error with a closed ``code``."""

    def __init__(self, code: str, message: str = "") -> None:
        self.code = code
        self.detail = message
        super().__init__(f"{code}: {message}" if message else code)


class ValidatorUnavailable(TuningError):
    """The TVL reference toolchain is missing or does not support TVL 1.7."""

    def __init__(self, reason: str = "") -> None:
        super().__init__(E_TVL_VALIDATOR_UNAVAILABLE, reason)
