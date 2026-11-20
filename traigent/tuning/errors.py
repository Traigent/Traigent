"""Closed error codes for the Continuous Tuning builders."""

from __future__ import annotations

from collections.abc import Sequence

E_FLOAT_RATE = "E_FLOAT_RATE"
E_RATE_PRECISION = "E_RATE_PRECISION"
E_INVALID_TUNING = "E_INVALID_TUNING"
E_TVL_VALIDATOR_UNAVAILABLE = "E_TVL_VALIDATOR_UNAVAILABLE"
E_NOT_ROUNDTRIPPABLE = "E_NOT_ROUNDTRIPPABLE"
E_TVL_VALIDATION_FAILED = "E_TVL_VALIDATION_FAILED"
E_NO_APPROVER_AUTHORITY = "E_NO_APPROVER_AUTHORITY"
E_APPROVER_NOT_AUTHORIZED = "E_APPROVER_NOT_AUTHORIZED"
E_AUTHORITY_CONFIG = "E_AUTHORITY_CONFIG"
E_UNSUPPORTED_IN_RELEASE = "E_UNSUPPORTED_IN_RELEASE"


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


class TvlValidationFailed(TuningError):
    """The TVL linter ran and reported error-severity diagnostics.

    The TVL codes are carried unchanged in ``diagnostics`` (and in the message).
    """

    def __init__(self, diagnostics: Sequence[object]) -> None:
        self.diagnostics = list(diagnostics)
        codes = ", ".join(str(getattr(d, "code", d)) for d in self.diagnostics)
        super().__init__(E_TVL_VALIDATION_FAILED, codes)
