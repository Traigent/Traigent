"""Shared guard for the Continuous Tuning tests: they need the TVL 1.7 toolchain.

Without it the tests skip (collection never imports TVL).  CI that must not lose
this coverage sets ``TRAIGENT_TVL_REQUIRED=1``: a missing TVL 1.7 then FAILS.
"""

from __future__ import annotations

import os

import pytest

from traigent.tuning import validator

REQUIRED_ENV = "TRAIGENT_TVL_REQUIRED"


def tvl_required() -> bool:
    return os.environ.get(REQUIRED_ENV, "").strip().lower() in {"1", "true", "yes"}


def check_toolchain() -> None:
    """Skip (or, with ``TRAIGENT_TVL_REQUIRED=1``, fail) when TVL 1.7 is missing."""
    if validator.validator_available():
        return
    reason = "TVL toolchain with 1.7 support not installed"
    if tvl_required():
        pytest.fail(f"{REQUIRED_ENV}=1 but {reason}", pytrace=False)
    pytest.skip(reason)


@pytest.fixture(autouse=True)
def _tvl_toolchain() -> None:
    check_toolchain()
