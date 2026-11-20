"""Shared guard for the Continuous Tuning tests: they need the TVL 1.7 toolchain.

Without it the tests skip (collection never imports TVL).  CI that must not lose
this coverage sets ``TRAIGENT_TVL_REQUIRED=1``: a missing TVL 1.7 then FAILS.
"""

from __future__ import annotations

import os
from collections.abc import Iterator

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


@pytest.fixture(autouse=True)
def _agent_code_version(monkeypatch: pytest.MonkeyPatch) -> None:
    """Promotion evidence needs an agent code version (no default exists).

    Most tests are not about the code version, so they pin one; the tests that
    are delete or change this variable themselves.
    """
    monkeypatch.setenv("TRAIGENT_AGENT_CODE_VERSION", "test-code-v1")


@pytest.fixture(autouse=True)
def _dataset_root(monkeypatch: pytest.MonkeyPatch, tmp_path: object) -> None:
    """The optimizer's loader only reads datasets under this root."""
    monkeypatch.setenv("TRAIGENT_DATASET_ROOT", str(tmp_path))


@pytest.fixture(autouse=True)
def _explicit_non_production_environment() -> Iterator[None]:
    """``local-trusted`` is refused unless the environment is explicitly non-production.

    Tests that exercise it run as ``development``; the tests that are about
    production or an unset environment change this themselves.  It uses its own
    patcher so a test calling ``monkeypatch.undo()`` does not drop the environment.
    """
    patcher = pytest.MonkeyPatch()
    for name in ("TRAIGENT_ENVIRONMENT", "TRAIGENT_ENV"):
        patcher.delenv(name, raising=False)
    patcher.setenv("ENVIRONMENT", "development")
    try:
        yield
    finally:
        patcher.undo()
