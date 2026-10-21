"""Trust-at-construction classes trust only validated operands (F1r)."""

from __future__ import annotations

import pytest

from traigent.core import best_config_diagnostics as diag
from traigent.core.best_config_runtime import (
    CloudBestConfigHeadRefusedError,
    CloudBestConfigHeadStaleError,
    CloudBestConfigStaleHeadError,
    HeadRefusalCode,
)

CANARY = "zq-acme-private-canary-91"


class _ForeignCode:
    value = CANARY

    def __str__(self) -> str:
        return CANARY

    __repr__ = __str__


class _LeakyInt(int):
    def __str__(self) -> str:
        return CANARY

    __repr__ = __str__

    def __format__(self, spec: str) -> str:
        return CANARY


def _renderings(exc: BaseException) -> str:
    return f"{exc!s}|{exc!r}|{exc.args!r}|{getattr(exc, 'message', '')}"


def test_foreign_code_object_is_untrusted_and_not_rendered() -> None:
    exc = CloudBestConfigHeadRefusedError(_ForeignCode())  # type: ignore[arg-type]
    assert not diag.is_safe(exc)
    assert CANARY not in _renderings(exc)


def test_str_lookalike_code_is_untrusted() -> None:
    exc = CloudBestConfigHeadRefusedError(CANARY)  # type: ignore[arg-type]
    assert not diag.is_safe(exc)
    assert CANARY not in _renderings(exc)


@pytest.mark.parametrize("code", list(HeadRefusalCode))
def test_real_codes_stay_trusted(code: HeadRefusalCode) -> None:
    exc = CloudBestConfigHeadRefusedError(code)
    assert diag.is_safe(exc)
    assert code.value in str(exc)


def test_head_stale_subclass_stays_trusted() -> None:
    assert diag.is_safe(CloudBestConfigHeadStaleError())


@pytest.mark.parametrize(
    "bad",
    [True, False, _LeakyInt(3), 3.0, "3", None, object(), _ForeignCode()],
)
@pytest.mark.parametrize("which", ["current", "expected"])
def test_non_int_generation_is_untrusted_and_not_rendered(bad, which) -> None:
    kwargs = {"current_generation": 1, "expected_generation": 2}
    kwargs[f"{which}_generation"] = bad
    exc = CloudBestConfigStaleHeadError(**kwargs)
    assert not diag.is_safe(exc)
    assert CANARY not in _renderings(exc)


def test_int_generations_stay_trusted() -> None:
    exc = CloudBestConfigStaleHeadError(current_generation=5, expected_generation=4)
    assert diag.is_safe(exc)
    assert "5" in str(exc) and "4" in str(exc)


def test_fs_guard_conversion_carries_no_context_or_cause() -> None:
    with pytest.raises(Exception) as info:
        with diag.fs_guard("Failed to read"):
            raise UnicodeDecodeError("utf-8", CANARY.encode(), 0, 1, CANARY)
    exc = info.value
    assert exc.__context__ is None
    assert exc.__cause__ is None
    assert CANARY not in _renderings(exc)


def test_fs_guard_oserror_carries_no_context_or_cause() -> None:
    with pytest.raises(Exception) as info:
        with diag.fs_guard("Failed to read"):
            raise OSError(2, CANARY)
    assert info.value.__context__ is None
    assert info.value.__cause__ is None
