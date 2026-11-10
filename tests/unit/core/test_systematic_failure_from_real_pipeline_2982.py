"""#2982: agent errors must not abort a run as provider failures.

Builds the error exactly as the evaluator does (``Function call failed: <text>``
with no exception type name) and runs it through the run-abort consumers.
"""

from __future__ import annotations

import copy
import pickle
import sqlite3

import pytest

from traigent.core.exception_handler import (
    VendorErrorCategory,
    classify_systematic_provider_failure,
)
from traigent.core.trial_result_factory import _provider_failure_summary
from traigent.evaluators.base import Dataset, EvaluationExample, SimpleScoringEvaluator
from traigent.evaluators.local import LocalEvaluator


class _ProviderQuotaError(Exception):
    """Stand-in for a provider SDK status error (module root is a provider SDK)."""

    __module__ = "openai"

    def __init__(self) -> None:
        super().__init__("Error code: 429 - insufficient_quota")
        self.status_code = 429


def _raise_sql(_q: str) -> str:
    return sqlite3.connect(":memory:").execute("select api_key from authentication")


def _raise_value(_q: str) -> str:
    raise ValueError("authentication header missing in tool schema")


def _raise_state(_q: str) -> str:
    raise RuntimeError("unauthorized transition: draft -> published")


def _raise_forbidden(_q: str) -> str:
    raise KeyError("user 7 forbidden from editing row")


def _raise_quota_table(_q: str) -> str:
    raise RuntimeError("quota table has no row for tenant 3")


def _raise_provider_quota(_q: str) -> str:
    raise _ProviderQuotaError()


async def _run(func, detailed: bool):
    dataset = Dataset([EvaluationExample({"_q": f"q{i}"}, "x") for i in range(3)])
    evaluator = LocalEvaluator(metrics=["accuracy"], detailed_metrics=detailed)
    return await evaluator.evaluate(func, {}, dataset)


@pytest.mark.asyncio
@pytest.mark.parametrize("detailed", [False, True])
@pytest.mark.parametrize(
    "func",
    [_raise_sql, _raise_value, _raise_state, _raise_forbidden, _raise_quota_table],
)
async def test_agent_error_text_does_not_become_a_provider_failure(
    func, detailed: bool
) -> None:
    result = await _run(func, detailed)
    assert result.errors and all(result.errors)
    assert _provider_failure_summary(result, 3) is None
    for error in result.errors:
        assert classify_systematic_provider_failure(error) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("detailed", [False, True])
async def test_real_provider_failure_still_aborts(detailed: bool) -> None:
    result = await _run(_raise_provider_quota, detailed)
    summary = _provider_failure_summary(result, 3)
    assert summary is not None
    assert summary["category"] == VendorErrorCategory.QUOTA_EXHAUSTED.value


@pytest.mark.asyncio
async def test_verdict_survives_copy_and_pickle() -> None:
    result = await _run(_raise_value, False)
    error = result.errors[0]
    for clone in (copy.deepcopy(error), pickle.loads(pickle.dumps(error))):
        assert classify_systematic_provider_failure(clone) is None


def _raise_user_provider_words(_q: str) -> str:
    raise RuntimeError("Error code: 429 - insufficient_quota invalid_api_key")


def _raise_user_value_provider_words(_q: str) -> str:
    raise ValueError("insufficient_quota: Error code: 401 invalid_api_key")


class _UserCodedError(Exception):
    """A user exception with its own error code and provider-looking words."""

    def __init__(self) -> None:
        super().__init__("insufficient_quota invalid_api_key Error code: 429")
        self.code = "E42"
        self.response = object()


class _NonSdkQuotaCodeError(Exception):
    """Non-SDK exception carrying a known provider error-code value."""

    def __init__(self) -> None:
        super().__init__("billing refused")
        self.code = "insufficient_quota"


class _NonSdkStatusError(Exception):
    def __init__(self, status: int, text: str = "x") -> None:
        super().__init__(text)
        self.status_code = status


def _raise_user_coded(_q: str) -> str:
    raise _UserCodedError()


def _raise_quota_code(_q: str) -> str:
    raise _NonSdkQuotaCodeError()


def test_non_sdk_exception_classified_from_structure_only() -> None:
    from traigent.core.exception_handler import classify_vendor_error

    assert (
        classify_vendor_error(_NonSdkStatusError(429), provider_text_only=True)
        == VendorErrorCategory.RATE_LIMIT
    )
    # words never decide, even next to a structured status
    assert (
        classify_vendor_error(
            _NonSdkStatusError(429, "insufficient_quota"), provider_text_only=True
        )
        == VendorErrorCategory.RATE_LIMIT
    )
    assert classify_vendor_error(_UserCodedError(), provider_text_only=True) is None
    assert (
        classify_vendor_error(_NonSdkQuotaCodeError(), provider_text_only=True)
        == VendorErrorCategory.QUOTA_EXHAUSTED
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("detailed", [False, True])
async def test_user_exception_with_own_code_and_provider_words_does_not_abort(
    detailed: bool,
) -> None:
    result = await _run(_raise_user_coded, detailed)
    assert _provider_failure_summary(result, 3) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("detailed", [False, True])
async def test_non_sdk_known_provider_code_still_aborts(detailed: bool) -> None:
    result = await _run(_raise_quota_code, detailed)
    summary = _provider_failure_summary(result, 3)
    assert summary is not None
    assert summary["category"] == VendorErrorCategory.QUOTA_EXHAUSTED.value


@pytest.mark.asyncio
@pytest.mark.parametrize("detailed", [False, True])
@pytest.mark.parametrize(
    "func", [_raise_user_provider_words, _raise_user_value_provider_words]
)
async def test_user_error_with_provider_marker_words_does_not_abort(
    func, detailed: bool
) -> None:
    result = await _run(func, detailed)
    assert _provider_failure_summary(result, 3) is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "func", [_raise_value, _raise_state, _raise_user_provider_words]
)
async def test_simple_scoring_evaluator_user_errors_do_not_abort(func) -> None:
    dataset = Dataset([EvaluationExample({"_q": f"q{i}"}, "x") for i in range(3)])
    evaluator = SimpleScoringEvaluator(
        scoring_function=lambda output, expected: 1.0, metrics=["accuracy"]
    )
    result = await evaluator.evaluate(func, {}, dataset)
    assert result.errors and all(result.errors)
    assert _provider_failure_summary(result, 3) is None


@pytest.mark.asyncio
async def test_simple_scoring_evaluator_provider_failure_still_aborts() -> None:
    dataset = Dataset([EvaluationExample({"_q": f"q{i}"}, "x") for i in range(3)])
    evaluator = SimpleScoringEvaluator(
        scoring_function=lambda output, expected: 1.0, metrics=["accuracy"]
    )
    result = await evaluator.evaluate(_raise_provider_quota, {}, dataset)
    summary = _provider_failure_summary(result, 3)
    assert summary is not None
    assert summary["category"] == VendorErrorCategory.QUOTA_EXHAUSTED.value
