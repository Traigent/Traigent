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
from traigent.evaluators.base import Dataset, EvaluationExample
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
