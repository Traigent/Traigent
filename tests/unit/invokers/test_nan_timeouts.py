"""NaN timeouts and NaN cost/coverage limits must be rejected, not ignored."""

from __future__ import annotations

import pytest

from traigent.cloud.validators import validate_comparability_metadata
from traigent.hooks.config import CostConstraints, HooksConfig
from traigent.invokers.batch import BatchInvoker
from traigent.invokers.local import LocalInvoker
from traigent.invokers.streaming import StreamingInvoker
from traigent.utils.exceptions import InvocationError

NAN = float("nan")


def test_local_invoker_rejects_nan_timeout() -> None:
    with pytest.raises(InvocationError):
        LocalInvoker(timeout=NAN)


def test_batch_invoker_rejects_nan_batch_timeout() -> None:
    with pytest.raises(InvocationError):
        BatchInvoker(batch_timeout=NAN)


def test_streaming_invoker_rejects_nan_chunk_timeout() -> None:
    with pytest.raises(InvocationError):
        StreamingInvoker(chunk_timeout=NAN)


def test_comparability_coverage_ratio_nan_rejected() -> None:
    with pytest.raises(ValueError, match="coverage_ratio"):
        validate_comparability_metadata({"coverage_ratio": NAN})


def test_hooks_cost_constraint_nan_rejected() -> None:
    with pytest.raises(ValueError, match="max_cost_per_query"):
        CostConstraints(max_cost_per_query=NAN)
    with pytest.raises(ValueError, match="max_cost_per_query"):
        HooksConfig.from_dict({"constraints": {"cost": {"max_cost_per_query": NAN}}})
