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


def test_hooks_cost_constraint_accepts_numpy_scalars() -> None:
    """numpy scalars are real numbers; only bool, NaN and inf are rejected."""
    np = pytest.importorskip("numpy")

    c = CostConstraints(
        max_cost_per_query=np.float32(0.1), max_monthly_budget=np.int64(50)
    )
    assert float(c.max_cost_per_query) == pytest.approx(0.1)
    with pytest.raises(ValueError, match="max_cost_per_query"):
        CostConstraints(max_cost_per_query=np.float64("nan"))
    with pytest.raises(ValueError, match="max_cost_per_query"):
        CostConstraints(max_cost_per_query=True)


@pytest.mark.parametrize(
    "build",
    [
        lambda v: LocalInvoker(timeout=v),
        lambda v: BatchInvoker(batch_timeout=v),
        lambda v: StreamingInvoker(chunk_timeout=v),
    ],
    ids=["timeout", "batch_timeout", "chunk_timeout"],
)
def test_invoker_timeouts_accept_numpy_and_reject_bool(build) -> None:
    np = pytest.importorskip("numpy")
    build(np.int64(2))
    build(np.float32(1.5))
    with pytest.raises(InvocationError):
        build(True)


def test_validate_number_accepts_numpy_scalars() -> None:
    np = pytest.importorskip("numpy")
    from traigent.utils.validation import Validators

    assert Validators.validate_number(np.float32(0.5), "x", 0, 1).is_valid
    assert Validators.validate_number(np.int64(3), "x", 0, 5).is_valid
    assert not Validators.validate_number(np.float64("nan"), "x").is_valid
    assert not Validators.validate_number(True, "x").is_valid
