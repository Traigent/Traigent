"""Small adaptive batch sizes must be able to grow (#3095, #3119)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from traigent.cloud.sessions import SessionManager
from traigent.invokers.batch import BatchInvoker
from traigent.utils.batch_processing import AdaptiveBatchSizer


@pytest.mark.parametrize("start", [1, 2, 3, 4])
def test_adaptive_batch_sizer_grows_from_small_sizes(start: int) -> None:
    sizer = AdaptiveBatchSizer(initial_batch_size=start, max_batch_size=start * 4)
    for i in range(1, 6):
        sizer.update_performance(
            batch_size=sizer.current_batch_size,
            throughput=float(i),
            memory_usage_mb=1.0,
            error_rate=0.0,
        )
    assert sizer.current_batch_size > start


@pytest.mark.parametrize("start", [1, 2, 3, 4])
def test_batch_invoker_adaptive_size_grows_from_small_sizes(start: int) -> None:
    invoker = BatchInvoker(batch_size=start, adaptive_batching=True)
    for _ in range(3):
        invoker._update_adaptive_batch_size(1.0, start)  # baseline rate
    for _ in range(10):
        invoker._update_adaptive_batch_size(0.1, start)  # much faster
    assert invoker.get_optimal_batch_size() > start


@pytest.mark.parametrize("dataset_size", [1, 2, 3, 4, 9])
@pytest.mark.parametrize("completed", [3, 12])
def test_subset_suggestion_never_empty(dataset_size: int, completed: int) -> None:
    session = SimpleNamespace(completed_trials=completed, optimization_strategy={})
    result = SessionManager._suggest_subset_indices(
        None,
        session,
        [],
        dataset_size,  # type: ignore[arg-type]
    )
    assert 1 <= len(result.indices) <= dataset_size
