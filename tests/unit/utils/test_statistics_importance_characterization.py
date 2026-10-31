"""Characterize StatisticsCallback parameter importance before extraction."""

from __future__ import annotations

from collections import Counter
from typing import Any

import pytest

from traigent.utils.callbacks import StatisticsCallback


class _CountingStats(dict[str, Any]):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.reads: Counter[str] = Counter()

    def __getitem__(self, key: str) -> Any:
        self.reads[key] += 1
        return super().__getitem__(key)


class _CountingList(list[Any]):
    def __init__(self, values: list[Any]) -> None:
        super().__init__(values)
        self.length_reads = 0
        self.index_reads: list[int] = []

    def __len__(self) -> int:
        self.length_reads += 1
        return super().__len__()

    def __getitem__(self, index: int) -> Any:
        self.index_reads.append(index)
        return super().__getitem__(index)


def test_parameter_importance_preserves_ordered_grouping_and_reads() -> None:
    callback = StatisticsCallback()
    scores = _CountingList([0.1, None, 0.9, 0.8])
    stats = _CountingStats(
        parameter_values={"model": ["a", "a", "b", "b"]},
        scores_by_trial=scores,
    )
    callback.stats = stats

    assert callback.get_parameter_importance() == {"model": 1.0}
    assert stats.reads == {"parameter_values": 1, "scores_by_trial": 11}
    assert scores.length_reads == 4
    assert scores.index_reads == [0, 0, 1, 2, 2, 3, 3]


def test_parameter_importance_preserves_zero_and_empty_results() -> None:
    callback = StatisticsCallback()
    callback.stats["parameter_values"] = {"single": ["only"], "same": [1, 1]}
    callback.stats["scores_by_trial"] = [0.7, 0.8]

    assert callback.get_parameter_importance() == {"single": 0.0, "same": 0.0}

    callback.stats["parameter_values"] = {}
    assert callback.get_parameter_importance() == {}


def test_parameter_importance_preserves_unhashable_value_exception_timing() -> None:
    callback = StatisticsCallback()
    scores = _CountingList([0.5])
    stats = _CountingStats(
        parameter_values={"unhashable": [[], "other"]},
        scores_by_trial=scores,
    )
    callback.stats = stats

    with pytest.raises(TypeError):
        callback.get_parameter_importance()

    assert stats.reads == {"parameter_values": 1, "scores_by_trial": 2}
    assert scores.length_reads == 1
    assert scores.index_reads == [0]
