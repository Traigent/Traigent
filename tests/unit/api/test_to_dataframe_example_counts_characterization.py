"""Characterization tests for the example-count columns of ``to_dataframe``.

These pin the observable behaviour of ``OptimizationResult.to_dataframe`` around
the ``successful_examples`` / ``examples_attempted`` / ``example_success_summary``
columns: values, column order, the exact order of metadata reads, and where an
exception stops the walk. They must keep passing unchanged across a refactor of
that block.
"""

from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

from traigent.api.types import (
    OptimizationResult,
    OptimizationStatus,
    TrialResult,
    TrialStatus,
)

BASE_COLUMNS = ["trial_id", "status", "duration", "timestamp"]
TS = datetime(2026, 1, 2, 3, 4, 5)


def _trial(
    trial_id: str = "t1",
    metadata: Any = None,
    metrics: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
) -> TrialResult:
    return TrialResult(
        trial_id=trial_id,
        config=config if config is not None else {"model": "m"},
        metrics=metrics if metrics is not None else {"accuracy": 0.5},
        status=TrialStatus.COMPLETED,
        duration=1.5,
        timestamp=TS,
        metadata=metadata,
    )


def _result(trials: list[Any]) -> OptimizationResult:
    return OptimizationResult(
        trials=trials,
        best_config={},
        best_score=0.5,
        optimization_id="opt",
        duration=1.0,
        convergence_info={},
        status=OptimizationStatus.COMPLETED,
        objectives=["accuracy"],
        algorithm="random",
        timestamp=TS,
    )


def _df(metadata: Any, **kwargs: Any) -> pd.DataFrame:
    return _result([_trial(metadata=metadata, **kwargs)]).to_dataframe()


# ---------------------------------------------------------------- values


def test_pandas_is_available() -> None:
    # Guards against these tests being silently vacuous.
    assert pd.DataFrame is not None


def test_no_example_keys_adds_no_example_columns() -> None:
    df = _df({})
    assert list(df.columns) == [
        *BASE_COLUMNS,
        "model",
        "accuracy",
        "avg_response_time",
        "avg_response_time_ms",
        "total_cost",
    ]


def test_only_successful_examples() -> None:
    df = _df({"successful_examples": 4})
    assert list(df.columns) == [
        *BASE_COLUMNS,
        "model",
        "accuracy",
        "avg_response_time",
        "avg_response_time_ms",
        "successful_examples",
        "total_cost",
    ]
    assert df.loc[0, "successful_examples"] == 4


def test_only_examples_attempted() -> None:
    df = _df({"examples_attempted": 9})
    assert list(df.columns) == [
        *BASE_COLUMNS,
        "model",
        "accuracy",
        "avg_response_time",
        "avg_response_time_ms",
        "examples_attempted",
        "total_cost",
    ]
    assert df.loc[0, "examples_attempted"] == 9


def test_both_keys_add_summary_in_insertion_order() -> None:
    df = _df({"examples_attempted": 5, "successful_examples": 3})
    assert list(df.columns) == [
        *BASE_COLUMNS,
        "model",
        "accuracy",
        "avg_response_time",
        "avg_response_time_ms",
        "successful_examples",
        "examples_attempted",
        "example_success_summary",
        "total_cost",
    ]
    assert df.loc[0, "successful_examples"] == 3
    assert df.loc[0, "examples_attempted"] == 5
    assert df.loc[0, "example_success_summary"] == "3/5 examples succeeded"


@pytest.mark.parametrize(
    ("succ", "att", "summary"),
    [
        (0, 0, "0/0 examples succeeded"),
        (0, 7, "0/7 examples succeeded"),
        (None, None, "None/None examples succeeded"),
        (None, 2, "None/2 examples succeeded"),
        ("", "", "/ examples succeeded"),
        (False, False, "False/False examples succeeded"),
        ("a", "b", "a/b examples succeeded"),
        (1.5, 2.5, "1.5/2.5 examples succeeded"),
    ],
)
def test_falsey_values_still_count_as_present(
    succ: Any, att: Any, summary: str
) -> None:
    row = (
        _result(
            [_trial(metadata={"successful_examples": succ, "examples_attempted": att})]
        )
        .to_dataframe()
        .iloc[0]
    )
    assert row["example_success_summary"] == summary
    assert "successful_examples" in row.index
    assert "examples_attempted" in row.index


@pytest.mark.parametrize("value", [0, None, "", False])
def test_falsey_single_key_is_present_without_summary(value: Any) -> None:
    df = _df({"successful_examples": value})
    assert "successful_examples" in df.columns
    assert "examples_attempted" not in df.columns
    assert "example_success_summary" not in df.columns
    df = _df({"examples_attempted": value})
    assert "examples_attempted" in df.columns
    assert "successful_examples" not in df.columns
    assert "example_success_summary" not in df.columns


def test_metadata_overwrites_metrics_key_without_moving_position() -> None:
    df = _df(
        {"successful_examples": 3, "examples_attempted": 5},
        metrics={"successful_examples": 99, "accuracy": 0.5},
    )
    assert list(df.columns) == [
        *BASE_COLUMNS,
        "model",
        "successful_examples",
        "accuracy",
        "avg_response_time",
        "avg_response_time_ms",
        "examples_attempted",
        "example_success_summary",
        "total_cost",
    ]
    assert df.loc[0, "successful_examples"] == 3


def test_metadata_none_and_empty_behave_alike() -> None:
    a = _df(None)
    b = _df({})
    assert list(a.columns) == list(b.columns)
    assert "example_success_summary" not in a.columns
    assert a.loc[0, "total_cost"] is None or pd.isna(a.loc[0, "total_cost"])


def test_empty_trials_gives_empty_frame() -> None:
    df = _result([]).to_dataframe()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0
    assert list(df.columns) == []


def test_multiple_trials_each_row_independent() -> None:
    result = _result(
        [
            _trial("a", {"successful_examples": 1, "examples_attempted": 2}),
            _trial("b", {}),
            _trial("c", {"examples_attempted": 8}),
        ]
    )
    df = result.to_dataframe()
    assert list(df["trial_id"]) == ["a", "b", "c"]
    assert df.loc[0, "example_success_summary"] == "1/2 examples succeeded"
    assert pd.isna(df.loc[1, "example_success_summary"])
    assert pd.isna(df.loc[2, "example_success_summary"])
    assert df.loc[2, "examples_attempted"] == 8
    assert pd.isna(df.loc[1, "successful_examples"])


# ---------------------------------------------- cost branch left unchanged


def test_total_example_cost_overrides_metrics_total_cost_in_place() -> None:
    df = _df(
        {"total_example_cost": 7.0, "successful_examples": 1, "examples_attempted": 1},
        metrics={"total_cost": 1.5, "accuracy": 0.5},
    )
    assert df.loc[0, "total_cost"] == 7.0
    assert list(df.columns).index("total_cost") < list(df.columns).index(
        "avg_response_time"
    )


def test_metrics_total_cost_kept_without_metadata_cost() -> None:
    df = _df({"successful_examples": 1}, metrics={"total_cost": 1.5})
    assert df.loc[0, "total_cost"] == 1.5


def test_metadata_total_cost_fallback_when_row_lacks_it() -> None:
    df = _df({"total_cost": 2.25, "successful_examples": 1})
    assert df.loc[0, "total_cost"] == 2.25
    assert list(df.columns)[-1] == "total_cost"


def test_total_example_cost_zero_still_wins() -> None:
    df = _df({"total_example_cost": 0}, metrics={"total_cost": 9.0})
    assert df.loc[0, "total_cost"] == 0


def test_response_time_floats_preserved() -> None:
    df = _df(
        {
            "example_results": [{"response_time": 0.5}, {"response_time": 1.5}],
            "successful_examples": 2,
            "examples_attempted": 2,
        }
    )
    assert df.loc[0, "avg_response_time"] == 1.0
    assert df.loc[0, "avg_response_time_ms"] == 1000.0
    assert isinstance(df.loc[0, "avg_response_time"], float)
    assert df.loc[0, "example_success_summary"] == "2/2 examples succeeded"


def test_response_time_none_when_unavailable() -> None:
    df = _df({"successful_examples": 1})
    assert pd.isna(df.loc[0, "avg_response_time"])
    assert pd.isna(df.loc[0, "avg_response_time_ms"])


# --------------------------------------------- duck-typed unbound receiver


def _stub_receiver(trials: list[Any], log: list[Any] | None = None) -> SimpleNamespace:
    calls = log if log is not None else []

    def avg(metadata: Any) -> Any:
        calls.append(("avg", metadata))
        return 2

    def avg_ms(metadata: Any) -> Any:
        calls.append(("avg_ms", metadata))
        return None

    return SimpleNamespace(
        trials=trials,
        _compute_average_response_time=avg,
        _compute_average_response_time_ms=avg_ms,
    )


def test_duck_typed_receiver_without_other_methods() -> None:
    log: list[Any] = []
    meta = {"successful_examples": 3, "examples_attempted": 4}
    receiver = _stub_receiver([_trial(metadata=meta)], log)
    df = OptimizationResult.to_dataframe(receiver)  # type: ignore[arg-type]
    assert log == [("avg", meta), ("avg_ms", meta)]
    row = df.iloc[0]
    assert row["avg_response_time"] == 2.0
    assert isinstance(row["avg_response_time"], float)
    assert pd.isna(row["avg_response_time_ms"])
    assert row["example_success_summary"] == "3/4 examples succeeded"


def test_duck_typed_receiver_gets_raw_metadata_even_when_none() -> None:
    log: list[Any] = []
    receiver = _stub_receiver([_trial(metadata=None)], log)
    OptimizationResult.to_dataframe(receiver)  # type: ignore[arg-type]
    assert log == [("avg", None), ("avg_ms", None)]


def test_duck_typed_receiver_empty_trials() -> None:
    df = OptimizationResult.to_dataframe(_stub_receiver([]))  # type: ignore[arg-type]
    assert len(df) == 0


# ------------------------------------------------------------- spies


class SpyMeta:
    """Mapping stand-in that records every access in order."""

    def __init__(
        self,
        data: dict[str, Any],
        log: list[str],
        fail_at: int | None = None,
        tag: str = "",
    ) -> None:
        self._tag = tag
        self._data = data
        self._log = log
        self._fail_at = fail_at
        self._count = 0

    def _hit(self, event: str) -> None:
        self._log.append(self._tag + event)
        if self._fail_at is not None and self._count == self._fail_at:
            raise RuntimeError(event)
        self._count += 1

    def __len__(self) -> int:
        self._hit("len")
        return len(self._data)

    def __contains__(self, key: object) -> bool:
        self._hit(f"contains:{key}")
        return key in self._data

    def get(self, key: str, default: Any = None) -> Any:
        self._hit(f"get:{key}")
        return self._data.get(key, default)


class SpyTrial:
    def __init__(
        self, meta: Any, log: list[str], trial_id: str = "t", tag: str = ""
    ) -> None:
        self._tag = tag
        self._meta = meta
        self._log = log
        self.trial_id = trial_id
        self.status = TrialStatus.COMPLETED
        self.duration = 1.0
        self.timestamp = TS
        self.config = {"model": "m"}
        self.metrics: dict[str, Any] = {"accuracy": 0.5}

    @property
    def metadata(self) -> Any:
        self._log.append(self._tag + "trial.metadata")
        return self._meta


def _spy_receiver(trials: list[Any], log: list[str]) -> SimpleNamespace:
    def avg(_metadata: Any) -> None:
        log.append("avg")

    def avg_ms(_metadata: Any) -> None:
        log.append("avg_ms")

    return SimpleNamespace(
        trials=trials,
        _compute_average_response_time=avg,
        _compute_average_response_time_ms=avg_ms,
    )


FULL_DATA = {
    "successful_examples": 3,
    "examples_attempted": 5,
    "total_example_cost": 1.0,
}

FULL_LOG = [
    "trial.metadata",
    "avg",
    "trial.metadata",
    "avg_ms",
    "trial.metadata",
    "len",
    "contains:successful_examples",
    "get:successful_examples",
    "contains:examples_attempted",
    "get:examples_attempted",
    "contains:successful_examples",
    "contains:examples_attempted",
    "get:successful_examples",
    "get:examples_attempted",
    "contains:total_example_cost",
    "get:total_example_cost",
]


def _run_spy(data: dict[str, Any], fail_at: int | None = None) -> list[str]:
    log: list[str] = []
    trial = SpyTrial(SpyMeta(data, log, fail_at), log)
    receiver = _spy_receiver([trial], log)
    try:
        OptimizationResult.to_dataframe(receiver)  # type: ignore[arg-type]
    except RuntimeError:
        pass
    return log


def test_spy_full_access_order() -> None:
    assert _run_spy(dict(FULL_DATA)) == FULL_LOG


def test_spy_only_successful_short_circuits_second_membership() -> None:
    data = {"successful_examples": 3}
    assert _run_spy(data) == [
        "trial.metadata",
        "avg",
        "trial.metadata",
        "avg_ms",
        "trial.metadata",
        "len",
        "contains:successful_examples",
        "get:successful_examples",
        "contains:examples_attempted",
        "contains:successful_examples",
        "contains:examples_attempted",
        "contains:total_example_cost",
        "get:total_cost",
    ]


def test_spy_only_attempted_short_circuits_after_first_membership() -> None:
    data = {"examples_attempted": 5}
    assert _run_spy(data) == [
        "trial.metadata",
        "avg",
        "trial.metadata",
        "avg_ms",
        "trial.metadata",
        "len",
        "contains:successful_examples",
        "contains:examples_attempted",
        "get:examples_attempted",
        "contains:successful_examples",
        "contains:total_example_cost",
        "get:total_cost",
    ]


def test_spy_cost_fallback_follows_count_block_unchanged() -> None:
    data = {"successful_examples": 1, "examples_attempted": 1, "total_cost": 3.0}
    log = _run_spy(data)
    assert log[-2:] == ["contains:total_example_cost", "get:total_cost"]
    assert log.count("trial.metadata") == 3


def test_spy_empty_mapping_is_replaced_by_plain_dict() -> None:
    log = _run_spy({})
    assert log == [
        "trial.metadata",
        "avg",
        "trial.metadata",
        "avg_ms",
        "trial.metadata",
        "len",
    ]


MAPPING_START = FULL_LOG.index("len")


@pytest.mark.parametrize("fail_at", range(len(FULL_LOG) - MAPPING_START))
def test_exception_halts_at_the_old_point_before_cost_access(fail_at: int) -> None:
    # fail_at counts mapping events only; the walk must stop right after the
    # failing call, so nothing later (including the cost reads) is touched.
    expected = FULL_LOG[: MAPPING_START + fail_at + 1]
    assert _run_spy(dict(FULL_DATA), fail_at=fail_at) == expected


@pytest.mark.parametrize(
    ("boom_key", "last_events"),
    [
        (
            "successful_examples",
            ["contains:examples_attempted", "get:successful_examples"],
        ),
        ("examples_attempted", ["get:successful_examples", "get:examples_attempted"]),
    ],
)
def test_format_failure_halts_before_cost_access(
    boom_key: str, last_events: list[str]
) -> None:
    class Boom:
        def __format__(self, spec: str) -> str:
            raise ValueError("format")

    log: list[str] = []
    data = {
        "successful_examples": 1,
        "examples_attempted": 5,
        "total_example_cost": 1,
        boom_key: Boom(),
    }
    trial = SpyTrial(SpyMeta(data, log), log)
    with pytest.raises(ValueError, match="format"):
        OptimizationResult.to_dataframe(_spy_receiver([trial], log))  # type: ignore[arg-type]
    assert log[-2:] == last_events
    assert "contains:total_example_cost" not in log


def test_non_mapping_truthy_metadata_raises_attribute_error_only_when_key_present() -> (
    None
):
    class Odd:
        def __bool__(self) -> bool:
            return True

        def __contains__(self, key: object) -> bool:
            return key == "successful_examples"

    log: list[str] = []
    trial = SpyTrial(Odd(), log)
    with pytest.raises(AttributeError):
        OptimizationResult.to_dataframe(_spy_receiver([trial], log))  # type: ignore[arg-type]
    assert log == [
        "trial.metadata",
        "avg",
        "trial.metadata",
        "avg_ms",
        "trial.metadata",
    ]


def test_later_trial_error_leaves_following_trials_untouched() -> None:
    log: list[str] = []
    trials = [
        SpyTrial(SpyMeta(dict(FULL_DATA), log, tag="a:"), log, "a", tag="a:"),
        SpyTrial(
            SpyMeta(dict(FULL_DATA), log, fail_at=2, tag="b:"), log, "b", tag="b:"
        ),
        SpyTrial(SpyMeta(dict(FULL_DATA), log, tag="c:"), log, "c", tag="c:"),
    ]
    with pytest.raises(RuntimeError):
        OptimizationResult.to_dataframe(_spy_receiver(trials, log))  # type: ignore[arg-type]
    first = [e[2:] for e in log if e.startswith("a:")]
    assert first == [e for e in FULL_LOG if e not in ("avg", "avg_ms")]
    assert [e for e in log if e.startswith("b:")][-1] == "b:get:successful_examples"
    assert not [e for e in log if e.startswith("c:")]
    # stubs ran for trials a and b only (two calls each), never for c
    assert log.count("avg") == 2
    assert log.count("avg_ms") == 2
