"""Characterization tests for ``traigent.api.types`` hot spots.

These pin the CURRENT observable behaviour of ``_json_safe_trial_value``,
``OptimizationResult._objective_orientation``, ``_aggregated_mean_metrics``,
``best_metrics`` and ``to_dataframe`` so that complexity-reducing refactors can
be shown to be behaviour-preserving. They document what the code does today,
including quirks; they are not a statement that every quirk is desirable.
"""

from __future__ import annotations

import json
import math
from datetime import datetime
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from traigent.api.types import (
    OptimizationResult,
    OptimizationStatus,
    TrialResult,
    TrialStatus,
    _json_safe_trial_value,
)

TS = datetime(2026, 1, 2, 3, 4, 5)


def _trial(
    trial_id: str = "t1",
    config: dict[str, Any] | None = None,
    metrics: dict[str, Any] | None = None,
    *,
    status: TrialStatus = TrialStatus.COMPLETED,
    metadata: dict[str, Any] | None = None,
    duration: float = 0.5,
) -> TrialResult:
    return TrialResult(
        trial_id=trial_id,
        config={"a": 1} if config is None else config,
        metrics={} if metrics is None else metrics,
        status=status,
        duration=duration,
        timestamp=TS,
        metadata={} if metadata is None else metadata,
    )


def _result(
    trials: list[TrialResult],
    *,
    best_config: dict[str, Any] | None = None,
    best_score: float | None = None,
    objectives: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> OptimizationResult:
    return OptimizationResult(
        trials=trials,
        best_config=best_config if best_config is not None else {},
        best_score=best_score,
        optimization_id="opt",
        duration=1.0,
        convergence_info={},
        status=OptimizationStatus.COMPLETED,
        objectives=objectives if objectives is not None else [],
        algorithm="random",
        timestamp=TS,
        metadata=metadata if metadata is not None else {},
    )


# --------------------------------------------------------------------------
# _json_safe_trial_value
# --------------------------------------------------------------------------


class _ToDictWithFormat:
    def to_dict(self, datetime_format: str = "iso") -> dict[str, Any]:
        return {"fmt": datetime_format, "when": TS}


class _ToDictNoArgs:
    def to_dict(self) -> dict[str, Any]:
        return {"plain": 1}


class _ToDictAlwaysTypeError:
    def to_dict(self, *args: Any, **kwargs: Any) -> Any:
        raise TypeError("nope")

    def __str__(self) -> str:
        return "ToDictAlwaysTypeError!"


class _ToDictNotCallable:
    to_dict = "not callable"

    def __str__(self) -> str:
        return "ToDictNotCallable!"


class _WithItem:
    def item(self) -> int:
        return 42


class _ItemRaisesTypeError:
    def item(self) -> int:
        raise TypeError("bad")

    def __str__(self) -> str:
        return "ItemRaisesTypeError!"


class _ItemRaisesValueError(_ItemRaisesTypeError):
    def item(self) -> int:
        raise ValueError("bad")


class _ItemNotCallable:
    item = 5

    def __str__(self) -> str:
        return "ItemNotCallable!"


class _Opaque:
    def __str__(self) -> str:
        return "opaque-object"


class TestJsonSafeTrialValue:
    @pytest.mark.parametrize("fmt", ["iso", "epoch"])
    def test_none(self, fmt):
        assert _json_safe_trial_value(None, datetime_format=fmt) is None

    @pytest.mark.parametrize(
        "value",
        ["text", "", "üñí", 0, 7, -3, 1.5, float("inf"), True, False],
    )
    def test_primitives_returned_as_is(self, value):
        out = _json_safe_trial_value(value, datetime_format="iso")
        assert out == value
        assert type(out) is type(value)

    def test_nan_passthrough(self):
        assert math.isnan(_json_safe_trial_value(float("nan"), datetime_format="iso"))

    def test_datetime_iso_and_epoch(self):
        assert _json_safe_trial_value(TS, datetime_format="iso") == TS.isoformat()
        assert _json_safe_trial_value(TS, datetime_format="epoch") == TS.timestamp()

    def test_datetime_invalid_format_raises(self):
        with pytest.raises(ValueError) as exc:
            _json_safe_trial_value(TS, datetime_format="bogus")  # type: ignore[arg-type]
        assert str(exc.value) == "datetime_format must be 'iso' or 'epoch', got 'bogus'"

    def test_dict_recurses_and_keeps_key_order(self):
        out = _json_safe_trial_value(
            {"z": TS, "a": {"inner": (1, 2)}, 3: None}, datetime_format="iso"
        )
        assert out == {"z": TS.isoformat(), "a": {"inner": [1, 2]}, 3: None}
        assert list(out) == ["z", "a", 3]

    def test_list_tuple_set_become_lists(self):
        assert _json_safe_trial_value([1, (2,), {3}], datetime_format="iso") == [
            1,
            [2],
            [3],
        ]
        assert _json_safe_trial_value((TS,), datetime_format="epoch") == [
            TS.timestamp()
        ]
        assert _json_safe_trial_value(set(), datetime_format="iso") == []

    def test_to_dict_with_datetime_format_kwarg(self):
        out = _json_safe_trial_value(_ToDictWithFormat(), datetime_format="epoch")
        assert out == {"fmt": "epoch", "when": TS.timestamp()}

    def test_to_dict_without_kwarg_falls_back_to_bare_call(self):
        out = _json_safe_trial_value(_ToDictNoArgs(), datetime_format="iso")
        assert out == {"plain": 1}

    def test_to_dict_type_error_both_ways_falls_through_to_str(self):
        out = _json_safe_trial_value(_ToDictAlwaysTypeError(), datetime_format="iso")
        assert out == "ToDictAlwaysTypeError!"

    def test_non_callable_to_dict_ignored(self):
        out = _json_safe_trial_value(_ToDictNotCallable(), datetime_format="iso")
        assert out == "ToDictNotCallable!"

    def test_item_callable_is_used(self):
        assert _json_safe_trial_value(_WithItem(), datetime_format="iso") == 42

    @pytest.mark.parametrize("cls", [_ItemRaisesTypeError, _ItemRaisesValueError])
    def test_item_raising_type_or_value_error_falls_through(self, cls):
        assert _json_safe_trial_value(cls(), datetime_format="iso") == (
            "ItemRaisesTypeError!"
        )

    def test_item_not_callable_falls_to_str(self):
        out = _json_safe_trial_value(_ItemNotCallable(), datetime_format="iso")
        assert out == "ItemNotCallable!"

    def test_unserializable_object_stringified(self):
        assert (
            _json_safe_trial_value(_Opaque(), datetime_format="iso") == "opaque-object"
        )
        assert _json_safe_trial_value(b"raw", datetime_format="iso") == "b'raw'"

    def test_circular_value_stringified_via_value_error(self):
        class Box:
            pass

        box = Box()
        box.self = box  # type: ignore[attr-defined]
        out = _json_safe_trial_value(box, datetime_format="iso")
        assert out == str(box)

    def test_result_is_json_dumpable_for_mixed_payload(self):
        payload = {"o": _Opaque(), "t": TS, "n": [None, _WithItem()]}
        out = _json_safe_trial_value(payload, datetime_format="iso")
        json.dumps(out)
        assert out == {"o": "opaque-object", "t": TS.isoformat(), "n": [None, 42]}


# --------------------------------------------------------------------------
# _objective_orientation
# --------------------------------------------------------------------------


class TestObjectiveOrientation:
    def test_metadata_none_uses_canonical_defaults(self):
        res = _result([], objectives=["accuracy"])
        res.metadata = None  # type: ignore[assignment]
        assert res._objective_orientation("accuracy") == "maximize"
        assert res._objective_orientation("cost") == "minimize"

    def test_unknown_name_without_declaration_raises(self):
        res = _result([])
        with pytest.raises(ValueError) as exc:
            res._objective_orientation("my_custom")
        assert str(exc.value).startswith(
            "Objective 'my_custom' has no declared orientation"
        )

    def test_schema_dict_with_dict_definitions(self):
        res = _result(
            [],
            metadata={
                "objective_schema": {
                    "objectives": [
                        {"name": "other", "orientation": "minimize"},
                        {"name": "my_custom", "orientation": "minimize"},
                    ]
                }
            },
        )
        assert res._objective_orientation("my_custom") == "minimize"
        assert res._objective_orientation("other") == "minimize"

    def test_schema_dict_wins_over_persisted_orientations(self):
        res = _result(
            [],
            metadata={
                "objective_schema": {
                    "objectives": [{"name": "my_custom", "orientation": "maximize"}]
                },
                "objective_orientations": {"my_custom": "minimize"},
            },
        )
        assert res._objective_orientation("my_custom") == "maximize"

    def test_schema_object_with_attribute_definitions(self):
        schema = SimpleNamespace(
            objectives=[
                SimpleNamespace(name="x", orientation="maximize"),
                SimpleNamespace(name="y", orientation="minimize"),
            ]
        )
        res = _result([], metadata={"objective_schema": schema})
        assert res._objective_orientation("y") == "minimize"
        assert res._objective_orientation("x") == "maximize"

    def test_schema_object_without_objectives_attribute(self):
        res = _result([], metadata={"objective_schema": object()})
        assert res._objective_orientation("accuracy") == "maximize"

    def test_schema_dict_without_objectives_key(self):
        res = _result([], metadata={"objective_schema": {}})
        assert res._objective_orientation("latency") == "minimize"

    def test_schema_objectives_none_is_treated_as_empty(self):
        res = _result([], metadata={"objective_schema": {"objectives": None}})
        assert res._objective_orientation("accuracy") == "maximize"

    def test_dict_definition_missing_name_never_matches(self):
        res = _result(
            [],
            metadata={
                "objective_schema": {"objectives": [{"orientation": "minimize"}]},
            },
        )
        assert res._objective_orientation("accuracy") == "maximize"

    def test_band_orientation_raises_with_exact_message(self):
        res = _result(
            [],
            metadata={
                "objective_schema": {
                    "objectives": [{"name": "tone", "orientation": "band"}]
                }
            },
        )
        with pytest.raises(ValueError) as exc:
            res._objective_orientation("tone")
        assert str(exc.value) == (
            "Objective 'tone' is target-banded and has no scalar "
            "maximize/minimize direction."
        )

    def test_band_for_other_objective_does_not_interfere(self):
        res = _result(
            [],
            metadata={
                "objective_schema": {
                    "objectives": [{"name": "tone", "orientation": "band"}]
                }
            },
        )
        assert res._objective_orientation("accuracy") == "maximize"

    @pytest.mark.parametrize("bad", ["sideways", None])
    def test_invalid_declared_orientation_raises(self, bad):
        res = _result(
            [],
            metadata={
                "objective_schema": {"objectives": [{"name": "q", "orientation": bad}]}
            },
        )
        with pytest.raises(ValueError) as exc:
            res._objective_orientation("q")
        assert str(exc.value) == (
            "Orientation for objective 'q' must be 'maximize' or "
            f"'minimize'; got {str(bad)!r}."
        )

    def test_dict_definition_without_orientation_key_raises(self):
        res = _result(
            [],
            metadata={"objective_schema": {"objectives": [{"name": "q"}]}},
        )
        with pytest.raises(ValueError, match="Orientation for objective 'q'"):
            res._objective_orientation("q")

    def test_first_matching_definition_wins(self):
        res = _result(
            [],
            metadata={
                "objective_schema": {
                    "objectives": [
                        {"name": "q", "orientation": "minimize"},
                        {"name": "q", "orientation": "maximize"},
                    ]
                }
            },
        )
        assert res._objective_orientation("q") == "minimize"

    @pytest.mark.parametrize(
        ("declared", "expected"),
        [
            (True, "maximize"),
            (False, "minimize"),
            ("minimize", "minimize"),
            ("maximize", "maximize"),
        ],
    )
    def test_persisted_orientations(self, declared, expected):
        res = _result([], metadata={"objective_orientations": {"q": declared}})
        assert res._objective_orientation("q") == expected

    def test_persisted_invalid_value_raises(self):
        res = _result([], metadata={"objective_orientations": {"q": "up"}})
        with pytest.raises(ValueError, match="Orientation for objective 'q'"):
            res._objective_orientation("q")

    def test_persisted_missing_name_falls_to_canonical(self):
        res = _result([], metadata={"objective_orientations": {"other": "minimize"}})
        assert res._objective_orientation("accuracy") == "maximize"

    def test_persisted_non_dict_ignored(self):
        res = _result([], metadata={"objective_orientations": ["q"]})
        assert res._objective_orientation("cost") == "minimize"

    def test_auto_detect_minimize_objectives_uses_orientation(self):
        res = _result(
            [],
            objectives=["accuracy", "cost", "latency"],
        )
        assert res._auto_detect_minimize_objectives() == ["cost", "latency"]


# --------------------------------------------------------------------------
# _aggregated_mean_metrics / best_metrics
# --------------------------------------------------------------------------

AGG = {"session_summary": {"selection_mode": "aggregated_mean"}}


def _agg_meta(**summary: Any) -> dict[str, Any]:
    return {"session_summary": {"selection_mode": "aggregated_mean", **summary}}


class TestAggregatedMeanMetrics:
    def test_not_aggregated_returns_none(self):
        res = _result(
            [_trial(metrics={"accuracy": 1.0})],
            best_config={"a": 1},
            best_score=1.0,
            objectives=["accuracy"],
        )
        assert res._aggregated_mean_metrics() is None

    def test_aggregated_without_best_config_returns_none(self):
        res = _result(
            [_trial(metrics={"accuracy": 1.0})],
            best_config={},
            best_score=1.0,
            objectives=["accuracy"],
            metadata=_agg_meta(),
        )
        assert res._aggregated_mean_metrics() is None

    def test_samples_per_config_marks_aggregated(self):
        res = _result(
            [
                _trial("1", metrics={"accuracy": 0.5}),
                _trial("2", metrics={"accuracy": 1.0}),
            ],
            best_config={"a": 1},
            best_score=0.75,
            objectives=["accuracy"],
            metadata={"session_summary": {"samples_per_config": {"x": 2}}},
        )
        assert res._aggregated_mean_metrics() == {"accuracy": 0.75}

    def test_config_equality_fallback_means_and_pins_primary(self):
        res = _result(
            [
                _trial("1", {"a": 1}, {"accuracy": 0.8, "cost": 1.0, "n": 3}),
                _trial("2", {"a": 2}, {"accuracy": 0.1, "cost": 9.0}),
                _trial("3", {"a": 1}, {"accuracy": 0.9, "cost": 2.0, "n": 4}),
            ],
            best_config={"a": 1},
            best_score=0.123,  # primary is pinned to best_score
            objectives=["accuracy"],
            metadata=_agg_meta(),
        )
        out = res._aggregated_mean_metrics()
        assert out == {"accuracy": 0.123, "cost": 1.5, "n": 3.5}
        assert list(out) == ["accuracy", "cost", "n"]

    def test_stamped_winning_ids_take_precedence_over_config(self):
        res = _result(
            [
                _trial("1", {"a": 1}, {"accuracy": 0.2}),
                _trial("2", {"a": 99}, {"accuracy": 0.6}),
                _trial("3", {"a": 98}, {"accuracy": 1.0}),
            ],
            best_config={"a": 1},
            best_score=None,
            objectives=["accuracy"],
            metadata=_agg_meta(winning_trial_ids=["2", 3]),
        )
        out = res._aggregated_mean_metrics()
        assert out == {"accuracy": pytest.approx(0.8)}

    def test_empty_or_nonlist_stamp_falls_back_to_config(self):
        for stamp in ([], "2", None, {"2"}):
            res = _result(
                [
                    _trial("1", {"a": 1}, {"accuracy": 0.2}),
                    _trial("2", {"a": 99}, {"accuracy": 0.6}),
                ],
                best_config={"a": 1},
                objectives=["accuracy"],
                metadata=_agg_meta(winning_trial_ids=stamp),
            )
            assert res._aggregated_mean_metrics() == {"accuracy": 0.2}

    def test_non_dict_session_summary_in_replicate_lookup(self):
        # _is_aggregated_run needs a dict summary, so this just pins the guard.
        res = _result(
            [_trial("1", metrics={"accuracy": 0.2})],
            best_config={"a": 1},
            metadata={"session_summary": "garbage"},
        )
        assert res._aggregated_mean_metrics() is None

    def test_failed_and_metricless_trials_excluded(self):
        res = _result(
            [
                _trial("1", metrics={"accuracy": 0.2}, status=TrialStatus.FAILED),
                _trial("2", metrics={}),
                _trial(
                    "3",
                    metrics={"accuracy": 0.4},
                    metadata={"successful_examples": 0},
                ),
                _trial("4", metrics={"accuracy": 0.6}),
            ],
            best_config={"a": 1},
            objectives=["accuracy"],
            metadata=_agg_meta(),
        )
        assert res._aggregated_mean_metrics() == {"accuracy": 0.6}

    def test_stamped_ids_also_filter_unsuccessful_trials(self):
        res = _result(
            [
                _trial("1", metrics={"accuracy": 0.2}, status=TrialStatus.FAILED),
                _trial("2", metrics={"accuracy": 0.6}),
            ],
            best_config={"a": 1},
            objectives=["accuracy"],
            metadata=_agg_meta(winning_trial_ids=["1", "2"]),
        )
        assert res._aggregated_mean_metrics() == {"accuracy": 0.6}

    def test_no_replicates_returns_none(self):
        res = _result(
            [_trial("1", {"a": 2}, {"accuracy": 0.2})],
            best_config={"a": 1},
            objectives=["accuracy"],
            metadata=_agg_meta(),
        )
        assert res._aggregated_mean_metrics() is None

    def test_stamped_ids_matching_nothing_returns_none(self):
        res = _result(
            [_trial("1", {"a": 1}, {"accuracy": 0.2})],
            best_config={"a": 1},
            objectives=["accuracy"],
            metadata=_agg_meta(winning_trial_ids=["zzz"]),
        )
        assert res._aggregated_mean_metrics() is None

    def test_non_numeric_bool_and_examples_attempted_skipped(self):
        res = _result(
            [
                _trial(
                    "1",
                    metrics={
                        "accuracy": 1,
                        "flag": True,
                        "label": "x",
                        "none": None,
                        "examples_attempted": 10,
                        "obj": {"k": 1},
                    },
                ),
            ],
            best_config={"a": 1},
            objectives=["accuracy"],
            metadata=_agg_meta(),
        )
        out = res._aggregated_mean_metrics()
        assert out == {"accuracy": 1.0}
        assert type(out["accuracy"]) is float

    def test_only_non_numeric_metrics_returns_none(self):
        res = _result(
            [_trial("1", metrics={"flag": True, "examples_attempted": 3, "s": "x"})],
            best_config={"a": 1},
            objectives=["flag"],
            metadata=_agg_meta(),
        )
        assert res._aggregated_mean_metrics() is None

    def test_primary_not_pinned_when_absent_or_no_best_score_or_no_objectives(self):
        trials = [_trial("1", metrics={"accuracy": 0.5, "cost": 2.0})]
        base: dict[str, Any] = {"best_config": {"a": 1}, "metadata": _agg_meta()}
        # primary absent from mean_metrics
        res = _result(trials, best_score=0.9, objectives=["latency"], **base)
        assert res._aggregated_mean_metrics() == {"accuracy": 0.5, "cost": 2.0}
        # best_score None
        res = _result(trials, best_score=None, objectives=["accuracy"], **base)
        assert res._aggregated_mean_metrics() == {"accuracy": 0.5, "cost": 2.0}
        # no objectives
        res = _result(trials, best_score=0.9, objectives=[], **base)
        assert res._aggregated_mean_metrics() == {"accuracy": 0.5, "cost": 2.0}
        # pinned
        res = _result(trials, best_score=0.9, objectives=["accuracy"], **base)
        assert res._aggregated_mean_metrics() == {"accuracy": 0.9, "cost": 2.0}

    def test_summation_order_and_float_values_are_exact(self):
        vals = [0.1, 0.2, 0.3, 1e16, -1e16]
        trials = [_trial(str(i), metrics={"m": v}) for i, v in enumerate(vals)]
        res = _result(
            trials, best_config={"a": 1}, objectives=["zz"], metadata=_agg_meta()
        )
        total = 0.0
        for v in vals:
            total = total + float(v)
        assert res._aggregated_mean_metrics() == {"m": total / len(vals)}

    def test_nan_and_inf_propagate(self):
        res = _result(
            [
                _trial("1", metrics={"m": float("nan"), "i": float("inf")}),
                _trial("2", metrics={"m": 1.0, "i": 1.0}),
            ],
            best_config={"a": 1},
            objectives=["zz"],
            metadata=_agg_meta(),
        )
        out = res._aggregated_mean_metrics()
        assert math.isnan(out["m"])
        assert out["i"] == float("inf")

    def test_metric_name_first_seen_order_and_uneven_counts(self):
        res = _result(
            [
                _trial("1", metrics={"b": 2.0, "a": 4.0}),
                _trial("2", metrics={"c": 6.0, "a": 2.0}),
            ],
            best_config={"a": 1},
            objectives=["zz"],
            metadata=_agg_meta(),
        )
        out = res._aggregated_mean_metrics()
        assert out == {"b": 2.0, "a": 3.0, "c": 6.0}
        assert list(out) == ["b", "a", "c"]


class TestBestMetrics:
    def test_no_winner_returns_empty(self):
        res = _result([_trial(metrics={"accuracy": 1.0})], best_config={})
        assert res.best_metrics == {}

    def test_empty_config_with_score_keeps_metrics_no_objectives(self):
        res = _result(
            [_trial(metrics={"accuracy": 1.0})], best_config={}, best_score=0.5
        )
        # first loop requires truthy best_config; second loop returns any trial
        assert res.best_metrics == {"accuracy": 1.0}

    def test_no_trials_returns_empty(self):
        res = _result([], best_config={"a": 1}, best_score=1.0)
        assert res.best_metrics == {}

    def test_aggregated_path_takes_precedence(self):
        res = _result(
            [
                _trial("1", metrics={"accuracy": 0.5}),
                _trial("2", metrics={"accuracy": 1.0}),
            ],
            best_config={"a": 1},
            best_score=0.75,
            objectives=["accuracy"],
            metadata=_agg_meta(),
        )
        assert res.best_metrics == {"accuracy": 0.75}

    def test_aggregated_none_falls_through_to_single_trial(self):
        res = _result(
            [_trial("1", metrics={"flag": True})],
            best_config={"a": 1},
            best_score=0.75,
            objectives=["flag"],
            metadata=_agg_meta(),
        )
        assert res.best_metrics == {"flag": True}

    def test_no_objectives_matching_successful_best_config_trial(self):
        res = _result(
            [
                _trial("1", {"a": 2}, {"accuracy": 0.1}),
                _trial("2", {"a": 1}, {"accuracy": 0.2}, status=TrialStatus.FAILED),
                _trial("3", {"a": 1}, {"accuracy": 0.3}),
            ],
            best_config={"a": 1},
            best_score=0.3,
        )
        out = res.best_metrics
        assert out == {"accuracy": 0.3}
        assert out is not res.trials[2].metrics  # copy

    def test_no_objectives_falls_back_to_first_trial_with_metrics(self):
        res = _result(
            [
                _trial("1", {"a": 2}, {}),
                _trial("2", {"a": 3}, {"accuracy": 0.4}),
            ],
            best_config={"a": 1},
            best_score=0.3,
        )
        assert res.best_metrics == {"accuracy": 0.4}

    def test_no_objectives_all_metricless_returns_empty(self):
        res = _result([_trial("1", {"a": 2}, {})], best_config={"a": 1}, best_score=1)
        assert res.best_metrics == {}

    def test_objectives_config_match_ignores_success_status(self):
        res = _result(
            [
                _trial("1", {"a": 1}, {"accuracy": 0.2}, status=TrialStatus.FAILED),
                _trial("2", {"a": 1}, {"accuracy": 0.9}),
            ],
            best_config={"a": 1},
            best_score=0.9,
            objectives=["accuracy"],
        )
        assert res.best_metrics == {"accuracy": 0.2}

    def test_objectives_config_match_skips_metricless_trial(self):
        res = _result(
            [
                _trial("1", {"a": 1}, {}),
                _trial("2", {"a": 1}, {"accuracy": 0.9}),
            ],
            best_config={"a": 1},
            best_score=0.9,
            objectives=["accuracy"],
        )
        assert res.best_metrics == {"accuracy": 0.9}

    def test_objectives_minimize_primary_returns_config_match_not_max(self):
        res = _result(
            [
                _trial("1", {"a": 2}, {"cost": 9.0}),
                _trial("2", {"a": 1}, {"cost": 1.0}),
            ],
            best_config={"a": 1},
            best_score=1.0,
            objectives=["cost"],
        )
        assert res.best_metrics == {"cost": 1.0}

    def test_objectives_no_config_match_uses_best_trial_id(self):
        res = _result(
            [
                _trial("1", {"a": 2}, {"accuracy": 0.1}),
                _trial("2", {"a": 3}, {"accuracy": 0.7}),
            ],
            best_config={"a": 1},
            best_score=0.7,
            objectives=["accuracy"],
            metadata={"best_trial_id": "2"},
        )
        assert res.best_metrics == {"accuracy": 0.7}

    def test_best_trial_id_without_metrics_skipped_then_last_resort(self):
        res = _result(
            [
                _trial("1", {"a": 2}, {"accuracy": 0.1}, status=TrialStatus.FAILED),
                _trial("2", {"a": 3}, {}),
                _trial("3", {"a": 4}, {"accuracy": 0.5}),
            ],
            best_config={"a": 1},
            best_score=0.7,
            objectives=["accuracy"],
            metadata={"best_trial_id": "2"},
        )
        assert res.best_metrics == {"accuracy": 0.5}

    def test_best_trial_id_unmatched_last_resort_requires_success(self):
        res = _result(
            [_trial("1", {"a": 2}, {"accuracy": 0.1}, status=TrialStatus.FAILED)],
            best_config={"a": 1},
            best_score=0.7,
            objectives=["accuracy"],
            metadata={"best_trial_id": "nope"},
        )
        assert res.best_metrics == {}

    def test_metadata_none_with_objectives_no_best_config(self):
        res = _result(
            [_trial("1", {"a": 2}, {"accuracy": 0.1})],
            best_config={},
            best_score=0.1,
            objectives=["accuracy"],
        )
        res.metadata = None  # type: ignore[assignment]
        assert res.best_metrics == {"accuracy": 0.1}

    def test_returns_copy_not_alias(self):
        res = _result(
            [_trial("1", {"a": 1}, {"accuracy": 0.1})],
            best_config={"a": 1},
            best_score=0.1,
            objectives=["accuracy"],
        )
        out = res.best_metrics
        out["accuracy"] = 99
        assert res.trials[0].metrics == {"accuracy": 0.1}


# --------------------------------------------------------------------------
# to_dataframe
# --------------------------------------------------------------------------


class TestToDataFrame:
    def test_empty_trials(self):
        df = _result([]).to_dataframe()
        assert_frame_equal(df, pd.DataFrame([]), check_dtype=True)
        assert df.shape == (0, 0)

    def test_basic_row_column_order_and_total_cost_none(self):
        res = _result([_trial("t1", {"a": 1}, {"accuracy": 0.5})])
        df = res.to_dataframe()
        expected = pd.DataFrame(
            [
                {
                    "trial_id": "t1",
                    "status": "completed",
                    "duration": 0.5,
                    "timestamp": TS,
                    "a": 1,
                    "accuracy": 0.5,
                    "avg_response_time": None,
                    "avg_response_time_ms": None,
                    "total_cost": None,
                }
            ]
        )
        assert_frame_equal(df, expected, check_dtype=True)
        assert list(df.columns) == [
            "trial_id",
            "status",
            "duration",
            "timestamp",
            "a",
            "accuracy",
            "avg_response_time",
            "avg_response_time_ms",
            "total_cost",
        ]
        assert isinstance(df.index, pd.RangeIndex)

    def test_metadata_example_fields_and_total_example_cost(self):
        res = _result(
            [
                _trial(
                    "t1",
                    {"a": 1},
                    {"accuracy": 0.5},
                    metadata={
                        "successful_examples": 3,
                        "examples_attempted": 4,
                        "total_example_cost": 0.25,
                    },
                )
            ]
        )
        df = res.to_dataframe()
        assert list(df.columns) == [
            "trial_id",
            "status",
            "duration",
            "timestamp",
            "a",
            "accuracy",
            "avg_response_time",
            "avg_response_time_ms",
            "successful_examples",
            "examples_attempted",
            "example_success_summary",
            "total_cost",
        ]
        row = df.iloc[0]
        assert row["successful_examples"] == 3
        assert row["examples_attempted"] == 4
        assert row["example_success_summary"] == "3/4 examples succeeded"
        assert row["total_cost"] == 0.25

    def test_only_one_of_success_fields(self):
        res = _result(
            [
                _trial("t1", metadata={"successful_examples": 3}),
                _trial("t2", metadata={"examples_attempted": 9}),
            ]
        )
        df = res.to_dataframe()
        assert "example_success_summary" not in df.columns
        assert list(df["successful_examples"].isna()) == [False, True]
        assert list(df["examples_attempted"].isna()) == [True, False]

    def test_summary_with_none_values_is_still_built(self):
        res = _result(
            [
                _trial(
                    "t1",
                    metadata={"successful_examples": None, "examples_attempted": 2},
                )
            ]
        )
        df = res.to_dataframe()
        assert df.iloc[0]["example_success_summary"] == "None/2 examples succeeded"

    def test_total_cost_from_metrics_kept_when_no_example_cost(self):
        res = _result(
            [_trial("t1", {"a": 1}, {"total_cost": 1.5}, metadata={"total_cost": 7})]
        )
        df = res.to_dataframe()
        assert df.iloc[0]["total_cost"] == 1.5
        # metrics key stays at its original position (before response times)
        assert list(df.columns).index("total_cost") < list(df.columns).index(
            "avg_response_time"
        )

    def test_total_example_cost_overrides_metrics_total_cost_in_place(self):
        res = _result(
            [
                _trial(
                    "t1",
                    {"a": 1},
                    {"total_cost": 1.5},
                    metadata={"total_example_cost": 0.1},
                )
            ]
        )
        df = res.to_dataframe()
        assert df.iloc[0]["total_cost"] == 0.1
        assert list(df.columns)[-1] == "avg_response_time_ms"

    def test_total_cost_from_metadata_when_not_in_metrics(self):
        res = _result([_trial("t1", metadata={"total_cost": 7})])
        assert res.to_dataframe().iloc[0]["total_cost"] == 7

    def test_response_times_populated_and_converted_to_float(self):
        res = _result(
            [
                _trial(
                    "t1",
                    metadata={
                        "example_results": [
                            {"response_time": 1},
                            {"response_time_ms": 3000},
                        ]
                    },
                )
            ]
        )
        df = res.to_dataframe()
        assert df.iloc[0]["avg_response_time"] == 2.0
        assert df.iloc[0]["avg_response_time_ms"] == 2000.0
        assert df["avg_response_time"].dtype == "float64"

    def test_failed_trials_and_mixed_columns(self):
        res = _result(
            [
                _trial("t1", {"a": 1}, {"accuracy": 0.5}),
                _trial(
                    "t2",
                    {"b": "x"},
                    {"cost": 2.0},
                    status=TrialStatus.FAILED,
                    duration=1.25,
                    metadata={
                        "total_example_cost": 3.0,
                        "measures": {"response_time": 2},
                    },
                ),
            ]
        )
        df = res.to_dataframe()
        expected = pd.DataFrame(
            [
                {
                    "trial_id": "t1",
                    "status": "completed",
                    "duration": 0.5,
                    "timestamp": TS,
                    "a": 1,
                    "accuracy": 0.5,
                    "avg_response_time": None,
                    "avg_response_time_ms": None,
                    "total_cost": None,
                },
                {
                    "trial_id": "t2",
                    "status": "failed",
                    "duration": 1.25,
                    "timestamp": TS,
                    "b": "x",
                    "cost": 2.0,
                    "avg_response_time": 2.0,
                    "avg_response_time_ms": 2000.0,
                    "total_cost": 3.0,
                },
            ]
        )
        assert_frame_equal(df, expected, check_dtype=True)
        assert list(df.columns) == [
            "trial_id",
            "status",
            "duration",
            "timestamp",
            "a",
            "accuracy",
            "avg_response_time",
            "avg_response_time_ms",
            "total_cost",
            "b",
            "cost",
        ]

    def test_metrics_override_reserved_column_names_in_place(self):
        res = _result([_trial("t1", {"a": 1}, {"duration": 99.0, "a": 5})])
        df = res.to_dataframe()
        assert df.iloc[0]["duration"] == 99.0
        assert df.iloc[0]["a"] == 5
        assert list(df.columns)[:5] == [
            "trial_id",
            "status",
            "duration",
            "timestamp",
            "a",
        ]

    def test_none_metadata_is_handled(self):
        t = _trial("t1")
        t.metadata = None  # type: ignore[assignment]
        df = _result([t]).to_dataframe()
        assert df.iloc[0]["trial_id"] == "t1"
        assert df.iloc[0]["total_cost"] is None

    def test_aggregated_repeated_configs_yield_one_row_per_trial(self):
        trials = [
            _trial("t1", {"a": 1}, {"accuracy": 0.5}),
            _trial("t2", {"a": 1}, {"accuracy": 0.7}),
        ]
        df = _result(
            trials, metadata=_agg_meta(samples_per_config={"x": 2})
        ).to_dataframe()
        expected = pd.DataFrame(
            {
                "trial_id": ["t1", "t2"],
                "status": ["completed", "completed"],
                "duration": [0.5, 0.5],
                "timestamp": [TS, TS],
                "a": [1, 1],
                "accuracy": [0.5, 0.7],
                "avg_response_time": [None, None],
                "avg_response_time_ms": [None, None],
                "total_cost": [None, None],
            }
        )
        assert_frame_equal(df, expected, check_dtype=True)
        assert list(df.index) == [0, 1]


# --------------------------------------------------------------------------
# Review round 1 additions
# --------------------------------------------------------------------------


class _FoldedName(str):
    """str whose ``__eq__`` casefolds while ``__ne__`` stays case-sensitive."""

    def __eq__(self, other: object) -> bool:
        return str(self).casefold() == str(other).casefold()

    def __hash__(self) -> int:
        return hash(str(self).casefold())


class _AttrDict(dict):
    """Empty dict definition carrying name/orientation as attributes."""

    name = "my_custom"
    orientation = "minimize"


class TestReviewRound1:
    def test_schema_name_matching_uses_eq_not_ne(self):
        res = _result(
            [],
            metadata={
                "objective_schema": {
                    "objectives": [
                        {"name": _FoldedName("ACCURACY"), "orientation": "minimize"}
                    ]
                }
            },
        )
        assert res._objective_orientation("accuracy") == "minimize"

    def test_dict_subclass_definition_falls_back_to_attributes(self):
        res = _result([], metadata={"objective_schema": {"objectives": [_AttrDict()]}})
        assert res._objective_orientation("my_custom") == "minimize"

    def test_dict_subclass_keys_override_attributes(self):
        definition = _AttrDict(name="other", orientation="maximize")
        res = _result([], metadata={"objective_schema": {"objectives": [definition]}})
        assert res._objective_orientation("other") == "maximize"
        with pytest.raises(ValueError, match="no declared orientation"):
            res._objective_orientation("my_custom")

    def test_best_metrics_preserves_raw_key_order(self):
        metrics = {"zeta": 1.0, "alpha": 2.0, "mid": 3.0}
        # no objectives, config match
        res = _result([_trial("1", {"a": 1}, dict(metrics))], best_config={"a": 1})
        res.best_score = 1.0
        assert list(res.best_metrics) == ["zeta", "alpha", "mid"]
        # no objectives, fallback to first trial with metrics
        res = _result([_trial("1", {"a": 2}, dict(metrics))], best_config={"a": 1})
        res.best_score = 1.0
        assert list(res.best_metrics) == ["zeta", "alpha", "mid"]
        # objectives, config match
        res = _result(
            [_trial("1", {"a": 1}, dict(metrics))],
            best_config={"a": 1},
            best_score=1.0,
            objectives=["alpha"],
        )
        assert list(res.best_metrics) == ["zeta", "alpha", "mid"]
        # objectives, best_trial_id and last resort
        for md in ({"best_trial_id": "1"}, {}):
            res = _result(
                [_trial("1", {"a": 2}, dict(metrics))],
                best_config={"a": 1},
                best_score=1.0,
                objectives=["alpha"],
                metadata=md,
            )
            assert list(res.best_metrics) == ["zeta", "alpha", "mid"]

    def test_stamped_ids_helper_guards(self):
        res = _result([], metadata={})
        res.metadata = None  # type: ignore[assignment]
        assert res._stamped_winning_trial_ids() is None
        res.metadata = {"session_summary": "garbage"}
        assert res._stamped_winning_trial_ids() is None
        res.metadata = {"session_summary": {"winning_trial_ids": [1, "b"]}}
        assert res._stamped_winning_trial_ids() == {"1", "b"}

    def test_opaque_helper_json_success_returns_same_object(self):
        from traigent.api.types import _json_safe_opaque_value  # noqa: PLC0415

        payload = {"a": [1, 2]}
        assert _json_safe_opaque_value(payload) is payload
