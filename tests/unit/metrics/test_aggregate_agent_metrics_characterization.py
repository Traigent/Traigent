"""Characterization tests for ``aggregate_agent_metrics``.

These pin the observable behaviour of the generic per-metric aggregation
(sum / mean / max / min) so that internal restructuring cannot change it.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping
from types import MappingProxyType

import pytest

from traigent.metrics.agent_metrics import aggregate_agent_metrics

UNKNOWN_MESSAGE = "Unknown aggregation method: {}"


class _Spy:
    """Float-convertible value that records the order of conversions."""

    def __init__(self, value: float, label: str, log: list[str]) -> None:
        self._value = value
        self._label = label
        self._log = log

    def __float__(self) -> float:
        self._log.append(self._label)
        return self._value


class _Unconvertible:
    """Value whose float conversion fails, recording the attempt."""

    def __init__(self, label: str, log: list[str]) -> None:
        self._label = label
        self._log = log

    def __float__(self) -> float:
        self._log.append(self._label)
        raise RuntimeError(f"cannot convert {self._label}")


class _RecordingAggregation:
    """Aggregation selector that records every equality probe it receives."""

    def __init__(self, matches: str | None = None) -> None:
        self.matches = matches
        self.probes: list[object] = []

    def __eq__(self, other: object) -> bool:
        self.probes.append(other)
        return other == self.matches

    __hash__ = None  # type: ignore[assignment]

    def __str__(self) -> str:
        return "recording-aggregation"


class _SizedMapping(Mapping):
    """Mapping with a configurable length, to exercise the empty-input guard."""

    def __init__(self, data: dict) -> None:
        self._data = data

    def __getitem__(self, key):
        return self._data[key]

    def __iter__(self) -> Iterator:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)


PER_AGENT = {
    "grader": {"cost": 0.5, "latency": 100.0},
    "generator": {"cost": 1.5, "latency": 300.0},
    "router": {"cost": 4.0, "latency": 50.0},
}


class TestReductions:
    def test_sum(self):
        assert aggregate_agent_metrics(PER_AGENT, aggregation="sum") == {
            "total_cost": 6.0,
            "total_latency": 450.0,
        }

    def test_mean(self):
        assert aggregate_agent_metrics(PER_AGENT, aggregation="mean") == {
            "mean_cost": 2.0,
            "mean_latency": 150.0,
        }

    def test_max(self):
        assert aggregate_agent_metrics(PER_AGENT, aggregation="max") == {
            "max_cost": 4.0,
            "max_latency": 300.0,
        }

    def test_min(self):
        assert aggregate_agent_metrics(PER_AGENT, aggregation="min") == {
            "min_cost": 0.5,
            "min_latency": 50.0,
        }

    def test_default_aggregation_is_sum(self):
        assert aggregate_agent_metrics(PER_AGENT) == aggregate_agent_metrics(
            PER_AGENT, aggregation="sum"
        )

    def test_aggregation_is_keyword_only(self):
        with pytest.raises(TypeError):
            aggregate_agent_metrics(PER_AGENT, "sum")  # type: ignore[misc]

    def test_single_agent_single_value(self):
        data = {"only": {"m": 3.25}}
        assert aggregate_agent_metrics(data, aggregation="sum") == {"total_m": 3.25}
        assert aggregate_agent_metrics(data, aggregation="mean") == {"mean_m": 3.25}
        assert aggregate_agent_metrics(data, aggregation="max") == {"max_m": 3.25}
        assert aggregate_agent_metrics(data, aggregation="min") == {"min_m": 3.25}

    def test_results_are_floats(self):
        data = {"a": {"m": 1}, "b": {"m": 2}}
        for aggregation, key in (
            ("sum", "total_m"),
            ("mean", "mean_m"),
            ("max", "max_m"),
            ("min", "min_m"),
        ):
            value = aggregate_agent_metrics(data, aggregation=aggregation)[key]
            assert type(value) is float

    def test_negative_values(self):
        data = {"a": {"m": -3.0}, "b": {"m": 1.0}, "c": {"m": -5.0}}
        assert aggregate_agent_metrics(data, aggregation="sum") == {"total_m": -7.0}
        assert aggregate_agent_metrics(data, aggregation="max") == {"max_m": 1.0}
        assert aggregate_agent_metrics(data, aggregation="min") == {"min_m": -5.0}


class TestGroupingAndOrder:
    def test_mixed_metric_groups_use_only_present_values(self):
        data = {
            "a": {"x": 1.0, "y": 10.0},
            "b": {"y": 20.0, "z": 5.0},
            "c": {"x": 3.0},
        }
        assert aggregate_agent_metrics(data, aggregation="sum") == {
            "total_x": 4.0,
            "total_y": 30.0,
            "total_z": 5.0,
        }
        # mean divides by the count of agents that reported that metric
        assert aggregate_agent_metrics(data, aggregation="mean") == {
            "mean_x": 2.0,
            "mean_y": 15.0,
            "mean_z": 5.0,
        }
        assert aggregate_agent_metrics(data, aggregation="max") == {
            "max_x": 3.0,
            "max_y": 20.0,
            "max_z": 5.0,
        }
        assert aggregate_agent_metrics(data, aggregation="min") == {
            "min_x": 1.0,
            "min_y": 10.0,
            "min_z": 5.0,
        }

    def test_result_keys_follow_first_seen_order(self):
        data = {
            "a": {"zeta": 1.0, "alpha": 2.0},
            "b": {"mid": 3.0, "alpha": 4.0, "zeta": 5.0},
        }
        result = aggregate_agent_metrics(data, aggregation="sum")
        assert list(result) == ["total_zeta", "total_alpha", "total_mid"]

    def test_values_keep_agent_then_metric_order_for_every_reduction(self):
        # max/min ties and NaN make reduction order observable.
        data = {"a": {"m": math.nan}, "b": {"m": 1.0}}
        assert math.isnan(aggregate_agent_metrics(data, aggregation="max")["max_m"])
        assert math.isnan(aggregate_agent_metrics(data, aggregation="min")["min_m"])
        data = {"a": {"m": 1.0}, "b": {"m": math.nan}}
        assert aggregate_agent_metrics(data, aggregation="max")["max_m"] == 1.0
        assert aggregate_agent_metrics(data, aggregation="min")["min_m"] == 1.0

    def test_agent_with_empty_metrics_is_skipped(self):
        data = {"a": {}, "b": {"m": 2.0}, "c": {}}
        assert aggregate_agent_metrics(data, aggregation="mean") == {"mean_m": 2.0}

    def test_non_string_metric_name_is_formatted_into_key(self):
        data = {"a": {5: 1.0}, "b": {5: 2.0}}
        assert aggregate_agent_metrics(data, aggregation="sum") == {"total_5": 3.0}

    def test_inputs_are_not_mutated(self):
        data = {"a": {"m": 1}, "b": {"m": 2}}
        aggregate_agent_metrics(data, aggregation="sum")
        assert data == {"a": {"m": 1}, "b": {"m": 2}}

    def test_accepts_read_only_mappings(self):
        data = MappingProxyType(
            {"a": MappingProxyType({"m": 1.0}), "b": MappingProxyType({"m": 2.0})}
        )
        assert aggregate_agent_metrics(data, aggregation="sum") == {"total_m": 3.0}


class TestEmptyInputAndInvalidAggregation:
    @pytest.mark.parametrize("aggregation", ["sum", "mean", "max", "min", "bogus"])
    def test_empty_input_returns_empty_before_validating_aggregation(self, aggregation):
        assert aggregate_agent_metrics({}, aggregation=aggregation) == {}

    def test_empty_custom_mapping_returns_empty_even_if_invalid(self):
        assert aggregate_agent_metrics(_SizedMapping({}), aggregation="bogus") == {}

    def test_empty_input_returns_a_fresh_dict(self):
        first = aggregate_agent_metrics({})
        first["x"] = 1.0
        assert aggregate_agent_metrics({}) == {}

    @pytest.mark.parametrize("aggregation", ["sum", "mean", "max", "min", "bogus"])
    def test_all_empty_agent_maps_return_empty(self, aggregation):
        data = {"a": {}, "b": {}}
        assert aggregate_agent_metrics(data, aggregation=aggregation) == {}

    def test_invalid_aggregation_raises_when_a_metric_exists(self):
        with pytest.raises(ValueError) as excinfo:
            aggregate_agent_metrics({"a": {"m": 1.0}}, aggregation="bogus")
        assert str(excinfo.value) == "Unknown aggregation method: bogus"

    @pytest.mark.parametrize(
        "aggregation",
        ["SUM", "Mean", " sum", "sum ", "", "average", "median", "total", None, 3],
    )
    def test_unknown_aggregation_text(self, aggregation):
        with pytest.raises(ValueError) as excinfo:
            aggregate_agent_metrics({"a": {"m": 1.0}}, aggregation=aggregation)
        assert str(excinfo.value) == UNKNOWN_MESSAGE.format(aggregation)

    def test_invalid_aggregation_reported_even_when_only_one_agent_has_metrics(self):
        data = {"a": {}, "b": {"m": 1.0}}
        with pytest.raises(ValueError, match="Unknown aggregation method: nope"):
            aggregate_agent_metrics(data, aggregation="nope")

    def test_selector_is_probed_in_chain_order_then_rejected(self):
        selector = _RecordingAggregation(matches=None)
        with pytest.raises(ValueError) as excinfo:
            aggregate_agent_metrics({"a": {"m": 1.0, "n": 2.0}}, aggregation=selector)  # type: ignore[arg-type]
        # Raised on the first metric: exactly one pass through the four tests.
        assert selector.probes == ["sum", "mean", "max", "min"]
        assert str(excinfo.value) == "Unknown aggregation method: recording-aggregation"

    @pytest.mark.parametrize(
        ("matches", "key", "probes"),
        [
            ("sum", "total_m", ["sum"]),
            ("mean", "mean_m", ["sum", "mean"]),
            ("max", "max_m", ["sum", "mean", "max"]),
            ("min", "min_m", ["sum", "mean", "max", "min"]),
        ],
    )
    def test_selector_matching_stops_probing(self, matches, key, probes):
        selector = _RecordingAggregation(matches=matches)
        result = aggregate_agent_metrics(
            {"a": {"m": 2.0}, "b": {"m": 4.0}}, aggregation=selector
        )  # type: ignore[arg-type]
        assert list(result) == [key]
        assert selector.probes == probes

    def test_selector_is_probed_once_per_metric(self):
        selector = _RecordingAggregation(matches="sum")
        aggregate_agent_metrics(
            {"a": {"m": 1.0, "n": 2.0}, "b": {"o": 3.0}},
            aggregation=selector,  # type: ignore[arg-type]
        )
        assert selector.probes == ["sum", "sum", "sum"]

    def test_selector_never_probed_for_empty_input(self):
        selector = _RecordingAggregation(matches="sum")
        assert aggregate_agent_metrics({}, aggregation=selector) == {}  # type: ignore[arg-type]
        assert aggregate_agent_metrics({"a": {}}, aggregation=selector) == {}  # type: ignore[arg-type]
        assert selector.probes == []


class TestFloatConversion:
    def test_each_value_converted_once_in_agent_then_metric_order(self):
        log: list[str] = []
        data = {
            "a": {"x": _Spy(1.0, "a.x", log), "y": _Spy(2.0, "a.y", log)},
            "b": {"y": _Spy(3.0, "b.y", log), "x": _Spy(4.0, "b.x", log)},
        }
        result = aggregate_agent_metrics(data, aggregation="sum")
        assert log == ["a.x", "a.y", "b.y", "b.x"]
        assert result == {"total_x": 5.0, "total_y": 5.0}

    def test_all_conversions_happen_before_invalid_aggregation_is_raised(self):
        log: list[str] = []
        data = {
            "a": {"x": _Spy(1.0, "a.x", log)},
            "b": {"y": _Spy(2.0, "b.y", log)},
            "c": {"x": _Spy(3.0, "c.x", log)},
        }
        with pytest.raises(ValueError, match="Unknown aggregation method: bogus"):
            aggregate_agent_metrics(data, aggregation="bogus")
        assert log == ["a.x", "b.y", "c.x"]

    def test_conversion_error_beats_invalid_aggregation(self):
        log: list[str] = []
        data = {
            "a": {"x": _Spy(1.0, "a.x", log)},
            "b": {"x": _Unconvertible("b.x", log)},
        }
        with pytest.raises(RuntimeError, match="cannot convert b.x"):
            aggregate_agent_metrics(data, aggregation="bogus")
        assert log == ["a.x", "b.x"]

    def test_conversion_error_halts_further_conversions(self):
        log: list[str] = []
        data = {
            "a": {"x": _Spy(1.0, "a.x", log), "y": _Unconvertible("a.y", log)},
            "b": {"x": _Spy(2.0, "b.x", log)},
        }
        with pytest.raises(RuntimeError, match="cannot convert a.y"):
            aggregate_agent_metrics(data, aggregation="sum")
        assert log == ["a.x", "a.y"]

    def test_empty_agent_maps_convert_nothing(self):
        log: list[str] = []
        # no values at all, so nothing to convert and nothing to reject
        assert aggregate_agent_metrics({"a": {}, "b": {}}, aggregation="bogus") == {}
        assert log == []

    def test_none_value_raises_type_error(self):
        with pytest.raises(TypeError):
            aggregate_agent_metrics({"a": {"m": None}}, aggregation="sum")  # type: ignore[dict-item]

    def test_unparseable_string_raises_value_error(self):
        with pytest.raises(ValueError, match="could not convert string to float"):
            aggregate_agent_metrics({"a": {"m": "abc"}}, aggregation="sum")  # type: ignore[dict-item]

    def test_numeric_strings_are_coerced(self):
        data = {"a": {"m": "2.5"}, "b": {"m": " 1 "}}
        assert aggregate_agent_metrics(data, aggregation="sum") == {"total_m": 3.5}  # type: ignore[dict-item]

    def test_ints_and_bools_are_coerced_to_float(self):
        data = {"a": {"m": True}, "b": {"m": 2}, "c": {"m": False}}
        assert aggregate_agent_metrics(data, aggregation="sum") == {"total_m": 3.0}
        assert aggregate_agent_metrics(data, aggregation="max") == {"max_m": 2.0}
        assert aggregate_agent_metrics(data, aggregation="min") == {"min_m": 0.0}
        assert aggregate_agent_metrics(data, aggregation="mean") == {"mean_m": 1.0}
        for aggregation, key in (("max", "max_m"), ("min", "min_m")):
            assert (
                type(aggregate_agent_metrics(data, aggregation=aggregation)[key])
                is float
            )

    def test_single_bool_reduces_to_float_not_bool(self):
        result = aggregate_agent_metrics({"a": {"m": True}}, aggregation="max")
        assert result == {"max_m": 1.0}
        assert type(result["max_m"]) is float


class TestSpecialFloats:
    def test_nan_poisons_sum_and_mean(self):
        data = {"a": {"m": math.nan}, "b": {"m": 1.0}}
        assert math.isnan(aggregate_agent_metrics(data, aggregation="sum")["total_m"])
        assert math.isnan(aggregate_agent_metrics(data, aggregation="mean")["mean_m"])

    def test_nan_in_max_min_depends_on_position(self):
        first = {"a": {"m": math.nan}, "b": {"m": 1.0}, "c": {"m": 2.0}}
        assert math.isnan(aggregate_agent_metrics(first, aggregation="max")["max_m"])
        assert math.isnan(aggregate_agent_metrics(first, aggregation="min")["min_m"])
        middle = {"a": {"m": 1.0}, "b": {"m": math.nan}, "c": {"m": 2.0}}
        assert aggregate_agent_metrics(middle, aggregation="max") == {"max_m": 2.0}
        assert aggregate_agent_metrics(middle, aggregation="min") == {"min_m": 1.0}

    def test_nan_from_string_value(self):
        data = {"a": {"m": "nan"}, "b": {"m": 1.0}}
        assert math.isnan(aggregate_agent_metrics(data, aggregation="sum")["total_m"])  # type: ignore[dict-item]

    def test_infinities(self):
        data = {"a": {"m": math.inf}, "b": {"m": 1.0}}
        assert aggregate_agent_metrics(data, aggregation="sum") == {"total_m": math.inf}
        assert aggregate_agent_metrics(data, aggregation="mean") == {"mean_m": math.inf}
        assert aggregate_agent_metrics(data, aggregation="max") == {"max_m": math.inf}
        assert aggregate_agent_metrics(data, aggregation="min") == {"min_m": 1.0}

    def test_opposite_infinities_sum_to_nan(self):
        data = {"a": {"m": math.inf}, "b": {"m": -math.inf}}
        assert math.isnan(aggregate_agent_metrics(data, aggregation="sum")["total_m"])
        assert math.isnan(aggregate_agent_metrics(data, aggregation="mean")["mean_m"])
        assert aggregate_agent_metrics(data, aggregation="max") == {"max_m": math.inf}
        assert aggregate_agent_metrics(data, aggregation="min") == {"min_m": -math.inf}

    def test_negative_infinity_mean(self):
        data = {"a": {"m": -math.inf}, "b": {"m": 1.0}}
        assert aggregate_agent_metrics(data, aggregation="mean") == {
            "mean_m": -math.inf
        }

    def test_signed_zero_max_min_keep_first_element(self):
        neg_first = {"a": {"m": -0.0}, "b": {"m": 0.0}}
        pos_first = {"a": {"m": 0.0}, "b": {"m": -0.0}}
        for aggregation, key in (("max", "max_m"), ("min", "min_m")):
            assert (
                math.copysign(
                    1.0,
                    aggregate_agent_metrics(neg_first, aggregation=aggregation)[key],
                )
                == -1.0
            )
            assert (
                math.copysign(
                    1.0,
                    aggregate_agent_metrics(pos_first, aggregation=aggregation)[key],
                )
                == 1.0
            )

    def test_signed_zero_sum_and_mean(self):
        data = {"a": {"m": -0.0}, "b": {"m": -0.0}}
        total = aggregate_agent_metrics(data, aggregation="sum")["total_m"]
        mean = aggregate_agent_metrics(data, aggregation="mean")["mean_m"]
        assert total == 0.0
        assert mean == 0.0
        assert math.copysign(1.0, mean) == math.copysign(1.0, total)

    def test_large_values_overflow_to_infinity_in_sum(self):
        data = {"a": {"m": 1.7e308}, "b": {"m": 1.7e308}}
        assert aggregate_agent_metrics(data, aggregation="sum") == {"total_m": math.inf}
