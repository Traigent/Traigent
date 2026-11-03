"""Characterization of unmeasured_example_ids row classification."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from traigent.utils.trial_costs import unmeasured_example_ids


def _row(example_id, metrics, **extra):
    return {"example_id": example_id, "metrics": metrics, **extra}


class TestUnreadableInput:
    @pytest.mark.parametrize("value", ["abc", b"abc", bytearray(b"abc"), 5, None, {}])
    def test_non_sequence_or_text_yields_empty(self, value):
        assert unmeasured_example_ids(value) == []

    def test_empty_sequence(self):
        assert unmeasured_example_ids([]) == []

    def test_tuple_accepted(self):
        assert unmeasured_example_ids((_row("a", {}),)) == ["a"]

    @pytest.mark.parametrize("metrics", [None, 5, "x", [("cost", 1)]])
    def test_row_without_mapping_metrics_is_skipped(self, metrics):
        assert unmeasured_example_ids([_row("a", metrics)]) == []

    def test_row_without_metrics_key_is_skipped(self):
        assert unmeasured_example_ids([{"example_id": "a"}, 7, None]) == []


class TestClassification:
    def test_measured_cost_not_reported(self):
        assert unmeasured_example_ids([_row("a", {"cost": 0.5})]) == []

    def test_zero_measured_cost_not_reported(self):
        assert unmeasured_example_ids([_row("a", {"total_cost": 0})]) == []

    def test_missing_cost_reported(self):
        assert unmeasured_example_ids([_row("a", {"latency": 1})]) == ["a"]

    @pytest.mark.parametrize("bad", [True, "0.5", None, float("nan"), float("inf")])
    def test_non_finite_or_non_numeric_cost_reported(self, bad):
        assert unmeasured_example_ids([_row("a", {"cost": bad})]) == ["a"]

    def test_flag_overrides_reported_cost(self):
        rows = [_row("a", {"cost": 0.5, "cost_unmeasured": True})]
        assert unmeasured_example_ids(rows) == ["a"]
        rows = [_row("a", {"cost": 0.5, "cost_unmeasured": 1.0})]
        assert unmeasured_example_ids(rows) == ["a"]

    @pytest.mark.parametrize("flag", [0, 0.0, False, -1, None, float("nan")])
    def test_non_positive_flag_does_not_flag(self, flag):
        rows = [_row("a", {"cost": 0.5, "cost_unmeasured": flag})]
        assert unmeasured_example_ids(rows) == []

    def test_failed_row_with_zero_attempts_is_not_reported(self):
        rows = [_row("a", {"provider_attempts": 0}, success=False)]
        assert unmeasured_example_ids(rows) == []

    @pytest.mark.parametrize("attempts", [1, 2, True, None, "0", float("nan")])
    def test_failed_row_with_other_attempts_reported(self, attempts):
        rows = [_row("a", {"provider_attempts": attempts}, success=False)]
        assert unmeasured_example_ids(rows) == ["a"]

    def test_successful_row_with_zero_attempts_reported(self):
        rows = [_row("a", {"provider_attempts": 0}, success=True)]
        assert unmeasured_example_ids(rows) == ["a"]

    def test_success_defaults_to_true_for_mapping_rows(self):
        rows = [_row("a", {"provider_attempts": 0})]
        assert unmeasured_example_ids(rows) == ["a"]

    def test_falsy_success_values_count_as_failed(self):
        for ok in (False, 0, None, ""):
            rows = [_row("a", {"provider_attempts": 0}, success=ok)]
            assert unmeasured_example_ids(rows) == [], ok

    def test_flagged_failed_zero_attempt_row_still_reported(self):
        metrics = {"provider_attempts": 0, "cost_unmeasured": True}
        assert unmeasured_example_ids([_row("a", metrics, success=False)]) == ["a"]

    def test_zero_attempts_read_from_metrics_not_from_row(self):
        row = _row("a", {}, success=False, provider_attempts=0)
        assert unmeasured_example_ids([row]) == ["a"]


class TestObjectRows:
    def test_attribute_rows_supported(self):
        row = SimpleNamespace(example_id="e1", metrics={}, success=True)
        assert unmeasured_example_ids([row]) == ["e1"]

    def test_attribute_row_success_defaults_true(self):
        row = SimpleNamespace(example_id="e1", metrics={"provider_attempts": 0})
        assert unmeasured_example_ids([row]) == ["e1"]

    def test_attribute_row_failed_zero_attempts_excluded(self):
        row = SimpleNamespace(
            example_id="e1", metrics={"provider_attempts": 0}, success=False
        )
        assert unmeasured_example_ids([row]) == []

    def test_attribute_row_without_metrics_skipped(self):
        assert unmeasured_example_ids([SimpleNamespace(example_id="e1")]) == []


class TestIdentifiers:
    def test_falsy_id_replaced_by_position(self):
        rows = [_row("a", {}), _row(None, {}), _row("", {}), _row(0, {}), _row("d", {})]
        assert unmeasured_example_ids(rows) == ["a", "#1", "#2", "#3", "d"]

    def test_position_counts_skipped_rows(self):
        rows = [_row("ok", {"cost": 1}), 5, _row(None, {})]
        assert unmeasured_example_ids(rows) == ["#2"]

    def test_id_stringified(self):
        assert unmeasured_example_ids([_row(42, {})]) == ["42"]

    def test_row_content_never_returned(self):
        rows = [_row("id-1", {}, prompt="secret text", output="more secret")]
        assert unmeasured_example_ids(rows) == ["id-1"]


class TestLimit:
    ROWS = [_row(f"r{i}", {}) for i in range(8)]

    def test_default_limit_is_five(self):
        assert unmeasured_example_ids(self.ROWS) == ["r0", "r1", "r2", "r3", "r4"]

    @pytest.mark.parametrize("limit,count", [(1, 1), (7, 7), (8, 8), (9, 8)])
    def test_explicit_limit(self, limit, count):
        assert unmeasured_example_ids(self.ROWS, limit=limit) == [
            f"r{i}" for i in range(count)
        ]

    def test_limit_applies_after_filtering(self):
        rows = [_row("m", {"cost": 1}), *self.ROWS]
        assert unmeasured_example_ids(rows, limit=2) == ["r0", "r1"]

    @pytest.mark.parametrize("limit", [0, -3])
    def test_non_positive_limit_still_returns_first_match(self, limit):
        # Existing behaviour: the cap is checked after appending.
        assert unmeasured_example_ids(self.ROWS, limit=limit) == ["r0"]
