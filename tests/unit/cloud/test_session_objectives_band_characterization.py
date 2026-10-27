"""Characterization of the canonical band translation for session objectives."""

from __future__ import annotations

import pytest

from traigent.cloud.session_objectives import _canonical_band


class _Recording(dict):
    """dict that records every ``get`` key, in order."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reads: list[str] = []

    def get(self, key, default=None):
        self.reads.append(key)
        return super().get(key, default)


class TestNotABand:
    @pytest.mark.parametrize("value", [None, [], [1, 2], (1, 2), "band", 5, object()])
    def test_non_dict_gives_none(self, value):
        assert _canonical_band(value, "t", 0.1) is None

    def test_empty_dict_gives_none(self):
        assert _canonical_band({}) is None


class TestExplicitTarget:
    def test_list_target_is_copied(self):
        source = [1, 2]
        out = _canonical_band({"target": source})
        assert out == {"target": [1, 2]}
        assert out["target"] is not source

    def test_tuple_target_becomes_list(self):
        assert _canonical_band({"target": (3, 4)}) == {"target": [3, 4]}

    def test_target_wins_over_other_spellings(self):
        band = {"target": [1, 2], "low": 8, "high": 9, "center": 5, "tol": 1}
        assert _canonical_band(band) == {"target": [1, 2]}

    def test_empty_target_is_kept(self):
        assert _canonical_band({"target": []}) == {"target": []}

    def test_falsy_zero_pair_is_kept(self):
        assert _canonical_band({"target": [0, 0]}) == {"target": [0, 0]}

    def test_non_iterable_target_raises(self):
        with pytest.raises(TypeError):
            _canonical_band({"target": 5})

    def test_string_target_is_split_into_characters(self):
        assert _canonical_band({"target": "ab"}) == {"target": ["a", "b"]}

    def test_reads_stop_after_target_test_alpha(self):
        band = _Recording(target=[1, 2])
        _canonical_band(band)
        assert band.reads == ["target", "test", "alpha"]

    def test_reads_skip_outer_supplied_options(self):
        band = _Recording(target=[1, 2])
        _canonical_band(band, "t", 0.5)
        assert band.reads == ["target"]


class TestLowHigh:
    def test_pair_becomes_target(self):
        assert _canonical_band({"low": 1, "high": 2}) == {"target": [1, 2]}

    def test_zero_bounds_are_valid(self):
        assert _canonical_band({"low": 0, "high": 0}) == {"target": [0, 0]}

    def test_missing_high_is_invalid(self):
        assert _canonical_band({"low": 1}) is None

    def test_missing_low_is_invalid(self):
        assert _canonical_band({"high": 1}) is None

    def test_half_pair_ignores_center_tolerance(self):
        assert _canonical_band({"low": 1, "center": 5, "tol": 1}) is None
        assert _canonical_band({"high": 1, "center": 5, "tol": 1}) is None

    def test_half_pair_never_reads_center(self):
        band = _Recording(low=1)
        _canonical_band(band)
        assert band.reads == ["target", "low", "high"]


class TestCenterTolerance:
    def test_center_and_tolerance_expand_to_bounds(self):
        assert _canonical_band({"center": 5, "tol": 2}) == {"target": [3, 7]}

    def test_zero_values_are_valid(self):
        assert _canonical_band({"center": 0, "tol": 1}) == {"target": [-1, 1]}
        assert _canonical_band({"center": 5, "tol": 0}) == {"target": [5, 5]}

    def test_missing_tolerance_is_invalid(self):
        assert _canonical_band({"center": 5}) is None

    def test_missing_center_is_invalid(self):
        assert _canonical_band({"tol": 1}) is None

    def test_float_arithmetic_is_plain(self):
        assert _canonical_band({"center": 0.5, "tol": 0.25}) == {"target": [0.25, 0.75]}

    def test_non_numeric_center_raises(self):
        with pytest.raises(TypeError):
            _canonical_band({"center": "a", "tol": 1})

    def test_read_order(self):
        band = _Recording(center=5, tol=1)
        _canonical_band(band)
        assert band.reads == ["target", "low", "high", "center", "tol", "test", "alpha"]

    def test_read_error_propagates(self):
        class Broken(dict):
            def get(self, key, default=None):
                if key == "tol":
                    raise RuntimeError("tol read")
                return super().get(key, default)

        with pytest.raises(RuntimeError, match="tol read"):
            _canonical_band(Broken(center=5, tol=1))


class TestTestAndAlpha:
    def test_outer_values_are_emitted(self):
        assert _canonical_band({"target": [1, 2]}, "z", 0.05) == {
            "target": [1, 2],
            "test": "z",
            "alpha": 0.05,
        }

    def test_nested_values_are_used_when_outer_absent(self):
        band = {"target": [1, 2], "test": "t", "alpha": 0.1}
        assert _canonical_band(band) == {"target": [1, 2], "test": "t", "alpha": 0.1}

    def test_outer_wins_over_nested(self):
        band = {"target": [1, 2], "test": "nested", "alpha": 0.1}
        assert _canonical_band(band, "outer", 0.2) == {
            "target": [1, 2],
            "test": "outer",
            "alpha": 0.2,
        }

    def test_each_option_is_chosen_independently(self):
        band = {"target": [1, 2], "test": "nested", "alpha": 0.1}
        assert _canonical_band(band, None, 0.2) == {
            "target": [1, 2],
            "test": "nested",
            "alpha": 0.2,
        }
        assert _canonical_band(band, "outer", None) == {
            "target": [1, 2],
            "test": "outer",
            "alpha": 0.1,
        }

    def test_falsy_outer_values_win_and_are_emitted(self):
        band = {"target": [1, 2], "test": "nested", "alpha": 0.1}
        assert _canonical_band(band, "", 0) == {
            "target": [1, 2],
            "test": "",
            "alpha": 0,
        }

    def test_absent_options_are_omitted(self):
        assert _canonical_band({"target": [1, 2]}) == {"target": [1, 2]}

    def test_none_nested_options_are_omitted(self):
        band = {"target": [1, 2], "test": None, "alpha": None}
        assert _canonical_band(band) == {"target": [1, 2]}

    def test_unknown_keys_are_never_forwarded(self):
        band = {"low": 1, "high": 2, "center": 9, "tol": 9, "extra": 1}
        assert _canonical_band(band) == {"target": [1, 2]}

    def test_options_apply_to_derived_targets_too(self):
        assert _canonical_band({"center": 5, "tol": 1}, "t", 0.5) == {
            "target": [4, 6],
            "test": "t",
            "alpha": 0.5,
        }

    def test_invalid_band_ignores_options(self):
        assert _canonical_band({"low": 1}, "t", 0.5) is None
