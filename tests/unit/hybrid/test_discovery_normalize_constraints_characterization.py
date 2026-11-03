"""Characterization for ConfigSpaceDiscovery._normalize_constraints_for_parsing."""

from __future__ import annotations

from traigent.hybrid.discovery import ConfigSpaceDiscovery

normalize = ConfigSpaceDiscovery._normalize_constraints_for_parsing


def test_text_entries_get_ids_in_group_order_with_per_group_index():
    out = normalize({"hard": ["a > 0", "b > 0"], "soft": ["c < 1"]})
    assert out == [
        {
            "id": "hard_0",
            "type": "expression",
            "rule": "a > 0",
            "error_message": "hard constraint 0 violated",
        },
        {
            "id": "hard_1",
            "type": "expression",
            "rule": "b > 0",
            "error_message": "hard constraint 1 violated",
        },
        {
            "id": "soft_0",
            "type": "expression",
            "rule": "c < 1",
            "error_message": "soft constraint 0 violated",
        },
    ]


def test_mixed_entries_keep_position_index_and_skip_other_types():
    d = {"id": "own"}
    out = normalize({"g": ["x", d, 5, None, "y"]})
    assert [r.get("id") for r in out] == ["g_0", "own", "g_4"]
    assert out[1] is d


def test_no_usable_entries_returns_the_original_mapping():
    c1 = {"hard": "nope", "soft": [1, None], "empty": []}
    assert normalize(c1) is c1
    c2 = {}
    assert normalize(c2) is c2


def test_typed_keys_detected_even_with_other_groups():
    c = {"hard": ["x"], "derived": []}
    assert normalize(c) is c
    c = {"structural": 1}
    assert normalize(c) is c


def test_group_name_is_stringified_in_id():
    out = normalize({1: ["x"]})
    assert out[0]["id"] == "1_0"


def test_empty_list_and_falsey_non_collections():
    lst: list = []
    assert normalize(lst) is lst
    assert normalize("") is None
    assert normalize(0) is None  # type: ignore[arg-type]
    assert normalize(()) is None  # type: ignore[arg-type]


def test_dict_subclass_lookup_and_iteration_order():
    class Logging(dict):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self.log = []

        def __contains__(self, key):
            self.log.append(("in", key))
            return super().__contains__(key)

        def items(self):
            self.log.append(("items", None))
            return super().items()

    c = Logging(hard=["x"])
    out = normalize(c)
    assert out[0]["id"] == "hard_0"
    assert c.log == [("in", "structural"), ("in", "derived"), ("items", None)]
    c2 = Logging(structural=[])
    assert normalize(c2) is c2
    assert c2.log == [("in", "structural")]
