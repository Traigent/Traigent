"""Golden characterization tests for traigent.core.utils.debug_print_object."""

from __future__ import annotations

from traigent.core.utils import debug_print_object as dpo


class Plain:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


class Slotted:
    __slots__ = ("a",)

    def __init__(self):
        self.a = 1

    def __str__(self):
        return "slotted-str"


class BadVars:
    @property
    def __dict__(self):  # type: ignore[override]
        raise RuntimeError("no introspection")

    def __str__(self):
        return "bad-vars-str"


def test_depth_limit_and_none():
    assert dpo({"a": 1}, max_depth=0) == "..."
    assert dpo(None) == "None"
    assert dpo(None, current_depth=1) == "  None"
    assert dpo(5, max_depth=3, current_depth=3) == "..."


def test_empty_containers():
    assert dpo({}) == "{}"
    assert dpo([]) == "[]"
    assert dpo((), current_depth=1) == "  []"


def test_nested_dict_golden():
    assert dpo({"a": 1, "b": {"c": 2}}, max_depth=3) == (
        "{\n  a:   1\n  b:   {\n    c:     2\n  }\n}"
    )


def test_dict_exact_output():
    out = dpo({"a": 1, "b": [2]})
    assert out == "{\n  a:   1\n  b:   [\n    ...\n  ]\n}"


def test_list_truncates_to_five_with_remainder_line():
    assert dpo(list(range(7))) == (
        "[\n    0\n    1\n    2\n    3\n    4\n  ... (2 more items)\n]"
    )


def test_list_exactly_five_has_no_remainder_line():
    assert "more items" not in dpo(list(range(5)))


def test_tuple_treated_like_list():
    assert dpo((1,)) == "[\n    1\n]"


def test_object_with_dict_limits_to_three_attributes():
    obj = Plain(a=1, b="x", c=None, d=4)
    out = dpo(obj, max_depth=3)
    assert out == (
        "Plain(\n  a:   1\n  b:   x\n  c:   None\n  ... (1 more attributes)\n)"
    )


def test_object_with_exactly_three_attributes_no_remainder():
    assert "more attributes" not in dpo(Plain(a=1, b=2, c=3), max_depth=3)


def test_object_with_empty_dict_falls_back_to_str():
    class Empty:
        def __str__(self):
            return "empty-str"

    assert dpo(Empty()) == "empty-str"
    assert dpo(Empty(), current_depth=1) == "  empty-str"


def test_object_without_dict_falls_back_to_str():
    assert dpo(Slotted()) == "slotted-str"
    assert dpo(42, current_depth=1) == "  42"
    assert dpo("s") == "s"


def test_introspection_failure_falls_back_to_str():
    assert dpo(BadVars()) == "bad-vars-str"
