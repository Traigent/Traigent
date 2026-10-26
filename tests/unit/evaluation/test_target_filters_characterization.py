"""Golden characterization for validate_target_filters (messages pinned)."""

from __future__ import annotations

import pytest

from traigent.evaluation.target_filters import validate_target_filters

ALLOWED = "environment, release, name, tags"


def test_none_and_empty():
    assert validate_target_filters(None) == {}
    assert validate_target_filters({}) == {}


def test_valid_returns_copy_in_input_order():
    src = {"tags": ["b", "a"], "environment": ["prod"]}
    out = validate_target_filters(src)
    assert out == src
    assert list(out) == ["tags", "environment"]
    assert out["tags"] is not src["tags"]


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ([], "target_filters must be an object (mapping)"),
        ("x", "target_filters must be an object (mapping)"),
        (
            {"bogus": ["a"]},
            f"target_filters contains an unsupported key; allowed keys: {ALLOWED}",
        ),
        ({"tags": "a"}, "target_filters.tags must be a list of strings"),
        ({"tags": ("a",)}, "target_filters.tags must be a list of strings"),
        ({"tags": []}, "target_filters.tags must contain 1 to 20 items"),
        (
            {"tags": [str(i) for i in range(21)]},
            "target_filters.tags must contain 1 to 20 items",
        ),
        ({"tags": [1]}, "target_filters.tags items must be strings"),
        ({"tags": [""]}, "target_filters.tags items must be 1 to 128 characters"),
        (
            {"environment": ["x" * 65]},
            "target_filters.environment items must be 1 to 64 characters",
        ),
        ({"name": ["a", "a"]}, "target_filters.name items must be unique"),
        # order of checks: bad item type is reported before duplicates
        ({"name": ["a", "a", 1]}, "target_filters.name items must be strings"),
        # first failing key wins, earlier valid keys do not matter
        (
            {"name": ["a"], "release": "r", "bogus": []},
            "target_filters.release must be a list of strings",
        ),
    ],
)
def test_errors(value, message):
    with pytest.raises(ValueError) as exc:
        validate_target_filters(value)
    assert str(exc.value) == message


def test_boundary_lengths_accepted():
    ok = {"environment": ["x" * 64], "tags": [str(i) for i in range(20)]}
    assert validate_target_filters(ok) == ok
