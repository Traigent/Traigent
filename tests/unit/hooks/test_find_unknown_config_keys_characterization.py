"""Golden characterization for find_unknown_config_keys."""

from __future__ import annotations

import pytest

from traigent.hooks.config import find_unknown_config_keys


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        (None, []),
        ([], []),
        ("x", []),
        ({}, []),
        ({"validation": {"enabled": True}}, []),
        ({"validatoin": 1}, [("validatoin", "validation")]),
        ({"zzz": 1}, [("zzz", None)]),
        (
            {"validation": {"enabld": 1, "hooks": 2}},
            [("validation.enabld", "validation.enabled")],
        ),
        # known key whose path has no sub-schema is not descended into
        ({"validation": {"hooks": {"anything": 1}}}, []),
        # known key with a non-dict value: walk returns without findings
        ({"validation": "text", "constraints": [1]}, []),
        (
            {"constraints": {"cost": {"max_cost_per_qeury": 1}, "bogus": 1}},
            [
                (
                    "constraints.cost.max_cost_per_qeury",
                    "constraints.cost.max_cost_per_query",
                ),
                ("constraints.bogus", None),
            ],
        ),
        # non-string keys are stringified
        ({1: 2}, [("1", None)]),
        # unknown key's value is never descended into
        ({"zzz": {"validation": {"qqq": 1}}}, [("zzz", None)]),
    ],
)
def test_find_unknown_config_keys_golden(data, expected):
    assert find_unknown_config_keys(data) == expected
