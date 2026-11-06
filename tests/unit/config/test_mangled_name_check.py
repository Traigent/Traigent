"""``_is_mangled_name`` must keep the semantics of the regex it replaced."""

import re

import pytest

from traigent.config.providers import _is_mangled_name

# The pre-refactor pattern (flagged super-linear by Sonar S8786).
_OLD = re.compile(r"^_\w+?__\w*(?<!__)$")


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("_Cls__attr", True),
        ("_C__a", True),
        ("_Cls__a__b", True),
        ("_a1__b2", True),
        ("__x", False),  # leading dunder only, no inner "__"
        ("_x__", False),  # trailing dunder
        ("__init__", False),
        ("__x__y__", False),  # ends with "__"
        ("_x_y", False),  # single underscores only
        ("x__y", False),  # no leading underscore
        ("_", False),
        ("", False),
        ("_a__", False),
        ("_a-b__c", False),  # non-word character
    ],
)
def test_is_mangled_name_matches_old_regex(name: str, expected: bool) -> None:
    assert _is_mangled_name(name) is expected
    assert bool(_OLD.match(name)) is expected


def test_is_mangled_name_is_linear_on_long_names() -> None:
    assert _is_mangled_name("_" + "a" * 200_000 + "__b") is True
    assert _is_mangled_name("_" + "_" * 200_000) is False
