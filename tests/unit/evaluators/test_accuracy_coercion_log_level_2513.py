"""#2513: routine accuracy coercion is not a WARNING.

A JSON-text answer compared against a dict gold label is the documented
structured-output path. Logged at WARNING once per example it was 94% of a
customer run log; it is now DEBUG. Equality decisions are unchanged.
"""

from __future__ import annotations

import logging

import pytest

from traigent.evaluators.base import _accuracy_values_match


@pytest.mark.parametrize(
    ("actual", "expected"),
    [
        pytest.param('{"a": 1}', {"a": 1}, id="json-to-dict"),
        pytest.param("[1, 2]", [1, 2], id="json-to-list"),
        pytest.param("1.0", "1", id="numeric-strings"),
    ],
)
def test_coercion_is_debug_not_warning(
    actual: str, expected: object, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG, logger="traigent.evaluators.base")
    assert _accuracy_values_match(actual, expected) is True
    coercion = [r for r in caplog.records if "Coercing" in r.getMessage()]
    assert coercion
    assert all(r.levelno == logging.DEBUG for r in coercion)
