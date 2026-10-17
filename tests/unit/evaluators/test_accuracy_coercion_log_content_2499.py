"""#2499: accuracy-coercion diagnostics name types, never example content.

Both coercion branches of ``_accuracy_values_match`` logged the raw output
(and, for numeric strings, the expected label) with ``%r``. Console logs are
retained by terminals and CI, so the message keeps the conversion category
and target type but omits the values. Equality decisions are unchanged.
"""

from __future__ import annotations

import logging

import pytest

from traigent.evaluators.base import _accuracy_values_match

_ACTUAL_MARKER = "ZZ_ACTUAL_SECRET_MARKER"
_EXPECTED_MARKER = "1.0"


@pytest.fixture
def records(caplog: pytest.LogCaptureFixture) -> pytest.LogCaptureFixture:
    caplog.set_level(logging.DEBUG, logger="traigent.evaluators.base")
    return caplog


def test_json_string_coercion_omits_output_content(records) -> None:
    actual = f'{{"label": "{_ACTUAL_MARKER}"}}'
    assert _accuracy_values_match(actual, {"label": _ACTUAL_MARKER}) is True
    coercion = [r for r in records.records if "Coercing" in r.getMessage()]
    assert coercion, "the conversion stays observable"
    assert "dict" in coercion[0].getMessage()
    assert _ACTUAL_MARKER not in records.text


def test_numeric_string_coercion_omits_output_and_expected(records) -> None:
    assert _accuracy_values_match("1.000000000000000042", "1.0") is True
    coercion = [r for r in records.records if "Coercing" in r.getMessage()]
    assert coercion and "numeric" in coercion[0].getMessage()
    assert "1.000000000000000042" not in records.text
    assert "'1.0'" not in records.text
