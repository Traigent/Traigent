"""#2790: the minimal_logging warning must fire only for a user-chosen value."""

from __future__ import annotations

import logging

import pytest

import traigent
from traigent.api.decorators import ExecutionOptions

_MARKER = "minimal_logging is only effective"


def _decorate(**extra):
    @traigent.optimize(
        configuration_space={"temperature": [0.0, 0.7]},
        objectives=["accuracy"],
        **extra,
    )
    def answer(question: str) -> str:
        return question

    return answer


def _warned(caplog: pytest.LogCaptureFixture) -> bool:
    return any(_MARKER in r.getMessage() for r in caplog.records)


def test_default_decoration_logs_no_minimal_logging_warning(caplog):
    with caplog.at_level(logging.WARNING):
        _decorate()
    assert not _warned(caplog)


def test_explicit_true_on_managed_path_still_warns_kwarg(caplog):
    with caplog.at_level(logging.WARNING):
        _decorate(minimal_logging=True)
    assert _warned(caplog)


def test_explicit_true_on_managed_path_still_warns_bundle(caplog):
    with caplog.at_level(logging.WARNING):
        _decorate(execution=ExecutionOptions(minimal_logging=True))
    assert _warned(caplog)


def test_explicit_false_does_not_warn(caplog):
    with caplog.at_level(logging.WARNING):
        _decorate(minimal_logging=False)
    assert not _warned(caplog)


def test_explicit_true_on_local_path_does_not_warn(caplog):
    with caplog.at_level(logging.WARNING):
        _decorate(minimal_logging=True, offline=True, algorithm="grid")
    assert not _warned(caplog)
