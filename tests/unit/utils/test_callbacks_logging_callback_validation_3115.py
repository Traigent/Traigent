"""LoggingCallback validates log_level and sanitizes the algorithm (#3115)."""

from __future__ import annotations

import logging

import pytest

from traigent.utils.callbacks import LoggingCallback


# --- #3115 -----------------------------------------------------------------
@pytest.mark.parametrize("bad", ["bogus", None, 5, ""])
def test_logging_callback_rejects_bad_log_level(bad: object) -> None:
    with pytest.raises(ValueError, match="log_level"):
        LoggingCallback(logging.getLogger("x"), bad)  # type: ignore[arg-type]


def test_logging_callback_accepts_level_names_case_insensitively() -> None:
    LoggingCallback(logging.getLogger("x"), "warning")


def test_logging_callback_sanitizes_algorithm(
    caplog: pytest.LogCaptureFixture,
) -> None:
    lg = logging.getLogger("x_3115")
    cb = LoggingCallback(lg)
    with caplog.at_level(logging.INFO, logger="x_3115"):
        cb.on_optimization_start({}, ["a"], "g\r\nforged")
    msgs = [r.getMessage() for r in caplog.records if r.name == "x_3115"]
    assert msgs
    assert all("\r" not in m and "\n" not in m for m in msgs)
