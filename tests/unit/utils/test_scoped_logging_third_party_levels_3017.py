"""#3017: scoped setup_logging must not reset the host's httpx/urllib3 levels.

``import traigent`` with TRAIGENT_LOG_LEVEL set calls
``setup_logging(level, logger_name="traigent")``, documented to leave the host's
logging configuration untouched. The legacy root path keeps its behaviour.
"""

from __future__ import annotations

import logging

import pytest

from traigent.utils.logging import apply_env_logging_level, setup_logging

_THIRD_PARTY = ("httpx", "urllib3")


@pytest.fixture
def restore_loggers():
    names = (*_THIRD_PARTY, "traigent", "")
    saved = {
        n: (
            logging.getLogger(n).level,
            list(logging.getLogger(n).handlers),
            logging.getLogger(n).propagate,
        )
        for n in names
    }
    yield
    for n, (level, handlers, propagate) in saved.items():
        lg = logging.getLogger(n)
        lg.setLevel(level)
        lg.handlers[:] = handlers
        lg.propagate = propagate


def test_scoped_setup_keeps_host_third_party_levels(restore_loggers):
    logging.getLogger("httpx").setLevel(logging.DEBUG)
    logging.getLogger("urllib3").setLevel(logging.INFO)
    setup_logging("DEBUG", logger_name="traigent")
    assert logging.getLogger("httpx").level == logging.DEBUG
    assert logging.getLogger("urllib3").level == logging.INFO


def test_env_level_at_import_keeps_host_third_party_levels(
    restore_loggers, monkeypatch
):
    monkeypatch.setenv("TRAIGENT_LOG_LEVEL", "DEBUG")
    logging.getLogger("httpx").setLevel(logging.DEBUG)
    logging.getLogger("urllib3").setLevel(logging.INFO)
    apply_env_logging_level()
    assert logging.getLogger("httpx").level == logging.DEBUG
    assert logging.getLogger("urllib3").level == logging.INFO


def test_root_setup_still_quiets_third_party(restore_loggers):
    logging.getLogger("httpx").setLevel(logging.DEBUG)
    logging.getLogger("urllib3").setLevel(logging.DEBUG)
    setup_logging("DEBUG")
    assert logging.getLogger("httpx").level == logging.WARNING
    assert logging.getLogger("urllib3").level == logging.WARNING
