"""Options that were accepted and ignored must take effect (#3126)."""

from __future__ import annotations

import logging

import pytest

from traigent.integrations.utils.discovery import ParameterDiscovery
from traigent.utils.logging import setup_logging


@pytest.fixture
def restore_root_logging():
    root = logging.getLogger()
    handlers, level = root.handlers[:], root.level
    yield
    root.handlers[:] = handlers
    root.setLevel(level)


def test_use_rich_installs_a_rich_handler(restore_root_logging) -> None:
    from rich.logging import RichHandler

    setup_logging("INFO", use_rich=True)
    assert [type(h) for h in logging.getLogger().handlers] == [RichHandler]
    setup_logging("INFO", use_rich=False)
    assert [type(h) for h in logging.getLogger().handlers] == [logging.StreamHandler]


@pytest.mark.parametrize(
    ("threshold", "expected"), [(0.0, "temperature_c"), (0.5, None), (1.0, None)]
)
def test_find_similar_parameters_threshold_applies(threshold, expected) -> None:
    got = ParameterDiscovery.find_similar_parameters(
        "tempe", ["temperature_c"], threshold=threshold
    )
    assert got == expected


def test_find_similar_parameters_default_keeps_substring_matching() -> None:
    assert (
        ParameterDiscovery.find_similar_parameters("tempe", ["temperature_c"])
        == "temperature_c"
    )
