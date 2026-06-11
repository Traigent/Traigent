"""Tests for SDK logging policy helpers."""

import logging
import types

from traigent.utils.logging import configure_litellm_logging


def test_litellm_logging_quiet_by_default(monkeypatch):
    """Normal SDK logging suppresses LiteLLM banners and duplicate propagation."""
    monkeypatch.delenv("TRAIGENT_LOG_LEVEL", raising=False)
    litellm_module = types.ModuleType("litellm")
    litellm_module.suppress_debug_info = False
    litellm_logger = logging.getLogger("LiteLLM")
    previous_level = litellm_logger.level
    previous_propagate = litellm_logger.propagate

    try:
        configure_litellm_logging(litellm_module=litellm_module)

        assert litellm_module.suppress_debug_info is True
        assert litellm_logger.level == logging.WARNING
        assert litellm_logger.propagate is False
    finally:
        litellm_logger.setLevel(previous_level)
        litellm_logger.propagate = previous_propagate


def test_litellm_logging_debug_opt_in_restores_verbose_logging(monkeypatch):
    """TRAIGENT_LOG_LEVEL=DEBUG opts back into verbose LiteLLM diagnostics."""
    monkeypatch.setenv("TRAIGENT_LOG_LEVEL", "DEBUG")
    litellm_module = types.ModuleType("litellm")
    litellm_module.suppress_debug_info = True
    litellm_logger = logging.getLogger("LiteLLM")
    previous_level = litellm_logger.level
    previous_propagate = litellm_logger.propagate

    try:
        configure_litellm_logging(litellm_module=litellm_module)

        assert litellm_module.suppress_debug_info is False
        assert litellm_logger.level == logging.DEBUG
        assert litellm_logger.propagate is True
    finally:
        litellm_logger.setLevel(previous_level)
        litellm_logger.propagate = previous_propagate
