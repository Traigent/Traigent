"""Tests for SDK logging policy helpers."""

import logging
import types

from traigent.utils.logging import configure_litellm_logging, setup_logging


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


def test_setup_logging_honors_traigent_log_level_env(monkeypatch):
    """The documented env knob must configure SDK loggers, not only LiteLLM."""
    root_logger = logging.getLogger()
    previous_level = root_logger.level
    previous_handlers = list(root_logger.handlers)
    litellm_logger = logging.getLogger("LiteLLM")
    previous_litellm_level = litellm_logger.level
    previous_litellm_propagate = litellm_logger.propagate
    monkeypatch.setenv("TRAIGENT_LOG_LEVEL", "DEBUG")

    try:
        setup_logging("WARNING")

        assert root_logger.level == logging.DEBUG
        assert litellm_logger.level == logging.DEBUG
        assert litellm_logger.propagate is True
    finally:
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        for handler in previous_handlers:
            root_logger.addHandler(handler)
        root_logger.setLevel(previous_level)
        litellm_logger.setLevel(previous_litellm_level)
        litellm_logger.propagate = previous_litellm_propagate


def test_setup_logging_default_still_clears_and_owns_root(monkeypatch):
    """Regression guard (Traigent/Traigent#1883): omitting ``logger_name`` must
    keep clearing + reconfiguring ROOT exactly as before — no default change."""
    root_logger = logging.getLogger()
    previous_level = root_logger.level
    previous_handlers = list(root_logger.handlers)
    monkeypatch.delenv("TRAIGENT_LOG_LEVEL", raising=False)

    sentinel_handler = logging.StreamHandler()
    root_logger.addHandler(sentinel_handler)

    try:
        setup_logging("ERROR")

        # The pre-existing root handler was removed (unchanged legacy behavior).
        assert sentinel_handler not in root_logger.handlers
        assert root_logger.level == logging.ERROR
        assert len(root_logger.handlers) == 1
    finally:
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        for handler in previous_handlers:
            root_logger.addHandler(handler)
        root_logger.setLevel(previous_level)


def test_setup_logging_with_logger_name_leaves_root_untouched(monkeypatch):
    """Traigent/Traigent#1883: opt-in ``logger_name`` scopes setup_logging() to a
    named logger tree and never touches ROOT — no removeHandler, no setLevel."""
    root_logger = logging.getLogger()
    previous_root_level = root_logger.level
    previous_root_handlers = list(root_logger.handlers)
    named_logger = logging.getLogger("traigent")
    previous_named_level = named_logger.level
    previous_named_handlers = list(named_logger.handlers)
    previous_named_propagate = named_logger.propagate
    monkeypatch.delenv("TRAIGENT_LOG_LEVEL", raising=False)

    # Simulate a host application that has already configured root logging.
    host_handler = logging.StreamHandler()
    root_logger.addHandler(host_handler)
    root_logger.setLevel(logging.WARNING)

    try:
        setup_logging("DEBUG", logger_name="traigent")

        # Host's root configuration survives untouched.
        assert host_handler in root_logger.handlers
        assert root_logger.level == logging.WARNING

        # The named logger was configured instead, with propagation off so
        # records are not re-emitted by the host's root handlers.
        assert named_logger.level == logging.DEBUG
        assert len(named_logger.handlers) == 1
        assert named_logger.propagate is False
    finally:
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        for handler in previous_root_handlers:
            root_logger.addHandler(handler)
        root_logger.setLevel(previous_root_level)

        for handler in named_logger.handlers[:]:
            named_logger.removeHandler(handler)
        for handler in previous_named_handlers:
            named_logger.addHandler(handler)
        named_logger.setLevel(previous_named_level)
        named_logger.propagate = previous_named_propagate


def test_setup_logging_with_logger_name_emits_each_record_exactly_once(monkeypatch):
    """Traigent/Traigent#1883 rework: with a host root handler installed AND
    setup_logging scoped to "traigent", each record is emitted exactly once —
    by the scoped handler — never a second time via propagation to root."""
    root_logger = logging.getLogger()
    previous_root_level = root_logger.level
    previous_root_handlers = list(root_logger.handlers)
    named_logger = logging.getLogger("traigent")
    previous_named_level = named_logger.level
    previous_named_handlers = list(named_logger.handlers)
    previous_named_propagate = named_logger.propagate
    monkeypatch.delenv("TRAIGENT_LOG_LEVEL", raising=False)

    class CountingHandler(logging.Handler):
        def __init__(self) -> None:
            super().__init__()
            self.records: list[logging.LogRecord] = []

        def emit(self, record: logging.LogRecord) -> None:
            self.records.append(record)

    host_root_handler = CountingHandler()
    root_logger.addHandler(host_root_handler)
    root_logger.setLevel(logging.INFO)

    try:
        setup_logging("INFO", logger_name="traigent")

        # Swap Traigent's own StreamHandler for a counting one so we can
        # observe scoped emissions without writing to stderr.
        scoped_handler = CountingHandler()
        for handler in named_logger.handlers[:]:
            named_logger.removeHandler(handler)
        named_logger.addHandler(scoped_handler)

        logging.getLogger("traigent.some.module").info("hello once")

        total_emissions = len(scoped_handler.records) + len(host_root_handler.records)
        assert total_emissions == 1
        assert len(scoped_handler.records) == 1  # emitted by the scoped tree
        assert len(host_root_handler.records) == 0  # never bubbled to root
    finally:
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        for handler in previous_root_handlers:
            root_logger.addHandler(handler)
        root_logger.setLevel(previous_root_level)

        for handler in named_logger.handlers[:]:
            named_logger.removeHandler(handler)
        for handler in previous_named_handlers:
            named_logger.addHandler(handler)
        named_logger.setLevel(previous_named_level)
        named_logger.propagate = previous_named_propagate
