"""Shared helper functions for walkthrough examples.

Common utilities for API key validation, logging setup, and environment configuration.
"""

import logging
import os

import traigent


# Estimated times for real examples (in seconds) - from test_all_examples.sh
EXAMPLE_ESTIMATED_TIMES: dict[str, int] = {
    "01_tuning_qa.py": 94,       # ~1m 34s
    "02_zero_code_change.py": 78,  # ~1m 18s
    "03_parameter_mode.py": 76,    # ~1m 16s
    "04_multi_objective.py": 63,   # ~1m 3s
    "05_rag_parallel.py": 55,      # ~0m 55s
    "06_custom_evaluator.py": 73,  # ~1m 13s
    "07_privacy_modes.py": 104,    # ~1m 44s
}


def _format_duration(seconds: int) -> str:
    """Format duration in human-readable form."""
    if seconds < 60:
        return f"~{seconds}s"
    minutes = seconds // 60
    secs = seconds % 60
    return f"~{minutes}m {secs}s"


def print_estimated_time(example_name: str) -> None:
    """Print estimated runtime for a real example.

    Skips output when running via test_all_examples.sh (which shows it already).

    Args:
        example_name: Name of the example file (e.g., "01_tuning_qa.py")
    """
    # Skip if running via shell script (which already shows estimated time)
    if os.getenv("TRAIGENT_BATCH_MODE", "").lower() in ("1", "true", "yes"):
        return
    estimated = EXAMPLE_ESTIMATED_TIMES.get(example_name)
    if estimated:
        print(f"Estimated time: {_format_duration(estimated)}")


def configure_logging() -> None:
    """Configure Traigent logging level from environment variable."""
    log_level = os.getenv("TRAIGENT_LOG_LEVEL", "WARNING").upper()
    traigent.configure(logging_level=log_level)


def is_valid_traigent_key(value: str) -> bool:
    """Validate Traigent API key format.

    Args:
        value: API key string to validate

    Returns:
        True if key matches expected format, False otherwise
    """
    prefix_lengths = {"tg_": 64, "uk_": 46}
    for prefix, expected_length in prefix_lengths.items():
        if value.startswith(prefix):
            if len(value) != expected_length:
                return False
            allowed = set(
                "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
            )
            return all(ch in allowed for ch in value[len(prefix) :])
    return False


def sanitize_traigent_api_key() -> None:
    """Remove invalid Traigent API keys from environment.

    Checks TRAIGENT_API_KEY and OPTIGEN_API_KEY; removes if invalid format.
    """
    key = os.getenv("TRAIGENT_API_KEY") or os.getenv("OPTIGEN_API_KEY")
    if key and not is_valid_traigent_key(key):
        print(
            "WARNING: Ignoring invalid TRAIGENT_API_KEY for this run. "
            "Set a valid Traigent key to enable cloud features."
        )
        os.environ.pop("TRAIGENT_API_KEY", None)
        os.environ.pop("OPTIGEN_API_KEY", None)


def require_openai_key(example_name: str) -> None:
    """Exit with error if OPENAI_API_KEY is not set.

    Args:
        example_name: Name of the example file (used in error message)

    Raises:
        SystemExit: If OPENAI_API_KEY environment variable is not set
    """
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit(
            "OPENAI_API_KEY not set. Export it to run real examples: "
            'export OPENAI_API_KEY="your-key". '
            f"To run without a key, use walkthrough/examples/mock/{example_name}."
        )


def setup_example_logger(name: str) -> logging.Logger:
    """Create a logger for walkthrough examples with simple formatting.

    Args:
        name: Logger name (typically the example module name)

    Returns:
        Configured logger instance
    """
    example_logger = logging.getLogger(f"traigent.walkthrough.{name}")
    if not example_logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        example_logger.addHandler(handler)
    example_logger.setLevel(logging.INFO)
    example_logger.propagate = False
    return example_logger
