"""Core constants and configuration values for Traigent optimization system.

This module centralizes all magic numbers, default values, and configuration
constants used throughout the core architecture. This promotes consistency
and makes it easier to modify system-wide defaults.
"""

# Traceability: CONC-Layer-Core CONC-Quality-Maintainability FUNC-ORCH-LIFECYCLE REQ-ORCH-003 SYNC-OptimizationFlow

from __future__ import annotations

from typing import Any

# =============================================================================
# MEMORY AND HISTORY LIMITS
# =============================================================================

# Maximum items in history collections to prevent unbounded memory growth
MAX_HISTORY_SIZE = 1000  # Default for most history collections
MAX_HISTORY_SIZE_LARGE = 5000  # For high-frequency data collection
MAX_HISTORY_SIZE_SMALL = 100  # For low-frequency or memory-constrained contexts

# Analytics-specific limits
MAX_ALERT_HISTORY_SIZE = 1000  # Maximum alerts to retain
MAX_OPTIMIZATION_HISTORY_SIZE = 1000  # Maximum optimization records
MAX_PERFORMANCE_HISTORY_SIZE = 1000  # Maximum performance data points
MAX_USAGE_HISTORY_SIZE = 10000  # Maximum usage metrics (high frequency)
MAX_REGRESSION_HISTORY_SIZE = 500  # Maximum regression analysis records

# Pruning ratio - when limit is reached, remove this fraction of oldest entries
HISTORY_PRUNE_RATIO = 0.1  # Remove 10% of oldest items when pruning

# =============================================================================
# TIMING AND PERFORMANCE CONSTANTS
# =============================================================================

# Default timeout for various operations (seconds)
DEFAULT_TIMEOUT = 60.0

# Default execution time for operations without timing (seconds)
DEFAULT_EXECUTION_TIME = 0.1

# Maximum number of retries for transient failures
MAX_RETRIES = 3

# Small epsilon value for floating point comparisons
EPSILON = 1e-10

# =============================================================================
# LLM AND MODEL CONSTANTS
# =============================================================================

# Default LLM model for optimization
DEFAULT_MODEL = "gpt-4o-mini"

# Default prompt style for optimization
DEFAULT_PROMPT_STYLE = "direct"

# Default temperature for LLM calls (deterministic)
DEFAULT_TEMPERATURE = 0.0

# Default maximum tokens for LLM responses
DEFAULT_MAX_TOKENS = 1000

# Maximum tokens for teach-style prompts (higher for explanations)
TEACH_MAX_TOKENS = 1500

# =============================================================================
# CONFIGURATION AND VALIDATION CONSTANTS
# =============================================================================

# Minimum weight value for objectives
MIN_OBJECTIVE_WEIGHT = 0.0

# Default objective weight when not specified
DEFAULT_OBJECTIVE_WEIGHT = 1.0

# Maximum number of test questions to use for validation
MAX_TEST_QUESTIONS = 10

# Default number of test questions for quick validation
DEFAULT_TEST_QUESTIONS = 2

# =============================================================================
# FILE AND PATH CONSTANTS
# =============================================================================

# Default file extension for dataset files
DATASET_FILE_EXTENSION = ".jsonl"

# Default configuration file name
CONFIG_FILE_NAME = "traigent_config.yaml"

# Default log directory name
LOG_DIRECTORY = "optimization_logs"

# =============================================================================
# METRICS AND AGGREGATION CONSTANTS
# =============================================================================

# Token metrics to aggregate
TOKEN_METRICS = ["input_tokens", "output_tokens", "total_tokens"]

# Cost metrics to aggregate
COST_METRICS = ["input_cost", "output_cost", "total_cost"]

# Response time metric
RESPONSE_TIME_METRIC = "avg_response_time"

# Aggregation methods available
AGGREGATION_METHODS = [
    "sum",  # Sum all values
    "mean",  # Average of values
    "max",  # Maximum value
    "min",  # Minimum value
    "median",  # Median value
]

# =============================================================================
# VALIDATION AND ERROR CONSTANTS
# =============================================================================

# Maximum length for error messages in logs
MAX_ERROR_MESSAGE_LENGTH = 500

# Validation error codes
VALIDATION_ERROR_CODES = {
    "MISSING_PARAMETER": "MISSING_PARAMETER",
    "INVALID_BOUNDS": "INVALID_BOUNDS",
    "DUPLICATE_NAME": "DUPLICATE_NAME",
    "INVALID_TYPE": "INVALID_TYPE",
    "CONSTRAINT_VIOLATION": "CONSTRAINT_VIOLATION",
}

# =============================================================================
# EXECUTION MODE CONSTANTS
# =============================================================================

# Historical execution mode constants. Current support validation lives in
# traigent.config.types.validate_execution_mode.
EXECUTION_MODES = [
    "edge_analytics",  # Client-side execution only
    "privacy",  # Legacy alias handled by config normalization
    "standard",  # Removed legacy mode
    "cloud",  # Reserved future remote execution
]

# Default execution mode
DEFAULT_EXECUTION_MODE = "edge_analytics"

# =============================================================================
# LOGGING AND DEBUGGING CONSTANTS
# =============================================================================

# Log levels
LOG_LEVELS = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

# Default log level
DEFAULT_LOG_LEVEL = "INFO"

# Debug mode flag
DEBUG_MODE = False

# =============================================================================
# TYPE-SAFE CONSTANT GROUPS
# =============================================================================

# Group related constants for better organization
TIMING_CONSTANTS = {
    "default_timeout": DEFAULT_TIMEOUT,
    "default_execution_time": DEFAULT_EXECUTION_TIME,
    "max_retries": MAX_RETRIES,
    "epsilon": EPSILON,
}

LLM_CONSTANTS = {
    "default_model": DEFAULT_MODEL,
    "default_temperature": DEFAULT_TEMPERATURE,
    "default_max_tokens": DEFAULT_MAX_TOKENS,
    "teach_max_tokens": TEACH_MAX_TOKENS,
}

VALIDATION_CONSTANTS = {
    "min_objective_weight": MIN_OBJECTIVE_WEIGHT,
    "default_objective_weight": DEFAULT_OBJECTIVE_WEIGHT,
    "max_test_questions": MAX_TEST_QUESTIONS,
    "default_test_questions": DEFAULT_TEST_QUESTIONS,
    "max_error_message_length": MAX_ERROR_MESSAGE_LENGTH,
}

MEMORY_CONSTANTS = {
    "max_history_size": MAX_HISTORY_SIZE,
    "max_history_size_large": MAX_HISTORY_SIZE_LARGE,
    "max_history_size_small": MAX_HISTORY_SIZE_SMALL,
    "max_alert_history_size": MAX_ALERT_HISTORY_SIZE,
    "max_optimization_history_size": MAX_OPTIMIZATION_HISTORY_SIZE,
    "max_performance_history_size": MAX_PERFORMANCE_HISTORY_SIZE,
    "max_usage_history_size": MAX_USAGE_HISTORY_SIZE,
    "max_regression_history_size": MAX_REGRESSION_HISTORY_SIZE,
    "history_prune_ratio": HISTORY_PRUNE_RATIO,
}

# =============================================================================
# UTILITY FUNCTIONS FOR CONSTANTS
# =============================================================================


def get_constant_group(group_name: str) -> dict[str, Any]:
    """Get a group of related constants.

    Args:
        group_name: Name of the constant group to retrieve

    Returns:
        Dictionary of constants in the specified group

    Raises:
        ValueError: If the group name is not recognized
    """
    groups: dict[str, dict[str, Any]] = {
        "timing": TIMING_CONSTANTS,
        "llm": LLM_CONSTANTS,
        "validation": VALIDATION_CONSTANTS,
        "memory": MEMORY_CONSTANTS,
    }

    if group_name not in groups:
        available_groups = ", ".join(groups.keys())
        raise ValueError(
            f"Unknown constant group '{group_name}'. Available groups: {available_groups}"
        )

    return groups[group_name]


def validate_constant_range(
    value: float, min_val: float, max_val: float, name: str
) -> None:
    """Validate that a constant value is within an acceptable range.

    Args:
        value: Value to validate
        min_val: Minimum acceptable value
        max_val: Maximum acceptable value
        name: Name of the constant for error messages

    Raises:
        ValueError: If the value is outside the acceptable range
    """
    if not (min_val <= value <= max_val):
        raise ValueError(
            f"Constant '{name}' value {value} is outside acceptable range [{min_val}, {max_val}]"
        )


def get_model_config(model_name: str) -> dict[str, Any]:
    """Get configuration for a specific model.

    Args:
        model_name: Name of the model to get config for

    Returns:
        Dictionary with model configuration
    """
    base_config = {"temperature": DEFAULT_TEMPERATURE, "max_tokens": DEFAULT_MAX_TOKENS}

    # Model-specific overrides
    model_overrides = {
        "gpt-4o": {
            "max_tokens": TEACH_MAX_TOKENS  # GPT-4o can handle longer responses
        },
        "gpt-4o-mini": {"max_tokens": DEFAULT_MAX_TOKENS},
        "gpt-3.5-turbo": {"max_tokens": 1000},  # GPT-3.5 has lower token limits
    }

    if model_name in model_overrides:
        base_config.update(model_overrides[model_name])

    return base_config
