"""Utility modules for TraiGent SDK."""

# Traceability: CONC-Layer-Infra CONC-Quality-Maintainability CONC-Quality-Observability FUNC-ORCH-LIFECYCLE REQ-ORCH-003 SYNC-OptimizationFlow

from __future__ import annotations

# Callback utilities
from traigent.utils.callbacks import (
    CallbackManager,
    DetailedProgressCallback,
    LoggingCallback,
    OptimizationCallback,
    ProgressBarCallback,
    ProgressInfo,
    SimpleProgressCallback,
    StatisticsCallback,
    get_default_callbacks,
    get_detailed_callbacks,
    get_verbose_callbacks,
)

# Exception utilities
from traigent.utils.exceptions import (
    ConfigurationError,
    EvaluationError,
    OptimizationError,
    TraigentError,
)
from traigent.utils.exceptions import (
    ValidationError as ValidationException,  # Alias to avoid conflict
)

# Logging utilities
from traigent.utils.logging import get_logger, setup_logging

# Retry utilities (consolidated)
from traigent.utils.retry import (
    CLOUD_API_RETRY_CONFIG,
    DEFAULT_RETRY,
    HTTP_RETRY_CONFIG,
    NonRetryableError,
    RetryableError,
    RetryConfig,
    RetryHandler,
    RetryResult,
    RetryStrategy,
    retry,
    retry_with_config,
)

# Validation utilities (consolidated)
from traigent.utils.validation import (
    ConfigurationValidator,
    CoreValidators,
    DatasetValidator,
    OptimizationValidator,
    ValidationError,
    ValidationResult,
    Validators,
    validate_and_suggest,
    validate_config_space,
    validate_dataset_path,
    validate_objectives,
    validate_or_raise,
    validate_positive_int,
    validate_probability,
)

__all__ = [
    # Callbacks
    "OptimizationCallback",
    "ProgressInfo",
    "SimpleProgressCallback",
    "DetailedProgressCallback",
    "ProgressBarCallback",
    "LoggingCallback",
    "StatisticsCallback",
    "CallbackManager",
    "get_default_callbacks",
    "get_detailed_callbacks",
    "get_verbose_callbacks",
    # Exceptions
    "TraigentError",
    "ConfigurationError",
    "EvaluationError",
    "OptimizationError",
    "ValidationException",
    # Logging
    "get_logger",
    "setup_logging",
    # Validation
    "ValidationError",
    "ValidationResult",
    "Validators",
    "CoreValidators",
    "validate_config_space",
    "validate_objectives",
    "validate_dataset_path",
    "validate_positive_int",
    "validate_probability",
    "validate_or_raise",
    "OptimizationValidator",
    "ConfigurationValidator",
    "DatasetValidator",
    "validate_and_suggest",
    # Retry
    "RetryConfig",
    "RetryHandler",
    "RetryStrategy",
    "RetryResult",
    "RetryableError",
    "NonRetryableError",
    "retry",
    "retry_with_config",
    "DEFAULT_RETRY",
    "HTTP_RETRY_CONFIG",
    "CLOUD_API_RETRY_CONFIG",
]
