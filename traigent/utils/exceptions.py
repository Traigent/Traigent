"""Custom exceptions for Traigent SDK."""

# Traceability: CONC-Layer-Infra CONC-Quality-Reliability CONC-Quality-Security FUNC-ORCH-LIFECYCLE FUNC-CLOUD-HYBRID FUNC-SECURITY REQ-ORCH-003 REQ-CLOUD-009 REQ-SEC-010 SYNC-OptimizationFlow SYNC-CloudHybrid

from __future__ import annotations

import os
import sys
from typing import Any


def _traigent_excepthook(exc_type, exc_value, exc_tb):
    """Custom exception hook for clean ConfigurationError output.

    Shows just the error message without full traceback for ConfigurationError,
    unless TRAIGENT_DEBUG is set.
    """
    if issubclass(exc_type, ConfigurationError) and not os.getenv("TRAIGENT_DEBUG"):
        # Clean output: just the exception type and message
        print(
            f"{exc_type.__module__}.{exc_type.__name__}: {exc_value}", file=sys.stderr
        )
        sys.exit(1)
    else:
        # Default behavior for other exceptions
        sys.__excepthook__(exc_type, exc_value, exc_tb)


# Install custom hook (only if not already customized)
if sys.excepthook is sys.__excepthook__:
    sys.excepthook = _traigent_excepthook


class TraigentError(Exception):
    """Base exception for all Traigent-related errors."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


class ConfigurationError(TraigentError):
    """Error in optimization configuration.

    This error is raised when:
    - Configuration values are invalid or malformed
    - Configuration features are not yet supported (e.g., cloud/hybrid execution modes)
    - Required configuration is missing

    Set TRAIGENT_DEBUG=1 to see full tracebacks.
    """


class ProviderValidationError(ConfigurationError):
    """Raised when provider API key validation fails.

    This error is raised when one or more provider API keys are invalid,
    missing, or when the provider SDK is not installed. Validation happens
    before optimization runs to fail fast and avoid wasted API costs.

    Attributes:
        failed_providers: List of (provider, error_type) tuples that failed.
        details: Dict with additional error context.

    Example:
        >>> from traigent.providers import validate_providers
        >>> results = validate_providers(["gpt-4o-mini", "claude-3-haiku-20240307"])
        >>> # If validation fails, ProviderValidationError is raised with:
        >>> # - List of failed providers and error types
        >>> # - Instructions for fixing the issue

    To skip provider validation:
        - Set TRAIGENT_SKIP_PROVIDER_VALIDATION=true in environment
        - Or use validate_providers=False in the @traigent.optimize decorator
    """

    def __init__(
        self,
        message: str,
        failed_providers: list[tuple[str, str]] | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, details)
        self.failed_providers = failed_providers or []


class ValidationError(TraigentError):
    """Error in input validation."""


class TraigentValidationError(ValidationError):
    """Enhanced validation error with helpful suggestions."""


class TVLValidationError(ValidationError):
    """Raised when a TVL specification or constraint is invalid."""


class TVLConstraintError(TVLValidationError):
    """Raised when a TVL constraint fails at runtime."""


class ClientError(TraigentError):
    """Error in client communication or configuration."""

    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, details)
        self.status_code = status_code


class StandardizedClientError(ClientError):
    """Standardized client error with consistent format."""


class AuthenticationError(TraigentError):
    """Error in authentication or authorization."""


class JWTValidationError(ValidationError):
    """Error in JWT token validation."""


class ConfigurationSpaceError(ConfigurationError):
    """Error in configuration space definition."""


class DatasetValidationError(ValidationError):
    """Error in dataset validation."""


class ObjectiveValidationError(ValidationError):
    """Error in objective validation."""


class InvocationError(TraigentError):
    """Error during function invocation."""

    def __init__(
        self,
        message: str,
        config: dict[str, Any] | None = None,
        input_data: dict[str, Any] | None = None,
        original_error: Exception | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, details)
        self.config = config
        self.input_data = input_data
        self.original_error = original_error


class EvaluationError(TraigentError):
    """Error during function evaluation."""

    def __init__(
        self,
        message: str,
        config: dict[str, Any] | None = None,
        original_error: Exception | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, details)
        self.config = config
        self.original_error = original_error


class OptimizationError(TraigentError):
    """Error in optimization process."""


class PluginError(TraigentError):
    """Error in plugin system."""


class PluginVersionError(PluginError):
    """Error raised when plugin version is incompatible with Traigent version.

    Example:
        raise PluginVersionError(
            plugin_name="my-plugin",
            plugin_version="1.0.0",
            required_traigent_version="2.0.0",
            current_traigent_version="1.5.0"
        )
    """

    def __init__(
        self,
        plugin_name: str,
        plugin_version: str,
        required_traigent_version: str,
        current_traigent_version: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        message = (
            f"Plugin '{plugin_name}' v{plugin_version} requires Traigent >= "
            f"{required_traigent_version}, but current version is {current_traigent_version}. "
            f"Please upgrade Traigent: pip install --upgrade traigent"
        )
        super().__init__(message, details)
        self.plugin_name = plugin_name
        self.plugin_version = plugin_version
        self.required_traigent_version = required_traigent_version
        self.current_traigent_version = current_traigent_version


class StorageError(TraigentError):
    """Error in storage operations."""


class TraigentStorageError(StorageError):
    """Specific storage error for Traigent Edge Analytics mode."""


class ServiceError(TraigentError):
    """Error in remote service operations."""

    def __init__(
        self,
        message: str,
        service_name: str | None = None,
        endpoint: str | None = None,
        status_code: int | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, details)
        self.service_name = service_name
        self.endpoint = endpoint
        self.status_code = status_code


class TraigentConnectionError(ServiceError):
    """Error connecting to remote service."""


class ServiceUnavailableError(ServiceError):
    """Remote service is temporarily unavailable."""


class QuotaExceededError(ServiceError):
    """Service quota or rate limit exceeded."""


class SessionError(TraigentError):
    """Error in session management."""


class AgentExecutionError(TraigentError):
    """Error during agent execution."""


class PlatformCapabilityError(TraigentError):
    """Raised when a platform doesn't support a requested capability."""


class RetryError(TraigentError):
    """Error in retry operations."""


class RetryableError(TraigentError):
    """Error that should trigger a retry.

    This is the canonical definition - use this instead of local definitions
    in retry.py or resilient_client.py.
    """

    def __init__(
        self,
        message: str,
        retry_after: float | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, details)
        self.retry_after = retry_after


class NonRetryableError(TraigentError):
    """Error that should not trigger a retry.

    This is the canonical definition - use this instead of local definitions
    in retry.py or resilient_client.py.
    """


class RateLimitError(RetryableError):
    """Rate limit exceeded error."""

    def __init__(
        self, message: str = "Rate limit exceeded", retry_after: float | None = None
    ) -> None:
        super().__init__(message, retry_after=retry_after)


class CostLimitExceeded(TraigentError):
    """Raised when accumulated cost exceeds the configured limit."""

    def __init__(self, accumulated: float, limit: float) -> None:
        super().__init__(f"Cost limit exceeded: ${accumulated:.2f} >= ${limit:.2f} USD")
        self.accumulated = accumulated
        self.limit = limit


class VendorPauseError(TraigentError):
    """Raised when a vendor error is classified as pause-worthy.

    Re-raised from TrialLifecycle._execute_trial_with_tracing to signal the
    orchestrator that the user should be prompted before continuing.

    Attributes:
        original_error: The underlying vendor exception.
        category: The VendorErrorCategory enum value (or None).
    """

    def __init__(
        self,
        message: str,
        original_error: Exception | None = None,
        category: Any = None,
    ) -> None:
        super().__init__(message)
        self.original_error = original_error
        self.category = category


class NetworkError(RetryableError):
    """Network-related error."""


class TraigentTimeoutError(TraigentError):
    """Operation timed out."""


class TrialPrunedError(TraigentError):
    """Signal that a trial has been pruned early.

    Attributes:
        step: The evaluation step at which pruning occurred.
        example_results: Partial example results collected before pruning.
            This allows capturing metrics from examples that were evaluated
            before the trial was pruned.
    """

    def __init__(
        self,
        message: str = "Trial pruned",
        step: int | None = None,
        example_results: list | None = None,
    ) -> None:
        super().__init__(message)
        self.step = step
        self.example_results = example_results or []


class FeatureNotAvailableError(TraigentError):
    """Error raised when a feature requires an uninstalled plugin.

    This error provides helpful guidance on which plugin to install
    to enable the requested feature.

    Example:
        raise FeatureNotAvailableError(
            "Parallel execution",
            plugin_name="traigent-parallel",
            install_hint="pip install traigent[ml]"
        )
    """

    def __init__(
        self,
        feature_name: str,
        plugin_name: str | None = None,
        install_hint: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        # Build helpful error message
        message = f"Feature '{feature_name}' is not available."
        if plugin_name:
            message += f" Requires the '{plugin_name}' plugin."
        if install_hint:
            message += f" Install with: {install_hint}"

        super().__init__(message, details)
        self.feature_name = feature_name
        self.plugin_name = plugin_name
        self.install_hint = install_hint


# =============================================================================
# Data Integrity and Validation Errors
# =============================================================================


class DataIntegrityError(TraigentError):
    """Data corruption or invalid conversion detected.

    Raised when data validation fails or conversion produces invalid results.
    Unlike silent failures that default to 0.0, this exception ensures data
    corruption is caught immediately.

    Attributes:
        message: Error description
        field: Optional field name that failed validation
        value: Optional invalid value that caused the error
        details: Additional context about the error
    """

    def __init__(
        self,
        message: str,
        field: str | None = None,
        value: Any = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, context)
        self.field = field
        self.value = value


class MetricExtractionError(DataIntegrityError):
    """Metric extraction or conversion failed.

    Raised when extracting metrics from function outputs fails due to:
    - Invalid types (non-numeric values)
    - NaN or Inf values
    - Missing required fields
    - Type conversion failures

    This prevents silent data corruption where invalid metrics are converted
    to 0.0 and corrupt optimization results.

    Attributes:
        message: Error description
        field: Metric field name that failed
        value: Invalid value that caused the error
        trial_id: Optional trial identifier
        example_id: Optional example identifier
        original_error: Optional underlying exception

    Example:
        raise MetricExtractionError(
            "Cannot convert metric 'accuracy' to numeric",
            field="accuracy",
            value="invalid",
            trial_id="trial-123",
            example_id="example-456",
        )
    """

    def __init__(
        self,
        message: str,
        field: str,
        value: Any,
        trial_id: str | None = None,
        example_id: str | None = None,
        original_error: Exception | None = None,
    ) -> None:
        context = {
            "trial_id": trial_id,
            "example_id": example_id,
            "original_error": str(original_error) if original_error else None,
        }
        super().__init__(message, field, value, context)
        self.trial_id = trial_id
        self.example_id = example_id
        self.original_error = original_error


class DTOSerializationError(DataIntegrityError):
    """DTO serialization or validation failed.

    Raised when a Data Transfer Object fails validation or produces invalid
    serialized data. This ensures DTOs maintain data integrity before sending
    to the backend.

    Attributes:
        message: Error description
        dto_class: Name of the DTO class that failed
        dto_id: Optional DTO identifier
        invalid_fields: Optional dict of field names to invalid values

    Example:
        raise DTOSerializationError(
            "ConfigurationRunDTO validation failed",
            dto_class="ConfigurationRunDTO",
            dto_id="config-run-123",
            invalid_fields={"measures": ["invalid", "types"]},
        )
    """

    def __init__(
        self,
        message: str,
        dto_class: str,
        dto_id: str | None = None,
        invalid_fields: dict[str, Any] | None = None,
    ) -> None:
        context = {
            "dto_class": dto_class,
            "dto_id": dto_id,
            "invalid_fields": invalid_fields,
        }
        super().__init__(message, context=context)
        self.dto_class = dto_class
        self.dto_id = dto_id
        self.invalid_fields = invalid_fields


# =============================================================================
# Warnings (for deprecation and discouraged patterns)
# =============================================================================


class TraigentWarning(UserWarning):
    """Base warning class for all Traigent-specific warnings.

    Use this as the base for warnings about deprecated features,
    discouraged patterns, or other non-fatal issues.
    """


class ConfigAccessWarning(TraigentWarning):
    """Warning for deprecated or discouraged configuration access patterns.

    Raised when using deprecated config access methods like get_current_config().
    Use traigent.get_config() instead, which works both during optimization
    trials and after apply_best_config().
    """


class TraigentDeprecationWarning(TraigentWarning):
    """Warning for deprecated Traigent features.

    Named TraigentDeprecationWarning to avoid shadowing Python's built-in
    DeprecationWarning. This is a Traigent-specific deprecation warning
    that won't be filtered by Python's default warning filters.
    """


# =============================================================================
# Lifecycle and State Errors
# =============================================================================


class OptimizationStateError(TraigentError):
    """Raised when accessing configuration in an invalid lifecycle state.

    This error is raised when:
    - Accessing current_config during an active optimization trial
    - Calling get_trial_config() outside of an active optimization trial
    - Calling get_config() when no trial or applied config is available
    - Attempting operations that are invalid for the current state

    Example:
        @traigent.optimize(configuration_space={"model": ["a", "b"]})
        def my_func():
            # Use get_config() - works during and after optimization
            cfg = traigent.get_config()
            return cfg["model"]

        # Run optimization
        result = my_func.optimize(...)
        print(result.best_config)  # Best config from optimization

        # Apply best config for production use
        my_func.apply_best_config()
        my_func("query")  # get_config() now returns applied config
    """

    def __init__(
        self,
        message: str,
        current_state: str | None = None,
        expected_states: list[str] | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, details)
        self.current_state = current_state
        self.expected_states = expected_states or []
