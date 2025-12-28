"""Extended type definitions for Traigent optimization system.

This module provides additional type definitions that extend the core types
with more specific and structured data types for better type safety and
code clarity throughout the system.
"""

# Traceability: CONC-Layer-Core CONC-Quality-Maintainability FUNC-ORCH-LIFECYCLE REQ-ORCH-003 SYNC-OptimizationFlow

from __future__ import annotations

from typing import Any, TypedDict

# =============================================================================
# LLM AND METRICS TYPES
# =============================================================================


class LLMMetrics(TypedDict):
    """Structured type for LLM usage metrics."""

    input_tokens: int
    output_tokens: int
    total_tokens: int
    input_cost: float
    output_cost: float
    total_cost: float
    response_time_ms: float
    model_name: str


class LLMResponseMetadata(TypedDict):
    """Metadata associated with LLM responses."""

    model: str
    temperature: float
    max_tokens: int
    finish_reason: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class CostBreakdown(TypedDict):
    """Detailed cost breakdown for LLM usage."""

    input_cost: float
    output_cost: float
    total_cost: float
    cost_per_token_input: float
    cost_per_token_output: float
    currency: str


# =============================================================================
# VALIDATION AND ERROR TYPES
# =============================================================================


class ValidationResult(TypedDict):
    """Result of a validation operation."""

    is_valid: bool
    errors: list[str]
    warnings: list[str]


class ErrorContext(TypedDict):
    """Context information for error reporting."""

    component: str
    operation: str
    parameters: dict[str, Any]
    timestamp: float
    error_code: str


class ValidationError(TypedDict):
    """Detailed validation error information."""

    field: str
    value: Any
    expected_type: str
    error_message: str
    suggestion: str | None


# =============================================================================
# CONFIGURATION AND PARAMETER TYPES
# =============================================================================


class ParameterBounds(TypedDict):
    """Type-safe bounds representation for parameters."""

    min: int | float
    max: int | float


class CategoricalChoices(TypedDict):
    """Type-safe categorical parameter choices."""

    values: list[Any]
    default: Any


class ParameterConfig(TypedDict):
    """Complete parameter configuration."""

    name: str
    type: str
    bounds: ParameterBounds | CategoricalChoices
    default: Any
    description: str | None


class ConfigurationValidation(TypedDict):
    """Result of configuration validation."""

    is_valid: bool
    missing_parameters: list[str]
    invalid_parameters: list[ValidationError]
    constraint_violations: list[str]


# =============================================================================
# OPTIMIZATION AND EXECUTION TYPES
# =============================================================================


class OptimizationContext(TypedDict):
    """Context information for optimization runs."""

    experiment_id: str
    session_id: str
    function_name: str
    dataset_name: str
    start_time: float
    config: dict[str, Any]


class TrialMetadata(TypedDict):
    """Metadata associated with optimization trials."""

    trial_id: str
    config: dict[str, Any]
    score: float
    metrics: dict[str, float]
    execution_time: float
    status: str
    error_message: str | None


class OptimizationSummary(TypedDict):
    """Summary of optimization results."""

    best_config: dict[str, Any]
    best_score: float
    total_trials: int
    successful_trials: int
    failed_trials: int
    total_time: float
    convergence_status: str


# =============================================================================
# EVALUATION AND METRICS TYPES
# =============================================================================


class ExampleResult(TypedDict):
    """Structured result from evaluating a single example."""

    example_id: str
    input_data: dict[str, Any]
    expected_output: Any
    actual_output: Any
    metrics: dict[str, float]
    execution_time: float
    success: bool
    error_message: str | None


class EvaluationSummary(TypedDict):
    """Summary of evaluation results across multiple examples."""

    total_examples: int
    successful_examples: int
    failed_examples: int
    aggregated_metrics: dict[str, float]
    average_execution_time: float
    error_rate: float


class MetricsAggregation(TypedDict):
    """Configuration for metrics aggregation."""

    method: str  # 'mean', 'sum', 'max', 'min', 'median'
    include_zeros: bool
    filter_outliers: bool
    outlier_threshold: float | None


# =============================================================================
# LOGGING AND MONITORING TYPES
# =============================================================================


class LogEntry(TypedDict):
    """Structured log entry."""

    timestamp: float
    level: str
    component: str
    message: str
    context: dict[str, Any]


class PerformanceMetrics(TypedDict):
    """Performance monitoring metrics."""

    operation: str
    start_time: float
    end_time: float
    duration: float
    memory_usage: int | None
    cpu_usage: float | None
    status: str


class HealthCheckResult(TypedDict):
    """Result of system health check."""

    component: str
    status: str  # 'healthy', 'degraded', 'unhealthy'
    checks: list[dict[str, Any]]
    timestamp: float


# =============================================================================
# API AND COMMUNICATION TYPES
# =============================================================================


class APIRequest(TypedDict):
    """API request structure."""

    endpoint: str
    method: str
    headers: dict[str, str]
    body: dict[str, Any] | None
    timeout: float


class APIResponse(TypedDict):
    """API response structure."""

    status_code: int
    headers: dict[str, str]
    body: dict[str, Any] | None
    response_time: float
    error: str | None


class WebSocketMessage(TypedDict):
    """WebSocket message structure."""

    type: str
    payload: dict[str, Any]
    timestamp: float
    correlation_id: str


# =============================================================================
# FILE AND STORAGE TYPES
# =============================================================================


class FileMetadata(TypedDict):
    """Metadata for stored files."""

    filename: str
    path: str
    size: int
    created_at: float
    modified_at: float
    checksum: str
    format: str


class StorageConfig(TypedDict):
    """Configuration for storage backends."""

    backend: str  # 'edge_analytics', 's3', 'gcs', etc.
    base_path: str
    credentials: dict[str, str] | None
    compression: bool
    encryption: bool


# =============================================================================
# UTILITY TYPES FOR TYPE SAFETY
# =============================================================================

# Type aliases for commonly used complex types
ConfigDict = dict[str, Any]
MetricsDict = dict[str, float]
ParameterDict = dict[str, ParameterBounds | CategoricalChoices]
ValidationErrors = list[ValidationError]

# Union types for flexibility
NumericType = int | float
BoundsType = tuple[NumericType, NumericType] | list[Any]
ConfigValue = str | int | float | bool | list[Any] | dict[str, Any]


# Protocol-like types for better type checking
class SupportsOptimization:
    """Protocol for objects that support optimization."""

    def optimize(self, *args, **kwargs) -> Any: ...


class SupportsEvaluation:
    """Protocol for objects that support evaluation."""

    def evaluate(self, *args, **kwargs) -> Any: ...


class SupportsValidation:
    """Protocol for objects that support validation."""

    def validate(self, *args, **kwargs) -> ValidationResult:
        """Protocol method for validation."""
        raise NotImplementedError("Subclasses must implement validate method")


# =============================================================================
# GENERIC UTILITY FUNCTIONS FOR TYPE SAFETY
# =============================================================================


def create_llm_metrics(
    input_tokens: int,
    output_tokens: int,
    input_cost: float,
    output_cost: float,
    response_time_ms: float,
    model_name: str,
) -> LLMMetrics:
    """Create a properly typed LLMMetrics object.

    Args:
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        input_cost: Cost for input tokens
        output_cost: Cost for output tokens
        response_time_ms: Response time in milliseconds
        model_name: Name of the model used

    Returns:
        Properly typed LLMMetrics object
    """
    return LLMMetrics(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
        input_cost=input_cost,
        output_cost=output_cost,
        total_cost=input_cost + output_cost,
        response_time_ms=response_time_ms,
        model_name=model_name,
    )


def create_validation_result(
    is_valid: bool, errors: list[str] | None = None, warnings: list[str] | None = None
) -> ValidationResult:
    """Create a properly typed ValidationResult object.

    Args:
        is_valid: Whether validation passed
        errors: List of error messages
        warnings: List of warning messages

    Returns:
        Properly typed ValidationResult object
    """
    return ValidationResult(
        is_valid=is_valid, errors=errors or [], warnings=warnings or []
    )


def create_error_context(
    component: str,
    operation: str,
    error_code: str,
    parameters: dict[str, Any] | None = None,
) -> ErrorContext:
    """Create a properly typed ErrorContext object.

    Args:
        component: Component where error occurred
        operation: Operation being performed
        error_code: Error code identifier
        parameters: Additional parameters for context

    Returns:
        Properly typed ErrorContext object
    """
    import time

    return ErrorContext(
        component=component,
        operation=operation,
        parameters=parameters or {},
        timestamp=time.time(),
        error_code=error_code,
    )


# =============================================================================
# TYPE GUARDS FOR RUNTIME TYPE CHECKING
# =============================================================================


def is_llm_metrics(obj: Any) -> bool:
    """Type guard for LLMMetrics objects."""
    if not isinstance(obj, dict):
        return False

    required_keys = {
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "input_cost",
        "output_cost",
        "total_cost",
        "response_time_ms",
        "model_name",
    }

    return all(key in obj for key in required_keys)


def is_validation_result(obj: Any) -> bool:
    """Type guard for ValidationResult objects."""
    if not isinstance(obj, dict):
        return False

    required_keys = {"is_valid", "errors", "warnings"}
    return all(key in obj for key in required_keys)


def is_example_result(obj: Any) -> bool:
    """Type guard for ExampleResult objects."""
    if not isinstance(obj, dict):
        return False

    required_keys = {
        "example_id",
        "input_data",
        "expected_output",
        "actual_output",
        "metrics",
        "execution_time",
        "success",
    }

    return all(key in obj for key in required_keys)
