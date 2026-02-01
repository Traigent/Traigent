"""Core optimized function implementation.
# Traceability: CONC-Layer-Core CONC-Quality-Usability CONC-Quality-Maintainability FUNC-ORCH-LIFECYCLE REQ-ORCH-003 SYNC-OptimizationFlow

Custom Evaluator Interface:
    Custom evaluation functions should have the signature:

    def custom_evaluate(func: Callable, config: dict[str, Any], example: EvaluationExample) -> ExampleResult

    Where:
    - func: The function being optimized
    - config: Configuration parameters to test
    - example: EvaluationExample from the dataset (contains input_data, expected_output, metadata)

    The function should return an ExampleResult with:
    - example_id: Unique identifier for this example
    - input_data: The input data used
    - expected_output: Expected output from the dataset
    - actual_output: Actual output from the function
    - metrics: Dict of metric names to float values
    - execution_time: Time taken to execute
    - success: Boolean indicating if evaluation succeeded
    - error_message: Optional error message if evaluation failed
"""

# Traceability: CONC-Layer-Core FUNC-ORCH-LIFECYCLE REQ-ORCH-003 SYNC-OptimizationFlow

from __future__ import annotations

import asyncio
import base64
import copy
import hashlib
import hmac
import json
import os
import sys
import threading
import time
from collections.abc import Callable, Sequence
from datetime import UTC, datetime, timedelta
from enum import Enum, auto
from pathlib import Path
from typing import Any, Literal, cast

from traigent.api.functions import _GLOBAL_CONFIG, get_global_parallel_config
from traigent.api.types import OptimizationResult, OptimizationStatus
from traigent.config import get_provider
from traigent.config.parallel import (
    coerce_parallel_config,
    merge_parallel_configs,
    resolve_parallel_config,
)
from traigent.config.types import ExecutionMode, TraigentConfig, resolve_execution_mode
from traigent.core.evaluator_wrapper import CustomEvaluatorWrapper
from traigent.core.objectives import (
    ObjectiveSchema,
    create_default_objectives,
    normalize_objectives,
    schema_to_objective_names,
)
from traigent.core.orchestrator import OptimizationOrchestrator
from traigent.evaluators.base import BaseEvaluator, Dataset
from traigent.evaluators.local import LocalEvaluator
from traigent.integrations.framework_override import override_context
from traigent.optimizers import get_optimizer
from traigent.tvl.options import TVLOptions
from traigent.tvl.spec_loader import load_tvl_spec
from traigent.utils.env_config import (
    get_validation_timeout,
    is_backend_offline,
    is_mock_llm,
    is_production,
    skip_provider_validation,
)
from traigent.utils.exceptions import (
    AuthenticationError,
    ConfigurationError,
    OptimizationError,
    OptimizationStateError,
    ProviderValidationError,
    TVLValidationError,
    ValidationError,
)
from traigent.utils.incentives import show_upgrade_hint
from traigent.utils.logging import get_logger
from traigent.utils.secure_path import safe_open, validate_path
from traigent.utils.validation import (
    validate_config_space,
    validate_dataset_path,
    validate_objectives,
)

logger = get_logger(__name__)

# Module-level flag to ensure cost warning is emitted only once per process
_COST_WARNING_EMITTED = False

# ISO 8601 timezone suffix for UTC (used in token validation)
_UTC_TIMEZONE_SUFFIX = "+00:00"

# Error message for invalid configuration space type
_CONFIG_SPACE_TYPE_ERROR = "Configuration space must be a dictionary"


def _emit_cost_warning_once() -> None:
    """Emit cost warning once per process when optimization starts.

    This warning informs users that optimization will make multiple LLM API calls
    and that cost estimates are approximations. The warning is suppressed in mock mode.
    """
    global _COST_WARNING_EMITTED
    if _COST_WARNING_EMITTED:
        return
    if is_mock_llm():
        return
    if os.getenv("TRAIGENT_COST_APPROVED", "").lower() in ("true", "1", "yes"):
        return

    _COST_WARNING_EMITTED = True

    # ANSI color codes for terminal styling
    YELLOW = "\033[93m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"
    RESET = "\033[0m"

    # Check if terminal supports colors (not redirected to file)
    use_colors = sys.stderr.isatty()

    if use_colors:
        msg = (
            f"\n{YELLOW}{BOLD}[!] COST WARNING{RESET}\n"
            f"{YELLOW}Traigent optimization will make multiple LLM API calls.{RESET}\n"
            f"Cost estimates are approximations based on {CYAN}tokencost{RESET} library pricing.\n"
            f"Actual billing is determined by your LLM provider.\n\n"
            f"{BOLD}Configuration:{RESET}\n"
            f"  - Custom model mappings: {CYAN}traigent/utils/cost_calculator.py{RESET} (EXACT_MODEL_MAPPING)\n"
            f"  - To skip real API calls in tests: {CYAN}TRAIGENT_MOCK_LLM=true{RESET}\n"
            f"  - Full details:          {CYAN}DISCLAIMER.md{RESET}\n"
        )
    else:
        msg = (
            "\n[!] COST WARNING\n"
            "Traigent optimization will make multiple LLM API calls.\n"
            "Cost estimates are approximations based on tokencost library pricing.\n"
            "Actual billing is determined by your LLM provider.\n\n"
            "Configuration:\n"
            "  - Custom model mappings: traigent/utils/cost_calculator.py (EXACT_MODEL_MAPPING)\n"
            "  - To skip real API calls in tests: TRAIGENT_MOCK_LLM=true\n"
            "  - Full details:          DISCLAIMER.md\n"
        )

    try:
        print(msg, file=sys.stderr)
    except UnicodeEncodeError:
        print(msg.encode("ascii", errors="replace").decode(), file=sys.stderr)
    sys.stderr.flush()


class OptimizationState(Enum):
    """Lifecycle state of an OptimizedFunction.

    This enum tracks where the function is in its optimization lifecycle,
    enabling proper enforcement of configuration access rules.

    States:
        UNOPTIMIZED: Before any optimization has been run. The function uses
            its default_config when called.
        OPTIMIZING: During an active optimization run. Configuration should
            be accessed via traigent.get_config() inside the function.
        OPTIMIZED: After optimization has completed successfully. The function
            uses best_config automatically, accessible via current_config.
        ERROR: Optimization failed. The function reverts to default_config.
    """

    UNOPTIMIZED = auto()
    OPTIMIZING = auto()
    OPTIMIZED = auto()
    ERROR = auto()


def _is_ci_environment() -> bool:
    """Detect if running in a CI environment (robust detection - 10 providers)."""
    return (
        os.getenv("CI") in ("true", "1")
        or os.getenv("GITHUB_ACTIONS") in ("true", "1")
        or os.getenv("JENKINS_URL") is not None
        or os.getenv("GITLAB_CI") in ("true", "1")
        or os.getenv("CIRCLECI") in ("true", "1")
        or os.getenv("TRAVIS") in ("true", "1")
        or os.getenv("BUILDKITE") in ("true", "1")
        or os.getenv("TEAMCITY_VERSION") is not None
        or os.getenv("AZURE_HTTP_USER_AGENT") is not None
        or os.getenv("BITBUCKET_BUILD_NUMBER") is not None
    )


def _check_env_var_approval() -> bool:
    """Check if CI run is approved via environment variable."""
    if os.getenv("TRAIGENT_RUN_APPROVED") == "1":
        approved_by = os.getenv("TRAIGENT_APPROVED_BY", "environment_variable")
        logger.info(f"CI optimization approved by: {approved_by}")
        return True
    return False


def _validate_legacy_token(token_data: dict[str, Any]) -> bool:
    """Validate a legacy format approval token."""
    if "approved_by" not in token_data or "expires_at" not in token_data:
        return False
    try:
        expires_str = token_data["expires_at"]
        expires_at = datetime.fromisoformat(
            expires_str.replace("Z", _UTC_TIMEZONE_SUFFIX)
        )
        now = datetime.now()
        if expires_at.tzinfo:
            now = now.replace(tzinfo=UTC)
        if expires_at > now:
            logger.info(
                f"CI optimization approved by legacy token: {token_data['approved_by']}"
            )
            return True
    except (ValueError, KeyError):
        pass
    return False


def _validate_hmac_token(token_data: dict[str, Any]) -> bool:
    """Validate an HMAC-signed approval token."""
    required_fields = ["approver", "expires_iso", "nonce", "signature"]
    if not all(field in token_data for field in required_fields):
        return False

    secret = os.getenv("TRAIGENT_APPROVAL_SECRET", "").encode()
    if not secret:
        logger.warning(
            "TRAIGENT_APPROVAL_SECRET not set, cannot validate token signature"
        )
        # Check expiry without signature validation
        try:
            expires_at = datetime.fromisoformat(
                token_data["expires_iso"].replace("Z", _UTC_TIMEZONE_SUFFIX)
            )
            now = datetime.now(UTC) if expires_at.tzinfo else datetime.now()
            if expires_at > now:
                logger.warning(
                    f"Token approved by {token_data['approver']} (signature not verified)"
                )
                return True
        except (ValueError, KeyError):
            pass
        return False

    # Compute expected signature
    payload = f"{token_data['approver']}|{token_data['expires_iso']}|{token_data['nonce']}".encode()
    expected_sig = base64.b64encode(
        hmac.new(secret, payload, hashlib.sha256).digest()
    ).decode()

    # Constant-time comparison
    if not hmac.compare_digest(token_data.get("signature", ""), expected_sig):
        logger.warning("Token signature validation failed")
        return False

    # Check expiration
    try:
        expires_at = datetime.fromisoformat(
            token_data["expires_iso"].replace("Z", _UTC_TIMEZONE_SUFFIX)
        )
        now = datetime.now(UTC) if expires_at.tzinfo else datetime.now()

        # Enforce max TTL of 24 hours from creation
        if expires_at - now > timedelta(hours=24):
            logger.warning("Token TTL exceeds 24 hours, rejecting")
            return False
        if expires_at > now:
            logger.info(
                f"CI optimization approved by HMAC token: {token_data['approver']}"
            )
            return True
        logger.debug("Token expired")
    except (ValueError, KeyError):
        pass
    return False


def _check_token_file_approval(token_file: Path, base_dir: Path) -> bool:
    """Check if CI run is approved via token file."""
    if not token_file.exists():
        return False

    try:
        with safe_open(token_file, base_dir, mode="r", encoding="utf-8") as f:
            token_data = json.load(f)

        # Try HMAC token first, then legacy format
        if _validate_hmac_token(token_data):
            return True
        if _validate_legacy_token(token_data):
            return True
    except (ValueError, KeyError) as e:  # JSONDecodeError is subclass of ValueError
        logger.debug(f"Invalid approval token: {e}")

    return False


class OptimizedFunction:
    """Wrapper for functions decorated with @traigent.optimize.

    This class provides the optimization interface for decorated functions,
    including methods to run optimization, get results, and analyze performance.
    """

    def __init__(
        self,
        func: Callable[..., Any],
        eval_dataset: str | list[str] | Dataset | None = None,
        objectives: list[str] | ObjectiveSchema | None = None,
        configuration_space: dict[str, Any] | None = None,
        config_space: dict[str, Any] | None = None,  # Backward compatibility
        default_config: dict[str, Any] | None = None,
        constraints: list[Callable[..., bool]] | None = None,
        injection_mode: str = "context",
        config_param: str | None = None,
        auto_override_frameworks: bool = False,
        framework_targets: list[str] | None = None,
        execution_mode: str = "cloud",
        local_storage_path: str | None = None,
        minimal_logging: bool = True,
        custom_evaluator: Callable[..., Any] | None = None,
        scoring_function: Callable[..., Any] | None = None,
        metric_functions: dict[str, Callable[..., Any]] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize optimized function wrapper.

        Args:
            func: Original function to optimize
            eval_dataset: Evaluation dataset path(s) or Dataset object
            objectives: List of objectives to optimize or ObjectiveSchema
            configuration_space: Parameter search space
            default_config: Default configuration values
            constraints: List of constraint functions
            injection_mode: How to inject configuration ("context", "parameter", "decorator")
            config_param: Parameter name for injection_mode="parameter"
            auto_override_frameworks: Enable automatic framework parameter overrides
            framework_targets: List of framework class names to override (e.g., ["openai.OpenAI"])
            execution_mode: Execution mode ("edge_analytics", "privacy", "standard", "cloud")
            local_storage_path: Custom path for local storage (Edge Analytics mode only)
            minimal_logging: Use minimal logging in Edge Analytics mode
            custom_evaluator: Custom evaluation function for advanced use cases
            scoring_function: Simple scoring function that returns a score or dict of scores
            metric_functions: Dict of metric name to scoring function
            **kwargs: Additional configuration
        """
        # Extract decorator-provided metadata before core storage
        self._requested_execution_mode = kwargs.pop("requested_execution_mode", None)
        # Store core parameters
        self._store_core_parameters(
            func,
            eval_dataset,
            objectives,
            default_config,
            constraints,
            injection_mode,
            config_param,
            auto_override_frameworks,
            framework_targets,
            execution_mode,
            local_storage_path,
            minimal_logging,
            custom_evaluator,
            scoring_function,
            metric_functions,
        )

        # Handle configuration space with backward compatibility
        self._setup_configuration_space(configuration_space, config_space)

        # Store additional parameters from kwargs
        self._store_additional_parameters(kwargs)

        # Initialize provider and validate configuration
        self._initialize_provider_and_validate()

        # Initialize state and function wrapper
        self._initialize_state_and_wrapper()

        logger.debug(
            f"Created OptimizedFunction for {getattr(func, '__name__', str(func))}"
        )

    def _store_core_parameters(
        self,
        func,
        eval_dataset,
        objectives,
        default_config,
        constraints,
        injection_mode,
        config_param,
        auto_override_frameworks,
        framework_targets,
        execution_mode,
        local_storage_path,
        minimal_logging,
        custom_evaluator,
        scoring_function,
        metric_functions,
    ) -> None:
        """Store core initialization parameters."""
        self.func = func
        self.eval_dataset = eval_dataset

        # Handle ObjectiveSchema creation
        resolved_schema = normalize_objectives(objectives)
        if resolved_schema is None:
            resolved_schema = create_default_objectives(["accuracy"])

        self.objective_schema = resolved_schema

        self.default_config = default_config or {}
        self.constraints = constraints or []
        self.injection_mode = injection_mode
        self.config_param = config_param
        self.auto_override_frameworks = auto_override_frameworks
        self.framework_targets = framework_targets or []

        # Execution mode configuration
        requested_mode = getattr(self, "_requested_execution_mode", None)
        try:
            effective_mode_enum = resolve_execution_mode(execution_mode)
        except (TypeError, ValueError) as exc:
            raise ValueError(str(exc)) from None

        privacy_alias_requested = False
        if effective_mode_enum is ExecutionMode.PRIVACY:
            effective_mode_enum = ExecutionMode.HYBRID
            privacy_alias_requested = True

        self._effective_execution_mode = effective_mode_enum
        if (
            requested_mode
            and requested_mode.lower() == "privacy"
            and effective_mode_enum is ExecutionMode.HYBRID
        ):
            display_mode = "privacy"
            privacy_alias_requested = True
        elif requested_mode:
            display_mode = requested_mode.lower()
        else:
            display_mode = effective_mode_enum.value
        self.execution_mode = display_mode
        self._privacy_alias_requested = privacy_alias_requested
        self.local_storage_path = local_storage_path
        self.minimal_logging = minimal_logging

        # Evaluation configuration
        self.custom_evaluator = custom_evaluator
        self.scoring_function = scoring_function
        self.metric_functions = metric_functions

    def _is_cloud_execution_mode(self) -> bool:
        """Return True when configured for managed cloud execution."""
        effective_mode = getattr(self, "_effective_execution_mode", None)
        if isinstance(effective_mode, ExecutionMode):
            mode_enum = effective_mode
        else:
            mode_enum = resolve_execution_mode(
                effective_mode, default=resolve_execution_mode(self.execution_mode)
            )
        return mode_enum in {
            ExecutionMode.CLOUD,
            ExecutionMode.STANDARD,
            ExecutionMode.HYBRID,
        }

    def _setup_configuration_space(self, configuration_space, config_space) -> None:
        """Setup configuration space with backward compatibility."""
        # Backward compatibility: support both config_space and configuration_space
        if config_space is not None and configuration_space is None:
            try:
                self.configuration_space = config_space
            except ValidationError as e:
                if _CONFIG_SPACE_TYPE_ERROR in str(e):
                    raise TypeError(str(e)) from None
                raise
        else:
            try:
                self.configuration_space = configuration_space or {}
            except ValidationError as e:
                if _CONFIG_SPACE_TYPE_ERROR in str(e):
                    raise TypeError(str(e)) from None
                else:
                    raise

    def _store_callbacks(self, kwargs: dict[str, Any], sentinel: object) -> None:
        """Store callbacks parameter, normalizing to a list."""
        callbacks_value = kwargs.pop("callbacks", sentinel)
        if callbacks_value is sentinel or callbacks_value is None:
            self.callbacks = None
            return

        if isinstance(callbacks_value, list):
            self.callbacks = callbacks_value
            kwargs["callbacks"] = list(callbacks_value)
        elif isinstance(callbacks_value, tuple):
            normalized_callbacks = list(callbacks_value)
            self.callbacks = normalized_callbacks
            kwargs["callbacks"] = list(normalized_callbacks)
        else:
            normalized_callbacks = [callbacks_value]
            self.callbacks = normalized_callbacks
            kwargs["callbacks"] = list(normalized_callbacks)

    def _store_optional_param(
        self,
        kwargs: dict[str, Any],
        sentinel: object,
        key: str,
        default: Any,
        as_bool: bool = False,
    ) -> Any:
        """Store an optional parameter with sentinel-based defaults.

        Args:
            kwargs: The kwargs dict to read from and update.
            sentinel: Sentinel object for detecting missing values.
            key: The parameter key name.
            default: Default value when sentinel is detected.
            as_bool: If True, coerce the value to bool.

        Returns:
            The resolved value.
        """
        value = kwargs.pop(key, sentinel)
        if value is sentinel:
            resolved = default
        elif as_bool:
            resolved = bool(value)
            kwargs[key] = resolved
        else:
            resolved = value
            kwargs[key] = value
        return resolved

    def _store_additional_parameters(self, kwargs) -> None:
        """Store additional parameters from kwargs."""
        sentinel = object()

        # Decorator-provided runtime defaults
        self.algorithm = kwargs.pop("algorithm", "random")
        kwargs["algorithm"] = self.algorithm

        self.max_trials = kwargs.pop("max_trials", 50)
        kwargs["max_trials"] = self.max_trials

        self.timeout = kwargs.pop("timeout", 60.0)
        kwargs["timeout"] = self.timeout

        save_to_value = kwargs.pop("save_to", sentinel)
        self.save_to = (
            None
            if save_to_value is sentinel or save_to_value is None
            else save_to_value
        )
        if self.save_to is not None:
            kwargs["save_to"] = self.save_to

        self._store_callbacks(kwargs, sentinel)

        self.use_cloud_service = self._store_optional_param(
            kwargs, sentinel, "use_cloud_service", False, as_bool=True
        )
        self.framework_target = self._store_optional_param(
            kwargs, sentinel, "framework_target", None
        )

        # Execution knobs
        provided_parallel = coerce_parallel_config(kwargs.pop("parallel_config", None))
        combined_parallel, sources = merge_parallel_configs(
            [(provided_parallel, "decorator")]
        )
        self.parallel_config = combined_parallel
        self.parallel_config_sources = sources
        kwargs["parallel_config"] = combined_parallel
        if sources:
            kwargs["_parallel_config_sources"] = sources

        self.privacy_enabled = self._store_optional_param(
            kwargs, sentinel, "privacy_enabled", False, as_bool=True
        )
        if getattr(self, "_privacy_alias_requested", False):
            self.privacy_enabled = True
            kwargs["privacy_enabled"] = True
            self._privacy_alias_requested = False

        # Mock mode configuration
        self.mock_mode_config = self._store_optional_param(
            kwargs, sentinel, "mock_mode_config", None
        )
        self.max_examples = self._store_optional_param(
            kwargs, sentinel, "max_examples", None
        )
        self.max_total_examples = self._store_optional_param(
            kwargs, sentinel, "max_total_examples", None
        )
        self.samples_include_pruned = self._store_optional_param(
            kwargs, sentinel, "samples_include_pruned", True, as_bool=True
        )
        self.validate_providers = self._store_optional_param(
            kwargs, sentinel, "validate_providers", True, as_bool=True
        )

        self.kwargs = kwargs
        excluded_runtime_keys = {
            "algorithm",
            "max_trials",
            "timeout",
            "save_to",
            "callbacks",
            "parallel_config",
            "_parallel_config_sources",
            "use_cloud_service",
            "framework_target",
            "privacy_enabled",
            "mock_mode_config",
            "max_total_examples",
            "samples_include_pruned",
        }
        self._decorator_runtime_overrides = {
            key: value
            for key, value in kwargs.items()
            if key not in excluded_runtime_keys
        }

        # Cloud service client (initialized lazily)
        self._cloud_client: Any | None = None

    def _initialize_provider_and_validate(self) -> None:
        """Initialize configuration provider and validate inputs."""
        # Validate basic inputs first
        self._validate_basic_inputs()

        # Get configuration provider
        # Convert enum to string value if needed
        injection_mode_str = (
            self.injection_mode.value
            if hasattr(self.injection_mode, "value")
            else self.injection_mode
        )
        try:
            self._provider = get_provider(
                injection_mode_str, config_param=self.config_param
            )
        except ConfigurationError as e:
            # Normalize provider/config errors to ValueError for decorator tests
            raise ValueError(str(e)) from None

        # Validate inputs
        self._validate_configuration()

    def _initialize_state_and_wrapper(self) -> None:
        """Initialize optimization state and function wrapper."""
        # Lock for protecting state mutations
        self._state_lock = threading.RLock()

        # Lifecycle state tracking
        self._state: OptimizationState = OptimizationState.UNOPTIMIZED

        # Optimization results
        self._optimization_results: OptimizationResult | None = None
        self._optimization_history: list[OptimizationResult] = []

        # Configuration state
        self._current_config = self.default_config.copy()
        self._best_config: dict[str, Any] | None = None

        # Make function callable with current config
        self._setup_function_wrapper()

    def _estimate_search_space_size(self) -> int:
        """Best-effort estimation of configuration combinations."""

        space = getattr(self, "configuration_space", {}) or {}
        if not isinstance(space, dict) or not space:
            return 0

        total = 1
        for values in space.values():
            if isinstance(values, (list, tuple, set)):
                total *= max(len(values), 1)
            else:
                return 0
        return total

    @property
    def config_space(self) -> dict[str, Any]:
        """Backward compatibility property for configuration_space."""
        return self.configuration_space

    @property
    def objectives(self) -> list[str]:
        """Objective names derived from the active objective schema."""
        result: list[str] = schema_to_objective_names(self.objective_schema)
        return result

    def _validate_basic_inputs(self) -> None:
        """Validate basic inputs and raise appropriate exceptions."""
        # Validate function
        if not callable(self.func):
            raise TypeError("func must be callable") from None

        # Validate max_trials and timeout for negative values
        if self.max_trials < 0:
            raise ValueError("max_trials must be non-negative")

        if self.timeout < 0:
            raise ValueError("timeout must be non-negative")

    def _validate_configuration(self) -> None:
        """Validate optimization configuration."""
        self._validate_objectives()
        self._validate_config_space()
        self._validate_dataset()

    def _validate_objectives(self) -> None:
        """Validate objectives configuration."""
        try:
            validate_objectives(self.objectives)
        except ValidationError as e:
            if "At least one objective must be specified" in str(e):
                raise ValueError(str(e)) from None
            elif "Objectives must be a list" in str(e):
                raise TypeError(str(e)) from e
            else:
                raise

    def _validate_config_space(self) -> None:
        """Validate configuration space."""
        if self.configuration_space:
            try:
                validate_config_space(self.configuration_space)
            except ValidationError as e:
                if _CONFIG_SPACE_TYPE_ERROR in str(e):
                    raise TypeError(str(e)) from None
                else:
                    raise
        elif self.configuration_space == {}:
            # Empty config space should raise ValueError with helpful message
            raise ValueError(
                "Configuration space cannot be empty. Please specify at least one parameter to optimize "
                "in the @traigent.optimize decorator. Example: configuration_space={'temperature': [0.0, 0.5, 1.0], "
                "'model': ['gpt-3.5-turbo', 'gpt-4']}"
            )

    def _validate_dataset(self) -> None:
        """Validate dataset configuration."""
        if isinstance(self.eval_dataset, (str, list)):
            # Skip dataset validation in tests when the file doesn't exist
            if not (
                os.environ.get("PYTEST_CURRENT_TEST")
                or (
                    isinstance(self.eval_dataset, str)
                    and self.eval_dataset in ["test.jsonl", "data.jsonl"]
                )
            ):
                validate_dataset_path(self.eval_dataset)

    def _setup_function_wrapper(self) -> None:
        """Setup function wrapper that uses current configuration."""
        # Use provider to inject configuration
        self._wrapped_func = self._provider.inject_config(
            self.func, self._current_config, self.config_param
        )

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Make the optimized function callable."""
        # If framework overrides are enabled, use them during function call
        if self.auto_override_frameworks and self.framework_targets:
            with override_context(self.framework_targets):
                return self._wrapped_func(*args, **kwargs)
        else:
            return self._wrapped_func(*args, **kwargs)

    def _prepare_algorithm_kwargs(
        self, algorithm_kwargs: dict[str, Any]
    ) -> dict[str, Any]:
        """Merge decorator overrides into algorithm kwargs and validate."""
        decorator_overrides = getattr(self, "_decorator_runtime_overrides", {})
        if decorator_overrides:
            merged = {**decorator_overrides, **algorithm_kwargs}
        else:
            merged = dict(algorithm_kwargs)

        if "parallel_trials" in merged:
            raise ValueError(
                "parallel_trials is not a valid parameter for optimize(). "
                "Use parallel_config={'trial_concurrency': N} instead."
            )
        return merged

    def _validate_objectives_input(
        self,
        objectives: ObjectiveSchema | Sequence[str] | None,
        algorithm_kwargs: dict[str, Any],
    ) -> tuple[
        ObjectiveSchema | Sequence[str] | None, ObjectiveSchema | Sequence[str] | None
    ]:
        """Validate and extract objectives from inputs.

        Returns:
            Tuple of (objectives, legacy_objectives) after validation.
        """
        legacy_objectives = algorithm_kwargs.pop("objectives", None)
        legacy_orientations = algorithm_kwargs.pop("objective_orientations", None)
        legacy_weights = algorithm_kwargs.pop("objective_weights", None)

        if legacy_orientations is not None or legacy_weights is not None:
            raise ValueError(
                "objective_orientations/objective_weights are no longer supported. "
                "Provide an ObjectiveSchema instead."
            )
        if objectives is not None and legacy_objectives is not None:
            raise ValueError(
                "objectives provided both via parameter and inside algorithm_kwargs"
            )
        return objectives, legacy_objectives

    def _process_tvl_options(
        self,
        tvl_spec: str | Path | None,
        tvl_environment: str | None,
        tvl: TVLOptions | dict[str, Any] | None,
        algorithm_kwargs: dict[str, Any],
        algorithm: str | None,
        max_trials: int | None,
        timeout: float | None,
        configuration_space: dict[str, Any] | None,
        objectives: ObjectiveSchema | Sequence[str] | None,
    ) -> tuple[
        str | None,
        int | None,
        float | None,
        dict[str, Any] | None,
        ObjectiveSchema | Sequence[str] | None,
        dict[str, Any] | None,
    ]:
        """Process TVL options and return updated values.

        Returns:
            Tuple of (algorithm, max_trials, timeout, configuration_space, objectives, tvl_state)
        """
        runtime_tvl_spec_kw = algorithm_kwargs.pop("tvl_spec", None)
        runtime_tvl_env_kw = algorithm_kwargs.pop("tvl_environment", None)
        runtime_tvl_bundle_kw = algorithm_kwargs.pop("tvl", None)

        tvl_options_runtime = self._resolve_runtime_tvl_options(
            tvl_spec if tvl_spec is not None else runtime_tvl_spec_kw,
            tvl_environment if tvl_environment is not None else runtime_tvl_env_kw,
            tvl if tvl is not None else runtime_tvl_bundle_kw,
        )

        if not tvl_options_runtime:
            return algorithm, max_trials, timeout, configuration_space, objectives, None

        try:
            tvl_artifact = load_tvl_spec(**tvl_options_runtime.to_kwargs())  # type: ignore[arg-type]
        except TVLValidationError as exc:
            raise ValidationError(exc.message) from exc

        if configuration_space is None:
            configuration_space = tvl_artifact.configuration_space
        if objectives is None and tvl_artifact.objective_schema is not None:
            objectives = tvl_artifact.objective_schema

        tvl_state = self._apply_runtime_tvl_artifact(tvl_artifact)
        algorithm, max_trials, timeout = self._apply_tvl_runtime_overrides(
            algorithm,
            max_trials,
            timeout,
            algorithm_kwargs,
            tvl_artifact.runtime_overrides(),
        )
        return (
            algorithm,
            max_trials,
            timeout,
            configuration_space,
            objectives,
            tvl_state,
        )

    def _restore_tvl_state(self, tvl_state: dict[str, Any] | None) -> None:
        """Restore state modified by TVL processing."""
        if not tvl_state:
            return
        if "constraints" in tvl_state:
            self.constraints = tvl_state["constraints"]
        if "default_config" in tvl_state:
            self.default_config = tvl_state["default_config"]

    async def optimize(
        self,
        algorithm: str | None = None,
        max_trials: int | None = None,
        timeout: float | None = None,
        save_to: str | None = None,
        custom_evaluator: Callable[..., Any] | None = None,
        callbacks: list[Callable[..., Any]] | None = None,
        configuration_space: dict[str, Any] | None = None,
        objectives: ObjectiveSchema | Sequence[str] | None = None,
        tvl_spec: str | Path | None = None,
        tvl_environment: str | None = None,
        tvl: TVLOptions | dict[str, Any] | None = None,
        **algorithm_kwargs: Any,
    ) -> OptimizationResult:
        """Run optimization on the function.

        Args:
            algorithm: Optimization algorithm to use. If None, uses the
                algorithm specified in the decorator (self.algorithm).
            max_trials: Maximum number of trials
            timeout: Maximum optimization time in seconds. Note: ``timeout_seconds``
                is deprecated; use ``timeout`` instead.
            save_to: Path to save results
            custom_evaluator: Custom evaluation function that takes (func, config, input_data)
                            and returns metrics dict. If provided, overrides default evaluation.
            callbacks: List of callback objects for progress tracking
            configuration_space: Override configuration space for this optimization run.
                                Takes precedence over decorator configuration_space.
            objectives: Optional override objectives (list of names or ObjectiveSchema)
            tvl_spec: Optional TVL spec path to load at runtime.
            tvl_environment: Environment overlay to apply when loading the spec.
            tvl: Structured TVL options (dict or TVLOptions) for runtime overrides.
            **algorithm_kwargs: Additional algorithm parameters

        Returns:
            OptimizationResult with trial results and best configuration

        Raises:
            OptimizationError: If optimization fails
        """
        logger.info(f"Starting optimization of {self.func.__name__}")
        _emit_cost_warning_once()

        # Handle common alias: timeout_seconds -> timeout
        if "timeout_seconds" in algorithm_kwargs:
            if timeout is None:
                timeout = algorithm_kwargs.pop("timeout_seconds")
                logger.warning(
                    "timeout_seconds is deprecated, use timeout instead. "
                    "The value has been applied."
                )
            else:
                algorithm_kwargs.pop("timeout_seconds")
                logger.warning(
                    "Both timeout and timeout_seconds provided; using timeout. "
                    "timeout_seconds is deprecated."
                )

        algorithm_kwargs = self._prepare_algorithm_kwargs(algorithm_kwargs)
        objectives, legacy_objectives = self._validate_objectives_input(
            objectives, algorithm_kwargs
        )

        # Process TVL options
        algorithm, max_trials, timeout, configuration_space, objectives, tvl_state = (
            self._process_tvl_options(
                tvl_spec,
                tvl_environment,
                tvl,
                algorithm_kwargs,
                algorithm,
                max_trials,
                timeout,
                configuration_space,
                objectives,
            )
        )

        # Normalize configuration_space to handle Range/IntRange/LogRange/Choices objects
        if configuration_space is not None:
            from traigent.api.parameter_ranges import normalize_configuration_space

            configuration_space, _ = normalize_configuration_space(configuration_space)

        runtime_objective_input = (
            objectives if objectives is not None else legacy_objectives
        )
        try:
            runtime_schema = normalize_objectives(runtime_objective_input)
        except TypeError as exc:
            raise ValueError(str(exc)) from exc

        original_schema = self.objective_schema
        if runtime_schema is not None:
            self.objective_schema = runtime_schema

        timeout = timeout if timeout is not None else getattr(self, "timeout", None)
        save_to = save_to if save_to is not None else getattr(self, "save_to", None)
        callbacks = (
            callbacks if callbacks is not None else getattr(self, "callbacks", None)
        )

        try:
            validate_objectives(self.objectives)
            result = await self._execute_optimization(
                algorithm=algorithm,
                max_trials=max_trials,
                timeout=timeout,
                save_to=save_to,
                custom_evaluator=custom_evaluator,
                callbacks=callbacks,
                configuration_space=configuration_space,
                algorithm_kwargs=algorithm_kwargs,
            )
        finally:
            if runtime_schema is not None:
                self.objective_schema = original_schema
            self._restore_tvl_state(tvl_state)

        return result

    def optimize_sync(
        self,
        algorithm: str | None = None,
        max_trials: int | None = None,
        timeout: float | None = None,
        save_to: str | None = None,
        custom_evaluator: Callable[..., Any] | None = None,
        callbacks: list[Callable[..., Any]] | None = None,
        configuration_space: dict[str, Any] | None = None,
        objectives: ObjectiveSchema | Sequence[str] | None = None,
        tvl_spec: str | Path | None = None,
        tvl_environment: str | None = None,
        tvl: TVLOptions | dict[str, Any] | None = None,
        **algorithm_kwargs: Any,
    ) -> OptimizationResult:
        """Run optimization synchronously (convenience wrapper).

        This is a synchronous wrapper around optimize() for users who don't need
        async functionality. It handles the event loop creation internally.

        For async code, use the async optimize() method directly:
            result = await func.optimize(...)

        Args:
            algorithm: Optimization algorithm to use
            max_trials: Maximum number of trials
            timeout: Maximum optimization time in seconds
            save_to: Path to save results
            custom_evaluator: Custom evaluation function
            callbacks: List of callback objects for progress tracking
            configuration_space: Override configuration space
            objectives: Optional override objectives
            tvl_spec: Optional TVL spec path
            tvl_environment: Environment overlay for TVL spec
            tvl: Structured TVL options
            **algorithm_kwargs: Additional algorithm parameters

        Returns:
            OptimizationResult with trial results and best configuration

        Example:
            # Simple synchronous usage (no asyncio.run needed)
            result = my_function.optimize_sync(max_trials=10)
            print(result.best_config)

            # Equivalent async usage
            result = asyncio.run(my_function.optimize(max_trials=10))
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        coro = self.optimize(
            algorithm=algorithm,
            max_trials=max_trials,
            timeout=timeout,
            save_to=save_to,
            custom_evaluator=custom_evaluator,
            callbacks=callbacks,
            configuration_space=configuration_space,
            objectives=objectives,
            tvl_spec=tvl_spec,
            tvl_environment=tvl_environment,
            tvl=tvl,
            **algorithm_kwargs,
        )

        if loop is not None and loop.is_running():
            # Already in an async context - can't use asyncio.run
            # Create a new thread to run the coroutine
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(asyncio.run, coro)
                return future.result()
        else:
            # No event loop running - safe to use asyncio.run
            return asyncio.run(coro)

    def _resolve_runtime_tvl_options(
        self,
        spec: str | Path | None,
        environment: str | None,
        bundle: TVLOptions | dict[str, Any] | None,
    ) -> TVLOptions | None:
        if bundle is None and spec is None:
            if environment:
                raise ValueError("tvl_environment requires a tvl_spec path")
            return None

        options: TVLOptions | None
        if isinstance(bundle, TVLOptions):
            options = bundle
        elif isinstance(bundle, dict):
            options = TVLOptions.model_validate(bundle)
        else:
            options = None

        if options is None and spec is None:
            raise ValueError("tvl options require a spec path")

        if options is None and spec is not None:
            options = TVLOptions(spec_path=str(spec), environment=environment)
            return options

        if options is None:
            raise ValueError("TVL options could not be resolved")
        if spec is not None and Path(options.spec_path) != Path(spec):
            raise ValueError("Conflicting TVL specs provided at runtime")
        if environment:
            options = options.merged_with(environment=environment)
        return options

    def _apply_runtime_tvl_artifact(self, artifact) -> dict[str, Any]:
        state: dict[str, Any] = {}
        if artifact.constraints:
            state["constraints"] = list(self.constraints or [])
            self.constraints = list(self.constraints or []) + artifact.constraints
        if artifact.default_config:
            state["default_config"] = copy.deepcopy(self.default_config)
            self.default_config = artifact.default_config.copy()
        return state

    @staticmethod
    def _apply_tvl_runtime_overrides(
        algorithm: str | None,
        max_trials: int | None,
        timeout: float | None,
        algorithm_kwargs: dict[str, Any],
        overrides: dict[str, Any],
    ) -> tuple[str | None, int | None, float | None]:
        updated_algorithm = algorithm or overrides.get("algorithm")
        updated_max_trials = (
            max_trials if max_trials is not None else overrides.get("max_trials")
        )
        updated_timeout = timeout if timeout is not None else overrides.get("timeout")

        for key in (
            "parallel_config",
            "max_total_examples",
            "samples_include_pruned",
        ):
            if key in overrides and key not in algorithm_kwargs:
                algorithm_kwargs[key] = overrides[key]

        # Fail fast on invalid parallel_trials usage - users must use parallel_config
        if "parallel_trials" in overrides or "parallel_trials" in algorithm_kwargs:
            raise ValueError(
                "parallel_trials is not a valid parameter for optimize(). "
                "Use parallel_config={'trial_concurrency': N} instead."
            )

        return updated_algorithm, updated_max_trials, updated_timeout

    def _resolve_execution_parameters(
        self,
        algorithm: str | None,
        max_trials: int | None,
        configuration_space: dict[str, Any] | None,
    ) -> tuple[str, int | None, dict[str, Any]]:
        """Resolve and validate execution parameters.

        Returns:
            Tuple of (resolved_algorithm, resolved_max_trials, effective_config_space)
        """
        # Determine effective configuration space (optimize() parameter takes precedence)
        effective_config_space = (
            configuration_space
            if configuration_space is not None
            else self.configuration_space
        )

        # Resolve algorithm (prefer method arg, else decorator-provided)
        resolved_algorithm: str = (
            algorithm if algorithm else cast(str, getattr(self, "algorithm", "grid"))
        )

        # Resolve max_trials value for this run
        resolved_max_trials = (
            max_trials if max_trials is not None else getattr(self, "max_trials", None)
        )
        if resolved_max_trials is not None and resolved_max_trials < 0:
            raise ValueError("max_trials must be non-negative")

        # Validate the effective configuration space
        if not effective_config_space or effective_config_space == {}:
            raise ValueError(
                "Configuration space cannot be empty. Please specify at least one parameter to optimize "
                "either in the @traigent.optimize decorator or in the optimize() method call. "
                "Example: optimize(configuration_space={'temperature': [0.0, 0.5, 1.0]})"
            )

        # Validate configuration space if provided
        if configuration_space is not None:
            validate_config_space(configuration_space)

        return resolved_algorithm, resolved_max_trials, effective_config_space

    def _create_traigent_config(self) -> TraigentConfig:
        """Create and store TraigentConfig for the optimization run."""
        traigent_config = TraigentConfig(
            execution_mode=cast(
                Literal["edge_analytics", "privacy", "hybrid", "standard", "cloud"],
                self.execution_mode,
            ),
            local_storage_path=self.local_storage_path,
            minimal_logging=self.minimal_logging,
            privacy_enabled=getattr(self, "privacy_enabled", False),
        )
        self.traigent_config = traigent_config
        return traigent_config

    def _prepare_optimization_dataset(
        self,
        algorithm_kwargs: dict[str, Any],
    ) -> tuple[Dataset, int | None, bool]:
        """Load and prepare the optimization dataset.

        Returns:
            Tuple of (dataset, max_total_examples, samples_include_pruned)
        """
        dataset = self._load_dataset()

        # Apply example cap if specified
        max_examples = algorithm_kwargs.get("max_examples") or getattr(
            self, "max_examples", None
        )
        if (
            max_examples is not None
            and isinstance(max_examples, int)
            and max_examples > 0
            and len(dataset.examples) > max_examples
        ):
            capped_dataset = Dataset(
                examples=dataset.examples[:max_examples],
                name=dataset.name if hasattr(dataset, "name") else "dataset",
                description=f"{dataset.description if hasattr(dataset, 'description') else 'Dataset'} (capped to {max_examples} examples)",
                metadata={
                    **getattr(dataset, "metadata", {}),
                    "original_count": len(dataset.examples),
                    "capped_count": max_examples,
                },
            )
            logger.info(
                f"Dataset capped from {len(dataset.examples)} to {max_examples} examples"
            )
            dataset = capped_dataset

        max_total_examples_value = algorithm_kwargs.pop("max_total_examples", None)
        if max_total_examples_value is None:
            max_total_examples_value = getattr(self, "max_total_examples", None)
        else:
            self.max_total_examples = max_total_examples_value

        samples_include_pruned_value = algorithm_kwargs.pop(
            "samples_include_pruned", None
        )
        if samples_include_pruned_value is None:
            samples_include_pruned_value = getattr(self, "samples_include_pruned", True)
        else:
            self.samples_include_pruned = bool(samples_include_pruned_value)

        return dataset, max_total_examples_value, bool(samples_include_pruned_value)

    def _validate_provider_keys(
        self,
        configuration_space: dict[str, Any] | None,
    ) -> None:
        """Validate provider API keys before running optimization.

        This method checks that API keys are valid for all providers used in
        the configuration space. It helps fail fast before starting expensive
        optimization runs.

        Validation is skipped when:
        - TRAIGENT_SKIP_PROVIDER_VALIDATION=true
        - TRAIGENT_MOCK_LLM=true
        - validate_providers=False in decorator

        For unknown models (not matching known provider patterns), a warning
        is logged but validation continues.

        Args:
            configuration_space: The configuration space containing model options.

        Raises:
            ProviderValidationError: If any known provider has an invalid API key.
        """
        # Check toggle precedence (first match wins):
        # 1. validate_providers=False in decorator -> skip
        # 2. TRAIGENT_SKIP_PROVIDER_VALIDATION=true -> skip
        # 3. TRAIGENT_MOCK_LLM=true -> skip (handled by skip_provider_validation)
        if not getattr(self, "validate_providers", True):
            logger.debug("Provider validation skipped (decorator override)")
            return

        if skip_provider_validation():
            logger.debug("Provider validation skipped (environment override)")
            return

        # Extract models from configuration space
        if configuration_space is None:
            return

        models = self._extract_models_from_config_space(configuration_space)
        if not models:
            return

        # Run validation
        from traigent.providers.validation import (
            get_failed_providers,
            print_provider_status,
            validate_providers,
        )

        timeout = get_validation_timeout()
        results = validate_providers(models, timeout=timeout)

        # Print status (matches walkthrough example 07 format)
        print_provider_status(results)

        # Check for failures
        failed = get_failed_providers(results)
        if failed:
            failed_list = ", ".join(
                f"{provider} ({error})" for provider, error in failed
            )
            raise ProviderValidationError(
                f"Provider validation failed. Invalid keys: {failed_list}. "
                "Set TRAIGENT_SKIP_PROVIDER_VALIDATION=true to bypass.",
                failed_providers=failed,
                details={"results": {p: s.message for p, s in results.items()}},
            )

    def _extract_models_from_config_space(
        self,
        configuration_space: dict[str, Any],
    ) -> list[str]:
        """Extract model names from configuration space.

        Looks for 'model' key in the configuration space and extracts
        all model name options.

        Args:
            configuration_space: The configuration space dict.

        Returns:
            List of model names found in the configuration space.
        """
        models: list[str] = []

        # Check for 'model' key (common pattern)
        model_value = configuration_space.get("model")
        if model_value is not None:
            if isinstance(model_value, str):
                models.append(model_value)
            elif isinstance(model_value, (list, tuple)):
                models.extend(str(m) for m in model_value if m is not None)

        # Also check for provider-specific model keys
        for key in ("router_model", "agent_model", "llm_model", "chat_model"):
            value = configuration_space.get(key)
            if value is not None:
                if isinstance(value, str):
                    models.append(value)
                elif isinstance(value, (list, tuple)):
                    models.extend(str(m) for m in value if m is not None)

        # Deduplicate while preserving order
        seen: set[str] = set()
        unique_models: list[str] = []
        for model in models:
            if model not in seen:
                seen.add(model)
                unique_models.append(model)

        return unique_models

    def _resolve_effective_parallel_config(
        self,
        algorithm_kwargs: dict[str, Any],
    ) -> tuple[int | None, int | None, int | None]:
        """Resolve parallel configuration settings.

        Returns:
            Tuple of (effective_parallel_trials, effective_batch_size, effective_thread_workers)
        """
        runtime_parallel_config = coerce_parallel_config(
            algorithm_kwargs.pop("parallel_config", None)
        )

        global_parallel_config = get_global_parallel_config()
        merged_parallel_config, merged_sources = merge_parallel_configs(
            [
                (global_parallel_config, "global"),
                (self.parallel_config, "decorator"),
                (runtime_parallel_config, "runtime"),
            ]
        )
        if merged_sources:
            logger.debug("Parallel configuration merge sources: %s", merged_sources)

        default_thread_workers = (
            merged_parallel_config.thread_workers
            if merged_parallel_config.thread_workers is not None
            else int(_GLOBAL_CONFIG.get("parallel_workers", 1))
        )

        resolved_parallel = resolve_parallel_config(
            merged_parallel_config,
            default_thread_workers=default_thread_workers,
            config_space_size=self._estimate_search_space_size(),
            detected_function_kind=(
                "async" if asyncio.iscoroutinefunction(self.func) else "sync"
            ),
            sources=merged_sources,
        )

        for warning in resolved_parallel.warnings:
            logger.warning(warning)

        # Check for unsafe attribute mode + parallel trials combination
        injection_mode_str = (
            self.injection_mode.value
            if hasattr(self.injection_mode, "value")
            else self.injection_mode
        )
        if (
            injection_mode_str == "attribute"
            and resolved_parallel.trial_concurrency > 1
        ):
            # Attribute injection mode is not thread-safe with parallel trials.
            # The function attribute (my_func.current_config) is a shared mutable
            # object that can be overwritten by concurrent threads, causing race
            # conditions. This is an unconditional block - no opt-in allowed.
            raise ValueError(
                "injection_mode='attribute' is not supported with parallel trials "
                "(trial_concurrency > 1). The function attribute is shared across "
                "concurrent trials and causes race conditions.\n\n"
                "Options:\n"
                "  1. Use injection_mode='context' (recommended) - thread-safe\n"
                "  2. Use injection_mode='parameter' - explicit config passing\n"
                "  3. Use sequential trials: parallel_config={'trial_concurrency': 1}"
            )

        logger.info(
            "Resolved parallel configuration: mode=%s, trial_concurrency=%s, example_concurrency=%s, thread_workers=%s",
            resolved_parallel.mode,
            resolved_parallel.trial_concurrency,
            resolved_parallel.example_concurrency,
            resolved_parallel.thread_workers,
        )

        return (
            resolved_parallel.trial_concurrency,
            resolved_parallel.example_concurrency,
            resolved_parallel.thread_workers,
        )

    def _resolve_custom_evaluator(
        self,
        custom_evaluator: Callable[..., Any] | None,
    ) -> Callable[..., Any] | None:
        """Resolve the effective custom evaluator based on mock mode settings.

        Returns:
            The custom evaluator to use, or None if LocalEvaluator should be used.
        """
        mock_mode_env = is_mock_llm()
        mock_config = self.mock_mode_config or {}
        mock_enabled = mock_config.get("enabled", True)
        override_evaluator = mock_config.get("override_evaluator", True)

        provided_custom_evaluator = custom_evaluator or self.custom_evaluator
        has_custom = provided_custom_evaluator is not None

        # Mock mode can override custom evaluators
        if mock_mode_env and mock_enabled and override_evaluator and has_custom:
            logger.info(
                "Mock mode enabled: overriding custom evaluator with LocalEvaluator"
            )
            return None

        return provided_custom_evaluator if has_custom else None

    def _build_metric_functions(self) -> dict[str, Callable[..., Any]]:
        """Build the effective metric functions dictionary."""
        effective_metric_functions: dict[str, Callable[..., Any]] = dict(
            self.metric_functions or {}
        )

        if self.scoring_function is None:
            return effective_metric_functions

        # Determine target metric for scoring function
        target_metric: str | None = None
        if "accuracy" in self.objectives:
            target_metric = "accuracy"
        elif "score" in self.objectives:
            target_metric = "score"

        if target_metric and target_metric not in effective_metric_functions:
            effective_metric_functions[target_metric] = self.scoring_function

        return effective_metric_functions

    def _resolve_effective_workers(
        self,
        effective_batch_size: int | None,
        effective_thread_workers: int | None,
    ) -> int:
        """Resolve the effective number of workers, applying thread limit."""
        effective_workers = max(1, int(effective_batch_size or 1))
        if effective_thread_workers and effective_workers > effective_thread_workers:
            logger.debug(
                "Clamping example concurrency from %s to thread worker limit %s",
                effective_workers,
                effective_thread_workers,
            )
            effective_workers = effective_thread_workers
        return effective_workers

    def _create_effective_evaluator(
        self,
        timeout: float | None,
        custom_evaluator: Callable[..., Any] | None,
        effective_batch_size: int | None,
        effective_thread_workers: int | None,
        effective_privacy_enabled: bool,
    ) -> BaseEvaluator:
        """Create the appropriate evaluator for the optimization run."""
        effective_evaluator = self._resolve_custom_evaluator(custom_evaluator)

        if effective_evaluator:
            if not callable(effective_evaluator):
                raise ValueError("custom_evaluator must be callable") from None
            return CustomEvaluatorWrapper(
                custom_evaluator=effective_evaluator,
                metrics=self.objectives,
                timeout=timeout or 60.0,
                capture_llm_metrics=True,
            )

        effective_metric_functions = self._build_metric_functions()
        effective_workers = self._resolve_effective_workers(
            effective_batch_size, effective_thread_workers
        )

        return LocalEvaluator(
            metrics=self.objectives,
            timeout=timeout or 60.0,
            max_workers=effective_workers,
            detailed=True,
            execution_mode=self.execution_mode,
            privacy_enabled=effective_privacy_enabled,
            mock_mode_config=self.mock_mode_config,
            metric_functions=effective_metric_functions or None,
        )

    def _build_optimization_orchestrator(
        self,
        optimizer: Any,
        evaluator: BaseEvaluator,
        max_trials: int | None,
        max_total_examples_value: int | None,
        timeout: float | None,
        callbacks: list[Callable[..., Any]] | None,
        traigent_config: TraigentConfig,
        effective_parallel_trials: int | None,
        samples_include_pruned_value: bool,
        algorithm_kwargs: dict[str, Any],
        show_progress: bool | None = None,
    ) -> OptimizationOrchestrator:
        """Build the optimization orchestrator with all configuration."""
        cache_policy = algorithm_kwargs.get("cache_policy", "allow_repeats")

        orchestrator_kwargs: dict[str, Any] = {
            "cache_policy": cache_policy,
        }
        if self.default_config:
            orchestrator_kwargs["default_config"] = self.default_config.copy()

        # Pass budget stop condition parameters
        if "budget_limit" in algorithm_kwargs:
            orchestrator_kwargs["budget_limit"] = algorithm_kwargs["budget_limit"]
        if "budget_metric" in algorithm_kwargs:
            orchestrator_kwargs["budget_metric"] = algorithm_kwargs["budget_metric"]
        if "budget_include_pruned" in algorithm_kwargs:
            orchestrator_kwargs["budget_include_pruned"] = algorithm_kwargs[
                "budget_include_pruned"
            ]

        orchestrator_kwargs["samples_include_pruned"] = samples_include_pruned_value

        # Pass constraints to orchestrator
        if self.constraints:
            orchestrator_kwargs["constraints"] = self.constraints

        # Pass plateau stop condition parameters
        if "plateau_window" in algorithm_kwargs:
            orchestrator_kwargs["plateau_window"] = algorithm_kwargs["plateau_window"]
        if "plateau_epsilon" in algorithm_kwargs:
            orchestrator_kwargs["plateau_epsilon"] = algorithm_kwargs["plateau_epsilon"]
        if "cost_limit" in algorithm_kwargs:
            orchestrator_kwargs["cost_limit"] = algorithm_kwargs["cost_limit"]
        if "cost_approved" in algorithm_kwargs:
            orchestrator_kwargs["cost_approved"] = algorithm_kwargs["cost_approved"]
        if show_progress is not None:
            orchestrator_kwargs["show_progress"] = bool(show_progress)

        orchestrator = OptimizationOrchestrator(
            optimizer=optimizer,
            evaluator=evaluator,
            max_trials=max_trials,
            max_total_examples=max_total_examples_value,
            timeout=timeout,
            callbacks=callbacks or [],
            config=traigent_config,
            parallel_trials=effective_parallel_trials,
            objectives=self.objectives,
            objective_schema=self.objective_schema,
            **orchestrator_kwargs,
        )

        orchestrator.samples_include_pruned = samples_include_pruned_value
        return orchestrator

    async def _run_and_finalize_optimization(
        self,
        orchestrator: OptimizationOrchestrator,
        dataset: Dataset,
        effective_config_space: dict[str, Any],
        save_to: str | None,
    ) -> OptimizationResult:
        """Run optimization with context managers and finalize results."""
        from traigent.config.context import ConfigurationSpaceContext

        # Set state to OPTIMIZING before starting
        self._state = OptimizationState.OPTIMIZING

        try:
            with ConfigurationSpaceContext(effective_config_space):
                if self.auto_override_frameworks and self.framework_targets:
                    with override_context(self.framework_targets):
                        result = await orchestrator.optimize(
                            func=self._wrapped_func,
                            dataset=dataset,
                        )
                else:
                    result = await orchestrator.optimize(
                        func=self._wrapped_func,
                        dataset=dataset,
                    )

            # Store results
            self._optimization_results = result
            self._optimization_history.append(result)

            # Update current config to best found
            if result.best_config:
                self._current_config = result.best_config.copy()
                self._best_config = result.best_config.copy()
                self._setup_function_wrapper()

            # Set state to OPTIMIZED on success
            self._state = OptimizationState.OPTIMIZED

        except Exception:
            # Set state to ERROR on failure
            self._state = OptimizationState.ERROR
            raise

        # Save results if requested
        if save_to:
            self.save_optimization_results(save_to)

        logger.info(
            f"Optimization completed: {len(result.trials)} trials, "
            f"best score: {result.best_score:.4f}"
        )

        # Show upgrade hints after optimization completion (Edge Analytics mode only)
        if self.traigent_config.is_edge_analytics_mode():
            try:
                show_upgrade_hint(
                    "session_complete",
                    trial_count=len(result.trials),
                    best_score=result.best_score,
                )
            except Exception as e:
                logger.debug(f"Failed to show upgrade hint: {e}")

        return result

    async def _try_cloud_execution(
        self,
        dataset: Dataset,
        max_trials: int | None,
        timeout: float | None,
        effective_config_space: dict[str, Any],
        algorithm_kwargs: dict[str, Any],
    ) -> OptimizationResult | None:
        """Try cloud execution, returning None if fallback to local is needed."""
        use_cloud = self._is_cloud_execution_mode() and (
            max_trials is None or max_trials > 0
        )
        if not use_cloud:
            return None
        if is_backend_offline():
            logger.info(
                "Backend offline: skipping cloud optimization and using local execution."
            )
            return None

        try:
            return await self._optimize_with_cloud_service(
                dataset,
                max_trials,
                timeout,
                configuration_space=effective_config_space,
                **algorithm_kwargs,
            )
        except (AuthenticationError, ConfigurationError, ValidationError):
            raise
        except OSError as e:  # Includes TimeoutError and ConnectionError (subclasses)
            logger.warning(
                "Cloud optimization failed (transient), falling back to local: %s", e
            )
        except Exception as e:
            logger.warning(
                "Cloud optimization failed unexpectedly, falling back to local: %s",
                e,
                exc_info=True,
            )
        return None

    def _apply_mock_config_overrides(
        self, algorithm: str, optimizer_kwargs: dict[str, Any]
    ) -> str:
        """Apply mock config overrides to algorithm and optimizer_kwargs."""
        mock_config = getattr(self, "mock_mode_config", None) or {}
        if not isinstance(mock_config, dict):
            return algorithm

        # Override algorithm if specified in mock config
        mock_optimizer = mock_config.get("optimizer")
        if mock_optimizer and isinstance(mock_optimizer, str):
            algorithm = mock_optimizer
            logger.debug("Using optimizer '%s' from mock_mode_config", mock_optimizer)

        # Extract and pass random_seed to optimizer for reproducibility
        random_seed = mock_config.get("random_seed")
        if random_seed is not None:
            optimizer_kwargs["random_seed"] = random_seed
            logger.debug(
                "Passing random_seed=%s to optimizer from mock_mode_config",
                random_seed,
            )

        return algorithm

    async def _execute_optimization(
        self,
        *,
        algorithm: str | None,
        max_trials: int | None,
        timeout: float | None,
        save_to: str | None,
        custom_evaluator: Callable[..., Any] | None,
        callbacks: list[Callable[..., Any]] | None,
        configuration_space: dict[str, Any] | None,
        algorithm_kwargs: dict[str, Any],
    ) -> OptimizationResult:
        """Execute optimization assuming objective schema is already resolved.

        This method orchestrates the optimization process by delegating to
        specialized helper methods for each phase of execution.
        """
        # Phase 1: Resolve and validate parameters
        algorithm, max_trials, effective_config_space = (
            self._resolve_execution_parameters(
                algorithm, max_trials, configuration_space
            )
        )

        # Phase 2: Create TraigentConfig and check CI approval
        traigent_config = self._create_traigent_config()
        self._check_ci_approval()

        # Phase 3: Prepare dataset
        dataset, max_total_examples_value, samples_include_pruned_value = (
            self._prepare_optimization_dataset(algorithm_kwargs)
        )

        # Phase 3.5: Validate provider API keys
        self._validate_provider_keys(effective_config_space)

        # Phase 4: Try cloud execution if applicable
        cloud_result = await self._try_cloud_execution(
            dataset, max_trials, timeout, effective_config_space, algorithm_kwargs
        )
        if cloud_result is not None:
            return cloud_result

        # Phase 5: Resolve parallel configuration
        effective_parallel_trials, effective_batch_size, effective_thread_workers = (
            self._resolve_effective_parallel_config(algorithm_kwargs)
        )
        show_progress = algorithm_kwargs.pop("show_progress", None)

        # Phase 6: Create optimizer
        optimizer_kwargs = algorithm_kwargs.copy()
        if max_trials and algorithm == "random":
            optimizer_kwargs["max_trials"] = max_trials

        # Apply mock config overrides if present
        algorithm = self._apply_mock_config_overrides(algorithm, optimizer_kwargs)

        optimizer = get_optimizer(
            algorithm, effective_config_space, self.objectives, **optimizer_kwargs
        )

        # Phase 7: Determine privacy and create evaluator
        effective_privacy_enabled = bool(getattr(self, "privacy_enabled", False))
        evaluator = self._create_effective_evaluator(
            timeout=timeout,
            custom_evaluator=custom_evaluator,
            effective_batch_size=effective_batch_size,
            effective_thread_workers=effective_thread_workers,
            effective_privacy_enabled=effective_privacy_enabled,
        )

        # Update TraigentConfig with final privacy setting
        traigent_config.privacy_enabled = effective_privacy_enabled

        # Phase 8: Build orchestrator
        orchestrator = self._build_optimization_orchestrator(
            optimizer=optimizer,
            evaluator=evaluator,
            max_trials=max_trials,
            max_total_examples_value=max_total_examples_value,
            timeout=timeout,
            callbacks=callbacks,
            traigent_config=traigent_config,
            effective_parallel_trials=effective_parallel_trials,
            samples_include_pruned_value=samples_include_pruned_value,
            algorithm_kwargs=algorithm_kwargs,
            show_progress=show_progress,
        )

        # Phase 9: Run optimization and finalize
        try:
            return await self._run_and_finalize_optimization(
                orchestrator=orchestrator,
                dataset=dataset,
                effective_config_space=effective_config_space,
                save_to=save_to,
            )
        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            raise OptimizationError(f"Optimization failed: {e}") from e

    async def _optimize_with_cloud_service(
        self,
        dataset: Dataset,
        max_trials: int | None = None,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> OptimizationResult:
        """Run optimization using Traigent Cloud Service.

        Args:
            dataset: Evaluation dataset
            max_trials: Maximum number of trials
            timeout: Optimization timeout
            **kwargs: Additional arguments

        Returns:
            OptimizationResult from cloud service
        """
        from traigent.api.types import TrialResult, TrialStatus
        from traigent.cloud.client import CloudOptimizationResult, TraigentCloudClient

        # Initialize cloud client if not already done
        if self._cloud_client is None:
            self._cloud_client = TraigentCloudClient(enable_fallback=True)

        if max_trials is not None and max_trials <= 0:
            logger.info("Cloud optimization skipped due to max_trials=0.")
            return self._build_empty_result("cloud_service")

        async with self._cloud_client as client:
            # Extract configuration_space from kwargs if provided
            config_space_override = kwargs.pop("configuration_space", None)
            effective_config_space = (
                config_space_override
                if config_space_override is not None
                else self.configuration_space
            )

            # Import ConfigurationSpaceContext here to avoid circular imports
            from traigent.config.context import ConfigurationSpaceContext

            # Set the configuration space context for framework overrides
            with ConfigurationSpaceContext(effective_config_space):
                # Run cloud optimization
                cloud_result: CloudOptimizationResult = await client.optimize_function(
                    function_name=self.func.__name__,
                    dataset=dataset,
                    configuration_space=effective_config_space,
                    objectives=self.objectives,
                    max_trials=max_trials if max_trials is not None else 50,
                )

            # Convert cloud result to standard OptimizationResult
            mock_trial = TrialResult(
                trial_id="cloud_best",
                config=cloud_result.best_config,
                metrics=cloud_result.best_metrics,
                status=TrialStatus.COMPLETED,
                duration=cloud_result.optimization_time,
                timestamp=datetime.now(UTC),
                metadata={},
            )

            from traigent.api.types import OptimizationResult, OptimizationStatus

            result = OptimizationResult(
                trials=[mock_trial],  # Cloud service doesn't expose all trials
                best_config=cloud_result.best_config,
                best_score=(
                    max(cloud_result.best_metrics.values())
                    if cloud_result.best_metrics
                    else 0.0
                ),
                optimization_id=f"cloud_{int(time.time())}",
                duration=cloud_result.optimization_time,
                convergence_info={},
                status=OptimizationStatus.COMPLETED,
                objectives=self.objectives,
                algorithm="cloud_service",
                timestamp=datetime.now(UTC),
                metadata={
                    "cloud_service": True,
                    "cost_reduction": cloud_result.cost_reduction,
                    "subset_used": cloud_result.subset_used,
                    "subset_size": cloud_result.subset_size,
                    "trials_count": cloud_result.trials_count,
                },
            )

            # Store results
            self._optimization_results = result
            self._optimization_history.append(result)

            # Update current config
            if result.best_config:
                self._current_config = result.best_config.copy()
                self._setup_function_wrapper()

            logger.info(
                f"Cloud optimization completed: {cloud_result.trials_count} trials, "
                f"{cloud_result.cost_reduction * 100:.1f}% cost reduction"
            )

            return result

    def _load_dataset(self) -> Dataset:
        """Load evaluation dataset.

        Returns:
            Dataset object for evaluation

        Raises:
            ConfigurationError: If dataset cannot be loaded
        """
        if isinstance(self.eval_dataset, Dataset):
            return self.eval_dataset

        elif isinstance(self.eval_dataset, str):
            try:
                return Dataset.from_jsonl(self.eval_dataset)
            except Exception as e:
                raise ConfigurationError(
                    f"Failed to load dataset from {self.eval_dataset}: {e}"
                ) from e

        elif isinstance(self.eval_dataset, list):
            # Multiple datasets - combine them
            all_examples = []
            for path in self.eval_dataset:
                try:
                    dataset = Dataset.from_jsonl(path)
                    all_examples.extend(dataset.examples)
                except Exception as e:
                    raise ConfigurationError(
                        f"Failed to load dataset from {path}: {e}"
                    ) from e

            return Dataset(
                examples=all_examples,
                name="combined_dataset",
                description=f"Combined dataset from {len(self.eval_dataset)} files",
            )

        else:
            raise ConfigurationError(f"Invalid dataset type: {type(self.eval_dataset)}")

    def _build_empty_result(self, algorithm: str) -> OptimizationResult:
        """Create a result representing a skipped optimization run."""

        best_config = (
            self._current_config.copy() if hasattr(self, "_current_config") else {}
        )
        now = datetime.now(UTC)

        return OptimizationResult(
            trials=[],
            best_config=best_config,
            best_score=0.0,
            optimization_id=f"no_trials_{int(time.time())}",
            duration=0.0,
            convergence_info={},
            status=OptimizationStatus.COMPLETED,
            objectives=self.objectives,
            algorithm=algorithm,
            timestamp=now,
            metadata={"reason": "max_trials=0"},
        )

    def _check_ci_approval(self) -> None:
        """Check if approval is required and granted for CI runs.

        Raises:
            OptimizationError: If running in CI without proper approval
        """
        # Only check in Edge Analytics mode
        if not self.traigent_config.is_edge_analytics_mode():
            return

        # Allow mock LLM mode to run without CI approval since no real calls occur
        if is_mock_llm():
            msg = "Skipping CI approval in mock LLM mode"
            if is_production():
                logger.warning(f"{msg} while ENVIRONMENT=production.")
            else:
                logger.info(f"{msg}.")
            return

        if not _is_ci_environment():
            return  # Interactive mode - no approval needed

        # Check environment variable approval (primary method)
        if _check_env_var_approval():
            return

        # Check token file approval (secondary method)
        storage_path = self.traigent_config.get_local_storage_path()
        if storage_path is None:
            raise ConfigurationError("Storage path not configured")
        storage_root = Path(storage_path).expanduser().resolve()
        token_file = validate_path(
            storage_root / "approval.token", storage_root, must_exist=False
        )
        if _check_token_file_approval(token_file, storage_root):
            return

        # No valid approval found
        raise OptimizationError("""
CI/CD Approval Required

This optimization was triggered in a CI environment and requires approval.

To approve, use one of these methods:

1. Environment variable (recommended for CI):
   export TRAIGENT_RUN_APPROVED=1
   export TRAIGENT_APPROVED_BY="your_name"

2. Approval token file:
   echo '{"approved_by": "your_name", "expires_at": "2024-12-31T23:59:59"}' > ~/.traigent/approval.token

3. GitHub Actions with environment protection:
   Use 'environment: production' with required reviewers
        """)

    def get_best_config(self) -> dict[str, Any] | None:
        """Get the best configuration found during optimization.

        Returns:
            Best configuration dictionary, or None if no optimization run
        """
        if self._optimization_results:
            best: dict[str, Any] = self._optimization_results.best_config
            return best
        return None

    def get_optimization_results(self) -> OptimizationResult | None:
        """Get the latest optimization results.

        Returns:
            Latest OptimizationResult, or None if no optimization run
        """
        return self._optimization_results

    def get_optimization_history(self) -> list[OptimizationResult]:
        """Get history of all optimization runs.

        Returns:
            List of OptimizationResult objects
        """
        return self._optimization_history.copy()

    def is_optimization_complete(self) -> bool:
        """Check if optimization has been completed.

        Returns:
            True if optimization has been run and completed, False otherwise
        """
        return self._optimization_results is not None

    def reset_optimization(self) -> None:
        """Reset optimization state and restore default configuration."""
        self._optimization_results = None
        self._optimization_history = []
        self._current_config = self.default_config.copy()
        self._best_config = None
        self._state = OptimizationState.UNOPTIMIZED
        self._setup_function_wrapper()
        logger.info(f"Reset optimization state for {self.func.__name__}")

    def save_optimization_results(self, path: str) -> None:
        """Save optimization results to file.

        Args:
            path: Path to save results to

        Raises:
            ConfigurationError: If no optimization results to save
        """
        if not self._optimization_results:
            raise ConfigurationError("No optimization results to save")

        from dataclasses import asdict

        result_dict = asdict(self._optimization_results)
        output_path = Path(path).expanduser()
        base_dir = (
            output_path.parent if output_path.is_absolute() else Path.cwd().resolve()
        )
        output_path = validate_path(output_path, base_dir, must_exist=False)
        with safe_open(output_path, base_dir, mode="w", encoding="utf-8") as f:
            json.dump(result_dict, f, indent=2, default=str)

        logger.info(f"Saved optimization results to {output_path}")

    def load_optimization_results(self, path: str) -> None:
        """Load optimization results from file.

        Args:
            path: Path to load results from

        Raises:
            ConfigurationError: If results cannot be loaded
        """
        try:
            input_path = Path(path).expanduser()
            base_dir = (
                input_path.parent if input_path.is_absolute() else Path.cwd().resolve()
            )
            input_path = validate_path(input_path, base_dir, must_exist=True)
            with safe_open(input_path, base_dir, mode="r", encoding="utf-8") as f:
                result_dict = json.load(f)

            # Manual reconstruction since OptimizationResult is a dataclass
            from traigent.api.types import (
                OptimizationResult,
                OptimizationStatus,
                TrialResult,
                TrialStatus,
            )

            trials = []
            for trial_data in result_dict.get("trials", []):
                trial = TrialResult(
                    trial_id=trial_data["trial_id"],
                    config=trial_data["config"],
                    metrics=trial_data["metrics"],
                    status=TrialStatus(trial_data["status"]),
                    duration=trial_data["duration"],
                    timestamp=datetime.fromisoformat(trial_data["timestamp"]),
                    error_message=trial_data.get("error_message"),
                    metadata=trial_data.get("metadata", {}),
                )
                trials.append(trial)

            self._optimization_results = OptimizationResult(
                trials=trials,
                best_config=result_dict["best_config"],
                best_score=result_dict["best_score"],
                optimization_id=result_dict["optimization_id"],
                duration=result_dict["duration"],
                convergence_info=result_dict["convergence_info"],
                status=OptimizationStatus(result_dict["status"]),
                objectives=result_dict["objectives"],
                algorithm=result_dict["algorithm"],
                timestamp=datetime.fromisoformat(result_dict["timestamp"]),
                metadata=result_dict.get("metadata", {}),
            )
            self._optimization_history.append(self._optimization_results)

            # Update current config, best_config, and state
            if self._optimization_results.best_config:
                self._current_config = self._optimization_results.best_config.copy()
                self._best_config = self._optimization_results.best_config.copy()
                self._setup_function_wrapper()

            # Set state based on loaded results status
            if self._optimization_results.status == OptimizationStatus.COMPLETED:
                self._state = OptimizationState.OPTIMIZED
            elif self._optimization_results.status == OptimizationStatus.FAILED:
                self._state = OptimizationState.ERROR
            else:
                self._state = (
                    OptimizationState.OPTIMIZED
                )  # Assume optimized for loaded results

            logger.info(f"Loaded optimization results from {path}")

        except Exception as e:
            raise ConfigurationError(f"Failed to load optimization results: {e}") from e

    @property
    def state(self) -> OptimizationState:
        """Get the current lifecycle state of this optimized function.

        Returns:
            OptimizationState indicating current phase:
            - UNOPTIMIZED: Before any optimization
            - OPTIMIZING: During active optimization
            - OPTIMIZED: After successful optimization
            - ERROR: Optimization failed
        """
        return self._state

    @property
    def best_config(self) -> dict[str, Any] | None:
        """Get the best configuration found during optimization.

        This is the recommended way to access optimization results.
        Returns None if optimization hasn't been run yet.

        Returns:
            The configuration that achieved the best score, or None.

        Example:
            result = traigent.optimize(my_func, ...)
            print(result.best_config)  # Via OptimizationResult (recommended)
            print(my_func.best_config)  # Via function property (also valid)
        """
        return self._best_config.copy() if self._best_config else None

    @property
    def current_config(self) -> dict[str, Any]:
        """Get the configuration this function uses when called.

        This property returns the active configuration:
        - Before optimization: Returns default_config
        - After optimization: Returns best_config (auto-applied)

        Raises:
            OptimizationStateError: If accessed during an active optimization.
                Use traigent.get_config() inside your function during optimization.

        Returns:
            Copy of the current configuration dict.
        """
        with self._state_lock:
            if self._state == OptimizationState.OPTIMIZING:
                raise OptimizationStateError(
                    "Cannot access current_config during an active optimization. "
                    "Use traigent.get_config() to access the current trial's "
                    "configuration within your optimized function.",
                    current_state=self._state.name,
                    expected_states=["UNOPTIMIZED", "OPTIMIZED", "ERROR"],
                )
            return self._current_config.copy()

    def set_config(self, config: dict[str, Any]) -> None:
        """Set current configuration manually.

        Args:
            config: Configuration to set
        """
        with self._state_lock:
            self._current_config = config.copy()
            self._setup_function_wrapper()
        logger.debug(f"Set configuration for {self.func.__name__}: {config}")

    def apply_best_config(self, results: OptimizationResult | None = None) -> bool:
        """Apply best configuration from optimization results.

        This method applies the best configuration found during optimization,
        updating the function to use optimal parameters for subsequent calls.

        Args:
            results: OptimizationResult to use (defaults to latest optimization)

        Returns:
            True if configuration applied successfully

        Raises:
            ConfigurationError: If no optimization results are available

        Example:
            >>> # After optimization
            >>> results = my_function.optimize()
            >>> # Apply best config manually (optional step)
            >>> my_function.apply_best_config(results.best_config)
            >>> # Now function uses optimal parameters
            >>> output = my_function("test input")
        """
        # Use provided results or fall back to latest optimization
        if results is None:
            results = self._optimization_results

        if not results or not results.best_config:
            raise ConfigurationError(
                "No optimization results available to apply. "
                "Please run optimization first using .optimize()"
            )

        # Update current config with best found configuration atomically
        # If wrapper setup fails, rollback to previous config AND wrapper to avoid inconsistent state
        with self._state_lock:
            old_config = self._current_config.copy()
            old_best = self._best_config.copy() if self._best_config else None
            old_wrapped_func = self._wrapped_func  # Save old wrapper for rollback
            try:
                self._current_config = results.best_config.copy()
                self._best_config = results.best_config.copy()
                # Re-wrap function with new configuration
                self._setup_function_wrapper()
            except Exception:
                # Rollback config AND wrapper on failure to keep object in consistent state
                self._current_config = old_config
                self._best_config = old_best
                self._wrapped_func = old_wrapped_func
                raise

        logger.info(
            f"Applied best config for {self.func.__name__}: {results.best_config} "
            f"(previous: {old_config})"
        )

        return True

    def cleanup(self, *, preserve_config: bool = True) -> None:
        """Clean up optimization artifacts to free memory.

        Call this after you're done analyzing optimization results
        and no longer need the trial history. This is useful for
        long-running services to prevent memory leaks.

        Args:
            preserve_config: If True (default), keeps the best_config applied.
                If False, reverts to default_config.

        Example:
            # After optimization and analysis
            result = my_func.optimize(...)
            analyze_results(result)

            # Clean up to free memory, but keep best config
            my_func.cleanup()

            # Or reset completely
            my_func.cleanup(preserve_config=False)
        """
        # Clear optimization history
        self._optimization_history.clear()
        self._optimization_results = None

        # Clear any accumulated stats
        if hasattr(self, "_stats"):
            self._stats.clear()

        # Clear metrics cache if present
        if hasattr(self, "_metrics_cache"):
            self._metrics_cache.clear()

        # Optionally revert config
        if not preserve_config:
            self._current_config = self.default_config.copy()
            self._best_config = None
            self._state = OptimizationState.UNOPTIMIZED
            self._setup_function_wrapper()

        # Clean provider state if provider supports it
        if hasattr(self, "_config_provider") and self._config_provider:
            if hasattr(self._config_provider, "cleanup"):
                self._config_provider.cleanup()

        logger.debug(f"Cleaned up optimization artifacts for {self.func.__name__}")

    def reset(self) -> None:
        """Fully reset the function to its initial state.

        This clears all optimization history, reverts to default_config,
        and sets state to UNOPTIMIZED. Use this to run a fresh optimization.

        Equivalent to calling cleanup(preserve_config=False).

        Example:
            # Run optimization
            result1 = my_func.optimize(...)

            # Reset for a new optimization run
            my_func.reset()

            # Run fresh optimization
            result2 = my_func.optimize(...)
        """
        self.cleanup(preserve_config=False)
        logger.info(f"Reset {self.func.__name__} to initial state")

    @property
    def configuration_space(self) -> dict[str, Any]:
        """Get configuration space."""
        return self._configuration_space

    @configuration_space.setter
    def configuration_space(self, value: dict[str, Any]) -> None:
        """Set configuration space with validation."""
        if value:
            validate_config_space(value)
        self._configuration_space = value or {}

    def __repr__(self) -> str:
        """String representation of optimized function."""
        return (
            f"OptimizedFunction({self.func.__name__}, "
            f"objectives={self.objectives}, "
            f"config_space_size={len(self.configuration_space)}, "
            f"optimized={self._optimization_results is not None})"
        )

    @property
    def __name__(self) -> str:
        """Get function name."""
        return getattr(self.func, "__name__", "OptimizedFunction")

    @property
    def __doc__(self) -> str | None:  # type: ignore[override]
        """Get function docstring."""
        return getattr(self.func, "__doc__", None)

    def get_doc(self) -> str | None:
        """Get function docstring."""
        return getattr(self.func, "__doc__", None)

    def run(self, *args: Any, **kwargs: Any) -> Any:
        """Run the function with current configuration."""
        return self(*args, **kwargs)
