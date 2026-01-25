"""Configuration types for Traigent SDK."""

# Traceability: CONC-ConfigInjection CONC-CloudService FUNC-API-ENTRY FUNC-CLOUD-HYBRID REQ-API-001 REQ-INJ-002 REQ-CLOUD-009 SYNC-CloudHybrid CONC-Layer-Core

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Literal

from traigent.utils.validation import Validators, validate_or_raise


class ExecutionMode(str, Enum):
    """Execution modes for Traigent optimization.

    Each mode provides different trade-offs between privacy, performance, and features:

    - EDGE_ANALYTICS: Optimization and LLM calls execute on the client.
      Non-sensitive telemetry syncs to the backend when available for analytics; runs continue
      if the backend is unreachable, but insights remain local.
    - PRIVACY: Legacy alias for hybrid mode with strict privacy toggles (no input/output sent).
    - STANDARD: Cloud orchestration with data sharing for balanced performance.
    - CLOUD: Full SaaS execution where optimization and trials run in the cloud.
    """

    EDGE_ANALYTICS = "edge_analytics"
    PRIVACY = "privacy"  # Back-compat alias; prefer HYBRID + privacy_enabled
    HYBRID = "hybrid"
    STANDARD = "standard"
    CLOUD = "cloud"


def resolve_execution_mode(
    mode: ExecutionMode | str | None,
    *,
    default: ExecutionMode = ExecutionMode.CLOUD,
) -> ExecutionMode:
    """Normalize user-provided execution mode values into an ExecutionMode enum."""

    if mode is None:
        return default
    if isinstance(mode, ExecutionMode):
        return mode
    if isinstance(mode, str):
        normalized = mode.strip().lower()
        if not normalized:
            return default
        try:
            return ExecutionMode(normalized)
        except ValueError as exc:  # pragma: no cover - defensive
            raise ValueError(
                f"execution_mode must be one of {[m.value for m in ExecutionMode]}, got '{mode}'"
            ) from exc
    raise TypeError(
        f"execution_mode must be a string or ExecutionMode, got {type(mode).__name__}"
    )


class InjectionMode(str, Enum):
    """Configuration injection modes for Traigent optimization.

    Each mode provides a different way to inject configuration into optimized functions:

    - CONTEXT: Uses Python's contextvars for thread-safe configuration (default)
    - PARAMETER: Adds explicit configuration parameter to function signature
    - SEAMLESS: Modifies variable assignments in function source code (zero code change)

    Note:
        ATTRIBUTE mode was removed in v2.x due to thread-safety issues with parallel
        trials. Use CONTEXT (recommended) or SEAMLESS instead.
    """

    CONTEXT = "context"
    PARAMETER = "parameter"
    SEAMLESS = "seamless"


def is_traigent_disabled() -> bool:
    """Check if Traigent is disabled via environment variable.

    When TRAIGENT_DISABLED=1 (or 'true'/'yes'), the @traigent.optimize decorator
    becomes a pass-through that returns the original function unchanged.

    This allows production deployments to disable Traigent without code changes.

    Returns:
        True if Traigent is disabled, False otherwise.

    Example:
        # In production, set TRAIGENT_DISABLED=1 to disable all optimization
        >>> import os
        >>> os.environ["TRAIGENT_DISABLED"] = "1"
        >>> is_traigent_disabled()
        True
    """
    val = os.getenv("TRAIGENT_DISABLED", "").lower()
    return val in ("1", "true", "yes")


@dataclass
class TraigentConfig:
    """Type-safe configuration container for Traigent optimization.

    This class provides structured access to configuration parameters
    with type hints and validation.

    Attributes:
        model: The model identifier (e.g., "gpt-4o-mini", "GPT-4o")
        temperature: Sampling temperature for randomness (0.0 to 2.0)
        max_tokens: Maximum number of tokens to generate
        top_p: Nucleus sampling parameter
        frequency_penalty: Penalty for repeated tokens
        presence_penalty: Penalty for new topics
        stop_sequences: List of sequences that stop generation
        seed: Random seed for reproducibility
        custom_params: Additional custom parameters

    Example:
        >>> config = TraigentConfig(
        ...     model="GPT-4o",
        ...     temperature=0.7,
        ...     max_tokens=1000
        ... )
        >>> config.model
        'GPT-4o'
    """

    # Core LLM parameters
    model: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    top_p: float | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None

    # Generation control
    stop_sequences: list[str] | None = None
    seed: int | None = None

    # Custom parameters
    custom_params: dict[str, Any] = field(default_factory=dict)

    # Execution mode and storage configuration
    execution_mode: Literal[
        "edge_analytics",
        "privacy",
        "hybrid",
        "standard",
        "cloud",
    ] = "edge_analytics"
    local_storage_path: str | None = None
    minimal_logging: bool = True
    auto_sync: bool = False  # Auto-sync to cloud when API key available
    # Privacy toggle: when True, do not log or transmit input/output/prompts (local/hybrid)
    privacy_enabled: bool = False

    # Metrics configuration
    strict_metrics_nulls: bool = (
        False  # When True, use None instead of 0.0 for missing metrics
    )

    # Analytics and telemetry settings
    enable_usage_analytics: bool = (
        True  # Send privacy-safe usage stats to encourage cloud adoption
    )
    analytics_endpoint: str | None = None  # Custom analytics endpoint
    anonymous_user_id: str | None = None  # Anonymous identifier (auto-generated)

    def __post_init__(self) -> None:
        """Validate configuration parameters using unified validators."""
        # Validate temperature
        if self.temperature is not None:
            result = Validators.validate_number(
                self.temperature, "temperature", 0.0, 2.0
            )
            validate_or_raise(result)

        # Validate max_tokens
        if self.max_tokens is not None:
            result = Validators.validate_positive_int(self.max_tokens, "max_tokens")
            validate_or_raise(result)

        # Validate top_p
        if self.top_p is not None:
            result = Validators.validate_probability(self.top_p, "top_p")
            validate_or_raise(result)

        # Validate penalties
        for penalty_name, penalty_value in [
            ("frequency_penalty", self.frequency_penalty),
            ("presence_penalty", self.presence_penalty),
        ]:
            if penalty_value is not None:
                result = Validators.validate_number(
                    penalty_value, penalty_name, -2.0, 2.0
                )
                validate_or_raise(result)

        # Validate execution mode using Enum helper
        mode_enum = resolve_execution_mode(self.execution_mode)
        if mode_enum is ExecutionMode.PRIVACY:
            # Map legacy 'privacy' to 'hybrid' + privacy_enabled=True
            mode_enum = ExecutionMode.HYBRID
            self.privacy_enabled = True

        self.execution_mode = mode_enum.value

        # Handle local storage path
        if self.is_edge_analytics_mode() and self.local_storage_path:
            # Validate and expand path
            self.local_storage_path = str(
                Path(self.local_storage_path).expanduser().resolve()
            )

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary.

        Returns:
            Dictionary representation of configuration, excluding None values and defaults.
        """
        result = {}

        # Define default values to exclude
        defaults = {
            "execution_mode": "cloud",
            "minimal_logging": True,
            "auto_sync": False,
            "privacy_enabled": False,
            "strict_metrics_nulls": False,
        }

        # Add defined parameters (only if not None and not default)
        for field_name in [
            "model",
            "temperature",
            "max_tokens",
            "top_p",
            "frequency_penalty",
            "presence_penalty",
            "stop_sequences",
            "seed",
            "execution_mode",
            "local_storage_path",
            "minimal_logging",
            "auto_sync",
            "privacy_enabled",
            "strict_metrics_nulls",
        ]:
            value = getattr(self, field_name)
            if value is not None and value != defaults.get(field_name):
                result[field_name] = value

        # Add custom parameters
        result.update(self.custom_params)

        return result

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> TraigentConfig:
        """Create TraigentConfig from dictionary.

        Args:
            config_dict: Dictionary of configuration parameters

        Returns:
            TraigentConfig instance
        """
        # Separate known fields from custom parameters
        known_fields = {
            "model",
            "temperature",
            "max_tokens",
            "top_p",
            "frequency_penalty",
            "presence_penalty",
            "stop_sequences",
            "seed",
            "execution_mode",
            "local_storage_path",
            "minimal_logging",
            "auto_sync",
            "privacy_enabled",
            "strict_metrics_nulls",
        }

        known_params = {k: v for k, v in config_dict.items() if k in known_fields}
        custom_params = {k: v for k, v in config_dict.items() if k not in known_fields}

        return cls(custom_params=custom_params, **known_params)

    def merge(self, other: TraigentConfig | dict[str, Any]) -> TraigentConfig:
        """Merge with another configuration.

        Args:
            other: Another TraigentConfig or dictionary to merge

        Returns:
            New TraigentConfig with merged values (other takes precedence)
        """
        if isinstance(other, dict):
            other = self.from_dict(other)

        # Start with current values
        merged_dict = self.to_dict()

        # Override with other values
        merged_dict.update(other.to_dict())

        return self.from_dict(merged_dict)

    def get(self, key: str, default: Any | None = None) -> Any:
        """Get configuration value with optional default.

        Args:
            key: Parameter name to retrieve
            default: Default value if key not found

        Returns:
            Parameter value or default
        """
        # Check if it's a standard field
        if hasattr(self, key):
            value = getattr(self, key)
            if value is not None:
                return value

        # Check custom parameters
        if key in self.custom_params:
            return self.custom_params[key]

        return default

    def __getitem__(self, key: str) -> Any:
        """Dictionary-style access to configuration."""
        value = self.get(key)
        if value is None and key not in self.to_dict():
            raise KeyError(f"Configuration key '{key}' not found")
        return value

    def __setattr__(self, key: str, value: Any) -> None:
        """Validate assignments for known configuration fields."""
        dataclass_fields = getattr(self, "__dataclass_fields__", {})
        if key in dataclass_fields and key != "custom_params":
            value = self._validated_assignment(key, value)
            object.__setattr__(self, key, value)
        elif key == "custom_params":
            if value is None:
                value = {}
            if not isinstance(value, dict):
                raise TypeError("custom_params must be a dictionary")
            object.__setattr__(self, key, value)
        else:
            object.__setattr__(self, key, value)

    def __setitem__(self, key: str, value: Any) -> None:
        """Dictionary-style setting of configuration."""
        # Check if it's a standard field
        if hasattr(self, key) and key != "custom_params":
            setattr(self, key, value)
        else:
            # Add to custom parameters
            self.custom_params[key] = value

    def __contains__(self, key: str) -> bool:
        """Check if configuration contains key."""
        return key in self.to_dict()

    def __repr__(self) -> str:
        """String representation of configuration."""
        params = []
        for key, value in self.to_dict().items():
            if isinstance(value, str):
                params.append(f"{key}='{value}'")
            else:
                params.append(f"{key}={value}")
        return f"TraigentConfig({', '.join(params)})"

    @property
    def execution_mode_enum(self) -> ExecutionMode:
        """Return execution mode as an ExecutionMode enum."""
        return resolve_execution_mode(self.execution_mode)

    def is_edge_analytics_mode(self) -> bool:
        """Check if configuration uses Edge Analytics mode."""
        return self.execution_mode_enum is ExecutionMode.EDGE_ANALYTICS

    def is_cloud_mode(self) -> bool:
        """Check if configuration is set to cloud mode."""
        return self.execution_mode_enum in {
            ExecutionMode.CLOUD,
            ExecutionMode.STANDARD,
        }

    def is_privacy_enabled(self) -> bool:
        """Whether privacy mode is enabled (content never logged or transmitted)."""
        return bool(self.privacy_enabled)

    def is_strict_metrics_nulls_enabled(self) -> bool:
        """Whether strict metrics nulls is enabled (use None instead of 0.0 for missing metrics)."""
        return bool(self.strict_metrics_nulls)

    def get_local_storage_path(self) -> str | None:
        """Get the local storage path, checking environment variables if not set."""
        explicit_path = (
            getattr(self, "_local_storage_path", None) or self.local_storage_path
        )
        if explicit_path:
            return explicit_path

        # Check environment variable (like DeepEval's DEEPEVAL_RESULTS_FOLDER)
        env_path = os.getenv("TRAIGENT_RESULTS_FOLDER")
        if env_path:
            return str(Path(env_path).expanduser().resolve())

        # Default to ~/.traigent/ for Edge Analytics mode
        if self.is_edge_analytics_mode():
            return str(Path.home() / ".traigent")

        return None

    def _validated_assignment(self, key: str, value: Any) -> Any:
        """Apply per-field validation when assigning configuration values."""
        if key == "temperature" and value is not None:
            validate_or_raise(
                Validators.validate_number(value, "temperature", 0.0, 2.0)
            )
        elif key == "max_tokens" and value is not None:
            validate_or_raise(Validators.validate_positive_int(value, "max_tokens"))
        elif key == "top_p" and value is not None:
            validate_or_raise(Validators.validate_probability(value, "top_p"))
        elif key in {"frequency_penalty", "presence_penalty"} and value is not None:
            validate_or_raise(Validators.validate_number(value, key, -2.0, 2.0))
        elif key == "stop_sequences" and value is not None:
            if not isinstance(value, list):
                raise TypeError("stop_sequences must be a list of strings")
        elif key == "execution_mode" and value is not None:
            resolved_mode = resolve_execution_mode(value)
            if resolved_mode is ExecutionMode.PRIVACY:
                # Promote legacy alias to hybrid and enable privacy
                object.__setattr__(self, "privacy_enabled", True)
                object.__setattr__(self, "_privacy_enforced_by_execution_mode", True)
                resolved_mode = ExecutionMode.HYBRID
            value = resolved_mode.value
        elif key == "local_storage_path" and value:
            value = str(Path(value).expanduser().resolve())
        elif key in {
            "minimal_logging",
            "auto_sync",
            "privacy_enabled",
            "strict_metrics_nulls",
        }:
            if key == "privacy_enabled" and getattr(
                self, "_privacy_enforced_by_execution_mode", False
            ):
                value = True
                object.__setattr__(self, "_privacy_enforced_by_execution_mode", False)
            else:
                value = bool(value)
        elif key == "enable_usage_analytics":
            value = bool(value)

        return value

    @classmethod
    def edge_analytics_mode(
        cls,
        storage_path: str | None = None,
        minimal_logging: bool = True,
        auto_sync: bool = False,
        **kwargs,
    ) -> TraigentConfig:
        """Create a configuration for Edge Analytics execution.

        Args:
            storage_path: Custom storage path for local data
            minimal_logging: Whether to use minimal logging
            auto_sync: Whether to auto-sync to cloud when API key is available
            **kwargs: Additional configuration parameters

        Returns:
            TraigentConfig configured for Edge Analytics mode
        """
        return cls(
            execution_mode=ExecutionMode.EDGE_ANALYTICS.value,
            local_storage_path=storage_path,
            minimal_logging=minimal_logging,
            auto_sync=auto_sync,
            **kwargs,
        )

    @classmethod
    def from_environment(cls) -> TraigentConfig:
        """Create configuration from environment variables.

        Checks for:
        - TRAIGENT_EDGE_ANALYTICS_MODE: Sets execution mode to edge_analytics if true
        - TRAIGENT_RESULTS_FOLDER: Sets local storage path
        - TRAIGENT_MINIMAL_LOGGING: Sets minimal logging preference
        - TRAIGENT_AUTO_SYNC: Sets auto-sync preference
        - TRAIGENT_PRIVACY_MODE: Sets privacy mode preference
        - TRAIGENT_STRICT_METRICS_NULLS: Use None instead of 0.0 for missing metrics

        Returns:
            TraigentConfig with environment-based settings
        """
        config = cls()

        # Check for Edge Analytics mode
        if os.getenv("TRAIGENT_EDGE_ANALYTICS_MODE", "").lower() in (
            "true",
            "1",
            "yes",
        ):
            config.execution_mode = ExecutionMode.EDGE_ANALYTICS.value

        # Set storage path from environment
        env_path = os.getenv("TRAIGENT_RESULTS_FOLDER")
        if env_path:
            config.local_storage_path = str(Path(env_path).expanduser().resolve())

        # Set logging preference
        if os.getenv("TRAIGENT_MINIMAL_LOGGING", "").lower() in ("false", "0", "no"):
            config.minimal_logging = False

        # Set auto-sync preference
        if os.getenv("TRAIGENT_AUTO_SYNC", "").lower() in ("true", "1", "yes"):
            config.auto_sync = True

        # Privacy mode toggle
        if os.getenv("TRAIGENT_PRIVACY_MODE", "").lower() in ("true", "1", "yes"):
            config.privacy_enabled = True

        # Strict metrics nulls toggle
        if os.getenv("TRAIGENT_STRICT_METRICS_NULLS", "").lower() in (
            "true",
            "1",
            "yes",
        ):
            config.strict_metrics_nulls = True

        return config
