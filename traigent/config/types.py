"""Configuration types for Traigent SDK."""

# Traceability: CONC-ConfigInjection CONC-CloudService FUNC-API-ENTRY FUNC-CLOUD-HYBRID REQ-API-001 REQ-INJ-002 REQ-CLOUD-009 SYNC-CloudHybrid CONC-Layer-Core

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import EnumType, StrEnum
from inspect import Parameter, Signature
from pathlib import Path
from typing import Any, Literal, cast

from traigent.utils.validation import Validators, validate_or_raise

_EMITTED_DEPRECATION_WARNINGS: set[str] = set()


def _reset_deprecation_warning_state_for_tests() -> None:
    """Reset once-per-process deprecation guards for isolated tests."""

    _EMITTED_DEPRECATION_WARNINGS.clear()


def _warn_deprecated_once(key: str, message: str, *, stacklevel: int = 3) -> None:
    """Emit one DeprecationWarning per process for a deprecated SDK surface."""

    if key in _EMITTED_DEPRECATION_WARNINGS:
        return
    _EMITTED_DEPRECATION_WARNINGS.add(key)

    import warnings

    warnings.warn(message, DeprecationWarning, stacklevel=stacklevel)


def removed_legacy_execution_mode_message(surface: str = "edge_analytics") -> str:
    """Migration error for the removed ``edge_analytics`` legacy selector.

    ``edge_analytics`` previously normalized to ``local`` with a warning, but
    runs configured through it could silently produce zero trials. Per the
    no-silent-legacy policy it now hard-fails with migration guidance.
    """

    return (
        f"{surface} has been removed and now fails instead of being silently "
        "remapped: legacy edge_analytics runs could silently produce zero "
        "trials. Migrate to the supported execution surface: offline=True with "
        "algorithm='grid' or algorithm='random' for local, zero-egress "
        "optimization, or omit execution_mode and use algorithm='auto' for "
        "managed optimization."
    )


class _ExecutionModeMeta(EnumType):
    """Hard-fail on the removed ``EDGE_ANALYTICS`` member alias."""

    def __getattribute__(cls, name: str) -> Any:
        if name == "EDGE_ANALYTICS":
            raise AttributeError(
                removed_legacy_execution_mode_message("ExecutionMode.EDGE_ANALYTICS")
            )
        return super().__getattribute__(name)

    def __getitem__(cls, name: str) -> Any:
        if name == "EDGE_ANALYTICS":
            raise KeyError(
                removed_legacy_execution_mode_message("ExecutionMode['EDGE_ANALYTICS']")
            )
        return super().__getitem__(name)


class ExecutionMode(StrEnum, metaclass=_ExecutionModeMeta):
    """Deprecated compatibility enum for runtime routing.

    Public SDK code should use ``algorithm`` plus ``offline``. ``LOCAL`` is the
    canonical local-only member. The former ``EDGE_ANALYTICS`` alias has been
    removed and hard-fails with migration guidance.
    """

    LOCAL = "local"
    HYBRID = "hybrid"
    HYBRID_API = "hybrid_api"

    @classmethod
    def _missing_(cls, value: object) -> ExecutionMode | None:
        if isinstance(value, str) and value.strip().lower() == "edge_analytics":
            raise ValueError(
                removed_legacy_execution_mode_message("execution_mode='edge_analytics'")
            )
        return None


class ExecutionIntent(StrEnum):
    """Resolved execution intent for the consolidated SDK surface."""

    CLOUD_BRAIN = "cloud_brain"
    CLOUD_REQUIRED = "cloud_required"
    LOCAL_ONLY = "local_only"


@dataclass(frozen=True)
class ResolvedExecutionPolicy:
    """Cloud/local execution policy resolved from public SDK knobs.

    The execution layer consumes this object. Legacy ``ExecutionMode`` remains
    available as an internal compatibility representation while the public
    selector moves to ``algorithm`` plus ``offline``.
    """

    intent: ExecutionIntent
    offline: bool
    require_cloud: bool
    algorithm: str
    source_hint: str
    legacy_execution_mode: ExecutionMode = ExecutionMode.LOCAL

    @property
    def allows_cloud_fallback(self) -> bool:
        """Whether managed optimization may fall back to local execution."""

        return self.intent is ExecutionIntent.CLOUD_BRAIN and not self.require_cloud

    @property
    def legacy_execution_mode_value(self) -> str:
        """Compatibility mode string for code not yet wired to policy."""

        return self.legacy_execution_mode.value


# Canonical runtime-supported modes and accepted public aliases.
_SUPPORTED_MODES = (
    ExecutionMode.LOCAL,
    ExecutionMode.HYBRID,
    ExecutionMode.HYBRID_API,
)
# No warn-and-remap aliases remain: removed legacy selectors hard-fail instead
# (no-silent-legacy policy; see removed_legacy_execution_mode_message).
_EXECUTION_MODE_ALIASES: dict[str, ExecutionMode] = {}
_DEPRECATED_CONFIG_EXECUTION_MODE_VALUES = frozenset(
    {ExecutionMode.HYBRID.value, ExecutionMode.HYBRID_API.value}
)
_FAIL_CLOSED_LEGACY_EXECUTION_MODES = frozenset({"privacy", "cloud"})
_REMOVED_LEGACY_EXECUTION_MODES = frozenset({"edge_analytics"})


def is_removed_legacy_execution_mode(mode: ExecutionMode | str | None) -> bool:
    """Return whether a raw legacy selector has been removed and must raise."""

    return (
        isinstance(mode, str)
        and mode.strip().lower() in _REMOVED_LEGACY_EXECUTION_MODES
    )
_NOT_YET_SUPPORTED_MODES: set[ExecutionMode] = set()
_TRUTHY_ENV_VALUES = frozenset({"1", "true", "yes", "on"})
_CONFIG_VALUE_UNSET = object()
_PURGED_PUBLIC_CONFIG_KEYS = frozenset(
    {
        "execution_mode",
        "privacy_enabled",
        "cloud_fallback_policy",
        "hybrid_api_endpoint",
    }
)
_LOCAL_ALGORITHMS = frozenset({"grid", "random"})
_SMART_ALGORITHMS = frozenset(
    {
        "bayesian",
        "optuna",
        "tpe",
        "optuna_tpe",
        "optuna_random",
        "optuna_grid",
        "optuna_cmaes",
        "optuna_nsga2",
        "nsga2",
        "cmaes",
        "nsgaii",
        "nsga_ii",
        "cma_es",
    }
)


def _read_bool_env(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in _TRUTHY_ENV_VALUES


def is_traigent_offline_requested() -> bool:
    """Return whether ``TRAIGENT_OFFLINE`` requests zero-egress local mode."""

    return _read_bool_env("TRAIGENT_OFFLINE")


def is_traigent_require_cloud_requested() -> bool:
    """Return whether ``TRAIGENT_REQUIRE_CLOUD`` disables local fallback."""

    return _read_bool_env("TRAIGENT_REQUIRE_CLOUD")


def normalize_algorithm_name(algorithm: str | None) -> str:
    """Normalize public algorithm names for resolution and validation."""

    if algorithm is None:
        return "auto"
    if not isinstance(algorithm, str):
        raise TypeError(f"algorithm must be a string, got {type(algorithm).__name__}")
    normalized = algorithm.strip().lower().replace("-", "_").replace(" ", "_")
    return normalized or "auto"


def is_local_algorithm(algorithm: str | None) -> bool:
    """Return True for local-only algorithms."""

    return normalize_algorithm_name(algorithm) in _LOCAL_ALGORITHMS


def is_smart_algorithm(algorithm: str | None) -> bool:
    """Return True for managed-service optimizer names."""

    normalized = normalize_algorithm_name(algorithm)
    return normalized in _SMART_ALGORITHMS or normalized.startswith("optuna_")


def accepted_algorithm_values() -> tuple[str, ...]:
    """Return known public algorithm names accepted by policy validation."""

    return tuple(sorted({"auto", *_LOCAL_ALGORITHMS, *_SMART_ALGORITHMS}))


def validate_algorithm_name(algorithm: str | None) -> str:
    """Validate and normalize the public algorithm selector."""

    normalized = normalize_algorithm_name(algorithm)
    # Validate strictly against KNOWN names only. is_smart_algorithm() keeps a
    # lenient ``optuna_`` prefix for internal classification/routing, but the
    # public boundary must reject unknown names (e.g. ``optuna_foo``) rather
    # than accept-then-fail later.
    if (
        normalized == "auto"
        or normalized in _LOCAL_ALGORITHMS
        or normalized in _SMART_ALGORITHMS
    ):
        return normalized
    valid = ", ".join(accepted_algorithm_values())
    raise ValueError(f"algorithm must be one of: {valid}; got {algorithm!r}")


def accepted_execution_mode_values() -> tuple[str, ...]:
    """Return deprecated compatibility selector strings for internal validators."""

    values = {mode.value for mode in _SUPPORTED_MODES}
    values.update(_EXECUTION_MODE_ALIASES)
    return tuple(sorted(values))


def _accepted_execution_mode_message() -> str:
    return (
        "use algorithm='auto', 'grid', or 'random'; set offline=True for "
        "local-only, zero-egress optimization"
    )


def _raw_execution_mode_token(mode: ExecutionMode | str) -> str:
    return mode.value if isinstance(mode, ExecutionMode) else str(mode)


def _normalized_execution_mode_token(mode: ExecutionMode | str) -> str:
    return _raw_execution_mode_token(mode).strip().lower()


def is_fail_closed_legacy_execution_mode(mode: ExecutionMode | str | None) -> bool:
    """Return whether a raw legacy selector must fail closed."""

    return isinstance(mode, str) and (
        _normalized_execution_mode_token(mode) in _FAIL_CLOSED_LEGACY_EXECUTION_MODES
    )


def fail_closed_legacy_execution_mode_message(mode: ExecutionMode | str) -> str:
    """Actionable error for legacy selectors that can otherwise hide egress."""

    raw_mode = _raw_execution_mode_token(mode)
    return (
        f"execution_mode={raw_mode!r} is a legacy selector and now fails closed "
        "at the public API boundary because compatibility normalization can route "
        "it to cloud egress. Remove execution_mode and use the supported execution "
        "surface instead: algorithm='auto' for managed/cloud-first optimization; "
        "offline=True, algorithm='grid', or algorithm='random' for local no-egress "
        "optimization."
    )


def _warn_deprecated_execution_mode_alias(mode: str, message: str) -> None:
    _warn_deprecated_once(
        f"execution_mode_alias:{mode.strip().lower() or '<empty>'}",
        f"execution_mode={mode!r} is deprecated. {message} This compatibility "
        "selector will be removed in a future major release.",
        stacklevel=4,
    )


def _warn_deprecated_config_execution_mode(
    mode: ExecutionMode | str,
    *,
    stacklevel: int = 4,
) -> None:
    raw_mode = mode.value if isinstance(mode, ExecutionMode) else str(mode)
    normalized = raw_mode.strip().lower()
    if normalized not in _DEPRECATED_CONFIG_EXECUTION_MODE_VALUES:
        return

    if normalized == "edge_analytics":
        guidance = (
            "Use offline=True with algorithm='grid' or algorithm='random' for "
            "local-only, zero-egress optimization. Prefer local over "
            "edge_analytics where a compatibility wire value is still required."
        )
    elif normalized == ExecutionMode.HYBRID.value:
        guidance = (
            "Omit execution_mode and use algorithm='auto' for managed "
            "optimization; set offline=True only when no egress is required."
        )
    else:
        guidance = (
            "Omit execution_mode and use algorithm='auto', 'grid', or 'random' "
            "with offline=True when no egress is required. Configure an "
            "external-service evaluator bundle separately."
        )

    _warn_deprecated_once(
        f"TraigentConfig.execution_mode:{normalized}",
        f"execution_mode={raw_mode!r} is deprecated for TraigentConfig. "
        f"{guidance} This compatibility selector will be removed in a future "
        "major release.",
        stacklevel=stacklevel,
    )


def _warn_deprecated_config_privacy_enabled(*, stacklevel: int = 4) -> None:
    _warn_deprecated_once(
        "TraigentConfig.privacy_enabled",
        "privacy_enabled is deprecated for TraigentConfig and has no effect in "
        "execution policy resolution. Use offline=True for no-egress local "
        "optimization. This compatibility flag will be removed in a future "
        "major release.",
        stacklevel=stacklevel,
    )


def resolve_execution_mode(
    mode: ExecutionMode | str | None,
    *,
    default: ExecutionMode = ExecutionMode.LOCAL,
) -> ExecutionMode:
    """Normalize legacy execution-mode values into an internal enum.

    New public code should call :func:`resolve_execution_policy` and use the
    returned :class:`ResolvedExecutionPolicy`.
    """
    if mode is None:
        return default
    if isinstance(mode, ExecutionMode):
        return mode
    if isinstance(mode, str):
        normalized = mode.strip().lower()
        if not normalized:
            return default
        if normalized == "standard":
            _warn_deprecated_execution_mode_alias(
                mode,
                "Omit execution_mode and use algorithm='auto' for cloud-first "
                "automatic optimization.",
            )
            return ExecutionMode.HYBRID
        if normalized in _REMOVED_LEGACY_EXECUTION_MODES:
            raise ValueError(
                removed_legacy_execution_mode_message(f"execution_mode={mode!r}")
            )
        if normalized in _FAIL_CLOSED_LEGACY_EXECUTION_MODES:
            raise ValueError(fail_closed_legacy_execution_mode_message(mode))
        if normalized in _EXECUTION_MODE_ALIASES:
            return _EXECUTION_MODE_ALIASES[normalized]
        try:
            return ExecutionMode(normalized)
        except ValueError as exc:  # pragma: no cover - defensive
            raise ValueError(
                f"Unsupported execution selector {mode!r}; "
                f"{_accepted_execution_mode_message()}."
            ) from exc
    raise TypeError(
        "Execution selector must be a string or compatibility enum, "
        f"got {type(mode).__name__}; {_accepted_execution_mode_message()}."
    )


def validate_execution_mode(mode: ExecutionMode | str | None) -> ExecutionMode:
    """Resolve and validate a deprecated compatibility selector."""
    from traigent.utils.exceptions import ConfigurationError

    try:
        resolved = resolve_execution_mode(mode)
    except ValueError as exc:
        if is_fail_closed_legacy_execution_mode(mode) or (
            is_removed_legacy_execution_mode(mode)
        ):
            raise ConfigurationError(str(exc)) from None
        raise ConfigurationError(
            f"Unsupported execution selector {mode!r}; "
            f"{_accepted_execution_mode_message()}."
        ) from None

    if resolved not in _SUPPORTED_MODES:
        raise ConfigurationError(
            f"Unsupported execution selector {resolved.value!r}; "
            f"{_accepted_execution_mode_message()}."
        )

    return resolved


def resolve_execution_policy(
    *,
    algorithm: str | None = "auto",
    offline: bool = False,
    execution_mode: ExecutionMode | str | None = None,
    privacy_enabled: bool | None = None,
    cloud_fallback_policy: str | None = None,
    require_cloud: bool | None = None,
    source_hint: str = "decorator",
) -> ResolvedExecutionPolicy:
    """Resolve public execution controls into a policy object.

    Semantics:
    - ``algorithm='auto'`` with egress allowed uses managed optimization when available.
    - ``grid``/``random`` resolve to local-only mode.
    - smart algorithms require managed optimization.
    - ``offline=True`` or ``TRAIGENT_OFFLINE=1`` forces local-only mode and
      rejects smart algorithms.
    - ``TRAIGENT_REQUIRE_CLOUD=1`` disables managed-to-local fallback.
    """
    from traigent.utils.exceptions import ConfigurationError

    if not isinstance(offline, bool):
        raise TypeError(f"offline must be a bool, got {type(offline).__name__}")

    normalized_algorithm = validate_algorithm_name(algorithm)
    offline_requested = offline or is_traigent_offline_requested()
    require_cloud_requested = (
        is_traigent_require_cloud_requested()
        if require_cloud is None
        else bool(require_cloud)
    )
    hint_parts = [source_hint]

    if execution_mode is not None:
        raw_mode = (
            execution_mode.value
            if isinstance(execution_mode, ExecutionMode)
            else str(execution_mode)
        )
        normalized_mode = raw_mode.strip().lower()
        hint_parts.append(f"legacy_execution_mode={normalized_mode}")

        if normalized_mode in _FAIL_CLOSED_LEGACY_EXECUTION_MODES:
            raise ConfigurationError(
                fail_closed_legacy_execution_mode_message(raw_mode)
            )
        if normalized_mode == "local":
            offline_requested = True
        elif normalized_mode in {"", "edge_analytics"}:
            _warn_deprecated_once(
                f"resolve_execution_policy.execution_mode:{normalized_mode or '<empty>'}",
                f"execution_mode={raw_mode!r} is deprecated. Mapping to "
                "offline=True / LOCAL_ONLY to preserve the legacy no-egress "
                "guarantee. Use offline=True with algorithm='grid' or "
                "algorithm='random' directly; prefer local over edge_analytics "
                "where a compatibility wire value is still required. This "
                "compatibility selector will be removed in a future major release.",
                stacklevel=3,
            )
            offline_requested = True
        elif normalized_mode in {"hybrid", "standard"}:
            _warn_deprecated_once(
                f"resolve_execution_policy.execution_mode:{normalized_mode}",
                f"execution_mode={raw_mode!r} is deprecated. Omit "
                "execution_mode and use algorithm='auto' for cloud-first "
                "automatic optimization. This compatibility selector will be "
                "removed in a future major release.",
                stacklevel=3,
            )
        elif normalized_mode == "hybrid_api":
            _warn_deprecated_once(
                "resolve_execution_policy.execution_mode:hybrid_api",
                "execution_mode='hybrid_api' is deprecated as a public execution "
                "selector. Configure an external-service evaluator bundle. This "
                "compatibility selector will be removed in a future major release.",
                stacklevel=3,
            )
        else:
            raise ConfigurationError(
                f"Unsupported execution selector {raw_mode!r}; use "
                "algorithm='auto', 'grid', or 'random'; set offline=True when "
                "no egress is required."
            )

    if privacy_enabled is not None:
        _warn_deprecated_once(
            "resolve_execution_policy.privacy_enabled",
            "privacy_enabled is deprecated and has no effect in execution policy "
            "resolution. Use offline=True for no egress. This compatibility flag "
            "will be removed in a future major release.",
            stacklevel=3,
        )
        hint_parts.append("legacy_privacy_enabled")

    if cloud_fallback_policy is not None:
        _warn_deprecated_once(
            "resolve_execution_policy.cloud_fallback_policy",
            "cloud_fallback_policy is deprecated and has no effect. Set "
            "TRAIGENT_REQUIRE_CLOUD=1 to disable local fallback. This "
            "compatibility flag will be removed in a future major release.",
            stacklevel=3,
        )
        hint_parts.append("legacy_cloud_fallback_policy")

    if offline_requested and is_smart_algorithm(normalized_algorithm):
        raise ConfigurationError(
            f"algorithm={normalized_algorithm!r} requires managed optimization and "
            "cannot be used with offline=True or TRAIGENT_OFFLINE=1."
        )

    if offline_requested:
        intent = ExecutionIntent.LOCAL_ONLY
    elif normalized_algorithm == "auto":
        intent = ExecutionIntent.CLOUD_BRAIN
    elif normalized_algorithm in _LOCAL_ALGORITHMS:
        intent = ExecutionIntent.LOCAL_ONLY
    else:
        intent = ExecutionIntent.CLOUD_REQUIRED

    legacy_mode = (
        ExecutionMode.LOCAL
        if intent is ExecutionIntent.LOCAL_ONLY
        else ExecutionMode.HYBRID
    )
    if execution_mode == ExecutionMode.HYBRID_API or (
        isinstance(execution_mode, str)
        and execution_mode.strip().lower() == ExecutionMode.HYBRID_API.value
    ):
        legacy_mode = ExecutionMode.HYBRID_API

    return ResolvedExecutionPolicy(
        intent=intent,
        offline=offline_requested,
        require_cloud=require_cloud_requested,
        algorithm=normalized_algorithm,
        source_hint=";".join(hint_parts),
        legacy_execution_mode=legacy_mode,
    )


class InjectionMode(StrEnum):
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
        algorithm: Optimizer selector. Public values are ``auto``, ``grid``, and
            ``random``; smart optimizer names are accepted where supported.
        offline: Force local-only, zero-egress optimization.
        local_storage_path: Path for local result storage.
        auto_sync: Whether local results may sync to the backend/portal when
            credentials are configured.
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

    # Public execution controls.
    algorithm: str = "auto"
    offline: bool = False

    # Deprecated compatibility state retained for existing config dictionaries.
    execution_mode: str | ExecutionMode | None | object = _CONFIG_VALUE_UNSET
    local_storage_path: str | None = None
    minimal_logging: bool = True
    auto_sync: bool = False  # Auto-sync to backend/portal when API key is available
    # Deprecated content-redaction compatibility flag; prefer offline/sync controls.
    privacy_enabled: bool | object = _CONFIG_VALUE_UNSET

    # Metrics configuration
    strict_metrics_nulls: bool = (
        False  # When True, use None instead of 0.0 for missing metrics
    )
    comparability_mode: Literal["legacy", "warn", "strict"] = "warn"

    # Analytics and telemetry settings
    enable_usage_analytics: bool = True  # Send privacy-safe usage stats when backend/portal integration is configured
    analytics_endpoint: str | None = None  # Custom analytics endpoint
    anonymous_user_id: str | None = None  # Anonymous identifier (auto-generated)

    # Execution-policy runtime state. These fields are intentionally omitted
    # from to_dict()/from_dict() so persisted user config does not capture
    # run-specific provenance or no-egress decisions.
    execution_policy: ResolvedExecutionPolicy | None = None
    no_egress: bool = False
    result_source: str | None = None
    fallback_reason: str | None = None
    persistence_status: str | None = None

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

        supplied_execution_mode = getattr(
            self, "_supplied_execution_mode_value", _CONFIG_VALUE_UNSET
        )
        if self.execution_mode is _CONFIG_VALUE_UNSET:
            object.__setattr__(self, "execution_mode", ExecutionMode.LOCAL.value)
        elif (
            supplied_execution_mode is not None
            and supplied_execution_mode is not _CONFIG_VALUE_UNSET
        ):
            _warn_deprecated_config_execution_mode(
                cast(ExecutionMode | str, supplied_execution_mode)
            )

        if self.privacy_enabled is _CONFIG_VALUE_UNSET:
            object.__setattr__(self, "privacy_enabled", False)
        if getattr(self, "_supplied_privacy_enabled", False):
            _warn_deprecated_config_privacy_enabled()

        # Validate deprecated compatibility selector against supported runtime modes.
        mode_enum = validate_execution_mode(
            cast(ExecutionMode | str | None, self.execution_mode)
        )
        object.__setattr__(self, "execution_mode", cast(str, mode_enum.value))
        object.__setattr__(self, "_warn_legacy_config_assignments", True)

        # Handle local storage path
        if self.is_local_mode() and self.local_storage_path:
            # Validate and expand path
            self.local_storage_path = str(
                Path(self.local_storage_path).expanduser().resolve()
            )

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary.

        Returns:
            Dictionary representation of configuration, excluding None values and defaults.
        """
        return self._to_dict(include_legacy=False)

    def _to_dict(self, *, include_legacy: bool) -> dict[str, Any]:
        result = {}

        # Define default values to exclude
        defaults = {
            "algorithm": "auto",
            "offline": False,
            "minimal_logging": True,
            "auto_sync": False,
            "strict_metrics_nulls": False,
            "comparability_mode": "warn",
        }
        if include_legacy:
            defaults.update(
                {
                    "execution_mode": "local",
                    "privacy_enabled": False,
                }
            )

        # Add defined parameters (only if not None and not default)
        field_names = [
            "model",
            "temperature",
            "max_tokens",
            "top_p",
            "frequency_penalty",
            "presence_penalty",
            "stop_sequences",
            "seed",
            "algorithm",
            "offline",
            "local_storage_path",
            "minimal_logging",
            "auto_sync",
            "strict_metrics_nulls",
            "comparability_mode",
        ]
        if include_legacy:
            field_names.extend(["execution_mode", "privacy_enabled"])

        for field_name in field_names:
            value = getattr(self, field_name)
            if value is not None and value != defaults.get(field_name):
                result[field_name] = value

        # Add custom parameters
        for key, value in self.custom_params.items():
            if include_legacy or key not in _PURGED_PUBLIC_CONFIG_KEYS:
                result[key] = value

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
            "algorithm",
            "offline",
            "execution_mode",
            "local_storage_path",
            "minimal_logging",
            "auto_sync",
            "privacy_enabled",
            "strict_metrics_nulls",
            "comparability_mode",
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
        explicit_override_keys: set[str] = set()
        if isinstance(other, dict):
            explicit_override_keys = set(other)
            other = self.from_dict(other)

        # Start with current values.
        merged_dict = self._to_dict(include_legacy=True)

        # Override with non-default values from the other config. For dict
        # inputs, preserve explicitly supplied defaults such as
        # {"execution_mode": "local"} so callers can intentionally
        # reset a value without default leakage from TraigentConfig().
        override_dict = other._to_dict(include_legacy=True)
        dataclass_fields = getattr(self, "__dataclass_fields__", {})
        for key in explicit_override_keys:
            if key in dataclass_fields and key != "custom_params":
                value = getattr(other, key)
                if value is not None:
                    override_dict[key] = value

        merged_dict.update(override_dict)

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
        """Return internal compatibility selector as an enum."""
        return resolve_execution_mode(
            cast(ExecutionMode | str | None, self.execution_mode)
        )

    def is_local_mode(self) -> bool:
        """Return whether the compatibility selector maps to local-only runtime."""
        return self.execution_mode_enum is ExecutionMode.LOCAL

    def is_edge_analytics_mode(self) -> bool:
        """Deprecated alias for :meth:`is_local_mode`."""
        _warn_deprecated_once(
            "TraigentConfig.is_edge_analytics_mode",
            "TraigentConfig.is_edge_analytics_mode() is deprecated. Use "
            "TraigentConfig.is_local_mode().",
            stacklevel=2,
        )
        return self.is_local_mode()

    def is_cloud_mode(self) -> bool:
        return False

    def is_privacy_enabled(self) -> bool:
        """Whether deprecated content-redaction compatibility is enabled."""
        return bool(self.privacy_enabled)

    def is_strict_metrics_nulls_enabled(self) -> bool:
        """Whether strict metrics nulls is enabled (use None instead of 0.0 for missing metrics)."""
        return bool(self.strict_metrics_nulls)

    def get_comparability_mode(self) -> Literal["legacy", "warn", "strict"]:
        """Return metric comparability handling mode."""
        mode = str(self.comparability_mode or "warn").strip().lower()
        if mode in {"legacy", "warn", "strict"}:
            return mode  # type: ignore[return-value]
        return "warn"

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

        # Default to ~/.traigent/ for local-only storage.
        if self.is_local_mode():
            return str(Path.home() / ".traigent")

        return None

    def _validated_assignment(self, key: str, value: Any) -> Any:
        """Apply per-field validation when assigning configuration values."""
        warn_legacy_assignment = getattr(self, "_warn_legacy_config_assignments", False)
        if key == "execution_mode":
            if not warn_legacy_assignment and value is not _CONFIG_VALUE_UNSET:
                object.__setattr__(self, "_supplied_execution_mode_value", value)
            elif warn_legacy_assignment and value is not None:
                _warn_deprecated_config_execution_mode(value)
        elif (
            key == "privacy_enabled"
            and not warn_legacy_assignment
            and value is not _CONFIG_VALUE_UNSET
        ):
            object.__setattr__(self, "_supplied_privacy_enabled", True)

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
        elif key == "algorithm":
            value = validate_algorithm_name(value)
        elif key == "offline":
            if not isinstance(value, bool):
                raise TypeError(f"offline must be a bool, got {type(value).__name__}")
        elif key == "stop_sequences" and value is not None:
            if not isinstance(value, list):
                raise TypeError("stop_sequences must be a list of strings")
        elif (
            key == "execution_mode"
            and value is not None
            and value is not _CONFIG_VALUE_UNSET
        ):
            resolved_mode = validate_execution_mode(value)
            value = resolved_mode.value
        elif key == "local_storage_path" and value:
            value = str(Path(value).expanduser().resolve())
        elif key in {
            "minimal_logging",
            "auto_sync",
            "privacy_enabled",
            "strict_metrics_nulls",
            "no_egress",
        }:
            if key == "privacy_enabled" and getattr(
                self, "_privacy_enforced_by_execution_mode", False
            ):
                value = True
                object.__setattr__(self, "_privacy_enforced_by_execution_mode", False)
            elif value is _CONFIG_VALUE_UNSET:
                return value
            else:
                if key == "privacy_enabled" and warn_legacy_assignment:
                    _warn_deprecated_config_privacy_enabled()
                value = bool(value)
        elif key == "comparability_mode":
            mode = str(value or "warn").strip().lower()
            if mode not in {"legacy", "warn", "strict"}:
                raise ValueError(
                    "comparability_mode must be one of: legacy, warn, strict"
                )
            value = mode
        elif key == "enable_usage_analytics":
            value = bool(value)

        return value

    @classmethod
    def local_mode(
        cls,
        storage_path: str | None = None,
        minimal_logging: bool = True,
        auto_sync: bool = False,
        **kwargs,
    ) -> TraigentConfig:
        """Compatibility constructor for local-only execution."""
        return cls(
            offline=True,
            execution_mode=ExecutionMode.LOCAL.value,
            local_storage_path=storage_path,
            minimal_logging=minimal_logging,
            auto_sync=auto_sync,
            **kwargs,
        )

    @classmethod
    def edge_analytics_mode(
        cls,
        storage_path: str | None = None,
        minimal_logging: bool = True,
        auto_sync: bool = False,
        **kwargs,
    ) -> TraigentConfig:
        """Deprecated compatibility constructor for offline local execution.

        Args:
            storage_path: Custom storage path for local data
            minimal_logging: Whether to use minimal logging
            auto_sync: Whether to auto-sync to backend/portal when an API key is available
            **kwargs: Additional configuration parameters

        Returns:
            TraigentConfig configured for local-only execution
        """
        _warn_deprecated_once(
            "TraigentConfig.edge_analytics_mode",
            "TraigentConfig.edge_analytics_mode() is deprecated. Use "
            "TraigentConfig(offline=True, algorithm='grid') or "
            "TraigentConfig(offline=True, algorithm='random') for local-only "
            "optimization; prefer local over edge_analytics where a "
            "compatibility wire value is still required. This compatibility "
            "constructor will be removed in a future major release.",
            stacklevel=2,
        )
        return cls.local_mode(
            storage_path=storage_path,
            minimal_logging=minimal_logging,
            auto_sync=auto_sync,
            **kwargs,
        )

    @classmethod
    def from_environment(cls) -> TraigentConfig:
        """Create configuration from environment variables.

        Checks for:
        - TRAIGENT_OFFLINE: Sets offline local-only execution if true
        - TRAIGENT_RESULTS_FOLDER: Sets local storage path
        - TRAIGENT_MINIMAL_LOGGING: Sets minimal logging preference
        - TRAIGENT_AUTO_SYNC: Sets auto-sync preference
        - TRAIGENT_EDGE_ANALYTICS_MODE: Deprecated compatibility alias for offline
        - TRAIGENT_PRIVACY_MODE: Deprecated compatibility content-redaction flag
        - TRAIGENT_STRICT_METRICS_NULLS: Use None instead of 0.0 for missing metrics
        - TRAIGENT_COMPARABILITY_MODE: one of legacy|warn|strict

        Returns:
            TraigentConfig with environment-based settings
        """
        config = cls()

        if is_traigent_offline_requested():
            config.offline = True
            object.__setattr__(config, "execution_mode", ExecutionMode.LOCAL.value)

        # Deprecated local-only compatibility env var.
        if os.getenv("TRAIGENT_EDGE_ANALYTICS_MODE", "").lower() in (
            "true",
            "1",
            "yes",
        ):
            _warn_deprecated_once(
                "TRAIGENT_EDGE_ANALYTICS_MODE",
                "TRAIGENT_EDGE_ANALYTICS_MODE is deprecated. Use "
                "TRAIGENT_OFFLINE=1 with algorithm='grid' or algorithm='random' "
                "for local-only, zero-egress optimization; prefer local over "
                "edge_analytics where a compatibility wire value is still "
                "required. This compatibility environment variable will be "
                "removed in a future major release.",
                stacklevel=2,
            )
            config.offline = True
            object.__setattr__(config, "execution_mode", ExecutionMode.LOCAL.value)

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

        # Deprecated content-redaction compatibility toggle.
        if os.getenv("TRAIGENT_PRIVACY_MODE", "").lower() in ("true", "1", "yes"):
            _warn_deprecated_once(
                "TRAIGENT_PRIVACY_MODE",
                "TRAIGENT_PRIVACY_MODE is deprecated. Use offline=True or "
                "TRAIGENT_OFFLINE=1 for no-egress local optimization. This "
                "compatibility environment variable will be removed in a future "
                "major release.",
                stacklevel=2,
            )
            object.__setattr__(config, "privacy_enabled", True)

        # Strict metrics nulls toggle
        if os.getenv("TRAIGENT_STRICT_METRICS_NULLS", "").lower() in (
            "true",
            "1",
            "yes",
        ):
            config.strict_metrics_nulls = True

        mode = os.getenv("TRAIGENT_COMPARABILITY_MODE", "").strip().lower()
        if mode in {"legacy", "warn", "strict"}:
            config.comparability_mode = mode  # type: ignore[assignment]

        return config


def _public_config_param(name: str, default: Any, annotation: Any) -> Parameter:
    return Parameter(
        name,
        kind=Parameter.KEYWORD_ONLY,
        default=default,
        annotation=annotation,
    )


TraigentConfig.__signature__ = Signature(  # type: ignore[attr-defined]
    parameters=[
        _public_config_param("model", None, str | None),
        _public_config_param("temperature", None, float | None),
        _public_config_param("max_tokens", None, int | None),
        _public_config_param("top_p", None, float | None),
        _public_config_param("frequency_penalty", None, float | None),
        _public_config_param("presence_penalty", None, float | None),
        _public_config_param("stop_sequences", None, list[str] | None),
        _public_config_param("seed", None, int | None),
        _public_config_param("algorithm", "auto", str),
        _public_config_param("offline", False, bool),
        _public_config_param("local_storage_path", None, str | None),
        _public_config_param("minimal_logging", True, bool),
        _public_config_param("auto_sync", False, bool),
        _public_config_param("strict_metrics_nulls", False, bool),
        _public_config_param(
            "comparability_mode", "warn", Literal["legacy", "warn", "strict"]
        ),
        _public_config_param("custom_params", None, dict[str, Any] | None),
    ]
)
