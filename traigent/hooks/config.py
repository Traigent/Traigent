"""Configuration loading for Traigent hooks.

Loads and validates traigent.yml configuration files that define
agent validation constraints.
"""

# Traceability: CONC-Layer-Config CONC-Quality-Reliability CONC-Quality-Maintainability FUNC-CONFIG-LOAD REQ-CONFIG-001

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from traigent.utils.logging import get_logger
from traigent.utils.secure_path import (
    safe_open,
    safe_write_text,
    validate_path,
)

logger = get_logger(__name__)

# Default configuration file names to search for
CONFIG_FILE_NAMES = ["traigent.yml", "traigent.yaml", ".traigent.yml", ".traigent.yaml"]


_KNOWN_KEYS: dict[str, set[str]] = {
    "": {"validation", "constraints"},
    "validation": {"enabled", "fail_on_warning", "skip_patterns", "hooks"},
    "constraints": {
        "cost",
        "performance",
        "models",
        "max_cost_per_query",
        "max_monthly_budget",
        "min_accuracy",
        "max_latency_ms",
        "allowed_models",
        "blocked_models",
    },
    "constraints.cost": {
        "max_cost_per_query",
        "max_monthly_budget",
        "warn_threshold_pct",
    },
    "constraints.performance": {"min_accuracy", "max_latency_ms", "min_success_rate"},
    "constraints.models": {"allowed", "blocked", "blocked_reasons"},
}


def find_unknown_config_keys(data: Any) -> list[tuple[str, str | None]]:
    """Return (dotted key path, did-you-mean path or None) for unrecognised keys.

    Only key names are inspected; values are never read into the result.
    """
    import difflib

    found: list[tuple[str, str | None]] = []

    def walk(node: Any, prefix: str) -> None:
        if not isinstance(node, dict):
            return
        known = _KNOWN_KEYS[prefix]
        for key in node:
            name = str(key)
            path = f"{prefix}.{name}" if prefix else name
            if name not in known:
                near = difflib.get_close_matches(name, sorted(known), n=1, cutoff=0.7)
                hint = (f"{prefix}.{near[0]}" if prefix else near[0]) if near else None
                found.append((path, hint))
            elif path in _KNOWN_KEYS:
                walk(node[key], path)

    walk(data, "")
    return found


def _first_set(*values: Any) -> Any:
    """Return the first value that is not None (0 counts as set)."""
    for value in values:
        if value is not None:
            return value
    return None


@dataclass
class CostConstraints:
    """Cost-related constraints for agent configurations."""

    max_cost_per_query: float | None = None  # Maximum $ per query
    max_monthly_budget: float | None = None  # Maximum $/month
    warn_threshold_pct: float = 0.8  # Warn at this % of budget


@dataclass
class PerformanceConstraints:
    """Performance-related constraints for agent configurations."""

    min_accuracy: float | None = None  # Minimum accuracy threshold (0.0-1.0)
    max_latency_ms: int | None = None  # Maximum response latency in ms
    min_success_rate: float | None = None  # Minimum success rate (0.0-1.0)


@dataclass
class ModelConstraints:
    """Model-related constraints for agent configurations."""

    allowed_models: list[str] = field(
        default_factory=list
    )  # Whitelist of allowed models
    blocked_models: list[str] = field(
        default_factory=list
    )  # Blacklist of blocked models
    blocked_reasons: dict[str, str] = field(
        default_factory=dict
    )  # Reasons for blocking


@dataclass
class HooksConstraints:
    """All constraint types combined."""

    cost: CostConstraints = field(default_factory=CostConstraints)
    performance: PerformanceConstraints = field(default_factory=PerformanceConstraints)
    models: ModelConstraints = field(default_factory=ModelConstraints)


@dataclass
class HooksConfig:
    """Configuration for Traigent hooks loaded from traigent.yml."""

    # Validation settings
    enabled: bool = True
    fail_on_warning: bool = False
    skip_patterns: list[str] = field(default_factory=list)  # File patterns to skip

    # Hooks to run
    pre_push_hooks: list[str] = field(default_factory=lambda: ["traigent-validate"])
    pre_commit_hooks: list[str] = field(default_factory=list)

    # Constraints
    constraints: HooksConstraints = field(default_factory=HooksConstraints)

    # Raw config for extension
    raw_config: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HooksConfig:
        """Create HooksConfig from parsed YAML data.

        Args:
            data: Dictionary from parsed YAML

        Returns:
            HooksConfig instance
        """
        validation_data = data.get("validation", {})
        constraints_data = data.get("constraints", {})

        # Parse hooks configuration
        hooks_data = validation_data.get("hooks", {})
        pre_push = hooks_data.get("pre-push", ["traigent-validate"])
        pre_commit = hooks_data.get("pre-commit", [])

        # Parse constraints
        cost_data = constraints_data.get("cost", {})
        cost_constraints = CostConstraints(
            max_cost_per_query=_first_set(
                cost_data.get("max_cost_per_query"),
                constraints_data.get("max_cost_per_query"),
            ),
            max_monthly_budget=_first_set(
                cost_data.get("max_monthly_budget"),
                constraints_data.get("max_monthly_budget"),
            ),
            warn_threshold_pct=cost_data.get("warn_threshold_pct", 0.8),
        )

        perf_data = constraints_data.get("performance", {})
        performance_constraints = PerformanceConstraints(
            min_accuracy=_first_set(
                perf_data.get("min_accuracy"), constraints_data.get("min_accuracy")
            ),
            max_latency_ms=_first_set(
                perf_data.get("max_latency_ms"),
                constraints_data.get("max_latency_ms"),
            ),
            min_success_rate=perf_data.get("min_success_rate"),
        )

        models_data = constraints_data.get("models", {})
        model_constraints = ModelConstraints(
            allowed_models=models_data.get("allowed")
            or constraints_data.get("allowed_models", []),
            blocked_models=models_data.get("blocked")
            or constraints_data.get("blocked_models", []),
            blocked_reasons=models_data.get("blocked_reasons", {}),
        )

        hooks_constraints = HooksConstraints(
            cost=cost_constraints,
            performance=performance_constraints,
            models=model_constraints,
        )

        return cls(
            enabled=validation_data.get("enabled", True),
            fail_on_warning=validation_data.get("fail_on_warning", False),
            skip_patterns=validation_data.get("skip_patterns", []),
            pre_push_hooks=pre_push if isinstance(pre_push, list) else [pre_push],
            pre_commit_hooks=(
                pre_commit if isinstance(pre_commit, list) else [pre_commit]
            ),
            constraints=hooks_constraints,
            raw_config=data,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary for serialization."""
        return {
            "validation": {
                "enabled": self.enabled,
                "fail_on_warning": self.fail_on_warning,
                "skip_patterns": self.skip_patterns,
                "hooks": {
                    "pre-push": self.pre_push_hooks,
                    "pre-commit": self.pre_commit_hooks,
                },
            },
            "constraints": {
                "max_cost_per_query": self.constraints.cost.max_cost_per_query,
                "max_monthly_budget": self.constraints.cost.max_monthly_budget,
                "min_accuracy": self.constraints.performance.min_accuracy,
                "max_latency_ms": self.constraints.performance.max_latency_ms,
                "allowed_models": self.constraints.models.allowed_models,
                "blocked_models": self.constraints.models.blocked_models,
            },
        }


def find_config_file(start_path: Path | None = None) -> Path | None:
    """Find traigent.yml configuration file by searching upward from start_path.

    Args:
        start_path: Starting directory (defaults to current directory)

    Returns:
        Path to config file if found, None otherwise
    """
    if start_path is None:
        start_path = Path.cwd()

    current = start_path.resolve()

    # Search up the directory tree
    while True:
        for config_name in CONFIG_FILE_NAMES:
            config_path = current / config_name
            if config_path.exists() and config_path.is_file():
                logger.debug(f"Found config file: {config_path}")
                return config_path

        parent = current.parent
        if parent == current:
            # Reached filesystem root
            break
        current = parent

    logger.debug(f"No config file found starting from {start_path}")
    return None


def load_hooks_config(config_path: Path | str | None = None) -> HooksConfig:
    """Load Traigent hooks configuration from YAML file.

    Args:
        config_path: Path to config file, or None to auto-detect

    Returns:
        HooksConfig instance

    Raises:
        FileNotFoundError: If config_path specified but not found
        yaml.YAMLError: If config file is invalid YAML
    """
    if config_path is None:
        config_path = find_config_file()

    if config_path is None:
        logger.info("No traigent.yml found, using default configuration")
        return HooksConfig()

    config_path = Path(config_path)
    allowed_base = (
        config_path.parent.resolve()
        if config_path.is_absolute()
        else Path.cwd().resolve()
    )
    config_path = validate_path(config_path, allowed_base, must_exist=False)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    logger.info(f"Loading hooks configuration from {config_path}")

    with safe_open(config_path, allowed_base, mode="r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    return HooksConfig.from_dict(data)


def create_default_config(output_path: Path | str | None = None) -> Path:
    """Create a default traigent.yml configuration file.

    Args:
        output_path: Where to create the file (defaults to ./traigent.yml)

    Returns:
        Path to created configuration file
    """
    if output_path is None:
        output_path = Path.cwd() / "traigent.yml"
    else:
        output_path = Path(output_path)

    allowed_base = (
        output_path.parent.resolve()
        if output_path.is_absolute()
        else Path.cwd().resolve()
    )
    output_path = validate_path(output_path, allowed_base, must_exist=False)

    default_config = """# Traigent Agent Configuration Constraints
# This file defines validation rules for agent configurations
# that are enforced by Git hooks before pushes.

validation:
  enabled: true
  fail_on_warning: false

constraints:
  # Cost constraints
  max_cost_per_query: 0.05      # $0.05 max per query

  # Model constraints
  allowed_models:
    - gpt-4o-mini
    - gpt-4o
    - claude-haiku-4-5-20251001
    - claude-sonnet-4-6
  blocked_models:
    - gpt-4-32k                 # Too expensive for production
"""

    safe_write_text(output_path, default_config, allowed_base, encoding="utf-8")

    logger.info(f"Created default configuration at {output_path}")
    return output_path
