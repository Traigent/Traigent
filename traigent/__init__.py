"""Traigent SDK - Open-source LLM optimization toolkit.

Traigent makes it effortless to optimize your LLM applications with a simple decorator.

Example:
    >>> import traigent
    >>> from traigent import Range, IntRange, Choices, LogRange
    >>>
    >>> # New SE-friendly syntax with Range/Choices classes
    >>> @traigent.optimize(
    ...     eval_dataset="evals.jsonl",
    ...     objectives=["accuracy", "cost"],
    ...     temperature=Range(0.0, 2.0),
    ...     max_tokens=IntRange(100, 4096),
    ...     model=Choices(["gpt-4o-mini", "gpt-4o"]),
    ... )
    ... def my_function(query: str) -> str:
    ...     return process_query(query)
    >>>
    >>> # Legacy syntax still works
    >>> @traigent.optimize(
    ...     configuration_space={
    ...         "model": ["gpt-4o-mini", "gpt-4o"],
    ...         "temperature": (0.0, 1.0)
    ...     }
    ... )
    ... def my_function(query: str) -> str:
    ...     return process_query(query)
    >>>
    >>> results = my_function.optimize()
    >>> best_config = my_function.get_best_config()
"""

# Traceability: CONC-Layer-API CONC-Quality-Usability CONC-Quality-Maintainability FUNC-API-ENTRY REQ-API-001 SYNC-OptimizationFlow

from __future__ import annotations

import builtins
import sys

from traigent._version import get_version

__version__ = get_version()
__author__ = "Traigent Team"
__email__ = "opensource@traigent.ai"

from traigent.api.config_space import ConfigSpace

# TVL constraint system
from traigent.api.constraints import (
    AndCondition,
    BoolExpr,
    Condition,
    Constraint,
    NotCondition,
    OrCondition,
    WhenBuilder,
    constraints_to_callables,
    implies,
    normalize_constraints,
    require,
    when,
)

# Public API exports
from traigent.api.decorators import optimize
from traigent.api.functions import (
    configure,
    get_available_strategies,
    get_config,
    get_current_config,
    get_optimization_insights,
    get_trial_config,
    get_version_info,
    initialize,
    override_config,
    set_strategy,
)

# SE-friendly parameter range classes
from traigent.api.parameter_ranges import (
    Choices,
    IntRange,
    LogRange,
    ParameterRange,
    Range,
)

# Core types
from traigent.api.types import (
    ConfigurationComparison,
    OptimizationResult,
    OptimizationStatus,
    ParetoFront,
    SensitivityAnalysis,
    StrategyConfig,
    TrialResult,
)
from traigent.api.validation_protocol import (
    ConstraintValidator,
    ConstraintViolation,
    PythonConstraintValidator,
    SatResult,
    SatStatus,
)
from traigent.api.validation_protocol import (
    ValidationResult as ConstraintValidationResult,
)

# Thread context helpers
from traigent.config.context import copy_context_to_thread

# Configuration types
from traigent.config.types import TraigentConfig

# Lifecycle and state management
from traigent.core.optimized_function import OptimizationState
from traigent.utils.callbacks import (
    LoggingCallback,
    ProgressBarCallback,
    StatisticsCallback,
    get_default_callbacks,
    get_verbose_callbacks,
)
from traigent.utils.constraints import (
    ConstraintManager,
    max_tokens_constraint,
    model_cost_constraint,
    temperature_constraint,
)

# Exceptions and warnings
from traigent.utils.exceptions import (
    ConfigAccessWarning,
    OptimizationStateError,
    TraigentDeprecationWarning,
    TraigentWarning,
)
from traigent.utils.importance import ParameterImportanceAnalyzer
from traigent.utils.multi_objective import MultiObjectiveMetrics, ParetoFrontCalculator

# Sprint 2 features
from traigent.utils.persistence import PersistenceManager
from traigent.utils.retry import RetryConfig, retry

# Sprint 3 features
from traigent.utils.validation import (
    OptimizationValidator,
    ValidationResult,
    validate_and_suggest,
)
from traigent.visualization.plots import PlotGenerator, create_quick_plot

__all__ = [
    # Main decorator
    "optimize",
    # SE-friendly parameter range classes
    "Range",
    "IntRange",
    "LogRange",
    "Choices",
    "ParameterRange",
    # TVL constraint system
    "AndCondition",
    "BoolExpr",
    "Condition",
    "Constraint",
    "ConfigSpace",
    "ConstraintValidator",
    "ConstraintViolation",
    "ConstraintValidationResult",
    "NotCondition",
    "OrCondition",
    "PythonConstraintValidator",
    "SatResult",
    "SatStatus",
    "WhenBuilder",
    "constraints_to_callables",
    "implies",
    "normalize_constraints",
    "require",
    "when",
    # Configuration functions
    "configure",
    "initialize",
    "override_config",
    "get_available_strategies",
    "get_config",
    "get_current_config",  # Deprecated: use get_trial_config
    "get_trial_config",  # New: use during optimization trials
    "get_optimization_insights",
    "get_version_info",
    "set_strategy",
    # Configuration types
    "TraigentConfig",
    # Lifecycle and state
    "OptimizationState",
    # Thread context helpers
    "copy_context_to_thread",
    # Exceptions and warnings
    "TraigentWarning",
    "TraigentDeprecationWarning",
    "OptimizationStateError",
    "ConfigAccessWarning",
    # Sprint 2 features
    "PersistenceManager",
    "ProgressBarCallback",
    "LoggingCallback",
    "StatisticsCallback",
    "get_default_callbacks",
    "get_verbose_callbacks",
    "ParetoFrontCalculator",
    "MultiObjectiveMetrics",
    "ParameterImportanceAnalyzer",
    "RetryConfig",
    "retry",
    # Sprint 3 features
    "OptimizationValidator",
    "ValidationResult",
    "validate_and_suggest",
    "ConstraintManager",
    "temperature_constraint",
    "model_cost_constraint",
    "max_tokens_constraint",
    "PlotGenerator",
    "create_quick_plot",
    # Result types
    "OptimizationResult",
    "TrialResult",
    "SensitivityAnalysis",
    "ConfigurationComparison",
    "ParetoFront",
    "OptimizationStatus",
    "StrategyConfig",
]

# Legacy compatibility: expose the package via the builtins namespace.
# DEPRECATED: This pollutes the global namespace and will be removed in v1.0.0.
# Tests should use explicit imports: `import traigent`
import os
import warnings

if os.environ.get("TRAIGENT_ENABLE_BUILTINS_COMPAT", "").lower() in ("1", "true"):
    # Only enable if explicitly requested (for legacy test compatibility)
    builtins.traigent = sys.modules[__name__]  # type: ignore[attr-defined]
elif os.environ.get("PYTEST_CURRENT_TEST"):
    # Auto-enable in pytest but warn about deprecation
    warnings.warn(
        "traigent is injected into builtins for test compatibility. "
        "This is deprecated and will be removed in v1.0.0. "
        "Use explicit 'import traigent' instead. "
        "Set TRAIGENT_ENABLE_BUILTINS_COMPAT=1 to suppress this warning.",
        DeprecationWarning,
        stacklevel=1,
    )
    builtins.traigent = sys.modules[__name__]  # type: ignore[attr-defined]
