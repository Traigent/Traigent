"""Utility helpers for Optuna-based optimizers.

This module centralises shared functionality for converting TraiGent configuration
spaces into Optuna distributions and for deriving optimisation directions. The
helpers are intentionally lightweight so they can be imported from both runtime
code and tests without introducing additional dependencies.
"""

# Traceability: CONC-Layer-Core CONC-Quality-Compatibility CONC-Quality-Maintainability FUNC-OPT-ALGORITHMS REQ-OPT-ALG-004 SYNC-OptimizationFlow

from __future__ import annotations

from typing import Any, Iterable

try:
    import optuna
    from optuna.distributions import (
        BaseDistribution,
        CategoricalDistribution,
        FloatDistribution,
        IntDistribution,
    )
except ImportError as exc:  # pragma: no cover - exercised in environments w/o Optuna
    optuna = None
    BaseDistribution = object
    CategoricalDistribution = object
    FloatDistribution = object
    IntDistribution = object
    OPTUNA_IMPORT_ERROR: ImportError | None = exc
else:
    OPTUNA_IMPORT_ERROR = None

from traigent.utils.exceptions import OptimizationError

MINIMIZE_KEYWORDS = {"cost", "latency", "error", "time", "memory", "loss"}


def ensure_optuna_available() -> None:
    """Ensure Optuna is installed before continuing."""

    if optuna is None:  # pragma: no cover - only triggered without Optuna
        raise OptimizationError(
            "Optuna is required for Optuna-based optimizers. Install the "
            "'optuna' package or include the optuna optional requirements."
        ) from OPTUNA_IMPORT_ERROR


def config_space_to_distributions(
    config_space: dict[str, Any],
    *,
    include_fixed: bool = True,
) -> dict[str, BaseDistribution]:
    """Convert a TraiGent configuration space into Optuna distributions."""

    ensure_optuna_available()

    distributions: dict[str, BaseDistribution] = {}

    for param_name, definition in config_space.items():
        if isinstance(definition, dict):
            param_type = (definition.get("type") or "categorical").lower()

            if param_type in {"fixed", "constant"}:
                if include_fixed:
                    distributions[param_name] = CategoricalDistribution(
                        [definition.get("value")]
                    )
                continue

            if param_type in {"categorical", "choice"}:
                choices = definition.get("choices") or definition.get("values")
                if not choices:
                    raise OptimizationError(
                        f"Categorical parameter '{param_name}' requires 'choices'"
                    )
                distributions[param_name] = CategoricalDistribution(list(choices))
                continue

            if param_type in {"int", "integer"}:
                low = definition.get("low")
                high = definition.get("high")
                if low is None or high is None:
                    raise OptimizationError(
                        f"Integer parameter '{param_name}' requires 'low' and 'high'"
                    )
                step = definition.get("step")
                log = bool(definition.get("log", False))
                if step:
                    distributions[param_name] = IntDistribution(
                        low=int(low), high=int(high), step=int(step), log=log
                    )
                else:
                    distributions[param_name] = IntDistribution(
                        low=int(low), high=int(high), log=log
                    )
                continue

            if param_type in {"float", "double", "loguniform", "qloguniform"}:
                low = definition.get("low")
                high = definition.get("high")
                if low is None or high is None:
                    raise OptimizationError(
                        f"Float parameter '{param_name}' requires 'low' and 'high'"
                    )
                step = definition.get("step")
                log = bool(definition.get("log", False)) or param_type in {
                    "loguniform",
                    "qloguniform",
                }
                if step:
                    distributions[param_name] = FloatDistribution(
                        low=float(low), high=float(high), step=float(step), log=log
                    )
                else:
                    distributions[param_name] = FloatDistribution(
                        low=float(low), high=float(high), log=log
                    )
                continue

            continue

        if isinstance(definition, list):
            distributions[param_name] = CategoricalDistribution(definition)
        elif isinstance(definition, tuple) and len(definition) == 2:
            low, high = definition
            if isinstance(low, int) and isinstance(high, int):
                distributions[param_name] = IntDistribution(low=low, high=high)
            else:
                distributions[param_name] = FloatDistribution(
                    low=float(low), high=float(high)
                )
        elif optuna and isinstance(definition, BaseDistribution):
            distributions[param_name] = definition
        elif include_fixed:
            distributions[param_name] = CategoricalDistribution([definition])

    return distributions


def infer_directions(objectives: Iterable[str]) -> list[str]:
    """Infer optimisation directions (minimize/maximize) from objective names."""

    directions: list[str] = []
    for name in objectives:
        if name is None:
            directions.append("maximize")
            continue

        normalised = name.lower()
        if normalised in MINIMIZE_KEYWORDS or normalised.startswith("min_"):
            directions.append("minimize")
        else:
            directions.append("maximize")

    return directions or ["maximize"]


def discretize_for_grid(
    config_space: dict[str, Any],
    *,
    n_bins: int = 10,
) -> dict[str, list[Any]]:
    """Discretize continuous configuration ranges for Optuna's GridSampler."""

    if n_bins < 2:
        raise OptimizationError("n_bins must be at least 2 for grid discretisation")

    discrete_space: dict[str, list[Any]] = {}
    for param, definition in config_space.items():
        if isinstance(definition, tuple) and len(definition) == 2:
            low, high = definition
            if isinstance(low, int) and isinstance(high, int):
                span = high - low
                if span <= 100:
                    discrete_space[param] = list(range(low, high + 1))
                else:
                    step = max(span // (n_bins - 1), 1)
                    discrete_space[param] = [low + idx * step for idx in range(n_bins)]
            else:
                interval = (float(high) - float(low)) / (n_bins - 1)
                discrete_space[param] = [
                    float(low) + idx * interval for idx in range(n_bins)
                ]
        elif isinstance(definition, list):
            discrete_space[param] = list(definition)
        else:
            discrete_space[param] = [definition]

    return discrete_space


def suggest_from_definition(
    trial: optuna.trial.Trial,
    name: str,
    definition: Any,
    current_config: dict[str, Any],
) -> Any:
    """Suggest a parameter value based on a flexible definition.

    Supports conditional parameters and richer metadata-based definitions.
    Returns ``None`` when the parameter should be omitted due to unmet
    conditions.
    """

    if isinstance(definition, dict):
        conditions = definition.get("conditions")
        if conditions:
            for key, expected in conditions.items():
                if current_config.get(key) != expected:
                    return (
                        definition.get("default") if "default" in definition else None
                    )

        param_type = (definition.get("type") or "categorical").lower()

        if param_type in {"fixed", "constant"}:
            return definition.get("value")

        if param_type in {"categorical", "choice"}:
            choices = definition.get("choices") or definition.get("values")
            if not choices:
                raise OptimizationError(
                    f"Categorical parameter '{name}' requires 'choices'"
                )
            return trial.suggest_categorical(name, list(choices))

        if param_type in {"int", "integer"}:
            low = definition.get("low")
            high = definition.get("high")
            if low is None or high is None:
                raise OptimizationError(
                    f"Integer parameter '{name}' requires 'low' and 'high'"
                )
            step = definition.get("step")
            log = bool(definition.get("log", False))
            if step:
                return trial.suggest_int(
                    name, int(low), int(high), step=int(step), log=log
                )
            return trial.suggest_int(name, int(low), int(high), log=log)

        if param_type in {"float", "double"}:
            low = definition.get("low")
            high = definition.get("high")
            if low is None or high is None:
                raise OptimizationError(
                    f"Float parameter '{name}' requires 'low' and 'high'"
                )
            step = definition.get("step")
            log = bool(definition.get("log", False))
            low_f = float(low)
            high_f = float(high)
            if step:
                return trial.suggest_float(
                    name, low_f, high_f, step=float(step), log=log
                )
            return trial.suggest_float(name, low_f, high_f, log=log)

        if param_type in {"qloguniform", "loguniform"}:
            low = definition.get("low")
            high = definition.get("high")
            if low is None or high is None:
                raise OptimizationError(
                    f"Log-uniform parameter '{name}' requires 'low' and 'high'"
                )
            return trial.suggest_float(name, float(low), float(high), log=True)

        raise OptimizationError(
            f"Unsupported parameter type '{param_type}' for '{name}'"
        )

    if isinstance(definition, list):
        return trial.suggest_categorical(name, definition)

    if isinstance(definition, tuple) and len(definition) == 2:
        low, high = definition
        if isinstance(low, int) and isinstance(high, int):
            return trial.suggest_int(name, low, high)
        return trial.suggest_float(name, float(low), float(high))

    if optuna and isinstance(definition, BaseDistribution):
        return trial._suggest(name, definition)

    return definition


def infer_distribution_from_value(name: str, value: Any) -> BaseDistribution:
    """Infer a distribution compatible with Optuna from a concrete value."""

    ensure_optuna_available()

    if isinstance(value, bool):
        return CategoricalDistribution([value])
    if isinstance(value, int):
        return IntDistribution(low=value, high=value)
    if isinstance(value, float):
        return FloatDistribution(low=value, high=value)
    return CategoricalDistribution([value])


def generate_optuna_visualizations(study: optuna.study.Study) -> dict[str, Any]:
    """Generate visualizations for an Optuna study."""

    ensure_optuna_available()

    try:
        from optuna.visualization import (
            plot_contour,
            plot_optimization_history,
            plot_parallel_coordinate,
            plot_param_importances,
        )
    except Exception as exc:  # pragma: no cover - visualization optional
        raise OptimizationError(
            f"Optuna visualization dependencies missing: {exc}"
        ) from exc

    figures: dict[str, Any] = {}
    if study.trials:
        figures["history"] = plot_optimization_history(study)
        figures["importance"] = plot_param_importances(study)
        figures["parallel"] = plot_parallel_coordinate(study)
        figures["contour"] = plot_contour(study)

    return figures
