"""Adapter utilities that expose Optuna through the TraiGent API surface."""

# Traceability: CONC-Layer-Core CONC-Quality-Compatibility CONC-Quality-Maintainability FUNC-OPT-ALGORITHMS REQ-OPT-ALG-004 SYNC-OptimizationFlow

from __future__ import annotations

from typing import Any, Callable, Iterable

from traigent.optimizers.optuna_utils import (
    config_space_to_distributions,
    discretize_for_grid,
    ensure_optuna_available,
    infer_directions,
)
from traigent.utils.exceptions import OptimizationError

try:  # pragma: no cover - only taken without Optuna installed
    import optuna
except ImportError as exc:  # pragma: no cover
    optuna = None
    OPTUNA_IMPORT_ERROR: ImportError | None = exc
else:
    OPTUNA_IMPORT_ERROR = None


def _require_optuna() -> None:
    if optuna is None:  # pragma: no cover
        raise OptimizationError(
            "Optuna is required for the OptunaAdapter"
        ) from OPTUNA_IMPORT_ERROR


class OptunaAdapter:
    """Expose Optuna's optimisation loop in a TraiGent-friendly way."""

    @staticmethod
    def optimize(
        func: Callable[..., Any],
        config_space: dict[str, Any],
        objectives: list[str],
        *,
        algorithm: str = "tpe",
        n_trials: int = 100,
        **study_kwargs: Any,
    ) -> dict[str, Any]:
        _require_optuna()

        search_space = config_space_to_distributions(config_space)
        sampler = OptunaAdapter._create_sampler(algorithm, config_space)
        directions = infer_directions(objectives)

        study = optuna.create_study(
            sampler=sampler,
            directions=directions,
            **study_kwargs,
        )

        def objective(trial: optuna.trial.Trial):
            params = {}
            for name, distribution in search_space.items():
                if isinstance(
                    distribution, optuna.distributions.CategoricalDistribution
                ):
                    params[name] = trial.suggest_categorical(name, distribution.choices)
                elif isinstance(distribution, optuna.distributions.FloatDistribution):
                    params[name] = trial.suggest_float(
                        name, distribution.low, distribution.high
                    )
                elif isinstance(distribution, optuna.distributions.IntDistribution):
                    params[name] = trial.suggest_int(
                        name, distribution.low, distribution.high
                    )
                else:  # pragma: no cover - defensive
                    raise OptimizationError(
                        f"Unsupported distribution type {type(distribution)} for {name}"
                    )

            result = func(**params)
            return OptunaAdapter._extract_objectives(result, objectives)

        study.optimize(objective, n_trials=n_trials)

        return OptunaAdapter._study_to_result(study)

    @staticmethod
    def _create_sampler(algorithm: str, config_space: dict[str, Any]):
        ensure_optuna_available()

        algo = algorithm.lower()
        if algo == "tpe":
            return optuna.samplers.TPESampler()
        if algo == "random":
            return optuna.samplers.RandomSampler()
        if algo == "grid":
            discrete = discretize_for_grid(config_space)
            return optuna.samplers.GridSampler(search_space=discrete)
        if algo == "cmaes":
            return optuna.samplers.CmaEsSampler()
        if algo in {"nsga2", "nsga-ii"}:
            return optuna.samplers.NSGAIISampler()

        raise OptimizationError(f"Unknown Optuna algorithm '{algorithm}'")

    @staticmethod
    def _extract_objectives(result: Any, objectives: Iterable[str]) -> Any:
        objective_list = list(objectives)
        if isinstance(result, dict):
            ordered = []
            for name in objective_list:
                if name not in result:
                    raise OptimizationError(
                        f"Objective '{name}' missing from result payload {result}"
                    )
                ordered.append(result[name])
            return ordered if len(ordered) > 1 else ordered[0]

        if isinstance(result, (list, tuple)):
            if len(result) != len(objective_list):
                raise OptimizationError(
                    "Result length does not match number of objectives"
                )
            return list(result) if len(result) > 1 else result[0]

        if len(objective_list) != 1:
            raise OptimizationError(
                "Multi-objective optimisation requires iterable return"
            )

        return result

    @staticmethod
    def _study_to_result(study: optuna.study.Study) -> dict[str, Any]:
        if not study.trials:
            return {
                "best_params": {},
                "best_values": [],
                "n_trials": 0,
                "trials": [],
            }

        try:
            best_params = study.best_params
            best_values = (
                study.best_value
                if hasattr(study, "best_value")
                else study.best_trials[0].values
            )
        except ValueError:
            best_params = study.best_trials[0].params
            best_values = study.best_trials[0].values

        history = []
        for trial in study.trials:
            history.append(
                {
                    "number": trial.number,
                    "params": trial.params,
                    "values": trial.values,
                    "state": trial.state.name,
                }
            )

        return {
            "study_name": study.study_name,
            "best_params": best_params,
            "best_values": best_values,
            "n_trials": len(study.trials),
            "trials": history,
        }
