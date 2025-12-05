"""Optuna-backed optimizers for TraiGent.

These optimizers follow the existing :class:`BaseOptimizer` interface while using
Optuna samplers under the hood.  They are implemented as additive features—the
original optimizers remain available and unchanged.
"""

# Traceability: CONC-Layer-Core CONC-Quality-Performance CONC-Quality-Reliability FUNC-OPT-ALGORITHMS REQ-OPT-ALG-004 SYNC-OptimizationFlow

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, cast

from traigent.api.types import TrialResult, TrialStatus
from traigent.optimizers.base import BaseOptimizer
from traigent.optimizers.optuna_checkpoint import OptunaCheckpointManager
from traigent.optimizers.optuna_utils import (
    config_space_to_distributions,
    ensure_optuna_available,
    infer_directions,
    infer_distribution_from_value,
    suggest_from_definition,
)
from traigent.optimizers.pruners import CeilingPruner
from traigent.telemetry.optuna_metrics import OptunaMetricsEmitter, sanitize_config
from traigent.utils.exceptions import OptimizationError
from traigent.utils.logging import get_logger

logger = get_logger(__name__)


OPTUNA_IMPORT_ERROR: ImportError | None = None

try:  # pragma: no cover - import guard is exercised only when Optuna missing
    import optuna
    from optuna import trial as optuna_trial
    from optuna.distributions import BaseDistribution
    from optuna.study import Study
    from optuna.trial import FrozenTrial
except ImportError as exc:  # pragma: no cover
    optuna = None
    Study = object
    BaseDistribution = object
    FrozenTrial = object
    optuna_trial = None
    OPTUNA_IMPORT_ERROR = exc


@dataclass
class OptunaTrialPayload:
    """Serializable payload used when exporting trial history."""

    trial_id: str
    config: dict[str, Any]
    metrics: dict[str, float]
    status: TrialStatus


class OptunaBaseOptimizer(BaseOptimizer):
    """Base implementation shared across specific Optuna optimizers."""

    sampler: optuna.samplers.BaseSampler | None = None

    def __init__(
        self,
        config_space: dict[str, Any],
        objectives: list[str],
        *,
        max_trials: int = 100,
        directions: Iterable[str] | None = None,
        storage: str | None = None,
        study_name: str | None = None,
        pruner: optuna.pruners.BasePruner | None = None,
        sampler: optuna.samplers.BaseSampler | None = None,
        checkpoint_manager: OptunaCheckpointManager | None = None,
        metrics_emitter: OptunaMetricsEmitter | None = None,
        mock_mode: bool = False,
        **kwargs: Any,
    ) -> None:
        ensure_optuna_available()

        self.max_trials = max_trials
        self._config_space = config_space
        self._synced_trials: set[str] = set()
        self._active_trials: dict[int, optuna.trial.Trial] = {}
        self._pending_configs: dict[int, dict[str, Any]] = {}
        self._checkpoint_manager = checkpoint_manager
        self._metrics_emitter = metrics_emitter
        self._mock_mode = mock_mode
        self._mock_config = self._build_mock_config() if mock_mode else None

        self._distributions = config_space_to_distributions(config_space)
        inferred = list(directions) if directions else infer_directions(objectives)

        sampler = sampler or self.sampler or optuna.samplers.TPESampler()
        self._sampler = sampler

        if pruner is None:
            if len(inferred) > 1:
                pruner = optuna.pruners.SuccessiveHalvingPruner()
            else:
                pruner = CeilingPruner(
                    min_completed_trials=2,
                    warmup_steps=2,
                    epsilon=1e-6,
                )

        # Create study upfront so suggestions are reproducible across calls.
        self._study = optuna.create_study(
            directions=inferred,
            sampler=sampler,
            pruner=pruner,
            storage=storage,
            study_name=study_name,
            load_if_exists=bool(storage and study_name),
        )

        self._study_directions = inferred

        self._restore_from_checkpoint()

        super().__init__(config_space, objectives, **kwargs)

    @property
    def study(self) -> Study:
        """Expose the underlying Optuna study."""

        return self._study

    @property
    def active_trials(self) -> dict[int, dict[str, Any]]:
        """Return a snapshot of currently active Optuna trials."""

        return {
            trial_id: dict(config) for trial_id, config in self._pending_configs.items()
        }

    def suggest_next_trial(self, history: list[TrialResult]) -> dict[str, Any]:
        """Suggest the next configuration via Optuna's ask/tell interface."""

        if self.trial_count >= self.max_trials:
            raise OptimizationError(
                f"Maximum number of trials reached ({self.max_trials})."
            )

        self._sync_history(history)

        if self._mock_mode and self._mock_config is not None:
            self._study.enqueue_trial(params=self._mock_config.copy())

        trial = self._study.ask()
        config = self._trial_to_config(trial)
        config["_optuna_trial_id"] = trial.number

        self._active_trials[trial.number] = trial
        self._pending_configs[trial.number] = config
        self._trial_count += 1

        logger.debug("Optuna suggested trial %s -> %s", trial.number, config)

        self._emit_event(
            "trial_suggested",
            trial.number,
            {"config": sanitize_config(config)},
        )
        self._persist_checkpoint()

        return config

    def report_intermediate_value(
        self, trial_id: int, step: int, value: float | Iterable[float]
    ) -> bool:
        """Report an intermediate metric to Optuna and return prune decision."""

        trial = self._active_trials.get(trial_id)
        if trial is None:
            logger.warning(
                "Attempted to report intermediate for unknown trial %s", trial_id
            )
            return False

        if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
            value_list = list(value)
        else:
            value_list = [float(value)]

        if len(self.objectives) > 1:
            # Optuna does not currently support reporting intermediate values for
            # multi-objective studies. We log and skip pruning in this case.
            logger.debug(
                "Skipping intermediate report for multi-objective trial=%s", trial_id
            )
            return False

        trial.report(value_list[0], step)
        should_prune = trial.should_prune()
        logger.debug(
            "Optuna intermediate report trial=%s step=%s value=%s prune=%s",
            trial_id,
            step,
            value,
            should_prune,
        )
        self._emit_event(
            "trial_intermediate",
            trial_id,
            {"step": step, "value": value_list[0]},
        )
        self._persist_checkpoint()
        return cast(bool, should_prune)

    def report_trial_result(
        self,
        trial_id: int,
        objectives: Iterable[float] | float,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Report a completed trial back to Optuna."""

        trial = self._active_trials.get(trial_id)
        if trial is None:
            raise OptimizationError(f"Unknown Optuna trial id {trial_id}")

        meta = metadata or {}

        if objectives is None:
            state_override = str(meta.get("state", "")).lower()
            if state_override == "pruned":
                step = int(meta.get("pruned_at_step", meta.get("step", 0)))
                self.report_trial_pruned(trial_id, step=step)
                return
            if state_override in {"failed", "error", "cancelled"}:
                message = meta.get("error") or meta.get("message") or "Trial failed"
                self.report_trial_failure(trial_id, str(message))
                return
            # If we reach here treat as failure with generic reason
            self.report_trial_failure(trial_id, "Trial reported without objectives")
            return

        trial = self._active_trials.pop(trial_id, None)
        if trial is None:  # pragma: no cover - defensive: race with concurrent pop
            raise OptimizationError(f"Unknown Optuna trial id {trial_id}")

        if isinstance(objectives, Iterable) and not isinstance(
            objectives, (str, bytes)
        ):
            values = list(objectives)
        else:
            values = float(objectives)  # type: ignore[assignment]

        logger.debug("Telling Optuna about completion of trial %s", trial_id)

        if isinstance(values, list):
            self._study.tell(trial, values=values)
            metrics = dict(zip(self.objectives, values))
        else:
            self._study.tell(trial, values)
            metrics = {self.objectives[0]: values}

        config = self._pending_configs.pop(trial_id, {}).copy()
        config.pop("_optuna_trial_id", None)

        trial_result = TrialResult(
            trial_id=str(trial_id),
            config=config,
            metrics=metrics,
            status=TrialStatus.COMPLETED,
            duration=0.0,
            timestamp=datetime.now(UTC),
            metadata=metadata or {},
        )
        self.update_best(trial_result)

        self._emit_event(
            "trial_completed",
            trial_id,
            {"metrics": metrics, "config": sanitize_config(config)},
        )
        self._persist_checkpoint()
        self._synced_trials.add(str(trial_id))

    def report_trial_failure(self, trial_id: int, error_message: str) -> None:
        """Mark an Optuna trial as failed."""

        trial = self._active_trials.pop(trial_id, None)
        if trial is None:
            logger.warning("Failed trial %s not tracked; ignoring", trial_id)
            return

        logger.warning("Marking trial %s as failed: %s", trial_id, error_message)
        self._study.tell(
            trial,
            values=None,
            state=optuna.trial.TrialState.FAIL,
        )
        config = self._pending_configs.pop(trial_id, {})

        self._emit_event(
            "trial_failed",
            trial_id,
            {"error": error_message, "config": sanitize_config(config)},
        )
        self._persist_checkpoint()
        self._synced_trials.add(str(trial_id))

    def report_trial_pruned(self, trial_id: int, step: int) -> None:
        """Mark a trial as pruned and remove it from active tracking."""

        trial = self._active_trials.pop(trial_id, None)
        if trial is None:
            logger.warning("Attempted to prune unknown trial %s", trial_id)
            return

        logger.info("Pruning trial %s at step %s", trial_id, step)
        self._study.tell(
            trial,
            values=None,
            state=optuna.trial.TrialState.PRUNED,
        )
        config = self._pending_configs.pop(trial_id, {})

        self._emit_event(
            "trial_pruned",
            trial_id,
            {"step": step, "config": sanitize_config(config)},
        )
        self._persist_checkpoint()
        self._synced_trials.add(str(trial_id))

    def should_stop(self, history: list[TrialResult]) -> bool:
        """Stop when the configured maximum number of trials is reached."""

        return self.trial_count >= self.max_trials

    # ------------------------------------------------------------------
    # Internal utilities
    # ------------------------------------------------------------------
    def _sync_history(self, history: list[TrialResult]) -> None:
        """Synchronise external history with the current Optuna study."""

        for entry in history:
            normalized = self._normalise_history_entry(entry)
            if normalized is None:
                continue

            trial_id = normalized["trial_id"]
            trial_id_key = str(trial_id)
            if trial_id_key in self._synced_trials:
                continue

            status = normalized["status"]
            if status not in {
                TrialStatus.COMPLETED,
                TrialStatus.FAILED,
                TrialStatus.CANCELLED,
            }:
                continue

            distributions: dict[str, BaseDistribution] = {}
            config = normalized["config"]
            for param_name, value in config.items():
                distributions[param_name] = infer_distribution_from_value(
                    param_name, value
                )
            values = None
            state = optuna.trial.TrialState.FAIL

            metrics = normalized["metrics"]
            if status == TrialStatus.COMPLETED:
                ordered = [metrics.get(obj, 0.0) for obj in self.objectives]
                values = ordered if len(ordered) > 1 else ordered[0]
                state = optuna.trial.TrialState.COMPLETE

            create_kwargs: dict[str, Any] = {
                "params": config,
                "distributions": distributions,
                "state": state,
            }
            if isinstance(values, list):
                create_kwargs["values"] = values
            else:
                create_kwargs["value"] = values

            frozen = optuna.trial.create_trial(**create_kwargs)

            self._study.add_trial(frozen)
            self._synced_trials.add(trial_id_key)

    def _normalise_history_entry(
        self, entry: TrialResult | dict[str, Any]
    ) -> dict[str, Any] | None:
        if isinstance(entry, TrialResult):
            return {
                "trial_id": entry.trial_id,
                "config": dict(entry.config),
                "status": entry.status,
                "metrics": dict(entry.metrics),
                "metadata": dict(entry.metadata),
            }

        if not isinstance(entry, dict):
            return None

        config = dict(entry.get("config") or {})
        trial_id = entry.get("trial_id") or config.get("_optuna_trial_id")
        if trial_id is None:
            return None

        raw_status = entry.get("status") or entry.get("state")
        status: TrialStatus
        if isinstance(raw_status, TrialStatus):
            status = raw_status
        elif isinstance(raw_status, str):
            lowered = raw_status.lower()
            try:
                status = TrialStatus(lowered)
            except ValueError:
                status = (
                    TrialStatus.COMPLETED
                    if entry.get("metrics")
                    else TrialStatus.FAILED
                )
        else:
            status = (
                TrialStatus.COMPLETED if entry.get("metrics") else TrialStatus.FAILED
            )

        metrics_obj = entry.get("metrics")
        if metrics_obj is None and "objectives" in entry:
            metrics_obj = entry["objectives"]

        metrics: dict[str, float]
        if isinstance(metrics_obj, dict):
            metrics = {str(k): float(v) for k, v in metrics_obj.items()}
        elif isinstance(metrics_obj, (list, tuple)):
            metrics = {
                objective: float(value)
                for objective, value in zip(self.objectives, metrics_obj)
            }
        elif metrics_obj is None:
            metrics = {}
        else:
            first_objective = self.objectives[0] if self.objectives else "value"
            metrics = {first_objective: float(metrics_obj)}

        config.pop("_optuna_trial_id", None)

        return {
            "trial_id": str(trial_id),
            "config": config,
            "status": status,
            "metrics": metrics,
            "metadata": dict(entry.get("metadata") or {}),
        }

    def _trial_to_config(self, trial: optuna.trial.Trial) -> dict[str, Any]:
        """Convert an Optuna trial into a plain configuration dictionary."""

        config: dict[str, Any] = {}
        for param_name, definition in self._config_space.items():
            suggestion = suggest_from_definition(trial, param_name, definition, config)
            if suggestion is not None:
                config[param_name] = suggestion

        return self._filter_provider_parameters(config)

    # ------------------------------------------------------------------
    # Metadata helpers
    # ------------------------------------------------------------------
    def get_algorithm_info(self) -> dict[str, Any]:
        info = super().get_algorithm_info()
        info.update(
            {
                "max_trials": self.max_trials,
                "optuna_sampler": self._sampler.__class__.__name__,
                "directions": self._study_directions,
            }
        )
        return info

    # ------------------------------------------------------------------
    # Recovery & telemetry helpers
    # ------------------------------------------------------------------
    def _restore_from_checkpoint(self) -> None:
        if not self._checkpoint_manager:
            return

        state = self._checkpoint_manager.load_state()
        pending = state.get("pending", [])
        if not pending:
            return

        restored = 0
        for params in pending:
            if not isinstance(params, dict) or not params:
                continue
            try:
                self._study.enqueue_trial(params=params)
                restored += 1
            except Exception:  # pragma: no cover - defensive, optuna internal failures
                logger.exception("Failed to enqueue trial from checkpoint: %s", params)

        if restored:
            logger.info("Restored %s pending trials from checkpoint", restored)

    def _persist_checkpoint(self) -> None:
        if not self._checkpoint_manager:
            return

        metadata = {
            "study_name": self._study.study_name,
            "objectives": self.objectives,
        }
        self._checkpoint_manager.save_state(
            self._pending_configs.values(), metadata=metadata
        )

    def _emit_event(
        self,
        event: str,
        trial_id: int,
        payload: dict[str, Any] | None = None,
    ) -> None:
        if not self._metrics_emitter:
            return

        try:
            self._metrics_emitter.emit_trial_update(
                event=event,
                trial_id=trial_id,
                study_name=self._study.study_name,
                payload=payload or {},
            )
        except Exception:  # pragma: no cover - defensive logging
            logger.exception("Failed to emit Optuna telemetry event")

    def _filter_provider_parameters(self, config: dict[str, Any]) -> dict[str, Any]:
        provider = config.get("provider")
        if not provider:
            return config

        filtered: dict[str, Any] = {}
        provider_prefix = f"{provider}."
        for key, value in config.items():
            if key.startswith(provider_prefix):
                filtered[key.split(".", 1)[1]] = value
            elif "." not in key:
                filtered[key] = value

        return filtered

    def _build_mock_config(self) -> dict[str, Any]:
        config: dict[str, Any] = {}
        for name, definition in self._config_space.items():
            if isinstance(definition, list):
                config[name] = definition[0]
            elif isinstance(definition, tuple) and len(definition) == 2:
                low, high = definition
                if isinstance(low, int) and isinstance(high, int):
                    config[name] = (low + high) // 2
                else:
                    config[name] = (float(low) + float(high)) / 2.0
            elif isinstance(definition, dict):
                if "default" in definition:
                    config[name] = definition["default"]
                elif definition.get("type") in {"categorical", "choice"}:
                    choices = (
                        definition.get("choices") or definition.get("values") or []
                    )
                    config[name] = choices[0] if choices else None
                elif definition.get("type") in {"int", "integer"}:
                    low = int(definition.get("low", 0))
                    high = int(definition.get("high", 0))
                    config[name] = (low + high) // 2
                elif definition.get("type") in {"float", "double"}:
                    low = float(definition.get("low", 0.0))
                    high = float(definition.get("high", 0.0))
                    config[name] = (low + high) / 2.0
                else:
                    config[name] = definition.get("value")
            else:
                config[name] = definition
        return config


class OptunaTPEOptimizer(OptunaBaseOptimizer):
    """Tree-structured Parzen Estimator sampler backed by Optuna."""

    sampler = optuna.samplers.TPESampler() if optuna else None  # pragma: no cover


class OptunaRandomOptimizer(OptunaBaseOptimizer):
    """Optuna random sampler."""

    sampler = optuna.samplers.RandomSampler() if optuna else None  # pragma: no cover


class OptunaCMAESOptimizer(OptunaBaseOptimizer):
    """CMA-ES sampler for continuous optimisation problems."""

    sampler = optuna.samplers.CmaEsSampler() if optuna else None  # pragma: no cover


class OptunaNSGAIIOptimizer(OptunaBaseOptimizer):
    """NSGA-II sampler for multi-objective optimisation."""

    sampler = optuna.samplers.NSGAIISampler() if optuna else None  # pragma: no cover


class OptunaGridOptimizer(OptunaBaseOptimizer):
    """Grid search implemented through Optuna's GridSampler."""

    def __init__(
        self,
        config_space: dict[str, Any],
        objectives: list[str],
        *,
        n_bins: int = 10,
        **kwargs: Any,
    ) -> None:
        from traigent.optimizers.optuna_utils import discretize_for_grid

        ensure_optuna_available()

        discrete_space = discretize_for_grid(config_space, n_bins=n_bins)
        sampler = optuna.samplers.GridSampler(search_space=discrete_space)
        super().__init__(
            config_space,
            objectives,
            sampler=sampler,
            **kwargs,
        )
