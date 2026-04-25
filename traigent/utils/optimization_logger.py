"""Unified optimization logging with versioning and secret sanitization."""

# Traceability: CONC-Layer-Core CONC-Quality-Observability CONC-Quality-Security FUNC-ORCH-LIFECYCLE FUNC-STORAGE FUNC-ANALYTICS REQ-ORCH-003 REQ-STOR-007 REQ-ANLY-011 SYNC-OptimizationFlow SYNC-StorageLogging

from __future__ import annotations

import json
import os
import random
import sys
import tempfile
import threading
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np

from traigent._version import __version__ as TRAIGENT_VERSION
from traigent.api.types import OptimizationResult, TrialResult
from traigent.config.types import ExecutionMode, resolve_execution_mode
from traigent.core.objectives import ObjectiveSchema, normalize_objectives
from traigent.utils.file_versioning import FileVersionManager, RunVersionInfo
from traigent.utils.logging import get_logger
from traigent.utils.secure_path import PathTraversalError, validate_path

logger = get_logger(__name__)


_SENSITIVE_KEYWORDS = {
    "api_key",
    "apikey",
    "api_token",
    "api_keys",
    "access_token",
    "refresh_token",
    "token",
    "authorization",
    "auth_header",
    "auth_token",
    "bearer",
    "bearer_token",
    "client_secret",
    "client_token",
    "secret",
    "password",
    "private_key",
    "jwt",
    "jwt_token",
}
_SECRET_VALUE_PREFIXES = (
    "sk-",
    "rk-",
    "pk-",
    "pk_",
    "sk_",
    "rk_",
    "token-",
    "token_",
    "bearer ",
)
_SECRET_VALUE_SUBSTRINGS = (
    "-----BEGIN",
    "PRIVATE KEY",
)


def _mask_string(value: str) -> str:
    """Mask a string value while keeping it recognizable."""
    if not value:
        return "***"
    if len(value) <= 4:
        return "***"
    if len(value) <= 8:
        return f"{value[:2]}..."
    return f"{value[:4]}...{value[-4:]}"


def _normalize_key_name(key: str) -> str:
    return key.lower().replace("-", "_")


def _is_sensitive_key(key: str) -> bool:
    normalized = _normalize_key_name(key)
    if normalized in _SENSITIVE_KEYWORDS:
        return True
    if normalized.endswith(("_key", "_token", "_secret", "_password")):
        return True
    if normalized.startswith(("token_", "secret_", "password_", "jwt_")):
        return True
    return False


def _looks_like_secret(value: str) -> bool:
    lowered = value.lower()
    if any(
        lowered.startswith(prefix.strip().lower()) for prefix in _SECRET_VALUE_PREFIXES
    ):
        return True
    if value.startswith("eyJ") and value.count(".") == 2:
        # Likely a JWT
        return True
    for marker in _SECRET_VALUE_SUBSTRINGS:
        if marker in value:
            return True
    if len(value) >= 32 and " " not in value and any(ch.isdigit() for ch in value):
        # Long token-like string
        return True
    return False


def sanitize_for_logging(
    value: Any,
    *,
    key_hint: str | None = None,
    _visited: set[int] | None = None,
    _memo: dict[int, Any] | None = None,
) -> Any:
    """Return a copy of ``value`` with sensitive information masked."""
    if _visited is None:
        _visited = set()
    if _memo is None:
        _memo = {}

    if isinstance(value, dict):
        obj_id = id(value)
        if obj_id in _memo:
            return _memo[obj_id]
        if obj_id in _visited:
            return "<recursion>"
        _visited.add(obj_id)
        sanitized: dict[str, Any] = {}
        _memo[obj_id] = sanitized
        for nested_key, nested_value in value.items():
            sanitized[nested_key] = sanitize_for_logging(
                nested_value,
                key_hint=str(nested_key),
                _visited=_visited,
                _memo=_memo,
            )
        return sanitized

    if isinstance(value, (list, tuple, set)):
        obj_id = id(value)
        if obj_id in _memo:
            return _memo[obj_id]
        if obj_id in _visited:
            return "<recursion>"
        _visited.add(obj_id)
        sanitized_sequence = [
            sanitize_for_logging(
                item, key_hint=key_hint, _visited=_visited, _memo=_memo
            )
            for item in value
        ]
        _memo[obj_id] = sanitized_sequence
        return sanitized_sequence

    if isinstance(value, str):
        if key_hint and _is_sensitive_key(key_hint):
            return _mask_string(value)
        if _looks_like_secret(value):
            return _mask_string(value)
        return value

    if isinstance(value, (int, float, bool)) or value is None:
        return value

    # Fallback to string for unsupported types (e.g., dataclasses, enums)
    return str(value)


def _extract_output_text(output: Any) -> str | None:
    """Best-effort extraction of the human-readable response text."""
    if output is None:
        return None
    if isinstance(output, str):
        return output
    if isinstance(output, dict):
        # Common shape: {"response": "...", "query": "..."}
        for key in ("response", "text", "content", "answer"):
            if key in output:
                return str(output[key])
    return None


def _serialize_example_result(ex: Any) -> dict[str, Any]:
    """Serialize a per-example result (dataclass or dict) for jsonl logging."""
    if isinstance(ex, dict):
        metrics = ex.get("metrics", {})
        actual = ex.get("actual_output")
        return {
            "example_id": ex.get("example_id"),
            "query": actual.get("query") if isinstance(actual, dict) else None,
            "response": _extract_output_text(actual),
            "expected": ex.get("expected_output"),
            "accuracy": metrics.get("accuracy") if isinstance(metrics, dict) else None,
            "cost_usd": ex.get("cost_usd", 0.0),
            "latency_ms": ex.get("latency_ms", 0.0),
            "error": ex.get("error"),
        }
    # Dataclass (e.g. HybridExampleResult)
    metrics = getattr(ex, "metrics", {}) or {}
    actual = getattr(ex, "actual_output", None)
    return {
        "example_id": getattr(ex, "example_id", None),
        "query": actual.get("query") if isinstance(actual, dict) else None,
        "response": _extract_output_text(actual),
        "expected": getattr(ex, "expected_output", None),
        "accuracy": metrics.get("accuracy") if isinstance(metrics, dict) else None,
        "cost_usd": getattr(ex, "cost_usd", 0.0),
        "latency_ms": getattr(ex, "latency_ms", 0.0),
        "error": getattr(ex, "error", None),
    }


class OptimizationLogger:
    """Thread-safe optimization logger with versioned files and sanitization."""

    ENV_BASE_PATH = "TRAIGENT_OPTIMIZATION_LOG_DIR"

    def __init__(
        self,
        experiment_name: str,
        session_id: str,
        execution_mode: ExecutionMode | str,
        base_path: Path | None = None,
        buffer_size: int = 10,
    ) -> None:
        self.experiment_name = self._sanitize_name(experiment_name)
        self.session_id = session_id
        self.execution_mode_enum = resolve_execution_mode(execution_mode)
        self.execution_mode = self.execution_mode_enum.value
        self.base_path = (
            Path(base_path) if base_path else self._resolve_default_base_path()
        )
        self.buffer_size = buffer_size

        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        session_short = session_id[:8] if len(session_id) >= 8 else session_id
        self.run_id = f"{timestamp}_{session_short}"

        self.experiment_path = self.base_path / "experiments" / self.experiment_name
        self.run_path = self.experiment_path / "runs" / self.run_id

        self._file_locks: dict[Path, threading.Lock] = {}
        self._trial_buffer: list[TrialResult] = []
        self._buffer_lock = threading.Lock()

        self._ensure_directories()
        self.start_time = datetime.now(UTC)

        self.file_manager = FileVersionManager(version="2")
        self.version_info = RunVersionInfo(self.run_path)
        self.version_info.create_version_info(
            traigent_version=TRAIGENT_VERSION,
            file_naming_version="2",
            custom_metadata={
                "experiment_name": experiment_name,
                "execution_mode": self.execution_mode,
                "session_id": session_id,
            },
        )

        logger.info(
            f"Initialized OptimizationLogger for {experiment_name} run {self.run_id}"
        )

    @classmethod
    def _resolve_default_base_path(cls) -> Path:
        override = os.getenv(cls.ENV_BASE_PATH) or os.getenv("TRAIGENT_RESULTS_FOLDER")
        if override:
            try:
                return Path(override).expanduser().resolve()
            except OSError:
                logger.debug(
                    "Failed to resolve override base path %s; falling back to workspace directory",
                    override,
                )
        try:
            return (Path.cwd() / ".traigent" / "optimization_logs").resolve()
        except OSError:
            fallback = Path(tempfile.gettempdir()) / "traigent" / "optimization_logs"
            logger.debug("Falling back to temporary optimization log path %s", fallback)
            return fallback

    def _sanitize_name(self, name: str) -> str:
        invalid_chars = ["/", "\\", ":", "*", "?", '"', "<", ">", "|"]
        for char in invalid_chars:
            name = name.replace(char, "_")
        return name[:100]

    def _ensure_directories(self) -> None:
        directories = [
            self.run_path / "meta",
            self.run_path / "trials",
            self.run_path / "metrics",
            self.run_path / "checkpoints",
            self.run_path / "artifacts",
            self.run_path / "logs",
        ]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def _file_lock(self, file_path: Path):
        if file_path not in self._file_locks:
            self._file_locks[file_path] = threading.Lock()
        with self._file_locks[file_path]:
            yield

    def _atomic_write(self, file_path: Path, data: Any) -> None:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = file_path.with_suffix(f"{file_path.suffix}.tmp.{os.getpid()}")
        sanitized = sanitize_for_logging(data)
        try:
            # nosec - temp_path derives from caller-supplied file_path which the
            # logger already constrains to its own run/experiment directory.
            with open(temp_path, "w", encoding="utf-8") as handle:
                json.dump(sanitized, handle, indent=2, default=str, ensure_ascii=False)
            temp_path.replace(file_path)
        except Exception as exc:
            logger.error(f"Failed to write {file_path}: {exc}")
            raise
        finally:
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except OSError:
                    pass

    def _append_jsonl(self, file_path: Path, data: dict[str, Any]) -> None:
        sanitized = sanitize_for_logging(data)
        with self._file_lock(file_path):
            with open(file_path, "a", encoding="utf-8") as handle:
                json.dump(sanitized, handle, default=str, ensure_ascii=False)
                handle.write("\n")

    def log_session_start(
        self,
        config: dict[str, Any],
        objectives: list[str] | ObjectiveSchema | None,
        algorithm: str | None,
        dataset_info: dict[str, Any] | None = None,
    ) -> None:
        session_metadata = {
            "session_id": self.session_id,
            "run_id": self.run_id,
            "experiment_name": self.experiment_name,
            "execution_mode": self.execution_mode,
            "start_time": self.start_time.isoformat(),
            "status": "running",
            "file_version": self.file_manager.version,
        }

        session_file = (
            self.run_path / "meta" / self.file_manager.get_filename("session")
        )
        self._atomic_write(session_file, session_metadata)

        config_file = self.run_path / "meta" / self.file_manager.get_filename("config")
        self._atomic_write(config_file, config or {})

        objective_schema: ObjectiveSchema | None
        objectives_data: dict[str, Any] | None
        objective_schema = normalize_objectives(objectives)
        objectives_data = objective_schema.to_dict() if objective_schema else None

        if objectives_data is not None and objective_schema is not None:
            objectives_data["metadata"] = {
                "algorithm": algorithm,
                "timestamp": datetime.now(UTC).isoformat(),
                "normalization_strategy": "min_max",
                "weights_normalized": True,
                "schema_version": objective_schema.schema_version,
            }
            objectives_data["algorithm"] = algorithm
            objectives_data["timestamp"] = datetime.now(UTC).isoformat()
            objectives_data["summary"] = {
                "names": [obj.name for obj in objective_schema.objectives],
                "orientations": {
                    obj.name: obj.orientation for obj in objective_schema.objectives
                },
                "weights": {
                    obj.name: obj.weight for obj in objective_schema.objectives
                },
                "normalized_weights": objective_schema.weights_normalized,
                "weights_sum": objective_schema.weights_sum,
                "bounds": {
                    obj.name: list(obj.bounds) if obj.bounds else None
                    for obj in objective_schema.objectives
                },
            }
            objectives_file = (
                self.run_path / "meta" / self.file_manager.get_filename("objectives")
            )
            self._atomic_write(objectives_file, objectives_data)

        if dataset_info:
            dataset_file = (
                self.run_path / "meta" / self.file_manager.get_filename("dataset")
            )
            self._atomic_write(dataset_file, dataset_info)

        environment_data = {
            "python_version": (
                f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
            ),
            "platform": sys.platform,
            "traigent_version": TRAIGENT_VERSION,
            "timestamp": datetime.now(UTC).isoformat(),
        }
        environment_file = (
            self.run_path / "meta" / self.file_manager.get_filename("environment")
        )
        self._atomic_write(environment_file, environment_data)

        manifest = self.file_manager.create_manifest(self.run_path)
        manifest_file = (
            self.run_path / "meta" / self.file_manager.get_filename("manifest")
        )
        self._atomic_write(manifest_file, manifest)

        self._update_status("running")
        logger.info(f"Logged session start for {self.run_id}")

    def log_config(self, config: dict[str, Any]) -> None:
        config_file = self.run_path / "meta" / self.file_manager.get_filename("config")
        self._atomic_write(config_file, config)
        logger.info(f"Logged optimization config for {self.run_id}")

    def log_objectives(self, objectives: Any) -> None:
        if hasattr(objectives, "to_dict"):
            objectives_data = objectives.to_dict()
            if "objectives" in objectives_data:
                objectives_data["summary"] = {
                    "names": [obj["name"] for obj in objectives_data["objectives"]],
                    "orientations": {
                        obj["name"]: obj["orientation"]
                        for obj in objectives_data["objectives"]
                    },
                    "weights": {
                        obj["name"]: obj["weight"]
                        for obj in objectives_data["objectives"]
                    },
                }
        else:
            objectives_data = objectives
        objectives_file = (
            self.run_path / "meta" / self.file_manager.get_filename("objectives")
        )
        self._atomic_write(objectives_file, objectives_data)
        logger.info(f"Logged objectives for {self.run_id}")

    def log_trial_result(self, trial_result: TrialResult) -> None:
        if self.execution_mode_enum is ExecutionMode.CLOUD:
            return
        with self._buffer_lock:
            self._trial_buffer.append(trial_result)
            if len(self._trial_buffer) >= self.buffer_size:
                self._flush_trial_buffer()

    def _flush_trial_buffer(self) -> None:
        if not self._trial_buffer:
            return
        trials_file = (
            self.run_path / "trials" / self.file_manager.get_filename("trials_stream")
        )
        for trial in self._trial_buffer:
            trial_data = {
                "trial_id": trial.trial_id,
                "config": trial.config,
                "metrics": trial.metrics,
                "status": trial.status,
                "duration": trial.duration,
                "timestamp": trial.timestamp.isoformat() if trial.timestamp else None,
            }
            serialized_examples: list[dict[str, Any]] | None = None
            example_results = (
                trial.metadata.get("example_results") if trial.metadata else None
            )
            if example_results:
                serialized_examples = [
                    _serialize_example_result(ex) for ex in example_results
                ]
                trial_data["example_results"] = serialized_examples
            self._append_jsonl(trials_file, trial_data)
            if example_results:
                trial_file = (
                    self.run_path
                    / "trials"
                    / self.file_manager.get_filename(
                        "trial_detail", trial_id=trial.trial_id
                    )
                )
                self._atomic_write(trial_file, trial_data)
            # Append human-readable summary table
            self._append_trial_summary(trial, serialized_examples)
        self._trial_buffer.clear()

    def _append_trial_summary(
        self,
        trial: Any,
        serialized_examples: list[dict[str, Any]] | None,
    ) -> None:
        """Append a human-readable summary table for one trial."""
        summary_file = self.run_path / "trials" / "trial_summary.txt"
        lines: list[str] = []
        acc = trial.metrics.get("accuracy", "?") if trial.metrics else "?"
        lines.append(
            f"=== {trial.trial_id}  |  accuracy={acc}  |  config={trial.config} ==="
        )
        if serialized_examples:
            hdr = f"  {'example_id':<14} {'acc':>4}  {'expected':<22} {'response'}"
            lines.append(hdr)
            lines.append(f"  {'-' * 14} {'-' * 4}  {'-' * 22} {'-' * 50}")
            for ex in serialized_examples:
                iid = str(ex.get("example_id") or "?")
                a = ex.get("accuracy", "?")
                exp = str(ex.get("expected") or "")[:22]
                resp = str(ex.get("response") or ex.get("query") or "")[:80]
                lines.append(f"  {iid:<14} {a!s:>4}  {exp:<22} {resp}")
        lines.append("")
        try:
            with open(summary_file, "a", encoding="utf-8") as fh:
                fh.write("\n".join(lines) + "\n")
        except OSError:
            pass

    def log_metrics_update(self, metrics: dict[str, Any]) -> None:
        timestamp = datetime.now(UTC).isoformat()
        for metric_name, metric_value in metrics.items():
            if metric_value is None:
                continue
            metric_file = (
                self.run_path
                / "metrics"
                / self.file_manager.get_filename(
                    "metric_stream", metric_name=metric_name
                )
            )
            metric_data = {"timestamp": timestamp, "value": metric_value}
            self._append_jsonl(metric_file, metric_data)

    def save_checkpoint(
        self,
        optimizer_state: dict[str, Any],
        trials_history: list[TrialResult],
        trial_count: int,
    ) -> None:
        checkpoint_data = {
            "timestamp": datetime.now(UTC).isoformat(),
            "trial_count": trial_count,
            "optimizer_state": optimizer_state,
            "random_state": {
                "numpy": (
                    np.random.get_state() if hasattr(np.random, "get_state") else None
                ),
                "python": random.getstate(),
            },
            "trials_summary": {
                "total": len(trials_history),
                "successful": sum(1 for t in trials_history if t.is_successful),
                "failed": sum(1 for t in trials_history if t.status == "failed"),
            },
        }
        checkpoint_file = (
            self.run_path
            / "checkpoints"
            / self.file_manager.get_filename("checkpoint", trial_count=trial_count)
        )
        self._atomic_write(checkpoint_file, checkpoint_data)

        history_file = (
            self.run_path
            / "checkpoints"
            / self.file_manager.get_filename("trial_history")
        )
        history_data = [
            {
                "trial_id": trial.trial_id,
                "config": trial.config,
                "metrics": trial.metrics,
                "status": trial.status,
                "duration": trial.duration,
            }
            for trial in trials_history
        ]
        self._atomic_write(history_file, history_data)

        latest_file = (
            self.run_path
            / "checkpoints"
            / self.file_manager.get_filename("checkpoint_latest")
        )
        latest_data = {
            "checkpoint_file": checkpoint_file.name,
            "trial_count": trial_count,
            "timestamp": datetime.now(UTC).isoformat(),
        }
        self._atomic_write(latest_file, latest_data)
        logger.debug(f"Saved checkpoint at trial {trial_count}")

    def log_results(self, results: Any) -> None:
        artifacts_dir = self.run_path / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        if hasattr(results, "best_config"):
            config = results.best_config
            score = results.best_score
            metrics = getattr(results, "best_metrics", {})
            full_results = {
                "trials": [
                    trial.__dict__ if hasattr(trial, "__dict__") else trial
                    for trial in results.trials
                ],
                "best_config": config,
                "best_score": score,
                "optimization_id": results.optimization_id,
                "duration": results.duration,
                "convergence_info": results.convergence_info,
                "status": str(results.status),
                "objectives": results.objectives,
                "algorithm": results.algorithm,
            }
        elif isinstance(results, dict):
            config = results.get("best_config", {})
            score = results.get("best_score")
            metrics = results.get("best_metrics", {})
            full_results = results
        else:
            config = {}
            score = None
            metrics = {}
            full_results = {"raw_results": results}

        file_path = artifacts_dir / self.file_manager.get_filename("best_config")
        payload = {
            "config": config,
            "score": score,
            "metrics": metrics,
            "timestamp": datetime.now(UTC).isoformat(),
            "full_results": full_results,
        }
        self._atomic_write(file_path, payload)
        logger.info(f"Logged results for {self.run_id}")

    def log_session_end(
        self,
        optimization_result: OptimizationResult,
        weighted_results: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        with self._buffer_lock:
            self._flush_trial_buffer()

        best_config_file = (
            self.run_path / "artifacts" / self.file_manager.get_filename("best_config")
        )
        best_config_data = {
            "config": optimization_result.best_config,
            "score": optimization_result.best_score,
            "objectives": optimization_result.objectives,
            "metrics": optimization_result.best_metrics,
        }
        self._atomic_write(best_config_file, best_config_data)

        if weighted_results:
            weighted_file = (
                self.run_path
                / "artifacts"
                / self.file_manager.get_filename("weighted_results")
            )
            self._atomic_write(weighted_file, weighted_results)

        if len(optimization_result.objectives) > 1:
            from traigent.utils.multi_objective import ParetoFrontCalculator

            calculator = ParetoFrontCalculator()
            pareto_front = calculator.calculate_pareto_front(
                optimization_result.trials, optimization_result.objectives
            )
            if pareto_front:
                pareto_data = {
                    "configurations": [p.config for p in pareto_front],
                    "objectives": [p.objectives for p in pareto_front],
                    "count": len(pareto_front),
                    "timestamp": datetime.now(UTC).isoformat(),
                }
                pareto_file = (
                    self.run_path
                    / "artifacts"
                    / self.file_manager.get_filename("pareto_front")
                )
                self._atomic_write(pareto_file, pareto_data)

        metrics_summary_file = (
            self.run_path
            / "metrics"
            / self.file_manager.get_filename("metrics_summary")
        )
        metrics_summary = {
            "total_trials": len(optimization_result.trials),
            "successful_trials": len(optimization_result.successful_trials),
            "failed_trials": len(optimization_result.failed_trials),
            "success_rate": optimization_result.success_rate,
            "duration": optimization_result.duration,
            "total_cost": optimization_result.total_cost,
            "total_tokens": optimization_result.total_tokens,
            "best_metrics": optimization_result.best_metrics,
            "algorithm": optimization_result.algorithm,
            "timestamp": datetime.now(UTC).isoformat(),
        }
        self._atomic_write(metrics_summary_file, metrics_summary)

        session_file = (
            self.run_path / "meta" / self.file_manager.get_filename("session")
        )
        with self._file_lock(session_file):
            if session_file.exists():
                with open(session_file) as handle:
                    session_data = json.load(handle)
            else:
                session_data = {}
            session_data.update(
                {
                    "end_time": datetime.now(UTC).isoformat(),
                    "duration": optimization_result.duration,
                    "status": "failed" if error else "completed",
                    "error": error,
                }
            )
            self._atomic_write(session_file, session_data)

        final_manifest = self.file_manager.create_manifest(self.run_path)
        final_manifest["finalized"] = True
        final_manifest["finalized_at"] = datetime.now(UTC).isoformat()
        manifest_file = (
            self.run_path / "meta" / self.file_manager.get_filename("manifest")
        )
        self._atomic_write(manifest_file, final_manifest)

        status = "failed" if error else "completed"
        self._update_status(status)
        self._update_experiment_index()
        logger.info(f"Logged session end for {self.run_id} with status: {status}")

    def _update_status(self, status: str) -> None:
        status_file = self.run_path / self.file_manager.get_filename("status")
        status_data = {
            "status": status,
            "timestamp": datetime.now(UTC).isoformat(),
        }
        self._atomic_write(status_file, status_data)

    def _update_experiment_index(self) -> None:
        index_file = self.base_path / "index.json"
        with self._file_lock(index_file):
            if index_file.exists():
                with open(index_file) as handle:
                    index_data = json.load(handle)
            else:
                index_data = {"experiments": {}, "version": "2.0.0"}
            experiments = index_data.setdefault("experiments", {})
            experiment_entry = experiments.setdefault(
                self.experiment_name,
                {"runs": [], "created": datetime.now(UTC).isoformat()},
            )
            run_info = {
                "run_id": self.run_id,
                "session_id": self.session_id,
                "timestamp": self.start_time.isoformat(),
                "execution_mode": self.execution_mode,
                "path": str(self.run_path),
                "file_version": self.file_manager.version,
            }
            if not any(
                run.get("run_id") == self.run_id for run in experiment_entry["runs"]
            ):
                experiment_entry["runs"].append(run_info)
            self._atomic_write(index_file, index_data)

    @classmethod
    def load_checkpoint(
        cls,
        experiment_name: str,
        run_id: str,
        base_path: Path | None = None,
    ) -> dict[str, Any]:
        base_path = base_path or cls._resolve_default_base_path()
        run_path = base_path / "experiments" / experiment_name / "runs" / run_id

        version_info = RunVersionInfo(run_path)
        compatibility = version_info.check_compatibility(TRAIGENT_VERSION)
        if not compatibility["compatible"]:
            logger.warning(f"Version compatibility issues: {compatibility['warnings']}")

        file_manager = FileVersionManager(version="2")
        latest_file = (
            run_path / "checkpoints" / file_manager.get_filename("checkpoint_latest")
        )
        if not latest_file.exists():
            raise FileNotFoundError(f"No checkpoint found for run {run_id}") from None

        with open(latest_file) as handle:
            latest_data = json.load(handle)

        # Constrain the checkpoint reference to the run's checkpoints directory.
        # The filename comes from latest.json on disk; a tampered manifest
        # could otherwise traverse out of run_path via "../" segments.
        checkpoints_dir = run_path / "checkpoints"
        try:
            checkpoint_file = validate_path(
                latest_data["checkpoint_file"], checkpoints_dir, must_exist=True
            )
        except PathTraversalError as exc:
            raise ValueError(
                f"Invalid checkpoint reference in {latest_file}: {exc}"
            ) from exc
        with open(checkpoint_file) as handle:
            checkpoint_data = json.load(handle)

        history_file = (
            run_path / "checkpoints" / file_manager.get_filename("trial_history")
        )
        with open(history_file) as handle:
            trial_history = json.load(handle)

        checkpoint_data["trial_history"] = trial_history
        checkpoint_data["run_path"] = str(run_path)
        checkpoint_data["version_compatibility"] = compatibility
        logger.info(
            f"Loaded checkpoint from {run_id} at trial {checkpoint_data.get('trial_count')}"
        )
        return cast(dict[str, Any], checkpoint_data)

    def generate_manifest(self) -> Path:
        manifest = self.file_manager.create_manifest(self.run_path)
        manifest_file = (
            self.run_path / "meta" / self.file_manager.get_filename("manifest")
        )
        with open(manifest_file, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2)
        return manifest_file

    def load_latest_checkpoint(self) -> dict[str, Any]:
        return self.load_checkpoint(self.experiment_name, self.run_id)

    @classmethod
    def load_latest_checkpoint_static(
        cls, experiment_name: str, run_id: str, base_path: Path | None = None
    ) -> dict[str, Any]:
        return cls.load_checkpoint(experiment_name, run_id, base_path)
