"""Checkpoint persistence utilities for Optuna-based optimizers."""

# Traceability: CONC-Layer-Core CONC-Quality-Reliability CONC-Quality-Maintainability FUNC-OPT-ALGORITHMS REQ-OPT-ALG-004 SYNC-OptimizationFlow

from __future__ import annotations

import json
import threading
import time
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, cast

from traigent.utils.logging import get_logger

try:  # pragma: no cover - imported lazily in some environments
    from optuna.trial import TrialState
except Exception:  # pragma: no cover - optuna optional at runtime
    TrialState = None


logger = get_logger(__name__)


class OptunaCheckpointManager:
    """Persist Optuna state for both incremental recovery and crash handling."""

    def __init__(
        self,
        path: str | Path | None = None,
        *,
        version: int = 1,
        checkpoint_dir: str | Path | None = None,
        file_prefix: str = "optuna_checkpoint",
        max_checkpoints: int | None = None,
    ) -> None:
        if path is None and checkpoint_dir is None:
            raise TypeError("Either path or checkpoint_dir must be provided")

        self.version = version
        self._lock = threading.RLock()

        self.path = Path(path) if path is not None else None
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)

        self._checkpoint_dir = (
            Path(checkpoint_dir) if checkpoint_dir is not None else None
        )
        if self._checkpoint_dir is not None:
            self._checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self._file_prefix = file_prefix
        self._max_checkpoints = max_checkpoints

    # ------------------------------------------------------------------
    # Legacy single-file persistence used by OptunaTPEOptimizer
    # ------------------------------------------------------------------
    def save_state(
        self,
        pending_configs: Iterable[dict[str, Any]],
        *,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Persist pending trial configurations to the legacy checkpoint file."""

        if self.path is None:
            raise RuntimeError("save_state requires a path-based checkpoint manager")

        pending_list = [dict(cfg) for cfg in pending_configs]
        for item in pending_list:
            item.pop("_optuna_trial_id", None)

        with self._lock:
            if not pending_list:
                if self.path.exists():
                    self.path.unlink()
                return

            payload = {
                "version": self.version,
                "updated_at": time.time(),
                "pending": pending_list,
                "metadata": metadata or {},
            }
            tmp_path = self.path.with_suffix(".tmp")
            tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
            tmp_path.replace(self.path)
            logger.debug(
                "Checkpoint persisted with %s pending configs at %s",
                len(pending_list),
                self.path,
            )

    def load_state(self) -> dict[str, Any]:
        """Load legacy checkpoint contents, returning pending configs and metadata."""

        if self.path is None:
            return {"pending": [], "metadata": {}}

        with self._lock:
            if not self.path.exists():
                return {"pending": [], "metadata": {}}

            try:
                payload = json.loads(self.path.read_text())
            except json.JSONDecodeError:
                logger.warning("Failed to decode Optuna checkpoint at %s", self.path)
                return {"pending": [], "metadata": {}}

            if payload.get("version") != self.version:
                logger.warning(
                    "Checkpoint version mismatch (%s != %s); ignoring",
                    payload.get("version"),
                    self.version,
                )
                return {"pending": [], "metadata": {}}

            return {
                "pending": payload.get("pending", []),
                "metadata": payload.get("metadata", {}),
            }

    def clear(self) -> None:
        """Delete any stored legacy checkpoint data."""

        if self.path is None:
            return

        with self._lock:
            if self.path.exists():
                self.path.unlink()

    # ------------------------------------------------------------------
    # Directory-based checkpoints for richer crash recovery flows
    # ------------------------------------------------------------------
    def save_checkpoint(
        self,
        *,
        study: Any | None,
        pending_trials: Mapping[int, Any] | Iterable[Any] | None,
        metadata: dict[str, Any] | None = None,
    ) -> Path:
        """Persist a snapshot of the study, pending trials, and metadata."""

        if self._checkpoint_dir is None:
            raise RuntimeError("checkpoint_dir must be configured for save_checkpoint")

        with self._lock:
            snapshot = {
                "version": self.version,
                "created_at": time.time(),
                "metadata": dict(metadata or {}),
            }
            snapshot.update(self._serialize_study(study))
            snapshot["pending_trials"] = self._serialize_pending_trials(pending_trials)

            checkpoint_path = self._next_checkpoint_path()
            tmp_path = checkpoint_path.with_suffix(".tmp")
            tmp_path.write_text(json.dumps(snapshot, indent=2, sort_keys=True))
            tmp_path.replace(checkpoint_path)

            if self._max_checkpoints is not None:
                self._prune_old_checkpoints()

            logger.debug("Wrote Optuna checkpoint to %s", checkpoint_path)
            return checkpoint_path

    def load_checkpoint(self, path: str | Path) -> dict[str, Any]:
        """Load a structured checkpoint previously written by save_checkpoint."""

        checkpoint_path = Path(path)
        with self._lock:
            try:
                payload = json.loads(checkpoint_path.read_text())
            except json.JSONDecodeError as exc:  # pragma: no cover - defensive
                raise ValueError(
                    f"Failed to decode checkpoint at {checkpoint_path}"
                ) from exc

        payload.setdefault("metadata", {})
        payload.setdefault("completed_trials", [])
        payload.setdefault("pruned_trials", [])
        payload.setdefault("failed_trials", [])
        payload.setdefault("pending_trials", [])
        return cast(dict[str, Any], payload)

    def get_latest_checkpoint(self) -> Path | None:
        """Return the most recent checkpoint file if one exists."""

        if self._checkpoint_dir is None:
            return None

        checkpoints = self.list_checkpoints()
        return checkpoints[-1] if checkpoints else None

    def list_checkpoints(self) -> list[Path]:
        """List checkpoint files in chronological order."""

        if self._checkpoint_dir is None:
            return []

        files = []
        for path in self._checkpoint_dir.glob("*.json"):
            if path.name.startswith(self._file_prefix):
                files.append(path)

        files.sort(key=lambda p: (p.stat().st_mtime, p.name))
        return files

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _serialize_study(self, study: Any | None) -> dict[str, list[dict[str, Any]]]:
        completed: list[dict[str, Any]] = []
        pruned: list[dict[str, Any]] = []
        failed: list[dict[str, Any]] = []

        trials = []
        if study is None:
            pass
        elif hasattr(study, "get_trials"):
            try:
                trials = study.get_trials(deepcopy=False)
            except TypeError:  # pragma: no cover - optuna<=2 compatibility fallback
                trials = study.get_trials()
        elif hasattr(study, "trials"):
            trials = study.trials

        for trial in trials or []:
            state_name = self._state_name(trial)
            payload = {
                "number": getattr(trial, "number", None),
                "params": dict(getattr(trial, "params", {}) or {}),
                "values": self._extract_values(trial),
            }

            if state_name == "COMPLETE":
                completed.append(payload)
            elif state_name == "PRUNED":
                payload["metadata"] = {
                    "pruned_step": getattr(trial, "last_step", None),
                }
                pruned.append(payload)
            elif state_name == "FAIL":
                failed.append(payload)

        return {
            "completed_trials": completed,
            "pruned_trials": pruned,
            "failed_trials": failed,
        }

    def _serialize_pending_trials(
        self, pending_trials: Mapping[int, Any] | Iterable[Any] | None
    ) -> list[dict[str, Any]]:
        if not pending_trials:
            return []

        entries: list[dict[str, Any]] = []
        items_list: list[tuple[int | None, Any]] = []

        if isinstance(pending_trials, Mapping):
            items_list.extend(pending_trials.items())
        else:
            for item in pending_trials:
                trial_id = getattr(item, "number", None)
                items_list.append((trial_id, item))

        for trial_id, payload in items_list:
            params: dict[str, Any]
            if hasattr(payload, "params"):
                params = dict(getattr(payload, "params", {}) or {})
            elif isinstance(payload, Mapping):
                params = dict(payload)
            else:
                params = {}

            entries.append(
                {
                    "trial_id": trial_id,
                    "params": params,
                }
            )

        return entries

    def _next_checkpoint_path(self) -> Path:
        assert self._checkpoint_dir is not None  # Guarded by caller

        timestamp = int(time.time() * 1000)
        counter = 0
        while True:
            suffix = f"{timestamp}_{counter:03d}" if counter else str(timestamp)
            candidate = self._checkpoint_dir / f"{self._file_prefix}_{suffix}.json"
            if not candidate.exists():
                return candidate
            counter += 1

    def _prune_old_checkpoints(self) -> None:
        checkpoints = self.list_checkpoints()
        while (
            self._max_checkpoints is not None
            and len(checkpoints) > self._max_checkpoints
        ):
            oldest = checkpoints.pop(0)
            try:
                oldest.unlink()
            except FileNotFoundError:  # pragma: no cover - raced deletion
                continue

    def _state_name(self, trial: Any) -> str | None:
        state = getattr(trial, "state", None)
        if state is None:
            return None

        if TrialState is not None and isinstance(state, TrialState):
            return str(state.name)

        name = getattr(state, "name", None)
        if name is not None:
            return str(name)

        if isinstance(state, str):
            return state.upper()

        return str(state)

    def _extract_values(self, trial: Any) -> list[float] | float | None:
        values = getattr(trial, "values", None)
        if values is not None:
            try:
                return list(values)
            except TypeError:  # pragma: no cover - single scalar
                return values  # type: ignore[no-any-return]

        value = getattr(trial, "value", None)
        if value is not None:
            return float(value)
        return None
