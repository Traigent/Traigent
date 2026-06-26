"""Configuration state management for optimized functions.

Manages the lifecycle of optimization configuration: loading, saving,
applying, and exporting best configurations. Extracted from
OptimizedFunction to reduce class complexity.
"""

# Traceability: CONC-Layer-Core FUNC-ORCH-LIFECYCLE REQ-ORCH-003

from __future__ import annotations

import json
import os
import threading
from collections.abc import Callable
from datetime import UTC, datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any

from traigent.api.types import OptimizationResult, OptimizationStatus
from traigent.core.best_config_runtime import (
    BEST_CONFIG_SCHEMA_VERSION,
    BestConfigSnapshot,
    BestConfigSource,
    BestConfigSourceMode,
    CloudPublishUnavailable,
    CloudPublishUnavailableReason,
    canonical_json,
    compute_spec_hash,
    function_ref_for,
    load_best_config_spec,
    resolve_cloud_cache_best_config,
    resolve_repo_best_config,
    sha256_digest,
    snapshot_from_spec,
    source_order_for_mode,
    thaw_config,
    write_cloud_cache_best_config,
    write_repo_best_config,
)
from traigent.utils.exceptions import ConfigurationError, OptimizationStateError
from traigent.utils.logging import get_logger
from traigent.utils.secure_path import (
    PathTraversalError,
    safe_open,
    validate_path,
    validate_user_path,
)

logger = get_logger(__name__)

DEFAULT_OPTIMIZATION_HISTORY_LIMIT = 100


class CloudBestConfigIntegrityError(ConfigurationError):
    """Raised when a cloud best-config response fails hash integrity checks."""


class OptimizationState(Enum):
    """Lifecycle state of an OptimizedFunction.

    States:
        UNOPTIMIZED: Before any optimization has been run.
        OPTIMIZING: During an active optimization run.
        OPTIMIZED: After optimization has completed successfully.
        ERROR: Optimization failed.
    """

    UNOPTIMIZED = auto()
    OPTIMIZING = auto()
    OPTIMIZED = auto()
    ERROR = auto()


class ConfigStateManager:
    """Manages optimization config state: results, history, load/save/export.

    This class is owned by OptimizedFunction and manages all config persistence
    and lifecycle state. It uses a callback to re-wrap the function when
    configuration changes.
    """

    def __init__(
        self,
        func: Callable[..., Any],
        default_config: dict[str, Any],
        local_storage_path: str | None,
        configuration_space: dict[str, Any] | None,
        auto_load_best: bool,
        load_from: str | None,
        setup_wrapper_callback: Callable[[], None],
        config_id: str | None = None,
        best_config_source: str = BestConfigSourceMode.OFF.value,
        best_config_strict: bool = False,
        best_config_cache_dir: str | None = None,
        best_config_cache_ttl_seconds: int = 24 * 60 * 60,
        best_config_stale_ok_ttl_seconds: int | None = None,
        best_config_environment: str | None = None,
        enable_auto_load_dev_logs: bool | None = None,
        optimization_history_limit: int = DEFAULT_OPTIMIZATION_HISTORY_LIMIT,
    ) -> None:
        """Initialize config state manager.

        Args:
            func: The original user function
            default_config: Default configuration to use
            local_storage_path: Path for local optimization logs
            configuration_space: Config space definition
            auto_load_best: Whether to auto-load best config on init
            load_from: Explicit path to load config from
            setup_wrapper_callback: Callback to re-wrap function with new config
            optimization_history_limit: Maximum number of optimization results to
                retain in in-memory history.
        """
        self.func = func
        self.default_config = default_config
        self.local_storage_path = local_storage_path
        self.configuration_space = configuration_space
        self._auto_load_best = auto_load_best
        self._load_from = load_from
        self._setup_wrapper_callback = setup_wrapper_callback
        self.config_id = config_id
        self.best_config_source = best_config_source
        self.best_config_strict = best_config_strict
        self.best_config_cache_dir = best_config_cache_dir
        self.best_config_cache_ttl_seconds = best_config_cache_ttl_seconds
        self.best_config_stale_ok_ttl_seconds = best_config_stale_ok_ttl_seconds
        self.best_config_environment = self._resolve_best_config_environment(
            best_config_environment
        )
        self.enable_auto_load_dev_logs = (
            auto_load_best
            if enable_auto_load_dev_logs is None
            else enable_auto_load_dev_logs
        )
        if optimization_history_limit < 1:
            raise ValueError("optimization_history_limit must be >= 1")
        self._optimization_history_limit = optimization_history_limit

        # Core state
        self._state = OptimizationState.UNOPTIMIZED
        self._state_lock = threading.RLock()
        self._optimization_results: OptimizationResult | None = None
        self._optimization_history: list[OptimizationResult] = []
        self._current_config: dict[str, Any] = default_config.copy()
        self._best_config: dict[str, Any] | None = None
        self._best_config_snapshot: BestConfigSnapshot = BestConfigSnapshot.from_config(
            self._current_config,
            config_id=self.config_id,
            source=BestConfigSource.DEFAULT.value,
        )
        self._override_sticky = False

    # -- Properties --------------------------------------------------------

    @property
    def state(self) -> OptimizationState:
        """Get the current lifecycle state."""
        return self._state

    @state.setter
    def state(self, value: OptimizationState) -> None:
        self._state = value

    @property
    def state_lock(self) -> threading.RLock:
        """Access the state lock for atomic operations."""
        return self._state_lock

    @property
    def optimization_results(self) -> OptimizationResult | None:
        return self._optimization_results

    @optimization_results.setter
    def optimization_results(self, value: OptimizationResult | None) -> None:
        self._optimization_results = value

    @property
    def optimization_history(self) -> list[OptimizationResult]:
        return self._optimization_history

    @property
    def current_config(self) -> dict[str, Any]:
        """Get the configuration this function uses when called.

        Raises:
            OptimizationStateError: If accessed during an active optimization.
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

    @property
    def current_config_raw(self) -> dict[str, Any]:
        """Direct access to current config without lock/state check (internal use)."""
        return self._current_config

    @current_config_raw.setter
    def current_config_raw(self, value: dict[str, Any]) -> None:
        self._current_config = value

    @property
    def best_config(self) -> dict[str, Any] | None:
        """Get the best configuration found during optimization."""
        return self._best_config.copy() if self._best_config else None

    @best_config.setter
    def best_config(self, value: dict[str, Any] | None) -> None:
        self._best_config = value

    @property
    def best_config_snapshot(self) -> BestConfigSnapshot:
        """Return the active immutable best-config snapshot."""
        return self._best_config_snapshot

    # -- Query methods -----------------------------------------------------

    def get_best_config(self) -> dict[str, Any] | None:
        """Get the best configuration found or loaded for runtime use."""
        if self._optimization_results and self._optimization_results.best_config:
            best: dict[str, Any] = self._optimization_results.best_config
            return best
        return self.best_config

    def get_optimization_results(self) -> OptimizationResult | None:
        """Get the latest optimization results."""
        return self._optimization_results

    def get_optimization_history(self) -> list[OptimizationResult]:
        """Get history of all optimization runs."""
        return self._optimization_history.copy()

    def append_optimization_result(self, result: OptimizationResult) -> None:
        """Append optimization result and enforce bounded history."""
        self._optimization_history.append(result)
        overflow = len(self._optimization_history) - self._optimization_history_limit
        if overflow > 0:
            del self._optimization_history[:overflow]

    def is_optimization_complete(self) -> bool:
        """Check if optimization has been completed."""
        return self._optimization_results is not None

    def _set_snapshot(
        self,
        snapshot: BestConfigSnapshot,
        *,
        best_config: dict[str, Any] | None,
    ) -> None:
        self._best_config_snapshot = snapshot
        self._current_config = thaw_config(snapshot.config)
        self._best_config = best_config.copy() if best_config else None

    def _reset_to_default_snapshot(self) -> None:
        snapshot = BestConfigSnapshot.from_config(
            self.default_config.copy(),
            config_id=self.config_id,
            source=BestConfigSource.DEFAULT.value,
        )
        self._set_snapshot(snapshot, best_config=None)

    def _snapshot_from_plain_config(
        self,
        config: dict[str, Any],
        *,
        source: str,
    ) -> BestConfigSnapshot:
        return BestConfigSnapshot.from_config(
            {**self.default_config, **config},
            config_id=self.config_id,
            source=source,
        )

    def _load_snapshot_from_file(
        self,
        path: str,
        *,
        source: str,
    ) -> BestConfigSnapshot | None:
        try:
            snapshot = load_best_config_spec(
                path,
                configuration_space=self.configuration_space,
                expected_config_id=self.config_id,
                expected_function_ref=function_ref_for(self.func),
                source=source,
                strict=True,
            )
            if snapshot is None:
                raise ConfigurationError(f"Failed to load best config from {path}")
            merged = {**self.default_config, **thaw_config(snapshot.config)}
            return BestConfigSnapshot.from_config(
                merged,
                config_id=snapshot.config_id,
                source=source,
                spec_hash=snapshot.spec_hash,
                provenance=thaw_config(snapshot.provenance),
            )
        except Exception as spec_exc:
            if self._path_has_best_config_schema(path):
                if self.best_config_strict:
                    if isinstance(spec_exc, ConfigurationError):
                        raise spec_exc
                    raise ConfigurationError(
                        f"Failed to load best config from {path}: {spec_exc}"
                    ) from spec_exc
                logger.warning(
                    "Ignoring invalid canonical best config %s: %s",
                    path,
                    spec_exc,
                )
                return None

            loaded_config = self._load_config_from_path(path)
            if loaded_config:
                return self._snapshot_from_plain_config(loaded_config, source=source)
            if self.best_config_strict:
                if isinstance(spec_exc, ConfigurationError):
                    raise spec_exc
                raise ConfigurationError(
                    f"Failed to load best config from {path}: {spec_exc}"
                ) from spec_exc
            logger.warning(
                "Failed to load best config from %s: %s. Function will use fallback.",
                path,
                spec_exc,
            )
            return None

    @staticmethod
    def _resolve_best_config_environment(explicit: str | None) -> str:
        raw = (
            explicit or os.environ.get("TRAIGENT_BEST_CONFIG_ENVIRONMENT") or "default"
        )
        normalized = raw.strip().lower()
        return normalized or "default"

    @staticmethod
    def _verify_cloud_response_hashes(
        data: dict[str, Any], spec: dict[str, Any]
    ) -> None:
        response_spec_hash = data.get("spec_hash")
        response_config_hash = data.get("config_hash")
        if not isinstance(response_spec_hash, str) or not response_spec_hash:
            raise CloudBestConfigIntegrityError(
                "Cloud best-config response did not include spec_hash"
            )
        if not isinstance(response_config_hash, str) or not response_config_hash:
            raise CloudBestConfigIntegrityError(
                "Cloud best-config response did not include config_hash"
            )
        config = spec.get("config")
        if not isinstance(config, dict):
            raise CloudBestConfigIntegrityError(
                "Cloud best-config spec.config must be an object"
            )

        actual_spec_hash = compute_spec_hash(spec)
        actual_config_hash = sha256_digest(canonical_json(config))
        if response_spec_hash != actual_spec_hash:
            raise CloudBestConfigIntegrityError("Cloud best-config spec_hash mismatch")
        if response_config_hash != actual_config_hash:
            raise CloudBestConfigIntegrityError(
                "Cloud best-config config_hash mismatch"
            )

    def _is_pure_cloud_source_mode(self) -> bool:
        mode_value = (
            self.best_config_source.value
            if isinstance(self.best_config_source, BestConfigSourceMode)
            else str(self.best_config_source)
        )
        return mode_value == BestConfigSourceMode.CLOUD.value

    def _resolve_cloud_fetch_best_config(self) -> BestConfigSnapshot | None:
        if not self.config_id:
            return None

        from traigent.cloud.backend_client import get_backend_client

        client = get_backend_client(enable_fallback=False)
        data = client.fetch_best_config_sync(
            self.config_id,
            environment=self.best_config_environment,
            function_ref=function_ref_for(self.func),
        )
        if not data:
            if self._is_pure_cloud_source_mode():
                raise CloudPublishUnavailable(
                    CloudPublishUnavailableReason.REQUEST_FAILED,
                    "Cloud best-config fetch returned no active config",
                )
            return None
        spec = data.get("spec")
        if not isinstance(spec, dict):
            raise ConfigurationError(
                "Cloud best-config response did not include a spec"
            )
        self._verify_cloud_response_hashes(data, spec)

        if self.best_config_cache_dir:
            try:
                write_cloud_cache_best_config(
                    self.best_config_cache_dir,
                    spec,
                    etag=(
                        data.get("etag") if isinstance(data.get("etag"), str) else None
                    ),
                    version=data.get("version"),
                )
            except Exception as cache_exc:
                if self.best_config_strict:
                    raise ConfigurationError(
                        f"Failed to update cloud best-config cache: {cache_exc}"
                    ) from cache_exc
                logger.warning(
                    "Fetched cloud best config but failed to update cache: %s",
                    cache_exc,
                )

        snapshot = snapshot_from_spec(
            spec,
            configuration_space=self.configuration_space,
            expected_config_id=self.config_id,
            expected_function_ref=function_ref_for(self.func),
            source=BestConfigSource.CLOUD_FETCH.value,
        )
        merged = {**self.default_config, **thaw_config(snapshot.config)}
        return BestConfigSnapshot.from_config(
            merged,
            config_id=snapshot.config_id,
            source=BestConfigSource.CLOUD_FETCH.value,
            spec_hash=snapshot.spec_hash,
            loaded_at=snapshot.loaded_at,
            provenance=thaw_config(snapshot.provenance),
        )

    def _resolve_mode_snapshot(self) -> BestConfigSnapshot | None:
        for source in source_order_for_mode(self.best_config_source):
            try:
                if source is BestConfigSource.REPO:
                    snapshot = resolve_repo_best_config(
                        config_id=self.config_id,
                        repo_root=Path.cwd(),
                        configuration_space=self.configuration_space,
                        expected_function_ref=function_ref_for(self.func),
                        strict=self.best_config_strict,
                    )
                elif source is BestConfigSource.CLOUD_CACHE:
                    snapshot = resolve_cloud_cache_best_config(
                        config_id=self.config_id,
                        cache_dir=self.best_config_cache_dir,
                        configuration_space=self.configuration_space,
                        ttl_seconds=self.best_config_cache_ttl_seconds,
                        stale_ok_ttl_seconds=self.best_config_stale_ok_ttl_seconds,
                        strict=self.best_config_strict,
                    )
                elif source is BestConfigSource.CLOUD_FETCH:
                    snapshot = self._resolve_cloud_fetch_best_config()
                else:
                    snapshot = None
            except Exception as exc:
                mode_value = (
                    self.best_config_source.value
                    if isinstance(self.best_config_source, BestConfigSourceMode)
                    else str(self.best_config_source)
                )
                if source is BestConfigSource.CLOUD_FETCH and (
                    mode_value == BestConfigSourceMode.CLOUD.value
                    or self.best_config_strict
                ):
                    if isinstance(exc, CloudPublishUnavailable):
                        raise
                    if isinstance(exc, CloudBestConfigIntegrityError):
                        raise CloudPublishUnavailable(
                            CloudPublishUnavailableReason.INTEGRITY_FAILED,
                            f"Cloud best-config integrity check failed: {exc}",
                        ) from exc
                    raise CloudPublishUnavailable(
                        CloudPublishUnavailableReason.REQUEST_FAILED,
                        f"Cloud best-config fetch failed: {exc}",
                    ) from exc
                if self.best_config_strict:
                    raise
                if isinstance(exc, CloudBestConfigIntegrityError):
                    logger.error(
                        "Cloud best-config integrity failure; ignoring %s source: %s",
                        source.value,
                        exc,
                    )
                else:
                    logger.warning(
                        "Ignoring %s best-config source after failure: %s",
                        source.value,
                        exc,
                    )
                snapshot = None
            if snapshot is not None:
                merged = {**self.default_config, **thaw_config(snapshot.config)}
                return BestConfigSnapshot.from_config(
                    merged,
                    config_id=snapshot.config_id,
                    source=snapshot.source,
                    spec_hash=snapshot.spec_hash,
                    loaded_at=snapshot.loaded_at,
                    expires_at=snapshot.expires_at,
                    provenance=thaw_config(snapshot.provenance),
                )
        return None

    # -- State mutation methods --------------------------------------------

    def reset_optimization(self) -> None:
        """Reset optimization state and restore default configuration."""
        self._optimization_results = None
        self._optimization_history = []
        self._override_sticky = False
        self._reset_to_default_snapshot()
        self._state = OptimizationState.UNOPTIMIZED
        self._setup_wrapper_callback()
        logger.info(f"Reset optimization state for {self.func.__name__}")

    def set_config(self, config: dict[str, Any]) -> None:
        """Set current configuration manually as a sticky override."""
        with self._state_lock:
            snapshot = self._snapshot_from_plain_config(
                config, source=BestConfigSource.OVERRIDE.value
            )
            self._set_snapshot(snapshot, best_config=config)
            self._override_sticky = True
            self._setup_wrapper_callback()
        logger.debug(f"Set configuration for {self.func.__name__}: {config}")

    def apply_best_config(
        self,
        results: OptimizationResult | None = None,
        *,
        get_wrapped_func: Callable[[], Any] | None = None,
        set_wrapped_func: Callable[[Any], None] | None = None,
    ) -> bool:
        """Apply best configuration from optimization results.

        Args:
            results: OptimizationResult to use (defaults to latest optimization)
            get_wrapped_func: Getter for the wrapped function (for rollback)
            set_wrapped_func: Setter for the wrapped function (for rollback)

        Returns:
            True if configuration applied successfully

        Raises:
            ConfigurationError: If no optimization results are available
        """
        if results is None:
            results = self._optimization_results

        if not results or not results.best_config:
            raise ConfigurationError(
                "No optimization results available to apply. "
                "Please run optimization first using .optimize()"
            )

        with self._state_lock:
            old_config = self._current_config.copy()
            old_best = self._best_config.copy() if self._best_config else None
            old_wrapped_func = get_wrapped_func() if get_wrapped_func else None
            old_snapshot = self._best_config_snapshot
            old_override_sticky = self._override_sticky
            try:
                snapshot = self._snapshot_from_plain_config(
                    results.best_config,
                    source=BestConfigSource.APPLY_BEST_CONFIG.value,
                )
                self._set_snapshot(snapshot, best_config=results.best_config)
                self._override_sticky = True
                self._setup_wrapper_callback()
            except Exception:
                self._current_config = old_config
                self._best_config = old_best
                self._best_config_snapshot = old_snapshot
                self._override_sticky = old_override_sticky
                if set_wrapped_func and old_wrapped_func is not None:
                    set_wrapped_func(old_wrapped_func)
                raise

        logger.info(
            f"Applied best config for {self.func.__name__}: {results.best_config} "
            f"(previous: {old_config})"
        )

        return True

    def clear_override(self) -> bool:
        """Clear any sticky per-instance config override and re-resolve sources."""
        with self._state_lock:
            had_override = self._override_sticky
            self._override_sticky = False
        self.maybe_auto_load_config()
        return had_override

    # -- Persistence methods -----------------------------------------------

    def save_optimization_results(self, path: str) -> None:
        """Save optimization results to file.

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

            from traigent.api.types import TrialResult, TrialStatus

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
            self.append_optimization_result(self._optimization_results)

            if self._optimization_results.best_config:
                snapshot = self._snapshot_from_plain_config(
                    self._optimization_results.best_config,
                    source=BestConfigSource.APPLY_BEST_CONFIG.value,
                )
                self._set_snapshot(
                    snapshot, best_config=self._optimization_results.best_config
                )
                self._override_sticky = True
                self._setup_wrapper_callback()

            if self._optimization_results.status == OptimizationStatus.COMPLETED:
                self._state = OptimizationState.OPTIMIZED
            elif self._optimization_results.status == OptimizationStatus.FAILED:
                self._state = OptimizationState.ERROR
            else:
                self._state = OptimizationState.OPTIMIZED

            logger.info(f"Loaded optimization results from {path}")

        except Exception as e:
            raise ConfigurationError(f"Failed to load optimization results: {e}") from e

    # -- Auto-load methods -------------------------------------------------

    def maybe_auto_load_config(self) -> None:
        """Resolve startup best config from explicit, repo, cache, or dev sources."""
        with self._state_lock:
            if self._override_sticky:
                return

        resolved: BestConfigSnapshot | None = None

        if self._load_from:
            logger.debug("Using explicit load_from path: %s", self._load_from)
            resolved = self._load_snapshot_from_file(
                self._load_from, source=BestConfigSource.LOAD_FROM.value
            )

        if resolved is None:
            env_path = os.environ.get("TRAIGENT_CONFIG_PATH")
            if env_path:
                logger.debug("Using TRAIGENT_CONFIG_PATH: %s", env_path)
                resolved = self._load_snapshot_from_file(
                    env_path, source=BestConfigSource.ENV.value
                )

        if resolved is None:
            resolved = self._resolve_mode_snapshot()

        if resolved is None and self.enable_auto_load_dev_logs:
            dev_log_path = self._find_latest_config_path()
            if dev_log_path:
                logger.debug("Auto-found latest dev log config: %s", dev_log_path)
                resolved = self._load_snapshot_from_file(
                    dev_log_path, source=BestConfigSource.DEV_LOG.value
                )

        with self._state_lock:
            if self._override_sticky:
                return
            if resolved is None:
                self._reset_to_default_snapshot()
            else:
                self._set_snapshot(resolved, best_config=thaw_config(resolved.config))
            self._setup_wrapper_callback()
            logger.debug(
                "Resolved best config for %s from %s",
                self.func.__name__,
                self._best_config_snapshot.source,
            )

    def _find_latest_config_path(self) -> str | None:
        """Find the most recent best_config file in optimization logs."""
        log_dirs = [
            Path.cwd() / ".traigent" / "optimization_logs",
            Path(os.environ.get("TRAIGENT_RESULTS_FOLDER", Path.home() / ".traigent"))
            / "optimization_logs",
        ]

        if self.local_storage_path:
            log_dirs.insert(0, Path(self.local_storage_path) / "optimization_logs")

        func_name = getattr(self.func, "__name__", "unknown")

        for log_dir in log_dirs:
            experiments_dir = log_dir / "experiments" / func_name / "runs"
            if not experiments_dir.exists():
                continue

            run_dirs = sorted(
                [d for d in experiments_dir.iterdir() if d.is_dir()],
                key=lambda d: d.stat().st_mtime,
                reverse=True,
            )

            for run_dir in run_dirs:
                for config_file in [
                    run_dir / "artifacts" / "best_config_v2.json",
                    run_dir / "artifacts" / "best_config.json",
                ]:
                    if config_file.exists():
                        return str(config_file)

        return None

    @staticmethod
    def _path_has_best_config_schema(path: str) -> bool:
        """Return True when a file declares the canonical best-config schema."""
        try:
            config_path = validate_user_path(path, for_write=False)
            with open(config_path, encoding="utf-8") as f:
                data = json.load(f)
            return (
                isinstance(data, dict)
                and data.get("schema_version") == BEST_CONFIG_SCHEMA_VERSION
            )
        except Exception:
            return False

    @staticmethod
    def _load_config_from_path(path: str) -> dict[str, Any] | None:
        """Load configuration from a JSON file."""
        try:
            config_path = validate_user_path(path, for_write=False)
            if not config_path.exists():
                logger.warning(f"Config file not found: {path}")
                return None

            with open(config_path, encoding="utf-8") as f:
                data = json.load(f)

            if "config" in data:
                return dict(data["config"])
            elif "best_config" in data:
                return dict(data["best_config"])
            elif isinstance(data, dict) and not any(
                k in data for k in ["trials", "metrics", "metadata"]
            ):
                return dict(data)
            else:
                logger.warning(
                    f"Unrecognized config format in {path}. "
                    "Expected 'config' or 'best_config' key, or direct config dict."
                )
                return None

        except json.JSONDecodeError as e:
            logger.warning(f"Invalid JSON in config file {path}: {e}")
            return None
        except PathTraversalError as e:
            logger.warning(f"Security: {e}")
            return None
        except Exception as e:
            logger.warning(f"Error loading config from {path}: {e}")
            return None

    # -- Export methods ----------------------------------------------------

    def export_config(
        self,
        path: str | Path,
        *,
        format: str = "slim",  # noqa: A002
        include_metadata: bool = True,
    ) -> Path:
        """Export the best configuration to a file."""
        if not self._best_config:
            raise ConfigurationError(
                "No best configuration available to export. "
                "Please run optimization first using .optimize() or load a config."
            )

        output_path: Path = validate_user_path(path, for_write=True)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if format == "slim":
            export_data = self._create_slim_export(include_metadata)
        elif format == "full":
            export_data = self._create_full_export(include_metadata)
        else:
            raise ConfigurationError(
                f"Unknown export format: {format}. Use 'slim' or 'full'."
            )

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=2, default=str)

        logger.info(f"Exported config for {self.func.__name__} to {output_path}")
        return output_path

    def export_best_config(
        self,
        directory: str | Path = ".traigent/best-configs",
        *,
        config_id: str | None = None,
        include_metadata: bool = True,
    ) -> Path:
        """Export the active best config as a canonical repo runtime spec."""
        if not self._best_config:
            raise ConfigurationError(
                "No best configuration available to export. "
                "Please run optimization first using .optimize() or load a config."
            )

        effective_config_id = config_id or self.config_id or self.func.__name__
        provenance = None
        if include_metadata:
            provenance = {
                "source": self._best_config_snapshot.source,
                "exported_at": datetime.now(UTC).isoformat(),
            }
            if self._optimization_results:
                provenance["optimization_id"] = (
                    self._optimization_results.optimization_id
                )

        return write_repo_best_config(
            directory,
            config_id=effective_config_id,
            config=self._best_config,
            function_ref=function_ref_for(self.func),
            provenance=provenance,
            environment=self.best_config_environment,
        )

    def publish_best_config(self, *, target: str = "cloud") -> dict[str, Any]:
        """Publish the active best config to durable backend cloud storage."""
        if target != "cloud":
            raise CloudPublishUnavailable(
                CloudPublishUnavailableReason.DISABLED_BY_CONFIG,
                "Only target='cloud' is reserved for best-config publishing. "
                "Use export_best_config() for local repo artifacts.",
            )
        if not self._best_config:
            raise ConfigurationError(
                "No best configuration available to publish. "
                "Please run optimization first using .optimize() or load a config."
            )

        from traigent.cloud.backend_client import get_backend_client

        effective_config_id = self.config_id or self.func.__name__
        provenance: dict[str, Any] = {
            "source": self._best_config_snapshot.source,
            "published_at": datetime.now(UTC).isoformat(),
        }
        if self._optimization_results:
            provenance["optimization_id"] = self._optimization_results.optimization_id
        spec = {
            "schema_version": BEST_CONFIG_SCHEMA_VERSION,
            "config_id": effective_config_id,
            "function_ref": function_ref_for(self.func),
            "environment": self.best_config_environment,
            "config": thaw_config(self._best_config),
            "provenance": provenance,
        }
        try:
            client = get_backend_client(enable_fallback=False)
            current = client.fetch_best_config_sync(
                effective_config_id,
                environment=self.best_config_environment,
                function_ref=function_ref_for(self.func),
            )
            if_match = (
                current.get("etag")
                if isinstance(current, dict) and isinstance(current.get("etag"), str)
                else None
            )
            result = client.publish_best_config_sync(
                spec,
                environment=self.best_config_environment,
                if_match=if_match,
            )
            self._verify_cloud_response_hashes(result, spec)
            return result
        except CloudPublishUnavailable:
            raise
        except CloudBestConfigIntegrityError as exc:
            raise CloudPublishUnavailable(
                CloudPublishUnavailableReason.INTEGRITY_FAILED,
                f"Cloud best-config publish integrity check failed: {exc}",
            ) from exc
        except Exception as exc:
            raise CloudPublishUnavailable(
                CloudPublishUnavailableReason.REQUEST_FAILED,
                f"Cloud best-config publish failed: {exc}",
            ) from exc

    def _create_slim_export(self, include_metadata: bool) -> dict[str, Any]:
        """Create a slim export suitable for git and deployment."""
        from traigent import __version__

        export: dict[str, Any] = {
            "config": dict(self._best_config) if self._best_config else {}
        }

        if include_metadata:
            export["function_name"] = getattr(self.func, "__name__", "unknown")
            export["exported_at"] = datetime.now(UTC).isoformat()
            export["traigent_version"] = __version__

            if self._optimization_results and self._optimization_results.best_metrics:
                export["metrics"] = {
                    k: v
                    for k, v in self._optimization_results.best_metrics.items()
                    if not k.startswith("_")
                }

        return export

    def _create_full_export(self, include_metadata: bool) -> dict[str, Any]:
        """Create a full export including trial history."""
        export = self._create_slim_export(include_metadata)

        if self._optimization_results:
            export["trials"] = [
                {
                    "trial_id": t.trial_id,
                    "config": t.config,
                    "metrics": t.metrics,
                    "status": t.status,
                }
                for t in self._optimization_results.trials
            ]

            export["optimization"] = {
                "total_trials": len(self._optimization_results.trials),
                "stop_reason": self._optimization_results.stop_reason,
                "duration_seconds": getattr(
                    self._optimization_results, "duration_seconds", None
                ),
            }

        if self.configuration_space:
            if hasattr(self.configuration_space, "to_dict"):
                export["configuration_space"] = self.configuration_space.to_dict()
            else:
                export["configuration_space"] = dict(self.configuration_space)

        return export
