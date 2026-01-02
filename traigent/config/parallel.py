"""Unified parallel execution configuration utilities."""

# Traceability: CONC-Orchestration CONC-Invocation FUNC-ORCH-LIFECYCLE FUNC-INVOKERS REQ-ORCH-003 REQ-INV-006 SYNC-OptimizationFlow CONC-Layer-Core

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Literal

from traigent.utils.validation import ValidationResult

ParallelMode = Literal["auto", "sequential", "parallel"]


@dataclass
class ParallelConfig:
    """User-facing configuration container for parallel execution.

    None indicates "inherit" / leave unspecified.
    """

    mode: ParallelMode | None = None
    trial_concurrency: int | None = None
    example_concurrency: int | None = None
    thread_workers: int | None = None

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> ParallelConfig:
        """Create ParallelConfig from a dictionary.

        Args:
            raw: Dictionary with parallel configuration keys.

        Returns:
            ParallelConfig instance.

        Raises:
            ValueError: If unknown keys are present in the dictionary.
        """
        _VALID_KEYS = {
            "mode",
            "trial_concurrency",
            "example_concurrency",
            "thread_workers",
        }

        unknown = set(raw.keys()) - _VALID_KEYS
        if unknown:
            raise ValueError(
                f"Unknown parallel_config key(s): {sorted(unknown)}. "
                "Valid keys: mode, trial_concurrency, example_concurrency, thread_workers"
            )

        return cls(
            mode=raw.get("mode"),
            trial_concurrency=raw.get("trial_concurrency"),
            example_concurrency=raw.get("example_concurrency"),
            thread_workers=raw.get("thread_workers"),
        )

    @classmethod
    def from_legacy(
        cls,
        *,
        parallel_trials: int | None = None,
        batch_size: int | None = None,
        parallel_workers: int | None = None,
    ) -> ParallelConfig:
        if parallel_trials is None and batch_size is None and parallel_workers is None:
            return cls()

        inferred_mode: ParallelMode | None = None
        if any(
            value is not None and value > 1
            for value in (parallel_trials, batch_size, parallel_workers)
        ):
            inferred_mode = "parallel"

        return cls(
            mode=inferred_mode,
            trial_concurrency=parallel_trials,
            example_concurrency=batch_size,
            thread_workers=parallel_workers,
        )

    def merge(self, override: ParallelConfig | None) -> ParallelConfig:
        if override is None:
            return self
        merged = ParallelConfig(
            mode=override.mode if override.mode is not None else self.mode,
            trial_concurrency=(
                override.trial_concurrency
                if override.trial_concurrency is not None
                else self.trial_concurrency
            ),
            example_concurrency=(
                override.example_concurrency
                if override.example_concurrency is not None
                else self.example_concurrency
            ),
            thread_workers=(
                override.thread_workers
                if override.thread_workers is not None
                else self.thread_workers
            ),
        )
        return merged


@dataclass
class ParallelConfigResolved:
    mode: Literal["sequential", "parallel"]
    trial_concurrency: int
    example_concurrency: int
    thread_workers: int
    sources: dict[str, str]
    warnings: list[str]

    @property
    def as_legacy_kwargs(self) -> dict[str, int]:
        return {
            "parallel_trials": self.trial_concurrency,
            "batch_size": self.example_concurrency,
            "parallel_workers": self.thread_workers,
        }


def coerce_parallel_config(
    value: ParallelConfig | dict[str, Any] | None,
) -> ParallelConfig | None:
    if value is None or isinstance(value, ParallelConfig):
        return value
    if isinstance(value, dict):
        return ParallelConfig.from_dict(value)
    raise TypeError(f"Unsupported parallel_config type: {type(value)!r}")


def merge_parallel_configs(
    configs: list[tuple[ParallelConfig | None, str]],
) -> tuple[ParallelConfig, dict[str, str]]:
    merged = ParallelConfig()
    sources: dict[str, str] = {}

    for config, source in configs:
        if config is None:
            continue
        if config.mode is not None:
            merged = replace(merged, mode=config.mode)
            sources["mode"] = source
        if config.trial_concurrency is not None:
            merged = replace(merged, trial_concurrency=config.trial_concurrency)
            sources["trial_concurrency"] = source
        if config.example_concurrency is not None:
            merged = replace(merged, example_concurrency=config.example_concurrency)
            sources["example_concurrency"] = source
        if config.thread_workers is not None:
            merged = replace(merged, thread_workers=config.thread_workers)
            sources["thread_workers"] = source

    return merged, sources


def _validate_parallel_config_inputs(
    config: ParallelConfig,
    thread_workers: int,
    validation: ValidationResult,
) -> None:
    """Validate input values for parallel configuration."""
    if config.mode not in (None, "auto", "sequential", "parallel"):
        validation.add_error(
            "parallel_config.mode",
            f"Unsupported mode '{config.mode}'. Expected 'sequential', 'parallel', or 'auto'.",
        )

    if thread_workers < 1:
        validation.add_error(
            "parallel_config.thread_workers",
            "thread_workers must be >= 1.",
        )

    if config.example_concurrency is not None and config.example_concurrency < 1:
        validation.add_error(
            "parallel_config.example_concurrency",
            "example_concurrency must be >= 1 when specified.",
        )
    if config.trial_concurrency is not None and config.trial_concurrency < 1:
        validation.add_error(
            "parallel_config.trial_concurrency",
            "trial_concurrency must be >= 1 when specified.",
        )


def _infer_resolved_mode(
    mode: ParallelMode | None,
    trial_concurrency: int | None,
    example_concurrency: int | None,
) -> Literal["sequential", "parallel"]:
    """Infer the resolved execution mode from config and concurrency values."""
    effective_mode = mode or "auto"
    if effective_mode == "auto":
        if (trial_concurrency or 0) > 1 or (example_concurrency or 0) > 1:
            return "parallel"
        return "sequential"
    return "parallel" if effective_mode == "parallel" else "sequential"


def _resolve_parallel_concurrency(
    config: ParallelConfig,
    thread_workers: int,
    config_space_size: int,
    sources: dict[str, str],
    validation: ValidationResult,
) -> tuple[int, int, int]:
    """Resolve concurrency values for parallel mode.

    Returns:
        tuple of (trial_concurrency, example_concurrency, thread_workers)
    """
    explicit_thread_workers = config.thread_workers is not None
    explicit_example_concurrency = config.example_concurrency is not None
    trial_concurrency = config.trial_concurrency
    example_concurrency = config.example_concurrency

    # Fill in defaults for missing values
    if trial_concurrency is None:
        inferred = min(
            max(thread_workers, 2),
            config_space_size or thread_workers or 2,
        )
        trial_concurrency = inferred

    if trial_concurrency is not None and not explicit_thread_workers:
        origin = sources.get("trial_concurrency")
        if origin != "runtime":
            thread_workers = max(thread_workers, trial_concurrency)

    if example_concurrency is None:
        example_concurrency = max(thread_workers, 1)
    elif (
        not explicit_example_concurrency
        and trial_concurrency is not None
        and example_concurrency < trial_concurrency
    ):
        example_concurrency = min(trial_concurrency, thread_workers)

    if trial_concurrency <= 1:
        validation.add_error(
            "parallel_config.trial_concurrency",
            "trial_concurrency must be greater than 1 in parallel mode.",
            suggestions=[
                "Lower mode to 'sequential', or increase parallel_config.trial_concurrency."
            ],
        )

    return trial_concurrency, example_concurrency, thread_workers


def _generate_parallel_warnings(
    resolved_mode: Literal["sequential", "parallel"],
    detected_function_kind: Literal["sync", "async"],
    trial_concurrency: int,
    example_concurrency: int,
    default_thread_workers: int,
) -> list[str]:
    """Generate warnings for parallel execution configuration."""
    warnings: list[str] = []

    if detected_function_kind == "async" and resolved_mode == "parallel":
        warnings.append(
            "Async optimized function detected. Ensure your implementation avoids blocking calls "
            "(e.g., replace time.sleep with await asyncio.sleep) to benefit from parallel execution."
        )

    if (
        resolved_mode == "parallel"
        and trial_concurrency * example_concurrency > max(default_thread_workers, 1) * 4
    ):
        warnings.append(
            "Requested concurrency is significantly higher than the available worker pool."
            " External providers may throttle or increase per-example latency; consider lowering"
            " trial_concurrency/example_concurrency or increasing thread_workers cautiously."
        )

    return warnings


def resolve_parallel_config(
    config: ParallelConfig,
    *,
    default_thread_workers: int,
    config_space_size: int,
    detected_function_kind: Literal["sync", "async"],
    sources: dict[str, str] | None = None,
) -> ParallelConfigResolved:
    """Resolve unspecified fields, enforcing defaults and validation."""
    validation = ValidationResult()
    sources = dict(sources or {})

    thread_workers = config.thread_workers or default_thread_workers

    # Validate inputs
    _validate_parallel_config_inputs(config, thread_workers, validation)

    # Infer the resolved mode
    resolved_mode = _infer_resolved_mode(
        config.mode, config.trial_concurrency, config.example_concurrency
    )

    # Resolve concurrency values based on mode
    if resolved_mode == "sequential":
        trial_concurrency = 1
        example_concurrency = 1
    else:
        trial_concurrency, example_concurrency, thread_workers = (
            _resolve_parallel_concurrency(
                config, thread_workers, config_space_size, sources, validation
            )
        )

    # Generate warnings
    warnings = _generate_parallel_warnings(
        resolved_mode,
        detected_function_kind,
        trial_concurrency or 1,
        example_concurrency or 1,
        default_thread_workers,
    )

    validation.raise_if_invalid()

    return ParallelConfigResolved(
        mode=resolved_mode,
        trial_concurrency=trial_concurrency or 1,
        example_concurrency=example_concurrency or 1,
        thread_workers=thread_workers or 1,
        sources=sources or {},
        warnings=warnings,
    )
