import pytest

from traigent.config.parallel import (
    ParallelConfig,
    coerce_parallel_config,
    merge_parallel_configs,
    resolve_parallel_config,
)
from traigent.utils.exceptions import ValidationError as ValidationException


def test_merge_parallel_configs_precedence() -> None:
    global_cfg = ParallelConfig(
        mode="sequential", trial_concurrency=1, thread_workers=2
    )
    decorator_cfg = ParallelConfig(mode="parallel", trial_concurrency=3)
    runtime_cfg = ParallelConfig(example_concurrency=5)

    merged, sources = merge_parallel_configs(
        [
            (global_cfg, "global"),
            (decorator_cfg, "decorator"),
            (runtime_cfg, "runtime"),
        ]
    )

    assert merged.mode == "parallel"
    assert merged.trial_concurrency == 3
    assert merged.example_concurrency == 5
    assert merged.thread_workers == 2
    # Runtime override should win for example concurrency, decorator for mode/trials.
    assert sources["mode"] == "decorator"
    assert sources["trial_concurrency"] == "decorator"
    assert sources["example_concurrency"] == "runtime"


def test_resolve_auto_parallel_defaults() -> None:
    cfg = ParallelConfig(mode="auto", trial_concurrency=4)
    resolved = resolve_parallel_config(
        cfg,
        default_thread_workers=2,
        config_space_size=10,
        detected_function_kind="sync",
        sources={"trial_concurrency": "runtime"},
    )

    assert resolved.mode == "parallel"
    assert resolved.trial_concurrency == 4
    # example_concurrency should default to max(thread_workers, 1)
    assert resolved.example_concurrency == 2
    assert resolved.thread_workers == 2
    assert not resolved.warnings


def test_resolve_sequential_forces_single_concurrency() -> None:
    cfg = ParallelConfig(mode="sequential", trial_concurrency=5, example_concurrency=7)
    resolved = resolve_parallel_config(
        cfg,
        default_thread_workers=4,
        config_space_size=0,
        detected_function_kind="sync",
    )

    assert resolved.mode == "sequential"
    assert resolved.trial_concurrency == 1
    assert resolved.example_concurrency == 1
    assert resolved.thread_workers == 4


def test_resolve_parallel_requires_trial_concurrency_gt_one() -> None:
    cfg = ParallelConfig(mode="parallel", trial_concurrency=1, example_concurrency=2)

    with pytest.raises(ValidationException) as exc:
        resolve_parallel_config(
            cfg,
            default_thread_workers=4,
            config_space_size=10,
            detected_function_kind="sync",
        )

    assert "trial_concurrency must be greater than 1" in str(exc.value)


def test_async_warning_emitted_for_parallel_mode() -> None:
    cfg = ParallelConfig(mode="parallel", trial_concurrency=3, example_concurrency=2)
    resolved = resolve_parallel_config(
        cfg,
        default_thread_workers=4,
        config_space_size=10,
        detected_function_kind="async",
    )

    assert resolved.mode == "parallel"
    assert any("Async optimized function" in warning for warning in resolved.warnings)


def test_high_concurrency_warning() -> None:
    cfg = ParallelConfig(
        mode="parallel",
        trial_concurrency=6,
        example_concurrency=6,
        thread_workers=2,
    )
    resolved = resolve_parallel_config(
        cfg,
        default_thread_workers=2,
        config_space_size=100,
        detected_function_kind="sync",
    )

    assert resolved.mode == "parallel"
    assert resolved.trial_concurrency == 6
    assert resolved.example_concurrency == 6
    assert any(
        "Requested concurrency is significantly higher" in warning
        for warning in resolved.warnings
    )


def test_coerce_parallel_config_from_dict() -> None:
    raw = {
        "mode": "parallel",
        "trial_concurrency": 4,
        "example_concurrency": 3,
        "thread_workers": 2,
    }
    cfg = coerce_parallel_config(raw)
    assert isinstance(cfg, ParallelConfig)
    assert cfg.mode == "parallel"
    assert cfg.trial_concurrency == 4
    assert cfg.example_concurrency == 3
    assert cfg.thread_workers == 2


def test_from_dict_rejects_unknown_keys() -> None:
    """Test that ParallelConfig.from_dict() rejects unknown keys."""
    raw = {
        "mode": "parallel",
        "trial_concurrency": 4,
        "parallel_trials": 2,  # Legacy key - should be rejected
    }
    with pytest.raises(ValueError) as exc:
        ParallelConfig.from_dict(raw)

    assert "Unknown parallel_config key(s)" in str(exc.value)
    assert "parallel_trials" in str(exc.value)
    assert "trial_concurrency" in str(exc.value)  # Valid keys listed


def test_from_dict_rejects_multiple_unknown_keys() -> None:
    """Test that ParallelConfig.from_dict() lists all unknown keys."""
    raw = {
        "parallel_trials": 2,
        "batch_size": 4,
        "max_workers": 8,
    }
    with pytest.raises(ValueError) as exc:
        ParallelConfig.from_dict(raw)

    error_msg = str(exc.value)
    assert "batch_size" in error_msg
    assert "max_workers" in error_msg
    assert "parallel_trials" in error_msg
