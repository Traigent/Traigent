"""Characterize trial-start config formatting before extraction."""

from __future__ import annotations

from typing import Any

import pytest

from traigent.utils.callbacks import SimpleProgressCallback


class _RecordingConfig(dict[str, Any]):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.events: list[str] = []

    def get(self, key: str, default: Any = None) -> Any:
        self.events.append(f"get:{key}")
        return super().get(key, default)

    def items(self) -> Any:
        self.events.append("items")
        return super().items()


def test_trial_start_preserves_primary_precedence_iteration_and_output_order() -> None:
    observations: list[tuple[int, str]] = []
    callback: SimpleProgressCallback

    def output(message: str) -> None:
        observations.append((callback.current_trial, message))

    callback = SimpleProgressCallback(output=output)
    callback.total_trials = 6
    config = _RecordingConfig(
        {
            "model": "",
            "approach": "cot",
            "method": "iterative",
            "temperature": 0.73456,
            "other": "kept",
            "optional": None,
        }
    )

    callback.on_trial_start(3, config)

    assert config.events == ["get:model", "get:approach", "items"]
    assert observations == [(4, "[4/6] Testing: cot, temperature=0.7, other=kept")]


def test_trial_start_falsey_primaries_keep_fallback_and_short_circuit_order() -> None:
    messages: list[str] = []
    callback = SimpleProgressCallback(output=messages.append)
    config = _RecordingConfig(
        {"model": "", "approach": None, "method": 0, "optional": None}
    )

    callback.on_trial_start(0, config)

    assert config.events == ["get:model", "get:approach", "get:method", "items"]
    assert messages == ["[1/0] Testing: config"]


def test_trial_start_without_details_updates_trial_without_reading_or_output() -> None:
    class _UnexpectedConfig(dict[str, Any]):
        def get(self, key: str, default: Any = None) -> Any:
            raise AssertionError("config.get should not be called")

        def items(self) -> Any:
            raise AssertionError("config.items should not be called")

    messages: list[str] = []
    callback = SimpleProgressCallback(output=messages.append, show_details=False)

    callback.on_trial_start(7, _UnexpectedConfig())

    assert callback.current_trial == 8
    assert messages == []


def test_trial_start_conversion_error_updates_trial_before_skipping_sink() -> None:
    class _BadString:
        def __str__(self) -> str:
            raise RuntimeError("conversion failed")

    messages: list[str] = []
    callback = SimpleProgressCallback(output=messages.append)

    with pytest.raises(RuntimeError, match="conversion failed"):
        callback.on_trial_start(2, {"value": _BadString()})

    assert callback.current_trial == 3
    assert messages == []
