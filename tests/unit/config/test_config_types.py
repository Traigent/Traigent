"""Validation tests for TraigentConfig assignments."""

import os
from pathlib import Path

import pytest

from traigent.config.types import ExecutionMode, TraigentConfig
from traigent.utils.exceptions import ValidationError as ValidationException


def test_setitem_validates_temperature() -> None:
    config = TraigentConfig()

    with pytest.raises(ValidationException):
        config["temperature"] = 3.0


def test_setitem_normalizes_execution_mode_privacy() -> None:
    config = TraigentConfig()
    config["execution_mode"] = "privacy"

    assert config.execution_mode == ExecutionMode.HYBRID.value
    assert config.privacy_enabled is True


def test_setitem_rejects_unknown_execution_mode() -> None:
    config = TraigentConfig()
    with pytest.raises(ValueError):
        config["execution_mode"] = "unsupported-mode"


def test_local_storage_path_expands() -> None:
    config = TraigentConfig(execution_mode=ExecutionMode.EDGE_ANALYTICS.value)
    config["local_storage_path"] = "~/tmp"

    expected = str(Path(os.path.expanduser("~/tmp")).resolve())
    assert config.local_storage_path == expected


def test_custom_params_assignment() -> None:
    config = TraigentConfig()
    config["custom_flag"] = True

    assert config.custom_params["custom_flag"] is True


def test_get_comparability_mode_falls_back_to_warn_for_invalid_value() -> None:
    config = TraigentConfig()
    object.__setattr__(config, "comparability_mode", "INVALID")

    assert config.get_comparability_mode() == "warn"


def test_setitem_rejects_unknown_comparability_mode() -> None:
    config = TraigentConfig()

    with pytest.raises(ValueError):
        config["comparability_mode"] = "unsupported"


def test_from_environment_reads_comparability_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TRAIGENT_COMPARABILITY_MODE", "strict")

    config = TraigentConfig.from_environment()

    assert config.comparability_mode == "strict"
