"""Tests for traigent_validation plugin registry."""

from __future__ import annotations

import logging
from typing import Any

import pytest

import traigent_validation.plugins as plugins
from traigent_validation.validators import (
    PythonConstraintValidator,
    SATConstraintValidator,
)


@pytest.fixture(autouse=True)
def _reset_plugin_registry() -> Any:
    """Isolate plugin registry state between tests."""
    plugins._reset_registry_for_testing()
    yield
    plugins._reset_registry_for_testing()


def test_builtin_python_validator_registered() -> None:
    """Built-in python validator is discoverable and instantiable."""
    validator = plugins.get_validator("python")
    sat_validator = plugins.get_validator("sat")

    assert validator is not None
    assert sat_validator is not None
    assert isinstance(validator, PythonConstraintValidator)
    assert isinstance(sat_validator, SATConstraintValidator)
    assert "python" in plugins.list_validators()
    assert "sat" in plugins.list_validators()


def test_custom_validator_registration() -> None:
    """Custom validators can be registered via plugin registry API."""

    class _CustomValidator:
        pass

    plugins.register_validator("unit-test-custom", _CustomValidator, overwrite=True)
    validator = plugins.get_validator("unit-test-custom")

    assert validator is not None
    assert validator.__class__.__name__ == "_CustomValidator"


def test_register_validator_without_overwrite_rejected() -> None:
    """Duplicate validator names require overwrite=True."""

    class _CustomValidator:
        pass

    plugins.register_validator("duplicate", _CustomValidator, overwrite=True)
    with pytest.raises(ValueError, match="already registered"):
        plugins.register_validator("duplicate", _CustomValidator, overwrite=False)


def test_get_validator_returns_none_when_factory_raises(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Factory construction errors should be logged and return None."""

    def _broken_factory() -> Any:
        raise RuntimeError("boom")

    plugins.register_validator("broken", _broken_factory, overwrite=True)
    with caplog.at_level(logging.WARNING):
        validator = plugins.get_validator("broken")

    assert validator is None
    assert "Failed to instantiate validator 'broken'" in caplog.text


def test_coerce_factory_keeps_callable_identity() -> None:
    """Callable entry points should be used directly as factories."""

    def _factory() -> object:
        return object()

    assert plugins._coerce_factory(_factory) is _factory


def test_load_entry_points_supports_legacy_metadata_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Legacy importlib.metadata API path should still register entry points."""

    class _LoadedValidator:
        pass

    class _EntryPoint:
        def __init__(self, name: str) -> None:
            self.name = name

        def load(self) -> type[_LoadedValidator]:
            return _LoadedValidator

    def _entry_points(*args: Any, **kwargs: Any) -> Any:
        if kwargs:
            raise TypeError("legacy api shape")
        return {plugins.ENTRY_POINT_GROUP: [_EntryPoint("legacy-ep")]}

    monkeypatch.setattr(plugins.importlib.metadata, "entry_points", _entry_points)
    plugins.load_entry_point_validators()

    validator = plugins.get_validator("legacy-ep")
    assert validator is not None
    assert isinstance(validator, _LoadedValidator)


def test_load_entry_points_ignores_non_callable_payload(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Non-callable entry-point payloads should be ignored with a warning."""

    class _EntryPoint:
        name = "bad-entry-point"

        @staticmethod
        def load() -> int:
            return 123

    def _entry_points(*args: Any, **kwargs: Any) -> list[_EntryPoint]:
        return [_EntryPoint()]

    monkeypatch.setattr(plugins.importlib.metadata, "entry_points", _entry_points)

    with caplog.at_level(logging.WARNING):
        plugins.load_entry_point_validators()

    assert "bad-entry-point" not in plugins.list_validators()
    assert "not callable" in caplog.text


def test_load_entry_points_handles_entry_point_load_error(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Failing entry-point loads should be logged and skipped."""

    class _EntryPoint:
        name = "broken-entry-point"

        @staticmethod
        def load() -> Any:
            raise RuntimeError("cannot import plugin")

    def _entry_points(*args: Any, **kwargs: Any) -> list[_EntryPoint]:
        return [_EntryPoint()]

    monkeypatch.setattr(plugins.importlib.metadata, "entry_points", _entry_points)

    with caplog.at_level(logging.WARNING):
        plugins.load_entry_point_validators()

    assert "broken-entry-point" not in plugins.list_validators()
    assert "Failed to load validator entry point" in caplog.text


def test_load_entry_points_handles_discovery_error(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Metadata discovery failures should not crash validator registration."""

    def _entry_points(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("metadata failure")

    monkeypatch.setattr(plugins.importlib.metadata, "entry_points", _entry_points)

    with caplog.at_level(logging.WARNING):
        plugins.load_entry_point_validators()

    assert "Failed to inspect validator entry points" in caplog.text
    assert isinstance(plugins.get_validator("python"), PythonConstraintValidator)


def test_reset_registry_for_testing_clears_custom_plugins() -> None:
    """Test helper should remove custom registration state."""

    class _CustomValidator:
        pass

    plugins.register_validator("temporary", _CustomValidator, overwrite=True)
    assert "temporary" in plugins.list_validators()

    plugins._reset_registry_for_testing()
    assert "temporary" not in plugins.list_validators()
