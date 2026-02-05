"""Tests for the SeamlessOptunaAdapter."""

from __future__ import annotations

import asyncio

import pytest

from traigent.config.context import TrialContext, get_trial_context
from traigent.config.seamless_optuna_adapter import (
    SeamlessOptunaAdapter,
    TypedConfigInjector,
    TypeValidationError,
)


def test_adapter_injects_configuration_and_context():
    adapter = SeamlessOptunaAdapter()
    trial_config = {
        "_optuna_trial_id": 42,
        "model": "gpt-4",
        "temperature": 0.2,
    }

    captured: dict[str, float] = {}

    def target(model: str, temperature: float) -> float:
        ctx = get_trial_context()
        assert ctx and ctx["trial_id"] == 42
        captured[model] = temperature
        return temperature

    wrapped = adapter.inject(target, trial_config)
    result = wrapped()

    assert result == 0.2
    assert captured == {"gpt-4": 0.2}


def test_adapter_does_not_mutate_trial_config():
    adapter = SeamlessOptunaAdapter()
    trial_config = {
        "_optuna_trial_id": 99,
        "model": "gpt-4o",
        "temperature": 0.5,
    }
    snapshot_before = dict(trial_config)

    def target(model: str, temperature: float) -> float:
        assert model == "gpt-4o"
        assert temperature == 0.5
        return temperature

    wrapped = adapter.inject(target, trial_config)
    wrapped()

    assert trial_config == snapshot_before


@pytest.mark.asyncio
async def test_adapter_supports_async_functions():
    adapter = SeamlessOptunaAdapter()
    trial_config = {
        "_optuna_trial_id": "async-1",
        "model": "claude",
        "temperature": 0.6,
    }

    async def async_target(model: str, temperature: float) -> str:
        ctx = get_trial_context()
        assert ctx and ctx["trial_id"] == "async-1"
        await asyncio.sleep(0)
        return f"{model}:{temperature}"

    wrapped = adapter.inject(async_target, trial_config)
    result = await wrapped()
    assert result == "claude:0.6"


@pytest.mark.asyncio
async def test_adapter_is_safe_for_parallel_invocations():
    adapter = SeamlessOptunaAdapter()
    trial_config = {
        "_optuna_trial_id": "parallel",
        "param": 1,
    }

    async def async_target(param: int) -> int:
        ctx = get_trial_context()
        assert ctx and ctx["trial_id"] == "parallel"
        snapshot = ctx["config_snapshot"]
        snapshot["param"] += 1
        await asyncio.sleep(0)
        return snapshot["param"]

    wrapped = adapter.inject(async_target, trial_config)
    results = await asyncio.gather(wrapped(), wrapped())

    assert results == [2, 2]
    assert trial_config["param"] == 1


def test_adapter_telemetry_hook_invoked():
    events: list[dict[str, object]] = []

    def telemetry(payload: dict[str, object]) -> None:
        events.append(payload)

    adapter = SeamlessOptunaAdapter(telemetry_hook=telemetry)
    trial_config = {
        "_optuna_trial_id": 7,
        "model": "cohere",
    }

    def target(model: str) -> str:
        return model

    wrapped = adapter.inject(target, trial_config)
    wrapped()

    assert events[0]["event"] == "trial_call_started"
    assert events[-1]["event"] == "trial_call_completed"
    assert all(event["trial_id"] == 7 for event in events)


def test_adapter_telemetry_hook_errors_are_suppressed(caplog):
    def telemetry(_: dict[str, object]) -> None:
        raise RuntimeError("telemetry boom")

    adapter = SeamlessOptunaAdapter(telemetry_hook=telemetry)
    trial_config = {"_optuna_trial_id": 8}

    def target() -> str:
        return "ok"

    wrapped = adapter.inject(target, trial_config)
    with caplog.at_level("WARNING"):
        assert wrapped() == "ok"
    assert any("telemetry boom" in record.message for record in caplog.records)


def test_adapter_requires_trial_id():
    adapter = SeamlessOptunaAdapter()
    with pytest.raises(ValueError):
        adapter.inject(lambda: None, {})


def test_trial_context_manager_manual_usage():
    payload = {"extra": "value"}
    with TrialContext(trial_id="manual", metadata=payload) as ctx:
        assert ctx["trial_id"] == "manual"
        assert ctx["extra"] == "value"
        assert get_trial_context()["trial_id"] == "manual"
    assert get_trial_context() is None


# =============================================================================
# TypedConfigInjector Tests (Issue #70)
# =============================================================================


class TestTypedConfigInjector:
    """Tests for TypedConfigInjector type validation."""

    def test_validate_basic_types(self) -> None:
        """Test validation of basic types (str, int, float)."""

        def func(name: str, count: int, temperature: float) -> str:
            return f"{name}:{count}:{temperature}"

        injector = TypedConfigInjector(func)
        errors = injector.validate_config(
            {"name": "test", "count": 5, "temperature": 0.7}
        )
        assert errors == []

    def test_validate_type_mismatch_str(self) -> None:
        """Test validation catches string type mismatch."""

        def func(name: str) -> str:
            return name

        injector = TypedConfigInjector(func)
        errors = injector.validate_config({"name": 123})
        assert len(errors) == 1
        assert "name" in errors[0]
        assert "str" in errors[0]
        assert "int" in errors[0]

    def test_validate_type_mismatch_int(self) -> None:
        """Test validation catches int type mismatch."""

        def func(count: int) -> int:
            return count

        injector = TypedConfigInjector(func)
        errors = injector.validate_config({"count": "five"})
        assert len(errors) == 1
        assert "count" in errors[0]
        assert "int" in errors[0]
        assert "str" in errors[0]

    def test_validate_int_accepted_for_float(self) -> None:
        """Test that int values are accepted for float parameters."""

        def func(temperature: float) -> float:
            return temperature

        injector = TypedConfigInjector(func)
        # int should be accepted for float
        errors = injector.validate_config({"temperature": 1})
        assert errors == []

    def test_validate_optional_type(self) -> None:
        """Test validation of Optional types."""

        def func(name: str | None) -> str:
            return name or "default"

        injector = TypedConfigInjector(func)

        # None should be valid
        errors = injector.validate_config({"name": None})
        assert errors == []

        # String should be valid
        errors = injector.validate_config({"name": "test"})
        assert errors == []

        # Int should be invalid
        errors = injector.validate_config({"name": 123})
        assert len(errors) == 1

    def test_validate_union_type(self) -> None:
        """Test validation of Union types."""

        def func(value: str | int) -> str:
            return str(value)

        injector = TypedConfigInjector(func)

        # String should be valid
        errors = injector.validate_config({"value": "test"})
        assert errors == []

        # Int should be valid
        errors = injector.validate_config({"value": 42})
        assert errors == []

        # Float should be invalid
        errors = injector.validate_config({"value": 3.14})
        assert len(errors) == 1

    def test_validate_union_type_pipe_syntax(self) -> None:
        """Test validation of Union types using Python 3.10+ pipe syntax."""

        def func(value: str | int) -> str:
            return str(value)

        injector = TypedConfigInjector(func)

        # String should be valid
        errors = injector.validate_config({"value": "test"})
        assert errors == []

        # Int should be valid
        errors = injector.validate_config({"value": 42})
        assert errors == []

        # Float should be invalid
        errors = injector.validate_config({"value": 3.14})
        assert len(errors) == 1

    def test_validate_list_type(self) -> None:
        """Test validation of list types."""

        def func(items: list[str]) -> int:
            return len(items)

        injector = TypedConfigInjector(func)

        # List should be valid
        errors = injector.validate_config({"items": ["a", "b"]})
        assert errors == []

        # String should be invalid
        errors = injector.validate_config({"items": "not a list"})
        assert len(errors) == 1

    def test_validate_dict_type(self) -> None:
        """Test validation of dict types."""

        def func(config: dict[str, int]) -> int:
            return sum(config.values())

        injector = TypedConfigInjector(func)

        # Dict should be valid
        errors = injector.validate_config({"config": {"a": 1, "b": 2}})
        assert errors == []

        # List should be invalid
        errors = injector.validate_config({"config": [1, 2, 3]})
        assert len(errors) == 1

    def test_validate_skips_params_without_type_hints(self) -> None:
        """Test validation skips parameters without type hints."""

        def func(typed: str, untyped) -> str:
            return f"{typed}:{untyped}"

        injector = TypedConfigInjector(func)
        errors = injector.validate_config({"typed": "test", "untyped": 123})
        assert errors == []

    def test_validate_strict_mode_requires_type_hints(self) -> None:
        """Test strict mode requires type hints for all params."""

        def func(typed: str, untyped) -> str:
            return f"{typed}:{untyped}"

        injector = TypedConfigInjector(func, strict=True)
        errors = injector.validate_config({"typed": "test", "untyped": 123})
        assert len(errors) == 1
        assert "untyped" in errors[0]
        assert "no type hint" in errors[0].lower()

    def test_validate_multiple_errors(self) -> None:
        """Test validation collects multiple errors."""

        def func(name: str, count: int, temperature: float) -> str:
            return f"{name}:{count}:{temperature}"

        injector = TypedConfigInjector(func)
        errors = injector.validate_config(
            {"name": 123, "count": "five", "temperature": "hot"}
        )
        assert len(errors) == 3

    def test_inject_raises_on_type_mismatch(self) -> None:
        """Test inject() raises TypeValidationError on type mismatch."""

        def func(name: str, count: int) -> str:
            return f"{name}:{count}"

        injector = TypedConfigInjector(func)

        with pytest.raises(TypeValidationError) as exc_info:
            injector.inject({"name": 123, "count": "five"})

        assert len(exc_info.value.errors) == 2
        assert exc_info.value.config == {"name": 123, "count": "five"}

    def test_inject_returns_config_on_success(self) -> None:
        """Test inject() returns config unchanged on successful validation."""

        def func(name: str, count: int) -> str:
            return f"{name}:{count}"

        injector = TypedConfigInjector(func)
        config = {"name": "test", "count": 5}
        result = injector.inject(config)

        assert result is config

    def test_handles_decorated_functions(self) -> None:
        """Test type validation works with decorated functions."""
        from functools import wraps

        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)

            return wrapper

        @decorator
        def func(name: str, count: int) -> str:
            return f"{name}:{count}"

        injector = TypedConfigInjector(func)
        errors = injector.validate_config({"name": "test", "count": 5})
        assert errors == []


class TestSeamlessOptunaAdapterTypeValidation:
    """Tests for SeamlessOptunaAdapter type validation integration."""

    def test_adapter_validates_types_by_default(self) -> None:
        """Test adapter validates types by default."""
        adapter = SeamlessOptunaAdapter()

        def target(model: str, temperature: float) -> str:
            return f"{model}:{temperature}"

        trial_config = {
            "_optuna_trial_id": 1,
            "model": 123,  # Wrong type
            "temperature": "hot",  # Wrong type
        }

        with pytest.raises(TypeValidationError) as exc_info:
            adapter.inject(target, trial_config)

        assert len(exc_info.value.errors) == 2

    def test_adapter_type_validation_can_be_disabled(self) -> None:
        """Test type validation can be disabled."""
        adapter = SeamlessOptunaAdapter(validate_types=False)

        def target(model: str, temperature: float) -> str:
            return f"{model}:{temperature}"

        trial_config = {
            "_optuna_trial_id": 1,
            "model": 123,  # Wrong type - should not raise
            "temperature": 0.5,
        }

        # Should not raise
        wrapped = adapter.inject(target, trial_config)
        # Note: calling wrapped() would fail since model=123 is wrong,
        # but inject() itself should succeed with validation disabled
        assert callable(wrapped)

    def test_adapter_strict_types_mode(self) -> None:
        """Test strict_types mode requires all params to have type hints."""
        adapter = SeamlessOptunaAdapter(validate_types=True, strict_types=True)

        def target(typed: str, untyped) -> str:
            return f"{typed}:{untyped}"

        trial_config = {
            "_optuna_trial_id": 1,
            "typed": "test",
            "untyped": 123,
        }

        with pytest.raises(TypeValidationError) as exc_info:
            adapter.inject(target, trial_config)

        assert len(exc_info.value.errors) == 1
        assert "untyped" in exc_info.value.errors[0]

    def test_adapter_caches_type_validators(self) -> None:
        """Test adapter caches TypedConfigInjector per function."""
        adapter = SeamlessOptunaAdapter()

        def target(model: str) -> str:
            return model

        trial_config = {"_optuna_trial_id": 1, "model": "gpt-4"}

        adapter.inject(target, trial_config)
        adapter.inject(target, trial_config)

        # Should have only one cached validator
        assert len(adapter._type_validators) == 1

    def test_adapter_valid_types_pass_through(self) -> None:
        """Test adapter allows valid types to pass through."""
        adapter = SeamlessOptunaAdapter()

        def target(model: str, temperature: float) -> str:
            return f"{model}:{temperature}"

        trial_config = {
            "_optuna_trial_id": 1,
            "model": "gpt-4",
            "temperature": 0.7,
        }

        wrapped = adapter.inject(target, trial_config)
        result = wrapped()

        assert result == "gpt-4:0.7"


class TestTypeValidationError:
    """Tests for TypeValidationError exception."""

    def test_error_message_format(self) -> None:
        """Test error message is properly formatted."""
        errors = ["Parameter 'name': expected str, got int"]
        config = {"name": 123}

        exc = TypeValidationError(errors, config)

        assert "Config type validation failed" in str(exc)
        assert "Parameter 'name'" in str(exc)

    def test_error_preserves_errors_and_config(self) -> None:
        """Test exception preserves errors list and config."""
        errors = ["error1", "error2"]
        config = {"a": 1, "b": 2}

        exc = TypeValidationError(errors, config)

        assert exc.errors == errors
        assert exc.config == config

    def test_error_is_subclass_of_typeerror(self) -> None:
        """Test TypeValidationError is a TypeError subclass."""
        assert issubclass(TypeValidationError, TypeError)
