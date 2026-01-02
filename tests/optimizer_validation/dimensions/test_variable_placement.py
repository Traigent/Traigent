"""Tests for tuned variable placement and usage patterns.

Validates that tuned variables can be used in various contexts within
the optimized function, such as:
- Multiple usages in the same function
- Passed to inner functions
- Used in dictionary/object construction
- Captured in closures
- Used in conditional logic
"""

from __future__ import annotations

import pytest

import traigent
from tests.optimizer_validation.specs import TestScenario


class DummyClient:
    """Dummy client for testing object construction."""

    def __init__(self, model: str, temperature: float):
        self.model = model
        self.temperature = temperature

    def complete(self, text: str) -> str:
        return f"{self.model}:{self.temperature}:{text}"


class TestVariableMultipleUsage:
    """Test tuned variable used multiple times in same function."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_multi_use_context(self, scenario_runner, result_validator):
        def multi_use_func(text: str = ""):
            temp = traigent.get_config().get("temperature")
            result = f"Using temp={temp}"
            return {"temp_used": temp, "output": result}

        scenario = TestScenario(
            name="multi_use_context",
            description="Variable used multiple times (context)",
            injection_mode="context",
            config_space={"temperature": [0.3, 0.7]},
            custom_function=multi_use_func,
            gist_template="multi-use-ctx -> {trial_count()} | {status()}",
        )
        _, result = await scenario_runner(scenario)
        assert not isinstance(result, Exception)

        # Verify trials executed
        if hasattr(result, "trials"):
            assert len(result.trials) >= 1, "Should complete at least one trial"
            for trial in result.trials:
                config = getattr(trial, "config", {})
                assert config, "Trial should have config"

        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_multi_use_parameter(self, scenario_runner, result_validator):
        def multi_use_func(text: str = "", traigent_config=None, **kwargs):
            temp = traigent_config.temperature
            result = f"Using temp={temp}"
            return {"temp_used": temp, "output": result}

        scenario = TestScenario(
            name="multi_use_parameter",
            description="Variable used multiple times (parameter)",
            injection_mode="parameter",
            config_space={"temperature": [0.3, 0.7]},
            custom_function=multi_use_func,
            gist_template="multi-use-param -> {trial_count()} | {status()}",
        )
        _, result = await scenario_runner(scenario)
        assert not isinstance(result, Exception)

        # Verify trials executed
        if hasattr(result, "trials"):
            assert len(result.trials) >= 1, "Should complete at least one trial"
            for trial in result.trials:
                config = getattr(trial, "config", {})
                assert config, "Trial should have config"

        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_multi_use_seamless(self, scenario_runner, result_validator):
        def multi_use_func(text: str = ""):
            # temperature injected by seamless mode
            temp = temperature  # type: ignore # noqa: F821
            result = f"Using temp={temperature}"  # type: ignore # noqa: F821
            return {"temp_used": temp, "output": result}

        scenario = TestScenario(
            name="multi_use_seamless",
            description="Variable used multiple times (seamless)",
            injection_mode="seamless",
            config_space={"temperature": [0.3, 0.7]},
            custom_function=multi_use_func,
            gist_template="multi-use-seamless -> {trial_count()} | {status()}",
        )
        _, result = await scenario_runner(scenario)
        assert not isinstance(result, Exception)

        # Verify trials executed
        if hasattr(result, "trials"):
            assert len(result.trials) >= 1, "Should complete at least one trial"
            for trial in result.trials:
                config = getattr(trial, "config", {})
                assert config, "Trial should have config"

        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()


class TestVariableInFunctionCalls:
    """Test tuned variable passed to inner/nested function calls."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_inner_call_context(self, scenario_runner, result_validator):
        def inner_call_func(text: str = ""):
            def inner(m):
                return f"model={m}"

            model = traigent.get_config().get("model")
            return inner(model)

        scenario = TestScenario(
            name="inner_call_context",
            description="Pass to inner function (context)",
            injection_mode="context",
            config_space={"model": ["gpt-3.5", "gpt-4"]},
            custom_function=inner_call_func,
            gist_template="inner-call-ctx -> {trial_count()} | {status()}",
        )
        _, result = await scenario_runner(scenario)
        assert not isinstance(result, Exception)

        # Verify trials executed
        if hasattr(result, "trials"):
            assert len(result.trials) >= 1, "Should complete at least one trial"
            for trial in result.trials:
                config = getattr(trial, "config", {})
                assert config, "Trial should have config"

        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_inner_call_parameter(self, scenario_runner, result_validator):
        def inner_call_func(text: str = "", traigent_config=None, **kwargs):
            def inner(m):
                return f"model={m}"

            return inner(traigent_config.model)

        scenario = TestScenario(
            name="inner_call_parameter",
            description="Pass to inner function (parameter)",
            injection_mode="parameter",
            config_space={"model": ["gpt-3.5", "gpt-4"]},
            custom_function=inner_call_func,
            gist_template="inner-call-param -> {trial_count()} | {status()}",
        )
        _, result = await scenario_runner(scenario)
        assert not isinstance(result, Exception)

        # Verify trials executed
        if hasattr(result, "trials"):
            assert len(result.trials) >= 1, "Should complete at least one trial"
            for trial in result.trials:
                config = getattr(trial, "config", {})
                assert config, "Trial should have config"

        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_inner_call_seamless(self, scenario_runner, result_validator):
        def inner_call_func(text: str = ""):
            def inner(m):
                return f"model={m}"

            return inner(model)  # type: ignore # noqa: F821

        scenario = TestScenario(
            name="inner_call_seamless",
            description="Pass to inner function (seamless)",
            injection_mode="seamless",
            config_space={"model": ["gpt-3.5", "gpt-4"]},
            custom_function=inner_call_func,
            gist_template="inner-call-seamless -> {trial_count()} | {status()}",
        )
        _, result = await scenario_runner(scenario)
        assert not isinstance(result, Exception)

        # Verify trials executed
        if hasattr(result, "trials"):
            assert len(result.trials) >= 1, "Should complete at least one trial"
            for trial in result.trials:
                config = getattr(trial, "config", {})
                assert config, "Trial should have config"

        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()


class TestVariableInClientConstruction:
    """Test tuned variable in API client/object construction."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_client_construction_context(self, scenario_runner, result_validator):
        def client_func(text: str = ""):
            client = DummyClient(
                model=traigent.get_config().get("model"),
                temperature=traigent.get_config().get("temperature"),
            )
            return client.complete(text)

        scenario = TestScenario(
            name="client_const_context",
            description="Client construction (context)",
            injection_mode="context",
            config_space={"model": ["gpt-4"], "temperature": [0.5]},
            custom_function=client_func,
            gist_template="client-ctx -> {trial_count()} | {status()}",
        )
        _, result = await scenario_runner(scenario)
        assert not isinstance(result, Exception)

        # Verify trials executed
        if hasattr(result, "trials"):
            assert len(result.trials) >= 1, "Should complete at least one trial"
            for trial in result.trials:
                config = getattr(trial, "config", {})
                assert config, "Trial should have config"

        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_client_construction_seamless(
        self, scenario_runner, result_validator
    ):
        def client_func(text: str = ""):
            client = DummyClient(
                model=model,  # type: ignore # noqa: F821
                temperature=temperature,  # type: ignore # noqa: F821
            )
            return client.complete(text)

        scenario = TestScenario(
            name="client_const_seamless",
            description="Client construction (seamless)",
            injection_mode="seamless",
            config_space={"model": ["gpt-4"], "temperature": [0.5]},
            custom_function=client_func,
            gist_template="client-seamless -> {trial_count()} | {status()}",
        )
        _, result = await scenario_runner(scenario)
        assert not isinstance(result, Exception)

        # Verify trials executed
        if hasattr(result, "trials"):
            assert len(result.trials) >= 1, "Should complete at least one trial"
            for trial in result.trials:
                config = getattr(trial, "config", {})
                assert config, "Trial should have config"

        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()


class TestVariableInDictConstruction:
    """Test tuned variable used to build dicts/kwargs."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_dict_construction_context(self, scenario_runner, result_validator):
        def dict_func(text: str = ""):
            params = {
                "model": traigent.get_config().get("model"),
                "max_tokens": traigent.get_config().get("max_tokens"),
            }
            return str(params)

        scenario = TestScenario(
            name="dict_const_context",
            description="Dict construction (context)",
            injection_mode="context",
            config_space={"model": ["gpt-4"], "max_tokens": [100]},
            custom_function=dict_func,
            gist_template="dict-ctx -> {trial_count()} | {status()}",
        )
        _, result = await scenario_runner(scenario)
        assert not isinstance(result, Exception)

        # Verify trials executed
        if hasattr(result, "trials"):
            assert len(result.trials) >= 1, "Should complete at least one trial"
            for trial in result.trials:
                config = getattr(trial, "config", {})
                assert config, "Trial should have config"

        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_dict_construction_seamless(self, scenario_runner, result_validator):
        def dict_func(text: str = ""):
            params = {
                "model": model,  # type: ignore # noqa: F821
                "max_tokens": max_tokens,  # type: ignore # noqa: F821
            }
            return str(params)

        scenario = TestScenario(
            name="dict_const_seamless",
            description="Dict construction (seamless)",
            injection_mode="seamless",
            config_space={"model": ["gpt-4"], "max_tokens": [100]},
            custom_function=dict_func,
            gist_template="dict-seamless -> {trial_count()} | {status()}",
        )
        _, result = await scenario_runner(scenario)
        assert not isinstance(result, Exception)

        # Verify trials executed
        if hasattr(result, "trials"):
            assert len(result.trials) >= 1, "Should complete at least one trial"
            for trial in result.trials:
                config = getattr(trial, "config", {})
                assert config, "Trial should have config"

        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()


class TestVariableInClosures:
    """Test tuned variable captured by closures/callbacks."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_closure_capture_context(self, scenario_runner, result_validator):
        def closure_func(text: str = ""):
            def callback():
                return traigent.get_config().get("temperature")

            return callback()

        scenario = TestScenario(
            name="closure_context",
            description="Closure capture (context)",
            injection_mode="context",
            config_space={"temperature": [0.5]},
            custom_function=closure_func,
            gist_template="closure-ctx -> {trial_count()} | {status()}",
        )
        _, result = await scenario_runner(scenario)
        assert not isinstance(result, Exception)

        # Verify trials executed
        if hasattr(result, "trials"):
            assert len(result.trials) >= 1, "Should complete at least one trial"
            for trial in result.trials:
                config = getattr(trial, "config", {})
                assert config, "Trial should have config"

        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_closure_capture_seamless(self, scenario_runner, result_validator):
        def closure_func(text: str = ""):
            def callback():
                return temperature  # type: ignore # noqa: F821

            return callback()

        scenario = TestScenario(
            name="closure_seamless",
            description="Closure capture (seamless)",
            injection_mode="seamless",
            config_space={"temperature": [0.5]},
            custom_function=closure_func,
            gist_template="closure-seamless -> {trial_count()} | {status()}",
        )
        _, result = await scenario_runner(scenario)
        assert not isinstance(result, Exception)

        # Verify trials executed
        if hasattr(result, "trials"):
            assert len(result.trials) >= 1, "Should complete at least one trial"
            for trial in result.trials:
                config = getattr(trial, "config", {})
                assert config, "Trial should have config"

        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()


class TestVariableInConditionals:
    """Test tuned variable in if/else/match logic."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_conditional_context(self, scenario_runner, result_validator):
        def cond_func(text: str = ""):
            if traigent.get_config().get("mode") == "fast":
                return "fast_mode"
            else:
                return "slow_mode"

        scenario = TestScenario(
            name="conditional_context",
            description="Conditional logic (context)",
            injection_mode="context",
            config_space={"mode": ["fast", "slow"]},
            custom_function=cond_func,
            gist_template="cond-ctx -> {trial_count()} | {status()}",
        )
        _, result = await scenario_runner(scenario)
        assert not isinstance(result, Exception)

        # Verify trials executed
        if hasattr(result, "trials"):
            assert len(result.trials) >= 1, "Should complete at least one trial"
            for trial in result.trials:
                config = getattr(trial, "config", {})
                assert config, "Trial should have config"

        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_conditional_seamless(self, scenario_runner, result_validator):
        def cond_func(text: str = ""):
            if mode == "fast":  # type: ignore # noqa: F821
                return "fast_mode"
            else:
                return "slow_mode"

        scenario = TestScenario(
            name="conditional_seamless",
            description="Conditional logic (seamless)",
            injection_mode="seamless",
            config_space={"mode": ["fast", "slow"]},
            custom_function=cond_func,
            gist_template="cond-seamless -> {trial_count()} | {status()}",
        )
        _, result = await scenario_runner(scenario)
        assert not isinstance(result, Exception)

        # Verify trials executed
        if hasattr(result, "trials"):
            assert len(result.trials) >= 1, "Should complete at least one trial"
            for trial in result.trials:
                config = getattr(trial, "config", {})
                assert config, "Trial should have config"

        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()
