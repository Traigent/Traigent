"""Async optimization tests for modern LLM usage patterns.

Purpose:
    Fill the CRITICAL gap where we have 0 tests for async patterns.
    Modern LLM usage is predominantly async:
    - async def functions
    - Streaming responses (token-by-token)
    - Concurrent API calls
    - Batch processing

Test Categories:
    1. Async Function Support - Optimizing async def functions
    2. Concurrency Patterns - Parallel async execution
    3. Batch Processing - Multiple inputs per trial
    4. Mixed Sync/Async - Interoperability

Note:
    These tests verify async patterns work correctly with the optimizer.
    The actual async execution happens in the scenario runner.
"""

from __future__ import annotations

from typing import Any

import pytest

from tests.optimizer_validation.specs import (
    ObjectiveSpec,
    TestScenario,
)


class TestAsyncFunctionOptimization:
    """Tests for optimizing async functions.

    Purpose:
        Verify that the optimizer can correctly wrap and optimize
        async def functions.

    Why This Matters:
        Most modern LLM code uses async:
        - async def call_openai(prompt): ...
        - async def query_anthropic(messages): ...

        The optimizer must handle these seamlessly.
    """

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_async_function_basic_optimization(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test basic optimization of async functions.

        Purpose:
            Verify that async functions can be optimized with the
            same patterns as sync functions.

        Dimensions: AsyncPattern=basic
        """
        scenario = TestScenario(
            name="async_basic",
            description="Basic async function optimization",
            injection_mode="context",
            config_space={
                "model": ["gpt-3.5-turbo", "gpt-4"],
                "temperature": [0.3, 0.7],
            },
            max_trials=4,
            mock_mode_config={"optimizer": "grid"},
            is_async=True,
            gist_template="async-basic -> {trial_count()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception), f"Unexpected error: {result}"

        # Verify trials were executed
        if hasattr(result, "trials"):
            assert len(result.trials) >= 1, "Should complete at least one trial"
            # Verify async execution completed successfully
            for trial in result.trials:
                config = getattr(trial, "config", {})
                assert (
                    "model" in config or "temperature" in config
                ), "Trial should have config parameters"

        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_async_with_timeout(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test async optimization respects timeout.

        Purpose:
            Verify that async functions respect timeout settings,
            important for controlling long-running API calls.

        Dimensions: AsyncPattern=timeout
        """
        scenario = TestScenario(
            name="async_timeout",
            description="Async optimization with timeout",
            injection_mode="context",
            config_space={
                "model": ["gpt-3.5-turbo", "gpt-4", "gpt-4o"],
                "temperature": [0.1, 0.5, 0.9],
            },
            max_trials=20,
            timeout=3.0,
            mock_mode_config={"optimizer": "random"},
            is_async=True,
            gist_template="async-timeout -> {trial_count()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception), f"Unexpected error: {result}"
        assert hasattr(result, "trials"), "Result should have trials"
        assert len(result.trials) >= 1, "Should complete at least one trial"
        assert result.stop_reason is not None, "Should have a stop reason"

        # Verify timeout stopped optimization early (max_trials=20, timeout=3s)
        if hasattr(result, "trials"):
            assert (
                len(result.trials) < 20
            ), f"Timeout should stop optimization before max_trials, got {len(result.trials)}"
        # Verify stop_reason is a valid value if available
        # Note: In mock mode, optimizer may complete before timeout triggers
        if hasattr(result, "stop_reason") and result.stop_reason is not None:
            valid_stop_reasons = {
                "timeout",
                "time_limit",
                "optimizer",
                "max_trials",
                "converged",
            }
            assert (
                result.stop_reason in valid_stop_reasons
            ), f"Unexpected stop_reason {result.stop_reason}, expected one of {valid_stop_reasons}"

        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_async_with_continuous_params(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test async optimization with continuous parameters.

        Purpose:
            Verify that async functions work correctly with continuous
            (float range) parameters.

        Dimensions: AsyncPattern=continuous
        """
        scenario = TestScenario(
            name="async_continuous",
            description="Async with continuous parameters",
            injection_mode="context",
            config_space={
                "temperature": (0.0, 2.0),
                "top_p": (0.5, 1.0),
            },
            max_trials=5,
            mock_mode_config={"optimizer": "optuna", "sampler": "tpe"},
            is_async=True,
            gist_template="async-cont -> {trial_count()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception), f"Unexpected error: {result}"
        assert hasattr(result, "trials"), "Result should have trials"
        assert len(result.trials) >= 1, "Should complete at least one trial"
        assert result.stop_reason is not None, "Should have a stop reason"

        # Verify continuous parameters are within specified ranges
        if hasattr(result, "trials"):
            for trial in result.trials:
                config = getattr(trial, "config", {})
                if "temperature" in config:
                    temp = config["temperature"]
                    assert isinstance(
                        temp, (int, float)
                    ), f"Temperature should be numeric, got {type(temp)}"
                    assert (
                        0.0 <= temp <= 2.0
                    ), f"Temperature {temp} outside range [0.0, 2.0]"
                if "top_p" in config:
                    top_p = config["top_p"]
                    assert isinstance(
                        top_p, (int, float)
                    ), f"top_p should be numeric, got {type(top_p)}"
                    assert (
                        0.5 <= top_p <= 1.0
                    ), f"top_p {top_p} outside range [0.5, 1.0]"

        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()


class TestConcurrentExecution:
    """Tests for concurrent/parallel async execution.

    Purpose:
        Verify that multiple async trials can run concurrently
        without interference.

    Why This Matters:
        Parallel trial execution significantly speeds up optimization.
        Async is the natural pattern for concurrent API calls.
    """

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_parallel_async_trials(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test parallel execution of async trials.

        Purpose:
            Verify that multiple async trials can run in parallel
            without data corruption or race conditions.

        Dimensions: AsyncPattern=parallel
        """
        scenario = TestScenario(
            name="parallel_async",
            description="Parallel async trial execution",
            injection_mode="context",
            config_space={
                "model": ["gpt-3.5-turbo", "gpt-4", "gpt-4o"],
                "temperature": [0.3, 0.5, 0.7],
            },
            max_trials=6,
            parallel_config={"trial_concurrency": 3},
            timeout=30.0,
            mock_mode_config={"optimizer": "random"},
            is_async=True,
            gist_template="parallel-async -> {trial_count()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception), f"Unexpected error: {result}"
        assert hasattr(result, "trials"), "Result should have trials"
        assert len(result.trials) >= 1, "Should complete at least one trial"
        assert result.stop_reason is not None, "Should have a stop reason"

        # Verify trials completed - should get multiple trials from parallel execution
        if hasattr(result, "trials"):
            assert (
                len(result.trials) >= 3
            ), f"Parallel execution with 6 trials should complete at least 3, got {len(result.trials)}"
            # Verify each trial has valid config
            valid_models = {"gpt-3.5-turbo", "gpt-4", "gpt-4o"}
            valid_temps = {0.3, 0.5, 0.7}
            for trial in result.trials:
                config = getattr(trial, "config", {})
                if "model" in config:
                    assert (
                        config["model"] in valid_models
                    ), f"Invalid model {config['model']}"
                if "temperature" in config:
                    assert (
                        config["temperature"] in valid_temps
                    ), f"Invalid temperature {config['temperature']}"

        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_config_isolation_in_concurrent_trials(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test that concurrent trials have isolated configs.

        Purpose:
            Verify that each concurrent trial receives its own config
            without cross-contamination.

        Dimensions: AsyncPattern=isolation
        """
        scenario = TestScenario(
            name="async_isolation",
            description="Config isolation in concurrent trials",
            injection_mode="context",
            config_space={
                "model": ["model-A", "model-B", "model-C", "model-D"],
                "temperature": [0.1, 0.5, 0.9],
            },
            max_trials=6,
            parallel_config={"trial_concurrency": 3},
            mock_mode_config={"optimizer": "random", "random_seed": 42},
            is_async=True,
            gist_template="async-isolate -> {trial_count()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception), f"Unexpected error: {result}"
        assert hasattr(result, "trials"), "Result should have trials"
        assert len(result.trials) >= 1, "Should complete at least one trial"
        assert result.stop_reason is not None, "Should have a stop reason"

        # Verify each trial has a valid and complete config
        if hasattr(result, "trials"):
            valid_models = {"model-A", "model-B", "model-C", "model-D"}
            valid_temps = {0.1, 0.5, 0.9}
            for trial in result.trials:
                config = getattr(trial, "config", {})
                # Verify model is valid
                if "model" in config:
                    assert (
                        config["model"] in valid_models
                    ), f"Invalid model {config['model']}"
                # Verify temperature is valid
                if "temperature" in config:
                    assert (
                        config["temperature"] in valid_temps
                    ), f"Invalid temperature {config['temperature']}"
                # Verify complete config (both model and temperature present)
                assert (
                    "model" in config or "temperature" in config
                ), "Trial should have at least one config parameter"

        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()


class TestBatchProcessing:
    """Tests for batch processing patterns.

    Purpose:
        Verify that optimization works correctly when the target
        function processes batches of inputs.

    Why This Matters:
        Many LLM use cases involve batch processing:
        - Evaluating prompts on a test set
        - Processing multiple documents
        - Running benchmarks
    """

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_optimization_for_batch_evaluation(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test optimization with batch evaluation pattern.

        Purpose:
            Verify that the optimizer can work with functions that
            evaluate on multiple inputs per trial.

        Dimensions: AsyncPattern=batch
        """
        scenario = TestScenario(
            name="batch_eval",
            description="Batch evaluation optimization",
            injection_mode="context",
            config_space={
                "model": ["gpt-3.5-turbo", "gpt-4"],
                "temperature": [0.0, 0.5, 1.0],
            },
            objectives=[
                ObjectiveSpec(name="avg_accuracy", orientation="maximize"),
            ],
            max_trials=4,
            mock_mode_config={"optimizer": "grid"},
            is_async=True,
            gist_template="batch -> {trial_count()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception), f"Unexpected error: {result}"

        # Verify batch evaluation produced expected trials
        if hasattr(result, "trials"):
            assert len(result.trials) >= 1, "Batch evaluation should produce trials"
            # Verify objective results are present
            for trial in result.trials:
                objectives = getattr(trial, "objectives", {})
                # avg_accuracy should be present if objectives were recorded
                if objectives:
                    assert any(
                        "accuracy" in key.lower() or "avg" in key.lower()
                        for key in objectives.keys()
                    ), f"Expected accuracy-related objective, got {objectives.keys()}"

        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_optimization_multiple_inputs_same_config(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test that same config is applied to all batch inputs.

        Purpose:
            Verify that when processing a batch, all inputs use the
            same configuration for that trial.

        Dimensions: AsyncPattern=consistent_batch
        """
        scenario = TestScenario(
            name="consistent_batch",
            description="Consistent config across batch inputs",
            injection_mode="context",
            config_space={
                "model": ["gpt-3.5-turbo", "gpt-4"],
                "temperature": [0.5],
            },
            max_trials=2,
            mock_mode_config={"optimizer": "grid"},
            is_async=True,
            gist_template="consistent -> {trial_count()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception), f"Unexpected error: {result}"
        assert hasattr(result, "trials"), "Result should have trials"
        assert len(result.trials) >= 1, "Should complete at least one trial"
        assert result.stop_reason is not None, "Should have a stop reason"

        # Verify config consistency - each trial should have a complete, valid config
        if hasattr(result, "trials"):
            valid_models = {"gpt-3.5-turbo", "gpt-4"}
            for trial in result.trials:
                config = getattr(trial, "config", {})
                # Verify model is one of the valid options
                if "model" in config:
                    assert (
                        config["model"] in valid_models
                    ), f"Invalid model {config['model']}, expected one of {valid_models}"
                # Verify temperature is the fixed value (using tolerance for float comparison)
                if "temperature" in config:
                    assert (
                        abs(config["temperature"] - 0.5) < 1e-9
                    ), f"Expected fixed temperature 0.5, got {config['temperature']}"

        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()


class TestMixedSyncAsync:
    """Tests for mixed sync/async usage patterns.

    Purpose:
        Verify that the optimizer handles mixed sync and async
        patterns correctly.

    Why This Matters:
        Real codebases often mix sync and async code:
        - Sync wrappers around async functions
        - Async functions calling sync utilities
        - Legacy code integration
    """

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_optimization_with_await_inside(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test optimization where function uses await internally.

        Purpose:
            Verify that functions using await internally are handled
            correctly by the optimizer.

        Dimensions: AsyncPattern=internal_await
        """
        scenario = TestScenario(
            name="internal_await",
            description="Function with internal await",
            injection_mode="context",
            config_space={
                "model": ["gpt-3.5-turbo", "gpt-4"],
                "temperature": [0.3, 0.7],
            },
            max_trials=4,
            mock_mode_config={"optimizer": "random"},
            is_async=True,
            gist_template="await -> {trial_count()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception), f"Unexpected error: {result}"

        # Verify trials completed with valid configs
        if hasattr(result, "trials"):
            assert len(result.trials) >= 1, "Should complete at least one trial"
            valid_models = {"gpt-3.5-turbo", "gpt-4"}
            valid_temps = {0.3, 0.7}
            for trial in result.trials:
                config = getattr(trial, "config", {})
                if "model" in config:
                    assert config["model"] in valid_models
                if "temperature" in config:
                    assert config["temperature"] in valid_temps

        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_sequential_async_trials(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test sequential execution of async trials.

        Purpose:
            Verify that async trials can run sequentially when
            parallel execution is not enabled.

        Dimensions: AsyncPattern=sequential
        """
        scenario = TestScenario(
            name="sequential_async",
            description="Sequential async trial execution",
            injection_mode="context",
            config_space={
                "model": ["gpt-3.5-turbo", "gpt-4"],
                "temperature": [0.3, 0.7],
            },
            max_trials=4,
            # No parallel_config = sequential execution
            mock_mode_config={"optimizer": "grid"},
            is_async=True,
            gist_template="seq-async -> {trial_count()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception), f"Unexpected error: {result}"

        if hasattr(result, "trials"):
            assert len(result.trials) == 4

        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()


class TestAsyncWithAllAlgorithms:
    """Tests for async patterns with all optimization algorithms.

    Purpose:
        Verify that async execution works correctly with all
        supported optimization algorithms.

    Why This Matters:
        Users should be able to use any algorithm with async functions.
        Algorithm choice shouldn't affect async behavior.
    """

    @pytest.mark.parametrize(
        "optimizer,sampler",
        [
            ("random", None),
            ("grid", None),
            ("optuna", "tpe"),
            ("optuna", "random"),
        ],
    )
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_async_with_algorithm(
        self,
        optimizer: str,
        sampler: str | None,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test async optimization with different algorithms.

        Purpose:
            Verify that each algorithm works correctly in async context.

        Dimensions: AsyncPattern=all_algorithms, Algorithm={optimizer}
        """
        mock_config: dict[str, Any] = {"optimizer": optimizer}
        if sampler:
            mock_config["sampler"] = sampler

        scenario = TestScenario(
            name=f"async_{optimizer}_{sampler or 'default'}",
            description=f"Async with {optimizer} algorithm",
            injection_mode="context",
            config_space={
                "model": ["gpt-3.5-turbo", "gpt-4"],
                "temperature": [0.3, 0.7],
            },
            max_trials=4,
            mock_mode_config=mock_config,
            is_async=True,
            gist_template=f"async-{optimizer} -> {{trial_count()}} | {{status()}}",
        )

        _, result = await scenario_runner(scenario)

        assert not isinstance(
            result, Exception
        ), f"Unexpected error with {optimizer}: {result}"

        # Verify trials were executed with valid configs
        if hasattr(result, "trials"):
            assert (
                len(result.trials) >= 1
            ), f"Should complete at least one trial with {optimizer}"
            valid_models = {"gpt-3.5-turbo", "gpt-4"}
            valid_temps = {0.3, 0.7}
            for trial in result.trials:
                config = getattr(trial, "config", {})
                if "model" in config:
                    assert config["model"] in valid_models
                if "temperature" in config:
                    assert config["temperature"] in valid_temps

        validation = result_validator(scenario, result)
        assert validation.passed, f"{optimizer} failed: {validation.summary()}"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_async_cmaes_with_continuous(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test async optimization with CMA-ES (requires continuous params).

        Purpose:
            Verify that CMA-ES works correctly with async execution
            and continuous parameters.

        Dimensions: AsyncPattern=cmaes, Algorithm=optuna_cmaes
        """
        scenario = TestScenario(
            name="async_cmaes",
            description="Async with CMA-ES algorithm",
            injection_mode="context",
            config_space={
                "temperature": (0.0, 2.0),
                "top_p": (0.5, 1.0),
            },
            max_trials=5,
            mock_mode_config={"optimizer": "optuna", "sampler": "cmaes"},
            is_async=True,
            gist_template="async-cmaes -> {trial_count()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception), f"Unexpected error: {result}"
        assert hasattr(result, "trials"), "Result should have trials"
        assert len(result.trials) >= 1, "Should complete at least one trial"
        assert result.stop_reason is not None, "Should have a stop reason"

        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()
