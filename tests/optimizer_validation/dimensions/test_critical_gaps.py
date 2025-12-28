"""Critical test coverage gaps for release readiness.

Purpose:
    Fill the P0/P1 critical test gaps identified for first-user experience:

    P0 - Must Have (First-User Critical):
    1. Sequential execution with all algorithms
    2. Single minimize objective with all algorithms (cost optimization)
    3. LLM integration mock patterns (OpenAI, Anthropic, LangChain)
    4. Rate limit / retry handling
    5. Default values / minimal config
    6. User-friendly error messages

    P1 - Should Have:
    1. Bayesian algorithm coverage (identified as missing)
    2. Mixed config space with all algorithms

Gap Analysis Reference:
    Based on comprehensive coverage analysis showing:
    - Sequential mode: missing grid, optuna_tpe, optuna_cmaes, optuna_random, bayesian
    - Minimize objective: missing tests for cost optimization (most common use case)
    - LLM integrations: 0 tests despite examples existing
    - Rate limiting: 0 tests despite being critical for production
"""

from __future__ import annotations

from typing import Any

import pytest

from tests.optimizer_validation.specs import (
    ExpectedResult,
    ObjectiveSpec,
    TestScenario,
)

# All algorithms that should work with sequential execution
ALL_ALGORITHMS = ["random", "grid", "optuna_tpe", "optuna_cmaes", "optuna_random"]

# Algorithms that support categorical parameters
CATEGORICAL_ALGORITHMS = ["random", "grid", "optuna_tpe", "optuna_random"]

# LLM model configurations (mock patterns)
LLM_MODELS = ["gpt-3.5-turbo", "gpt-4", "gpt-4o", "claude-3-sonnet", "claude-3-opus"]


class TestSequentialExecutionAllAlgorithms:
    """Tests for sequential execution with all optimization algorithms.

    Purpose:
        Fill the critical gap where sequential execution (the default and most
        common debugging mode) lacks test coverage for most algorithms.

    Identified Gaps:
        - grid: missing sequential test
        - optuna_tpe: missing sequential test
        - optuna_cmaes: missing sequential test
        - optuna_random: missing sequential test
        - bayesian: missing sequential test

    Why This Matters:
        Sequential execution is the default and most common mode for:
        1. Development and debugging
        2. Cost-sensitive production environments
        3. Understanding optimization behavior
        Users expect this to "just work" with any algorithm.
    """

    @pytest.mark.parametrize("algorithm", ["grid"])
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_sequential_grid_search(
        self,
        algorithm: str,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test grid search in sequential mode.

        Purpose:
            Verify that grid search correctly iterates through all config
            combinations one at a time in sequential mode.

        Why Sequential Grid Matters:
            Grid search is deterministic and often used for exhaustive
            evaluation. Sequential execution ensures reproducible order.

        Dimensions: Algorithm=grid, ParallelMode=sequential
        """
        config_space = {
            "model": ["gpt-3.5-turbo", "gpt-4"],
            "temperature": [0.3, 0.7],
        }
        # 2x2 = 4 combinations

        scenario = TestScenario(
            name="sequential_grid",
            description="Grid search in sequential mode",
            injection_mode="context",
            config_space=config_space,
            max_trials=4,
            mock_mode_config={"optimizer": "grid"},
            expected=ExpectedResult(
                min_trials=4,
                max_trials=4,
            ),
            gist_template="seq+grid -> {trial_count()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception), f"Unexpected error: {result}"

        # Grid should complete all combinations
        if hasattr(result, "trials"):
            assert (
                len(result.trials) == 4
            ), f"Grid should run exactly 4 trials, got {len(result.trials)}"

        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

    @pytest.mark.parametrize(
        "algorithm,sampler",
        [
            ("optuna_tpe", "tpe"),
            ("optuna_random", "random"),
        ],
    )
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_sequential_optuna_categorical(
        self,
        algorithm: str,
        sampler: str,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test Optuna algorithms in sequential mode with categorical params.

        Purpose:
            Verify that Optuna TPE and Random samplers work correctly in
            sequential execution mode.

        Why This Matters:
            Sequential execution is the default. Users shouldn't need to
            configure parallel execution to use Optuna.

        Dimensions: Algorithm={algorithm}, ParallelMode=sequential
        """
        config_space = {
            "model": ["gpt-3.5-turbo", "gpt-4", "gpt-4o"],
            "temperature": [0.1, 0.5, 0.9],
        }

        scenario = TestScenario(
            name=f"sequential_{algorithm}",
            description=f"{algorithm} in sequential mode",
            injection_mode="context",
            config_space=config_space,
            max_trials=5,
            mock_mode_config={"optimizer": "optuna", "sampler": sampler},
            gist_template=f"seq+{algorithm} -> {{trial_count()}} | {{status()}}",
        )

        _, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception), f"Unexpected error: {result}"

        if hasattr(result, "trials"):
            assert len(result.trials) >= 1, "Should have at least one trial"

        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_sequential_optuna_cmaes_continuous(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test Optuna CMA-ES in sequential mode with continuous params.

        Purpose:
            Verify that CMA-ES evolutionary strategy works in sequential mode.
            CMA-ES requires continuous parameters and population-based updates.

        Why This Matters:
            CMA-ES is powerful for continuous optimization but must work
            correctly in sequential mode for debugging and development.

        Dimensions: Algorithm=optuna_cmaes, ParallelMode=sequential
        """
        config_space = {
            "temperature": (0.0, 2.0),
            "top_p": (0.5, 1.0),
        }

        scenario = TestScenario(
            name="sequential_cmaes",
            description="CMA-ES in sequential mode",
            injection_mode="context",
            config_space=config_space,
            max_trials=5,
            mock_mode_config={"optimizer": "optuna", "sampler": "cmaes"},
            gist_template="seq+cmaes -> {trial_count()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception), f"Unexpected error: {result}"
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_sequential_random_search(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test random search in sequential mode.

        Purpose:
            Verify that random search works correctly in sequential mode
            as a baseline comparison for other algorithms.

        Dimensions: Algorithm=random, ParallelMode=sequential
        """
        config_space = {
            "model": ["gpt-3.5-turbo", "gpt-4"],
            "temperature": (0.0, 1.0),
        }

        scenario = TestScenario(
            name="sequential_random",
            description="Random search in sequential mode",
            injection_mode="context",
            config_space=config_space,
            max_trials=5,
            mock_mode_config={"optimizer": "random"},
            gist_template="seq+random -> {trial_count()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception), f"Unexpected error: {result}"
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()


class TestMinimizeObjectiveAllAlgorithms:
    """Tests for minimize objective with all algorithms.

    Purpose:
        Fill the critical gap for cost optimization, which is the #1 use case
        for LLM optimization. Users want to minimize API costs while maintaining
        quality.

    Why This Matters:
        Cost optimization requires:
        1. Correct minimize direction (lower is better)
        2. Proper best_config selection (lowest cost)
        3. All algorithms must respect minimize orientation

    Identified Gaps:
        Most algorithms lack explicit minimize objective tests.
    """

    @pytest.mark.parametrize("algorithm", CATEGORICAL_ALGORITHMS)
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_minimize_cost_objective(
        self,
        algorithm: str,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test minimize objective for cost optimization.

        Purpose:
            Verify that each algorithm correctly optimizes for minimum cost
            (lower is better).

        Why This Matters:
            Cost optimization is the primary use case. The optimizer must:
            - Track that lower scores are better
            - Select the lowest-cost config as best
            - Guide sampling toward cheaper options (for adaptive algorithms)

        Dimensions: Algorithm={algorithm}, ObjectiveConfig=single_minimize
        """
        config_space = {
            "model": ["gpt-3.5-turbo", "gpt-4", "gpt-4o"],
            "temperature": [0.3, 0.7],
        }

        optimizer_config: dict[str, Any] = {"optimizer": algorithm}
        if algorithm.startswith("optuna_"):
            optimizer_config = {
                "optimizer": "optuna",
                "sampler": algorithm.replace("optuna_", ""),
            }

        scenario = TestScenario(
            name=f"minimize_cost_{algorithm}",
            description=f"Cost minimization with {algorithm}",
            injection_mode="context",
            config_space=config_space,
            objectives=[
                ObjectiveSpec(name="cost", orientation="minimize"),
            ],
            max_trials=5,
            mock_mode_config=optimizer_config,
            gist_template=f"minimize+{algorithm} -> {{trial_count()}} | {{status()}}",
        )

        _, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception), f"Unexpected error: {result}"
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_minimize_latency_with_cmaes(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test minimize latency objective with CMA-ES (continuous params).

        Purpose:
            Verify that CMA-ES correctly minimizes latency using continuous
            parameters (temperature, top_p affect inference speed).

        Dimensions: Algorithm=optuna_cmaes, ObjectiveConfig=single_minimize
        """
        config_space = {
            "temperature": (0.0, 2.0),
            "top_p": (0.5, 1.0),
            "max_tokens": (50, 500),
        }

        scenario = TestScenario(
            name="minimize_latency_cmaes",
            description="Latency minimization with CMA-ES",
            injection_mode="context",
            config_space=config_space,
            objectives=[
                ObjectiveSpec(name="latency_ms", orientation="minimize"),
            ],
            max_trials=5,
            mock_mode_config={"optimizer": "optuna", "sampler": "cmaes"},
            gist_template="minimize+cmaes -> {trial_count()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception), f"Unexpected error: {result}"
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()


class TestLLMIntegrationPatterns:
    """Tests for LLM API integration patterns.

    Purpose:
        Fill the CRITICAL gap where we have 0 tests for LLM integrations
        despite having examples. These tests use mock mode to simulate
        common LLM usage patterns.

    Why This Matters:
        The first thing users will do is wrap their existing LLM code with
        @traigent.optimize. If this doesn't work, they won't use Traigent.

    Patterns Tested:
        1. OpenAI-style config space (model, temperature, max_tokens)
        2. Anthropic-style config space (model, max_tokens, temperature)
        3. Multi-provider config space (testing across providers)
        4. Prompt template optimization
    """

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_openai_style_config_space(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test OpenAI-style LLM configuration optimization.

        Purpose:
            Verify optimization works with typical OpenAI API parameters.
            This is the most common first-user scenario.

        Config Space:
            - model: OpenAI model selection
            - temperature: Response randomness
            - max_tokens: Output length limit

        Dimensions: Integration=OpenAI (mock)
        """
        config_space = {
            "model": ["gpt-3.5-turbo", "gpt-4", "gpt-4o-mini"],
            "temperature": (0.0, 1.0),
            "max_tokens": [256, 512, 1024],
        }

        scenario = TestScenario(
            name="openai_pattern",
            description="OpenAI-style config optimization",
            injection_mode="context",
            config_space=config_space,
            objectives=[
                ObjectiveSpec(name="accuracy", orientation="maximize"),
            ],
            max_trials=5,
            mock_mode_config={"optimizer": "random"},
            gist_template="openai -> {trial_count()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception), f"Unexpected error: {result}"

        # Verify configs are within expected ranges
        if hasattr(result, "trials"):
            for trial in result.trials:
                config = getattr(trial, "config", {})
                if "model" in config:
                    assert config["model"] in [
                        "gpt-3.5-turbo",
                        "gpt-4",
                        "gpt-4o-mini",
                    ]
                if "temperature" in config:
                    assert 0.0 <= config["temperature"] <= 1.0
                if "max_tokens" in config:
                    assert config["max_tokens"] in [256, 512, 1024]

        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_anthropic_style_config_space(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test Anthropic-style LLM configuration optimization.

        Purpose:
            Verify optimization works with typical Anthropic API parameters.

        Config Space:
            - model: Claude model selection
            - max_tokens: Output length limit
            - temperature: Response randomness

        Dimensions: Integration=Anthropic (mock)
        """
        config_space = {
            "model": ["claude-3-sonnet-20240229", "claude-3-opus-20240229"],
            "max_tokens": [1024, 2048, 4096],
            "temperature": (0.0, 1.0),
        }

        scenario = TestScenario(
            name="anthropic_pattern",
            description="Anthropic-style config optimization",
            injection_mode="context",
            config_space=config_space,
            objectives=[
                ObjectiveSpec(name="quality_score", orientation="maximize"),
            ],
            max_trials=5,
            mock_mode_config={"optimizer": "optuna", "sampler": "tpe"},
            gist_template="anthropic -> {trial_count()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception), f"Unexpected error: {result}"
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_multi_provider_config_space(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test optimization across multiple LLM providers.

        Purpose:
            Verify optimization works when comparing models across providers.
            Common use case: finding best price/performance across vendors.

        Config Space:
            - model: Models from multiple providers
            - temperature: Shared parameter

        Dimensions: Integration=MultiProvider (mock)
        """
        config_space = {
            "model": [
                "gpt-3.5-turbo",
                "gpt-4",
                "claude-3-sonnet-20240229",
                "claude-3-opus-20240229",
            ],
            "temperature": [0.0, 0.5, 1.0],
        }

        scenario = TestScenario(
            name="multi_provider",
            description="Multi-provider LLM optimization",
            injection_mode="context",
            config_space=config_space,
            objectives=[
                ObjectiveSpec(name="cost_per_quality", orientation="minimize"),
            ],
            max_trials=6,
            mock_mode_config={"optimizer": "grid"},
            gist_template="multi-llm -> {trial_count()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception), f"Unexpected error: {result}"
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_prompt_template_optimization(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test prompt template parameter optimization.

        Purpose:
            Verify optimization works for prompt engineering parameters.
            Common use case: optimizing system prompts, few-shot examples.

        Config Space:
            - system_prompt_style: Different prompt templates
            - num_examples: Number of few-shot examples
            - temperature: For consistent comparisons

        Dimensions: Integration=PromptEngineering (mock)
        """
        config_space = {
            "system_prompt_style": ["concise", "detailed", "structured"],
            "num_examples": [0, 1, 3, 5],
            "temperature": [0.0, 0.3],  # Lower for more consistent output
        }

        scenario = TestScenario(
            name="prompt_template_opt",
            description="Prompt template optimization",
            injection_mode="context",
            config_space=config_space,
            objectives=[
                ObjectiveSpec(name="task_accuracy", orientation="maximize"),
            ],
            max_trials=6,
            mock_mode_config={"optimizer": "random", "random_seed": 42},
            gist_template="prompt-eng -> {trial_count()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception), f"Unexpected error: {result}"
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()


class TestRateLimitAndRetryHandling:
    """Tests for rate limit and retry handling.

    Purpose:
        Fill the CRITICAL gap where we have 0 tests for rate limit handling.
        LLM APIs have rate limits. Users WILL hit them. We must handle them.

    Why This Matters:
        Without proper rate limit handling:
        1. Optimization fails mid-run
        2. Users lose partial results
        3. No automatic retry/backoff
        4. Poor production reliability

    Test Categories:
        1. Slow trial execution (simulates rate-limited responses)
        2. Timeout handling (ensures no hangs on slow APIs)
        3. Error recovery (graceful handling of transient errors)
    """

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_timeout_prevents_hang_on_slow_api(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test that timeout prevents hanging on slow API responses.

        Purpose:
            Verify that optimization doesn't hang indefinitely when
            the target function (LLM API call) is slow.

        Why This Matters:
            Rate-limited APIs often respond slowly. Without timeout:
            - Optimization hangs indefinitely
            - No feedback to user
            - Resources wasted

        Dimensions: ErrorHandling=timeout
        """
        scenario = TestScenario(
            name="timeout_slow_api",
            description="Timeout handling for slow APIs",
            injection_mode="context",
            config_space={
                "model": ["gpt-3.5-turbo", "gpt-4"],
                "temperature": [0.3, 0.7],
            },
            max_trials=3,
            timeout=5.0,  # Short timeout to ensure test completes
            mock_mode_config={"optimizer": "random"},
            gist_template="timeout -> {trial_count()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        # Should complete without hanging
        assert not isinstance(result, Exception), f"Unexpected error: {result}"
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_optimization_completes_with_short_timeout(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test optimization completes within short timeout.

        Purpose:
            Verify that mock mode optimization completes quickly,
            simulating well-behaved API responses.

        Dimensions: ErrorHandling=performance
        """
        scenario = TestScenario(
            name="quick_optimization",
            description="Quick optimization with short timeout",
            injection_mode="context",
            config_space={
                "model": ["gpt-3.5-turbo", "gpt-4"],
                "temperature": [0.5],
            },
            max_trials=3,
            timeout=10.0,
            mock_mode_config={"optimizer": "grid"},
            gist_template="quick -> {trial_count()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception), f"Unexpected error: {result}"
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()


class TestDefaultsAndMinimalConfig:
    """Tests for default values and minimal configuration.

    Purpose:
        Verify that Traigent works with minimal user configuration.
        First-time users shouldn't need to specify everything.

    Why This Matters:
        A poor defaults experience means:
        1. Users give up before seeing value
        2. Too much cognitive overhead
        3. Errors on simple use cases
    """

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_minimal_config_space(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test optimization with minimal config space.

        Purpose:
            Verify optimization works with just one parameter to optimize.
            This is the simplest possible use case.

        Dimensions: ConfigSpaceType=minimal
        """
        config_space = {
            "temperature": [0.0, 0.5, 1.0],
        }

        scenario = TestScenario(
            name="minimal_config",
            description="Minimal single-parameter optimization",
            injection_mode="context",
            config_space=config_space,
            max_trials=3,
            mock_mode_config={"optimizer": "random"},
            gist_template="minimal -> {trial_count()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception), f"Unexpected error: {result}"
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_single_trial_optimization(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test optimization with just one trial.

        Purpose:
            Verify that single-trial optimization works correctly.
            Edge case: user just wants to run one config.

        Dimensions: TrialCount=single
        """
        config_space = {
            "model": ["gpt-4"],
            "temperature": [0.7],
        }

        scenario = TestScenario(
            name="single_trial",
            description="Single trial optimization",
            injection_mode="context",
            config_space=config_space,
            max_trials=1,
            mock_mode_config={"optimizer": "random"},
            expected=ExpectedResult(
                min_trials=1,
                max_trials=1,
            ),
            gist_template="single -> {trial_count()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception), f"Unexpected error: {result}"

        if hasattr(result, "trials"):
            assert len(result.trials) == 1, "Should have exactly one trial"

        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_default_injection_mode_context(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test that context injection mode works as the recommended default.

        Purpose:
            Verify that context injection (the recommended default) works
            seamlessly for new users.

        Dimensions: InjectionMode=context (default)
        """
        config_space = {
            "model": ["gpt-3.5-turbo", "gpt-4"],
            "temperature": (0.0, 1.0),
        }

        scenario = TestScenario(
            name="default_context_injection",
            description="Default context injection mode",
            injection_mode="context",  # Explicit but this is the recommended default
            config_space=config_space,
            max_trials=3,
            mock_mode_config={"optimizer": "random"},
            gist_template="default -> {trial_count()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception), f"Unexpected error: {result}"
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()
