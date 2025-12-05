import asyncio
from concurrent.futures import ThreadPoolExecutor

import pytest

import traigent
from traigent.api.types import TrialStatus
from traigent.config.context import ConfigurationContext
from traigent.config.providers import SeamlessParameterProvider
from traigent.config.types import TraigentConfig
from traigent.evaluators.base import Dataset, EvaluationExample
from traigent.utils.exceptions import ConfigurationError


def _reset_provider() -> SeamlessParameterProvider:
    return SeamlessParameterProvider(max_cache_size=32)


def test_seamless_runtime_shim_applies_signature_default() -> None:
    provider = _reset_provider()

    def fn(question: str, model: str = "claude-3-haiku") -> str:
        return model

    wrapped = provider.inject_config(fn, {"model": "claude-3-sonnet"})
    assert wrapped("What is 2+2?") == "claude-3-sonnet"


def test_seamless_runtime_shim_respects_required_parameter() -> None:
    provider = _reset_provider()

    def fn(question: str, model: str) -> str:
        return model

    wrapped = provider.inject_config(fn, {"model": "claude-3-sonnet"})
    assert wrapped("What is 2+2?") == "claude-3-sonnet"


def test_seamless_runtime_shim_handles_keyword_only() -> None:
    provider = _reset_provider()

    def fn(question: str, *, model: str = "claude-3-haiku") -> str:
        return model

    wrapped = provider.inject_config(fn, {"model": "claude-3-sonnet"})
    assert wrapped("What is 2+2?") == "claude-3-sonnet"


def test_seamless_runtime_shim_handles_methods() -> None:
    provider = _reset_provider()

    class Agent:
        def process(self, text: str, model: str = "claude-3-haiku") -> str:
            return model

    wrapped = provider.inject_config(Agent.process, {"model": "claude-3-sonnet"})
    assert wrapped(Agent(), "hello") == "claude-3-sonnet"


@pytest.mark.asyncio
async def test_seamless_runtime_shim_handles_async() -> None:
    provider = _reset_provider()

    async def fn(question: str, model: str = "claude-3-haiku") -> str:
        await asyncio.sleep(0)
        return model

    wrapped = provider.inject_config(fn, {"model": "claude-3-sonnet"})
    assert await wrapped("What is 2+2?") == "claude-3-sonnet"


def test_seamless_runtime_shim_does_not_override_explicit_arguments() -> None:
    provider = _reset_provider()

    def fn(question: str, model: str = "claude-3-haiku") -> str:
        return model

    wrapped = provider.inject_config(fn, {"model": "claude-3-sonnet"})
    assert wrapped("What is 2+2?", model="manual-model") == "manual-model"


def test_seamless_runtime_shim_keeps_assignment_path() -> None:
    provider = _reset_provider()

    def fn(question: str) -> str:
        model = "claude-3-haiku"
        return model

    wrapped = provider.inject_config(fn, {"model": "claude-3-sonnet"})
    assert wrapped("What is 2+2?") == "claude-3-sonnet"


def test_seamless_runtime_shim_caches_per_config() -> None:
    provider = _reset_provider()

    def fn(question: str, model: str = "claude-3-haiku") -> str:
        return model

    wrapped_haiku = provider.inject_config(fn, {"model": "claude-3-haiku"})
    wrapped_sonnet = provider.inject_config(fn, {"model": "claude-3-sonnet"})

    assert wrapped_haiku("Q") == "claude-3-haiku"
    assert wrapped_sonnet("Q") == "claude-3-sonnet"
    # Reuse cached wrappers to ensure cache isolation
    assert wrapped_sonnet("Q") == "claude-3-sonnet"


def test_seamless_runtime_shim_thread_safety() -> None:
    provider = _reset_provider()

    def fn(question: str, model: str = "claude-3-haiku") -> str:
        return model

    wrapped = provider.inject_config(fn, {"model": "claude-3-sonnet"})

    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(lambda _: wrapped("Q"), range(8)))

    assert all(result == "claude-3-sonnet" for result in results)


@pytest.mark.parametrize(
    "initial_platform,config_platform,initial_model,config_model,config_temperature",
    [
        ("openai", "anthropic", "gpt-3.5-turbo", "claude-3-sonnet", 0.1),
        ("anthropic", "openai", "claude-3-haiku", "gpt-4o", 0.2),
        ("langchain", "langchain", "gpt-3.5-turbo", "gpt-4o-mini", 0.3),
    ],
)
def test_seamless_runtime_shim_multiple_parameters(
    initial_platform: str,
    config_platform: str,
    initial_model: str,
    config_model: str,
    config_temperature: float,
) -> None:
    provider = _reset_provider()

    def fn(
        question: str,
        platform: str = initial_platform,
        model: str = initial_model,
        temperature: float = 0.7,
    ) -> tuple[str, str, float]:
        return platform, model, temperature

    wrapped = provider.inject_config(
        fn,
        {
            "platform": config_platform,
            "model": config_model,
            "temperature": config_temperature,
        },
    )

    observed_platform, observed_model, observed_temperature = wrapped("Q")

    assert observed_platform == config_platform
    assert observed_model == config_model
    assert observed_temperature == config_temperature


@pytest.mark.parametrize("execution_mode", ["edge_analytics", "hybrid", "cloud"])
def test_seamless_runtime_shim_respects_configuration_context(
    execution_mode: str,
) -> None:
    provider = _reset_provider()

    def fn(question: str, model: str = "claude-3-haiku", temperature: float = 0.5):
        return model, temperature

    wrapped = provider.inject_config(
        fn,
        {
            "model": "claude-3-sonnet",
            "temperature": 0.2,
        },
    )

    # Without context, config should win
    assert wrapped("Q") == ("claude-3-sonnet", 0.2)

    # With context, context overrides the injected config
    context_config = TraigentConfig(
        execution_mode=execution_mode,
        model="context-model",
        temperature=0.9,
    )

    with ConfigurationContext(context_config):
        assert wrapped("Q") == ("context-model", 0.9)


def test_seamless_runtime_shim_positional_only_parameters() -> None:
    provider = _reset_provider()

    def fn(question, /, model: str = "claude-3-haiku"):
        return model

    wrapped = provider.inject_config(fn, {"model": "claude-3-sonnet"})
    assert wrapped("Q") == "claude-3-sonnet"


def test_seamless_runtime_shim_raises_when_fallback_fails(monkeypatch) -> None:
    provider = _reset_provider()

    def fn() -> str:
        return "value"

    def raise_transform(*args, **kwargs):  # type: ignore[unused-argument]
        raise RuntimeError("transform")

    def raise_build(*args, **kwargs):  # type: ignore[unused-argument]
        raise RuntimeError("build")

    monkeypatch.setattr(provider, "_transform_function", raise_transform)
    monkeypatch.setattr("traigent.config.providers.create_runtime_shim", raise_build)

    wrapped = provider.inject_config(fn, {"model": "claude"})

    with pytest.raises(ConfigurationError):
        wrapped()


def test_seamless_runtime_shim_invalid_configuration_raises() -> None:
    provider = _reset_provider()

    def fn(question: str, model: str = "baseline") -> str:
        if model == "override":
            raise RuntimeError("model_not_found: unsupported")
        return model

    wrapped = provider.inject_config(fn, {"model": "override"})

    with pytest.raises(ConfigurationError) as exc:
        wrapped("Q")
    assert "configuration" in str(exc.value)


@pytest.mark.asyncio
async def test_seamless_invalid_configuration_marks_trial_failed() -> None:
    dataset = Dataset(
        examples=[
            EvaluationExample(
                input_data={"question": "Any question"},
                expected_output="unused",
            )
        ],
        name="invalid_model_dataset",
        description="Dataset ensuring invalid configs fail",
    )

    @traigent.optimize(
        eval_dataset=dataset,
        configuration_space={"model": ["invalid-model-slug"]},
        injection_mode="seamless",
        algorithm="grid",
        max_trials=1,
    )
    def target_fn(question: str, model: str = "baseline") -> str:
        if model == "invalid-model-slug":
            raise RuntimeError("model_not_found: invalid-model-slug")
        return "baseline"

    result = await target_fn.optimize(max_trials=1)

    assert result.trials, "Expected at least one trial"
    trial = result.trials[0]
    assert trial.status == TrialStatus.FAILED
    assert trial.error_message is not None
    assert "Failed to execute with injected configuration" in trial.error_message


def test_seamless_runtime_shim_stats_access() -> None:
    provider = _reset_provider()

    def fn(question: str, model: str = "claude-3-haiku") -> str:
        return model

    wrapped = provider.inject_config(fn, {"model": "claude-3-sonnet"})
    assert wrapped("Q") == "claude-3-sonnet"

    stats = provider.get_stats()
    assert stats["runtime_shims"] >= 1
    assert "fallback_triggers" in stats
    assert isinstance(stats["fallback_triggers"], dict)
