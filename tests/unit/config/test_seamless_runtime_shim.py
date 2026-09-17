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


def test_seamless_fails_closed_when_config_has_no_injectable_target(
    monkeypatch,
) -> None:
    """Issue #2298: a non-empty config space with zero injectable targets must
    raise before running the (unvaried) function, instead of silently
    completing the whole search with a phantom "best_config" (the #1451
    WARNING-only behavior this overturns)."""
    monkeypatch.delenv("TRAIGENT_ALLOW_SEAMLESS_NO_TARGETS", raising=False)
    provider = _reset_provider()

    def fn(question: str) -> str:
        model_name = "claude-3-haiku"
        return model_name

    wrapped = provider.inject_config(fn, {"model": "claude-3-sonnet"})

    with pytest.raises(ConfigurationError) as exc:
        wrapped("What is 2+2?")

    assert "no injectable target" in str(exc.value)
    assert "model" in str(exc.value)
    assert provider.get_stats()["fallback_triggers"]["no_injection"] == [["model"]]

    # C1 (cache-poisoning): the raising branch must never populate
    # `_compiled_cache`. A second and third call with the identical
    # (func, config) pair -- same cache_key -- must raise again, not
    # silently return a cached, unvaried `func` from `_lookup_cached`.
    with pytest.raises(ConfigurationError, match="no injectable target"):
        wrapped("What is 2+2?")
    with pytest.raises(ConfigurationError, match="no injectable target"):
        wrapped("What is 2+2?")
    assert provider.get_stats()["fallback_triggers"]["no_injection"] == [
        ["model"],
        ["model"],
        ["model"],
    ]


def test_seamless_no_injectable_target_opt_out_env_var(monkeypatch, caplog) -> None:
    """TRAIGENT_ALLOW_SEAMLESS_NO_TARGETS restores the pre-#2298 warning-only
    behavior for the rare intentional case."""
    monkeypatch.setenv("TRAIGENT_ALLOW_SEAMLESS_NO_TARGETS", "true")
    provider = _reset_provider()

    def fn(question: str) -> str:
        model_name = "claude-3-haiku"
        return model_name

    wrapped = provider.inject_config(fn, {"model": "claude-3-sonnet"})

    with caplog.at_level("WARNING", logger="traigent.config.providers"):
        assert wrapped("What is 2+2?") == "claude-3-haiku"

    assert "found no injectable targets" in caplog.text


def test_seamless_fallback_fails_closed_for_dangerous_name(monkeypatch) -> None:
    """I2: `_seamless_fallback` builds the runtime shim unconditionally when
    `_transform_function` raises -- here because `_is_safe_function` rejects
    a function name containing "eval". With no parameter matching the config
    key either, the shim would silently run the unvaried function and still
    count as a `runtime_shims` hit. Must raise instead, and raise again on
    repeat calls (no cache poisoning)."""
    monkeypatch.delenv("TRAIGENT_ALLOW_SEAMLESS_NO_TARGETS", raising=False)
    provider = _reset_provider()

    def evaluate_q(q: str) -> str:
        model_name = "orig"
        return model_name

    wrapped = provider.inject_config(evaluate_q, {"model": "varied"})

    for _ in range(2):
        with pytest.raises(ConfigurationError, match="no injectable target"):
            wrapped("q")

    stats = provider.get_stats()
    assert stats["runtime_shims"] == 0
    assert stats["fallback_triggers"]["no_injection"] == [["model"], ["model"]]


def test_seamless_fallback_fails_closed_for_exec_defined_function(monkeypatch) -> None:
    """I2: an exec()-defined function has no retrievable source, so
    `_transform_function` raises for a different reason than the dangerous-name
    case above; the same no-target check must still fire before the shim
    silently runs the unvaried function."""
    monkeypatch.delenv("TRAIGENT_ALLOW_SEAMLESS_NO_TARGETS", raising=False)
    provider = _reset_provider()

    ns: dict[str, object] = {}
    exec("def dyn(q):\n    model_name = 'orig'\n    return model_name\n", ns)
    wrapped = provider.inject_config(ns["dyn"], {"model": "varied"})

    with pytest.raises(ConfigurationError, match="no injectable target"):
        wrapped("q")
    with pytest.raises(ConfigurationError, match="no injectable target"):
        wrapped("q")

    assert provider.get_stats()["runtime_shims"] == 0


def test_seamless_fallback_opt_out_env_var_still_shims(monkeypatch) -> None:
    """The opt-out restores the pre-#2298 warning-only fallback shim for the
    dangerous-name/no-source case, same as the AST path's opt-out."""
    monkeypatch.setenv("TRAIGENT_ALLOW_SEAMLESS_NO_TARGETS", "true")
    provider = _reset_provider()

    def evaluate_q(q: str) -> str:
        model_name = "orig"
        return model_name

    wrapped = provider.inject_config(evaluate_q, {"model": "varied"})
    assert wrapped("q") == "orig"
    assert provider.get_stats()["runtime_shims"] == 1


def test_seamless_does_not_warn_for_assignment_or_parameter_injection(caplog) -> None:
    provider = _reset_provider()

    def assignment_fn(question: str) -> str:
        model = "claude-3-haiku"
        return model

    def parameter_fn(question: str, model: str = "claude-3-haiku") -> str:
        return model

    assignment_wrapped = provider.inject_config(
        assignment_fn, {"model": "claude-3-sonnet"}
    )
    parameter_wrapped = provider.inject_config(
        parameter_fn, {"model": "claude-3-sonnet"}
    )

    with caplog.at_level("WARNING", logger="traigent.config.providers"):
        assert assignment_wrapped("What is 2+2?") == "claude-3-sonnet"
        assert parameter_wrapped("What is 2+2?") == "claude-3-sonnet"

    assert "found no injectable targets" not in caplog.text


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


@pytest.mark.parametrize("execution_mode", ["local", "hybrid"])
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


def test_assert_injectable_raises_once_per_run_for_zero_targets(monkeypatch) -> None:
    """C2: `assert_injectable` is the once-per-run structural probe -- it
    must raise for a config with zero injectable targets WITHOUT ever
    calling the function body, and it must respect the opt-out env var like
    the per-call checks do."""
    monkeypatch.delenv("TRAIGENT_ALLOW_SEAMLESS_NO_TARGETS", raising=False)
    provider = _reset_provider()
    calls: list[str] = []

    def fn(question: str) -> str:
        calls.append(question)
        model_name = "claude-3-haiku"
        return model_name

    with pytest.raises(ConfigurationError, match="no injectable target"):
        provider.assert_injectable(fn, {"model": "claude-3-sonnet"})
    assert calls == []

    monkeypatch.setenv("TRAIGENT_ALLOW_SEAMLESS_NO_TARGETS", "true")
    # Opt-out: returns the (empty) covered set instead of raising.
    assert provider.assert_injectable(fn, {"model": "claude-3-sonnet"}) == set()
    assert calls == []


def test_assert_injectable_empty_config_is_a_noop() -> None:
    provider = _reset_provider()

    def fn(question: str) -> str:
        return question

    assert provider.assert_injectable(fn, {}) == set()


def test_assert_injectable_warns_on_partial_coverage(caplog) -> None:
    """I5: a config with SOME injectable keys and some uncovered ones must
    warn (naming the uncovered keys) rather than fail closed or stay
    silent -- partial coverage does not fail closed."""
    provider = _reset_provider()

    def fn(q: str) -> str:
        model = "orig-model"
        return f"{model}|temp-literal-0.7"

    with caplog.at_level("WARNING", logger="traigent.config.providers"):
        covered = provider.assert_injectable(
            fn, {"model": "varied", "temperature": 0.1}
        )

    assert covered == {"model"}
    assert "temperature" in caplog.text
    assert "does not fail closed" in caplog.text


def test_seamless_optimize_sync_fails_before_first_trial() -> None:
    """Issue #2298 (C2), end-to-end mock test: the issue's literal-kwarg
    repro function has zero injectable targets for the whole
    configuration_space, so the once-per-run structural check must abort
    the run -- raising before ``optimize_sync`` creates a single trial --
    instead of the pre-fix behavior of every trial completing with the
    unvaried function and the run finishing ``status: completed`` with a
    phantom ``best_config``."""
    calls: list[str] = []
    dataset = Dataset(
        examples=[
            EvaluationExample(input_data={"q": "question 0"}, expected_output="orig"),
            EvaluationExample(input_data={"q": "question 1"}, expected_output="orig"),
        ],
        name="no_injectable_target_dataset",
    )

    with pytest.raises(ConfigurationError, match="no injectable target"):

        @traigent.optimize(
            eval_dataset=dataset,
            configuration_space={"model": ["gpt-4o-mini", "gpt-4o"]},
            injection_mode="seamless",
            objectives=["accuracy"],
        )
        def seamless_literal(q: str) -> str:
            model_name = "orig"
            calls.append(model_name)
            return model_name

        # Unreachable: decoration above already raised. Left in place so the
        # test still documents (and would enforce, if the abort point ever
        # moved later) that `optimize_sync` must never create a trial.
        seamless_literal.optimize_sync(algorithm="grid", max_trials=2)  # type: ignore[name-defined]

    assert calls == [], "the function body must never execute"


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
    )
    def target_fn(question: str, model: str = "baseline") -> str:
        if model == "invalid-model-slug":
            raise RuntimeError("model_not_found: invalid-model-slug")
        return "baseline"

    result = await target_fn.optimize(algorithm="grid", max_trials=1)

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


def test_positional_only_parameters_are_injectable_not_rejected() -> None:
    """A positional-only parameter is a real injection target, so the
    fail-closed guard must not treat it as "no injectable targets".

    ``runtime_injector`` binds by name into ``BoundArguments.arguments`` and
    skips only ``VAR_POSITIONAL``/``VAR_KEYWORD``, so ``def f(model, /)`` is
    injectable. An earlier revision of this guard listed only
    POSITIONAL_OR_KEYWORD and KEYWORD_ONLY, which turned a working
    optimization into ``SeamlessNoInjectableTargetsError`` — a hard failure
    where the previous behaviour was correct.
    """

    def positional_only(model: str = "old", /) -> str:
        return model

    provider = SeamlessParameterProvider()

    assert provider.inject_config(positional_only, {"model": "new"})() == "new"
    provider.assert_injectable(positional_only, {"model": "new"})


def test_matched_param_names_covers_every_injectable_kind() -> None:
    """Keep the guard's notion of "injectable" identical to the injector's.

    The injector rejects only the two variadic kinds; anything else it can
    bind by name. Asserting on the complement makes a future narrowing of
    either list fail here instead of at a user's call site.
    """
    import inspect

    def every_kind(
        pos_only: int = 1,
        /,
        normal: int = 2,
        *args: int,
        kw_only: int = 3,
        **kwargs: int,
    ) -> None:
        return None

    signature = inspect.signature(every_kind)
    config = {"pos_only": 9, "normal": 9, "kw_only": 9, "args": 9, "kwargs": 9}

    matched = SeamlessParameterProvider()._matched_param_names(signature, config)

    assert matched == {"pos_only", "normal", "kw_only"}
