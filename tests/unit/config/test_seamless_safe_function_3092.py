"""Seamless safety check matches exact names / path segments, not substrings."""

import pytest

from traigent.config.providers import SeamlessParameterProvider
from traigent.utils.exceptions import ConfigurationError


def _provider() -> SeamlessParameterProvider:
    return SeamlessParameterProvider(max_cache_size=8)


def evaluate_answer(q: str) -> str:
    model = "orig"
    return model


def _renamed(name: str = "", module: str = ""):
    def fn(q: str) -> str:
        model = "orig"
        return model

    if name:
        fn.__name__ = name
    if module:
        fn.__module__ = module
    return fn


@pytest.mark.parametrize(
    "name",
    ["evaluate_answer", "executor", "compile_prompt", "evaluation_helper", "run"],
)
def test_substring_names_are_safe(name: str) -> None:
    assert _provider()._is_safe_function(_renamed(name=name))


@pytest.mark.parametrize(
    "module",
    ["cost_agent", "post_processing", "photos", "systems", "my.osmosis.agent"],
)
def test_substring_modules_are_safe(module: str) -> None:
    assert _provider()._is_safe_function(_renamed(module=module))


@pytest.mark.parametrize("name", ["exec", "eval", "compile", "__init__"])
def test_exact_dangerous_names_still_rejected(name: str) -> None:
    assert not _provider()._is_safe_function(_renamed(name=name))


@pytest.mark.parametrize(
    "module", ["os", "sys", "builtins", "subprocess", "os.path", "pkg.sys.helpers"]
)
def test_dangerous_module_segments_still_rejected(module: str) -> None:
    assert not _provider()._is_safe_function(_renamed(module=module))


def test_function_with_eval_substring_is_injected() -> None:
    wrapped = _provider().inject_config(evaluate_answer, {"model": "varied"})
    assert wrapped("q") == "varied"


def test_rejection_explains_itself() -> None:
    wrapped = _provider().inject_config(_renamed(name="eval"), {"model": "varied"})
    with pytest.raises(ConfigurationError) as excinfo:
        wrapped("q")
    message = str(excinfo.value)
    assert "reserved dangerous name" in message
    assert "'eval'" in message
