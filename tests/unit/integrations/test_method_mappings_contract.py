"""Contract tests: METHOD_MAPPINGS entries must name params the method really takes."""

from __future__ import annotations

import importlib
import inspect

import pytest

from traigent.config.context import config_context, config_space_context
from traigent.integrations.framework_override import FrameworkOverrideManager
from traigent.integrations.mappings import (
    METHOD_MAPPINGS,
    METHOD_PARAMETER_TRANSLATIONS,
    PARAMETER_MAPPINGS,
)


def _resolve_method(class_path: str, method: str):
    """Resolve ``module.Class`` + simple method name; None if not importable."""
    module_name, _, cls_name = class_path.rpartition(".")
    try:
        cls = getattr(importlib.import_module(module_name), cls_name)
    except (ImportError, AttributeError):
        return None
    return getattr(cls, method, None)


def _direct_entries():
    for class_path, methods in METHOD_MAPPINGS.items():
        for method, params in methods.items():
            if "." not in method and params:
                yield class_path, method, params


@pytest.mark.unit
@pytest.mark.parametrize("class_path,method,params", list(_direct_entries()))
def test_importable_direct_methods_accept_every_listed_param(
    class_path, method, params
):
    fn = _resolve_method(class_path, method)
    if fn is None:
        pytest.skip(f"{class_path} not importable")
    sig = inspect.signature(fn)
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        pytest.skip("method takes **kwargs; signature cannot prove rejection")
    translations = METHOD_PARAMETER_TRANSLATIONS.get(class_path, {}).get(method, {})
    class_map = PARAMETER_MAPPINGS.get(class_path, {})
    framework_names = [translations.get(p, class_map.get(p, p)) for p in params]
    missing = [p for p in framework_names if p not in sig.parameters]
    assert not missing, f"{class_path}.{method} does not accept {missing}"


@pytest.mark.unit
def test_pydantic_ai_agent_has_no_top_level_kwarg_injection():
    """PydanticAI takes sampling params in model_settings, never as run() kwargs."""
    assert "pydantic_ai.Agent" not in METHOD_MAPPINGS


@pytest.mark.unit
@pytest.mark.parametrize("cls", ["cohere.Client", "cohere.AsyncClient"])
def test_cohere_method_tables_use_canonical_names(cls):
    canonical = set(PARAMETER_MAPPINGS[cls])
    for method, params in METHOD_MAPPINGS[cls].items():
        unknown = [p for p in params if p not in canonical]
        assert not unknown, f"{cls}.{method} lists non-canonical params {unknown}"


@pytest.mark.unit
@pytest.mark.parametrize("cls", ["cohere.Client", "cohere.AsyncClient"])
@pytest.mark.parametrize("method", ["generate", "chat"])
def test_cohere_top_p_top_k_are_injected_as_p_and_k(cls, method):
    mgr = FrameworkOverrideManager()

    def original(instance, *args, **kwargs):
        return kwargs

    wrapped = mgr._create_override_method(original, cls, method)
    cfg = {"top_p": 0.5, "top_k": 3, "temperature": 0.2}
    mgr._override_active.enabled = True
    t1 = config_context.set(cfg)
    t2 = config_space_context.set(None)
    try:
        kwargs = wrapped(object(), message="hi")
    finally:
        config_context.reset(t1)
        config_space_context.reset(t2)
        mgr._override_active.enabled = False
    assert kwargs["p"] == 0.5
    assert kwargs["k"] == 3
    assert "top_p" not in kwargs and "top_k" not in kwargs
