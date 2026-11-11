"""#3201: the real seamless walkthrough must be rewritable by the seamless provider.

The example runner skips real examples, so this runs the provider's rewrite on
the example's own function body (extracted from source, provider stubbed) and
never calls a model.
"""

from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path

import pytest

from traigent.config.providers import (
    SeamlessNoInjectableTargetsError,
    SeamlessParameterProvider,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
WALKTHROUGH = REPO_ROOT / "walkthrough"
CONFIG = {"model": "gpt-4.1", "temperature": 0.1}

_STUB_HEADER = """
CALLS = []


class ChatOpenAI:
    def __init__(self, **kwargs):
        CALLS.append(kwargs)

    def invoke(self, prompt):
        class _R:
            content = "stub"

        return _R()

"""


def _load_function(tmp_path: Path, example: Path, name: str, modname: str):
    """Compile just ``name`` from ``example`` into a real module file."""
    tree = ast.parse(example.read_text())
    node = next(
        n
        for n in tree.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name
    )
    node.decorator_list = []
    path = tmp_path / f"{modname}.py"
    path.write_text(_STUB_HEADER + "\n\n" + ast.unparse(node) + "\n")
    spec = importlib.util.spec_from_file_location(modname, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[modname] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(modname, None)
        raise
    return module


def test_real_zero_code_change_example_is_seamless_rewritable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("TRAIGENT_ALLOW_SEAMLESS_NO_TARGETS", raising=False)
    example = WALKTHROUGH / "real" / "02_zero_code_change.py"
    module = _load_function(tmp_path, example, "answer_question", "_zcc_real_3201")
    wrapped = SeamlessParameterProvider().inject_config(
        module.answer_question, dict(CONFIG)
    )
    wrapped("What is the capital of France?")
    assert module.CALLS == [{"model": "gpt-4.1", "temperature": 0.1}]


@pytest.mark.parametrize("twin", ["real", "mock"])
def test_zero_code_change_declares_local_assignments_named_after_config_keys(
    twin: str,
) -> None:
    tree = ast.parse((WALKTHROUGH / twin / "02_zero_code_change.py").read_text())
    func = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "answer_question"
    )
    assigned = {
        t.id
        for node in ast.walk(func)
        if isinstance(node, ast.Assign)
        for t in node.targets
        if isinstance(t, ast.Name)
    }
    assert {"model", "temperature"} <= assigned


def test_literal_kwargs_form_is_rejected_by_the_rewriter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Control: the pre-fix shape really is refused, so the test above bites."""
    monkeypatch.delenv("TRAIGENT_ALLOW_SEAMLESS_NO_TARGETS", raising=False)
    path = tmp_path / "_literal_kwargs_3201.py"
    path.write_text(
        _STUB_HEADER
        + """
def answer_question(question):
    llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.7)
    return llm.invoke(question).content
"""
    )
    spec = importlib.util.spec_from_file_location("_literal_kwargs_3201", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["_literal_kwargs_3201"] = module
    spec.loader.exec_module(module)
    wrapped = SeamlessParameterProvider().inject_config(
        module.answer_question, dict(CONFIG)
    )
    with pytest.raises(SeamlessNoInjectableTargetsError):
        wrapped("q")
