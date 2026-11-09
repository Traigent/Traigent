"""Regression tests for #3170: traigent_ui problem registry must load all problems.

Six problem modules used to import ``examples.langchain_problems.base`` (not in
the tree or the wheel) and ``load_all_problems()`` skipped them silently.
"""

from __future__ import annotations

import importlib
import logging
import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
PLUGIN_ROOT = REPO_ROOT / "plugins" / "traigent-ui"
PROBLEMS_DIR = PLUGIN_ROOT / "traigent_ui" / "langchain_problems"
PKG = "traigent_ui.langchain_problems"

FORMERLY_BROKEN = [
    "assembly_code_logic_puzzles",
    "customer_support_request_classification",
    "javascript_coding_puzzles",
    "medical_diagnostic_questions",
    "smoking_cessation_guidance",
    "text_to_sql_query_generation",
]


def _expected_modules() -> list[str]:
    return sorted(
        p.stem
        for p in PROBLEMS_DIR.glob("*.py")
        if p.name not in ("__init__.py", "base.py") and not p.name.startswith("test_")
    )


def _registered_modules(plugin) -> list[str]:
    """Module stems that contributed at least one registered problem class.

    Registry keys are not always the module stem (basic_arithmetics registers
    itself as ``basic_arithmetic``), so map each class back to its module.
    """
    return sorted(
        {
            plugin.get_problem_class(n).__module__.rsplit(".", 1)[-1]
            for n in plugin.get_available_problems()
        }
    )


def _purge() -> None:
    for name in [
        m for m in sys.modules if m.split(".")[0] in ("traigent_ui", "examples")
    ]:
        del sys.modules[name]


@pytest.fixture
def plugin(monkeypatch):
    """Import traigent_ui from the plugin dir only, with langchain stubbed."""
    # Nothing but the plugin dir may provide the package: no repo root, no cwd.
    clean = [p for p in sys.path if p not in ("", str(REPO_ROOT))]
    monkeypatch.setattr(sys, "path", [str(PLUGIN_ROOT), *clean])
    _purge()

    lc = types.ModuleType("langchain")
    chains = types.ModuleType("langchain.chains")
    chains.LLMChain = object
    prompts = types.ModuleType("langchain.prompts")
    prompts.ChatPromptTemplate = object
    openai = types.ModuleType("langchain_openai")
    openai.ChatOpenAI = object
    for name, mod in {
        "langchain": lc,
        "langchain.chains": chains,
        "langchain.prompts": prompts,
        "langchain_openai": openai,
    }.items():
        monkeypatch.setitem(sys.modules, name, mod)

    pkg = importlib.import_module(PKG)
    yield pkg
    _purge()


def test_expected_module_count_is_thirteen():
    assert len(_expected_modules()) == 13


def test_load_all_problems_registers_every_module(plugin, caplog):
    with caplog.at_level(logging.WARNING):
        plugin.load_all_problems()
    assert _registered_modules(plugin) == _expected_modules()
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


@pytest.mark.parametrize("module_name", FORMERLY_BROKEN)
def test_formerly_broken_module_imports_without_repo_root(plugin, module_name):
    assert "examples" not in sys.modules
    mod = importlib.import_module(f"{PKG}.{module_name}")
    assert mod.__name__ == f"{PKG}.{module_name}"
    assert not any(
        name == "examples" or name.startswith("examples.") for name in sys.modules
    )


def test_broken_module_logs_warning_naming_it(plugin, caplog, monkeypatch):
    broken = "assembly_code_logic_puzzles"
    real_import = importlib.import_module

    def fake_import(name, package=None):
        if name == f".{broken}":
            raise ModuleNotFoundError("No module named 'examples'")
        return real_import(name, package)

    monkeypatch.setattr(importlib, "import_module", fake_import)
    with caplog.at_level(logging.WARNING, logger=PKG):
        plugin.load_all_problems()

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert broken in warnings[0].getMessage()
    registered = _registered_modules(plugin)
    assert broken not in registered
    assert len(registered) == 12


def test_load_all_problems_twice_returns_same_registry(plugin):
    plugin.load_all_problems()
    first = {n: plugin.get_problem_class(n) for n in plugin.get_available_problems()}
    plugin.load_all_problems()
    second = {n: plugin.get_problem_class(n) for n in plugin.get_available_problems()}
    assert first == second
    assert "basic_arithmetic" in second


def test_module_that_sys_exits_at_import_warns_and_loading_continues(
    plugin, caplog, monkeypatch
):
    broken = "code_review"
    real_import = importlib.import_module

    def fake_import(name, package=None):
        if name == f".{broken}":
            raise SystemExit(1)
        return real_import(name, package)

    monkeypatch.setattr(importlib, "import_module", fake_import)
    with caplog.at_level(logging.WARNING, logger=PKG):
        plugin.load_all_problems()

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert broken in warnings[0].getMessage()
    registered = _registered_modules(plugin)
    assert broken not in registered
    assert len(registered) == 12
