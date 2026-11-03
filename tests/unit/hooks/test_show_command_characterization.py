"""Characterization tests pinning the ``hooks show`` command output."""

from __future__ import annotations

import io
from types import SimpleNamespace
from unittest.mock import patch

from click.testing import CliRunner
from rich.console import Console

from traigent.cli import hooks_commands
from traigent.cli.hooks_commands import hooks


def _fn(name, objectives=("accuracy",), dataset=None, space=None):
    cfg = {} if space is None else {"configuration_space": space}
    return SimpleNamespace(
        name=name,
        objectives=list(objectives),
        eval_dataset=dataset,
        decorator_config=cfg,
    )


def _run(funcs=None, exc=None, args=("m.py",)):
    buf = io.StringIO()
    console = Console(file=buf, width=200, color_system=None, highlight=False)

    def fake(path, names):
        fake.calls.append((path, names))
        if exc:
            raise exc
        return funcs

    fake.calls = []
    with (
        patch.object(hooks_commands, "console", console),
        patch("traigent.cli.function_discovery.discover_optimized_functions", fake),
        CliRunner().isolated_filesystem(),
    ):
        open("m.py", "w").close()
        result = CliRunner().invoke(hooks, ["show", *args])
    return result, buf.getvalue(), fake.calls


def test_load_error_exits_1():
    result, out, _ = _run(exc=RuntimeError("boom"))
    assert result.exit_code == 1
    assert "Error loading module: boom" in out


def test_no_functions_returns_zero():
    result, out, _ = _run(funcs=[])
    assert result.exit_code == 0
    assert "No @traigent.optimize decorated functions found" in out
    assert "Agent Configurations" not in out


def test_function_filter_is_passed_through():
    _, _, calls = _run(funcs=[], args=("m.py", "-f", "foo"))
    assert calls == [("m.py", ["foo"])]
    _, _, calls = _run(funcs=[], args=("m.py",))
    assert calls == [("m.py", None)]


def test_full_rendering_exact():
    funcs = [
        _fn(
            "a",
            objectives=("accuracy", "cost"),
            dataset="d.jsonl",
            space={"model": ["x", "y"], "temp": (0.0, 1.0), "k": 5, "e": []},
        ),
        _fn("b", objectives=()),
        _fn("c", space={}),
    ]
    result, out, _ = _run(funcs=funcs)
    assert result.exit_code == 0
    assert out == (
        "\nAgent Configurations in m.py\n\n"
        "a\n"
        "  Objectives: accuracy, cost\n"
        "  Dataset: d.jsonl\n"
        "  Configuration Space:\n"
        "    model: ['x', 'y']\n"
        "    temp: [0.0 - 1.0]\n"
        "    k: 5\n"
        "    e: []\n"
        "\n"
        "b\n"
        "  Objectives: \n"
        "\n"
        "c\n"
        "  Objectives: accuracy\n"
        "\n"
    )
