"""Characterization tests pinning the ``hooks validate`` command output."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from traigent.cli.hooks_commands import hooks


def _issue(sev, msg, suggestion=None):
    return SimpleNamespace(severity=sev, message=msg, suggestion=suggestion)


def _res(name, valid=True, models=(), cost=None, issues=(), warnings=()):
    return SimpleNamespace(
        function_name=name,
        is_valid=valid,
        models_found=list(models),
        estimated_cost_per_query=cost,
        issues=list(issues),
        warnings=[SimpleNamespace(message=w) for w in warnings],
    )


OK = _res("ok_fn", models=["gpt-4o", "claude"], cost=0.00123)
BAD = _res(
    "bad_fn",
    valid=False,
    issues=[_issue("error", "too pricey", "use cheaper"), _issue("warning", "hmm")],
)
WARN = _res("warn_fn", warnings=["careful", "really"])

# name -> (args, results, enabled, fail_on_warning, load_error, target_kind)
CASES = {
    "load_error": (["x"], [], True, False, True, "dir"),
    "load_error_verbose": (["x", "-v"], [], True, False, True, "dir"),
    "disabled": (["x"], [OK], False, False, False, "dir"),
    "empty_default": ([], [], True, False, False, "dir"),
    "empty_explicit": (["x"], [], True, False, False, "dir"),
    "empty_explicit_exit": (["x", "--exit-code"], [], True, False, False, "dir"),
    "empty_explicit_exit_allow": (
        ["x", "--exit-code", "--allow-empty"],
        [],
        True,
        False,
        False,
        "dir",
    ),
    "empty_default_exit": (["--exit-code"], [], True, False, False, "dir"),
    "ok_dir": (["x"], [OK], True, False, False, "dir"),
    "ok_file_verbose": (
        ["f.py", "-v", "-c", "cfg.yml"],
        [OK],
        True,
        False,
        False,
        "file",
    ),
    "bad_exit": (["x", "--exit-code"], [OK, BAD], True, False, False, "dir"),
    "bad_noexit_verbose": (["x", "-v"], [OK, BAD], True, False, False, "dir"),
    "warn_only": (["x", "--exit-code"], [WARN], True, False, False, "dir"),
    "warn_fail_on_warning": (
        ["x", "--exit-code"],
        [OK, WARN],
        True,
        True,
        False,
        "dir",
    ),
    "warn_fail_on_warning_noexit": (["x"], [WARN], True, True, False, "dir"),
    "bad_and_warn_fail_on_warning": (
        ["x", "--exit-code"],
        [BAD, WARN],
        True,
        True,
        False,
        "dir",
    ),
    "clean_fail_on_warning": (["x", "--exit-code"], [OK], True, True, False, "dir"),
}


def _snapshot(name):
    args, results, enabled, fail_on_warning, load_error, kind = CASES[name]
    cfg = SimpleNamespace(enabled=enabled, fail_on_warning=fail_on_warning)
    validator = MagicMock()
    validator.validate_file.return_value = list(results)
    validator.validate_directory.return_value = list(results)
    loader = MagicMock(return_value=cfg)
    if load_error:
        loader.side_effect = RuntimeError("bad cfg")
    runner = CliRunner()
    with runner.isolated_filesystem():
        import pathlib

        pathlib.Path("x").mkdir()
        pathlib.Path("f.py").write_text("")
        pathlib.Path("cfg.yml").write_text("")
        with (
            patch("traigent.hooks.config.load_hooks_config", loader),
            patch("traigent.hooks.validator.AgentValidator", return_value=validator),
        ):
            result = runner.invoke(hooks, ["validate", *args])
    return {
        "exit": result.exit_code,
        "output": result.output,
        "calls": repr(
            (
                loader.call_args,
                validator.validate_file.call_args_list,
                validator.validate_directory.call_args_list,
            )
        ),
    }


GOLDEN: dict = {
    "load_error": {
        "exit": 1,
        "output": "\n"
        "Traigent Agent Validation\n"
        "Target: x\n"
        "\n"
        "Error loading configuration: bad cfg\n",
        "calls": "(call(None), [], [])",
    },
    "load_error_verbose": {
        "exit": 1,
        "output": "\n"
        "Traigent Agent Validation\n"
        "Target: x\n"
        "\n"
        "Error loading configuration: bad cfg\n",
        "calls": "(call(None), [], [])",
    },
    "disabled": {
        "exit": 0,
        "output": "\n"
        "Traigent Agent Validation\n"
        "Target: x\n"
        "\n"
        "Traigent hooks are disabled in configuration\n",
        "calls": "(call(None), [], [])",
    },
    "empty_default": {
        "exit": 0,
        "output": "\n"
        "Traigent Agent Validation\n"
        "Target: .\n"
        "\n"
        "No @traigent.optimize decorated functions found\n",
        "calls": "(call(None), [], [call(PosixPath('.'))])",
    },
    "empty_explicit": {
        "exit": 0,
        "output": "\n"
        "Traigent Agent Validation\n"
        "Target: x\n"
        "\n"
        "No @traigent.optimize decorated functions found\n",
        "calls": "(call(None), [], [call(PosixPath('x'))])",
    },
    "empty_explicit_exit": {
        "exit": 1,
        "output": "\n"
        "Traigent Agent Validation\n"
        "Target: x\n"
        "\n"
        "No @traigent.optimize decorated functions found\n",
        "calls": "(call(None), [], [call(PosixPath('x'))])",
    },
    "empty_explicit_exit_allow": {
        "exit": 0,
        "output": "\n"
        "Traigent Agent Validation\n"
        "Target: x\n"
        "\n"
        "No @traigent.optimize decorated functions found\n",
        "calls": "(call(None), [], [call(PosixPath('x'))])",
    },
    "empty_default_exit": {
        "exit": 0,
        "output": "\n"
        "Traigent Agent Validation\n"
        "Target: .\n"
        "\n"
        "No @traigent.optimize decorated functions found\n",
        "calls": "(call(None), [], [call(PosixPath('.'))])",
    },
    "ok_dir": {
        "exit": 0,
        "output": "\n"
        "Traigent Agent Validation\n"
        "Target: x\n"
        "\n"
        "Found 1 decorated function(s)\n"
        "\n"
        "✓ ok_fn\n"
        "  Models: gpt-4o, claude\n"
        "  Est. cost: $0.0012/query\n"
        "\n"
        "Summary:\n"
        "  Passed: 1\n"
        "  Failed: 0\n"
        "\n"
        "All validations passed!\n",
        "calls": "(call(None), [], [call(PosixPath('x'))])",
    },
    "ok_file_verbose": {
        "exit": 0,
        "output": "\n"
        "Traigent Agent Validation\n"
        "Target: f.py\n"
        "\n"
        "Configuration loaded from: cfg.yml\n"
        "\n"
        "Found 1 decorated function(s)\n"
        "\n"
        "✓ ok_fn\n"
        "  Models: gpt-4o, claude\n"
        "  Est. cost: $0.0012/query\n"
        "\n"
        "Summary:\n"
        "  Passed: 1\n"
        "  Failed: 0\n"
        "\n"
        "All validations passed!\n",
        "calls": "(call('cfg.yml'), [call(PosixPath('f.py'))], [])",
    },
    "bad_exit": {
        "exit": 1,
        "output": "\n"
        "Traigent Agent Validation\n"
        "Target: x\n"
        "\n"
        "Found 2 decorated function(s)\n"
        "\n"
        "✓ ok_fn\n"
        "  Models: gpt-4o, claude\n"
        "  Est. cost: $0.0012/query\n"
        "\n"
        "✗ bad_fn\n"
        "  ERROR: too pricey\n"
        "  WARNING: hmm\n"
        "\n"
        "Summary:\n"
        "  Passed: 1\n"
        "  Failed: 1\n"
        "\n"
        "Validation failed!\n",
        "calls": "(call(None), [], [call(PosixPath('x'))])",
    },
    "bad_noexit_verbose": {
        "exit": 0,
        "output": "\n"
        "Traigent Agent Validation\n"
        "Target: x\n"
        "\n"
        "Configuration loaded from: auto-detected\n"
        "\n"
        "Found 2 decorated function(s)\n"
        "\n"
        "✓ ok_fn\n"
        "  Models: gpt-4o, claude\n"
        "  Est. cost: $0.0012/query\n"
        "\n"
        "✗ bad_fn\n"
        "  ERROR: too pricey\n"
        "    Suggestion: use cheaper\n"
        "  WARNING: hmm\n"
        "\n"
        "Summary:\n"
        "  Passed: 1\n"
        "  Failed: 1\n"
        "\n"
        "Validation failed!\n",
        "calls": "(call(None), [], [call(PosixPath('x'))])",
    },
    "warn_only": {
        "exit": 0,
        "output": "\n"
        "Traigent Agent Validation\n"
        "Target: x\n"
        "\n"
        "Found 1 decorated function(s)\n"
        "\n"
        "✓ warn_fn\n"
        "  WARNING: careful\n"
        "  WARNING: really\n"
        "\n"
        "Summary:\n"
        "  Passed: 1\n"
        "  Failed: 0\n"
        "\n"
        "All validations passed!\n",
        "calls": "(call(None), [], [call(PosixPath('x'))])",
    },
    "warn_fail_on_warning": {
        "exit": 1,
        "output": "\n"
        "Traigent Agent Validation\n"
        "Target: x\n"
        "\n"
        "Found 2 decorated function(s)\n"
        "\n"
        "✓ ok_fn\n"
        "  Models: gpt-4o, claude\n"
        "  Est. cost: $0.0012/query\n"
        "\n"
        "✓ warn_fn\n"
        "  WARNING: careful\n"
        "  WARNING: really\n"
        "\n"
        "Summary:\n"
        "  Passed: 2\n"
        "  Failed: 0\n"
        "\n"
        "Warnings present and fail_on_warning is enabled.\n"
        "\n"
        "Validation failed!\n",
        "calls": "(call(None), [], [call(PosixPath('x'))])",
    },
    "warn_fail_on_warning_noexit": {
        "exit": 0,
        "output": "\n"
        "Traigent Agent Validation\n"
        "Target: x\n"
        "\n"
        "Found 1 decorated function(s)\n"
        "\n"
        "✓ warn_fn\n"
        "  WARNING: careful\n"
        "  WARNING: really\n"
        "\n"
        "Summary:\n"
        "  Passed: 1\n"
        "  Failed: 0\n"
        "\n"
        "Warnings present and fail_on_warning is enabled.\n"
        "\n"
        "Validation failed!\n",
        "calls": "(call(None), [], [call(PosixPath('x'))])",
    },
    "bad_and_warn_fail_on_warning": {
        "exit": 1,
        "output": "\n"
        "Traigent Agent Validation\n"
        "Target: x\n"
        "\n"
        "Found 2 decorated function(s)\n"
        "\n"
        "✗ bad_fn\n"
        "  ERROR: too pricey\n"
        "  WARNING: hmm\n"
        "\n"
        "✓ warn_fn\n"
        "  WARNING: careful\n"
        "  WARNING: really\n"
        "\n"
        "Summary:\n"
        "  Passed: 1\n"
        "  Failed: 1\n"
        "\n"
        "Validation failed!\n",
        "calls": "(call(None), [], [call(PosixPath('x'))])",
    },
    "clean_fail_on_warning": {
        "exit": 0,
        "output": "\n"
        "Traigent Agent Validation\n"
        "Target: x\n"
        "\n"
        "Found 1 decorated function(s)\n"
        "\n"
        "✓ ok_fn\n"
        "  Models: gpt-4o, claude\n"
        "  Est. cost: $0.0012/query\n"
        "\n"
        "Summary:\n"
        "  Passed: 1\n"
        "  Failed: 0\n"
        "\n"
        "All validations passed!\n",
        "calls": "(call(None), [], [call(PosixPath('x'))])",
    },
}


@pytest.mark.parametrize("name", list(CASES))
def test_validate_output_exit_and_calls(name):
    assert _snapshot(name) == GOLDEN[name]
