"""Regression tests for hooks cost checks (#2640)."""

# Traceability: CONC-Quality-Reliability REQ-ORCH-003

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner

from traigent.cli.hooks_commands import hooks
from traigent.hooks.config import HooksConfig
from traigent.hooks.validator import AgentInfo, AgentValidator


def _agent(models: list[str]) -> AgentInfo:
    return AgentInfo(
        name="f",
        file_path="x.py",
        configuration_space={"model": models},
        objectives=["accuracy"],
        constraints=[],
    )


def test_zero_nested_cost_cap_is_not_overridden_by_flat_key() -> None:
    cfg = HooksConfig.from_dict(
        {"constraints": {"cost": {"max_cost_per_query": 0}, "max_cost_per_query": 5}}
    )
    assert cfg.constraints.cost.max_cost_per_query == 0


def test_unpriced_model_under_cost_cap_is_an_error_not_free() -> None:
    cfg = HooksConfig.from_dict({"constraints": {"max_cost_per_query": 0.01}})
    result = AgentValidator(cfg).validate_agent(_agent(["totally-unknown-model-xyz"]))
    assert result.has_errors
    assert not result.is_valid
    assert any(
        i.code == "COST_UNKNOWN" and i.severity == "error" for i in result.issues
    )
    assert "totally-unknown-model-xyz" not in "".join(i.message for i in result.issues)


def test_cli_exit_code_nonzero_for_unpriced_model_under_cost_cap(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "a.py").write_text(
        "import traigent\n"
        "@traigent.optimize(configuration_space={'model': ['totally-unknown-model-xyz']})\n"
        "def f(q):\n    return q\n"
    )
    cfgfile = tmp_path / "traigent.yml"
    cfgfile.write_text("constraints:\n  max_cost_per_query: 0.01\n")
    res = CliRunner().invoke(
        hooks, ["validate", "a.py", "-c", str(cfgfile), "--exit-code"]
    )
    assert res.exit_code == 1, res.output


def test_unpriced_model_mixed_with_priced_still_warns() -> None:
    cfg = HooksConfig.from_dict({"constraints": {"max_cost_per_query": 100}})
    result = AgentValidator(cfg).validate_agent(
        _agent(["gpt-4o-mini", "totally-unknown-model-xyz"])
    )
    assert any(i.code == "COST_UNKNOWN" for i in result.issues)


def test_no_cost_cap_no_unknown_warning() -> None:
    result = AgentValidator(HooksConfig()).validate_agent(
        _agent(["totally-unknown-model-xyz"])
    )
    assert not any(i.code == "COST_UNKNOWN" for i in result.issues + result.warnings)


def test_cli_validate_exit_code_honours_fail_on_warning(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "a.py").write_text("x = 1\n")
    cfgfile = tmp_path / "traigent.yml"
    cfgfile.write_text("validation:\n  fail_on_warning: true\n")

    from traigent.hooks.validator import ValidationIssue, ValidationResult

    warn = ValidationResult(
        function_name="f",
        is_valid=True,
        warnings=[ValidationIssue("warning", "COST_UNKNOWN", "m")],
    )
    with patch.object(AgentValidator, "validate_file", return_value=[warn]):
        res = CliRunner().invoke(
            hooks, ["validate", "a.py", "-c", str(cfgfile), "--exit-code"]
        )
    assert res.exit_code == 1, res.output
