"""Golden characterization for the generate-config command's control flow."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from traigent.cli.generate_config_command import generate_config
from traigent.config_generator.types import (
    AutoConfigResult,
    ObjectiveSpec,
    TVarSpec,
)

SAMPLE_SOURCE = """\
from langchain_openai import ChatOpenAI

def my_agent(query: str) -> str:
    llm = ChatOpenAI(model="gpt-4o", temperature=0.7)
    return llm.invoke(query)
"""


def _make_result() -> AutoConfigResult:
    return AutoConfigResult(
        tvars=(
            TVarSpec(
                name="temperature",
                range_type="Range",
                range_kwargs={"low": 0.0, "high": 2.0},
                source="preset",
            ),
        ),
        objectives=(
            ObjectiveSpec(name="accuracy", orientation="maximize", weight=1.0),
        ),
        safety_constraints=(),
        benchmarks=(),
        recommendations=(),
        agent_type="general_llm",
    )


GEN = "traigent.config_generator.generate_config"


@pytest.fixture
def src(tmp_path: Path) -> Path:
    f = tmp_path / "agent.py"
    f.write_text(SAMPLE_SOURCE)
    return f


def _invoke(*args):
    return CliRunner().invoke(generate_config, list(args))


def test_unknown_subsystems_rejected_before_generation(src):
    with patch(GEN) as gen:
        res = _invoke(str(src), "--only", "tvars, bogus ,zzz")
    assert res.exit_code == 1
    assert "Unknown subsystems: bogus, zzz. Valid: " in res.output
    gen.assert_not_called()


def test_only_is_parsed_into_frozenset_and_passed(src):
    with patch(GEN, return_value=_make_result()) as gen:
        res = _invoke(str(src), "--only", "tvars, objectives", "-f", "my_agent")
    assert res.exit_code == 0
    kwargs = gen.call_args.kwargs
    assert kwargs["subsystems"] == frozenset({"tvars", "objectives"})
    assert kwargs["function_name"] == "my_agent"
    assert kwargs["enrich"] is False
    assert kwargs["model"] == "gpt-4o-mini"
    assert kwargs["budget_usd"] == 0.10
    assert gen.call_args.args == (src,)


def test_no_only_passes_none_and_empty_only_too(src):
    with patch(GEN, return_value=_make_result()) as gen:
        _invoke(str(src))
        assert gen.call_args.kwargs["subsystems"] is None
        _invoke(str(src), "--only", "")
        assert gen.call_args.kwargs["subsystems"] is None


@pytest.mark.parametrize(
    "exc,text",
    [
        (FileNotFoundError("x"), "File not found: "),
        (
            SyntaxError("bad", ("f", 3, 1, "t")),
            "Error: could not parse agent.py: syntax error at line 3",
        ),
        (RuntimeError("boom"), "Error: boom"),
    ],
)
def test_generation_errors_exit_1(src, exc, text):
    with patch(GEN, side_effect=exc):
        res = _invoke(str(src))
    assert res.exit_code == 1
    assert text in res.output


def test_apply_requires_function(src):
    with patch(GEN, return_value=_make_result()):
        res = _invoke(str(src), "--apply")
    assert res.exit_code == 1
    assert "--apply requires --function/-f" in res.output


def test_apply_with_and_without_backup(src):
    target = "traigent.config_generator.apply.apply_config"
    with patch(GEN, return_value=_make_result()), patch(target, return_value=src) as ap:
        res = _invoke(str(src), "--apply", "-f", "my_agent")
        assert res.exit_code == 0
        assert f"Applied @traigent.optimize() to my_agent in {src}" in res.output
        assert f"Backup saved to {src.with_suffix('.py.bak')}" in res.output
        assert ap.call_args.kwargs == {"backup": True}
        res = _invoke(str(src), "--apply", "-f", "my_agent", "--no-backup")
        assert "Backup saved" not in res.output
        assert ap.call_args.kwargs == {"backup": False}


def test_apply_skips_output_formatting(src):
    target = "traigent.config_generator.apply.apply_config"
    with patch(GEN, return_value=_make_result()), patch(target, return_value=src):
        res = _invoke(str(src), "--apply", "-f", "f", "-o", "python")
    assert "@traigent.optimize" not in res.output.replace(
        "Applied @traigent.optimize()", ""
    )


def test_json_detailed_flag_overrides_format(src):
    with patch(GEN, return_value=_make_result()):
        plain = _invoke(str(src), "-o", "json")
        detailed = _invoke(str(src), "-o", "json-detailed")
        flagged = _invoke(str(src), "-o", "python", "--json-detailed")
    assert flagged.output == detailed.output
    assert plain.exit_code == 0
    assert flagged.output.lstrip().startswith("{")


def test_tvl_value_error_becomes_click_exception(src):
    with (
        patch(GEN, return_value=_make_result()),
        patch(
            "traigent.cli.generate_config_command._output_tvl",
            side_effect=ValueError("no tvl"),
        ),
    ):
        res = _invoke(str(src), "-o", "tvl")
    assert res.exit_code == 1
    assert "Error: no tvl" in res.output
