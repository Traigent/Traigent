"""Continuous Tuning: builders compile to TVL 1.7 and the TVL toolchain validates."""

from __future__ import annotations

import json
import os
from decimal import Decimal
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

import traigent
from traigent.cli.main import cli
from traigent.tuning import (
    E_FLOAT_RATE,
    E_TVL_VALIDATOR_UNAVAILABLE,
    FailureRate,
    ModelReleased,
    OnBreach,
    ProposeFix,
    RandomAudit,
    Tuning,
    TuningError,
    ValidatorUnavailable,
    validator,
)

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "tuning"
GOLDEN = sorted((FIXTURES / "golden").glob("*.json"))
SLICE_CASE = FIXTURES / "external-slice-trace.yml"
BODY = ("requirements", "monitors", "responses", "tvl_version")


def _golden(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize("path", GOLDEN, ids=lambda p: p.stem)
def test_sugar_compiles_to_golden(path: Path) -> None:
    g = _golden(path)
    doc = Tuning.from_sugar(g["sugar"]).to_tvl(g["module_context"])
    assert json.loads(json.dumps(doc)) == g["expected_tvl"]


@pytest.mark.parametrize("path", GOLDEN, ids=lambda p: p.stem)
def test_sugar_round_trip(path: Path) -> None:
    g = _golden(path)
    assert Tuning.from_sugar(g["sugar"]).to_sugar() == g["sugar"]


@pytest.mark.parametrize("path", GOLDEN, ids=lambda p: p.stem)
def test_golden_lint(path: Path) -> None:
    g = _golden(path)
    diags = Tuning.from_sugar(g["sugar"]).validate(g["module_context"])
    errors = [d for d in diags if d.severity == "error"]
    expected = g.get("expected_diagnostics")
    if expected is None:
        assert errors == []
    else:
        assert [(d.code, d.severity) for d in errors] == [
            (e["code"], e["severity"]) for e in expected
        ]


@pytest.mark.parametrize("path", GOLDEN, ids=lambda p: p.stem)
def test_from_tvl_round_trip(path: Path) -> None:
    doc = _golden(path)["expected_tvl"]
    rebuilt = Tuning.from_tvl(doc)
    assert rebuilt.to_tvl() == doc


def test_slice_matches_tvl_slice_case_module() -> None:
    case = yaml.safe_load(SLICE_CASE.read_text(encoding="utf-8"))
    g = _golden(FIXTURES / "golden" / "slice.json")
    assert g["expected_tvl"] == json.loads(json.dumps(case["module"]))


@pytest.mark.parametrize("bad", [0.08, True])
def test_floats_rejected(bad: object) -> None:
    with pytest.raises(TuningError) as err:
        FailureRate("customer:c", at_most=bad)  # type: ignore[arg-type]
    assert err.value.code == E_FLOAT_RATE
    with pytest.raises(TuningError) as err:
        RandomAudit(0.02)  # type: ignore[arg-type]
    assert err.value.code == E_FLOAT_RATE


def test_decimal_accepted() -> None:
    k = FailureRate("customer:c", at_most=Decimal("0.08"), name="q")
    assert Tuning(keep=[k]).to_tvl()["requirements"][0]["max_rate"] == 0.08


def test_approval_none_rejected() -> None:
    with pytest.raises(TuningError):
        ProposeFix(approval="none")


def test_tvl_fragment_exclusive_with_kwargs(tmp_path: Path) -> None:
    frag = tmp_path / "p.tvl.yml"
    frag.write_text(yaml.safe_dump({"responses": []}))
    with pytest.raises(TypeError):
        Tuning(keep=[FailureRate("customer:c", "0.1")], tvl=frag)


def test_tvl_fragment_compiles(tmp_path: Path) -> None:
    g = _golden(FIXTURES / "golden" / "slice.json")
    frag = tmp_path / "p.tvl.yml"
    body = {k: g["expected_tvl"][k] for k in ("requirements", "monitors", "responses")}
    frag.write_text(yaml.safe_dump(body))
    doc = Tuning(tvl=frag).to_tvl(g["module_context"])
    assert doc == g["expected_tvl"]


def test_multiple_keeps_get_one_breach_rule_each() -> None:
    t = Tuning(
        keep=[
            FailureRate("customer:a", "0.1", name="a"),
            FailureRate("customer:b", "0.2", name="b"),
        ],
        on=[OnBreach()],
    )
    rules = t.to_tvl()["responses"][0]["rules"]
    assert [r["id"] for r in rules] == ["on_breach_a", "on_breach_b"]
    assert [r["trigger"]["monitor"] for r in rules] == ["a_prod", "b_prod"]
    assert Tuning.from_tvl(t.to_tvl()).to_tvl() == t.to_tvl()


def test_observe_attaches_tuning_state(tmp_path: Path) -> None:
    dataset = tmp_path / "d.jsonl"
    dataset.write_text('{"input": {"q": "a"}, "output": "a"}\n')
    policy = Tuning(
        keep=[FailureRate("customer:doc_label_check", "0.08", name="quality_floor")],
        audit=RandomAudit("0.02"),
        on=[
            ModelReleased(family=["anthropic"], tier=["small"], admit="model"),
            OnBreach(),
        ],
    )

    @traigent.observe(agent="doc-classifier", tune=policy)
    @traigent.optimize(
        configuration_space={
            "model": ["anthropic/claude-haiku-4-5", "anthropic/claude-sonnet-4-6"]
        },
        objectives=["accuracy"],
        eval_dataset=str(dataset),
    )
    def classify(q: str) -> str:
        return q

    state = classify.tuning
    assert state.validated and state.agent == "doc-classifier"
    assert state.errors == []
    assert state.tvl["tvars"][0]["domain"] == [
        "anthropic/claude-haiku-4-5",
        "anthropic/claude-sonnet-4-6",
    ]
    assert state.tvl["tvl_version"] == "1.7"


def test_observe_without_tune_is_unchanged() -> None:
    @traigent.observe()
    def plain(x: int) -> int:
        return x + 1

    assert not hasattr(plain, "tuning")
    assert plain(1) == 2


def test_observe_decoration_survives_missing_validator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _hide_tvl(monkeypatch)

    @traigent.observe(
        agent="a", tune=Tuning(keep=[FailureRate("customer:c", "0.1", name="q")])
    )
    def f(x: int) -> int:
        return x

    assert f.tuning.validated is False
    with pytest.raises(ValidatorUnavailable) as err:
        f.tuning.require_validated()
    assert err.value.code == E_TVL_VALIDATOR_UNAVAILABLE
    assert f(3) == 3


def _hide_tvl(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys

    for name in [n for n in sys.modules if n == "tvl" or n.startswith("tvl.")]:
        monkeypatch.setitem(sys.modules, name, None)  # import raises ImportError
    validator.reset_validator_cache()


@pytest.fixture(autouse=True)
def _reset_cache():
    validator.reset_validator_cache()
    yield
    validator.reset_validator_cache()


def test_unavailable_refuses_validate(monkeypatch: pytest.MonkeyPatch) -> None:
    _hide_tvl(monkeypatch)
    with pytest.raises(ValidatorUnavailable) as err:
        Tuning().validate({})
    assert err.value.code == E_TVL_VALIDATOR_UNAVAILABLE


def test_cli_check_and_simulate_refuse_when_unavailable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    f = tmp_path / "m.yml"
    f.write_text("tvl_version: '1.7'\n")
    h = tmp_path / "h.yml"
    h.write_text("steps: []\n")
    _hide_tvl(monkeypatch)
    runner = CliRunner()
    for args in (
        ["tune", "check", str(f)],
        ["tune", "simulate", str(f), "--history", str(h)],
    ):
        result = runner.invoke(cli, args)
        assert result.exit_code == 2
        assert E_TVL_VALIDATOR_UNAVAILABLE in result.output


@pytest.mark.parametrize("path", GOLDEN, ids=lambda p: p.stem)
def test_cli_check_goldens(path: Path, tmp_path: Path) -> None:
    g = _golden(path)
    f = tmp_path / "m.yml"
    f.write_text(yaml.safe_dump(g["expected_tvl"]))
    result = CliRunner().invoke(cli, ["tune", "check", str(f)])
    if "expected_diagnostics" in g:
        assert result.exit_code == 1
        assert g["expected_diagnostics"][0]["code"] in result.output
    else:
        assert result.exit_code == 0, result.output


def _simulate(case_path: Path, tmp_path: Path) -> None:
    case = yaml.safe_load(case_path.read_text(encoding="utf-8"))
    module = tmp_path / "module.yml"
    module.write_text(yaml.safe_dump(case["module"]))
    history = tmp_path / "history.yml"
    history.write_text(yaml.safe_dump(case["input"]["history"]))
    result = CliRunner().invoke(
        cli,
        [
            "tune",
            "simulate",
            str(module),
            "--history",
            str(history),
            "--format",
            "json",
        ],
    )
    assert result.exit_code == 0, result.output
    trace = json.loads(result.output)
    expected = case["expected"]
    assert trace["facts"] == expected["facts"]
    assert trace["head_generations"] == expected["head_generations"]
    assert trace["max_open_attempts"] == expected["max_open_attempts"]


def test_cli_simulate_vendored_slice_case(tmp_path: Path) -> None:
    _simulate(SLICE_CASE, tmp_path)


@pytest.mark.skipif(not os.environ.get("TVL_REPO"), reason="TVL_REPO not set")
def test_cli_simulate_slice_case_from_tvl_repo(tmp_path: Path) -> None:
    case = (
        Path(os.environ["TVL_REPO"])
        / "conformance/cases/runtime/semantic_traces/external-slice-trace.yml"
    )
    assert case.read_bytes() == SLICE_CASE.read_bytes()
    _simulate(case, tmp_path)


def test_cli_compile(tmp_path: Path) -> None:
    dataset = tmp_path / "d.jsonl"
    dataset.write_text('{"input": {"q": "a"}, "output": "a"}\n')
    src = tmp_path / "agent_mod.py"
    src.write_text(
        "import traigent\n"
        "from traigent.tuning import Tuning, FailureRate, OnBreach\n"
        "@traigent.observe(agent='a', tune=Tuning(keep=[FailureRate('customer:c', '0.1', name='q')], on=[OnBreach()]))\n"
        f"@traigent.optimize(configuration_space={{'model': ['x', 'y']}}, objectives=['accuracy'], eval_dataset={str(dataset)!r})\n"
        "def run(q):\n    return q\n"
    )
    out = tmp_path / "out.yml"
    result = CliRunner().invoke(cli, ["tune", "compile", f"{src}:run", "-o", str(out)])
    assert result.exit_code == 0, result.output
    doc = yaml.safe_load(out.read_text())
    assert doc["tvl_version"] == "1.7" and doc["requirements"][0]["name"] == "q"
    check = CliRunner().invoke(cli, ["tune", "check", str(out)])
    assert check.exit_code == 0, check.output
