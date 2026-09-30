"""Unit tests for examples/diagnostic/run_diagnostic.py (mock only, zero network)."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = PROJECT_ROOT / "examples/diagnostic/run_diagnostic.py"
CONFIG = PROJECT_ROOT / "examples/diagnostic/configs/text_to_sql.openrouter.json"
DATASET = PROJECT_ROOT / "examples/datasets/text-to-sql/evaluation_set.jsonl"

_spec = importlib.util.spec_from_file_location("run_diagnostic", SCRIPT)
assert _spec and _spec.loader
rd = importlib.util.module_from_spec(_spec)
sys.modules["run_diagnostic"] = rd
_spec.loader.exec_module(rd)

FORBIDDEN = ("certified", "certificate", "optimal")


def _rows(n: int = 20) -> list[dict[str, str]]:
    return [{"input": f"q{i}", "expected": f"SELECT {i}"} for i in range(n)]


class ScriptedProvider:
    """Provider whose per-(model) behaviour is scripted by a callable."""

    name = "scripted"

    def __init__(self, fn: Any) -> None:
        self.fn = fn
        self.calls = 0

    def complete(self, model, system, user, *, temperature, max_tokens, timeout_s):
        self.calls += 1
        return self.fn(model, user)


def _ctx(provider: Any, cap: float = 100.0, reserve: float = 0.0, retries: int = 1):
    cfg = {**rd.DEFAULTS, "max_retries": retries}
    return rd.Context(
        provider=provider,
        cfg=cfg,
        system_prompt="sys",
        mock=True,
        cap_usd=cap,
        holdout_reserve_usd=reserve,
    )


@pytest.mark.unit
def test_split_is_deterministic_disjoint_and_groups_duplicates() -> None:
    rows = _rows(30) + [{"input": "q3", "expected": "SELECT 3"}] * 2
    a = rd.split_rows(rows, 0.3, seed=5)
    b = rd.split_rows(rows, 0.3, seed=5)
    assert a == b
    search, holdout = a
    assert set(search).isdisjoint(holdout)
    assert sorted(search + holdout) == list(range(len(rows)))
    assert holdout and search
    search_inputs = {rows[i]["input"] for i in search}
    holdout_inputs = {rows[i]["input"] for i in holdout}
    assert search_inputs.isdisjoint(holdout_inputs)
    assert rd.split_rows(rows, 0.3, seed=6) != a


def _stats(q: float, c: float | None) -> dict[str, Any]:
    return {"n": 10, "quality": q, "cost_per_request_usd": c}


@pytest.mark.unit
def test_selection_rule_cases() -> None:
    base = _stats(0.9, 1.0)
    # cheaper and within margin qualifies; lowest cost wins
    sel, _ = rd.select_configuration(
        {"b": base, "x": _stats(0.89, 0.5), "y": _stats(0.9, 0.3)},
        "b",
        ["x", "y"],
        0.02,
    )
    assert sel == "y"
    # equal cost is rejected (strictly cheaper required)
    sel, notes = rd.select_configuration(
        {"b": base, "x": _stats(0.9, 1.0)}, "b", ["x"], 0.02
    )
    assert sel is None and "not strictly cheaper" in notes[0]
    # none qualifies on quality
    sel, _ = rd.select_configuration(
        {"b": base, "x": _stats(0.5, 0.1)}, "b", ["x"], 0.02
    )
    assert sel is None
    # tie on cost -> higher quality
    sel, _ = rd.select_configuration(
        {"b": base, "x": _stats(0.88, 0.4), "y": _stats(0.9, 0.4)},
        "b",
        ["x", "y"],
        0.02,
    )
    assert sel == "y"
    # unknown candidate cost never qualifies
    sel, _ = rd.select_configuration(
        {"b": base, "x": _stats(0.9, None)}, "b", ["x"], 0.02
    )
    assert sel is None


@pytest.mark.unit
def test_unknown_cost_stays_unknown_and_is_counted() -> None:
    prov = ScriptedProvider(lambda m, u: rd.Completion("SELECT 1", 5, 5, None))
    ctx = _ctx(prov)
    rec = rd.run_request(ctx, "m", 0, {"input": "q", "expected": "SELECT 1"}, "search")
    assert rec is not None
    assert rec["cost_usd"] is None
    assert ctx.unknown_cost_calls == 1
    assert ctx.spend_usd == 0.0
    s = rd.summarize(ctx.records)
    assert s["cost_per_request_usd"] is None and s["unknown_cost_calls"] == 1


@pytest.mark.unit
def test_failed_call_is_quality_zero_and_its_cost_counts() -> None:
    def fn(model: str, user: str) -> rd.Completion:
        raise rd.ProviderError("http 500", cost_usd=0.01)

    ctx = _ctx(ScriptedProvider(fn), retries=1)
    rec = rd.run_request(ctx, "m", 0, {"input": "q", "expected": "x"}, "holdout")
    assert rec is not None
    assert rec["quality"] == 0.0 and rec["error"]
    assert rec["cost_usd"] == pytest.approx(0.02)  # two attempts, both charged
    assert ctx.spend_usd == pytest.approx(0.02)
    assert rd.summarize(ctx.records)["errors"] == 1


@pytest.mark.unit
def test_spend_guard_stops_before_next_call() -> None:
    prov = ScriptedProvider(lambda m, u: rd.Completion("a", 1, 1, 0.6))
    ctx = _ctx(prov, cap=1.0)
    row = {"input": "q", "expected": "a"}
    assert rd.run_request(ctx, "m", 0, row, "search") is not None
    assert rd.run_request(ctx, "m", 1, row, "search") is not None  # 1.2 >= cap now
    calls = prov.calls
    assert rd.run_request(ctx, "m", 2, row, "search") is None
    assert prov.calls == calls
    assert ctx.stopped and "spend guard" in ctx.stopped


@pytest.mark.unit
def test_holdout_reserve_stops_search_early() -> None:
    prov = ScriptedProvider(lambda m, u: rd.Completion("a", 1, 1, 0.5))
    ctx = _ctx(prov, cap=2.0, reserve=1.5)
    row = {"input": "q", "expected": "a"}
    assert rd.run_request(ctx, "m", 0, row, "search") is not None
    assert rd.run_request(ctx, "m", 1, row, "search") is None  # 0.5 >= 2.0-1.5
    ctx.stopped = None
    assert rd.run_request(ctx, "m", 1, row, "holdout") is not None


@pytest.mark.unit
def test_incomplete_run_is_marked_in_report(tmp_path: Path) -> None:
    cfg = rd.load_config(CONFIG)
    cfg["max_spend_usd"] = 0.48  # passes pre-flight (estimate ~0.47)
    cfg["samples_path"] = str(DATASET)
    cfg["schema_path"] = str(PROJECT_ROOT / "examples/core/text-to-sql/schema.sql")
    # Under-estimating provider: estimate passes, actual spend hits the cap.
    real = rd.MockProvider(rd.load_samples(DATASET))

    class Costly:
        name = "mock"

        def complete(self, *a: Any, **kw: Any) -> rd.Completion:
            comp = real.complete(*a, **kw)
            comp.cost_usd = 0.05
            return comp

    res = rd.run_diagnostic(cfg, tmp_path, mock=True, provider=Costly())
    assert res["status"] == "incomplete"
    assert "INCOMPLETE" in (tmp_path / "report.md").read_text()


@pytest.mark.unit
def test_preflight_refuses_when_estimate_exceeds_cap(tmp_path: Path) -> None:
    cfg = rd.load_config(CONFIG)
    cfg["max_spend_usd"] = 0.0001
    with pytest.raises(rd.DiagnosticError, match="exceeds max_spend_usd"):
        rd.run_diagnostic(cfg, tmp_path, mock=True)
    frozen = json.loads((tmp_path / "results.json").read_text())
    assert frozen["status"] == "frozen_plan"  # plan is frozen before any call


@pytest.mark.unit
def test_unknown_price_requires_flag(tmp_path: Path) -> None:
    cfg = rd.load_config(CONFIG)
    cfg["baseline"] = {"model": "no-such-vendor/no-such-model-xyz"}
    with pytest.raises(rd.DiagnosticError, match="allow-unknown-price"):
        rd.run_diagnostic(cfg, tmp_path, mock=False, provider=ScriptedProvider(None))


@pytest.mark.unit
def test_bootstrap_claim_logic() -> None:
    same = [0.0] * 200
    iv = rd.paired_bootstrap(same, 500, seed=1)
    ok, text = rd.noninferiority_claim(iv, 0.02, 200)
    assert ok and text.startswith("non-inferior")
    noisy = [1.0, -1.0, 0.0, 0.0, 0.0, -1.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0]
    iv = rd.paired_bootstrap(noisy, 500, seed=1)
    ok, text = rd.noninferiority_claim(iv, 0.02, len(noisy))
    assert not ok and text == "non-inferiority unestablished (holdout n=12)"
    assert rd.paired_bootstrap([0.0, 1.0], 100, 3) == rd.paired_bootstrap(
        [0.0, 1.0], 100, 3
    )


@pytest.mark.unit
def test_nearest_rank_percentiles() -> None:
    vals = [float(i) for i in range(1, 13)]
    assert rd.nearest_rank(vals, 95) == 12.0  # ceil(.95*12)=12 -> max
    assert rd.nearest_rank(vals, 50) == 6.0
    assert rd.nearest_rank([], 95) is None
    assert rd.nearest_rank([float(i) for i in range(1, 101)], 95) == 95.0


@pytest.mark.unit
def test_metric_normalization() -> None:
    assert rd.score("SELECT  1 ;", "select 1") == 1.0
    assert rd.score("select 2", "select 1") == 0.0


@pytest.mark.unit
def test_cli_mock_end_to_end_no_network(tmp_path: Path, monkeypatch) -> None:
    import httpx

    def boom(*a: Any, **kw: Any) -> None:
        raise AssertionError("network client used in mock mode")

    monkeypatch.setattr(httpx, "Client", boom)
    monkeypatch.setattr(httpx, "AsyncClient", boom)
    monkeypatch.chdir(PROJECT_ROOT)
    rc = rd.main(["--config", str(CONFIG), "--out", str(tmp_path), "--mock"])
    assert rc == 0
    report = (tmp_path / "report.md").read_text(encoding="utf-8")
    res = json.loads((tmp_path / "results.json").read_text(encoding="utf-8"))
    for needle in (
        "holdout n=",
        "cost/1k requests (provider)",
        "p95",
        "(descriptive)",
        "## Caveats",
        "stop-on-actual",
        "best observed configuration under the tested scope",
        "textual agreement",
    ):
        assert needle in report, needle
    for word in FORBIDDEN:
        assert word not in report.lower()
        assert word not in json.dumps(res).lower()
    assert res["status"] == "complete"
    assert res["selected_model"] == "google/gemini-2.5-flash-lite"
    assert set(res["split"]["search_ids"]).isdisjoint(res["split"]["holdout_ids"])
    assert len(res["holdout"]["pairs"]) == res["split"]["n_holdout"]
    # every config covered every search row through the shared function
    per = res["search"]["per_config"]
    assert all(v["n"] == res["split"]["n_search"] for v in per.values())


@pytest.mark.unit
@pytest.mark.timeout(300)
def test_cli_subprocess_mock_env_var(tmp_path: Path) -> None:
    env = os.environ.copy()
    env["TRAIGENT_MOCK_LLM"] = "true"
    env["PYTHONPATH"] = str(PROJECT_ROOT)
    env.pop("OPENROUTER_API_KEY", None)
    done = subprocess.run(
        [sys.executable, str(SCRIPT), "--config", str(CONFIG), "--out", str(tmp_path)],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=280,
    )
    assert done.returncode == 0, done.stderr[-2000:]
    assert (tmp_path / "report.md").exists() and (tmp_path / "results.json").exists()
