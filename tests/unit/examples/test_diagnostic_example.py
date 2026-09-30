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
    assert ok and text.startswith("non-inferior at")
    noisy = [1.0, -1.0, 0.0, 0.0, 0.0, -1.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0]
    iv = rd.paired_bootstrap(noisy, 500, seed=1)
    ok, text = rd.noninferiority_claim(iv, 0.02, len(noisy))
    assert not ok and text.startswith("non-inferiority unestablished (holdout n=12;")
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
        "seeded test database",
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


# ---------------------------------------------------------------------------
# Review-fix regressions
# ---------------------------------------------------------------------------

SCHEMA = PROJECT_ROOT / "examples/core/text-to-sql/schema.sql"
SEED = PROJECT_ROOT / "examples/diagnostic/data/text_to_sql_seed.sql"


def _sql_scorer() -> Any:
    return rd.SqlScorer(
        SCHEMA.read_text(encoding="utf-8"), SEED.read_text(encoding="utf-8")
    )


@pytest.mark.unit
def test_n_required_matches_reviewer_figure() -> None:
    assert rd.n_required(0.02) == 183


@pytest.mark.unit
def test_all_zero_differences_at_n12_is_unestablished() -> None:
    diffs = [0.0] * 12
    iv = rd.paired_bootstrap(diffs, 500, seed=1)
    assert iv == (0.0, 0.0)  # degenerate interval must not be enough
    ok, text = rd.noninferiority_claim(iv, 0.02, 12, 12, True)
    assert not ok
    assert text == (
        "non-inferiority unestablished (holdout n=12; at least 183 independent "
        "pairs needed at margin 0.02)"
    )


@pytest.mark.unit
def test_claim_needs_n_required_independent_pairs() -> None:
    diffs = [0.0] * 183
    iv = rd.paired_bootstrap(diffs, 200, seed=1)
    assert rd.noninferiority_claim(iv, 0.02, 183, 183, True)[0] is True
    assert rd.noninferiority_claim(iv, 0.02, 183, 182, True)[0] is False
    # duplicates do not count as independent pairs
    assert rd.noninferiority_claim(iv, 0.02, 400, 100, True)[0] is False
    # incomplete holdout suppresses the claim even with plenty of pairs
    ok, text = rd.noninferiority_claim(iv, 0.02, 183, 183, False)
    assert not ok and "no inferential claim" in text


@pytest.mark.unit
def test_retry_cannot_exceed_the_guard() -> None:
    def fn(model: str, user: str) -> rd.Completion:
        raise rd.ProviderError("http 500", cost_usd=0.6)

    prov = ScriptedProvider(fn)
    ctx = _ctx(prov, cap=0.5, retries=3)
    rec = rd.run_request(ctx, "m", 0, {"input": "q", "expected": "x"}, "holdout")
    assert prov.calls == 1  # charged failure, then the retry is blocked
    assert rec is not None and rec["error"]
    assert ctx.spend_usd == pytest.approx(0.6)
    assert ctx.stopped and "spend guard" in ctx.stopped


@pytest.mark.unit
def test_unknown_cost_stops_further_calls_and_spend_is_unknown(tmp_path: Path) -> None:
    prov = ScriptedProvider(lambda m, u: rd.Completion("a", 1, 1, None))
    ctx = _ctx(prov)
    row = {"input": "q", "expected": "a"}
    assert rd.run_request(ctx, "m", 0, row, "search") is not None
    assert rd.run_request(ctx, "m", 1, row, "search") is None
    assert prov.calls == 1
    assert ctx.stopped and "unknown charge" in ctx.stopped

    cfg = rd.load_config(CONFIG)
    real = rd.MockProvider(rd.load_samples(DATASET))

    class NoCost:
        name = "mock"

        def complete(self, *a: Any, **kw: Any) -> rd.Completion:
            comp = real.complete(*a, **kw)
            comp.cost_usd = None
            return comp

    res = rd.run_diagnostic(cfg, tmp_path, mock=True, provider=NoCost())
    assert res["status"] == "incomplete"
    assert res["spend"]["total_usd"] is None
    assert res["spend"]["known_subtotal_usd"] == 0.0
    report = (tmp_path / "report.md").read_text()
    assert "Total spend (search + holdout, provider-reported): unknown" in report
    assert "Actual total" not in report
    assert "Known subtotal" in report


@pytest.mark.unit
def test_incomplete_holdout_suppresses_inferential_claims(tmp_path: Path) -> None:
    cfg = rd.load_config(CONFIG)
    real = rd.MockProvider(rd.load_samples(DATASET))
    state = {"holdout_calls": 0}
    holdout_inputs = {
        r["input"]
        for i, r in enumerate(rd.load_samples(DATASET))
        if i in rd.split_rows(rd.load_samples(DATASET), 0.3, 7)[1]
    }

    class StopsMidHoldout:
        name = "mock"

        def complete(self, model, system, user, **kw: Any) -> rd.Completion:
            comp = real.complete(model, system, user, **kw)
            if user in holdout_inputs:
                state["holdout_calls"] += 1
                if state["holdout_calls"] >= 5:
                    comp.cost_usd = None  # unknown charge mid-holdout
            return comp

    res = rd.run_diagnostic(cfg, tmp_path, mock=True, provider=StopsMidHoldout())
    h = res["holdout"]
    assert res["status"] == "incomplete" and h["complete"] is False
    assert h["quality_diff_interval_95"] is None
    assert h["quality_diff_mean"] is None
    assert "no inferential claim" in h["claim"]
    assert "non-inferior at" not in (tmp_path / "report.md").read_text()


@pytest.mark.unit
def test_holdout_calls_exactly_baseline_and_selected(tmp_path: Path) -> None:
    cfg = rd.load_config(CONFIG)
    rows = rd.load_samples(DATASET)
    prov = rd.MockProvider(rows)
    res = rd.run_diagnostic(cfg, tmp_path, mock=True, provider=prov)
    baseline = cfg["baseline"]["model"]
    selected = res["selected_model"]
    assert selected and selected != baseline
    for rid in res["split"]["holdout_ids"]:
        models = [m for m, u in prov.call_log if u == rows[rid]["input"]]
        assert sorted(models) == sorted([baseline, selected])
    pair_models = {
        (p["baseline"]["model"], p["selected"]["model"])
        for p in res["holdout"]["pairs"]
    }
    assert pair_models == {(baseline, selected)}


@pytest.mark.unit
def test_config_provider_mock_is_honored(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("TRAIGENT_MOCK_LLM", raising=False)
    cfg = rd.load_config(CONFIG)
    cfg["provider"] = "mock"
    res = rd.run_diagnostic(cfg, tmp_path, mock=False)
    assert res["mock"] is True and res["provider"] == "mock"


@pytest.mark.unit
def test_openrouter_models_fetch_never_called_in_mock_mode(
    tmp_path: Path, monkeypatch
) -> None:
    def boom(*a: Any, **kw: Any) -> None:
        raise AssertionError("OpenRouter /models fetched in mock mode")

    monkeypatch.setattr(rd, "fetch_openrouter_prices", boom)
    cfg = rd.load_config(CONFIG)
    assert rd.run_diagnostic(cfg, tmp_path, mock=True)["status"] == "complete"


@pytest.mark.unit
def test_openrouter_prices_used_with_provenance(tmp_path: Path, monkeypatch) -> None:
    def fake_fetch(models: list[str]) -> dict[str, Any]:
        return dict.fromkeys(models, (1e-06, 2e-06, "OpenRouter /models at T"))

    monkeypatch.setattr(rd, "fetch_openrouter_prices", fake_fetch)
    cfg = rd.load_config(CONFIG)
    real = rd.MockProvider(rd.load_samples(DATASET))

    class Named:
        name = "openrouter"

        def complete(self, *a: Any, **kw: Any) -> rd.Completion:
            return real.complete(*a, **kw)

    res = rd.run_diagnostic(cfg, tmp_path, mock=False, provider=Named())
    prov = res["preflight"]["pricing_provenance"]
    assert set(prov.values()) == {"OpenRouter /models at T"}


@pytest.mark.unit
def test_openrouter_request_has_no_deprecated_usage_field(monkeypatch) -> None:
    import httpx

    sent: dict[str, Any] = {}

    class FakeClient:
        def __init__(self, *a: Any, **kw: Any) -> None:
            pass

        def __enter__(self) -> FakeClient:
            return self

        def __exit__(self, *a: Any) -> None:
            return None

        def post(self, url: str, json: dict, headers: dict) -> Any:
            sent.update(json)

            class R:
                status_code = 200

                def json(self) -> dict:
                    return {
                        "choices": [{"message": {"content": "SELECT 1"}}],
                        "usage": {"prompt_tokens": 1, "completion_tokens": 1},
                    }

            return R()

    monkeypatch.setattr(httpx, "Client", FakeClient)
    comp = rd.OpenRouterProvider(api_key="test-key").complete(
        "m", "s", "u", temperature=0.0, max_tokens=5, timeout_s=1.0
    )
    assert "usage" not in sent
    assert comp.cost_usd is None  # absent usage.cost means unknown


@pytest.mark.unit
def test_sql_execution_match_cases() -> None:
    sc = _sql_scorer()
    ref = "SELECT name FROM customers WHERE status = 'active'"
    # equivalent query, different text
    assert sc.score("select name from customers where status='active';", ref) == 1.0
    # result mismatch
    assert sc.score("SELECT name FROM customers WHERE status = 'inactive'", ref) == 0.0
    # order-insensitive without ORDER BY
    assert sc.score(f"{ref} ORDER BY name DESC", ref) == 1.0
    # order-sensitive when the reference has ORDER BY
    ordered = "SELECT name FROM customers ORDER BY name ASC"
    assert sc.score("SELECT name FROM customers ORDER BY name DESC", ordered) == 0.0
    assert sc.score("SELECT name FROM customers ORDER BY name ASC", ordered) == 1.0
    # candidate error
    assert sc.score("SELECT nope FROM nowhere", ref) == 0.0
    # code fences stripped
    assert sc.score(f"```sql\n{ref}\n```", ref) == 1.0
    # non-SELECT refused, and the database is untouched
    assert sc.score("DELETE FROM customers", ref) == 0.0
    assert (
        sc.score("SELECT COUNT(*) FROM customers", "SELECT COUNT(*) FROM customers")
        == 1.0
    )
    # invalid references are flagged, not scored
    assert sc.score("SELECT 1", "SELECT * FROM customers WHERE 1 = 0") is None
    assert sc.score("SELECT 1", "SELECT * FROM missing_table") is None
    assert sc.score("SELECT 1", "DROP TABLE customers") is None


@pytest.mark.unit
def test_reference_invalid_rows_are_excluded_from_quality() -> None:
    ctx = _ctx(ScriptedProvider(lambda m, u: rd.Completion("SELECT 1", 1, 1, 0.0)))
    ctx.scorer = _sql_scorer().score
    bad = {"input": "q", "expected": "SELECT * FROM customers WHERE 1 = 0"}
    good = {"input": "g", "expected": "SELECT 1"}
    r1 = rd.run_request(ctx, "m", 0, bad, "search")
    r2 = rd.run_request(ctx, "m", 1, good, "search")
    assert r1 and r1["reference_invalid"] and r1["quality"] is None
    assert r2 and r2["quality"] == 1.0
    s = rd.summarize(ctx.records)
    assert s["quality"] == 1.0 and s["reference_invalid"] == 1 and s["n"] == 2


@pytest.mark.unit
def test_shipped_seed_keeps_reference_queries_valid() -> None:
    sc = _sql_scorer()
    invalid = [
        r["expected"]
        for r in rd.load_samples(DATASET)
        if sc.reference_rows(r["expected"]) is None
    ]
    assert len(invalid) <= 4, invalid


@pytest.mark.unit
def test_sql_metric_requires_schema_and_seed(tmp_path: Path) -> None:
    cfg_path = tmp_path / "c.json"
    base = json.loads(CONFIG.read_text())
    base.pop("seed_path")
    cfg_path.write_text(json.dumps(base))
    with pytest.raises(rd.DiagnosticError, match="seed_path"):
        rd.load_config(cfg_path)
