#!/usr/bin/env python3
"""LLM cost & quality diagnostic.

Compares a baseline model against the best observed cheaper configuration on a
held-out split of a customer's sample, and writes ``report.md`` + ``results.json``.

    # no key, no network
    python examples/diagnostic/run_diagnostic.py \
        --config examples/diagnostic/configs/text_to_sql.openrouter.json \
        --out /tmp/diag --mock

    # real run (OPENROUTER_API_KEY in the environment)
    python examples/diagnostic/run_diagnostic.py --config my.json --out ./diag

Flow: load + split -> freeze plan -> pre-flight estimate -> search (traigent.optimize,
grid, search split only) -> explicit selection -> one holdout pass (baseline and
selected, interleaved) -> paired bootstrap -> report.

Every call, in search and holdout, goes through ``run_request`` so scoring,
latency, cost and error accounting are identical everywhere.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import hashlib
import json
import math
import os
import random
import re
import sys
import time
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
METRIC_NAME = "normalized_exact_match"

DEFAULTS: dict[str, Any] = {
    "temperature": 0.0,
    "max_tokens": 512,
    "max_retries": 1,
    "timeout_s": 60.0,
    "holdout_fraction": 0.3,
    "seed": 0,
    "metric": METRIC_NAME,
    "margin": 0.02,
    "bootstrap_resamples": 2000,
    "provider": "openrouter",
}
REQUIRED = ("samples_path", "baseline", "candidate_models", "prompt", "max_spend_usd")

CAP_DESCRIPTION = (
    "stop-on-actual: the run stops before the next call once actual spend reaches "
    "the cap; a single in-flight call may exceed it"
)


class DiagnosticError(Exception):
    """User-facing configuration or pre-flight failure."""


# ---------------------------------------------------------------------------
# Config, samples, split
# ---------------------------------------------------------------------------


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    cfg = {**DEFAULTS, **json.loads(path.read_text(encoding="utf-8"))}
    missing = [k for k in REQUIRED if k not in cfg]
    if missing:
        raise DiagnosticError(f"config is missing required keys: {missing}")
    if cfg["metric"] != METRIC_NAME:
        raise DiagnosticError(f"only metric {METRIC_NAME!r} is supported")
    if not isinstance(cfg["baseline"], dict) or "model" not in cfg["baseline"]:
        raise DiagnosticError("baseline must be an object with a 'model' key")
    if not cfg["candidate_models"]:
        raise DiagnosticError("candidate_models must not be empty")
    if not 0.0 < float(cfg["holdout_fraction"]) < 1.0:
        raise DiagnosticError("holdout_fraction must be strictly between 0 and 1")
    if int(cfg["max_tokens"]) <= 0 or int(cfg["max_retries"]) < 0:
        raise DiagnosticError("max_tokens must be > 0 and max_retries >= 0")
    if float(cfg["max_spend_usd"]) <= 0:
        raise DiagnosticError("max_spend_usd must be > 0")
    if cfg["provider"] not in ("openrouter", "mock"):
        raise DiagnosticError("provider must be 'openrouter' or 'mock'")
    base = path.parent
    for key in ("samples_path", "schema_path"):
        if cfg.get(key) and not Path(cfg[key]).is_absolute():
            cfg[key] = str(_resolve_relative(cfg[key], base))
    return cfg


def _resolve_relative(rel: str, config_dir: Path) -> Path:
    """Resolve against the working directory first, then the config directory."""
    for candidate in (Path.cwd() / rel, config_dir / rel):
        if candidate.exists():
            return candidate
    return Path.cwd() / rel


def load_samples(path: str | Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for lineno, line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        obj = json.loads(line)
        if "input" not in obj or "expected" not in obj:
            raise DiagnosticError(f"{path}:{lineno}: row needs 'input' and 'expected'")
        rows.append({"input": str(obj["input"]), "expected": str(obj["expected"])})
    if len(rows) < 2:
        raise DiagnosticError("need at least 2 sample rows to split")
    return rows


def split_rows(
    rows: list[dict[str, str]], holdout_fraction: float, seed: int
) -> tuple[list[int], list[int]]:
    """Seeded split into (search_ids, holdout_ids) of row indexes.

    Rows with an identical ``input`` are kept together, so duplicates never
    straddle the split.
    """
    groups: dict[str, list[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        groups[row["input"]].append(idx)
    if len(groups) < 2:
        raise DiagnosticError("need at least 2 distinct inputs to split")
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    target = max(1, round(len(rows) * holdout_fraction))
    holdout: list[int] = []
    used = 0
    for key in keys[:-1]:  # always leave at least one group for search
        if used >= target:
            break
        holdout.extend(groups[key])
        used += len(groups[key])
    held = set(holdout)
    search = [i for i in range(len(rows)) if i not in held]
    return sorted(search), sorted(holdout)


# ---------------------------------------------------------------------------
# Metric and statistics
# ---------------------------------------------------------------------------


def normalize(text: str) -> str:
    text = re.sub(r"\s+", " ", text.strip().lower())
    return text.rstrip(";").strip()


def score(output: str, expected: str) -> float:
    return 1.0 if normalize(output) == normalize(expected) else 0.0


def nearest_rank(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, math.ceil(pct / 100.0 * len(ordered)))
    return ordered[rank - 1]


def paired_bootstrap(
    diffs: list[float], resamples: int, seed: int
) -> tuple[float, float] | None:
    """Seeded 95% percentile interval of the mean paired difference."""
    if not diffs:
        return None
    rng = random.Random(seed)
    n = len(diffs)
    means = sorted(
        sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(resamples)
    )
    lo = means[int(0.025 * (resamples - 1))]
    hi = means[int(math.ceil(0.975 * (resamples - 1)))]
    return lo, hi


def noninferiority_claim(
    interval: tuple[float, float] | None, margin: float, n: int
) -> tuple[bool, str]:
    if interval is not None and interval[0] > -margin:
        return True, (
            f"non-inferior at margin {margin:g} (95% paired bootstrap lower bound "
            f"{interval[0]:+.3f} > {-margin:+.3f}, holdout n={n})"
        )
    return False, f"non-inferiority unestablished (holdout n={n})"


# ---------------------------------------------------------------------------
# Providers
# ---------------------------------------------------------------------------


@dataclass
class Completion:
    text: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cost_usd: float | None = None  # provider-reported; None means unknown
    simulated_latency_s: float = 0.0


class ProviderError(Exception):
    def __init__(
        self, message: str, cost_usd: float | None = None, simulated_latency_s=0.0
    ):
        super().__init__(message)
        self.cost_usd = cost_usd
        self.simulated_latency_s = simulated_latency_s


class Provider(Protocol):
    name: str

    def complete(
        self,
        model: str,
        system: str,
        user: str,
        *,
        temperature: float,
        max_tokens: int,
        timeout_s: float,
    ) -> Completion: ...


class OpenRouterProvider:
    name = "openrouter"

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        if not self.api_key:
            raise DiagnosticError(
                "OPENROUTER_API_KEY is not set (or use --mock for a no-key dry run)"
            )

    def complete(
        self, model, system, user, *, temperature, max_tokens, timeout_s
    ) -> Completion:
        import httpx

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "usage": {"include": True},  # ask OpenRouter for usage.cost
        }
        try:
            with httpx.Client(timeout=timeout_s) as client:
                resp = client.post(
                    OPENROUTER_URL,
                    json=payload,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                )
        except httpx.HTTPError as exc:
            raise ProviderError(f"transport error: {type(exc).__name__}") from exc
        try:
            body = resp.json()
        except ValueError:
            body = {}
        usage = body.get("usage") or {}
        cost = usage.get("cost")
        cost = float(cost) if isinstance(cost, (int, float)) else None
        if resp.status_code >= 400 or "choices" not in body:
            raise ProviderError(f"http {resp.status_code}", cost_usd=cost)
        content = (body["choices"][0].get("message") or {}).get("content") or ""
        return Completion(
            text=str(content),
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            cost_usd=cost,
        )


# Deterministic mock model behaviour: (error_rate, usd_per_1k_input, usd_per_1k_output,
# base_latency_s). The mock also invents a per-row latency jitter from a hash.
MOCK_PROFILES: dict[str, tuple[float, float, float, float]] = {
    "mock/frontier": (0.0, 0.003, 0.015, 1.2),
    "mock/cheap-equal": (0.0, 0.0002, 0.0006, 0.5),
    "mock/cheap-worse": (0.25, 0.0001, 0.0004, 0.4),
    "anthropic/claude-sonnet-4.5": (0.0, 0.003, 0.015, 1.3),
    "openai/gpt-4o-mini": (0.25, 0.00015, 0.0006, 0.6),
    "google/gemini-2.5-flash-lite": (0.0, 0.0001, 0.0004, 0.4),
    "anthropic/claude-haiku-4.5": (0.05, 0.001, 0.005, 0.7),
}


def _unit_hash(*parts: str) -> float:
    digest = hashlib.sha256("|".join(parts).encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def mock_profile(model: str) -> tuple[float, float, float, float]:
    if model in MOCK_PROFILES:
        return MOCK_PROFILES[model]
    u = _unit_hash("profile", model)
    return (0.3 * u, 0.0002 + 0.002 * u, 0.0008 + 0.01 * u, 0.4 + u)


def _approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


class MockProvider:
    """Deterministic, zero-network provider. Knows the answers (an oracle)."""

    name = "mock"

    def __init__(self, samples: list[dict[str, str]]) -> None:
        self._answers: dict[str, list[str]] = defaultdict(list)
        for row in samples:
            self._answers[row["input"]].append(row["expected"])
        self.calls = 0

    def complete(
        self, model, system, user, *, temperature, max_tokens, timeout_s
    ) -> Completion:
        self.calls += 1
        err, pin, pout, base = mock_profile(model)
        expected = self._answers[user][0] if self._answers.get(user) else ""
        wrong = _unit_hash("wrong", model, user) < err
        text = f"{expected} /* wrong */ LIMIT 0" if wrong else expected
        ptoks = _approx_tokens(system + user)
        ctoks = min(_approx_tokens(text), max_tokens)
        return Completion(
            text=text,
            prompt_tokens=ptoks,
            completion_tokens=ctoks,
            cost_usd=(ptoks * pin + ctoks * pout) / 1000.0,
            simulated_latency_s=base * (0.8 + 0.4 * _unit_hash("lat", model, user)),
        )


def price_per_token(model: str, mock: bool) -> tuple[float, float, str] | None:
    """(input $/token, output $/token, provenance) or None if unknown."""
    if mock:
        _, pin, pout, _ = mock_profile(model)
        return pin / 1000.0, pout / 1000.0, "mock_profile"
    try:
        from traigent.utils.cost_calculator import get_model_token_pricing

        return get_model_token_pricing(model)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Shared invocation + scoring + accounting
# ---------------------------------------------------------------------------


@dataclass
class Context:
    provider: Provider
    cfg: dict[str, Any]
    system_prompt: str
    mock: bool
    cap_usd: float
    holdout_reserve_usd: float = 0.0
    spend_usd: float = 0.0
    unknown_cost_calls: int = 0
    stopped: str | None = None
    records: list[dict[str, Any]] = field(default_factory=list)
    _sleep: Any = None

    def budget_left_for(self, phase: str) -> bool:
        limit = self.cap_usd - (self.holdout_reserve_usd if phase == "search" else 0.0)
        return self.spend_usd < limit


def run_request(
    ctx: Context, model: str, row_id: int, row: dict[str, str], phase: str
) -> dict[str, Any] | None:
    """The one place a model is invoked, scored and accounted for.

    Returns None (and sets ``ctx.stopped``) if the spend guard stops the run
    before the call. Latency is monotonic elapsed time around the whole request
    including retries.
    """
    if ctx.stopped:
        return None
    if not ctx.budget_left_for(phase):
        ctx.stopped = (
            f"spend guard: actual spend ${ctx.spend_usd:.6f} reached the limit "
            f"before a {phase} call"
        )
        return None
    cfg = ctx.cfg
    attempts = 1 + int(cfg["max_retries"])
    cost_total = 0.0
    unknown_attempts = 0
    ptoks = ctoks = 0
    tokens_known = True
    simulated = 0.0
    error: str | None = None
    text = ""
    start = time.monotonic()
    for _ in range(attempts):
        try:
            comp = ctx.provider.complete(
                model,
                ctx.system_prompt,
                row["input"],
                temperature=float(cfg["temperature"]),
                max_tokens=int(cfg["max_tokens"]),
                timeout_s=float(cfg["timeout_s"]),
            )
        except ProviderError as exc:
            error = str(exc)
            simulated += exc.simulated_latency_s
            if exc.cost_usd is None:
                unknown_attempts += 1
            else:
                cost_total += exc.cost_usd
            tokens_known = False
            continue
        error = None
        simulated += comp.simulated_latency_s
        if comp.cost_usd is None:
            unknown_attempts += 1
        else:
            cost_total += comp.cost_usd
        if comp.prompt_tokens is None or comp.completion_tokens is None:
            tokens_known = False
        else:
            ptoks += comp.prompt_tokens
            ctoks += comp.completion_tokens
        text = comp.text
        break
    latency = time.monotonic() - start + simulated
    ctx.spend_usd += cost_total
    ctx.unknown_cost_calls += 1 if unknown_attempts else 0
    rec = {
        "phase": phase,
        "model": model,
        "row_id": row_id,
        "quality": 0.0 if error else score(text, row["expected"]),
        "error": error,
        "cost_usd": None if unknown_attempts else cost_total,
        "known_cost_usd": cost_total,
        "prompt_tokens": ptoks if tokens_known else None,
        "completion_tokens": ctoks if tokens_known else None,
        "latency_s": latency,
        "output": text,
    }
    ctx.records.append(rec)
    return rec


# ---------------------------------------------------------------------------
# Search stats + selection
# ---------------------------------------------------------------------------


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(records)
    unknown = sum(1 for r in records if r["cost_usd"] is None)
    known = sum(r["known_cost_usd"] for r in records)
    lat = [r["latency_s"] for r in records]
    return {
        "n": n,
        "quality": sum(r["quality"] for r in records) / n if n else None,
        "errors": sum(1 for r in records if r["error"]),
        "unknown_cost_calls": unknown,
        "cost_per_request_usd": (known / n) if n and unknown == 0 else None,
        "known_cost_usd": known,
        "latency_p50_s": nearest_rank(lat, 50),
        "latency_p95_s": nearest_rank(lat, 95),
    }


def select_configuration(
    stats: dict[str, dict[str, Any]],
    baseline: str,
    candidates: list[str],
    margin: float,
) -> tuple[str | None, list[str]]:
    """Selection rule, recorded verbatim in the report.

    Among candidates whose provider-reported search cost per request is STRICTLY
    below the baseline's and whose search quality >= baseline quality - margin,
    pick the lowest cost; tie-break higher quality.
    """
    notes: list[str] = []
    base = stats.get(baseline)
    if not base or base["quality"] is None or base["cost_per_request_usd"] is None:
        return None, [
            "baseline search cost or quality is unknown; nothing can be selected"
        ]
    eligible: list[tuple[float, float, int, str]] = []
    for order, model in enumerate(candidates):
        s = stats.get(model)
        if not s or s["quality"] is None:
            notes.append(f"{model}: rejected (not evaluated)")
        elif s["cost_per_request_usd"] is None:
            notes.append(f"{model}: rejected (provider cost unknown)")
        elif not s["cost_per_request_usd"] < base["cost_per_request_usd"]:
            notes.append(f"{model}: rejected (not strictly cheaper than baseline)")
        elif s["quality"] < base["quality"] - margin:
            notes.append(
                f"{model}: rejected (search quality {s['quality']:.3f} < "
                f"{base['quality'] - margin:.3f})"
            )
        else:
            notes.append(f"{model}: eligible")
            eligible.append((s["cost_per_request_usd"], -s["quality"], order, model))
    if not eligible:
        return None, notes
    eligible.sort()
    return eligible[0][3], notes


# ---------------------------------------------------------------------------
# traigent.optimize search
# ---------------------------------------------------------------------------


def run_search(
    ctx: Context,
    rows: list[dict[str, str]],
    search_ids: list[int],
    models: list[str],
    tmp_dir: Path,
) -> dict[str, Any]:
    """Grid search over models on the search split using traigent.optimize."""
    import traigent
    from traigent.core.objectives import create_default_objectives

    dataset = tmp_dir / "search_split.jsonl"
    dataset.write_text(
        "\n".join(json.dumps(rows[i]) for i in search_ids) + "\n", encoding="utf-8"
    )
    queue: dict[str, list[int]] = defaultdict(list)  # input -> row ids (in order)
    for i in search_ids:
        queue[rows[i]["input"]].append(i)
    seen: dict[tuple[str, str], int] = defaultdict(int)

    def metric(output: str, expected: str, **_: object) -> float:
        return score(output, expected)

    @traigent.optimize(
        eval_dataset=str(dataset),
        objectives=create_default_objectives(
            [METRIC_NAME], orientations={METRIC_NAME: "maximize"}
        ),
        configuration_space={"model": list(models)},
        metric_functions={METRIC_NAME: metric},
        injection_mode="context",
        offline=True,
    )
    def generate(input: str) -> str:  # noqa: A002 - dataset key is "input"
        model = str(traigent.get_config()["model"])
        ids = queue[input]
        nth = seen[(model, input)]
        seen[(model, input)] += 1
        row_id = ids[nth % len(ids)]
        rec = run_request(ctx, model, row_id, rows[row_id], "search")
        return "" if rec is None else rec["output"]

    result = asyncio.run(generate.optimize(algorithm="grid", max_trials=len(models)))
    return {
        "traigent_trials": len(result.trials),
        "traigent_best_config_informational": dict(result.best_config or {}),
    }


# ---------------------------------------------------------------------------
# Pre-flight estimate
# ---------------------------------------------------------------------------


def estimate_cost(
    cfg: dict[str, Any],
    models: list[str],
    n_search: int,
    n_holdout: int,
    mock: bool,
    system_prompt: str,
    avg_input_chars: float,
) -> dict[str, Any]:
    in_tok = _approx_tokens(system_prompt) + int(avg_input_chars / 4) + 1
    out_tok = int(cfg["max_tokens"])  # conservative upper bound
    per_model: dict[str, float | None] = {}
    provenance: dict[str, str] = {}
    for model in models:
        price = price_per_token(model, mock)
        if price is None:
            per_model[model] = None
            provenance[model] = "unknown"
        else:
            per_model[model] = in_tok * price[0] + out_tok * price[1]
            provenance[model] = price[2]
    base = cfg["baseline"]["model"]
    cands = [m for m in models if m != base]
    unknown = [m for m, v in per_model.items() if v is None]
    search_est = sum((v or 0.0) * n_search for v in per_model.values())
    known_cands = [per_model[m] or 0.0 for m in cands]
    holdout_reserve = (per_model[base] or 0.0) * n_holdout + max(
        known_cands, default=0.0
    ) * n_holdout
    return {
        "calls_search": n_search * len(models),
        "calls_holdout": 2 * n_holdout,
        "assumed_input_tokens_per_call": in_tok,
        "assumed_output_tokens_per_call": out_tok,
        "per_call_usd": per_model,
        "pricing_provenance": provenance,
        "unknown_price_models": unknown,
        "search_usd": None if unknown else search_est,
        "holdout_reserve_usd": None if unknown else holdout_reserve,
        "total_usd": None if unknown else search_est + holdout_reserve,
        "note": "assumes max_tokens output on every call and no retries",
    }


# ---------------------------------------------------------------------------
# Holdout + report
# ---------------------------------------------------------------------------


def run_holdout(
    ctx: Context,
    rows: list[dict[str, str]],
    holdout_ids: list[int],
    baseline: str,
    selected: str,
) -> list[dict[str, Any]]:
    """Evaluate baseline and selected on every holdout row, interleaved per row."""
    pairs: list[dict[str, Any]] = []
    for rid in holdout_ids:
        b = run_request(ctx, baseline, rid, rows[rid], "holdout")
        if b is None:
            break
        c = run_request(ctx, selected, rid, rows[rid], "holdout")
        if c is None:
            break
        pairs.append({"row_id": rid, "baseline": b, "selected": c})
    return pairs


def _fmt_usd(v: float | None) -> str:
    return "unknown" if v is None else f"${v:.6f}"


def _fmt_s(v: float | None) -> str:
    return "n/a" if v is None else f"{v:.2f}s"


def arm_summary(
    records: list[dict[str, Any]], model: str, mock: bool
) -> dict[str, Any]:
    s = summarize(records)
    s["model"] = model
    s["cost_per_1k_requests_usd"] = (
        None if s["cost_per_request_usd"] is None else s["cost_per_request_usd"] * 1000
    )
    est = None
    price = price_per_token(model, mock)
    if price and records and all(r["prompt_tokens"] is not None for r in records):
        tot = sum(
            r["prompt_tokens"] * price[0] + r["completion_tokens"] * price[1]
            for r in records
        )
        est = tot / len(records) * 1000
    s["price_table_estimate_per_1k_requests_usd"] = est
    s["price_table_provenance"] = price[2] if price else "unknown"
    return s


def render_report(res: dict[str, Any]) -> str:
    cfg = res["config"]
    lines = ["# LLM cost & quality diagnostic", ""]
    if res["status"] != "complete":
        lines += [f"**INCOMPLETE: {res['incomplete_reason']}**", ""]
    if res["mock"]:
        lines += [
            "> MOCK MODE: synthetic models, costs and latencies. Numbers below "
            "exercise the flow only and say nothing about real models.",
            "",
        ]
    sp = res["split"]
    lines += [
        "## Setup",
        f"- Samples: {sp['n_total']} rows; search n={sp['n_search']}, "
        f"holdout n={sp['n_holdout']} (seed {cfg['seed']}, duplicates kept together)",
        f"- Metric: {METRIC_NAME} (lowercase, collapse whitespace, strip trailing ';')",
        f"- Baseline: {cfg['baseline']['model']}; candidates: "
        f"{', '.join(cfg['candidate_models'])}",
        f"- Temperature {cfg['temperature']}, max_tokens {cfg['max_tokens']}, "
        f"max_retries {cfg['max_retries']}, margin {cfg['margin']} (absolute)",
        f"- Spend cap USD {cfg['max_spend_usd']}: {CAP_DESCRIPTION}",
        "",
        "## Search (search split only)",
        "| model | quality | cost/request (provider) | errors | unknown-cost calls |",
        "|---|---|---|---|---|",
    ]
    for model, s in res["search"]["per_config"].items():
        q = "n/a" if s["quality"] is None else f"{s['quality']:.3f}"
        lines.append(
            f"| {model} | {q} | {_fmt_usd(s['cost_per_request_usd'])} | "
            f"{s['errors']} | {s['unknown_cost_calls']} |"
        )
    if res["status"] == "frozen_plan":
        return "\n".join(lines + ["Plan frozen; no calls were made.", ""])
    lines += ["", "### Selection rule and outcome"]
    lines.append(
        "Among candidates whose provider-reported search cost per request is strictly "
        "below the baseline's and whose search quality is at least baseline quality "
        "minus the margin, the lowest cost is chosen; ties go to higher quality."
    )
    lines += [f"- {n}" for n in res["search"]["selection_notes"]]
    sel = res["selected_model"]
    if sel is None:
        lines += ["", "**No cheaper configuration met the quality bar.**"]
    else:
        lines += [
            "",
            f"Selected (best observed configuration under the tested scope): {sel}",
        ]
    if res.get("holdout"):
        h = res["holdout"]
        lines += [
            "",
            "## Holdout (evaluated once, after selection)",
            "| arm | model | quality | cost/1k requests (provider) | "
            "cost/1k (price-table estimate) | p50 | p95 | errors |",
            "|---|---|---|---|---|---|---|---|",
        ]
        p95_label = " (descriptive)" if h["n"] < 20 else ""
        for arm in ("baseline", "selected"):
            a = h[arm]
            est = a["price_table_estimate_per_1k_requests_usd"]
            est_s = "unknown" if est is None else f"${est:.4f} (estimate)"
            lines.append(
                f"| {arm} | {a['model']} | {a['quality']:.3f} | "
                f"{_fmt_usd(a['cost_per_1k_requests_usd'])} | {est_s} | "
                f"{_fmt_s(a['latency_p50_s'])} | "
                f"{_fmt_s(a['latency_p95_s'])}{p95_label} | {a['errors']} |"
            )
        iv = h["quality_diff_interval_95"]
        lines += [
            "",
            f"- Holdout n={h['n']}; quality difference (selected - baseline) = "
            f"{h['quality_diff_mean']:+.3f}",
            "- Paired bootstrap 95% interval: "
            + ("n/a" if iv is None else f"[{iv[0]:+.3f}, {iv[1]:+.3f}]")
            + f" ({cfg['bootstrap_resamples']} resamples, seeded)",
            f"- **{h['claim']}**",
        ]
        if h["cost_saving_pct"] is not None:
            lines.append(
                f"- Provider-reported cost change vs baseline: "
                f"{-h['cost_saving_pct']:+.1f}% per request"
            )
        if h["n"] < 20:
            lines.append("- p95 latency is descriptive only at this sample size.")
    sp_ = res["spend"]
    lines += [
        "",
        "## Spend",
        f"- Actual total (search + holdout, provider-reported): "
        f"{_fmt_usd(sp_['actual_total_usd'])}",
        f"- Calls with unknown provider cost (excluded from the total): "
        f"{sp_['unknown_cost_calls']}",
        f"- Pre-flight estimate: {_fmt_usd(res['preflight']['total_usd'])}",
        "",
        "## Caveats",
        f"- Small samples: search n={sp['n_search']}, holdout n={sp['n_holdout']}. "
        "One row changes quality by 1/n; differences of a few points are noise.",
        "- Normalized exact match measures textual agreement with the reference, "
        "not execution correctness (for SQL, an equivalent query can score 0).",
        "- Results describe the tested models, prompt and sample only; customer "
        "sampling bias limits extrapolation. This is not a production guarantee.",
        "",
    ]
    return "\n".join(lines)


def write_outputs(out: Path, res: dict[str, Any]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(
        json.dumps(res, indent=2, sort_keys=True), encoding="utf-8"
    )
    (out / "report.md").write_text(render_report(res), encoding="utf-8")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _traigent_env(results_dir: Path, dataset_root: Path, mock: bool) -> Iterator[None]:
    updates = {
        "TRAIGENT_RESULTS_FOLDER": str(results_dir),
        "TRAIGENT_COST_APPROVED": "true",
        "TRAIGENT_DATASET_ROOT": str(dataset_root),
    }
    if mock:
        updates["TRAIGENT_MOCK_LLM"] = "true"
    saved = {k: os.environ.get(k) for k in updates}
    os.environ.update(updates)
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def run_diagnostic(
    cfg: dict[str, Any],
    out: Path,
    *,
    mock: bool,
    allow_unknown_price: bool = False,
    provider: Provider | None = None,
) -> dict[str, Any]:
    rows = load_samples(cfg["samples_path"])
    schema_text = ""
    if cfg.get("schema_path"):
        schema_text = Path(cfg["schema_path"]).read_text(encoding="utf-8")
    system_prompt = str(cfg["prompt"]).replace("{schema_text}", schema_text)
    baseline = cfg["baseline"]["model"]
    models = [baseline] + [
        m for m in dict.fromkeys(cfg["candidate_models"]) if m != baseline
    ]
    candidates = models[1:]
    if not candidates:
        raise DiagnosticError(
            "candidate_models must contain a model other than the baseline"
        )

    search_ids, holdout_ids = split_rows(
        rows, float(cfg["holdout_fraction"]), int(cfg["seed"])
    )
    avg_chars = sum(len(r["input"]) for r in rows) / len(rows)
    est = estimate_cost(
        cfg, models, len(search_ids), len(holdout_ids), mock, system_prompt, avg_chars
    )
    if provider is None:
        provider = MockProvider(rows) if mock else OpenRouterProvider()
    res: dict[str, Any] = {
        "status": "frozen_plan",
        "incomplete_reason": None,
        "mock": mock,
        "provider": provider.name,
        "config": dict(cfg),
        "metric": METRIC_NAME,
        "split": {
            "n_total": len(rows),
            "n_search": len(search_ids),
            "n_holdout": len(holdout_ids),
            "search_ids": search_ids,
            "holdout_ids": holdout_ids,
        },
        "models": models,
        "preflight": est,
        "spend_cap_semantics": CAP_DESCRIPTION,
    }
    # Freeze split, configs and metric before any call.
    write_outputs(
        out,
        res
        | {
            "status": "frozen_plan",
            "incomplete_reason": "no calls made yet",
            "search": {"per_config": {}, "selection_notes": []},
            "selected_model": None,
            "spend": {"actual_total_usd": 0.0, "unknown_cost_calls": 0},
        },
    )
    if est["unknown_price_models"] and not allow_unknown_price:
        raise DiagnosticError(
            f"pricing unknown for {est['unknown_price_models']}; the pre-flight "
            "estimate is 'unknown'. Re-run with --allow-unknown-price to proceed."
        )
    cap = float(cfg["max_spend_usd"])
    if est["total_usd"] is not None and est["total_usd"] > cap:
        raise DiagnosticError(
            f"pre-flight estimate ${est['total_usd']:.4f} exceeds max_spend_usd "
            f"${cap:.2f}; refusing to start"
        )
    ctx = Context(
        provider=provider,
        cfg=cfg,
        system_prompt=system_prompt,
        mock=mock,
        cap_usd=cap,
        holdout_reserve_usd=est["holdout_reserve_usd"] or 0.0,
    )

    work = out / "work"
    work.mkdir(parents=True, exist_ok=True)
    with _traigent_env(out / ".traigent", work, mock):
        res["search"] = run_search(ctx, rows, search_ids, models, work)

    per_config: dict[str, Any] = {}
    complete_search = True
    for m in models:
        recs = [r for r in ctx.records if r["phase"] == "search" and r["model"] == m]
        if len(recs) != len(search_ids):
            complete_search = False
        per_config[m] = summarize(recs)
    selected, notes = (None, ["search incomplete; no selection made"])
    if complete_search:
        selected, notes = select_configuration(
            per_config, baseline, candidates, float(cfg["margin"])
        )
    res["search"] |= {"per_config": per_config, "selection_notes": notes}
    res["selected_model"] = selected
    res["selection_rule"] = (
        "strictly lower provider-reported search cost per request than baseline, "
        "search quality >= baseline - margin; lowest cost, tie-break higher quality"
    )

    pairs: list[dict[str, Any]] = []
    if selected is not None and not ctx.stopped:
        pairs = run_holdout(ctx, rows, holdout_ids, baseline, selected)
    if pairs:
        b_recs = [p["baseline"] for p in pairs]
        s_recs = [p["selected"] for p in pairs]
        diffs = [p["selected"]["quality"] - p["baseline"]["quality"] for p in pairs]
        iv = paired_bootstrap(
            diffs, int(cfg["bootstrap_resamples"]), int(cfg["seed"]) + 1
        )
        _, claim = noninferiority_claim(iv, float(cfg["margin"]), len(pairs))
        b_arm = arm_summary(b_recs, baseline, mock)
        s_arm = arm_summary(s_recs, selected, mock)
        saving = None
        if b_arm["cost_per_request_usd"] and s_arm["cost_per_request_usd"] is not None:
            saving = 100.0 * (
                1 - s_arm["cost_per_request_usd"] / b_arm["cost_per_request_usd"]
            )
        res["holdout"] = {
            "n": len(pairs),
            "baseline": b_arm,
            "selected": s_arm,
            "quality_diff_mean": sum(diffs) / len(diffs),
            "quality_diff_interval_95": iv,
            "claim": claim,
            "cost_saving_pct": saving,
            "pairs": pairs,
        }
    reason = None
    if ctx.stopped:
        reason = ctx.stopped
    elif not complete_search:
        reason = "search did not cover every row for every configuration"
    elif selected is not None and len(pairs) != len(holdout_ids):
        reason = "holdout did not cover every row"
    res["status"] = "complete" if reason is None else "incomplete"
    res["incomplete_reason"] = reason
    res["spend"] = {
        "actual_total_usd": ctx.spend_usd,
        "unknown_cost_calls": ctx.unknown_cost_calls,
        "semantics": CAP_DESCRIPTION,
    }
    res["search_records"] = [r for r in ctx.records if r["phase"] == "search"]
    write_outputs(out, res)
    return res


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mock", action="store_true", help="deterministic, zero network")
    ap.add_argument("--allow-unknown-price", action="store_true")
    args = ap.parse_args(argv)
    mock = args.mock or os.environ.get("TRAIGENT_MOCK_LLM", "").lower() in {
        "1",
        "true",
        "yes",
        "y",
    }
    try:
        cfg = load_config(args.config)
        res = run_diagnostic(
            cfg, Path(args.out), mock=mock, allow_unknown_price=args.allow_unknown_price
        )
    except DiagnosticError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"status: {res['status']}; report: {Path(args.out) / 'report.md'}")
    return 0 if res["status"] == "complete" else 1


if __name__ == "__main__":
    sys.exit(main())
