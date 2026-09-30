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
import sqlite3
import sys
import threading
import time
from collections import Counter, defaultdict
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime, UTC
from pathlib import Path
from typing import Any, Protocol

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"
METRIC_NAME = "normalized_exact_match"
SQL_METRIC_NAME = "sql_execution_match"
METRICS = (METRIC_NAME, SQL_METRIC_NAME)

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
    if cfg["metric"] not in METRICS:
        raise DiagnosticError(f"metric must be one of {list(METRICS)}")
    if cfg["metric"] == SQL_METRIC_NAME and not (
        cfg.get("schema_path") and cfg.get("seed_path")
    ):
        raise DiagnosticError(
            f"metric {SQL_METRIC_NAME!r} needs both schema_path and seed_path"
        )
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
    for key in ("samples_path", "schema_path", "seed_path"):
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


_FENCE = re.compile(r"```[a-zA-Z0-9_+-]*[ \t]*\n?(.*?)```", re.DOTALL)
_ORDER_BY = re.compile(r"\border\s+by\b", re.IGNORECASE)
_SQL_LITERALS_AND_COMMENTS = re.compile(
    r"'(?:[^']|'')*'|\"(?:[^\"]|\"\")*\"|--[^\n]*|/\*.*?\*/", re.DOTALL
)
_SQL_MAX_ROWS = 10_000  # more rows than this scores 0 ("result too large")
_SQL_MAX_VALUE_BYTES = 1_000_000  # per value; randomblob/zeroblob beyond fails
_READ_ONLY_START = re.compile(r"^(select|with)\b", re.IGNORECASE)
_SQL_MAX_PROGRESS_TICKS = 5000  # x1000 VM ops: aborts runaway queries


def strip_code_fences(text: str) -> str:
    """Return the body of the first markdown code fence, else the stripped text."""
    m = _FENCE.search(text)
    return (m.group(1) if m else text).strip()


def _row_key(row: tuple[Any, ...]) -> tuple[Any, ...]:
    return tuple(round(v, 6) if isinstance(v, float) else v for v in row)


def has_outer_order_by(sql: str) -> bool:
    """True if ORDER BY appears in the outermost query, ignoring string
    literals, quoted identifiers, comments and anything inside parentheses."""
    text = _SQL_LITERALS_AND_COMMENTS.sub(" ", sql)
    depth = 0
    top: list[str] = []
    for ch in text:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(0, depth - 1)
        elif depth == 0:
            top.append(ch)
    return bool(_ORDER_BY.search("".join(top)))


class SqlScorer:
    """Scores SQL by executing it against a seeded in-memory SQLite database.

    One connection is used for both the candidate and the reference query, so
    date functions such as ``date('now')`` see the same instant. The database is
    made read-only (``PRAGMA query_only``) after seeding and only ``SELECT`` /
    ``WITH`` statements are executed.
    """

    def __init__(self, schema_sql: str, seed_sql: str) -> None:
        self._db = sqlite3.connect(":memory:", check_same_thread=False)
        self._db.executescript(schema_sql)
        self._db.executescript(seed_sql)
        self._db.commit()
        self._db.execute("PRAGMA query_only = ON")
        with contextlib.suppress(AttributeError, ValueError, sqlite3.Error):
            # Python >= 3.11: SQLite rejects any string/blob over the limit.
            self._db.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, _SQL_MAX_VALUE_BYTES)
        self.last_error: str | None = None
        self._lock = threading.Lock()
        self._ref_cache: dict[str, list[tuple[Any, ...]] | None] = {}

    def _run(self, sql: str) -> list[tuple[Any, ...]]:
        stmt = strip_code_fences(sql).strip().rstrip(";").strip()
        if not _READ_ONLY_START.match(stmt):
            raise ValueError("only SELECT/WITH statements are executed")
        ticks = 0

        def guard() -> int:
            nonlocal ticks
            ticks += 1
            return 1 if ticks > _SQL_MAX_PROGRESS_TICKS else 0

        self._db.set_progress_handler(guard, 1000)
        try:
            rows = self._db.execute(stmt).fetchmany(_SQL_MAX_ROWS + 1)
        finally:
            self._db.set_progress_handler(None, 0)
        if len(rows) > _SQL_MAX_ROWS:
            raise ValueError("result too large")
        for row in rows:
            for v in row:
                if isinstance(v, (str, bytes)) and len(v) > _SQL_MAX_VALUE_BYTES:
                    raise ValueError("result too large")
        return rows

    def reference_rows(self, expected: str) -> list[tuple[Any, ...]] | None:
        """Rows of the expected query, or None if it errors or returns no rows."""
        with self._lock:
            if expected not in self._ref_cache:
                try:
                    rows = self._run(expected)
                except (sqlite3.Error, ValueError):
                    rows = []
                self._ref_cache[expected] = rows or None
            return self._ref_cache[expected]

    def score(self, output: str, expected: str) -> float | None:
        """1.0 match, 0.0 mismatch/error, None if the reference is invalid."""
        ref = self.reference_rows(expected)
        if ref is None:
            return None
        with self._lock:
            try:
                got = self._run(output)
            except (sqlite3.Error, ValueError) as exc:
                self.last_error = str(exc)
                return 0.0
        self.last_error = None
        if has_outer_order_by(expected):
            return (
                1.0 if [_row_key(r) for r in got] == [_row_key(r) for r in ref] else 0.0
            )
        same = Counter(_row_key(r) for r in got) == Counter(_row_key(r) for r in ref)
        return 1.0 if same else 0.0


def make_scorer(cfg: dict[str, Any]) -> Any:
    """Return ``scorer(output, expected) -> float | None`` for the configured metric."""
    if cfg["metric"] == SQL_METRIC_NAME:
        sql = SqlScorer(
            Path(cfg["schema_path"]).read_text(encoding="utf-8"),
            Path(cfg["seed_path"]).read_text(encoding="utf-8"),
        )
        return sql.score
    return score


def nearest_rank(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, math.ceil(pct / 100.0 * len(ordered)))
    return ordered[rank - 1]


def paired_bootstrap(
    diffs: list[float],
    resamples: int,
    seed: int,
    groups: list[Any] | None = None,
) -> tuple[float, float] | None:
    """Seeded 95% percentile interval of the mean paired difference.

    With ``groups`` (one key per difference, e.g. the input text) the bootstrap
    resamples whole groups, so duplicated inputs cannot narrow the interval; each
    resample's mean is the size-weighted mean over the drawn groups."""
    if not diffs:
        return None
    rng = random.Random(seed)
    if groups is None:
        groups = list(range(len(diffs)))
    clusters: dict[Any, list[float]] = {}
    for g, d in zip(groups, diffs, strict=True):
        clusters.setdefault(g, []).append(d)
    stats = [(sum(v), len(v)) for v in clusters.values()]
    m = len(stats)
    means = []
    for _ in range(resamples):
        drawn = [stats[rng.randrange(m)] for _ in range(m)]
        means.append(sum(t for t, _ in drawn) / sum(c for _, c in drawn))
    means.sort()
    lo = means[int(0.025 * (resamples - 1))]
    hi = means[int(math.ceil(0.975 * (resamples - 1)))]
    return lo, hi


def n_required(margin: float) -> int:
    """Independent pairs needed so that zero observed regressions rule out a
    regression rate >= margin with 97.5% one-sided confidence:
    ceil(ln(0.025) / ln(1 - margin))."""
    if not 0.0 < margin < 1.0:
        raise ValueError("margin must be strictly between 0 and 1")
    return math.ceil(math.log(0.025) / math.log(1.0 - margin))


def noninferiority_claim(
    interval: tuple[float, float] | None,
    margin: float,
    n: int,
    n_independent: int | None = None,
    complete: bool = True,
) -> tuple[bool, str]:
    """Claim non-inferiority only with a complete holdout, enough independent
    pairs and a bootstrap lower bound above -margin. The interval alone is never
    enough: with few pairs a zero-variance sample gives a degenerate interval."""
    if not complete:
        return False, "no inferential claim: the holdout is incomplete"
    indep = n if n_independent is None else n_independent
    need = n_required(margin)
    if indep >= need and interval is not None and interval[0] > -margin:
        return True, (
            f"non-inferior at margin {margin:g} (95% paired bootstrap lower bound "
            f"{interval[0]:+.3f} > {-margin:+.3f}; holdout n={n}, {indep} independent "
            f"pairs >= {need} required)"
        )
    return False, (
        f"non-inferiority unestablished (holdout n={n}; at least {need} independent "
        f"pairs needed at margin {margin:g})"
    )


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
        }  # usage.cost is returned by default; absent means unknown
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
        self.call_log: list[tuple[str, str]] = []  # (model, user) per call

    def complete(
        self, model, system, user, *, temperature, max_tokens, timeout_s
    ) -> Completion:
        self.calls += 1
        self.call_log.append((model, user))
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


def fetch_openrouter_prices(
    models: list[str], timeout_s: float = 30.0
) -> dict[str, tuple[float, float, str]]:
    """Per-token prices for ``models`` from OpenRouter's public /models listing.

    Never called in mock mode. Returns only the ids it found; callers fall back.
    """
    import httpx

    with httpx.Client(timeout=timeout_s) as client:
        resp = client.get(OPENROUTER_MODELS_URL)
    resp.raise_for_status()
    stamp = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    wanted = set(models)
    found: dict[str, tuple[float, float, str]] = {}
    for item in resp.json().get("data", []):
        mid = item.get("id")
        pricing = item.get("pricing") or {}
        if mid not in wanted:
            continue
        try:
            found[mid] = (
                float(pricing["prompt"]),
                float(pricing["completion"]),
                f"OpenRouter /models at {stamp}",
            )
        except (KeyError, TypeError, ValueError):
            continue
    return found


def price_per_token(
    model: str,
    mock: bool,
    table: dict[str, tuple[float, float, str]] | None = None,
) -> tuple[float, float, str] | None:
    """(input $/token, output $/token, provenance) or None if unknown."""
    if mock:
        _, pin, pout, _ = mock_profile(model)
        return pin / 1000.0, pout / 1000.0, "mock_profile"
    if table and model in table:
        return table[model]
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
    scorer: Any = None
    stopped: str | None = None
    records: list[dict[str, Any]] = field(default_factory=list)
    unpaired_holdout: list[dict[str, Any]] = field(default_factory=list)
    _sleep: Any = None

    def budget_left_for(self, phase: str) -> bool:
        limit = self.cap_usd - (self.holdout_reserve_usd if phase == "search" else 0.0)
        return self.spend_usd < limit


def run_request(
    ctx: Context, model: str, row_id: int, row: dict[str, str], phase: str
) -> dict[str, Any] | None:
    """The one place a model is invoked, scored and accounted for.

    The spend guard is checked before EVERY provider attempt (retries included)
    and each attempt's charge is booked as soon as it returns, success or
    failure. A charge that cannot be determined stops all further provider calls.
    Returns None (and sets ``ctx.stopped``) if no attempt was made. Latency is
    monotonic elapsed time around the whole request including retries.
    """
    cfg = ctx.cfg
    attempts = 1 + int(cfg["max_retries"])
    cost_total = 0.0
    unknown_attempts = 0
    made = 0
    ptoks = ctoks = 0
    tokens_known = True
    simulated = 0.0
    error: str | None = None
    text = ""
    start = time.monotonic()
    for _ in range(attempts):
        if ctx.stopped:
            break
        if not ctx.budget_left_for(phase):
            ctx.stopped = (
                f"spend guard: actual spend ${ctx.spend_usd:.6f} reached the limit "
                f"before a {phase} call"
            )
            break
        made += 1
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
            _book(ctx, exc.cost_usd)
            if exc.cost_usd is None:
                unknown_attempts += 1
            else:
                cost_total += exc.cost_usd
            tokens_known = False
            continue
        error = None
        simulated += comp.simulated_latency_s
        _book(ctx, comp.cost_usd)
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
    if made == 0:
        return None
    latency = time.monotonic() - start + simulated
    ctx.unknown_cost_calls += 1 if unknown_attempts else 0
    scorer = ctx.scorer or score
    quality = 0.0 if error else scorer(text, row["expected"])
    if error:
        # the reference may still be invalid; do not penalise both arms for it
        ref_check = scorer("", row["expected"])
        quality = None if ref_check is None else 0.0
    rec = {
        "phase": phase,
        "model": model,
        "row_id": row_id,
        "quality": quality,
        "reference_invalid": quality is None,
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


def _book(ctx: Context, cost: float | None) -> None:
    """Account one attempt's charge; an undeterminable charge stops the run."""
    if cost is None:
        ctx.stopped = (
            "unknown charge: a provider call returned no cost; further provider "
            "calls were stopped"
        )
    else:
        ctx.spend_usd += cost


# ---------------------------------------------------------------------------
# Search stats + selection
# ---------------------------------------------------------------------------


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(records)
    unknown = sum(1 for r in records if r["cost_usd"] is None)
    known = sum(r["known_cost_usd"] for r in records)
    lat = [r["latency_s"] for r in records]
    valid = [r["quality"] for r in records if r["quality"] is not None]
    return {
        "n": n,
        "n_scored": len(valid),
        "reference_invalid": n - len(valid),
        "quality": sum(valid) / len(valid) if valid else None,
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

    scorer = ctx.scorer or score
    metric_name = str(ctx.cfg["metric"])

    def metric(output: str, expected: str, **_: object) -> float:
        value = scorer(output, expected)  # informational for Traigent only
        return 0.0 if value is None else value

    @traigent.optimize(
        eval_dataset=str(dataset),
        objectives=create_default_objectives(
            [metric_name], orientations={metric_name: "maximize"}
        ),
        configuration_space={"model": list(models)},
        metric_functions={metric_name: metric},
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
    prices: dict[str, tuple[float, float, str]] | None = None,
) -> dict[str, Any]:
    in_tok = _approx_tokens(system_prompt) + int(avg_input_chars / 4) + 1
    out_tok = int(cfg["max_tokens"])  # conservative upper bound
    per_model: dict[str, float | None] = {}
    provenance: dict[str, str] = {}
    for model in models:
        price = price_per_token(model, mock, prices)
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
            ctx.unpaired_holdout.append({"row_id": rid, "baseline": b})
            break
        pairs.append({"row_id": rid, "baseline": b, "selected": c})
    return pairs


def _fmt_usd(v: float | None) -> str:
    return "unknown" if v is None else f"${v:.6f}"


def _fmt_s(v: float | None) -> str:
    return "n/a" if v is None else f"{v:.2f}s"


def arm_summary(
    records: list[dict[str, Any]],
    model: str,
    mock: bool,
    prices: dict[str, tuple[float, float, str]] | None = None,
) -> dict[str, Any]:
    s = summarize(records)
    s["model"] = model
    s["cost_per_1k_requests_usd"] = (
        None if s["cost_per_request_usd"] is None else s["cost_per_request_usd"] * 1000
    )
    est = None
    price = price_per_token(model, mock, prices)
    if price and records and all(r["prompt_tokens"] is not None for r in records):
        tot = sum(
            r["prompt_tokens"] * price[0] + r["completion_tokens"] * price[1]
            for r in records
        )
        est = tot / len(records) * 1000
    s["price_table_estimate_per_1k_requests_usd"] = est
    s["price_table_provenance"] = price[2] if price else "unknown"
    return s


def _metric_line(metric: str) -> str:
    if metric == SQL_METRIC_NAME:
        return (
            f"{SQL_METRIC_NAME} (same result rows as the reference query on a seeded "
            "in-memory SQLite test database; order-sensitive only if the reference "
            "has ORDER BY)"
        )
    return f"{METRIC_NAME} (lowercase, collapse whitespace, strip trailing ';')"


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
    if res.get("spend", {}).get("total_usd", 0) is None:
        lines += [
            "**Total spend is UNKNOWN: at least one provider call returned no cost.**",
            "",
        ]
    lines += [
        "## Setup",
        f"- Samples: {sp['n_total']} rows; search n={sp['n_search']}, "
        f"holdout n={sp['n_holdout']} (seed {cfg['seed']}, duplicates kept together)",
        f"- Metric: {_metric_line(res['metric'])}",
        f"- Baseline: {cfg['baseline']['model']}; candidates: "
        f"{', '.join(cfg['candidate_models'])}",
        f"- Temperature {cfg['temperature']}, max_tokens {cfg['max_tokens']}, "
        f"max_retries {cfg['max_retries']}, margin {cfg['margin']} (absolute)",
        f"- Spend cap USD {cfg['max_spend_usd']}: {CAP_DESCRIPTION}",
        "",
        "## Search (search split only)",
        "| model | quality | cost/request (provider) | errors | unknown-cost calls | "
        "reference_invalid rows |",
        "|---|---|---|---|---|---|",
    ]
    for model, s in res["search"]["per_config"].items():
        q = "n/a" if s["quality"] is None else f"{s['quality']:.3f}"
        lines.append(
            f"| {model} | {q} | {_fmt_usd(s['cost_per_request_usd'])} | "
            f"{s['errors']} | {s['unknown_cost_calls']} | "
            f"{s.get('reference_invalid', 0)} |"
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
            q = "n/a" if a["quality"] is None else f"{a['quality']:.3f}"
            lines.append(
                f"| {arm} | {a['model']} | {q} | "
                f"{_fmt_usd(a['cost_per_1k_requests_usd'])} | {est_s} | "
                f"{_fmt_s(a['latency_p50_s'])} | "
                f"{_fmt_s(a['latency_p95_s'])}{p95_label} | {a['errors']} |"
            )
        iv = h["quality_diff_interval_95"]
        dm = h["quality_diff_mean"]
        lines += [
            "",
            f"- Holdout n={h['n']} of {sp['n_holdout']} planned rows "
            f"({h['n_independent']} independent pairs; "
            f"{h['reference_invalid']} reference_invalid rows excluded)",
            "- Quality difference (selected - baseline) = "
            + ("n/a" if dm is None else f"{dm:+.3f}"),
            "- Paired bootstrap 95% interval (descriptive): "
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
        f"- Total spend (search + holdout, provider-reported): "
        f"{_fmt_usd(sp_['total_usd'])}",
        f"- Known subtotal (calls with a reported cost): "
        f"{_fmt_usd(sp_['known_subtotal_usd'])}",
        f"- Calls with unknown provider cost: {sp_['unknown_cost_calls']}",
        f"- Pre-flight estimate: {_fmt_usd(res['preflight']['total_usd'])}",
        "",
        "## Caveats",
        f"- Small samples: search n={sp['n_search']}, holdout n={sp['n_holdout']}. "
        "One row changes quality by 1/n; differences of a few points are noise.",
        *_metric_caveat(res["metric"]),
        "- Results describe the tested models, prompt and sample only; customer "
        "sampling bias limits extrapolation. This is not a production guarantee.",
        "",
    ]
    return "\n".join(lines)


def _metric_caveat(metric: str) -> list[str]:
    if metric == SQL_METRIC_NAME:
        return [
            "- SQL execution match compares result rows on a seeded test database; "
            "it is not proof of correctness on the customer's real data. Rows whose "
            "reference query errors or returns nothing are excluded as "
            "reference_invalid.",
        ]
    return [
        "- Normalized exact match measures textual agreement with the reference, "
        "not execution correctness (for SQL, an equivalent query can score 0).",
    ]


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
    mock = mock or cfg.get("provider") == "mock"
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
    if provider is None:
        provider = MockProvider(rows) if mock else OpenRouterProvider()
    prices: dict[str, tuple[float, float, str]] = {}
    if not mock and provider.name == "openrouter":
        try:
            prices = fetch_openrouter_prices(models)
        except Exception as exc:  # falls back to traigent pricing, then unknown
            print(
                f"warning: OpenRouter /models fetch failed ({type(exc).__name__}); "
                "falling back to Traigent pricing",
                file=sys.stderr,
            )
    est = estimate_cost(
        cfg,
        models,
        len(search_ids),
        len(holdout_ids),
        mock,
        system_prompt,
        avg_chars,
        prices,
    )
    res: dict[str, Any] = {
        "status": "frozen_plan",
        "incomplete_reason": None,
        "mock": mock,
        "provider": provider.name,
        "config": dict(cfg),
        "metric": cfg["metric"],
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
            "spend": {
                "total_usd": 0.0,
                "known_subtotal_usd": 0.0,
                "unknown_cost_calls": 0,
            },
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
        scorer=make_scorer(cfg),
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
    res["holdout_unpaired"] = ctx.unpaired_holdout  # spend stays traceable
    holdout_complete = (
        selected is not None and len(pairs) == len(holdout_ids) and not ctx.stopped
    )
    if pairs:
        b_recs = [p["baseline"] for p in pairs]
        s_recs = [p["selected"] for p in pairs]
        valid = [
            p
            for p in pairs
            if p["selected"]["quality"] is not None
            and p["baseline"]["quality"] is not None
        ]
        diffs = [p["selected"]["quality"] - p["baseline"]["quality"] for p in valid]
        n_indep = len({rows[p["row_id"]]["input"] for p in valid})
        iv = None
        dmean = None
        if holdout_complete and diffs:
            iv = paired_bootstrap(
                diffs,
                int(cfg["bootstrap_resamples"]),
                int(cfg["seed"]) + 1,
                groups=[rows[p["row_id"]]["input"] for p in valid],
            )
            dmean = sum(diffs) / len(diffs)
        _, claim = noninferiority_claim(
            iv, float(cfg["margin"]), len(pairs), n_indep, holdout_complete
        )
        b_arm = arm_summary(b_recs, baseline, mock, prices)
        s_arm = arm_summary(s_recs, selected, mock, prices)
        saving = None
        if (
            holdout_complete
            and b_arm["cost_per_request_usd"]
            and s_arm["cost_per_request_usd"] is not None
        ):
            saving = 100.0 * (
                1 - s_arm["cost_per_request_usd"] / b_arm["cost_per_request_usd"]
            )
        res["holdout"] = {
            "n": len(pairs),
            "n_independent": n_indep,
            "n_required_independent": n_required(float(cfg["margin"])),
            "complete": holdout_complete,
            "reference_invalid": len(pairs) - len(valid),
            "baseline": b_arm,
            "selected": s_arm,
            "quality_diff_mean": dmean,
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
        "total_usd": None if ctx.unknown_cost_calls else ctx.spend_usd,
        "known_subtotal_usd": ctx.spend_usd,
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
