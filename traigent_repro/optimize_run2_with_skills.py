#!/usr/bin/env python
"""Run #2 — Optimize the demo_sql_spider agent WITH the traigent-skills as guide.

Follows traigent-boost-agent + traigent-composite-knobs + traigent-configuration-space:
  - shape = single LLM call with sampling upside.
  - TVARs from recommend_configuration_space("code_gen"): schema_context,
    generation_path, candidate_count (directly-applicable subset).
  - composite = self_consistency, implemented MANUALLY (n-sampling + majority vote)
    rather than the composite factory, because the factory (output, metrics)
    tuple-return path does NOT invoke custom metric_functions (documented known
    SDK issue) and our metric needs db_path (built-in evaluator cannot score it).
  - same metric/dataset as run #1 for a fair comparison.

Env: same switches as optimize_run1_no_skills.py.
"""
from __future__ import annotations

import asyncio
import os
import re
import sys
from collections import Counter
from pathlib import Path

DEMO_ROOT = Path(os.environ.get("DEMO_SQL_SPIDER_ROOT") or (Path.cwd() / "demo_sql_spider")).resolve()
if not (DEMO_ROOT / "text2sql").is_dir():
    raise SystemExit(
        "Set DEMO_SQL_SPIDER_ROOT to a clone of https://github.com/Traigent/demo_sql_spider"
    )
sys.path.insert(0, str(DEMO_ROOT))

os.environ.setdefault("TRAIGENT_RESULTS_FOLDER", str(Path(__file__).parent / ".traigent_results_run2"))
os.environ.setdefault("TRAIGENT_COST_APPROVED", "true")
os.environ.setdefault("TRAIGENT_RUN_COST_LIMIT", "6.0")

MOCK = os.getenv("TRAIGENT_MOCK_LLM", "").lower() in {"1", "true", "yes"}
EXEC_MODE = os.getenv("TR_EXEC_MODE", "edge_analytics")
MODELS = [m.strip() for m in os.getenv("TR_MODELS", "gpt-4o-mini,gpt-4o").split(",") if m.strip()]
N = int(os.getenv("TR_N", "30"))

import traigent  # noqa: E402

from text2sql.agent import _strip_sql  # noqa: E402
from text2sql.dataset import load_examples, row_fields, sample  # noqa: E402
from text2sql.execaccuracy import execution_accuracy  # noqa: E402
from text2sql.prompts import build_messages  # noqa: E402


def build_dataset() -> list[dict]:
    src = DEMO_ROOT / "eval" / "spider_lite_30.jsonl"
    rows = sample(load_examples(src), n=N, seed=0)
    out = []
    for raw in rows:
        r = row_fields(raw, DEMO_ROOT)
        out.append({"input": {"question": r["question"], "schema": r["schema"]},
                    "output": r["gold"], "db_path": r["db_path"], "db_id": r["db_id"]})
    return out


DATASET = build_dataset()


def exec_accuracy_metric(output, expected, metadata, **_) -> float:
    return execution_accuracy(output or "", expected, metadata["db_path"])


_COT_SYSTEM = (
    "You are an expert SQL assistant. Think step by step about the database schema "
    "and the query plan needed to answer the question. Then output the FINAL answer "
    "as a single SQL query inside a ```sql code block. Only the fenced query is used."
)


def _messages_for(question, schema, schema_context, generation_path):
    include_schema = schema_context != "none"
    if generation_path == "query_plan_cot":
        system = _COT_SYSTEM
        if include_schema and schema:
            system += f"\n\nDatabase schema:\n{schema}"
        return [{"role": "system", "content": system}, {"role": "user", "content": question}]
    return build_messages(question, schema=schema, include_schema=include_schema, prompt_style="direct")


def _norm_sql(sql): return re.sub(r"\s+", " ", sql.strip().rstrip(";").lower())


def _majority_vote(candidates):
    cleaned = [c for c in (s.strip() for s in candidates) if c]
    if not cleaned:
        return ""
    winner = Counter(_norm_sql(c) for c in cleaned).most_common(1)[0][0]
    for c in cleaned:
        if _norm_sql(c) == winner:
            return c
    return cleaned[0]


def boosted_generate_sql(question, schema, cfg) -> str:
    model = str(cfg.get("model", "gpt-4o-mini"))
    schema_context = str(cfg.get("schema_context", "full_ddl_fk"))
    generation_path = str(cfg.get("generation_path", "direct"))
    k = int(cfg.get("candidate_count", 1))
    temperature = 0.0 if k <= 1 else 0.7
    messages = _messages_for(question, schema, schema_context, generation_path)
    max_tokens = 256 if generation_path == "direct" else 512

    import litellm

    def _one_call(n):
        resp = litellm.completion(model=model, messages=messages, temperature=temperature,
                                  max_tokens=max_tokens, n=n)
        return [_strip_sql(ch["message"]["content"] or "") for ch in resp["choices"]]

    if k <= 1:
        return _one_call(1)[0]
    candidates = []
    try:
        candidates = _one_call(k)         # one n-sampled call...
    except Exception:                      # ...or fall back to looping if n>1 unsupported
        candidates = []
    while len(candidates) < k:
        candidates.extend(_one_call(1))
    return _majority_vote(candidates[:k])


@traigent.optimize(
    eval_dataset=DATASET,
    objectives=["exec_accuracy"],          # <-- custom name; coerced on the portal (see README)
    configuration_space={
        "model": MODELS,
        "schema_context": ["none", "full_ddl_fk"],
        "generation_path": ["direct", "query_plan_cot"],
        "candidate_count": [1, 3],
    },
    metric_functions={"exec_accuracy": exec_accuracy_metric},
    injection_mode="context",
    execution_mode=EXEC_MODE,
)
def sql_agent_boosted(question: str, schema: str = "") -> str:
    cfg = traigent.get_config()
    if MOCK:
        return "SELECT 1"
    return boosted_generate_sql(question, schema, cfg)


async def main() -> None:
    combos = len(MODELS) * 2 * 2 * 2
    print(f"RUN #2 (skill-guided) | examples={len(DATASET)} models={MODELS} "
          f"combos={combos} exec_mode={EXEC_MODE} mock={MOCK}")
    result = await sql_agent_boosted.optimize(algorithm="grid", max_trials=combos)
    print("best_config :", result.best_config)
    print("best_score  :", result.best_score)
    print("trials      :", len(result.trials))
    print("stop_reason :", getattr(result, "stop_reason", None))
    try:
        print("success_rate:", f"{result.success_rate:.0%}")
    except Exception:
        pass
    tc = getattr(result, "total_cost", None)
    if tc is not None:
        print("total_cost  :", f"${tc:.4f}")
    try:
        from traigent.utils.insights import get_optimization_insights
        pi = get_optimization_insights(result).get("parameter_insights")
        if pi:
            print("parameter_insights:", {k: round(v.get("performance_impact", 0), 3) for k, v in pi.items()})
    except Exception as exc:
        print("[insights unavailable:", exc, "]")
    print("\nper-trial exec_accuracy:")
    rows = []
    for tr in result.trials:
        m = getattr(tr, "metrics", None) or {}
        s = getattr(tr, "score", None)
        s = s if s is not None else (m.get("exec_accuracy") if isinstance(m, dict) else None)
        cfg = getattr(tr, "config", getattr(tr, "configuration", {})) or {}
        rows.append((s or 0.0, {k: cfg.get(k) for k in ("model", "schema_context", "generation_path", "candidate_count") if k in cfg}))
    for s, cfg in sorted(rows, key=lambda x: x[0], reverse=True):
        print(f"  {s:>6.1%}  {cfg}")


if __name__ == "__main__":
    asyncio.run(main())
