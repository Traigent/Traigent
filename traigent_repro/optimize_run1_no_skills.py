#!/usr/bin/env python
"""Run #1 — Optimize the demo_sql_spider agent with Traigent, WITHOUT traigent-skills.

Built only from the Traigent SDK itself (README + bundled examples + docstrings).

  dataset : the demo's eval/spider_lite_30.jsonl, reshaped IN-MEMORY into Traigent
            examples {input:{question,schema}, output:<gold sql>, db_path, db_id}.
  agent   : the repo's own text2sql.agent.generate_sql (one litellm call).
  metric  : the repo's own execution_accuracy (run predicted+gold SQL on the
            vendored SQLite DB and compare result sets).
  knobs   : the agent's honest set — temperature, include_schema, prompt_style
            (+ model).

Env:
  TRAIGENT_MOCK_LLM=true    -> agent returns canned SQL, no LLM call, no cost.
  TR_EXEC_MODE=edge_analytics (default, local) | hybrid (portal-tracked)
  TR_MODELS="openrouter/openai/gpt-4o-mini,openrouter/openai/gpt-4o"
  TR_N=30
  TRAIGENT_PROJECT_ID=<project>   (hybrid only)
"""
from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

# Point this at a clone of github.com/Traigent/demo_sql_spider (the agent under test).
DEMO_ROOT = Path(os.environ.get("DEMO_SQL_SPIDER_ROOT") or (Path.cwd() / "demo_sql_spider")).resolve()
if not (DEMO_ROOT / "text2sql").is_dir():
    raise SystemExit(
        "Set DEMO_SQL_SPIDER_ROOT to a clone of https://github.com/Traigent/demo_sql_spider"
    )
sys.path.insert(0, str(DEMO_ROOT))

os.environ.setdefault("TRAIGENT_RESULTS_FOLDER", str(Path(__file__).parent / ".traigent_results_run1"))
os.environ.setdefault("TRAIGENT_COST_APPROVED", "true")

MOCK = os.getenv("TRAIGENT_MOCK_LLM", "").lower() in {"1", "true", "yes"}
EXEC_MODE = os.getenv("TR_EXEC_MODE", "edge_analytics")
MODELS = [m.strip() for m in os.getenv("TR_MODELS", "gpt-4o-mini").split(",") if m.strip()]
N = int(os.getenv("TR_N", "30"))

import traigent  # noqa: E402

from text2sql.agent import generate_sql as repo_generate_sql  # noqa: E402
from text2sql.dataset import load_examples, row_fields, sample  # noqa: E402
from text2sql.execaccuracy import execution_accuracy  # noqa: E402


def build_dataset() -> list[dict]:
    src = DEMO_ROOT / "eval" / "spider_lite_30.jsonl"
    rows = sample(load_examples(src), n=N, seed=0)
    out = []
    for raw in rows:
        r = row_fields(raw, DEMO_ROOT)  # resolves db_path to an absolute path
        # NOTE: extra TOP-LEVEL keys become example.metadata. A nested
        # {"metadata": {...}} key is NOT honoured for in-memory dicts (it lands
        # as metadata["metadata"]); only the file loader special-cases it.
        out.append({
            "input": {"question": r["question"], "schema": r["schema"]},
            "output": r["gold"],
            "db_path": r["db_path"],
            "db_id": r["db_id"],
        })
    return out


DATASET = build_dataset()


def exec_accuracy_metric(output, expected, metadata, **_) -> float:
    return execution_accuracy(output or "", expected, metadata["db_path"])


@traigent.optimize(
    eval_dataset=DATASET,
    objectives=["exec_accuracy"],          # <-- custom name; see README "no accuracy on portal"
    configuration_space={
        "model": MODELS,
        "temperature": [0.0, 0.7],
        "include_schema": ["true", "false"],
        "prompt_style": ["direct", "few_shot"],
    },
    metric_functions={"exec_accuracy": exec_accuracy_metric},
    injection_mode="context",
    execution_mode=EXEC_MODE,
)
def sql_agent(question: str, schema: str = "") -> str:
    cfg = traigent.get_config()
    if MOCK:
        return "SELECT 1"
    return repo_generate_sql(
        question, schema=schema,
        model=str(cfg.get("model", "gpt-4o-mini")),
        temperature=float(cfg.get("temperature", 0.0)),
        include_schema=str(cfg.get("include_schema", "true")).lower() == "true",
        prompt_style=str(cfg.get("prompt_style", "direct")),
    )


async def main() -> None:
    combos = len(MODELS) * 2 * 2 * 2
    print(f"RUN #1 (no skills) | examples={len(DATASET)} models={MODELS} "
          f"combos={combos} exec_mode={EXEC_MODE} mock={MOCK}")
    result = await sql_agent.optimize(algorithm="grid", max_trials=combos)
    print("best_config :", result.best_config)
    print("best_score  :", result.best_score)
    print("trials      :", len(result.trials))
    print("\nper-trial exec_accuracy:")
    rows = []
    for tr in result.trials:
        m = getattr(tr, "metrics", None) or {}
        s = getattr(tr, "score", None)
        s = s if s is not None else (m.get("exec_accuracy") if isinstance(m, dict) else None)
        cfg = getattr(tr, "config", getattr(tr, "configuration", {})) or {}
        rows.append((s or 0.0, {k: cfg.get(k) for k in ("model", "temperature", "include_schema", "prompt_style") if k in cfg}))
    for s, cfg in sorted(rows, key=lambda x: x[0], reverse=True):
        print(f"  {s:>6.1%}  {cfg}")


if __name__ == "__main__":
    asyncio.run(main())
