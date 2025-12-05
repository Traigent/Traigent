#!/usr/bin/env python3
"""Smoke test: run a real_api optimization and verify DB measures.

This script:
- Runs a tiny optimization using a function that calls a real LLM (OpenAI v1).
- Requires OPENAI_API_KEY set and access to your backend DB.
- Queries the database for the latest configuration_runs entry and prints measures.

Env vars you can set:
- OPENAI_API_KEY: API key for OpenAI (required for real metrics)
- TRAIGENT_API_URL: Backend base URL if not default (optional)
- DB_URL: Postgres URL (e.g., postgresql://user:pass@host:5432/optigen)

Usage:
  python scripts/smoke/smoke_real_api_measures.py
"""

import asyncio
import json
import os
import sys
import tempfile
from datetime import datetime, timezone


def _ensure_openai():
    try:
        import openai  # noqa: F401
        from openai import OpenAI  # noqa: F401

        return True
    except Exception as e:
        print(f"OpenAI SDK not available: {e}")
        print("Install with: pip install openai>=1.0.0")
        return False


def _ensure_asyncpg():
    try:
        import asyncpg  # noqa: F401

        return True
    except Exception:
        return False


def _make_dataset_file() -> str:
    data = [
        {"input": {"text": "Reply with the single word: ok"}, "output": None},
        {"input": {"text": "Say: hello"}, "output": None},
    ]
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False)
    for item in data:
        tmp.write(json.dumps(item) + "\n")
    tmp.flush()
    tmp.close()
    return tmp.name


def _build_optimized_function(dataset_path: str):
    # Real LLM call using OpenAI v1 SDK; returns full response object with usage
    from openai import OpenAI

    import traigent

    client = OpenAI()

    @traigent.optimize(
        eval_dataset=dataset_path,
        objectives=["accuracy"],
        configuration_space={
            "model": [os.environ.get("OPENAI_MODEL", "gpt-4o-mini")],
            "temperature": [0.0],
            "max_tokens": [64],
        },
        execution_mode="standard",  # triggers real_api path and per-example measures
        minimal_logging=False,
    )
    def real_llm(
        text: str,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        max_tokens: int = 64,
    ):
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": text}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        # Return full response so metrics extractor can read usage
        return resp

    return real_llm


async def _run_optimization():
    dataset = _make_dataset_file()
    func = _build_optimized_function(dataset)
    print("Running optimization (1 trial) with real LLM call...")
    result = await func.optimize(algorithm="grid", max_trials=1)
    print("Optimization complete.")
    print(f"Best score: {result.best_score}")
    return result


async def _query_db():
    db_url = os.environ.get(
        "DB_URL", "postgresql://optigen:optigen_local@localhost:5432/optigen"
    )
    if not _ensure_asyncpg():
        print("asyncpg not installed; attempting psql fallback...")
        import subprocess

        try:
            print("Latest experiments:")
            cmd = [
                "psql",
                db_url,
                "-c",
                "SELECT id, name, created_at FROM experiments ORDER BY created_at DESC LIMIT 3",
            ]
            subprocess.run(cmd, check=False)
            print("\nLatest configuration runs for latest experiment (measure sample):")
            cmd2 = [
                "psql",
                db_url,
                "-c",
                (
                    "SELECT id, created_at, left(measures::text, 200) AS measures_sample "
                    "FROM configuration_runs "
                    "WHERE experiment_id = (SELECT id FROM experiments ORDER BY created_at DESC LIMIT 1) "
                    "ORDER BY created_at DESC LIMIT 5"
                ),
            ]
            subprocess.run(cmd2, check=False)
        except Exception as e:
            print(f"psql fallback failed: {e}")
        return

    import asyncpg  # type: ignore

    conn = await asyncpg.connect(db_url)
    try:
        exp = await conn.fetchrow(
            """
            SELECT id, name, created_at
            FROM experiments
            ORDER BY created_at DESC
            LIMIT 1
            """
        )

        if not exp:
            print("No experiments found in DB.")
            return

        print(
            f"Latest experiment: id={exp['id']}, name={exp['name']}, created_at={exp['created_at']}"
        )

        runs = await conn.fetch(
            """
            SELECT id, measures, created_at
            FROM configuration_runs
            WHERE experiment_id = $1
            ORDER BY created_at DESC
            LIMIT 5
            """,
            exp["id"],
        )

        if not runs:
            print("No configuration_runs found for latest experiment.")
            return

        print(f"Found {len(runs)} configuration runs. Inspecting most recent:")
        run = runs[0]
        measures = run["measures"] or []
        print(f"Run id={run['id']}, created_at={run['created_at']}")
        print(f"Measures count: {len(measures)}")

        if measures:
            print("First measure sample:")
            first = measures[0]
            print(json.dumps(first, indent=2))
            # Basic presence checks for real_api metrics
            keys = set(first.keys())
            expected_any = {"input_tokens", "output_tokens", "total_tokens"}
            latency_keys = {"response_time", "response_time_ms"}
            has_tokens = bool(expected_any & keys)
            has_latency = bool(latency_keys & keys)
            print(f"Has token metrics: {has_tokens}")
            print(f"Has latency: {has_latency}")
        else:
            print("Measures array is empty. Investigate orchestrator/evaluator paths.")
    finally:
        await conn.close()


async def main():
    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is not set; cannot perform real_api smoke test.")
        sys.exit(1)
    if not _ensure_openai():
        sys.exit(1)

    print(f"Starting smoke test at {datetime.now(timezone.utc).isoformat()}")
    await _run_optimization()
    print("\nQuerying DB for latest measures...")
    await _query_db()


if __name__ == "__main__":
    asyncio.run(main())
