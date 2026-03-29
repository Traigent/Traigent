#!/usr/bin/env python3
"""
Text-to-SQL Query Generation - Optimize natural language to SQL conversion.

Demonstrates how to use Traigent to tune a text-to-SQL pipeline over a
telecom (Amdocs-style) database schema. Evaluates generated SQL using a
normalized exact-match metric and a keyword-coverage fallback.

Run without an API key (mock mode):
    TRAIGENT_MOCK_LLM=true python examples/core/text-to-sql/run.py

Run with a real LLM (requires ANTHROPIC_API_KEY):
    python examples/core/text-to-sql/run.py
"""

from __future__ import annotations

import asyncio
import os
import re
import sys
from pathlib import Path

MOCK = str(os.getenv("TRAIGENT_MOCK_LLM", "")).lower() in {"1", "true", "yes", "y"}
BASE = Path(__file__).parent
if MOCK:
    os.environ["HOME"] = str(BASE)
    results_dir = BASE / ".traigent_local"
    results_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TRAIGENT_RESULTS_FOLDER"] = str(results_dir)

os.environ.setdefault("TRAIGENT_COST_APPROVED", "true")

try:
    import traigent
except ImportError:  # pragma: no cover
    import importlib

    module_path = Path(__file__).resolve()
    for depth in (2, 3):
        try:
            sys.path.append(str(module_path.parents[depth]))
        except IndexError:
            continue
    traigent = importlib.import_module("traigent")

DATA_ROOT = Path(__file__).resolve().parents[2] / "datasets" / "text-to-sql"
DATASET = str(DATA_ROOT / "evaluation_set.jsonl")
SCHEMA = (BASE / "schema.sql").read_text()

if MOCK:
    try:
        traigent.initialize(execution_mode="edge_analytics")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Evaluation metric
# ---------------------------------------------------------------------------

def _normalize_sql(sql: str) -> str:
    """Lowercase, collapse whitespace, strip trailing semicolon."""
    sql = sql.strip().rstrip(";").lower()
    return re.sub(r"\s+", " ", sql)


def sql_accuracy(output: str, expected: str, **_: object) -> float:
    """Score SQL output against expected query.

    Returns 1.0 for a normalized exact match, 0.5 if all expected SQL
    keywords and table names are present, otherwise 0.0.
    """
    norm_out = _normalize_sql(output)
    norm_exp = _normalize_sql(expected)

    if norm_out == norm_exp:
        return 1.0

    # Partial credit: check that key tokens from the expected query appear
    sql_keywords = {"select", "from", "where", "join", "on", "group by",
                    "order by", "having", "limit", "count", "sum", "avg",
                    "max", "min", "distinct", "not in"}
    exp_tokens = set(norm_exp.split())
    out_tokens = set(norm_out.split())
    # Table/column names are any word that is not a SQL keyword
    stop = {"select", "from", "where", "join", "on", "by", "and", "or",
            "not", "in", "as", "is", "null", "the", "a", "an", "*", ">",
            "<", "=", ">=", "<=", "<>", "(", ")", ",", ".", "'", "1", "5"}
    exp_names = {t for t in exp_tokens if t not in stop and not t.replace(".", "").isnumeric()}
    matched = sum(1 for t in exp_names if t in out_tokens)
    coverage = matched / len(exp_names) if exp_names else 0.0
    return round(0.5 * coverage, 3)


# ---------------------------------------------------------------------------
# Mock responses (deterministic; no API key required)
# ---------------------------------------------------------------------------

_MOCK_ANSWERS: dict[str, str] = {
    "show all active customers":
        "SELECT * FROM customers WHERE status = 'active'",
    "count how many customers are in new york":
        "SELECT COUNT(*) FROM customers WHERE city = 'New York'",
    "find customers with unpaid bills":
        "SELECT DISTINCT c.name FROM customers c JOIN billing b ON c.customer_id = b.customer_id WHERE b.status = 'unpaid'",
    "get the total billing revenue":
        "SELECT SUM(amount) FROM billing",
    "list the top 5 customers by data usage":
        "SELECT c.name, SUM(nu.data_gb) AS total_data FROM customers c JOIN network_usage nu ON c.customer_id = nu.customer_id GROUP BY c.customer_id, c.name ORDER BY total_data DESC LIMIT 5",
    "find all premium plan subscribers":
        "SELECT * FROM subscriptions WHERE plan_name = 'premium'",
    "get average monthly rate per plan":
        "SELECT plan_name, AVG(monthly_rate) AS avg_rate FROM subscriptions GROUP BY plan_name",
    "count customers by city":
        "SELECT city, COUNT(*) AS customer_count FROM customers GROUP BY city ORDER BY customer_count DESC",
    "find overdue unpaid bills":
        "SELECT * FROM billing WHERE status = 'unpaid' AND due_date < CURRENT_DATE",
    "show total call minutes by customer":
        "SELECT c.name, SUM(nu.call_minutes) AS total_minutes FROM customers c JOIN network_usage nu ON c.customer_id = nu.customer_id GROUP BY c.customer_id, c.name ORDER BY total_minutes DESC",
    "find customers with no network usage":
        "SELECT name FROM customers WHERE customer_id NOT IN (SELECT DISTINCT customer_id FROM network_usage)",
    "list customers on the enterprise plan":
        "SELECT c.name, c.city FROM customers c JOIN subscriptions s ON c.customer_id = s.customer_id WHERE s.plan_name = 'enterprise'",
    "get the highest monthly billing amount":
        "SELECT MAX(amount) FROM billing",
    "show customers who have more than one subscription":
        "SELECT customer_id, COUNT(*) AS sub_count FROM subscriptions GROUP BY customer_id HAVING COUNT(*) > 1",
    "find the average data usage per customer":
        "SELECT c.name, AVG(nu.data_gb) AS avg_data FROM customers c JOIN network_usage nu ON c.customer_id = nu.customer_id GROUP BY c.customer_id, c.name",
}


def _mock_generate_sql(question: str) -> str:
    key = question.strip().lower()
    return _MOCK_ANSWERS.get(key, f"SELECT * FROM customers -- unknown: {question}")


# ---------------------------------------------------------------------------
# Optimized function
# ---------------------------------------------------------------------------

@traigent.optimize(
    eval_dataset=DATASET,
    objectives=["sql_accuracy"],
    configuration_space={
        "model": ["claude-3-haiku-20240307", "claude-3-5-sonnet-20241022"],
        "temperature": [0.0, 0.2],
        "include_schema": ["true", "false"],
    },
    metric_functions={"sql_accuracy": sql_accuracy},
    injection_mode="seamless",
    execution_mode="edge_analytics",
)
def generate_sql(question: str) -> str:
    """Convert a natural language question into a SQL query.

    Args:
        question: The natural language database question.

    Returns:
        A syntactically valid SQL query string.
    """
    if MOCK:
        return _mock_generate_sql(question)

    assert os.getenv("ANTHROPIC_API_KEY"), (
        "Set ANTHROPIC_API_KEY or run with TRAIGENT_MOCK_LLM=true"
    )

    from langchain_anthropic import ChatAnthropic
    from langchain_core.messages import HumanMessage, SystemMessage

    config = traigent.get_config()
    model = str(config.get("model", "claude-3-5-sonnet-20241022"))
    temperature = float(config.get("temperature", 0.0))
    include_schema = str(config.get("include_schema", "true")) == "true"

    system_parts = [
        "You are an expert SQL assistant. Convert the user's question into a valid SQL query.",
        "Return ONLY the SQL query — no explanation, no markdown, no code fences.",
    ]
    if include_schema:
        system_parts.append(f"\nDatabase schema:\n{SCHEMA}")

    llm = ChatAnthropic(
        model_name=model,
        temperature=temperature,
        timeout=None,
        stop=None,
    )
    response = llm.invoke([
        SystemMessage(content="\n".join(system_parts)),
        HumanMessage(content=question),
    ])
    return str(response.content).strip()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        print("=" * 60)
        print("Text-to-SQL Optimization Example")
        print("=" * 60)
        print("Schema: Telecom customer database (customers, subscriptions,")
        print("        billing, network_usage)")
        print("Objective: sql_accuracy (normalize + exact-match scoring)")
        print("Configuration space:")
        print("  - model: claude-3-haiku, claude-3-5-sonnet")
        print("  - temperature: 0.0, 0.2")
        print("  - include_schema: true, false")
        print(f"Mode: {'MOCK (no API key required)' if MOCK else 'REAL (requires ANTHROPIC_API_KEY)'}")
        print("-" * 60)

        async def main() -> None:
            trials = 12  # covers all 8 config combos + oversampling
            result = await generate_sql.optimize(algorithm="grid", max_trials=trials)

            print("\n" + "=" * 60)
            print("OPTIMIZATION COMPLETE")
            print("=" * 60)
            print(f"Best config : {result.best_config}")
            print(f"Best score  : {result.best_score:.3f}")
            print(f"Total trials: {len(result.trials)}")

            df = result.to_aggregated_dataframe(primary_objective="sql_accuracy")
            preferred = ["model", "temperature", "include_schema",
                         "samples_count", "sql_accuracy", "cost", "duration"]
            cols = [c for c in preferred if c in df.columns]
            if cols:
                df = df.sort_values("sql_accuracy", ascending=False)
                print("\nAggregated results:")
                print(df[cols].to_string(index=False))

        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nCancelled by user.")
        raise SystemExit(130) from None
