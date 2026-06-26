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

Dataset note:
    This demo ships with a small telecom (Amdocs-style) eval set covering
    the customers, subscriptions, billing, and network_usage tables (40 rows
    as of this writing). Questions span simple filters, aggregations, JOINs,
    GROUP BY / HAVING, subqueries, NULL handling, and date arithmetic.

    For production-scale text2SQL benchmarking, use the public SPIDER dataset
    (https://yale-lily.github.io/spider) — download Spider 1.0, convert rows
    to the same {"question": ..., "expected": ...} JSONL format (one object
    per line), and pass the path via eval_dataset= in the @traigent.optimize
    decorator or as a CLI argument. SPIDER is released under a non-commercial
    research license; do not redistribute modified copies.
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

    _sdk = os.environ.get("TRAIGENT_SDK_PATH")
    if _sdk:
        sys.path.insert(0, _sdk)
    else:
        module_path = Path(__file__).resolve()
        for depth in (2, 3):
            try:
                sys.path.append(str(module_path.parents[depth]))
            except IndexError:
                continue
    traigent = importlib.import_module("traigent")

DATA_ROOT = Path(__file__).resolve().parents[2] / "datasets" / "text-to-sql"
DATASET = str(DATA_ROOT / "evaluation_set.jsonl")
_schema_path = BASE / "schema.sql"
SCHEMA = _schema_path.read_text() if _schema_path.exists() else ""

if MOCK:
    try:
        traigent.initialize(offline=True)
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

    Returns 1.0 for a normalized exact match, partial credit (up to 0.5)
    based on coverage of expected table/column names, otherwise 0.0.
    """
    norm_out = _normalize_sql(output)
    norm_exp = _normalize_sql(expected)

    if norm_out == norm_exp:
        return 1.0

    # Partial credit: check that key tokens from the expected query appear
    exp_tokens = set(norm_exp.split())
    out_tokens = set(norm_out.split())
    # Table/column names are any word that is not a SQL keyword
    stop = {
        "select",
        "from",
        "where",
        "join",
        "on",
        "by",
        "and",
        "or",
        "not",
        "in",
        "as",
        "is",
        "null",
        "the",
        "a",
        "an",
        "*",
        ">",
        "<",
        "=",
        ">=",
        "<=",
        "<>",
        "(",
        ")",
        ",",
        ".",
        "'",
        "1",
        "5",
    }
    exp_names = {
        t for t in exp_tokens if t not in stop and not t.replace(".", "").isnumeric()
    }
    matched = sum(1 for t in exp_names if t in out_tokens)
    coverage = matched / len(exp_names) if exp_names else 0.0
    return round(0.5 * coverage, 3)


# ---------------------------------------------------------------------------
# Mock responses (deterministic; no API key required)
# ---------------------------------------------------------------------------

_MOCK_ANSWERS: dict[str, str] = {
    # --- original 15 rows ---
    "show all active customers": "SELECT * FROM customers WHERE status = 'active'",
    "count how many customers are in new york": "SELECT COUNT(*) FROM customers WHERE city = 'New York'",
    "find customers with unpaid bills": "SELECT DISTINCT c.name FROM customers c JOIN billing b ON c.customer_id = b.customer_id WHERE b.status = 'unpaid'",
    "get the total billing revenue": "SELECT SUM(amount) FROM billing",
    "list the top 5 customers by data usage": "SELECT c.name, SUM(nu.data_gb) AS total_data FROM customers c JOIN network_usage nu ON c.customer_id = nu.customer_id GROUP BY c.customer_id, c.name ORDER BY total_data DESC LIMIT 5",
    "find all premium plan subscribers": "SELECT * FROM subscriptions WHERE plan_name = 'premium'",
    "get average monthly rate per plan": "SELECT plan_name, AVG(monthly_rate) AS avg_rate FROM subscriptions GROUP BY plan_name",
    "count customers by city": "SELECT city, COUNT(*) AS customer_count FROM customers GROUP BY city ORDER BY customer_count DESC",
    "find overdue unpaid bills": "SELECT * FROM billing WHERE status = 'unpaid' AND due_date < CURRENT_DATE",
    "show total call minutes by customer": "SELECT c.name, SUM(nu.call_minutes) AS total_minutes FROM customers c JOIN network_usage nu ON c.customer_id = nu.customer_id GROUP BY c.customer_id, c.name ORDER BY total_minutes DESC",
    "find customers with no network usage": "SELECT name FROM customers WHERE customer_id NOT IN (SELECT DISTINCT customer_id FROM network_usage)",
    "list customers on the enterprise plan": "SELECT c.name, c.city FROM customers c JOIN subscriptions s ON c.customer_id = s.customer_id WHERE s.plan_name = 'enterprise'",
    "get the highest monthly billing amount": "SELECT MAX(amount) FROM billing",
    "show customers who have more than one subscription": "SELECT customer_id, COUNT(*) AS sub_count FROM subscriptions GROUP BY customer_id HAVING COUNT(*) > 1",
    "find the average data usage per customer": "SELECT c.name, AVG(nu.data_gb) AS avg_data FROM customers c JOIN network_usage nu ON c.customer_id = nu.customer_id GROUP BY c.customer_id, c.name",
    # --- expanded rows (rows 16–40) ---
    "list all distinct cities where we have customers": "SELECT DISTINCT city FROM customers ORDER BY city",
    "find the most expensive subscription plan": "SELECT plan_name, monthly_rate FROM subscriptions ORDER BY monthly_rate DESC LIMIT 1",
    "count the total number of subscriptions": "SELECT COUNT(*) FROM subscriptions",
    "show customers who started subscriptions in 2024": "SELECT DISTINCT c.name FROM customers c JOIN subscriptions s ON c.customer_id = s.customer_id WHERE strftime('%Y', s.start_date) = '2024'",
    "find all bills due this month": "SELECT * FROM billing WHERE strftime('%Y-%m', due_date) = strftime('%Y-%m', CURRENT_DATE)",
    "show total data usage per city": "SELECT c.city, SUM(nu.data_gb) AS total_data FROM customers c JOIN network_usage nu ON c.customer_id = nu.customer_id GROUP BY c.city ORDER BY total_data DESC",
    "find customers with both unpaid bills and no network usage": "SELECT DISTINCT c.name FROM customers c JOIN billing b ON c.customer_id = b.customer_id WHERE b.status = 'unpaid' AND c.customer_id NOT IN (SELECT DISTINCT customer_id FROM network_usage)",
    "show the minimum and maximum billing amounts": "SELECT MIN(amount) AS min_amount, MAX(amount) AS max_amount FROM billing",
    "list all customers sorted by name alphabetically": "SELECT * FROM customers ORDER BY name ASC",
    "find customers on plans costing more than 50 per month": "SELECT DISTINCT c.name, c.city FROM customers c JOIN subscriptions s ON c.customer_id = s.customer_id WHERE s.monthly_rate > 50",
    "show the count of paid versus unpaid bills": "SELECT status, COUNT(*) AS bill_count FROM billing GROUP BY status",
    "find customers with total call minutes over 1000": "SELECT c.name, SUM(nu.call_minutes) AS total_minutes FROM customers c JOIN network_usage nu ON c.customer_id = nu.customer_id GROUP BY c.customer_id, c.name HAVING SUM(nu.call_minutes) > 1000",
    "list plans ordered by average monthly rate descending": "SELECT plan_name, AVG(monthly_rate) AS avg_rate FROM subscriptions GROUP BY plan_name ORDER BY avg_rate DESC",
    "show the total amount billed to each customer": "SELECT c.name, SUM(b.amount) AS total_billed FROM customers c JOIN billing b ON c.customer_id = b.customer_id GROUP BY c.customer_id, c.name ORDER BY total_billed DESC",
    "find customers whose name starts with a": "SELECT * FROM customers WHERE name LIKE 'A%'",
    "count how many customers have at least one subscription": "SELECT COUNT(DISTINCT customer_id) FROM subscriptions",
    "show network usage records from the last 30 days": "SELECT * FROM network_usage WHERE record_date >= DATE(CURRENT_DATE, '-30 days')",
    "find the customer with the highest single bill": "SELECT c.name, b.amount FROM customers c JOIN billing b ON c.customer_id = b.customer_id ORDER BY b.amount DESC LIMIT 1",
    "list inactive customers with no unpaid bills": "SELECT c.name FROM customers c WHERE c.status = 'inactive' AND c.customer_id NOT IN (SELECT customer_id FROM billing WHERE status = 'unpaid')",
    "show average call minutes and average data usage per customer": "SELECT c.name, AVG(nu.call_minutes) AS avg_minutes, AVG(nu.data_gb) AS avg_data FROM customers c JOIN network_usage nu ON c.customer_id = nu.customer_id GROUP BY c.customer_id, c.name",
    "find cities with more than 3 customers": "SELECT city, COUNT(*) AS customer_count FROM customers GROUP BY city HAVING COUNT(*) > 3",
    "list customers along with their plan names and monthly rates": "SELECT c.name, s.plan_name, s.monthly_rate FROM customers c JOIN subscriptions s ON c.customer_id = s.customer_id ORDER BY c.name",
    "show total revenue from paid bills only": "SELECT SUM(amount) AS paid_revenue FROM billing WHERE status = 'paid'",
    "find customers who have network usage records but no subscription": "SELECT DISTINCT c.name FROM customers c JOIN network_usage nu ON c.customer_id = nu.customer_id WHERE c.customer_id NOT IN (SELECT DISTINCT customer_id FROM subscriptions)",
    "show the earliest subscription start date": "SELECT MIN(start_date) AS earliest_start FROM subscriptions",
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
        "model": ["claude-haiku-4-5-20251001", "claude-sonnet-4-6"],
        "temperature": [0.0, 0.2],
        "include_schema": ["true", "false"],
    },
    metric_functions={"sql_accuracy": sql_accuracy},
    injection_mode="seamless",
    offline=True,
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

    assert os.getenv(
        "ANTHROPIC_API_KEY"
    ), "Set ANTHROPIC_API_KEY or run with TRAIGENT_MOCK_LLM=true"

    from langchain_anthropic import ChatAnthropic
    from langchain_core.messages import HumanMessage, SystemMessage

    config = traigent.get_config()
    model = str(config.get("model", "claude-sonnet-4-6"))
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
    response = llm.invoke(
        [
            SystemMessage(content="\n".join(system_parts)),
            HumanMessage(content=question),
        ]
    )
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
        print("  - model: claude-haiku-4-5-20251001, claude-sonnet-4-6")
        print("  - temperature: 0.0, 0.2")
        print("  - include_schema: true, false")
        print(
            f"Mode: {'MOCK (no API key required)' if MOCK else 'REAL (requires ANTHROPIC_API_KEY)'}"
        )
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
            preferred = [
                "model",
                "temperature",
                "include_schema",
                "samples_count",
                "sql_accuracy",
                "cost",
                "duration",
            ]
            cols = [c for c in preferred if c in df.columns]
            if cols:
                df = df.sort_values("sql_accuracy", ascending=False)
                print("\nAggregated results:")
                print(df[cols].to_string(index=False))

        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nCancelled by user.")
        raise SystemExit(130) from None
