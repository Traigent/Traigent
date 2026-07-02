# Text-to-SQL Optimization Example

Demonstrates how to use Traigent to optimize a natural language to SQL
pipeline over a telecom (Amdocs-style) database schema.

## Schema

Four tables live in `schema.sql`:

| Table | Key columns |
|---|---|
| `customers` | `customer_id`, `name`, `city`, `status` |
| `subscriptions` | `subscription_id`, `customer_id`, `plan_name`, `monthly_rate`, `start_date` |
| `billing` | `billing_id`, `customer_id`, `amount`, `status`, `due_date` |
| `network_usage` | `usage_id`, `customer_id`, `call_minutes`, `data_gb`, `record_date` |

## Running the demo

**Mock mode** — no API key required, deterministic answers:

```bash
TRAIGENT_MOCK_LLM=true python examples/core/text-to-sql/run.py
```

**Real LLM mode — Anthropic** (requires `ANTHROPIC_API_KEY`):

```bash
python examples/core/text-to-sql/run.py
```

**Real LLM mode — OpenRouter** (requires `OPENROUTER_API_KEY`):

See [`run-plan.txt2sql-example.md`](run-plan.txt2sql-example.md) for a
pre-filled run plan. Use a **paid** model (e.g. `openai/gpt-4o-mini`) —
free-tier OpenRouter slots (`:free` suffix) hit 429 rate limits under the
optimizer's trial concurrency and will silently score 0% accuracy for all
their trials.

Traigent explores the configuration space (model, temperature, include_schema)
and reports the best-scoring combination according to the `sql_accuracy` metric.

## Eval set

The bundled eval set is at
`examples/datasets/text-to-sql/evaluation_set.jsonl` (40 rows). Each row is a
JSON object with two keys:

```json
{"input": "find customers with unpaid bills", "expected": "SELECT DISTINCT c.name FROM customers c JOIN billing b ON c.customer_id = b.customer_id WHERE b.status = 'unpaid'"}
```

Questions cover:

- Simple filters (`WHERE status = ...`, `LIKE`)
- Aggregations (`COUNT`, `SUM`, `AVG`, `MIN`, `MAX`)
- `GROUP BY` with and without `HAVING`
- Inner and left-join patterns
- Correlated subqueries and `NOT IN` NULL-safe patterns
- `ORDER BY ... LIMIT`
- Date arithmetic (`strftime`, `DATE(... '-30 days')`)
- Multi-table joins across all four tables

## Using SPIDER for rigorous benchmarking

The built-in eval set is intentionally small — enough to explore the
configuration space quickly. For production-grade text2SQL evaluation, the
SPIDER benchmark (Yale, 2018) is the standard choice: 10,181 questions across
200 databases and 138 domains.

**Steps to integrate SPIDER:**

1. Download Spider 1.0 from https://yale-lily.github.io/spider
   (requires agreeing to a non-commercial research license).

2. Convert the `train_spider.json` or `dev.json` rows to JSONL format:

   ```python
   import json, pathlib

   spider_dev = json.loads(pathlib.Path("spider/dev.json").read_text())
   with open("spider_eval.jsonl", "w") as f:
       for row in spider_dev:
           f.write(json.dumps({
               "input": row["question"],
               "expected": row["query"],
           }) + "\n")
   ```

3. Update the `eval_dataset=` path in `run.py` (or pass it at runtime):

   ```python
   @traigent.optimize(
       eval_dataset="path/to/spider_eval.jsonl",
       ...
   )
   def generate_sql(question: str) -> str:
       ...
   ```

4. Update the system prompt in `generate_sql` to include the relevant SPIDER
   database schema for each question. SPIDER questions reference many different
   databases, so you will need to look up the schema per `db_id`.

**License note:** SPIDER is released for non-commercial research use only. Do
not redistribute modified copies of the dataset or include SPIDER data in this
repository.

## Scoring

The `sql_accuracy` metric in `run.py` first attempts a normalized exact match
(case-insensitive, whitespace-collapsed). On a mismatch it falls back to
token-coverage partial credit (up to 0.5), scoring how many schema names from
the expected query appear in the generated query.

For SPIDER-style evaluation you may want to replace this with an
execution-accuracy metric: run both the expected and generated queries against
the actual database and compare result sets. That requires a SQLite (or
Postgres) instance loaded with each SPIDER database.
