# Traigent optimization repro — `demo_sql_spider` (text2SQL), and why accuracy isn't shown on the portal

Reproduction of two real Traigent optimization runs over the **`demo_sql_spider`**
NL→SQL agent, run on **`execution_mode="hybrid"`** (local trials + portal tracking).
This harness lives in the SDK repo; the agent under test is a separate clone
(`github.com/Traigent/demo_sql_spider`), located via `DEMO_SQL_SPIDER_ROOT`.
It also documents an SDK behaviour: **a custom-named objective/metric (`exec_accuracy`)
never appears under its own name on the portal** — the portal shows only a blended
"Score". Root cause + fix are in [§5](#5-why-no-accuracy-shows-on-the-portal).

Companion GitHub issue: see the issue that links here.

---

## 0. Environment

| | |
|---|---|
| Traigent SDK | `0.13.0` (installed from `github.com/Traigent/Traigent`; bundles `litellm==1.87.1`) |
| Python | 3.12 |
| Provider | OpenAI models via **OpenRouter** through litellm (`openrouter/openai/gpt-4o-mini`, `openrouter/openai/gpt-4o`) |
| Execution | `execution_mode="hybrid"` → backend `https://portal.traigent.ai`, `/api/v1` |
| Agent | the demo's `text2sql.agent.generate_sql` (one litellm call) |
| Metric | the demo's `text2sql.execaccuracy.execution_accuracy` (runs predicted + gold SQL on the vendored SQLite DB, compares result sets) |
| Dataset | the demo's `eval/spider_lite_30.jsonl` (30 Spider-lite examples) |

> **Only the Traigent SDK is installed** — nothing else. It bundles `litellm`, which is
> all the agent needs. The demo's own `requirements.txt` (deepeval/langfuse) is NOT used.

## 1. Setup

```bash
# 1. clone both repos
git clone https://github.com/Traigent/demo_sql_spider.git          # the agent under test
git clone -b repro/portal-custom-metric-not-shown \
    https://github.com/Traigent/Traigent.git                       # this harness lives here

# 2. venv + install the SDK only (it bundles litellm; nothing else is needed)
python -m venv .venv && . .venv/bin/activate
pip install -U pip
pip install ./Traigent            # or: pip install "git+https://github.com/Traigent/Traigent.git"

# 3. point the harness at the agent clone
export DEMO_SQL_SPIDER_ROOT="$PWD/demo_sql_spider"
cd Traigent                        # run the commands below from the SDK repo root
```

## 2. Credentials (env vars — never commit them)

```bash
export OPENROUTER_API_KEY="sk-or-v1-..."     # an OpenRouter INFERENCE key (not a provisioning key)
export TRAIGENT_API_KEY="uk_..."             # a Traigent portal key scoped to your project
export TRAIGENT_PROJECT_ID="project_..."     # the project you want experiments to land in
```

Notes that cost real debugging time during the original run:
- An OpenRouter **provisioning/management** key returns `200` on `/api/v1/auth/key` but
  `401 {"error":{"message":"User not found."}}` on `/chat/completions`. You need a plain
  **inference** key (`is_management_key:false`, `is_provisioning_key:false`).
- A Traigent key only sees its **own** project. If the portal UI shows a different project
  than the key's, experiments won't appear where you're looking (`PROJECT_NOT_FOUND`).
  Create the key from inside the project you intend to view, and pin `TRAIGENT_PROJECT_ID`.

## 3. Validate with no cost first (recommended)

```bash
# Layer 1 — dataset/metric/db_path plumbing (no LLM, no cost). Expect all rows = 1.0:
python traigent_repro/selftest_metric.py

# Layer 2 — full Traigent wiring in mock mode (no keys, no cost). Expect 8/16 trials, 0 failures:
TRAIGENT_MOCK_LLM=true TR_EXEC_MODE=edge_analytics TR_N=4 TR_MODELS=gpt-4o-mini \
  python traigent_repro/optimize_run1_no_skills.py
```

## 4. The two real runs (portal-tracked)

```bash
# RUN #1 — no skills: model × temperature × include_schema × prompt_style (16-config grid)
TR_EXEC_MODE=hybrid TR_N=30 \
  TR_MODELS="openrouter/openai/gpt-4o-mini,openrouter/openai/gpt-4o" \
  python traigent_repro/optimize_run1_no_skills.py

# RUN #2 — skill-guided: model × schema_context × generation_path(CoT) × candidate_count(self-consistency)
TR_EXEC_MODE=hybrid TR_N=30 \
  TR_MODELS="openrouter/openai/gpt-4o-mini,openrouter/openai/gpt-4o" \
  python traigent_repro/optimize_run2_with_skills.py
```

### Results observed (execution accuracy = exact, deterministic; computed locally)

**Run #1 (no skills)** — best **60.0%** · `gpt-4o, include_schema=true, direct, temp 0`

| include_schema | exec_accuracy range |
|---|---|
| `true`  | 46.7% – 60.0% |
| `false` | **6.7%** (every config) |

**Run #2 (skill-guided)** — best **63.3%** · `gpt-4o, schema_context=full_ddl_fk, direct, candidate_count=1`

`parameter_insights` performance_impact: **schema_context 1.09** (dominant) · model 0.13 ·
generation_path/CoT **0.03** · candidate_count/self-consistency **0.009**.

**Verdict:** the skill-guided boosters (chain-of-thought, self-consistency) gave **no real lift** on
this task — the winning config has both OFF, and 60.0%→63.3% is one example (gpt-4o is not fully
deterministic at temp 0). `include_schema` / `schema_context` dominates (≈7× swing). ~50–63% is the
realistic ceiling for a deliberately-basic single-call agent on Spider-lite, so this is an honest
"no-boost" result (which `traigent-boost-agent` step 10 says to report as such).

---

## 5. Why no accuracy shows on the portal

**Symptom:** in the portal's *Configurations Results* table you see a **`Score`** column (plus
Latency, Total Cost) but **no accuracy** — and the `Score` doesn't read as a % correct. On a 1-example
run the `Score` even ranks `include_schema=false` (worst accuracy, cheapest) as "BEST".

**Two compounding causes:**

**(a) Custom objective names are coerced to a fixed vocabulary.** The objective here is named
`exec_accuracy`. The backend bridge maps SDK objective names to a fixed set of measure IDs and
**defaults anything unknown to `accuracy`** — the custom name never reaches the portal:

```python
# traigent/cloud/backend_bridges.py  ->  _map_objectives_to_measures()
objective_mapping = {
    "accuracy": "accuracy", "cost": "cost", "latency": "latency",
    "success_rate": "accuracy", "error_rate": "accuracy",
    "contextual_precision": "contextual_precision", "ragas": "ragas",
}
measure_id = objective_mapping.get(objective.lower(), "accuracy")  # exec_accuracy -> "accuracy"
```

So `exec_accuracy` (or `sql_accuracy`, or any custom metric name) is never displayed under its own
name; at best it is collapsed into the generic `accuracy` measure.

**(b) The portal headline `Score` is a normalized, weighted composite, not raw accuracy.** The
*Configurations Results* table tags filters as `… SCORED · 1 PARETO · 3 WEIGHTS` and shows a blended
`Score` (objective + cost + latency, normalized). That is why `include_schema=false` (cheapest/fastest)
can outrank schema-on configs despite far worse SQL correctness, and why fractional `Score`s appear
even on a 1-example run (raw accuracy on one example can only be 0% or 100%).

**Evidence captured during the run**
- `GET /api/v1beta/projects/<proj>/experiments/<exp>` → top-level `"measures": []`; per-config rows
  expose `Score`/Latency/Cost; the UI hint is literally *"Click on any row to view detailed measures."*
- The mapping code above (objective → fixed measure id, default `accuracy`).

**Expected vs actual**
- *Expected:* an objective/metric named `exec_accuracy` is shown on the portal under that name (or at
  least the raw accuracy % is a first-class, visible column).
- *Actual:* no `exec_accuracy` anywhere; only a blended `Score`. The real accuracy is, at best, hidden
  in per-config "detailed measures" under the generic `accuracy` id.

**Fix / workaround**
- *Workaround now:* name the objective a recognized id — `objectives=["accuracy"]` and
  `metric_functions={"accuracy": exec_accuracy_metric}` (same computation). Then the portal can show
  it as the `accuracy` measure. Optionally `["accuracy","cost"]` for an accuracy-vs-cost view.
- *SDK/portal fix:* preserve arbitrary custom measure names end-to-end (don't silently coerce to a
  fixed vocabulary), and/or label the portal's `Score` column with its weighting so it isn't mistaken
  for the objective value.

---

## 6. Other issues encountered (full list in the companion issue)

1. **In-memory dataset drops a nested `metadata` key** — extra *top-level* keys become
   `example.metadata`, but a literal `{"metadata": {...}}` lands as `metadata["metadata"]`; the file
   loader *does* special-case it. (Worked around: put `db_path`/`db_id` at top level.)
2. **Custom-metric exceptions are swallowed → `0.0`** (warning only) in `evaluators/local.py`; a wrong
   `db_path` looks like "bad agent" (all 0%), not broken wiring.
3. **Cost calculator can't price `openrouter/*` models** → `UnknownModelError` raised 240× from
   `utils/cost_calculator.py` (only on the multi-sample `candidate_count=3` configs); caught after
   scoring, but spams tracebacks and makes `total_cost` undercounted for OpenRouter models.
4. **`examples/core/text-to-sql/run.py` needs `pandas`** (`to_aggregated_dataframe()`) which isn't a
   core dependency → `ModuleNotFoundError` on a clean install.
5. **`traigent-composite-knobs`**: the composite `(output, metrics)` tuple-return path does **not**
   invoke custom `metric_functions` (silently `0.0`), so composites don't compose with a custom
   deterministic metric — self-consistency had to be hand-rolled.
6. **Mock-mode activation is inconsistent** across skill/examples/SDK (`TRAIGENT_MOCK_LLM` vs
   `enable_mock_mode_for_quickstart()` vs `TRAIGENT_OFFLINE_MODE`).
