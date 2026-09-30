# LLM cost & quality diagnostic

A runnable recipe for a short engagement: take a customer's sample of real
requests, measure their current model (the **baseline**), let Traigent search a
few cheaper models on part of the sample, then compare the baseline with the
**best observed configuration under the tested scope** on a held-out part of the
sample. The output is `report.md` plus `results.json`.

Providers: `openrouter` (real) and `mock` (deterministic, zero network).

## What the customer provides

1. A sample of real requests as JSONL, one object per line with `input` and
   `expected` (the answer they consider correct). Aim for 100+ rows; below about
   40 the holdout is too small to say much.
2. Their current model id and the system prompt they use today.
3. Optionally a schema or context file referenced from the prompt as `{schema_text}`.

Two built-in metrics, selected by the config field `metric`:

- `normalized_exact_match`: lowercase, collapse whitespace, strip a trailing `;`,
  then compare. This is textual agreement with the reference: an equivalent SQL
  query written differently scores 0.
- `sql_execution_match` (needs `schema_path` and `seed_path`): both the model's
  SQL and the reference SQL are executed on an in-memory SQLite database built
  from the schema plus a deterministic seed file
  (`data/text_to_sql_seed.sql`, dates relative to `date('now')`). Score is 1 if
  the result rows match as a multiset (order-sensitive only if the reference has
  `ORDER BY`), else 0; a SQL error scores 0. Markdown code fences are stripped,
  only `SELECT`/`WITH` statements run, and the database is read-only. Rows whose
  reference query errors or returns no rows are marked `reference_invalid`,
  excluded from quality means and counted in the report. This shows the same
  results on a seeded test database; it is still not proof of correctness on the
  customer's real data.

## Config (JSON)

See `diagnostic.example.json` and `configs/text_to_sql.openrouter.json`.

| key | meaning | default |
|---|---|---|
| `samples_path` | JSONL of `input`/`expected` rows | required |
| `schema_path` | optional text file substituted for `{schema_text}` in `prompt` | none |
| `baseline` | `{"model": "<id>"}`, the customer's current model | required |
| `candidate_models` | list of model ids to try | required |
| `prompt` | fixed system prompt | required |
| `temperature` | fixed for every call | 0 |
| `max_tokens` | output bound per call | 512 |
| `max_retries` | retries after a failed call | 1 |
| `timeout_s` | per-attempt timeout | 60 |
| `holdout_fraction` | share of rows held out | 0.3 |
| `seed` | split and bootstrap seed | 0 |
| `margin` | absolute quality margin for selection and non-inferiority | 0.02 |
| `max_spend_usd` | spend cap (see below) | required |
| `bootstrap_resamples` | paired bootstrap resamples | 2000 |
| `metric` | `normalized_exact_match` or `sql_execution_match` | `normalized_exact_match` |
| `seed_path` | SQL seed file (required for `sql_execution_match`) | none |
| `provider` | `openrouter` or `mock` (`--mock` / `TRAIGENT_MOCK_LLM` force mock) | `openrouter` |

Relative paths resolve against the working directory first, then the config's
directory. **Model ids must be checked against OpenRouter's model list
(https://openrouter.ai/api/v1/models) before a real run**; the ids in the demo
config are placeholders that may be renamed or retired.

## Run

```bash
# Dry run, no key, no network (also enabled by TRAIGENT_MOCK_LLM=true)
python examples/diagnostic/run_diagnostic.py \
  --config examples/diagnostic/configs/text_to_sql.openrouter.json \
  --out ./diag-mock --mock

# Real run
export OPENROUTER_API_KEY=...   # never commit this
python examples/diagnostic/run_diagnostic.py \
  --config examples/diagnostic/configs/text_to_sql.openrouter.json \
  --out ./diag
```

For `provider=openrouter` the pre-flight prices come from OpenRouter's public
`/models` listing, fetched at run time (provenance `OpenRouter /models at <UTC>`),
then Traigent's pricing tables, else "unknown". Mock mode never fetches. Add
`--allow-unknown-price` if a model has no price anywhere; the pre-flight estimate
is then "unknown" and only the actual-spend guard applies.
Mock mode prints a banner in the report: its models, costs and latencies are synthetic.

## What happens, in order

1. Load samples. Rows with identical `input` are grouped so duplicates never
   straddle the split. A seeded split produces a search and a holdout set. The
   split, configs and metric are written to `results.json` before any call.
2. Pre-flight: estimated calls x tokens x price (Traigent's pricing utilities;
   output assumed at `max_tokens`, no retries). Holdout spend is reserved up
   front; the run refuses to start if the estimate exceeds `max_spend_usd`.
3. Search: `traigent.optimize` with `algorithm="grid"` over the baseline plus the
   candidates, on the search split only, serially, every row for every config.
   Quality, cost, tokens, latency and errors come from this script's own ledger,
   written by the same function that later serves the holdout.
4. Selection (recorded in the report): among candidates whose provider-reported
   search cost per request is strictly below the baseline's and whose search
   quality is at least baseline minus `margin`, pick the lowest cost; ties go to
   higher quality. If none qualifies the report says so; that is a valid outcome.
5. Holdout, run once after selection: baseline and selected config on every
   holdout row, interleaved per row, with the same timeout and retries. Latency
   is monotonic time around the whole request including retries. Errors score 0
   in the denominator; failed calls' cost counts toward spend.
6. A seeded paired bootstrap, resampling unique holdout inputs (duplicates
   form one cluster, so they cannot narrow it), gives a 95%
   interval, shown as descriptive. "Non-inferior at margin m" is claimed only if
   the holdout is complete AND the number of independent pairs (unique holdout
   inputs) is at least `ceil(ln(0.025)/ln(1-m))` (183 at m=0.02) AND the bootstrap
   lower bound is above -m. Otherwise the report says "non-inferiority
   unestablished (holdout n=...; at least N independent pairs needed at margin
   m)". An incomplete holdout (spend-guard stop, unknown cost, missing rows)
   suppresses every inferential claim. A final baseline call whose candidate
   call was stopped is kept in `results.json` as `holdout_unpaired`.
   Generated SQL is bounded: at most 10,000 result rows and 1,000,000 bytes per
   value (`randomblob` beyond that fails) and 20,000,000 bytes cumulative across
   the result, checked row by row so oversized results stop early; exceeding any
   of these scores 0.

## Reading the report

- **Cost per 1k requests** is provider-reported (`usage.cost` from OpenRouter).
  A call with no reported cost is "unknown", never zero, and is counted
  separately. A price-table figure, if shown, is labelled "estimate".
- **p95 latency** uses nearest-rank and is marked descriptive below 20 rows.
- **Spend cap** is stop-on-actual: the guard is checked before every provider
  attempt, retries included, and each attempt's charge is booked immediately. A
  single in-flight call may exceed the cap. A call whose cost cannot be determined
  stops all further provider calls; the report is marked INCOMPLETE, total spend
  is shown as "unknown" and the known subtotal is listed separately.
- Outputs land in `--out`: `report.md`, `results.json` (paired per-row
  measurements including model outputs, pricing provenance, config, split ids,
  spend), and `work/` (the search split and Traigent's local run state). Treat
  the whole directory as customer data.

## Wording rules for what you hand over

Say "best observed configuration under the tested scope". Do not describe the
result as a guarantee, a proof, or a production commitment: it is one small sample,
one metric, one prompt, fixed temperature.

## Limits

Two providers only, JSON config only, two metrics, fixed prompt and temperature,
serial execution. The search split shares Traigent's local run state under `--out`;
no data leaves the machine except the model calls themselves.
