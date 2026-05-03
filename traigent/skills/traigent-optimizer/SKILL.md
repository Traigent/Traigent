---
name: traigent-optimizer
description: "Adopt Traigent in an existing codebase. Use when a user asks an AI coding assistant to find optimization opportunities, scan a repo for Traigent candidates, propose @traigent.optimize wiring, or generate a TVL/decorator adoption plan. Runs static scan/decorate helpers first, requires objective and dataset confirmation before writes, and keeps real LLM optimization behind explicit user approval."
license: Apache-2.0
metadata:
  author: Traigent
  version: "1.0"
---

# Traigent Optimizer Adoption

## Role

Help the user adopt Traigent in an existing codebase. Prefer the SDK CLI as the
source of truth. Do not invent a separate scanner, do not call an LLM provider
from inside the SDK flow, and do not spend real model tokens unless the user
explicitly approves a real optimization run.

The default optimizer CLI flow is static and review-first:

1. `traigent optimizer scan` ranks functions worth optimizing.
2. The user selects a candidate.
3. `traigent optimizer decorate` emits a dry-run plan for one function.
4. The user confirms objectives and dataset fields.
5. Only then should code changes or real optimization be considered.

## Helper Boundaries

| Helper | Use it for |
| --- | --- |
| `traigent optimizer scan/decorate` | Default Python adoption workflow for existing code. |
| `traigent optimizer --agent codex/claude-code/github-models` | Optional coding-agent enrichment. Use only after explicit user approval because it spends that agent/provider budget. |
| `traigent detect-tvars` | Low-level tuned-variable discovery debugging. |
| `traigent generate-config` | Advanced config generation; `--enrich` may call an LLM and must be budgeted. |
| JS `traigent detect tuned-variables` | Current JS discovery path until JS optimizer parity lands. |
| JS `traigent migrate seamless` | Reviewed JS codemod path for hardcoded local tuned variables. |
| MCP/hybrid | Backend/portal integration transport, not first-run adoption. |
| Governed autosearch | Advanced TVL search when that feature is available and a TVL program already exists. |

## Python Workflow

### 1. Scan

Run a static scan from the repo root or a focused package path:

```bash
traigent optimizer scan . --top 5
```

For machine-readable output:

```bash
mkdir -p .traigent
traigent optimizer scan . --top 5 --output .traigent/optimizer-scan.json
```

Do not pass `--agent` by default. If the user explicitly wants coding-agent
enrichment, use one of:

```bash
traigent optimizer scan . --top 5 --agent codex --agent-enrich-top-n 3
traigent optimizer scan . --top 5 --agent claude-code --agent-enrich-top-n 3
traigent optimizer scan . --top 5 --agent github-models --agent-enrich-top-n 3
```

Agent enrichment is one-shot, read-only, schema-validated, and provenance is
recorded in `agent_enrichment`. When operating on one file, pass `--project-root`
if the repo root is not the file's parent. Treat any `validation_status: invalid`
or `rejected_by_policy` as a review signal, not as a hard failure.

Report the top candidates with function, file, rank reason, proposed objective
candidates, and missing dataset signals. Do not present scan output as
authoritative; it is a ranked adoption aid.

### 2. Ask The User To Pick

Ask the user to choose one function unless they already named it. Do not decorate
multiple functions in one step. Multi-function patterns can be noted from scan
results, but v1 decoration is single-function.

### 3. Decorate Dry Run

Run:

```bash
mkdir -p .traigent
traigent optimizer decorate path/to/file.py --function function_name --output .traigent/function_name.decorate.json
```

If the user already confirmed objectives or a dataset:

```bash
traigent optimizer decorate path/to/file.py \
  --function function_name \
  --objective accuracy \
  --objective cost \
  --dataset data/eval.jsonl \
  --output .traigent/function_name.decorate.json
```

If the user approved enrichment for the chosen function, add the selected agent:

```bash
traigent optimizer decorate path/to/file.py \
  --function function_name \
  --agent codex \
  --output .traigent/function_name.decorate.json
```

Current Python slice emits a dry-run plan. If `--write` is unavailable or returns
"not implemented", do not treat that as a failure. Review the plan and apply
changes manually only after confirmation.

### 4. Confirm Objectives And Dataset

Apply this policy:

- `cost` and `latency` are auto-measurable once the runtime captures usage and timing.
- Quality objectives such as `accuracy`, `recall_at_k`, helpfulness, or safety require user confirmation and matching dataset fields.
- Do not silently select a quality objective just because the scanner proposed it.
- If no dataset exists, create only a stub or TODO plan unless the user provides ground-truth data.

State expected dataset fields explicitly. Examples:

```text
accuracy requires: input, expected_output
recall_at_k requires: input, relevant_doc_ids
```

### 5. Apply Changes

Before editing source, summarize the intended changes: emit mode, proposed tvars,
selected objectives, dataset path or stub path, and whether any objective still
requires confirmation. Then edit minimally. Prefer TVL when the plan resolves to
`tvl`; prefer inline only when the plan resolves to `inline` and the search
space is small.

### 6. Validate Without Cost

After applying changes, validate locally:

```bash
traigent validate path/to/eval.jsonl
traigent check path/to/file.py --functions="function_name" --dry-run
```

For tutorial or smoke runs, use mock mode before any real model call:

```python
from traigent.testing import enable_mock_mode_for_quickstart

enable_mock_mode_for_quickstart()
```

Do not run a real optimization until the user explicitly approves the cost.

## JS Workflow

The unified `traigent optimizer scan/decorate` command is not available in the
JS CLI in this checkout yet. For JS projects:

```bash
traigent detect tuned-variables src/agent.ts --function answerQuestion
traigent migrate seamless src/agent.ts
```

Only run `traigent migrate seamless --write` after reviewing diagnostics and
confirming the user wants the codemod applied.

## What To Tell The User

```text
Scan found 5 candidates. The best target is answer_question because it calls
OpenAI and has model/temperature literals. Decorate dry-run proposes model and
temperature tvars, cost/latency auto-measurable objectives, and accuracy as a
quality objective requiring expected_output labels.
```

Avoid vague claims like "fully optimized" or "ready for production" after a scan
or decorate dry run. The assistant found an adoption path; optimization results
only exist after a validated run.
