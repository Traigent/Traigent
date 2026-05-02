---
name: traigent-optimizer
description: "Adopt Traigent in an existing codebase. Use when a user asks an AI coding assistant to find optimization opportunities, scan a repo for Traigent candidates, propose @traigent.optimize wiring, or generate a TVL/decorator adoption plan. Runs static scan/decorate helpers first, requires objective and dataset confirmation before writes, and keeps real LLM optimization behind explicit user approval."
license: Apache-2.0
metadata:
  author: Traigent
  version: "1.0"
---

# Traigent Optimizer Adoption

Use the SDK CLI as the source of truth. The optimizer adoption flow is static and
review-first: scan, let the user choose one function, produce a decorate dry-run
plan, confirm objectives and dataset fields, then apply minimal changes.

## Workflow

1. Run a static scan:

```bash
traigent optimizer scan . --top 5
```

2. Present the top candidates with file, function, score, detected signals,
proposed tvars, objective candidates, and missing dataset signals.

3. Ask the user to pick one function unless they already named it.

4. Produce a dry-run decorate plan:

```bash
traigent optimizer decorate path/to/file.py --function function_name --output function_name.decorate.json
```

5. Confirm objectives and dataset fields before edits. Cost and latency are
auto-measurable; quality metrics such as accuracy, recall, helpfulness, and
safety require explicit user confirmation and matching dataset labels.

6. If `--write` is unavailable, do not treat that as failure. Apply the accepted
plan manually and minimally.

7. Validate without cost:

```bash
traigent validate path/to/eval.jsonl
traigent check path/to/file.py --functions="function_name" --dry-run
```

Do not run real optimization until the user explicitly approves provider cost.

## Boundaries

- `traigent optimizer scan/decorate` is the Python adoption path.
- `traigent detect-tvars` is lower-level tuned-variable discovery.
- `traigent generate-config --enrich` may call an LLM and must be budgeted.
- JS does not have `traigent optimizer scan/decorate` in this checkout yet; use
  `traigent detect tuned-variables` and reviewed `traigent migrate seamless`.
- MCP/hybrid is backend/portal transport, not first-run adoption.
- Governed autosearch is advanced TVL search after a TVL program exists.
