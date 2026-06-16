#!/usr/bin/env python
"""Layer-1 self-test (no LLM, no cost): dataset transform + db_path + metric.

Runs gold SQL against itself for every example. Every row MUST score 1.0;
any 0.0 means a broken db_path or a gold query that doesn't execute.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from optimize_run1_no_skills import DATASET, exec_accuracy_metric  # noqa: E402

bad = []
for i, ex in enumerate(DATASET):
    md = {"db_path": ex["db_path"], "db_id": ex["db_id"]}
    if not Path(md["db_path"]).exists():
        bad.append((i, md["db_id"], "DB FILE MISSING", md["db_path"]))
        continue
    if exec_accuracy_metric(ex["output"], ex["output"], md) != 1.0:
        bad.append((i, md["db_id"], "score!=1.0", ex["output"][:60]))

print(f"examples checked: {len(DATASET)}")
if bad:
    print(f"FAILURES: {len(bad)}")
    for r in bad:
        print("  ", r)
    raise SystemExit(1)
print("ALL ROWS score 1.0 on gold-vs-gold — dataset/metric/db_path plumbing OK.")
