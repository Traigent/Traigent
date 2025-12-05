"""Quick coverage summary over TraceSync runtime log.

Reads newline-delimited JSON records from TRACE_SYNC_LOG (default: runtime/traces/runtime.log)
and reports which concept_ids/req_ids/sync_ids were observed. Targets are a
lightweight set aligned to current demo flows; extend TARGET_* sets as needed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Set

# Minimal targets aligned with instrumentation plan focus areas
TARGET_CONCEPTS = {
    "CONC-Layer-API",
    "CONC-Layer-Core",
    "CONC-Layer-Infra",
    "CONC-Layer-CrossCutting",
}

TARGET_REQS = {
    "REQ-API-001",
    "REQ-INV-006",
    "REQ-INJ-002",
    "REQ-EVAL-005",
    "REQ-ORCH-003",
    "REQ-STOR-007",
    "REQ-SEC-010",
    "REQ-ANLY-011",
    "REQ-TVLSPEC-012",
}

TARGET_SYNCS = {
    "SYNC-OptimizationFlow",
    "SYNC-StorageLogging",
    "SYNC-CloudHybrid",
    "SYNC-Observability",
    "SYNC-IntegrationHook",
}


def _extract_list(payload: dict[str, Any], key: str) -> Iterable[str]:
    value = payload.get(key) or []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple, set)):
        return [str(v) for v in value]
    return []


def summarize_log(log_path: Path) -> dict[str, Set[str]]:
    concepts: Set[str] = set()
    reqs: Set[str] = set()
    syncs: Set[str] = set()

    if not log_path.exists():
        raise SystemExit(f"log not found: {log_path}")

    with log_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            concept_id = record.get("concept_id")
            if concept_id:
                concepts.add(concept_id)

            inputs = record.get("inputs") or {}
            concepts.update(_extract_list(inputs, "concept_id"))
            reqs.update(_extract_list(inputs, "req_ids"))
            syncs.update(_extract_list(inputs, "sync_ids"))

    return {
        "concepts": concepts,
        "reqs": reqs,
        "syncs": syncs,
    }


def render_report(observed: dict[str, Set[str]]) -> str:
    def pct(hit: int, total: int) -> str:
        return f"{(hit/total*100):.1f}%" if total else "n/a"

    hit_concepts = observed["concepts"]
    hit_reqs = observed["reqs"]
    hit_syncs = observed["syncs"]

    lines = []
    lines.append("Coverage summary")
    lines.append(
        f"- concepts: {len(hit_concepts)}/{len(TARGET_CONCEPTS)} ({pct(len(hit_concepts), len(TARGET_CONCEPTS))})"
    )
    lines.append(f"  hit: {sorted(hit_concepts)}")
    lines.append(
        f"- req_ids: {len(hit_reqs)}/{len(TARGET_REQS)} ({pct(len(hit_reqs), len(TARGET_REQS))})"
    )
    lines.append(f"  hit: {sorted(hit_reqs)}")
    lines.append(
        f"- sync_ids: {len(hit_syncs)}/{len(TARGET_SYNCS)} ({pct(len(hit_syncs), len(TARGET_SYNCS))})"
    )
    lines.append(f"  hit: {sorted(hit_syncs)}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="TraceSync runtime coverage summary")
    parser.add_argument(
        "--log",
        type=Path,
        default=Path("runtime/traces/runtime.log"),
        help="Path to runtime trace log",
    )
    args = parser.parse_args()

    observed = summarize_log(args.log)
    print(render_report(observed))


if __name__ == "__main__":
    main()
