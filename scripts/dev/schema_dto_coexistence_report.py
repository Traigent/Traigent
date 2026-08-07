#!/usr/bin/env python3
"""One-off analysis: for every hand-written dataclass DTO in
traigent/cloud/dtos.py, traigent/admin/dtos.py, and traigent/core_metrics/dtos.py,
find the best-matching generated schema type (traigent/generated/schema_types.py)
and report whether their fields agree.

Three match tiers, in order:
  1. EXACT   - DTO name (minus "DTO"/"Response" suffix noise) snake_cases to
               exactly one generated class's schema stem.
  2. FUZZY   - the DTO's name tokens are a subset of a generated class's schema
               stem tokens (container/dashboard schemas named more verbosely
               than the DTO, e.g. ProjectAnalyticsSummaryDTO ->
               project_scoped_analytics_summary_schema.json).
  3. STRUCTURAL - no name-based candidate; search every object-shaped node
               (top-level OR nested `properties`) across the WHOLE schema
               corpus for the best Jaccard similarity of property-key sets to
               the DTO's field names. This is how nested per-dashboard
               fragments (TrendPointDTO, HistogramBucketDTO, ...) that have no
               schema file of their own get matched to the schema fragment
               they actually correspond to.

This script is read-only analysis; it does not modify the generator or its
committed output. Run after `python scripts/generate_schema_types.py`.
"""

from __future__ import annotations

import dataclasses
import importlib
import json
import re
import sys
from pathlib import Path
from typing import Any, get_type_hints

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

DTO_MODULES = [
    "traigent.cloud.dtos",
    "traigent.admin.dtos",
    "traigent.core_metrics.dtos",
]


def snake(name: str) -> str:
    name = re.sub(r"(?<!^)(?=[A-Z])", "_", name)
    return name.lower()


def load_dto_classes() -> list[tuple[str, type]]:
    out = []
    for modname in DTO_MODULES:
        mod = importlib.import_module(modname)
        for _, obj in vars(mod).items():
            if dataclasses.is_dataclass(obj) and obj.__module__ == modname:
                out.append((modname, obj))
    out.sort(key=lambda t: (t[0], t[1].__name__))
    return out


def load_generated_classes() -> dict[str, type]:
    mod = importlib.import_module("traigent.generated.schema_types")
    return {
        name: obj for name, obj in vars(mod).items() if dataclasses.is_dataclass(obj)
    }


def dto_field_info(cls: type) -> dict[str, tuple[bool, Any]]:
    """name -> (required, type_repr). Required == no default and no default_factory."""
    hints = _safe_type_hints(cls)
    info = {}
    for f in dataclasses.fields(cls):
        required = (
            f.default is dataclasses.MISSING
            and f.default_factory is dataclasses.MISSING
        )  # type: ignore[misc]
        info[f.name] = (required, hints.get(f.name, f.type))
    return info


def _safe_type_hints(cls: type) -> dict[str, Any]:
    try:
        return get_type_hints(cls)
    except Exception:
        return {}


def type_repr(t: Any) -> str:
    return str(t).replace("typing.", "").replace("NoneType", "None")


# ---------------------------------------------------------------------------
# Schema corpus: reload raw JSON so we can search NESTED property sets too
# (the committed generated module only carries top-level shapes).
# ---------------------------------------------------------------------------


def load_schema_corpus(schema_root: Path) -> list[tuple[str, str, frozenset[str]]]:
    """Returns (relpath, json-pointer-ish label, propkeys) for every
    object-shaped node (has non-empty "properties") anywhere in the corpus."""
    candidates: list[tuple[str, str, frozenset[str]]] = []
    for path in sorted(schema_root.rglob("*.json")):
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(doc, dict) and "openapi" in doc:
            continue
        rel = path.relative_to(schema_root).as_posix()

        def walk(node: Any, label: str, rel: str = rel) -> None:
            if isinstance(node, dict):
                props = node.get("properties")
                if isinstance(props, dict) and props:
                    candidates.append((rel, label, frozenset(props.keys())))
                    # Recurse INTO each property's own schema (this is where a
                    # nested object like "job"/"context" actually lives) —
                    # deliberately not skipped, only the "properties" dict
                    # itself is skipped below so it isn't re-walked as a
                    # generic mapping.
                    for key, value in props.items():
                        walk(value, f"{label}/properties/{key}")
                for key, value in node.items():
                    if key == "properties":
                        continue
                    walk(value, f"{label}/{key}")
            elif isinstance(node, list):
                for i, value in enumerate(node):
                    walk(value, f"{label}[{i}]")

        walk(doc, "$")
    return candidates


def jaccard(a: frozenset, b: frozenset) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def overlap_ratio(wanted: frozenset[str], candidate: frozenset[str]) -> float:
    if not wanted:
        return 0.0
    return len(wanted & candidate) / len(wanted)


def pascal_case(stem: str) -> str:
    stem = re.sub(r"_schema$", "", stem)
    parts = re.split(r"[_-]+", stem)
    return "".join(p[:1].upper() + p[1:] for p in parts if p)


def main() -> int:
    import os

    schema_repo = Path(
        os.environ.get("TRAIGENT_SCHEMA_REPO") or (ROOT.parent / "TraigentSchema")
    ).resolve()
    schema_root = schema_repo / "traigent_schema" / "schemas"

    dto_classes = load_dto_classes()
    generated = load_generated_classes()
    corpus = load_schema_corpus(schema_root)

    print(f"Loaded {len(dto_classes)} hand-written DTOs")
    print(
        f"Loaded {len(generated)} generated schema types (records+aliases with fields)"
    )
    print(f"Loaded {len(corpus)} object-shaped nodes from the raw schema corpus\n")

    rows = []
    for modname, cls in dto_classes:
        base = cls.__name__
        if base.endswith("DTO"):
            base = base[:-3]
        snake_name = snake(base)
        wanted_name = pascal_case(snake_name) + "Schema"

        dto_fields = dto_field_info(cls)
        dto_field_names = frozenset(dto_fields.keys())

        nominal_kind = None
        nominal_label = None
        nominal_fields: frozenset[str] = frozenset()

        if wanted_name in generated and dataclasses.fields(generated[wanted_name]):
            nominal_kind = "EXACT"
            nominal_label = wanted_name
            nominal_fields = frozenset(dto_field_info(generated[wanted_name]).keys())
        else:
            want_tokens = set(snake_name.split("_"))
            fuzzy_candidates = []
            for name, gcls in generated.items():
                if not name.endswith("Schema") or not dataclasses.fields(gcls):
                    continue
                stem_tokens = set(snake(name[:-6]).split("_"))
                if want_tokens <= stem_tokens:
                    fuzzy_candidates.append((len(stem_tokens - want_tokens), name))
            if fuzzy_candidates:
                fuzzy_candidates.sort()
                nominal_kind = "FUZZY"
                nominal_label = fuzzy_candidates[0][1]
                nominal_fields = frozenset(
                    dto_field_info(generated[nominal_label]).keys()
                )

        # Best global structural candidate, computed unconditionally: even a
        # name-based (EXACT/FUZZY) match can be a thin envelope around the
        # DTO's real shape (see ProjectExportJobDTO below), so always know
        # what the best independent structural candidate would have been.
        best = None
        best_score = 0.0
        for rel, label, propkeys in corpus:
            s = jaccard(dto_field_names, propkeys)
            if s > best_score:
                best_score = s
                best = (rel, label, propkeys)
        structural_candidate = None
        if best and best_score >= 0.4:
            structural_candidate = (
                f"{best[0]}{best[1]}",
                best[2],
                round(best_score, 2),
            )

        nominal_ratio = (
            overlap_ratio(dto_field_names, nominal_fields) if nominal_kind else -1.0
        )
        struct_ratio = (
            overlap_ratio(dto_field_names, structural_candidate[1])
            if structural_candidate
            else -1.0
        )

        # Winner-take-all: report whichever candidate actually covers more of
        # the DTO's fields as the PRIMARY match, and note the runner-up when it
        # is a different schema location (rather than silently keeping a name
        # match that turned out to be a thin envelope, e.g. EvaluatorDTO ->
        # EvaluatorConfigSchema by name vs. evaluator_definition_schema.json by
        # actual field overlap).
        match_kind = match_label = None
        schema_field_names: frozenset[str] = frozenset()
        score = None
        weak_nominal_note = None

        if nominal_kind and (not structural_candidate or nominal_ratio >= struct_ratio):
            match_kind, match_label, schema_field_names = (
                nominal_kind,
                nominal_label,
                nominal_fields,
            )
        elif structural_candidate:
            match_kind = "STRUCTURAL"
            match_label, schema_field_names, score = structural_candidate
            if nominal_kind and nominal_label != match_label:
                weak_nominal_note = (
                    f"name-based ({nominal_kind}) candidate was {nominal_label!r}, sharing only "
                    f"{len(dto_field_names & nominal_fields)}/{len(dto_field_names)} fields; the "
                    f"structural match above covers {len(dto_field_names & schema_field_names)}"
                    f"/{len(dto_field_names)} and is reported instead — likely a nested/wrapped "
                    "fragment, not the name-matched top-level schema"
                )
        elif nominal_kind:
            match_kind, match_label, schema_field_names = (
                nominal_kind,
                nominal_label,
                nominal_fields,
            )

        dto_only = sorted(dto_field_names - schema_field_names)
        schema_only = sorted(schema_field_names - dto_field_names)
        shared = sorted(dto_field_names & schema_field_names)

        rows.append(
            {
                "module": modname.rsplit(".", 1)[-1],
                "dto": cls.__name__,
                "kind": match_kind or "NONE",
                "match": match_label or "-",
                "score": score,
                "dto_field_count": len(dto_field_names),
                "schema_field_count": len(schema_field_names) if match_kind else None,
                "shared": len(shared),
                "dto_only": dto_only,
                "schema_only": schema_only,
                "weak_nominal_note": weak_nominal_note,
            }
        )

    # ---- print report ----
    for r in rows:
        header = f"{r['module']:16s} {r['dto']:42s} [{r['kind']:10s}] -> {r['match']}"
        if r["score"] is not None:
            header += f" (jaccard={r['score']})"
        print(header)
        if r["kind"] != "NONE":
            print(
                f"    dto_fields={r['dto_field_count']} schema_fields={r['schema_field_count']} "
                f"shared={r['shared']}"
            )
            if r["dto_only"]:
                print(f"    DTO-only (not in schema match): {r['dto_only']}")
            if r["schema_only"]:
                print(f"    schema-only (not in DTO): {r['schema_only']}")
            if r["weak_nominal_note"]:
                print(f"    NOTE: {r['weak_nominal_note']}")
        print()

    kinds = {}
    for r in rows:
        kinds[r["kind"]] = kinds.get(r["kind"], 0) + 1
    print("=== summary ===")
    for k, v in sorted(kinds.items()):
        print(f"{k}: {v}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
