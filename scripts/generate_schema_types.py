#!/usr/bin/env python3
"""Generate committed Python dataclasses from TraigentSchema JSON Schema files.

Ports the pattern already proven in traigent-js
(``scripts/generate-schema-types.mjs``) to the Python SDK:

* output is COMMITTED (``traigent/generated/schema_types.py``), so a schema
  change shows up as a reviewable diff instead of a build-time-only artifact
  nobody looks at;
* the header carries a ``source_sha256`` over every source schema file, plus
  one ``# source:`` line per file, so the committed artifact is traceable to
  the exact inputs that produced it;
* ``--check`` regenerates in memory and fails (naming the drifted file) on any
  difference from the committed artifact;
* determinism: two consecutive regenerations from the same inputs are
  byte-identical (files are walked in sorted order, dict iteration order is
  insertion order, no wall-clock/timestamp is embedded).

Scope note (read before extending): TypeScript can express an anonymous
nested object inline (``{ a: string; b: number }``); a Python ``@dataclass``
cannot without a name. So unlike the JS generator, a JSON object that appears
NESTED inside another schema (not as the top-level shape of its own schema
file) is rendered as ``dict[str, Any]`` rather than a synthesized nested
class. Every *top-level* schema file still gets a fully-typed dataclass (or a
type alias when its top-level shape isn't a record — an enum, a scalar
wrapper, or a union of other generated types). This is the single largest
divergence from the JS reference implementation; see the module docstring in
that file for the structural-typing tricks that are simply unavailable here.

Usage:
    python scripts/generate_schema_types.py            # regenerate in place
    python scripts/generate_schema_types.py --check     # verify committed output is current, exit 1 with a diff-style message if not

Environment:
    TRAIGENT_SCHEMA_REPO   Path to a TraigentSchema checkout (defaults to a
                            sibling ``../TraigentSchema`` directory, matching
                            the traigent-js convention).
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import keyword
import re
import sys
from os import environ
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
SCHEMA_REPO_ROOT = Path(
    environ.get("TRAIGENT_SCHEMA_REPO") or (ROOT.parent / "TraigentSchema")
).resolve()
SCHEMA_ROOT = SCHEMA_REPO_ROOT / "traigent_schema" / "schemas"
OUT_FILE = ROOT / "traigent" / "generated" / "schema_types.py"
GENERATOR_LABEL = "scripts/generate_schema_types.py"

_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_ABS_URL_RE = re.compile(r"^https?://")


# ---------------------------------------------------------------------------
# Schema discovery
# ---------------------------------------------------------------------------


def discover_files() -> list[Path]:
    """All *.json files under SCHEMA_ROOT, sorted for determinism."""
    if not SCHEMA_ROOT.is_dir():
        raise SystemExit(
            f"Schema root not found: {SCHEMA_ROOT}\n"
            "Set TRAIGENT_SCHEMA_REPO to a TraigentSchema checkout."
        )
    files = list(SCHEMA_ROOT.rglob("*.json"))
    files.sort(key=lambda p: p.relative_to(SCHEMA_ROOT).as_posix())
    return files


def is_openapi_document(doc: Any) -> bool:
    """*_endpoints.json (and mep_endpoints.json) are OpenAPI documents, not
    JSON Schema — they carry an 'openapi' top-level key and no $id/type."""
    return isinstance(doc, dict) and "openapi" in doc


def pascal_case(stem: str) -> str:
    stem = re.sub(r"_schema$", "", stem)
    parts = re.split(r"[_-]+", stem)
    return "".join(part[:1].upper() + part[1:] for part in parts if part)


# ---------------------------------------------------------------------------
# Load every eligible schema file up front, indexed by resolved path and by
# $id, mirroring the JS generator's schemaByFile / nameByFile / nameById /
# fileById maps.
# ---------------------------------------------------------------------------


class SchemaIndex:
    def __init__(self, files: list[Path]) -> None:
        self.files = files
        self.schema_by_file: dict[Path, Any] = {}
        self.name_by_file: dict[Path, str] = {}
        self.name_by_id: dict[str, str] = {}
        self.file_by_id: dict[str, Path] = {}
        self.skipped_openapi: list[Path] = []

        seen_names: dict[str, Path] = {}
        for path in files:
            with path.open(encoding="utf-8") as handle:
                doc = json.load(handle)
            if is_openapi_document(doc):
                self.skipped_openapi.append(path)
                continue
            self.schema_by_file[path] = doc
            name = pascal_case(path.stem) + "Schema"
            if name in seen_names:
                raise SystemExit(
                    f"Class name collision: {name!r} generated for both "
                    f"{seen_names[name]} and {path} — schema stems must be "
                    "unique across the corpus."
                )
            seen_names[name] = path
            self.name_by_file[path] = name
            schema_id = doc.get("$id") if isinstance(doc, dict) else None
            if isinstance(schema_id, str):
                self.name_by_id[schema_id] = name
                self.file_by_id[schema_id] = path

    @property
    def eligible_files(self) -> list[Path]:
        return [p for p in self.files if p not in self.skipped_openapi]


# ---------------------------------------------------------------------------
# $ref / JSON-pointer resolution (ports refToType / resolveJsonPointer from
# the JS generator)
# ---------------------------------------------------------------------------


def resolve_json_pointer(document: Any, fragment: str) -> Any:
    if not fragment:
        return document
    if not fragment.startswith("/"):
        return None
    node = document
    for raw_part in fragment[1:].split("/"):
        part = raw_part.replace("~1", "/").replace("~0", "~")
        if isinstance(node, dict) and part in node:
            node = node[part]
        else:
            return None
    return node


class Resolver:
    def __init__(self, index: SchemaIndex) -> None:
        self.index = index

    def target_file_for_ref(self, ref: str, from_file: Path) -> tuple[Path | None, str]:
        """Return (resolved_file_or_None, fragment) for a $ref string."""
        path_part, _, fragment = ref.partition("#")
        if not path_part:
            return from_file, fragment
        if _ABS_URL_RE.match(path_part):
            return self.index.file_by_id.get(path_part), fragment
        target = (from_file.parent / path_part).resolve()
        return target, fragment

    def ref_to_pytype(
        self, ref: str, from_file: Path, seen_refs: frozenset[str]
    ) -> str:
        path_part, _, fragment = ref.partition("#")
        target_file, _ = self.target_file_for_ref(ref, from_file)
        if not fragment:
            # Whole-file reference: point at that file's own generated name
            # (a class or a type alias), not into its innards.
            if not path_part:
                return self.index.name_by_file.get(from_file, "Any")
            if _ABS_URL_RE.match(path_part):
                return self.index.name_by_id.get(path_part, "Any")
            return (
                self.index.name_by_file.get(target_file, "Any")
                if target_file
                else "Any"
            )
        if target_file is None:
            return "Any"
        document = self.index.schema_by_file.get(target_file)
        if document is None:
            return "Any"
        resolved = resolve_json_pointer(document, fragment)
        if resolved is None:
            return "Any"
        ref_key = f"{target_file}#{fragment}"
        if ref_key in seen_refs:
            return "Any"  # cycle guard
        return self.schema_to_pytype(resolved, target_file, seen_refs | {ref_key})

    def deref_to_node(
        self, schema: Any, from_file: Path, guard: int = 0
    ) -> tuple[Any, Path]:
        """Follow a chain of single $refs to the concrete node, tracking the
        file each hop lives in (ports derefToNode)."""
        node, file = schema, from_file
        while (
            isinstance(node, dict) and isinstance(node.get("$ref"), str) and guard < 32
        ):
            guard += 1
            target_file, fragment = self.target_file_for_ref(node["$ref"], file)
            if target_file is None:
                return node, file
            document = self.index.schema_by_file.get(target_file)
            if document is None:
                return node, file
            resolved = (
                resolve_json_pointer(document, fragment) if fragment else document
            )
            if resolved is None:
                return node, file
            node, file = resolved, target_file
        return node, file

    # -- object-shape flattening (ports collectObjectShape / renderObjectShape) --

    def collect_object_shape(
        self, schema: Any, from_file: Path, acc: dict[str, Any]
    ) -> None:
        node, file = self.deref_to_node(schema, from_file)
        if not isinstance(node, dict):
            return
        if isinstance(node.get("allOf"), list):
            for member in node["allOf"]:
                self.collect_object_shape(member, file, acc)
        if node.get("type") == "object" or node.get("properties"):
            acc["has_object"] = True
            if node.get("additionalProperties") is False:
                acc["closed"] = True
            for key in node.get("required") or []:
                acc["required"].add(key)
            for key, nested in (node.get("properties") or {}).items():
                if key in acc["props"] and not self._carries_renderable_type(nested):
                    continue
                acc["props"][key] = (nested, file)

    @staticmethod
    def _carries_renderable_type(schema: Any) -> bool:
        if not isinstance(schema, dict):
            return False
        return bool(
            isinstance(schema.get("$ref"), str)
            or "const" in schema
            or isinstance(schema.get("enum"), list)
            or schema.get("type") is not None
            or isinstance(schema.get("allOf"), list)
            or isinstance(schema.get("anyOf"), list)
            or isinstance(schema.get("oneOf"), list)
            or (isinstance(schema.get("properties"), dict) and schema["properties"])
        )

    @staticmethod
    def new_shape_accumulator() -> dict[str, Any]:
        return {"props": {}, "required": set(), "closed": False, "has_object": False}

    # -- top-level dispatch: object record, alias, or skip --------------------

    def top_level_shape(self, document: Any, file: Path) -> tuple[str, Any]:
        """Classify a top-level schema document.

        Returns (kind, payload):
          kind == "record" -> payload is the shape accumulator (dataclass)
          kind == "alias"  -> payload is a python type expression (string)
          kind == "skip"   -> payload is None (no renderable top-level shape,
                              e.g. a pure definitions library referenced only
                              via #/definitions/... fragments elsewhere)
        """
        acc = self.new_shape_accumulator()
        self.collect_object_shape(document, file, acc)
        if acc["has_object"]:
            return "record", acc
        # Not an object at the top level: fall back to the general scalar/
        # union renderer. A bare "Any" means nothing meaningful is expressible
        # (e.g. a pure {"definitions": {...}} library) -> skip.
        expr = self.schema_to_pytype(document, file, frozenset(), top_level=True)
        if expr == "Any":
            return "skip", None
        return "alias", expr

    def render_object_shape(
        self, acc: dict[str, Any], seen_refs: frozenset[str]
    ) -> tuple[list[tuple[str, str]], list[tuple[str, str]], list[str]]:
        """Returns (required_fields, optional_fields, sanitize_notes) where
        each field is (python_name, type_expr)."""
        required: list[tuple[str, str]] = []
        optional: list[tuple[str, str]] = []
        notes: list[str] = []
        used_names: set[str] = set()
        for key in acc["props"]:
            nested, file = acc["props"][key]
            py_name = self._sanitize_field_name(key, used_names)
            if py_name != key:
                notes.append(f"{py_name} <- JSON property {key!r}")
            used_names.add(py_name)
            type_expr = self.schema_to_pytype(nested, file, seen_refs)
            if key in acc["required"]:
                required.append((py_name, type_expr))
            else:
                if not type_expr.endswith("None") and " | None" not in type_expr:
                    type_expr = f"{type_expr} | None"
                optional.append((py_name, type_expr))
        return required, optional, notes

    @staticmethod
    def _sanitize_field_name(key: str, used: set[str]) -> str:
        name = re.sub(r"[^A-Za-z0-9_]", "_", key)
        if not name or name[0].isdigit():
            name = f"_{name}"
        if keyword.iskeyword(name):
            name = f"{name}_"
        base = name
        suffix = 2
        while name in used:
            name = f"{base}_{suffix}"
            suffix += 1
        return name

    # -- general schema -> python type expression (ports schemaToType) --------

    def schema_to_pytype(
        self,
        schema: Any,
        from_file: Path,
        seen_refs: frozenset[str],
        top_level: bool = False,
    ) -> str:
        if not isinstance(schema, dict):
            return "Any"
        if isinstance(schema.get("$ref"), str):
            return self.ref_to_pytype(schema["$ref"], from_file, seen_refs)
        if isinstance(schema.get("allOf"), list):
            return self._allof_to_pytype(schema, from_file, seen_refs, top_level)
        if "const" in schema:
            value = schema["const"]
            if _is_literal_safe(value):
                return f"Literal[{py_literal(value)}]"
            return _fallback_scalar_type(value, schema.get("type"))
        if isinstance(schema.get("enum"), list) and schema["enum"]:
            enum_values = schema["enum"]
            if all(_is_literal_safe(v) for v in enum_values):
                values = ", ".join(py_literal(v) for v in enum_values)
                return f"Literal[{values}]"
            # PEP 586: Literal only admits bool/int/str/bytes/None (not float) —
            # a float-valued enum (e.g. a const tolerance/epsilon) falls back to
            # its declared/inferred scalar type instead of an invalid Literal.
            return _fallback_scalar_type(enum_values[0], schema.get("type"))
        if isinstance(schema.get("type"), list):
            variants = []
            for t in schema["type"]:
                variant = dict(schema)
                variant["type"] = t
                variants.append(self.schema_to_pytype(variant, from_file, seen_refs))
            return _dedup_union(variants)
        has_object_shape = schema.get("type") == "object" or bool(
            schema.get("properties")
        )
        if (schema.get("anyOf") or schema.get("oneOf")) and not has_object_shape:
            members = schema.get("anyOf") or schema.get("oneOf")
            if not top_level and len(members) > 6:
                return "Any"  # documented cap: an unwieldy nested union degrades to Any
            variants = [self.schema_to_pytype(m, from_file, seen_refs) for m in members]
            return _dedup_union(variants)
        if (schema.get("anyOf") or schema.get("oneOf")) and has_object_shape:
            # Composite object (base shape + anyOf/oneOf refinement). Python
            # dataclasses can't express the refinement; at the top level the
            # caller renders the base object as the record and this path is
            # unreachable (top_level_shape short-circuits on has_object).
            # Reached only for a NESTED property -> degrade to dict[str, Any].
            return "dict[str, Any]"
        schema_type = schema.get("type")
        if schema_type == "string":
            return "str"
        if schema_type in ("integer", "number"):
            return "int" if schema_type == "integer" else "float"
        if schema_type == "boolean":
            return "bool"
        if schema_type == "null":
            return "None"
        if schema_type == "array":
            item_type = self.schema_to_pytype(
                schema.get("items") or {}, from_file, seen_refs
            )
            return f"list[{item_type}]"
        if schema_type == "object" or schema.get("properties"):
            if top_level:
                # Handled by top_level_shape via collect_object_shape; this
                # branch only fires for a map-typed top-level (no properties).
                additional = schema.get("additionalProperties")
                if isinstance(additional, dict):
                    value_type = self.schema_to_pytype(additional, from_file, seen_refs)
                    return f"dict[str, {value_type}]"
                return "dict[str, Any]"
            return (
                "dict[str, Any]"  # nested object: no inline structural type in Python
            )
        return "Any"

    def _allof_to_pytype(
        self,
        schema: dict[str, Any],
        from_file: Path,
        seen_refs: frozenset[str],
        top_level: bool,
    ) -> str:
        acc = self.new_shape_accumulator()
        self.collect_object_shape(schema, from_file, acc)
        if acc["has_object"] and acc["props"]:
            return "dict[str, Any]"  # nested composed object: no inline type in Python
        # Scalar composition (e.g. a description wrapper around one $ref'd
        # primitive/definition): honor a local const/enum, else adopt the
        # single referenced member's type. This is the common
        # "allOf: [{$ref: .../Identifier}]" wrapper pattern.
        if "const" in schema:
            value = schema["const"]
            if _is_literal_safe(value):
                return f"Literal[{py_literal(value)}]"
            return _fallback_scalar_type(value, schema.get("type"))
        if isinstance(schema.get("enum"), list) and schema["enum"]:
            enum_values = schema["enum"]
            if all(_is_literal_safe(v) for v in enum_values):
                return f"Literal[{', '.join(py_literal(v) for v in enum_values)}]"
            return _fallback_scalar_type(enum_values[0], schema.get("type"))
        for member in schema["allOf"]:
            candidate = self.schema_to_pytype(member, from_file, seen_refs)
            if candidate != "Any":
                return candidate
        return "Any"


def py_literal(value: Any) -> str:
    if value is True:
        return "True"
    if value is False:
        return "False"
    if value is None:
        return "None"
    if isinstance(value, str):
        return repr(value)
    if isinstance(value, (int, float)):
        return repr(value)
    return repr(json.dumps(value, sort_keys=True))


def _is_literal_safe(value: Any) -> bool:
    """PEP 586: typing.Literal only admits bool/int/str/bytes/None (not float,
    not dict/list). bool is an int subclass and IS allowed, so it is checked
    first only to route through py_literal's True/False branch correctly."""
    if value is None or isinstance(value, (bool, int, str)):
        return True
    return False


def _fallback_scalar_type(sample_value: Any, declared_type: Any) -> str:
    """Base scalar type to use when const/enum can't render as Literal."""
    if isinstance(declared_type, str):
        mapping = {
            "string": "str",
            "integer": "int",
            "number": "float",
            "boolean": "bool",
            "null": "None",
        }
        if declared_type in mapping:
            return mapping[declared_type]
    if isinstance(sample_value, bool):
        return "bool"
    if isinstance(sample_value, int):
        return "int"
    if isinstance(sample_value, float):
        return "float"
    if isinstance(sample_value, str):
        return "str"
    if sample_value is None:
        return "None"
    return "Any"


def _dedup_union(variants: list[str]) -> str:
    seen: list[str] = []
    for v in variants:
        if v not in seen:
            seen.append(v)
    return " | ".join(seen) if seen else "Any"


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def render_record(
    name: str, acc: dict[str, Any], resolver: Resolver, source: str
) -> str:
    required, optional, notes = resolver.render_object_shape(acc, frozenset())
    lines = ["@dataclass(frozen=True)", f"class {name}:"]
    doc = f'    """Generated from {source}."""'
    lines.append(doc)
    if notes:
        lines.append("    #")
        for note in notes:
            lines.append(f"    # sanitized field name: {note}")
    if not required and not optional:
        lines.append("")
    for py_name, type_expr in required:
        lines.append(f"    {py_name}: {type_expr}")
    for py_name, type_expr in optional:
        lines.append(f"    {py_name}: {type_expr} = None")
    return "\n".join(lines)


def render_alias(name: str, expr: str, source: str) -> str:
    return f"# Generated from {source}\n{name} = {expr}"


def generate() -> str:
    files = discover_files()
    index = SchemaIndex(files)
    resolver = Resolver(index)

    hash_ = hashlib.sha256()
    source_lines: list[str] = []
    for path in index.eligible_files:
        rel = path.relative_to(SCHEMA_REPO_ROOT).as_posix()
        content = path.read_bytes()
        hash_.update(rel.encode("utf-8"))
        hash_.update(content)
        source_lines.append(f"# source: {rel}")

    body_blocks: list[str] = []
    skipped_no_shape: list[str] = []
    record_count = 0
    alias_count = 0
    for path in index.eligible_files:
        document = index.schema_by_file[path]
        name = index.name_by_file[path]
        rel = path.relative_to(SCHEMA_REPO_ROOT).as_posix()
        kind, payload = resolver.top_level_shape(document, path)
        if kind == "record":
            body_blocks.append(render_record(name, payload, resolver, rel))
            record_count += 1
        elif kind == "alias":
            body_blocks.append(render_alias(name, payload, rel))
            alias_count += 1
        else:
            skipped_no_shape.append(rel)

    header = [
        '"""Generated by scripts/generate_schema_types.py. Do not edit by hand.',
        "",
        f"source_sha256: {hash_.hexdigest()}",
        "",
        f"Records: {record_count}  Aliases: {alias_count}  "
        f"Skipped (no top-level shape): {len(skipped_no_shape)}",
        '"""',
        "",
        "# ruff: noqa",
        "# fmt: off",
        "from __future__ import annotations",
        "",
        "from dataclasses import dataclass",
        "from typing import Any, Literal",
        "",
        *source_lines,
    ]
    if skipped_no_shape:
        header.append("#")
        header.append(
            "# Skipped (pure definitions library, no top-level shape to render):"
        )
        for rel in skipped_no_shape:
            header.append(f"# skipped: {rel}")
    header.append("")

    content = "\n".join(header) + "\n\n\n".join(body_blocks) + "\n"
    # Belt-and-suspenders: this generator must never emit trailing whitespace
    # (a pre-commit hook silently strips it, permanently desyncing the
    # checked-in artifact from a fresh --check regeneration; see the module
    # docstring history in the PR that added this check).
    for lineno, line in enumerate(content.splitlines(), start=1):
        if line != line.rstrip():
            raise AssertionError(
                f"generator produced trailing whitespace on line {lineno}; this is a "
                "generator bug, fix it before committing (see module docstring)"
            )
    return content


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify the committed artifact matches a fresh regeneration; exit 1 on drift.",
    )
    args = parser.parse_args(argv)

    content = generate()

    if args.check:
        existing = OUT_FILE.read_text(encoding="utf-8") if OUT_FILE.exists() else ""
        if existing != content:
            diff = "".join(
                difflib.unified_diff(
                    existing.splitlines(keepends=True),
                    content.splitlines(keepends=True),
                    fromfile=f"{OUT_FILE} (committed)",
                    tofile=f"{OUT_FILE} (regenerated)",
                    n=2,
                )
            )
            print(
                f"{OUT_FILE} is stale. Run: python {GENERATOR_LABEL}\n\n{diff}",
                file=sys.stderr,
            )
            return 1
        print(f"{OUT_FILE} is up to date.")
        return 0

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(content, encoding="utf-8")
    print(f"Wrote {OUT_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
