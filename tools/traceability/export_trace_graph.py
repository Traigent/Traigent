#!/usr/bin/env python3
"""
Export traceability graph data for interactive viewers (e.g., ReGraph/Cytoscape).

Reads:
- docs/traceability/requirements.yml
- docs/traceability/functionalities.yml
- docs/traceability/concepts/*.yml
- docs/traceability/syncs/*.yml
- docs/traceability/reports/code_summaries.json
- docs/traceability/reports/trace_links.json

Writes:
- docs/traceability/reports/graph.json with nodes/edges suitable for graph UIs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[2]
TRACE_DIR = ROOT / "docs" / "traceability"


def load_yaml(path: Path) -> Any:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def add_node(
    nodes: dict[str, dict[str, Any]], node_id: str, label: str, type_: str, **attrs: Any
) -> None:
    if node_id in nodes:
        nodes[node_id].update({k: v for k, v in attrs.items() if v is not None})
        return
    nodes[node_id] = {
        "id": node_id,
        "label": label,
        "type": type_,
        **{k: v for k, v in attrs.items() if v is not None},
    }


def add_edge(
    edges: list[dict[str, Any]],
    source: str,
    target: str,
    type_: str,
    label: str | None = None,
    **attrs: Any,
) -> None:
    edge_id = f"{source}->{target}:{type_}"
    edge = {"id": edge_id, "source": source, "target": target, "type": type_}
    if label:
        edge["label"] = label
    edge.update({k: v for k, v in attrs.items() if v is not None})
    edges.append(edge)


def build_graph() -> dict[str, Any]:
    nodes: dict[str, dict[str, Any]] = {}
    edges: list[dict[str, Any]] = []

    # Requirements
    reqs = load_yaml(TRACE_DIR / "requirements.yml") or []
    for req in reqs:
        add_node(
            nodes,
            req["id"],
            req["id"],
            "requirement",
            name=req.get("name"),
            title=req.get("title"),
            priority=req.get("priority"),
            status=req.get("status"),
            tags=req.get("tags"),
        )

    # Functionalities
    funcs = load_yaml(TRACE_DIR / "functionalities.yml") or []
    for func in funcs:
        func_id = func["id"]
        add_node(
            nodes,
            func_id,
            func_id,
            "functionality",
            name=func.get("name"),
            tags=func.get("tags"),
            layer=func.get("layer"),
            status=func.get("status"),
        )
        for req_id in func.get("requirements", []) or []:
            add_edge(edges, req_id, func_id, "satisfied_by")

    # Concepts
    for concept_path in sorted((TRACE_DIR / "concepts").glob("*.yml")):
        concept = load_yaml(concept_path) or {}
        concept_id = concept.get("id") or concept_path.stem.upper()
        add_node(
            nodes,
            concept_id,
            concept_id,
            "concept",
            name=concept.get("name"),
            status=concept.get("status"),
        )

    # Syncs
    for sync_path in sorted((TRACE_DIR / "syncs").glob("*.yml")):
        sync = load_yaml(sync_path) or {}
        sync_id = sync.get("id") or sync_path.stem.upper()
        add_node(
            nodes,
            sync_id,
            sync_id,
            "sync",
            name=sync.get("name"),
            status=sync.get("status"),
        )
        for func_id in sync.get("functionalities", []) or []:
            add_edge(edges, func_id, sync_id, "implemented_by")
        # Link sync to referenced concepts (when/then sections)
        for clause in sync.get("when", []) or []:
            if isinstance(clause, dict) and "concept" in clause:
                add_edge(edges, sync_id, clause["concept"], "touches")
        for clause in sync.get("then", []) or []:
            if isinstance(clause, dict) and "concept" in clause:
                add_edge(edges, sync_id, clause["concept"], "invokes")

    # Functionality -> Concept edges from functionalities.yml
    for func in funcs:
        func_id = func["id"]
        for concept_id in func.get("concepts", []) or []:
            add_edge(edges, func_id, concept_id, "realized_by")
        for sync_id in func.get("syncs", []) or []:
            add_edge(edges, func_id, sync_id, "implemented_by")

    # Code summaries (files and symbols)
    code_summaries_path = TRACE_DIR / "reports" / "code_summaries.json"
    code_entries = (
        json.loads(code_summaries_path.read_text(encoding="utf-8"))
        if code_summaries_path.exists()
        else []
    )
    for entry in code_entries:
        file_id = f"code:{entry['file']}"
        add_node(
            nodes,
            file_id,
            entry["file"],
            "code_file",
            summary=entry.get("doc"),
            concept_id=entry.get("concept_id"),
        )
        # Add symbol child nodes
        for cls_name in entry.get("classes") or []:
            cls_id = f"{file_id}::{cls_name}"
            add_node(
                nodes, cls_id, cls_name, "code_symbol", kind="class", parent=file_id
            )
            add_edge(edges, file_id, cls_id, "contains")
        for fn_name in entry.get("functions") or []:
            fn_id = f"{file_id}::{fn_name}"
            add_node(
                nodes, fn_id, fn_name, "code_symbol", kind="function", parent=file_id
            )
            add_edge(edges, file_id, fn_id, "contains")

    # Trace links (code to concepts/functionality/requirements)
    trace_links_path = TRACE_DIR / "reports" / "trace_links.json"
    trace_links = (
        json.loads(trace_links_path.read_text(encoding="utf-8"))
        if trace_links_path.exists()
        else []
    )
    for link in trace_links:
        code_id = f"code:{link['code_unit']}"
        concept_id = link.get("concept_id")
        if concept_id:
            add_edge(
                edges, code_id, concept_id, "implements", status=link.get("status")
            )
        for func_id in link.get("functionalities") or []:
            add_edge(edges, code_id, func_id, "supports", status=link.get("status"))
            for req_id in link.get("requirements") or []:
                add_edge(edges, func_id, req_id, "traces_to", status=link.get("status"))
        for req_id in link.get("requirements") or []:
            add_edge(edges, code_id, req_id, "addresses", status=link.get("status"))

    return {"nodes": list(nodes.values()), "edges": edges}


def main() -> None:
    graph = build_graph()
    output_path = TRACE_DIR / "reports" / "graph.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(graph, indent=2), encoding="utf-8")
    print(
        f"Wrote graph with {len(graph['nodes'])} nodes and {len(graph['edges'])} edges to {output_path}"
    )


if __name__ == "__main__":
    main()
