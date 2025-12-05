"""Layered dependency views and violations report."""

from __future__ import annotations

import argparse
import csv
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

try:  # pragma: no cover
    from .dependency_graph import Canvas
    from .viz_atlas import (
        AtlasConfig,
        draw_text,
        load_config,
        load_graphml,
        matches_patterns,
        strongly_connected_components,
    )
except ImportError:  # pragma: no cover
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from scripts.code_analysis.dependency_graph import Canvas
    from scripts.code_analysis.viz_atlas import (
        AtlasConfig,
        draw_text,
        load_config,
        load_graphml,
        matches_patterns,
        strongly_connected_components,
    )


def build_adjacency(
    nodes: Sequence[str], edges: Iterable[Tuple[str, str, float]]
) -> Dict[str, List[str]]:
    neighbors: Dict[str, List[str]] = {node: [] for node in nodes}
    for src, dst, _ in edges:
        if src in neighbors and dst in neighbors:
            neighbors[src].append(dst)
    return neighbors


def compute_layer(module: str, path: str, patterns: Sequence[re.Pattern[str]]) -> int:
    for idx, pattern in enumerate(patterns):
        if pattern.search(module) or pattern.search(path):
            return idx
    return len(patterns)


def layer_name(index: int, config: AtlasConfig) -> str:
    if index < len(config.vertical_layers):
        return config.vertical_layers[index]
    return "other"


def draw_layered_dag(
    components: List[List[str]],
    component_edges: Dict[Tuple[int, int], float],
    component_layers: Dict[int, int],
    component_representative: Dict[int, str],
    output_path: Path,
    config: AtlasConfig,
) -> None:
    layer_groups: Dict[int, List[int]] = defaultdict(list)
    for comp_id, layer_idx in component_layers.items():
        layer_groups[layer_idx].append(comp_id)
    ordered_layers = sorted(layer_groups.keys())
    margin_x = 140
    margin_y = 140
    max_layer_size = max((len(layer_groups[layer]) for layer in ordered_layers), default=1)
    width = min(config.image_max_edge_px, max(900, margin_x * 2 + max_layer_size * 280))
    height = min(config.image_max_edge_px, max(700, margin_y * 2 + max(len(ordered_layers) - 1, 1) * 220))
    if not ordered_layers:
        width = height = 800
    canvas = Canvas(width, height)

    if len(ordered_layers) == 1:
        y_positions = {ordered_layers[0]: height // 2}
    else:
        layer_spacing = (height - 2 * margin_y) / (len(ordered_layers) - 1)
        y_positions = {
            layer: int(margin_y + idx * layer_spacing)
            for idx, layer in enumerate(ordered_layers)
        }

    positions: Dict[int, Tuple[int, int]] = {}
    for layer in ordered_layers:
        comps = sorted(layer_groups[layer], key=lambda cid: component_representative[cid])
        count = len(comps)
        if count == 1:
            xs = [width // 2]
        else:
            layer_width = width - 2 * margin_x
            step = layer_width / (count - 1)
            xs = [int(margin_x + i * step) for i in range(count)]
        for x, comp_id in zip(xs, comps):
            positions[comp_id] = (x, y_positions[layer])

    max_weight = max(component_edges.values()) if component_edges else 1.0
    for (src, dst), weight in component_edges.items():
        src_pos = positions[src]
        dst_pos = positions[dst]
        thickness = max(1, int(round(2 * weight / max_weight)))
        color = (120, 120, 120, 160)
        canvas.draw_line(src_pos[0], src_pos[1], dst_pos[0], dst_pos[1], color, thickness)

    palette = [
        (70, 130, 180, 235),
        (95, 158, 160, 235),
        (60, 179, 113, 235),
        (218, 165, 32, 235),
        (147, 112, 219, 235),
        (205, 92, 92, 235),
        (128, 128, 128, 235),
    ]

    for comp_id, nodes in enumerate(components):
        if comp_id not in positions:
            continue
        x, y = positions[comp_id]
        layer_idx = component_layers.get(comp_id, len(config.vertical_layers))
        color = palette[layer_idx % len(palette)]
        radius = max(8, min(22, int(6 + math.sqrt(len(nodes) * 20) / 6)))
        canvas.draw_circle(x, y, radius, color)
        label = component_representative.get(comp_id, str(comp_id)).replace("traigent.", "")
        draw_text(canvas, x - len(label) * 3, y + radius + 10, label[:18])
        draw_text(canvas, x - 30, y - radius - 20, f"L{layer_idx}")

    draw_text(canvas, 20, 20, "LAYERED COMPONENT DAG", scale=2)
    canvas.save_png(output_path)


def generate_layers(
    graph_path: Path,
    output_dir: Path,
    config: AtlasConfig,
) -> None:
    node_paths, raw_edges = load_graphml(graph_path)
    patterns = [re.compile(pattern, re.IGNORECASE) for pattern in config.vertical_layers]

    filtered_nodes = [
        node
        for node, rel_path in node_paths.items()
        if not matches_patterns(rel_path, config.exclude_patterns)
    ]
    filtered_set = set(filtered_nodes)
    edges = [
        (edge.source, edge.target, edge.weight)
        for edge in raw_edges
        if edge.source in filtered_set and edge.target in filtered_set
    ]
    adjacency = build_adjacency(filtered_nodes, edges)

    components_list = strongly_connected_components(filtered_nodes, adjacency)
    component_map: Dict[str, int] = {}
    component_representative: Dict[int, str] = {}
    component_layers: Dict[int, int] = {}
    for comp_id, nodes in enumerate(components_list):
        for node in nodes:
            component_map[node] = comp_id
        representative = min(nodes)
        component_representative[comp_id] = representative
        node_layers = [
            compute_layer(node, node_paths.get(node, ""), patterns)
            for node in nodes
        ]
        component_layers[comp_id] = min(node_layers) if node_layers else len(patterns)

    component_edges: Dict[Tuple[int, int], float] = defaultdict(float)
    for src, dst, weight in edges:
        comp_src = component_map[src]
        comp_dst = component_map[dst]
        if comp_src == comp_dst:
            continue
        component_edges[(comp_src, comp_dst)] += weight

    output_dir.mkdir(parents=True, exist_ok=True)
    layered_png = output_dir / "layered_deps.png"
    draw_layered_dag(
        components_list,
        component_edges,
        component_layers,
        component_representative,
        layered_png,
        config,
    )

    violations_path = output_dir / "violations.csv"
    with violations_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["source_module", "source_layer", "target_module", "target_layer", "edge_weight"])
        for src, dst, weight in sorted(edges, key=lambda row: (-row[2], row[0], row[1])):
            src_layer = compute_layer(src, node_paths.get(src, ""), patterns)
            dst_layer = compute_layer(dst, node_paths.get(dst, ""), patterns)
            if dst_layer < src_layer:
                writer.writerow(
                    [
                        src,
                        layer_name(src_layer, config),
                        dst,
                        layer_name(dst_layer, config),
                        f"{weight:.2f}",
                    ]
                )

    index_lines = [
        "<!DOCTYPE html>",
        "<html lang=\"en\">",
        "<head>",
        "  <meta charset=\"utf-8\">",
        "  <title>Layered Views</title>",
        "  <style>body{font-family:Arial,Helvetica,sans-serif;margin:40px;}figure{margin:20px 0;}h2{margin-top:36px;}table{border-collapse:collapse;margin-top:16px;}th,td{border:1px solid #ccc;padding:6px 10px;text-align:left;} .viz-frame{max-width:100%;overflow:auto;border:1px solid #ccc;border-radius:4px;background:#fff;padding:8px;} .viz-frame img{display:block;}</style>",
        "</head>",
        "<body>",
        "  <h1>Layered Dependency Views</h1>",
        f"  <p>Input graph: {graph_path}</p>",
        "  <figure>",
        "    <figcaption>Layered Component DAG</figcaption>",
        "    <div class=\"viz-frame\">",
        "      <img src=\"layered_deps.png\" alt=\"Layered dependency graph\">",
        "    </div>",
        "  </figure>",
        "  <h2>Layer Violations</h2>",
        "  <p><a href=\"violations.csv\">Download CSV</a></p>",
        "</body>",
        "</html>",
    ]
    (output_dir / "index.html").write_text("\n".join(index_lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Layer analysis for dependency graph")
    parser.add_argument("--graph", type=Path, required=True, help="Path to deps.graphml")
    parser.add_argument("--out", type=Path, required=True, help="Output directory for layer artifacts")
    parser.add_argument(
        "--config", type=Path, default=Path("config/viz.yaml"), help="Visualization configuration"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    generate_layers(args.graph, args.out, config)


if __name__ == "__main__":  # pragma: no cover
    main()
