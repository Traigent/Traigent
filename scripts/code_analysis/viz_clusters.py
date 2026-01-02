"""Clustered capability views built from the dependency graph."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

from traigent.utils.secure_path import PathTraversalError, validate_path

try:  # pragma: no cover - allow direct execution
    from .dependency_graph import Canvas, layout_positions
    from .viz_atlas import (
        AtlasConfig,
        draw_text,
        load_config,
        load_graphml,
        load_metrics,
        matches_patterns,
    )
except ImportError:  # pragma: no cover
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from scripts.code_analysis.dependency_graph import Canvas, layout_positions
    from scripts.code_analysis.viz_atlas import (
        AtlasConfig,
        draw_text,
        load_config,
        load_graphml,
        load_metrics,
        matches_patterns,
    )


@dataclass
class ModuleInfo:
    name: str
    path: str
    sloc: int
    fan_in: int
    fan_out: int


Graph = Dict[str, Dict[str, float]]


def build_module_info(
    node_paths: Mapping[str, str],
    metrics: Mapping[str, Mapping[str, object]],
) -> Dict[str, ModuleInfo]:
    modules: Dict[str, ModuleInfo] = {}
    for module, rel_path in node_paths.items():
        metric_row = metrics.get(module, {})
        sloc = int(float(metric_row.get("sloc", 0) or 0))
        fan_in = int(float(metric_row.get("fan_in", 0) or 0))
        fan_out = int(float(metric_row.get("fan_out", 0) or 0))
        modules[module] = ModuleInfo(
            name=module,
            path=rel_path or module.replace(".", "/"),
            sloc=sloc,
            fan_in=fan_in,
            fan_out=fan_out,
        )
    return modules


def undirected_weights(edges: Iterable[Tuple[str, str, float]]) -> Tuple[Graph, Dict[Tuple[str, str], float]]:
    adjacency: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    pair_weights: Dict[Tuple[str, str], float] = defaultdict(float)
    for src, dst, weight in edges:
        if src == dst:
            pair = (src, dst)
        else:
            pair = tuple(sorted((src, dst)))
        pair_weights[pair] += weight
    for (u, v), weight in pair_weights.items():
        adjacency.setdefault(u, defaultdict(float))
        adjacency.setdefault(v, defaultdict(float))
        adjacency[u][v] += weight
        adjacency[v][u] += weight
    return {node: dict(neighbors) for node, neighbors in adjacency.items()}, pair_weights


def louvain_one_level(graph: Graph) -> Dict[str, int]:
    nodes = list(graph.keys())
    node2com = {node: idx for idx, node in enumerate(nodes)}
    node_degree = {node: sum(neighbors.values()) for node, neighbors in graph.items()}
    m2 = sum(node_degree.values())
    if m2 == 0:
        return dict.fromkeys(nodes, 0)
    community_weight = defaultdict(float)
    for node, deg in node_degree.items():
        community_weight[node2com[node]] += deg
    improved = True
    while improved:
        improved = False
        for node in nodes:
            current_com = node2com[node]
            k_i = node_degree[node]
            if k_i == 0:
                continue
            neighbor_weights = defaultdict(float)
            for neighbor, weight in graph[node].items():
                neighbor_com = node2com[neighbor]
                neighbor_weights[neighbor_com] += weight
            community_weight[current_com] -= k_i
            best_com = current_com
            best_gain = 0.0
            for community, total_weight in neighbor_weights.items():
                gain = total_weight - (community_weight[community] * k_i) / m2
                if gain > best_gain + 1e-12:
                    best_gain = gain
                    best_com = community
            if best_com != current_com:
                node2com[node] = best_com
                community_weight[best_com] += k_i
                improved = True
            else:
                community_weight[current_com] += k_i
    # Compress community ids to contiguous range
    remap: Dict[int, int] = {}
    counter = 0
    result: Dict[str, int] = {}
    for node, community in node2com.items():
        if community not in remap:
            remap[community] = counter
            counter += 1
        result[node] = remap[community]
    return result


def aggregate_graph(graph: Graph, partition: Mapping[str, int]) -> Graph:
    aggregated_weights: Dict[Tuple[int, int], float] = defaultdict(float)
    for node, neighbors in graph.items():
        com_u = partition[node]
        for neighbor, weight in neighbors.items():
            com_v = partition[neighbor]
            if com_u == com_v:
                aggregated_weights[(com_u, com_v)] += weight
            else:
                key = (min(com_u, com_v), max(com_u, com_v))
                aggregated_weights[key] += weight
    communities = set(partition.values())
    new_graph: Dict[int, Dict[int, float]] = {community: defaultdict(float) for community in communities}
    for (u, v), weight in aggregated_weights.items():
        new_graph.setdefault(u, defaultdict(float))
        new_graph.setdefault(v, defaultdict(float))
        new_graph[u][v] += weight
        if u != v:
            new_graph[v][u] += weight
    return {node: dict(neighbors) for node, neighbors in new_graph.items()}


def louvain_clustering(graph: Graph) -> Dict[str, int]:
    if not graph:
        return {}
    current_graph: Graph = {node: dict(neighbors) for node, neighbors in graph.items()}
    hierarchy: List[Dict[str, int]] = []
    while True:
        partition = louvain_one_level(current_graph)
        hierarchy.append(partition)
        communities = set(partition.values())
        if len(communities) == len(current_graph):
            break
        current_graph = aggregate_graph(current_graph, partition)
    # Reconstruct mapping back to original nodes
    final_map: Dict[int, int] = {node: idx for idx, node in enumerate(current_graph.keys())}
    for partition in reversed(hierarchy):
        next_map: Dict[str, int] = {}
        for node, community in partition.items():
            next_map[node] = final_map[community]
        final_map = next_map  # type: ignore[assignment]
    # final_map currently maps original nodes to arbitrary ids; normalize
    id_map: Dict[int, int] = {}
    counter = 0
    result: Dict[str, int] = {}
    for node, community in final_map.items():
        if community not in id_map:
            id_map[community] = counter
            counter += 1
        result[node] = id_map[community]
    return result


def compute_density(cluster_nodes: Sequence[str], pair_weights: Mapping[Tuple[str, str], float]) -> float:
    n = len(cluster_nodes)
    if n <= 1:
        return 0.0
    cluster_set = set(cluster_nodes)
    edge_count = 0
    for (u, v), _ in pair_weights.items():
        if u in cluster_set and v in cluster_set and u != v:
            edge_count += 1
    max_edges = n * (n - 1) / 2
    if max_edges == 0:
        return 0.0
    return edge_count / max_edges


def chunked(iterable: Sequence[str], size: int) -> List[List[str]]:
    return [list(iterable[i : i + size]) for i in range(0, len(iterable), size)]


def draw_cluster_image(
    nodes: Sequence[str],
    edges: Sequence[Tuple[str, str, float]],
    module_info: Mapping[str, ModuleInfo],
    caption: str,
    output: Path,
    max_edge: int,
) -> None:
    count = len(nodes)
    width = min(max_edge, max(640, count * 60))
    height = min(max_edge, max(640, count * 60))
    canvas = Canvas(width, height)
    top_margin = 120
    coords = layout_positions(max(count, 1), width, height - top_margin)
    positions: Dict[str, Tuple[int, int]] = {}
    if count == 1:
        positions[nodes[0]] = (width // 2, top_margin + (height - top_margin) // 2)
    else:
        for node, (x, y) in zip(nodes, coords):
            positions[node] = (x, y + top_margin)
    edge_weights = edges
    if edge_weights:
        max_weight = max(weight for (_, _, weight) in edge_weights)
    else:
        max_weight = 1.0
    for src, dst, weight in edge_weights:
        if src not in positions or dst not in positions:
            continue
        src_pos = positions[src]
        dst_pos = positions[dst]
        thickness = max(1, int(round(2 * weight / max_weight)))
        canvas.draw_line(src_pos[0], src_pos[1], dst_pos[0], dst_pos[1], (120, 120, 120, 160), thickness)
    for node in nodes:
        x, y = positions[node]
        sloc = module_info.get(node, ModuleInfo(node, node, 0, 0, 0)).sloc
        radius = max(6, min(16, int(6 + math.log(sloc + 2, 2))))
        canvas.draw_circle(x, y, radius, (65, 105, 225, 230))
        label = node.replace("traigent.", "")[:18]
        draw_text(canvas, x - len(label) * 3, y + radius + 8, label)
    draw_text(canvas, 20, 30, caption, scale=2)
    canvas.save_png(output)


def summarize_cluster(
    cluster_id: int,
    modules: Sequence[str],
    module_info: Mapping[str, ModuleInfo],
    directed_edges: Sequence[Tuple[str, str, float]],
    output_path: Path,
    density: float,
) -> None:
    info_rows = [module_info.get(name, ModuleInfo(name, name, 0, 0, 0)) for name in modules]
    total_sloc = sum(row.sloc for row in info_rows)
    top_modules = sorted(info_rows, key=lambda row: (-row.sloc, row.name))[:10]
    internal_edges = [
        (src, dst, weight)
        for src, dst, weight in directed_edges
        if src in modules and dst in modules and src != dst
    ]
    internal_edges.sort(key=lambda row: (-row[2], row[0], row[1]))
    external_dependents = [
        (src, dst, weight)
        for src, dst, weight in directed_edges
        if src not in modules and dst in modules
    ]
    external_dependents.sort(key=lambda row: (-row[2], row[0], row[1]))
    lines: List[str] = []
    lines.append(f"# Cluster {cluster_id:02d}")
    lines.append("")
    lines.append(f"- Modules: {len(modules)}")
    lines.append(f"- Total SLOC: {total_sloc}")
    lines.append(f"- Density: {density:.3f}")
    lines.append("")
    lines.append("## Top Modules by SLOC")
    for row in top_modules:
        lines.append(
            f"- {row.name} — {row.sloc} SLOC (fan-in {row.fan_in}, fan-out {row.fan_out})"
        )
    if internal_edges:
        lines.append("")
        lines.append("## Strongest Internal Edges")
        for src, dst, weight in internal_edges[:15]:
            lines.append(f"- {src} → {dst} — weight {weight:.2f}")
    if external_dependents:
        lines.append("")
        lines.append("## Top External Dependents")
        for src, dst, weight in external_dependents[:15]:
            lines.append(f"- {src} → {dst} — weight {weight:.2f}")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def build_clusters(
    graph_path: Path,
    output_dir: Path,
    config: AtlasConfig,
) -> None:
    node_paths, raw_edges = load_graphml(graph_path)
    metrics_path = graph_path.with_name("metrics.csv")
    metrics = load_metrics(metrics_path) if metrics_path.exists() else {}
    module_info = build_module_info(node_paths, metrics)

    filtered_nodes = [
        node
        for node in node_paths
        if not matches_patterns(module_info[node].path, config.exclude_patterns)
    ]
    filtered_set = set(filtered_nodes)
    directed_edges = [
        (edge.source, edge.target, edge.weight)
        for edge in raw_edges
        if edge.source in filtered_set and edge.target in filtered_set
    ]

    undirected_graph, pair_weights = undirected_weights(directed_edges)
    for node in filtered_nodes:
        undirected_graph.setdefault(node, {})
    partition = louvain_clustering(undirected_graph)
    if not partition:
        partition = dict.fromkeys(filtered_nodes, 0)

    clusters: Dict[int, List[str]] = defaultdict(list)
    for node, cluster_id in partition.items():
        clusters[cluster_id].append(node)
    sorted_clusters = sorted(clusters.items(), key=lambda item: (-len(item[1]), item[0]))
    cluster_order_map = {cluster_id: idx for idx, (cluster_id, _) in enumerate(sorted_clusters)}

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "clusters").mkdir(parents=True, exist_ok=True)

    cluster_metadata: Dict[str, Dict[str, object]] = {}
    for original_cluster_id, modules in sorted_clusters:
        new_cluster_id = cluster_order_map[original_cluster_id]
        modules.sort()
        density = compute_density(modules, pair_weights)
        chunks = chunked(modules, config.max_nodes_per_view)
        for index, chunk in enumerate(chunks, start=1):
            chunk_edges = [
                (src, dst, weight)
                for src, dst, weight in directed_edges
                if src in chunk and dst in chunk
            ]
            caption = (
                f"CLUSTER {new_cluster_id:02d} ({index}/{len(chunks)}) - {len(chunk)} MODULES"
                if len(chunks) > 1
                else f"CLUSTER {new_cluster_id:02d} - {len(chunk)} MODULES"
            )
            filename = (
                f"cluster_{new_cluster_id:02d}_part_{index:02d}.png"
                if len(chunks) > 1
                else f"cluster_{new_cluster_id:02d}.png"
            )
            output_path = output_dir / "clusters" / filename
            draw_cluster_image(chunk, chunk_edges, module_info, caption, output_path, config.image_max_edge_px)
        summary_path = output_dir / "clusters" / f"cluster_{new_cluster_id:02d}.md"
        summarize_cluster(
            new_cluster_id,
            modules,
            module_info,
            directed_edges,
            summary_path,
            density,
        )
        for module in modules:
            cluster_metadata[module] = {
                "cluster": new_cluster_id,
                "size": len(modules),
                "density": density,
            }

    (output_dir / "clusters.json").write_text(
        json.dumps(cluster_metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    index_lines = [
        "<!DOCTYPE html>",
        "<html lang=\"en\">",
        "<head>",
        "  <meta charset=\"utf-8\">",
        "  <title>Cluster Views</title>",
        "  <style>body{font-family:Arial,Helvetica,sans-serif;margin:40px;}figure{margin:20px 0;}h2{margin-top:36px;}figcaption{font-weight:bold;margin-bottom:6px;} .viz-frame{max-width:100%;overflow:auto;border:1px solid #ccc;border-radius:4px;background:#fff;padding:8px;} .viz-frame img{display:block;}</style>",
        "</head>",
        "<body>",
        "  <h1>Clustered Capability Views</h1>",
        f"  <p>Input graph: {graph_path}</p>",
    ]
    for original_cluster_id, modules in sorted_clusters:
        cluster_id = cluster_order_map[original_cluster_id]
        chunks = chunked(modules, config.max_nodes_per_view)
        index_lines.append(f"  <h2>Cluster {cluster_id:02d}</h2>")
        index_lines.append(f"  <p>{len(modules)} modules, density {compute_density(modules, pair_weights):.3f}</p>")
        for idx in range(1, len(chunks) + 1):
            filename = (
                f"cluster_{cluster_id:02d}_part_{idx:02d}.png"
                if len(chunks) > 1
                else f"cluster_{cluster_id:02d}.png"
            )
            caption = (
                f"Cluster {cluster_id:02d} ({idx}/{len(chunks)})"
                if len(chunks) > 1
                else f"Cluster {cluster_id:02d}"
            )
            index_lines.append("  <figure>")
            index_lines.append(f"    <figcaption>{caption}</figcaption>")
            index_lines.append("    <div class=\"viz-frame\">")
            index_lines.append(f"      <img src=\"clusters/{filename}\" alt=\"{caption}\">")
            index_lines.append("    </div>")
            index_lines.append("  </figure>")
        index_lines.append(
            f"  <p><a href=\"clusters/cluster_{cluster_id:02d}.md\">Summary (Markdown)</a></p>"
        )
    index_lines.extend(["</body>", "</html>"])
    (output_dir / "index.html").write_text("\n".join(index_lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate clustered views from dependency graph")
    parser.add_argument("--graph", type=Path, required=True, help="Path to deps.graphml")
    parser.add_argument("--out", type=Path, required=True, help="Output directory for cluster artifacts")
    parser.add_argument(
        "--config", type=Path, default=Path("config/viz.yaml"), help="Visualization configuration file"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_dir = Path.cwd()
    try:
        graph_path = validate_path(args.graph, base_dir, must_exist=True)
        output_dir = validate_path(args.out, base_dir)
        config_path = validate_path(args.config, base_dir, must_exist=True)
    except (PathTraversalError, FileNotFoundError) as exc:
        raise SystemExit(f"Error: {exc}") from exc
    config = load_config(config_path)
    build_clusters(graph_path, output_dir, config)


if __name__ == "__main__":  # pragma: no cover
    main()
