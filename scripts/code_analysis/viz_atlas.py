"""Repository atlas generator.

Builds high-level, skimmable visualizations from the existing code-analysis
artifacts (`inventory.csv`, `metrics.csv`, `deps.graphml`).
"""

from __future__ import annotations

import argparse
import csv
import fnmatch
import math
import re
from collections import defaultdict

# Use defusedxml to prevent XXE attacks
try:
    import defusedxml.ElementTree as ET
except ImportError as exc:
    raise RuntimeError(
        "defusedxml is required for XML parsing in scripts/code_analysis. "
        "Install it (e.g. `pip install defusedxml` or enable the TraiGent "
        "security extras)."
    ) from exc
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

try:  # pragma: no cover - allow running as a script
    from .dependency_graph import Canvas, layout_positions
except ImportError:  # pragma: no cover
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from scripts.code_analysis.dependency_graph import Canvas, layout_positions


@dataclass
class AtlasConfig:
    max_nodes_per_view: int = 120
    image_max_edge_px: int = 4000
    top_hubs: int = 15
    exclude_patterns: Sequence[str] = ()
    vertical_layers: Sequence[str] = ()


@dataclass
class GraphEdge:
    source: str
    target: str
    weight: float


@dataclass
class ModuleData:
    name: str
    path: str
    sloc: int
    fan_in: int
    fan_out: int
    test_count: int


FONT_5x7: dict[str, Sequence[str]] = {
    " ": (
        "     ",
        "     ",
        "     ",
        "     ",
        "     ",
        "     ",
        "     ",
    ),
    "?": (
        " ### ",
        "#   #",
        "    #",
        "   # ",
        "  #  ",
        "     ",
        "  #  ",
    ),
    "#": (
        " # # ",
        "# # #",
        " # # ",
        "# # #",
        " # # ",
        "     ",
        "     ",
    ),
    "0": (
        " ### ",
        "#   #",
        "#  ##",
        "# # #",
        "##  #",
        "#   #",
        " ### ",
    ),
    "1": (
        "  #  ",
        " ##  ",
        "# #  ",
        "  #  ",
        "  #  ",
        "  #  ",
        "#####",
    ),
    "2": (
        " ### ",
        "#   #",
        "    #",
        "   # ",
        "  #  ",
        " #   ",
        "#####",
    ),
    "3": (
        " ### ",
        "#   #",
        "    #",
        "  ## ",
        "    #",
        "#   #",
        " ### ",
    ),
    "4": (
        "#   #",
        "#   #",
        "#   #",
        "#####",
        "    #",
        "    #",
        "    #",
    ),
    "5": (
        "#####",
        "#    ",
        "#    ",
        "#### ",
        "    #",
        "#   #",
        " ### ",
    ),
    "6": (
        " ### ",
        "#   #",
        "#    ",
        "#### ",
        "#   #",
        "#   #",
        " ### ",
    ),
    "7": (
        "#####",
        "    #",
        "   # ",
        "  #  ",
        "  #  ",
        "  #  ",
        "  #  ",
    ),
    "8": (
        " ### ",
        "#   #",
        "#   #",
        " ### ",
        "#   #",
        "#   #",
        " ### ",
    ),
    "9": (
        " ### ",
        "#   #",
        "#   #",
        " ####",
        "    #",
        "#   #",
        " ### ",
    ),
    "A": (
        "  #  ",
        " # # ",
        "#   #",
        "#####",
        "#   #",
        "#   #",
        "#   #",
    ),
    "B": (
        "#### ",
        "#   #",
        "#   #",
        "#### ",
        "#   #",
        "#   #",
        "#### ",
    ),
    "C": (
        " ### ",
        "#   #",
        "#    ",
        "#    ",
        "#    ",
        "#   #",
        " ### ",
    ),
    "D": (
        "#### ",
        "#   #",
        "#   #",
        "#   #",
        "#   #",
        "#   #",
        "#### ",
    ),
    "E": (
        "#####",
        "#    ",
        "#    ",
        "#### ",
        "#    ",
        "#    ",
        "#####",
    ),
    "F": (
        "#####",
        "#    ",
        "#    ",
        "#### ",
        "#    ",
        "#    ",
        "#    ",
    ),
    "G": (
        " ### ",
        "#   #",
        "#    ",
        "# ###",
        "#   #",
        "#   #",
        " ### ",
    ),
    "H": (
        "#   #",
        "#   #",
        "#   #",
        "#####",
        "#   #",
        "#   #",
        "#   #",
    ),
    "I": (
        " ### ",
        "  #  ",
        "  #  ",
        "  #  ",
        "  #  ",
        "  #  ",
        " ### ",
    ),
    "J": (
        "  ###",
        "   # ",
        "   # ",
        "   # ",
        "#  # ",
        "#  # ",
        " ##  ",
    ),
    "K": (
        "#   #",
        "#  # ",
        "# #  ",
        "##   ",
        "# #  ",
        "#  # ",
        "#   #",
    ),
    "L": (
        "#    ",
        "#    ",
        "#    ",
        "#    ",
        "#    ",
        "#    ",
        "#####",
    ),
    "M": (
        "#   #",
        "## ##",
        "# # #",
        "#   #",
        "#   #",
        "#   #",
        "#   #",
    ),
    "N": (
        "#   #",
        "##  #",
        "# # #",
        "#  ##",
        "#   #",
        "#   #",
        "#   #",
    ),
    "O": (
        " ### ",
        "#   #",
        "#   #",
        "#   #",
        "#   #",
        "#   #",
        " ### ",
    ),
    "P": (
        "#### ",
        "#   #",
        "#   #",
        "#### ",
        "#    ",
        "#    ",
        "#    ",
    ),
    "Q": (
        " ### ",
        "#   #",
        "#   #",
        "#   #",
        "# # #",
        "#  # ",
        " ## #",
    ),
    "R": (
        "#### ",
        "#   #",
        "#   #",
        "#### ",
        "# #  ",
        "#  # ",
        "#   #",
    ),
    "S": (
        " ### ",
        "#   #",
        "#    ",
        " ### ",
        "    #",
        "#   #",
        " ### ",
    ),
    "T": (
        "#####",
        "  #  ",
        "  #  ",
        "  #  ",
        "  #  ",
        "  #  ",
        "  #  ",
    ),
    "U": (
        "#   #",
        "#   #",
        "#   #",
        "#   #",
        "#   #",
        "#   #",
        " ### ",
    ),
    "V": (
        "#   #",
        "#   #",
        "#   #",
        "#   #",
        "#   #",
        " # # ",
        "  #  ",
    ),
    "W": (
        "#   #",
        "#   #",
        "#   #",
        "# # #",
        "# # #",
        "## ##",
        "#   #",
    ),
    "X": (
        "#   #",
        "#   #",
        " # # ",
        "  #  ",
        " # # ",
        "#   #",
        "#   #",
    ),
    "Y": (
        "#   #",
        "#   #",
        " # # ",
        "  #  ",
        "  #  ",
        "  #  ",
        "  #  ",
    ),
    "Z": (
        "#####",
        "    #",
        "   # ",
        "  #  ",
        " #   ",
        "#    ",
        "#####",
    ),
    "-": (
        "     ",
        "     ",
        "     ",
        " ### ",
        "     ",
        "     ",
        "     ",
    ),
    "/": (
        "    #",
        "   # ",
        "   # ",
        "  #  ",
        " #   ",
        " #   ",
        "#    ",
    ),
    "(": (
        "   # ",
        "  #  ",
        " #   ",
        " #   ",
        " #   ",
        "  #  ",
        "   # ",
    ),
    ")": (
        "#   ",
        " #  ",
        "  # ",
        "  # ",
        "  # ",
        " #  ",
        "#   ",
    ),
    ":": (
        "     ",
        "  #  ",
        "     ",
        "     ",
        "     ",
        "  #  ",
        "     ",
    ),
    ",": (
        "     ",
        "     ",
        "     ",
        "     ",
        "  ## ",
        "  #  ",
        " #   ",
    ),
    ".": (
        "     ",
        "     ",
        "     ",
        "     ",
        "     ",
        "  ## ",
        "  ## ",
    ),
    "&": (
        " ##  ",
        "#  # ",
        " ##  ",
        " ## #",
        "#  # ",
        "#  # ",
        " ## #",
    ),
}


def draw_text(
    canvas: Canvas,
    x: int,
    y: int,
    text: str,
    *,
    color: tuple[int, int, int, int] = (20, 20, 20, 255),
    scale: int = 1,
) -> None:
    """Render uppercase text onto the canvas using a simple 5x7 bitmap font."""

    cursor_x = x
    cursor_y = y
    line_height = (7 + 1) * scale
    for ch in text.upper():
        if ch == "\n":
            cursor_x = x
            cursor_y += line_height
            continue
        pattern = FONT_5x7.get(ch, FONT_5x7.get("?"))
        if not pattern:
            continue
        height = len(pattern)
        width = len(pattern[0]) if pattern else 0
        for row in range(height):
            for col in range(width):
                if pattern[row][col] != " ":
                    for dy in range(scale):
                        for dx in range(scale):
                            canvas.set_pixel(
                                cursor_x + col * scale + dx,
                                cursor_y + row * scale + dy,
                                color,
                            )
        cursor_x += (width + 1) * scale


def load_config(path: Path) -> AtlasConfig:
    defaults = AtlasConfig()
    if not path.exists():
        return defaults

    def parse_scalar(value: str):
        value = value.strip().strip('"')
        if not value:
            return ""
        if value.isdigit():
            return int(value)
        try:
            return float(value)
        except ValueError:
            return value

    data: dict[str, object] = {}
    current_key: str | None = None
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            if not raw_line.strip() or raw_line.lstrip().startswith("#"):
                continue
            if raw_line.startswith("-") and current_key:
                value = raw_line[1:].strip()
                parsed = parse_scalar(value)
                data.setdefault(current_key, []).append(parsed)
                continue
            if ":" in raw_line:
                key, value = raw_line.split(":", 1)
                key = key.strip()
                value = value.strip()
                if value:
                    data[key] = parse_scalar(value)
                    current_key = None
                else:
                    data[key] = []
                    current_key = key
    return AtlasConfig(
        max_nodes_per_view=int(data.get("max_nodes_per_view", defaults.max_nodes_per_view)),
        image_max_edge_px=int(data.get("image_max_edge_px", defaults.image_max_edge_px)),
        top_hubs=int(data.get("top_hubs", defaults.top_hubs)),
        exclude_patterns=tuple(
            str(item) for item in data.get("exclude_patterns", defaults.exclude_patterns)
        ),
        vertical_layers=tuple(
            str(item) for item in data.get("vertical_layers", defaults.vertical_layers)
        ),
    )


def load_inventory(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return {row["module"]: row for row in reader}


def load_metrics(path: Path) -> dict[str, dict[str, object]]:
    results: dict[str, dict[str, object]] = {}
    if not path.exists():
        return results
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            module = row["module"]
            results[module] = {
                "path": row.get("path", ""),
                "sloc": int(float(row.get("sloc", "0") or 0)),
                "fan_in": int(float(row.get("fan_in", "0") or 0)),
                "fan_out": int(float(row.get("fan_out", "0") or 0)),
                "test_count": int(float(row.get("test_count", "0") or 0)),
            }
    return results


def load_graphml(path: Path) -> tuple[dict[str, str], list[GraphEdge]]:
    if not path.exists():
        return {}, []
    tree = ET.parse(str(path))
    root = tree.getroot()
    ns = {"g": "http://graphml.graphdrawing.org/xmlns"}

    key_map: dict[str, str] = {}
    for key in root.findall("g:key", ns):
        attr_name = key.attrib.get("attr_name")
        if attr_name:
            key_map[attr_name] = key.attrib["id"]
    node_key = key_map.get("path", "d0")
    edge_key = key_map.get("weight", "d1")

    node_paths: dict[str, str] = {}
    for node in root.findall("g:graph/g:node", ns):
        node_id = node.attrib["id"]
        path_value = ""
        for data in node.findall("g:data", ns):
            if data.attrib.get("key") == node_key and data.text:
                path_value = data.text.strip()
        node_paths[node_id] = path_value

    edges: list[GraphEdge] = []
    for edge in root.findall("g:graph/g:edge", ns):
        weight = 1.0
        for data in edge.findall("g:data", ns):
            if data.attrib.get("key") == edge_key and data.text:
                try:
                    weight = float(data.text.strip())
                except ValueError:
                    weight = 1.0
        edges.append(GraphEdge(edge.attrib["source"], edge.attrib["target"], weight))
    return node_paths, edges


def sanitize_filename(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", name)


def top_package(module: str) -> str:
    parts = module.split(".")
    if len(parts) >= 2:
        return ".".join(parts[:2])
    return module


def matches_patterns(path: str, patterns: Sequence[str]) -> bool:
    normalized = path.replace("\\", "/")
    for pattern in patterns:
        if fnmatch.fnmatch(normalized, pattern):
            return True
    return False


def strongly_connected_components(
    nodes: Sequence[str], adjacency: dict[str, Sequence[str]]
) -> list[list[str]]:
    index_map: dict[str, int] = {}
    lowlink: dict[str, int] = {}
    index = 0
    stack: list[str] = []
    on_stack: dict[str, bool] = {}
    components: list[list[str]] = []

    def strongconnect(node: str) -> None:
        nonlocal index
        index_map[node] = index
        lowlink[node] = index
        index += 1
        stack.append(node)
        on_stack[node] = True

        for neighbor in adjacency.get(node, []):
            if neighbor not in index_map:
                strongconnect(neighbor)
                lowlink[node] = min(lowlink[node], lowlink[neighbor])
            elif on_stack.get(neighbor):
                lowlink[node] = min(lowlink[node], index_map[neighbor])

        if lowlink[node] == index_map[node]:
            component: list[str] = []
            while True:
                w = stack.pop()
                on_stack[w] = False
                component.append(w)
                if w == node:
                    break
            components.append(sorted(component))

    for node in nodes:
        if node not in index_map:
            strongconnect(node)
    return components


class TreeNode:
    def __init__(self, name: str) -> None:
        self.name = name
        self.children: dict[str, TreeNode] = {}
        self.is_terminal = False

    def child(self, name: str) -> TreeNode:
        if name not in self.children:
            self.children[name] = TreeNode(name)
        return self.children[name]


class AtlasBuilder:
    def __init__(self, input_dir: Path, output_dir: Path, config: AtlasConfig) -> None:
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.config = config
        self.inventory: dict[str, dict[str, str]] = {}
        self.metrics: dict[str, dict[str, object]] = {}
        self.node_paths: dict[str, str] = {}
        self.edges: list[GraphEdge] = []
        self.module_data: dict[str, ModuleData] = {}
        self.filtered_modules: list[str] = []
        self.adjacency: dict[str, list[str]] = {}
        self.inverse_adjacency: dict[str, list[str]] = {}
        self.edge_weights: dict[tuple[str, str], float] = {}
        self.packages: dict[str, list[str]] = {}
        self.package_chunks: list[dict[str, object]] = []
        self.hub_cards: list[dict[str, object]] = []
        self.components: list[dict[str, object]] = []
        self.component_edges: dict[tuple[int, int], float] = {}
        self.component_positions: dict[int, tuple[int, int]] = {}
        self.layer_order: list[list[int]] = []

    def load_sources(self) -> None:
        self.inventory = load_inventory(self.input_dir / "inventory.csv")
        self.metrics = load_metrics(self.input_dir / "metrics.csv")
        self.node_paths, self.edges = load_graphml(self.input_dir / "deps.graphml")
        modules = sorted(self.node_paths.keys())

        module_info: dict[str, ModuleData] = {}
        for module in modules:
            inventory_row = self.inventory.get(module, {})
            metrics_row = self.metrics.get(module, {})
            path = (
                inventory_row.get("path")
                or metrics_row.get("path")
                or self.node_paths.get(module)
                or module.replace(".", "/") + ".py"
            )
            sloc = int(float(metrics_row.get("sloc", inventory_row.get("sloc", 0)) or 0))
            fan_in = int(float(metrics_row.get("fan_in", 0) or 0))
            fan_out = int(float(metrics_row.get("fan_out", 0) or 0))
            test_count = int(float(metrics_row.get("test_count", 0) or 0))
            module_info[module] = ModuleData(module, path, sloc, fan_in, fan_out, test_count)
        self.module_data = module_info

        included: list[str] = []
        for module, data in module_info.items():
            if matches_patterns(data.path, self.config.exclude_patterns):
                continue
            included.append(module)
        self.filtered_modules = sorted(included)

        adjacency: dict[str, list[str]] = {node: [] for node in self.filtered_modules}
        reverse: dict[str, list[str]] = {node: [] for node in self.filtered_modules}
        weights: dict[tuple[str, str], float] = {}
        for edge in self.edges:
            if edge.source not in adjacency or edge.target not in adjacency:
                continue
            adjacency[edge.source].append(edge.target)
            reverse[edge.target].append(edge.source)
            weights[(edge.source, edge.target)] = (
                weights.get((edge.source, edge.target), 0.0) + edge.weight
            )
        self.adjacency = adjacency
        self.inverse_adjacency = reverse
        self.edge_weights = weights

        packages: dict[str, list[str]] = defaultdict(list)
        for module in self.filtered_modules:
            packages[top_package(module)].append(module)
        for pkg in packages.values():
            pkg.sort()
        self.packages = dict(sorted(packages.items()))

    def prepare_directories(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "by_top_package").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "neighborhoods").mkdir(parents=True, exist_ok=True)

    def build_module_tree(self) -> None:
        root = TreeNode("(root)")
        for module in self.filtered_modules:
            parts = module.split(".")
            node = root
            for idx, part in enumerate(parts):
                node = node.child(part)
                if idx == len(parts) - 1:
                    node.is_terminal = True

        mermaid_lines = ["graph TD", '    root["ROOT"]']
        markdown_lines: list[str] = []
        counter = 0

        def visit_mermaid(node: TreeNode, parent_id: str) -> None:
            nonlocal counter
            if node.name == "(root)":
                current_id = parent_id
            else:
                current_id = f"n{counter}"
                counter += 1
                label = node.name.replace('"', "'")
                mermaid_lines.append(f'    {current_id}["{label}"]')
                mermaid_lines.append(f"    {parent_id} --> {current_id}")
            for child in sorted(node.children.values(), key=lambda c: c.name):
                visit_mermaid(child, current_id)

        def visit_markdown(node: TreeNode, depth: int) -> None:
            if node.name != "(root)":
                suffix = ""
                if node.is_terminal and node.children:
                    suffix = " (module & package)"
                elif node.is_terminal:
                    suffix = " (module)"
                bullet = "  " * depth + f"- {node.name}{suffix}"
                markdown_lines.append(bullet)
                next_depth = depth + 1
            else:
                next_depth = depth
            for child in sorted(node.children.values(), key=lambda c: c.name):
                visit_markdown(child, next_depth)

        visit_mermaid(root, "root")
        visit_markdown(root, 0)

        (self.output_dir / "modules_tree.mmd").write_text(
            "\n".join(mermaid_lines), encoding="utf-8"
        )
        (self.output_dir / "modules_tree.md").write_text(
            "\n".join(markdown_lines), encoding="utf-8"
        )

    def build_component_overview(self) -> None:
        components = strongly_connected_components(self.filtered_modules, self.adjacency)
        component_map: dict[str, int] = {}
        component_data: list[dict[str, object]] = []
        for idx, nodes in enumerate(components):
            component_map.update(dict.fromkeys(nodes, idx))
            total_sloc = sum(self.module_data[node].sloc for node in nodes)
            representative = sorted(nodes, key=lambda n: (-self.module_data[n].sloc, n))[0]
            component_data.append(
                {
                    "id": idx,
                    "nodes": nodes,
                    "size": len(nodes),
                    "sloc": total_sloc,
                    "representative": representative,
                }
            )
        edge_map: dict[tuple[int, int], float] = {}
        for (src, dst), weight in self.edge_weights.items():
            comp_src = component_map[src]
            comp_dst = component_map[dst]
            if comp_src == comp_dst:
                continue
            edge_map[(comp_src, comp_dst)] = edge_map.get((comp_src, comp_dst), 0.0) + weight
        self.components = component_data
        self.component_edges = edge_map

        patterns = [re.compile(layer, re.IGNORECASE) for layer in self.config.vertical_layers]
        layer_buckets: dict[int, list[int]] = defaultdict(list)
        for comp in component_data:
            representative = comp["representative"]
            layer_index = len(patterns)
            for idx, pattern in enumerate(patterns):
                if pattern.search(representative):
                    layer_index = idx
                    break
            layer_buckets[layer_index].append(comp["id"])
        ordered_layers = [layer_buckets[idx] for idx in sorted(layer_buckets.keys())]
        for layer in ordered_layers:
            layer.sort(key=lambda cid: self.components[cid]["representative"])  # type: ignore[index]
        self.layer_order = ordered_layers

        positions: dict[int, tuple[int, int]] = {}
        margin_x = 140
        margin_y = 140
        layer_count = max(1, len(ordered_layers))
        largest_layer = max((len(layer) for layer in ordered_layers), default=1)
        width = min(self.config.image_max_edge_px, max(800, margin_x * 2 + largest_layer * 260))
        height = min(
            self.config.image_max_edge_px,
            max(600, margin_y * 2 + max(layer_count - 1, 1) * 220),
        )
        if layer_count == 1:
            y_positions = [height // 2]
        else:
            layer_spacing = (height - 2 * margin_y) / (layer_count - 1)
            y_positions = [int(margin_y + i * layer_spacing) for i in range(layer_count)]
        for layer_idx, layer in enumerate(ordered_layers):
            count = max(1, len(layer))
            if count == 1:
                xs = [width // 2]
            else:
                layer_width = width - 2 * margin_x
                step = layer_width / (count - 1)
                xs = [int(margin_x + i * step) for i in range(count)]
            for node_idx, component_id in enumerate(layer):
                positions[component_id] = (xs[node_idx], y_positions[layer_idx])
        self.component_positions = positions

        canvas = Canvas(width, height)
        max_weight = max(edge_map.values()) if edge_map else 1.0
        layer_colors = [
            (70, 130, 180, 235),
            (95, 158, 160, 235),
            (60, 179, 113, 235),
            (218, 165, 32, 235),
            (147, 112, 219, 235),
            (205, 92, 92, 235),
        ]
        for (src, dst), weight in edge_map.items():
            src_pos = positions[src]
            dst_pos = positions[dst]
            thickness = max(1, int(round(2 * weight / max_weight)))
            color = (120, 120, 120, 160)
            canvas.draw_line(src_pos[0], src_pos[1], dst_pos[0], dst_pos[1], color, thickness)
        for comp in component_data:
            comp_id = comp["id"]
            x, y = positions[comp_id]
            radius = max(8, min(24, int(6 + math.sqrt(comp["sloc"] + 1) / 8)))
            color = layer_colors[comp_id % len(layer_colors)]
            canvas.draw_circle(x, y, radius, color)
            label = comp["representative"].replace("traigent.", "")
            draw_text(canvas, x - len(label) * 3, y + radius + 10, label[:18])
            draw_text(canvas, x - 30, y - radius - 20, f"{comp['size']} MOD"[:18])
        draw_text(canvas, 20, 20, "COMPONENT DAG OVERVIEW", scale=2)
        canvas.save_png(self.output_dir / "deps_overview.png")

    def build_package_views(self) -> None:
        image_entries: list[dict[str, object]] = []
        max_nodes = self.config.max_nodes_per_view
        max_edge = self.config.image_max_edge_px

        for package, modules in self.packages.items():
            if not modules:
                continue
            chunks = [modules[i : i + max_nodes] for i in range(0, len(modules), max_nodes)]
            chunk_records: list[dict[str, object]] = []
            for index, chunk in enumerate(chunks, start=1):
                chunk_nodes = chunk
                chunk_edges = [
                    (src, dst, self.edge_weights[(src, dst)])
                    for (src, dst) in self.edge_weights
                    if src in chunk_nodes and dst in chunk_nodes
                ]
                count = len(chunk_nodes)
                width = min(max_edge, max(640, count * 60))
                height = min(max_edge, max(640, count * 60))
                canvas = Canvas(width, height)
                top_margin = 120
                radius = min(width, height - top_margin) // 2 - 40
                if radius < 60:
                    radius = min(width, height) // 2 - 40
                center_x = width // 2
                center_y = top_margin + radius
                positions = {}
                if count == 1:
                    positions[chunk_nodes[0]] = (center_x, center_y)
                else:
                    coords = layout_positions(count, width, height - top_margin)
                    for node, (x, y) in zip(chunk_nodes, coords):
                        positions[node] = (x, y + top_margin)
                max_weight = max((w for (_, _, w) in chunk_edges), default=1.0)
                for src, dst, weight in chunk_edges:
                    src_pos = positions[src]
                    dst_pos = positions[dst]
                    thickness = max(1, int(round(2 * weight / max_weight)))
                    canvas.draw_line(
                        src_pos[0],
                        src_pos[1],
                        dst_pos[0],
                        dst_pos[1],
                        (130, 130, 130, 150),
                        thickness,
                    )
                for node in chunk_nodes:
                    x, y = positions[node]
                    radius_node = max(
                        6,
                        min(14, int(6 + math.log(self.module_data[node].sloc + 1, 2))),
                    )
                    canvas.draw_circle(x, y, radius_node, (65, 105, 225, 220))
                    label = node.replace("traigent.", "")
                    draw_text(canvas, x - len(label[:16]) * 3, y + radius_node + 8, label[:16])
                caption = f"{package.upper()} ({index}/{len(chunks)}) - {count} MODULES"
                draw_text(canvas, 20, 30, caption, scale=2)
                filename = f"{sanitize_filename(package.replace('.', '_'))}_{index:02d}_of_{len(chunks):02d}.png"
                output_path = self.output_dir / "by_top_package" / filename
                canvas.save_png(output_path)
                chunk_records.append(
                    {
                        "filename": f"by_top_package/{filename}",
                        "caption": caption,
                    }
                )
            image_entries.append(
                {
                    "package": package,
                    "chunks": chunk_records,
                }
            )
        self.package_chunks = image_entries

    def build_neighborhoods(self) -> None:
        modules = sorted(
            self.filtered_modules,
            key=lambda name: max(self.module_data[name].fan_in, self.module_data[name].fan_out),
            reverse=True,
        )[: self.config.top_hubs]

        entries: list[dict[str, object]] = []
        max_edge = self.config.image_max_edge_px
        for module in modules:
            inbound = [
                (neighbor, self.edge_weights[(neighbor, module)])
                for neighbor in self.inverse_adjacency.get(module, [])
                if (neighbor, module) in self.edge_weights
            ]
            outbound = [
                (neighbor, self.edge_weights[(module, neighbor)])
                for neighbor in self.adjacency.get(module, [])
                if (module, neighbor) in self.edge_weights
            ]
            inbound.sort(key=lambda item: (-item[1], item[0]))
            outbound.sort(key=lambda item: (-item[1], item[0]))
            inbound = inbound[:10]
            outbound = outbound[:10]

            width = min(max_edge, 1200)
            height = min(max_edge, 900)
            canvas = Canvas(width, height)
            top_margin = 120
            center_x = width // 2
            center_y = (height + top_margin) // 2
            radius = min(width, height - top_margin) // 2 - 100
            if radius < 120:
                radius = min(width, height) // 2 - 120

            def fan_positions(
                count: int, start_angle: float, end_angle: float
            ) -> list[tuple[int, int]]:
                if count == 0:
                    return []
                if count == 1:
                    angles = [0.5 * (start_angle + end_angle)]
                else:
                    step = (end_angle - start_angle) / (count - 1)
                    angles = [start_angle + i * step for i in range(count)]
                coords: list[tuple[int, int]] = []
                for angle in angles:
                    x = int(center_x + radius * math.cos(angle))
                    y = int(center_y + radius * math.sin(angle))
                    coords.append((x, y))
                return coords

            inbound_positions = fan_positions(len(inbound), math.pi * 0.75, math.pi * 1.25)
            outbound_positions = fan_positions(len(outbound), -math.pi * 0.25, math.pi * 0.25)
            max_in_weight = max((w for (_, w) in inbound), default=1.0)
            max_out_weight = max((w for (_, w) in outbound), default=1.0)

            for (neighbor, weight), (x, y) in zip(inbound, inbound_positions):
                canvas.draw_line(
                    x,
                    y,
                    center_x,
                    center_y,
                    (65, 105, 225, 200),
                    max(1, int(round(3 * weight / max_in_weight))),
                )
                canvas.draw_circle(x, y, 8, (65, 105, 225, 255))
                draw_text(canvas, x - 40, y + 14, neighbor.replace("traigent.", "")[:16])
            for (neighbor, weight), (x, y) in zip(outbound, outbound_positions):
                canvas.draw_line(
                    center_x,
                    center_y,
                    x,
                    y,
                    (34, 139, 34, 200),
                    max(1, int(round(3 * weight / max_out_weight))),
                )
                canvas.draw_circle(x, y, 8, (34, 139, 34, 255))
                draw_text(canvas, x - 40, y + 14, neighbor.replace("traigent.", "")[:16])

            center_color = (205, 92, 92, 240)
            canvas.draw_circle(center_x, center_y, 14, center_color)
            title = module.upper()
            draw_text(canvas, 20, 30, f"{title} NEIGHBORHOOD", scale=2)
            draw_text(
                canvas,
                20,
                70,
                f"FAN-IN {self.module_data[module].fan_in} | FAN-OUT {self.module_data[module].fan_out}",
                scale=1,
            )
            draw_text(
                canvas,
                center_x - len(module.replace("traigent.", "")) * 3,
                center_y - 26,
                module.replace("traigent.", "")[:18],
            )

            filename = sanitize_filename(module.replace(".", "_")) + ".png"
            output_path = self.output_dir / "neighborhoods" / filename
            canvas.save_png(output_path)
            entries.append(
                {
                    "module": module,
                    "filename": f"neighborhoods/{filename}",
                    "inbound": inbound,
                    "outbound": outbound,
                }
            )
        self.hub_cards = entries

    def write_index(self) -> None:
        html_lines = [
            "<!DOCTYPE html>",
            '<html lang="en">',
            "<head>",
            '  <meta charset="utf-8">',
            "  <title>Repository Atlas</title>",
            "  <style>body{font-family:Arial,Helvetica,sans-serif;margin:40px;}figure{margin:20px 0;}h2{margin-top:48px;}figcaption{font-weight:bold;margin-bottom:8px;}ul{line-height:1.5;} .viz-frame{max-width:100%;overflow:auto;border:1px solid #ccc;border-radius:4px;background:#fff;padding:8px;} .viz-frame img{display:block;} </style>",
            "</head>",
            "<body>",
            "  <h1>Repository Atlas</h1>",
            f"  <p>Input: {self.input_dir}</p>",
            '  <p><a href="modules_tree.md">Module Tree (Markdown)</a> | <a href="modules_tree.mmd">Module Tree (Mermaid)</a></p>',
            "  <h2>Overview</h2>",
            "  <figure>",
            "    <figcaption>Strongly Connected Components DAG</figcaption>",
            '    <div class="viz-frame">',
            '      <img src="deps_overview.png" alt="Component overview">',
            "    </div>",
            "  </figure>",
            "  <h2>By Package</h2>",
        ]
        for entry in self.package_chunks:
            html_lines.append(f"  <h3>{entry['package']}</h3>")
            for chunk in entry["chunks"]:
                html_lines.append("  <figure>")
                html_lines.append(f"    <figcaption>{chunk['caption']}</figcaption>")
                html_lines.append('    <div class="viz-frame">')
                html_lines.append(f'      <img src="{chunk["filename"]}" alt="{chunk["caption"]}">')
                html_lines.append("    </div>")
                html_lines.append("  </figure>")
        html_lines.append("  <h2>Neighborhoods</h2>")
        for entry in self.hub_cards:
            caption = f"{entry['module']} (top {len(entry['inbound'])} inbound / {len(entry['outbound'])} outbound)"
            html_lines.append("  <figure>")
            html_lines.append(f"    <figcaption>{caption}</figcaption>")
            html_lines.append('    <div class="viz-frame">')
            html_lines.append(f'      <img src="{entry["filename"]}" alt="{caption}">')
            html_lines.append("    </div>")
            html_lines.append("  </figure>")
        html_lines.extend(["</body>", "</html>"])
        (self.output_dir / "index.html").write_text("\n".join(html_lines), encoding="utf-8")

    def build(self) -> None:
        self.load_sources()
        self.prepare_directories()
        self.build_module_tree()
        self.build_component_overview()
        self.build_package_views()
        self.build_neighborhoods()
        self.write_index()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate repository atlas from code analysis artifacts."
    )
    parser.add_argument(
        "--in",
        dest="input_dir",
        type=Path,
        required=True,
        help="Input directory containing inventory.csv, metrics.csv, deps.graphml",
    )
    parser.add_argument(
        "--out",
        dest="output_dir",
        type=Path,
        required=True,
        help="Output directory for atlas artifacts",
    )
    parser.add_argument(
        "--config",
        dest="config_path",
        type=Path,
        default=Path("config/viz.yaml"),
        help="Path to viz configuration file",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config_path)
    builder = AtlasBuilder(args.input_dir, args.output_dir, config)
    builder.build()


if __name__ == "__main__":  # pragma: no cover
    main()
