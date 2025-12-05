"""Build a static import graph for source modules."""

from __future__ import annotations

import argparse
import ast
import math
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:  # pragma: no cover
    from .analysis_utils import load_ast, safe_relpath, to_module_name
except ImportError:  # pragma: no cover
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from scripts.code_analysis.analysis_utils import load_ast, safe_relpath, to_module_name


@dataclass
class ModuleRecord:
    name: str
    path: Path
    ast_tree: Optional[ast.AST]


@dataclass
class Edge:
    source: str
    target: str
    weight: int


class ModuleIndex:
    def __init__(self, source_root: Path) -> None:
        self.source_root = source_root
        self.prefix = source_root.name
        self.module_to_path: Dict[str, Path] = {}
        self.path_to_module: Dict[Path, str] = {}
        self.modules: Dict[str, ModuleRecord] = {}
        self._build()

    def _build(self) -> None:
        for path in sorted(self.source_root.rglob("*.py")):
            module_name = to_module_name(self.source_root, path)
            tree = load_ast(path)
            record = ModuleRecord(name=module_name, path=path, ast_tree=tree)
            self.module_to_path[module_name] = path
            self.path_to_module[path] = module_name
            self.modules[module_name] = record

    def has_module(self, dotted: str) -> bool:
        return dotted in self.module_to_path

    def closest_module(self, dotted: str) -> Optional[str]:
        candidates = [dotted]
        if dotted and not dotted.startswith(f"{self.prefix}."):
            candidates.append(f"{self.prefix}.{dotted}")
        if dotted.startswith(f"{self.prefix}."):
            without_prefix = dotted[len(self.prefix) + 1 :]
            candidates.append(without_prefix)
        if not dotted:
            candidates.append(self.prefix)

        for candidate in candidates:
            if not candidate:
                continue
            parts = [part for part in candidate.split(".") if part]
            while parts:
                name = ".".join(parts)
                canonical = self._canonical(name)
                if canonical in self.module_to_path:
                    return canonical
                parts.pop()
        if self.prefix in self.module_to_path:
            return self.prefix
        return None

    def _canonical(self, name: str) -> str:
        if not name:
            return self.prefix
        if name == self.prefix:
            return name
        if name.startswith(f"{self.prefix}."):
            return name
        return f"{self.prefix}.{name}"


def resolve_import(module: str, node: ast.AST) -> List[str]:
    targets: List[str] = []
    if isinstance(node, ast.Import):
        for alias in node.names:
            if alias.name:
                targets.append(alias.name)
    elif isinstance(node, ast.ImportFrom):
        base_module = node.module or ""
        if node.level:
            base_module = resolve_relative(module, node.level, base_module)
        if not node.names:
            if base_module:
                targets.append(base_module)
        else:
            for alias in node.names:
                if alias.name == "*":
                    if base_module:
                        targets.append(base_module)
                    continue
                if base_module:
                    targets.append(f"{base_module}.{alias.name}")
                else:
                    prefix = resolve_relative(module, node.level, "") if node.level else module
                    target_name = f"{prefix}.{alias.name}" if prefix else alias.name
                    targets.append(target_name)
    return targets


def resolve_relative(module_name: str, level: int, suffix: str) -> str:  # type: ignore[override]
    base_parts = module_name.split(".")
    if level > len(base_parts):
        trimmed: List[str] = []
    else:
        trimmed = base_parts[: len(base_parts) - level]
    if suffix:
        trimmed.append(suffix)
    return ".".join(trimmed)


def normalize_target(target: str) -> str:
    return target.strip(".")


def build_edges(index: ModuleIndex) -> List[Edge]:
    counts: Dict[Tuple[str, str], int] = {}
    for module_name, record in index.modules.items():
        tree = record.ast_tree
        if tree is None:
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                for target in resolve_import(module_name, node):
                    target = normalize_target(target)
                    if not target:
                        continue
                    internal = index.closest_module(target)
                    if not internal:
                        continue
                    if internal == module_name:
                        continue
                    key = (module_name, internal)
                    counts[key] = counts.get(key, 0) + 1
    return [Edge(source=src, target=dst, weight=w) for (src, dst), w in counts.items()]


def write_graphml(nodes: Dict[str, ModuleRecord], edges: List[Edge], output: Path, project_root: Path) -> None:
    from xml.etree.ElementTree import Element, SubElement, tostring

    graphml = Element("graphml", xmlns="http://graphml.graphdrawing.org/xmlns")
    key_path = SubElement(graphml, "key", id="d0", **{"for": "node"}, attr_name="path", attr_type="string")
    key_weight = SubElement(graphml, "key", id="d1", **{"for": "edge"}, attr_name="weight", attr_type="double")
    graph = SubElement(graphml, "graph", edgedefault="directed")

    for name, record in sorted(nodes.items()):
        node_el = SubElement(graph, "node", id=name)
        data_el = SubElement(node_el, "data", key="d0")
        data_el.text = safe_relpath(record.path, project_root)

    for edge in edges:
        edge_el = SubElement(graph, "edge", source=edge.source, target=edge.target)
        data_el = SubElement(edge_el, "data", key="d1")
        data_el.text = str(edge.weight)

    xml_bytes = tostring(graphml, encoding="utf-8", xml_declaration=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(xml_bytes)


class Canvas:
    def __init__(self, width: int, height: int, background: Tuple[int, int, int, int] = (255, 255, 255, 255)) -> None:
        self.width = width
        self.height = height
        self.pixels = bytearray(width * height * 4)
        color_bytes = bytes(background)
        self.pixels[:] = color_bytes * (width * height)

    def set_pixel(self, x: int, y: int, color: Tuple[int, int, int, int]) -> None:
        if 0 <= x < self.width and 0 <= y < self.height:
            idx = (y * self.width + x) * 4
            self.pixels[idx : idx + 4] = bytes(color)

    def fill_rect(self, x: int, y: int, width: int, height: int, color: Tuple[int, int, int, int]) -> None:
        if width <= 0 or height <= 0:
            return
        x0 = max(0, x)
        y0 = max(0, y)
        x1 = min(self.width, x + width)
        y1 = min(self.height, y + height)
        if x0 >= x1 or y0 >= y1:
            return
        span = x1 - x0
        color_bytes = bytes(color)
        row_bytes = color_bytes * span
        for row in range(y0, y1):
            start = (row * self.width + x0) * 4
            end = start + span * 4
            self.pixels[start:end] = row_bytes

    def draw_line(self, x0: int, y0: int, x1: int, y1: int, color: Tuple[int, int, int, int], thickness: int = 1) -> None:
        dx = abs(x1 - x0)
        dy = -abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx + dy
        while True:
            self._draw_point(x0, y0, color, thickness)
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x0 += sx
            if e2 <= dx:
                err += dx
                y0 += sy

    def _draw_point(self, x: int, y: int, color: Tuple[int, int, int, int], radius: int) -> None:
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dx * dx + dy * dy <= radius * radius:
                    self.set_pixel(x + dx, y + dy, color)

    def draw_circle(self, x: int, y: int, radius: int, color: Tuple[int, int, int, int]) -> None:
        self._draw_point(x, y, color, radius)

    def save_png(self, path: Path) -> None:
        width = self.width
        height = self.height
        raw = bytearray()
        for y in range(height):
            raw.append(0)
            start = y * width * 4
            end = start + width * 4
            raw.extend(self.pixels[start:end])
        compressed = zlib.compress(bytes(raw), level=9)
        path.parent.mkdir(parents=True, exist_ok=True)

        def chunk(name: bytes, data: bytes) -> bytes:
            return len(data).to_bytes(4, "big") + name + data + zlib.crc32(name + data).to_bytes(4, "big")

        header = b"\x89PNG\r\n\x1a\n"
        ihdr = chunk(b"IHDR", width.to_bytes(4, "big") + height.to_bytes(4, "big") + b"\x08\x06\x00\x00\x00")
        idat = chunk(b"IDAT", compressed)
        iend = chunk(b"IEND", b"")
        path.write_bytes(header + ihdr + idat + iend)


def layout_positions(n: int, width: int, height: int, margin: int = 40) -> List[Tuple[int, int]]:
    if n == 0:
        return []
    radius = min(width, height) // 2 - margin
    cx = width // 2
    cy = height // 2
    positions: List[Tuple[int, int]] = []
    for i in range(n):
        angle = 2 * math.pi * i / n
        x = int(cx + radius * math.cos(angle))
        y = int(cy + radius * math.sin(angle))
        positions.append((x, y))
    return positions


def render_png(nodes: Dict[str, ModuleRecord], edges: List[Edge], output: Path) -> None:
    count = len(nodes)
    if count == 0:
        Canvas(200, 200).save_png(output)
        return
    width = min(max(600, count * 40), 4000)
    height = width
    canvas = Canvas(width, height)
    ordered_nodes = sorted(nodes.keys())
    positions = {name: pos for name, pos in zip(ordered_nodes, layout_positions(count, width, height))}
    max_weight = max((edge.weight for edge in edges), default=1)
    for edge in edges:
        src_pos = positions.get(edge.source)
        dst_pos = positions.get(edge.target)
        if not src_pos or not dst_pos:
            continue
        thickness = max(1, int(round(3 * edge.weight / max_weight)))
        alpha = max(80, min(255, int(255 * edge.weight / max_weight)))
        color = (30, 30, 30, alpha)
        canvas.draw_line(src_pos[0], src_pos[1], dst_pos[0], dst_pos[1], color, thickness)
    for name, pos in positions.items():
        canvas.draw_circle(pos[0], pos[1], 8, (65, 105, 225, 255))
    canvas.save_png(output)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate dependency graph outputs.")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--source-root", type=Path, default=Path("traigent"))
    parser.add_argument("--graphml", type=Path, required=True)
    parser.add_argument("--png", type=Path, required=True)
    args = parser.parse_args()

    source_root = (args.project_root / args.source_root).resolve()
    index = ModuleIndex(source_root)
    edges = build_edges(index)
    write_graphml(index.modules, edges, args.graphml, args.project_root.resolve())
    render_png(index.modules, edges, args.png)


if __name__ == "__main__":  # pragma: no cover
    main()
