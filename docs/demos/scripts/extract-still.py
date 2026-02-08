#!/usr/bin/env python3
"""Extract a deterministic static frame from an animated svg-term SVG."""

from __future__ import annotations

import argparse
import copy
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

SVG_NS = "http://www.w3.org/2000/svg"
XLINK_NS = "http://www.w3.org/1999/xlink"
NS = {"svg": SVG_NS, "xlink": XLINK_NS}
XLINK_HREF = f"{{{XLINK_NS}}}href"


def _build_symbol_text(root: ET.Element) -> dict[str, str]:
    symbols: dict[str, str] = {}
    for symbol in root.findall(".//svg:symbol", NS):
        symbol_id = symbol.attrib.get("id")
        if not symbol_id:
            continue
        fragments: list[str] = []
        for text_node in symbol.findall("svg:text", NS):
            fragments.append("".join(text_node.itertext()))
        symbols[symbol_id] = " ".join(fragment for fragment in fragments if fragment)
    return symbols


def _iter_frames(root: ET.Element, symbols: dict[str, str]):
    for frame in root.findall(".//svg:svg[@x]", NS):
        x_raw = frame.attrib.get("x")
        if not x_raw or not x_raw.isdigit():
            continue

        lines: list[tuple[float, str]] = []
        for use_node in frame.findall("svg:use", NS):
            href = use_node.attrib.get(XLINK_HREF, "")
            if not href.startswith("#"):
                continue
            symbol_id = href[1:]
            text = symbols.get(symbol_id, "").strip()
            if not text:
                continue
            y_raw = use_node.attrib.get("y", "0")
            try:
                y_val = float(y_raw)
            except ValueError:
                y_val = 0.0
            lines.append((y_val, text))

        if not lines:
            continue

        lines.sort(key=lambda item: item[0])
        top_y = lines[0][0]
        top_line = lines[0][1]
        full_text = "\n".join(text for _, text in lines)
        yield int(x_raw), frame, top_y, top_line, full_text


def _pick_frame(
    root: ET.Element, required: list[str], top_contains: str | None
) -> tuple[int, ET.Element]:
    symbols = _build_symbol_text(root)
    matches: list[tuple[int, ET.Element, float]] = []
    for x_val, frame, top_y, top_line, full_text in _iter_frames(root, symbols):
        if top_contains and top_contains not in top_line:
            continue
        if any(token not in full_text for token in required):
            continue
        matches.append((x_val, frame, top_y))

    if matches:
        for x_val, frame, top_y in matches:
            if top_y == 0.0:
                return x_val, frame
        return matches[0][0], matches[0][1]

    requirement_summary = ", ".join(required) if required else "no required tokens"
    top_summary = top_contains or "none"
    raise ValueError(
        "No frame matched. "
        f"top_contains={top_summary!r}, required=[{requirement_summary}]"
    )


def _find_animated_group(root: ET.Element) -> ET.Element:
    for group in root.findall(".//svg:g", NS):
        style = group.attrib.get("style", "")
        if "animation:" in style or "animation-name" in style or "animation-duration" in style:
            return group
    raise ValueError("Animated frame group with CSS animation was not found")


def extract_still(
    input_path: Path, output_path: Path, required: list[str], top_contains: str | None
) -> int:
    tree = ET.parse(input_path)
    root = tree.getroot()

    selected_x, selected_frame = _pick_frame(root, required=required, top_contains=top_contains)
    animated_group = _find_animated_group(root)

    selected_copy = copy.deepcopy(selected_frame)
    selected_copy.attrib.pop("x", None)
    if selected_copy.attrib.get("width"):
        selected_copy.attrib["x"] = "0"

    animated_group.clear()
    animated_group.append(selected_copy)
    animated_group.attrib.pop("style", None)

    ET.register_namespace("", SVG_NS)
    ET.register_namespace("xlink", XLINK_NS)
    output_path.write_text(ET.tostring(root, encoding="unicode"))
    return selected_x


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Input animated SVG path")
    parser.add_argument("--output", required=True, help="Output still SVG path")
    parser.add_argument(
        "--require",
        action="append",
        default=[],
        help="Substring that must appear in the selected frame (repeatable)",
    )
    parser.add_argument(
        "--top-contains",
        default=None,
        help="Substring that must appear in the top visible line",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"Input file not found: {input_path}", file=sys.stderr)
        return 1

    try:
        selected_x = extract_still(
            input_path=input_path,
            output_path=output_path,
            required=args.require,
            top_contains=args.top_contains,
        )
    except Exception as exc:  # pragma: no cover - script-level error path
        print(f"Failed to extract still from {input_path}: {exc}", file=sys.stderr)
        return 1

    print(f"  {output_path} (frame x={selected_x})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
