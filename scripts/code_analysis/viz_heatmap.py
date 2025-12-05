"""Visualize risk scores as a heatmap-style table image."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

try:  # pragma: no cover
    from .dependency_graph import Canvas
    from .viz_atlas import draw_text
except ImportError:  # pragma: no cover
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from scripts.code_analysis.dependency_graph import Canvas
    from scripts.code_analysis.viz_atlas import draw_text


HEADER_BG = (240, 248, 255, 255)
ROW_BG = (255, 255, 255, 255)
ALT_ROW_BG = (245, 245, 245, 255)
BORDER_COLOR = (200, 200, 200, 255)
HEAT_LOW = (99, 190, 123, 255)
HEAT_HIGH = (205, 92, 92, 255)
TEXT_COLOR = (30, 30, 30, 255)

FONT_SCALE = 1
ROW_HEIGHT = 28
HEADER_HEIGHT = 36
COLUMN_WIDTHS = [320, 110, 110, 110, 110, 90, 90, 90]
COLUMNS = [
    ("module", "Module"),
    ("risk", "Risk"),
    ("complexity", "Complexity"),
    ("fan_in", "Fan-In"),
    ("fan_out", "Fan-Out"),
    ("violations", "Viol"),
    ("clones", "Clones"),
    ("churn", "Churn"),
]


def lerp_color(value: float) -> Tuple[int, int, int, int]:
    value = min(max(value, 0.0), 1.0)
    r = int(HEAT_LOW[0] + (HEAT_HIGH[0] - HEAT_LOW[0]) * value)
    g = int(HEAT_LOW[1] + (HEAT_HIGH[1] - HEAT_LOW[1]) * value)
    b = int(HEAT_LOW[2] + (HEAT_HIGH[2] - HEAT_LOW[2]) * value)
    a = 255
    return (r, g, b, a)


def parse_csv(path: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(row)
    return rows


def compute_heat_values(rows: Sequence[Dict[str, str]]) -> Tuple[Dict[str, float], float, float]:
    risks = [float(row.get("risk", 0.0) or 0.0) for row in rows]
    if risks:
        min_risk = min(risks)
        max_risk = max(risks)
    else:
        min_risk = 0.0
        max_risk = 0.0
    range_risk = max(max_risk - min_risk, 1e-6)
    normalized: Dict[str, float] = {}
    for row in rows:
        risk = float(row.get("risk", 0.0) or 0.0)
        normalized[row.get("module", "")] = (risk - min_risk) / range_risk
    return normalized, min_risk, max_risk


def draw_table(rows: Sequence[Dict[str, str]], output: Path, rows_per_image: int = 150) -> None:
    if not rows:
        Canvas(800, 200).save_png(output)
        return
    normalized, min_risk, max_risk = compute_heat_values(rows)
    columns = COLUMN_WIDTHS
    total_width = sum(columns) + 80
    pages = [rows[i : i + rows_per_image] for i in range(0, len(rows), rows_per_image)]
    for index, page in enumerate(pages, start=1):
        height = HEADER_HEIGHT + len(page) * ROW_HEIGHT + 60
        canvas = Canvas(total_width, height)
        canvas.fill_rect(0, 0, total_width, HEADER_HEIGHT, HEADER_BG)
        canvas.fill_rect(0, HEADER_HEIGHT, total_width, 1, BORDER_COLOR)
        # Column headers
        x_offset = 40
        for (key, title), col_width in zip(COLUMNS, columns):
            draw_text(canvas, x_offset, 8, title, color=TEXT_COLOR, scale=FONT_SCALE)
            x_offset += col_width
        # Rows
        for row_idx, row in enumerate(page):
            y_top = HEADER_HEIGHT + row_idx * ROW_HEIGHT + 2
            bg_color = ROW_BG if row_idx % 2 == 0 else ALT_ROW_BG
            canvas.fill_rect(0, y_top, total_width, ROW_HEIGHT, bg_color)
            x_offset = 40
            for (key, _title), col_width in zip(COLUMNS, columns):
                value = row.get(key, "")
                if key == "risk":
                    module = row.get("module", "")
                    color = lerp_color(normalized.get(module, 0.0))
                    canvas.fill_rect(x_offset, y_top, col_width, ROW_HEIGHT, color)
                    draw_text(
                        canvas,
                        x_offset + 8,
                        y_top + 8,
                        f"{float(value):.3f}" if value else "",
                        color=TEXT_COLOR,
                        scale=FONT_SCALE,
                    )
                else:
                    draw_text(canvas, x_offset + 4, y_top + 8, str(value)[:20], color=TEXT_COLOR, scale=FONT_SCALE)
                x_offset += col_width
        footer_y = HEADER_HEIGHT + len(page) * ROW_HEIGHT + 20
        draw_text(canvas, 40, footer_y, f"Risk range: {min_risk:.3f}−{max_risk:.3f}")
        if len(pages) > 1:
            draw_text(canvas, total_width - 160, footer_y, f"Page {index}/{len(pages)}")
        filename = output if len(pages) == 1 else output.with_name(f"{output.stem}_p{index}.png")
        canvas.save_png(filename)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render risk heatmap PNG")
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rows-per-page", type=int, default=150)
    args = parser.parse_args()
    rows = parse_csv(args.csv)
    if not rows:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        Canvas(800, 200).save_png(args.out)
        return
    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows_per_image = max(25, min(args.rows_per_page, 200))
    draw_table(rows, args.out, rows_per_image)


if __name__ == "__main__":  # pragma: no cover
    main()
