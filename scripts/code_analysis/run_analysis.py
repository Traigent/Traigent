"""Entrypoint orchestrating repository code analysis."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

try:  # pragma: no cover - support execution via python path/to/script.py
    from .analysis_utils import load_coverage_map, load_lint_map, write_csv
    from .dependency_graph import ModuleIndex, build_edges, render_png, write_graphml
    from .inventory import gather_inventory, load_codeowners, run_coverage, run_lint
    from .metrics import compute_metrics
except ImportError:  # pragma: no cover
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from scripts.code_analysis.analysis_utils import load_coverage_map, load_lint_map, write_csv
    from scripts.code_analysis.dependency_graph import (
        ModuleIndex,
        build_edges,
        render_png,
        write_graphml,
    )
    from scripts.code_analysis.inventory import (
        gather_inventory,
        load_codeowners,
        run_coverage,
        run_lint,
    )
    from scripts.code_analysis.metrics import compute_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run codebase inventory and metrics collection.")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--source-root", type=Path, default=Path("traigent"))
    parser.add_argument("--tests-root", type=Path, default=Path("tests"))
    parser.add_argument("--output-root", type=Path, default=Path("reports/1_quality/analysis"))
    parser.add_argument("--tag", type=str, help="Optional output directory tag")
    parser.add_argument("--skip-coverage", action="store_true")
    parser.add_argument("--skip-lint", action="store_true")
    parser.add_argument("--skip-owners", action="store_true")
    return parser.parse_args()


def ensure_analysis_outputs(args: argparse.Namespace) -> None:
    project_root = args.project_root.resolve()
    source_root = (args.project_root / args.source_root).resolve()
    tests_root = (args.project_root / args.tests_root).resolve()

    tag = args.tag or datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    output_dir = (args.project_root / args.output_root / tag).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    coverage_xml = output_dir / "coverage.xml"
    lint_json = output_dir / "lint.json"
    inventory_csv = output_dir / "inventory.csv"
    metrics_csv = output_dir / "metrics.csv"
    graphml_path = output_dir / "deps.graphml"
    png_path = output_dir / "deps.png"

    if not args.skip_coverage:
        run_coverage(project_root, coverage_xml)
    if not args.skip_lint:
        run_lint(project_root, lint_json)

    coverage_map = load_coverage_map(coverage_xml, project_root)
    lint_map = load_lint_map(lint_json, project_root)

    codeowners = [] if args.skip_owners else load_codeowners(project_root)

    rows = list(
        gather_inventory(
            project_root,
            source_root,
            coverage_map,
            lint_map,
            codeowners,
        )
    )
    header = [
        "module",
        "path",
        "language",
        "sloc",
        "file_size_bytes",
        "last_modified_iso",
        "owners",
        "test_coverage_percent",
        "lint_error_count",
    ]
    write_csv(inventory_csv, header, rows)

    compute_metrics(project_root, source_root, tests_root, metrics_csv)

    index = ModuleIndex(source_root)
    edges = build_edges(index)
    write_graphml(index.modules, edges, graphml_path, project_root)
    render_png(index.modules, edges, png_path)

    summary_path = output_dir / "SUMMARY.txt"
    summary_lines = [
        "Code analysis completed.",
        f"Inventory: {inventory_csv}",
        f"Metrics: {metrics_csv}",
        f"GraphML: {graphml_path}",
        f"PNG: {png_path}",
    ]
    summary_path.write_text("\n".join(str(line) for line in summary_lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    ensure_analysis_outputs(args)


if __name__ == "__main__":  # pragma: no cover
    main()
