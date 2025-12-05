#!/usr/bin/env python3
"""
Generate focused, digestible architecture diagrams.

Creates:
1. Top 20 high-complexity modules (easy to read)
2. Per-package complexity views
3. Core class hierarchy only (most important classes)
4. Summary statistics charts
"""

from __future__ import annotations

import json
from pathlib import Path


def generate_complexity_top20_dot(data: dict) -> str:
    """Generate DOT for top 20 most complex modules only."""
    lines = [
        "digraph ComplexityTop20 {",
        "    rankdir=LR;",
        '    node [shape=box, style=filled, fontname="Helvetica", fontsize=11];',
        '    label="Top 20 High-Complexity Modules\\n(Max Cyclomatic Complexity)";',
        "    labelloc=t;",
        "    fontsize=16;",
        "",
    ]

    high_complexity = data.get("complexity", {}).get("high_complexity_modules", [])[:20]

    for i, (module, avg_cc, max_cc) in enumerate(high_complexity):
        # Color based on max complexity
        if max_cc >= 50:
            color = "#d32f2f"  # Dark red
        elif max_cc >= 30:
            color = "#ef5350"  # Red
        elif max_cc >= 20:
            color = "#ffcc80"  # Orange
        else:
            color = "#fff9c4"  # Yellow

        # Extract module name
        parts = module.split("/")
        pkg = parts[1] if len(parts) > 1 else "root"
        name = parts[-1].replace(".py", "")

        node_id = f"mod{i}"
        label = f"{pkg}/{name}\\nMax CC: {max_cc}\\nAvg CC: {avg_cc:.1f}"
        lines.append(f'    {node_id} [label="{label}" fillcolor="{color}"];')

    # Add legend
    lines.extend(
        [
            "",
            "    subgraph cluster_legend {",
            '        label="Complexity Scale";',
            "        style=dashed;",
            "        rankdir=TB;",
            '        leg1 [label="Critical (50+)" fillcolor="#d32f2f"];',
            '        leg2 [label="Very High (30-50)" fillcolor="#ef5350"];',
            '        leg3 [label="High (20-30)" fillcolor="#ffcc80"];',
            '        leg4 [label="Moderate (15-20)" fillcolor="#fff9c4"];',
            "    }",
        ]
    )

    lines.append("}")
    return "\n".join(lines)


def generate_package_summary_dot(data: dict) -> str:
    """Generate DOT showing package-level metrics summary."""
    lines = [
        "digraph PackageSummary {",
        "    rankdir=TB;",
        '    node [shape=box, style=filled, fontname="Helvetica", fontsize=10];',
        '    label="TraiGent Package Summary\\n(Files / Lines / Classes)";',
        "    labelloc=t;",
        "    fontsize=16;",
        "",
    ]

    # Extract packages from nested structure
    packages = {}
    pkg_struct = data.get("package_structure", {})
    for pkg_name, pkg_data in pkg_struct.get("subpackages", {}).items():
        packages[pkg_name] = {
            "files": pkg_data.get("total_files", 0),
            "lines": pkg_data.get("total_lines", 0),
            "classes": pkg_data.get("total_classes", 0),
        }

    # Sort by lines of code
    sorted_pkgs = sorted(
        packages.items(), key=lambda x: x[1].get("lines", 0), reverse=True
    )

    for pkg_name, pkg_data in sorted_pkgs[:15]:  # Top 15 packages
        files = pkg_data.get("files", 0)
        lines_count = pkg_data.get("lines", 0)
        classes = pkg_data.get("classes", 0)

        # Color based on size
        if lines_count > 10000:
            color = "#ef9a9a"  # Large - red
        elif lines_count > 5000:
            color = "#ffcc80"  # Medium - orange
        elif lines_count > 2000:
            color = "#fff9c4"  # Small - yellow
        else:
            color = "#c8e6c9"  # Tiny - green

        node_id = pkg_name.replace("/", "_").replace(".", "_")
        label = f"{pkg_name}\\n{files} files | {lines_count:,}L | {classes}C"
        lines.append(f'    {node_id} [label="{label}" fillcolor="{color}"];')

    lines.append("}")
    return "\n".join(lines)


def generate_hub_modules_dot(data: dict) -> str:
    """Generate DOT for hub modules (high fan-in)."""
    lines = [
        "digraph HubModules {",
        "    rankdir=LR;",
        '    node [shape=box, style=filled, fontname="Helvetica", fontsize=11];',
        '    label="Hub Modules (High Fan-In)\\nModules imported by many others";',
        "    labelloc=t;",
        "    fontsize=16;",
        "",
    ]

    hub_modules = data.get("hub_modules", [])[:15]

    for i, item in enumerate(hub_modules):
        module = item.get("module", "") if isinstance(item, dict) else item[0]
        fan_in = item.get("fan_in", 0) if isinstance(item, dict) else item[1]
        # Color based on fan-in
        if fan_in >= 50:
            color = "#d32f2f"  # Critical - dark red
        elif fan_in >= 20:
            color = "#ef5350"  # High - red
        elif fan_in >= 10:
            color = "#ffcc80"  # Medium - orange
        else:
            color = "#fff9c4"  # Low - yellow

        # Shorten module name
        short_name = module.replace("traigent.", "")

        node_id = f"hub{i}"
        label = f"{short_name}\\n{fan_in} dependents"
        lines.append(f'    {node_id} [label="{label}" fillcolor="{color}"];')

    # Add legend
    lines.extend(
        [
            "",
            "    subgraph cluster_legend {",
            '        label="Stability Risk";',
            "        style=dashed;",
            '        leg1 [label="Critical (50+)" fillcolor="#d32f2f"];',
            '        leg2 [label="High (20-50)" fillcolor="#ef5350"];',
            '        leg3 [label="Medium (10-20)" fillcolor="#ffcc80"];',
            '        leg4 [label="Low (5-10)" fillcolor="#fff9c4"];',
            "    }",
        ]
    )

    lines.append("}")
    return "\n".join(lines)


def generate_class_top15_dot(data: dict) -> str:
    """Generate DOT for top 15 classes by method count."""
    lines = [
        "digraph TopClasses {",
        "    rankdir=LR;",
        '    node [shape=box, style=filled, fontname="Helvetica", fontsize=11];',
        '    label="Top 15 Largest Classes\\n(by method count)";',
        "    labelloc=t;",
        "    fontsize=16;",
        "",
    ]

    # Extract from JSON - would need class data
    # For now, use hardcoded from report
    top_classes = [
        ("AuthManager", "cloud.auth", 65),
        ("OptimizationOrchestrator", "core.orchestrator", 61),
        ("BackendIntegratedClient", "cloud.backend_client", 55),
        ("CostOptimizationAI", "analytics.intelligence", 54),
        ("OptimizedFunction", "core.optimized_function", 50),
        ("TraiGentCloudClient", "cloud.client", 46),
        ("RemoteServiceRegistry", "optimizers.service_registry", 40),
        ("OptimizationResult", "api.types", 28),
        ("SpecificationGenerator", "agents.specification_generator", 26),
        ("SDKBackendBridge", "cloud.backend_bridges", 25),
        ("DatasetConverter", "cloud.dataset_converter", 25),
        ("ProductionMCPClient", "cloud.production_mcp_client", 25),
        ("ApiOperations", "cloud.api_operations", 24),
        ("BaseOverrideManager", "integrations.base", 23),
        ("FrameworkOverride", "integrations.framework_override", 23),
    ]

    for i, (name, pkg, methods) in enumerate(top_classes):
        # Color based on method count (complexity indicator)
        if methods >= 50:
            color = "#ef9a9a"  # Very large
        elif methods >= 30:
            color = "#ffcc80"  # Large
        elif methods >= 20:
            color = "#fff9c4"  # Medium
        else:
            color = "#c8e6c9"  # Small

        node_id = f"cls{i}"
        label = f"{name}\\n{pkg}\\n{methods} methods"
        lines.append(f'    {node_id} [label="{label}" fillcolor="{color}"];')

    lines.append("}")
    return "\n".join(lines)


def main():
    output_dir = Path(__file__).parent / "output"
    data_file = output_dir / "architecture_data.json"

    if not data_file.exists():
        print(f"❌ Data file not found: {data_file}")
        print("   Run generate_diagrams.py --all-features first")
        return

    print("📊 Loading architecture data...")
    data = json.loads(data_file.read_text())

    # Create focused output directory
    focused_dir = output_dir / "focused"
    focused_dir.mkdir(exist_ok=True)

    print(f"\n📁 Generating focused diagrams in {focused_dir}/")

    # Generate focused DOT files
    diagrams = [
        ("complexity_top20.dot", generate_complexity_top20_dot(data)),
        ("package_summary.dot", generate_package_summary_dot(data)),
        ("hub_modules.dot", generate_hub_modules_dot(data)),
        ("top_classes.dot", generate_class_top15_dot(data)),
    ]

    for filename, dot_content in diagrams:
        dot_file = focused_dir / filename
        dot_file.write_text(dot_content)
        print(f"   ✅ {filename}")

        # Render to SVG
        svg_file = focused_dir / filename.replace(".dot", ".svg")
        import subprocess

        result = subprocess.run(
            ["dot", "-Tsvg", str(dot_file), "-o", str(svg_file)],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            print(f"   ✅ {svg_file.name}")
        else:
            print(f"   ❌ Failed to render {svg_file.name}: {result.stderr}")

    print("\n✨ Done! Open the SVG files in your browser.")


if __name__ == "__main__":
    main()
