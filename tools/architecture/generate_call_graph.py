#!/usr/bin/env python3
"""
Call Graph Generator for TraiGent SDK

Generates function-level call graphs showing which functions call which others.
Useful for understanding code flow, identifying hot paths, and refactoring analysis.

Usage:
    python tools/architecture/generate_call_graph.py [options]
    .venv/bin/python tools/architecture/generate_call_graph.py [options]

Options:
    --output-dir DIR    Output directory (default: tools/architecture/output)
    --package PATH      Package to analyze (default: traigent)
    --module MODULE     Analyze specific module (e.g., traigent.core.orchestrator)
    --depth N           Max call depth to trace (default: 3)
    --min-calls N       Min calls to include function (default: 2)
    --focus FUNC        Focus on calls to/from specific function
    --include-stdlib    Include standard library calls
    --verbose           Show detailed progress

Outputs:
    - call_graph.dot           Full call graph
    - call_graph_focused.dot   Focused view (high-traffic functions)
    - call_graph_data.json     Machine-readable call data
    - hot_paths.md             Report of most-called functions
"""

from __future__ import annotations

import ast
import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ============================================================================
# Data Structures
# ============================================================================


@dataclass
class FunctionInfo:
    """Information about a function definition."""

    name: str
    module: str
    class_name: str | None = None
    lineno: int = 0
    is_async: bool = False
    is_method: bool = False
    parameters: list[str] = field(default_factory=list)
    calls: list[str] = field(default_factory=list)  # Functions this function calls
    called_by: list[str] = field(default_factory=list)  # Functions that call this

    @property
    def fqn(self) -> str:
        """Fully qualified name."""
        if self.class_name:
            return f"{self.module}.{self.class_name}.{self.name}"
        return f"{self.module}.{self.name}"

    @property
    def short_name(self) -> str:
        """Short display name."""
        if self.class_name:
            return f"{self.class_name}.{self.name}"
        return self.name


@dataclass
class CallEdge:
    """A function call relationship."""

    caller: str  # FQN of calling function
    callee: str  # FQN or name of called function
    count: int = 1  # How many times this call appears
    lineno: int = 0


@dataclass
class CallGraphAnalysis:
    """Complete call graph analysis results."""

    functions: dict[str, FunctionInfo] = field(default_factory=dict)
    edges: list[CallEdge] = field(default_factory=list)
    modules_analyzed: int = 0
    total_functions: int = 0
    total_calls: int = 0

    # Computed metrics
    hot_functions: list[tuple[str, int]] = field(
        default_factory=list
    )  # (fqn, call_count)
    entry_points: list[str] = field(
        default_factory=list
    )  # Functions not called by others
    leaf_functions: list[str] = field(
        default_factory=list
    )  # Functions that don't call others


# ============================================================================
# AST Analysis
# ============================================================================


class CallGraphVisitor(ast.NodeVisitor):
    """AST visitor to extract function calls."""

    def __init__(self, module_name: str, root_package: str):
        self.module_name = module_name
        self.root_package = root_package
        self.current_class: str | None = None
        self.current_function: FunctionInfo | None = None
        self.functions: dict[str, FunctionInfo] = {}
        self.calls: list[CallEdge] = []

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Track class context for method analysis."""
        old_class = self.current_class
        self.current_class = node.name
        self.generic_visit(node)
        self.current_class = old_class

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Analyze regular function definition."""
        self._analyze_function(node, is_async=False)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """Analyze async function definition."""
        self._analyze_function(node, is_async=True)

    def _analyze_function(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef, is_async: bool
    ) -> None:
        """Extract function info and its calls."""
        func_info = FunctionInfo(
            name=node.name,
            module=self.module_name,
            class_name=self.current_class,
            lineno=node.lineno,
            is_async=is_async,
            is_method=self.current_class is not None,
            parameters=[arg.arg for arg in node.args.args],
        )

        # Store function
        self.functions[func_info.fqn] = func_info

        # Track current function for call analysis
        old_function = self.current_function
        self.current_function = func_info

        # Visit function body to find calls
        self.generic_visit(node)

        self.current_function = old_function

    def visit_Call(self, node: ast.Call) -> None:
        """Track function calls."""
        if self.current_function is None:
            # Call at module level (not in a function)
            self.generic_visit(node)
            return

        callee_name = self._extract_call_name(node)
        if callee_name:
            self.current_function.calls.append(callee_name)
            self.calls.append(
                CallEdge(
                    caller=self.current_function.fqn,
                    callee=callee_name,
                    lineno=node.lineno,
                )
            )

        self.generic_visit(node)

    def _extract_call_name(self, node: ast.Call) -> str | None:
        """Extract the name of the called function."""
        func = node.func

        if isinstance(func, ast.Name):
            # Simple call: func()
            return func.id

        elif isinstance(func, ast.Attribute):
            # Method call: obj.method() or module.func()
            parts = []
            current = func
            while isinstance(current, ast.Attribute):
                parts.append(current.attr)
                current = current.value
            if isinstance(current, ast.Name):
                parts.append(current.id)
            return ".".join(reversed(parts))

        elif isinstance(func, ast.Subscript):
            # Generic call: Type[T]()
            return self._extract_call_name(
                ast.Call(func=func.value, args=[], keywords=[])
            )

        return None


def analyze_module_calls(
    file_path: Path, module_name: str, root_package: str
) -> tuple[dict[str, FunctionInfo], list[CallEdge]]:
    """Analyze a single module for function calls."""
    try:
        content = file_path.read_text(encoding="utf-8")
        tree = ast.parse(content, filename=str(file_path))

        visitor = CallGraphVisitor(module_name, root_package)
        visitor.visit(tree)

        return visitor.functions, visitor.calls

    except SyntaxError as e:
        print(f"  ⚠️  Syntax error in {file_path}: {e}", file=sys.stderr)
    except Exception as e:
        print(f"  ⚠️  Error analyzing {file_path}: {e}", file=sys.stderr)

    return {}, []


def analyze_package_calls(
    package_path: Path,
    root_package: str,
    verbose: bool = False,
    import_prefix: str = "",
) -> CallGraphAnalysis:
    """Recursively analyze a package for function calls."""
    analysis = CallGraphAnalysis()

    current_prefix = import_prefix or root_package

    def process_directory(dir_path: Path, prefix: str) -> None:
        for item in sorted(dir_path.iterdir()):
            if item.name.startswith("__pycache__"):
                continue

            if item.is_file() and item.suffix == ".py":
                module_name = (
                    f"{prefix}.{item.stem}" if item.stem != "__init__" else prefix
                )

                if verbose:
                    print(f"  Analyzing {module_name}...")

                functions, calls = analyze_module_calls(item, module_name, root_package)
                analysis.functions.update(functions)
                analysis.edges.extend(calls)
                analysis.modules_analyzed += 1

            elif item.is_dir() and (item / "__init__.py").exists():
                subprefix = f"{prefix}.{item.name}"
                process_directory(item, subprefix)

    process_directory(package_path, current_prefix)

    # Compute statistics
    analysis.total_functions = len(analysis.functions)
    analysis.total_calls = len(analysis.edges)

    # Build reverse lookup (called_by)
    call_counts: dict[str, int] = defaultdict(int)
    for edge in analysis.edges:
        call_counts[edge.callee] += 1
        # Update called_by if callee is in our functions
        for fqn, func in analysis.functions.items():
            if (
                edge.callee == func.name
                or edge.callee == func.short_name
                or edge.callee == fqn
            ):
                func.called_by.append(edge.caller)

    # Find hot functions (most called)
    analysis.hot_functions = sorted(
        call_counts.items(), key=lambda x: x[1], reverse=True
    )[:50]

    # Find entry points (functions not called by internal code)
    called_names = {e.callee for e in analysis.edges}
    for fqn, func in analysis.functions.items():
        if (
            func.name not in called_names
            and func.short_name not in called_names
            and fqn not in called_names
        ):
            if not func.name.startswith("_"):  # Skip private functions
                analysis.entry_points.append(fqn)

    # Find leaf functions (don't call other functions)
    for fqn, func in analysis.functions.items():
        if not func.calls:
            analysis.leaf_functions.append(fqn)

    return analysis


# ============================================================================
# Output Generation
# ============================================================================


def generate_call_graph_dot(
    analysis: CallGraphAnalysis,
    min_calls: int = 2,
    include_stdlib: bool = False,
    focus_function: str | None = None,
    max_nodes: int = 100,
) -> str:
    """Generate DOT graph for function calls."""
    lines = [
        "digraph CallGraph {",
        "    rankdir=LR;",
        '    node [shape=box, style=filled, fontname="Helvetica", fontsize=9];',
        "    edge [fontsize=8];",
        "",
        "    // Color scheme:",
        "    // Blue: Entry points (not called by others)",
        "    // Green: Regular functions",
        "    // Orange: Hot functions (called frequently)",
        "    // Red: Very hot functions (>10 calls)",
        "    // Gray: External/stdlib functions",
        "",
    ]

    # Build set of functions to include
    included_functions: set[str] = set()
    included_edges: list[CallEdge] = []

    # Filter edges by call count
    edge_counts: dict[tuple[str, str], int] = defaultdict(int)
    for edge in analysis.edges:
        edge_counts[(edge.caller, edge.callee)] += 1

    for (caller, callee), count in edge_counts.items():
        if count < min_calls:
            continue

        # Skip stdlib unless requested
        if not include_stdlib:
            if callee.split(".")[0] in {
                "os",
                "sys",
                "json",
                "re",
                "time",
                "logging",
                "typing",
                "collections",
                "functools",
                "itertools",
                "pathlib",
                "datetime",
                "asyncio",
                "inspect",
                "abc",
                "dataclasses",
                "enum",
            }:
                continue

        # Focus mode
        if focus_function:
            if focus_function not in caller and focus_function not in callee:
                continue

        included_functions.add(caller)
        included_functions.add(callee)
        included_edges.append(CallEdge(caller=caller, callee=callee, count=count))

    # Limit nodes for readability
    if len(included_functions) > max_nodes:
        # Keep only the most-connected functions
        connection_count: dict[str, int] = defaultdict(int)
        for edge in included_edges:
            connection_count[edge.caller] += 1
            connection_count[edge.callee] += 1

        top_functions = sorted(
            connection_count.items(), key=lambda x: x[1], reverse=True
        )[:max_nodes]
        included_functions = {f for f, _ in top_functions}
        included_edges = [
            e
            for e in included_edges
            if e.caller in included_functions and e.callee in included_functions
        ]

    # Get call counts for coloring
    callee_counts: dict[str, int] = defaultdict(int)
    for edge in included_edges:
        callee_counts[edge.callee] += edge.count

    # Group by module
    modules: dict[str, list[str]] = defaultdict(list)
    for fqn in included_functions:
        parts = fqn.split(".")
        if len(parts) >= 2:
            module = ".".join(parts[:2])  # traigent.core
        else:
            module = "external"
        modules[module].append(fqn)

    # Create subgraphs for modules
    for module_name, funcs in sorted(modules.items()):
        if module_name == "external":
            continue

        module_id = module_name.replace(".", "_")
        lines.append(f"    subgraph cluster_{module_id} {{")
        lines.append(f'        label="{module_name}";')
        lines.append("        style=rounded;")
        lines.append("        color=gray;")

        for fqn in funcs:
            node_id = fqn.replace(".", "_").replace("-", "_")
            func_info = analysis.functions.get(fqn)

            # Determine color
            call_count = callee_counts.get(fqn, 0)
            if call_count > 10:
                color = "#ef9a9a"  # Red - very hot
            elif call_count > 5:
                color = "#ffcc80"  # Orange - hot
            elif fqn in analysis.entry_points:
                color = "#bbdefb"  # Blue - entry point
            else:
                color = "#c8e6c9"  # Green - regular

            # Build label
            if func_info:
                short_name = func_info.short_name
                if func_info.is_async:
                    short_name = f"async {short_name}"
            else:
                short_name = fqn.split(".")[-1]

            label = f"{short_name}"
            if call_count > 0:
                label += f"\\n({call_count} calls)"

            lines.append(f'        {node_id} [label="{label}" fillcolor="{color}"];')

        lines.append("    }")
        lines.append("")

    # Add external functions
    external_funcs = [f for f in included_functions if f not in analysis.functions]
    if external_funcs:
        lines.append("    // External functions")
        for fqn in external_funcs[:20]:  # Limit external
            node_id = fqn.replace(".", "_").replace("-", "_")
            short_name = fqn.split(".")[-1]
            lines.append(
                f'    {node_id} [label="{short_name}" fillcolor="#e0e0e0" style="filled,dashed"];'
            )
        lines.append("")

    # Add edges
    lines.append("    // Call edges")
    for edge in included_edges:
        if (
            edge.caller not in included_functions
            or edge.callee not in included_functions
        ):
            continue
        caller_id = edge.caller.replace(".", "_").replace("-", "_")
        callee_id = edge.callee.replace(".", "_").replace("-", "_")
        if edge.count > 1:
            lines.append(f'    {caller_id} -> {callee_id} [label="{edge.count}"];')
        else:
            lines.append(f"    {caller_id} -> {callee_id};")

    lines.append("}")
    return "\n".join(lines)


def generate_focused_call_graph_dot(
    analysis: CallGraphAnalysis, top_n: int = 30
) -> str:
    """Generate DOT graph focused on most-called functions."""
    # Get top N most-called functions
    top_functions = set(f for f, _ in analysis.hot_functions[:top_n])

    # Also include their direct callers
    for edge in analysis.edges:
        if edge.callee in top_functions:
            # Find the full FQN for the callee
            for fqn in analysis.functions:
                if edge.callee in fqn:
                    top_functions.add(edge.caller)
                    break

    lines = [
        "digraph FocusedCallGraph {",
        "    rankdir=TB;",
        '    node [shape=box, style=filled, fontname="Helvetica", fontsize=10];',
        "    edge [fontsize=8];",
        '    label="Top Called Functions";',
        "    labelloc=t;",
        "    fontsize=14;",
        "",
    ]

    # Collect edges for included functions
    included_edges = []
    for edge in analysis.edges:
        caller_match = any(
            edge.caller.endswith(f) or f in edge.caller for f in top_functions
        )
        callee_match = any(
            edge.callee == f or edge.callee in f
            for f, _ in analysis.hot_functions[:top_n]
        )
        if caller_match or callee_match:
            included_edges.append(edge)

    # Get call counts
    callee_counts: dict[str, int] = defaultdict(int)
    for _, callee in analysis.hot_functions[:top_n]:
        callee_counts[_] = callee

    # Add nodes
    added_nodes: set[str] = set()
    for func_name, count in analysis.hot_functions[:top_n]:
        if func_name in added_nodes:
            continue
        added_nodes.add(func_name)

        node_id = func_name.replace(".", "_").replace("-", "_")

        # Color by call frequency
        if count > 20:
            color = "#d32f2f"  # Dark red
        elif count > 10:
            color = "#ef5350"  # Red
        elif count > 5:
            color = "#ffcc80"  # Orange
        else:
            color = "#fff9c4"  # Yellow

        short_name = func_name.split(".")[-1] if "." in func_name else func_name
        label = f"{short_name}\\n{count} calls"
        lines.append(f'    {node_id} [label="{label}" fillcolor="{color}"];')

    lines.append("")
    lines.append("    // Call edges")

    # Add edges between included nodes
    edge_counts: dict[tuple[str, str], int] = defaultdict(int)
    for edge in analysis.edges:
        if edge.callee in added_nodes:
            edge_counts[(edge.caller, edge.callee)] += 1

    for (caller, callee), count in edge_counts.items():
        if callee not in added_nodes:
            continue
        caller_id = caller.replace(".", "_").replace("-", "_")
        callee_id = callee.replace(".", "_").replace("-", "_")

        # Only show significant call relationships
        if count >= 2:
            lines.append(f'    {caller_id} -> {callee_id} [label="{count}"];')

    lines.append("}")
    return "\n".join(lines)


def generate_hot_paths_report(analysis: CallGraphAnalysis) -> str:
    """Generate markdown report of most-called functions."""
    lines = [
        "# Call Graph Analysis Report",
        "",
        f"**Generated from**: {analysis.modules_analyzed} modules",
        f"**Total Functions**: {analysis.total_functions}",
        f"**Total Call Relationships**: {analysis.total_calls}",
        "",
        "## Most Called Functions (Hot Paths)",
        "",
        "These functions are called most frequently and are critical paths in the codebase.",
        "",
        "| Rank | Function | Calls | Module |",
        "|------|----------|-------|--------|",
    ]

    for i, (func_name, count) in enumerate(analysis.hot_functions[:30], 1):
        parts = func_name.split(".")
        short_name = parts[-1] if parts else func_name
        module = ".".join(parts[:-1]) if len(parts) > 1 else "unknown"
        lines.append(f"| {i} | `{short_name}` | {count} | {module} |")

    # Entry points section
    lines.extend(
        [
            "",
            "## Entry Points",
            "",
            "Functions that are not called by other functions in the codebase (potential API surface):",
            "",
        ]
    )

    entry_by_module: dict[str, list[str]] = defaultdict(list)
    for fqn in analysis.entry_points[:50]:
        parts = fqn.split(".")
        module = ".".join(parts[:2]) if len(parts) >= 2 else "root"
        entry_by_module[module].append(fqn)

    for module, funcs in sorted(entry_by_module.items()):
        lines.append(f"### {module}")
        for fqn in funcs[:10]:
            short_name = fqn.split(".")[-1]
            lines.append(f"- `{short_name}`")
        lines.append("")

    # Leaf functions section
    lines.extend(
        [
            "## Leaf Functions",
            "",
            f"Functions that don't call other functions ({len(analysis.leaf_functions)} total).",
            "These are typically simple utilities or terminal operations.",
            "",
        ]
    )

    leaf_count = min(20, len(analysis.leaf_functions))
    for fqn in analysis.leaf_functions[:leaf_count]:
        short_name = fqn.split(".")[-1]
        lines.append(f"- `{short_name}`")

    # Recommendations
    lines.extend(
        [
            "",
            "## Recommendations",
            "",
            "### High-Traffic Functions to Optimize",
            "",
            "Consider profiling and optimizing these frequently-called functions:",
            "",
        ]
    )

    for func_name, count in analysis.hot_functions[:5]:
        short_name = func_name.split(".")[-1]
        lines.append(
            f"1. **{short_name}** ({count} calls) - High impact on performance"
        )

    lines.extend(
        [
            "",
            "### Potential Refactoring Targets",
            "",
            "Functions with many callers may benefit from interface stabilization:",
            "",
        ]
    )

    for func_name, count in analysis.hot_functions[5:10]:
        short_name = func_name.split(".")[-1]
        lines.append(f"- `{short_name}` ({count} callers)")

    return "\n".join(lines)


def generate_call_graph_json(analysis: CallGraphAnalysis) -> dict[str, Any]:
    """Generate JSON representation of call graph."""
    return {
        "summary": {
            "modules_analyzed": analysis.modules_analyzed,
            "total_functions": analysis.total_functions,
            "total_calls": analysis.total_calls,
            "entry_points_count": len(analysis.entry_points),
            "leaf_functions_count": len(analysis.leaf_functions),
        },
        "hot_functions": [
            {"name": name, "calls": count}
            for name, count in analysis.hot_functions[:50]
        ],
        "entry_points": analysis.entry_points[:50],
        "leaf_functions": analysis.leaf_functions[:50],
        "functions": {
            fqn: {
                "name": func.name,
                "module": func.module,
                "class": func.class_name,
                "is_async": func.is_async,
                "is_method": func.is_method,
                "calls_count": len(func.calls),
                "called_by_count": len(func.called_by),
            }
            for fqn, func in list(analysis.functions.items())[:200]  # Limit for size
        },
        "edges": [
            {"caller": e.caller, "callee": e.callee, "count": e.count}
            for e in analysis.edges[:500]  # Limit for size
        ],
    }


# ============================================================================
# Main
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="Generate function call graphs")
    parser.add_argument(
        "--output-dir", default="tools/architecture/output", help="Output directory"
    )
    parser.add_argument("--package", default="traigent", help="Package to analyze")
    parser.add_argument("--module", help="Analyze specific module only")
    parser.add_argument("--depth", type=int, default=3, help="Max call depth")
    parser.add_argument("--min-calls", type=int, default=2, help="Min calls to include")
    parser.add_argument("--focus", help="Focus on specific function")
    parser.add_argument(
        "--include-stdlib", action="store_true", help="Include stdlib calls"
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    # Find project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    package_path = project_root / args.package

    if not package_path.exists():
        print(f"❌ Package not found: {package_path}", file=sys.stderr)
        sys.exit(1)

    print(f"🔍 Analyzing call graph for {args.package}...")

    # Analyze
    analysis = analyze_package_calls(
        package_path,
        args.package,
        verbose=args.verbose,
    )

    print(f"\n📊 Analysis complete:")
    print(f"   Modules analyzed: {analysis.modules_analyzed}")
    print(f"   Functions found: {analysis.total_functions}")
    print(f"   Call relationships: {analysis.total_calls}")
    print(f"   Entry points: {len(analysis.entry_points)}")
    print(f"   Leaf functions: {len(analysis.leaf_functions)}")

    # Generate outputs
    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n📁 Writing outputs to {output_dir}/")

    # Full call graph
    output_file = output_dir / "call_graph.dot"
    dot = generate_call_graph_dot(
        analysis,
        min_calls=args.min_calls,
        include_stdlib=args.include_stdlib,
        focus_function=args.focus,
    )
    output_file.write_text(dot)
    print(f"   ✅ {output_file.name}")

    # Focused call graph
    output_file = output_dir / "call_graph_focused.dot"
    dot = generate_focused_call_graph_dot(analysis, top_n=30)
    output_file.write_text(dot)
    print(f"   ✅ {output_file.name}")

    # JSON data
    output_file = output_dir / "call_graph_data.json"
    data = generate_call_graph_json(analysis)
    output_file.write_text(json.dumps(data, indent=2))
    print(f"   ✅ {output_file.name}")

    # Hot paths report
    output_file = output_dir / "hot_paths.md"
    report = generate_hot_paths_report(analysis)
    output_file.write_text(report)
    print(f"   ✅ {output_file.name}")

    # Top hot functions
    print("\n🔥 Top 10 Most Called Functions:")
    for i, (func_name, count) in enumerate(analysis.hot_functions[:10], 1):
        short_name = func_name.split(".")[-1] if "." in func_name else func_name
        print(f"   {i}. {short_name}: {count} calls")

    print("\n💡 To render DOT files as images:")
    print(f"   dot -Tsvg {output_dir}/call_graph.dot -o {output_dir}/call_graph.svg")
    print(
        f"   dot -Tsvg {output_dir}/call_graph_focused.dot -o {output_dir}/call_graph_focused.svg"
    )

    print("\n✨ Done!")


if __name__ == "__main__":
    main()
