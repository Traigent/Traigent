#!/usr/bin/env python3
"""
Architecture Diagram Generator for Traigent SDK

Generates package structure, module dependency graphs, class hierarchies,
complexity analysis, and test coverage maps.

Usage:
    python tools/architecture/generate_diagrams.py [options]
    .venv/bin/python tools/architecture/generate_diagrams.py [options]

Options:
    --output-dir DIR    Output directory (default: tools/architecture/output)
    --format FORMAT     Output format: text, dot, json, all (default: all)
    --package PATH      Package to analyze (default: traigent)
    --include-tests     Include tests/ in analysis
    --verbose           Show detailed progress
    --complexity        Run complexity analysis (requires radon)
    --coverage FILE     Path to coverage JSON file (from pytest --cov-report=json)
    --class-hierarchy   Generate class hierarchy diagrams

Outputs:
    - package_structure.txt      Tree view with metrics
    - module_dependencies.dot    Graphviz dependency graph
    - class_hierarchy.dot        Class inheritance diagram
    - complexity_heatmap.dot     Modules colored by complexity
    - coverage_map.dot           Modules colored by test coverage
    - architecture_data.json     Machine-readable data
    - architecture_report.md     Gap analysis and recommendations
"""

from __future__ import annotations

import ast
import argparse
import json
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any


# ============================================================================
# Data Structures
# ============================================================================


@dataclass
class ClassInfo:
    """Information about a class definition."""

    name: str
    bases: List[str] = field(default_factory=list)
    methods: int = 0
    is_abstract: bool = False
    docstring: Optional[str] = None
    module_path: str = ""


@dataclass
class ComplexityMetrics:
    """Complexity metrics for a module."""

    cyclomatic_complexity: float = 0.0  # Average CC
    max_complexity: int = 0  # Highest single function CC
    maintainability_index: float = 100.0  # MI score (0-100)
    halstead_volume: float = 0.0
    functions_analyzed: int = 0
    high_complexity_functions: List[Tuple[str, int]] = field(
        default_factory=list
    )  # (name, cc)


@dataclass
class ModuleMetrics:
    """Metrics for a single Python module."""

    path: Path
    lines: int = 0
    classes: int = 0
    functions: int = 0
    imports: List[str] = field(default_factory=list)
    internal_imports: List[str] = field(default_factory=list)
    external_imports: List[str] = field(default_factory=list)
    docstring: Optional[str] = None
    # New fields for enhanced analysis
    class_info: List[ClassInfo] = field(default_factory=list)
    complexity: Optional[ComplexityMetrics] = None
    coverage_percent: Optional[float] = None

    @property
    def has_docstring(self) -> bool:
        return self.docstring is not None and len(self.docstring.strip()) > 0

    @property
    def avg_complexity(self) -> float:
        if self.complexity:
            return self.complexity.cyclomatic_complexity
        return 0.0


@dataclass
class PackageMetrics:
    """Aggregated metrics for a package/directory."""

    path: Path
    name: str
    modules: Dict[str, ModuleMetrics] = field(default_factory=dict)
    subpackages: Dict[str, "PackageMetrics"] = field(default_factory=dict)

    @property
    def total_files(self) -> int:
        count = len(self.modules)
        for sub in self.subpackages.values():
            count += sub.total_files
        return count

    @property
    def total_lines(self) -> int:
        lines = sum(m.lines for m in self.modules.values())
        for sub in self.subpackages.values():
            lines += sub.total_lines
        return lines

    @property
    def total_classes(self) -> int:
        count = sum(m.classes for m in self.modules.values())
        for sub in self.subpackages.values():
            count += sub.total_classes
        return count

    @property
    def total_functions(self) -> int:
        count = sum(m.functions for m in self.modules.values())
        for sub in self.subpackages.values():
            count += sub.total_functions
        return count


@dataclass
class DependencyEdge:
    """A dependency between two modules."""

    source: str  # Module that imports
    target: str  # Module being imported
    import_type: str  # "direct" or "from"


@dataclass
class ArchitectureAnalysis:
    """Complete architecture analysis results."""

    root_package: PackageMetrics
    all_modules: Dict[str, ModuleMetrics] = field(default_factory=dict)
    dependencies: List[DependencyEdge] = field(default_factory=list)
    circular_dependencies: List[List[str]] = field(default_factory=list)
    orphaned_modules: List[str] = field(default_factory=list)
    hub_modules: List[Tuple[str, int]] = field(default_factory=list)  # (module, fan_in)
    layering_violations: List[Tuple[str, str, str]] = field(default_factory=list)
    # New fields for enhanced analysis
    all_classes: Dict[str, ClassInfo] = field(default_factory=dict)  # FQN -> ClassInfo
    inheritance_edges: List[Tuple[str, str]] = field(
        default_factory=list
    )  # (child, parent)
    complexity_stats: Dict[str, Any] = field(default_factory=dict)
    coverage_stats: Dict[str, float] = field(default_factory=dict)


# ============================================================================
# AST Analysis
# ============================================================================


class ModuleAnalyzer(ast.NodeVisitor):
    """AST visitor to extract module metrics including class hierarchy."""

    def __init__(self, root_package: str, module_path: str = ""):
        self.root_package = root_package
        self.module_path = module_path
        self.classes = 0
        self.functions = 0
        self.imports: List[str] = []
        self.internal_imports: List[str] = []
        self.external_imports: List[str] = []
        self.docstring: Optional[str] = None
        self.class_info: List[ClassInfo] = []

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.classes += 1

        # Extract base classes
        bases = []
        for base in node.bases:
            base_name = self._get_base_name(base)
            if base_name:
                bases.append(base_name)

        # Count methods
        methods = sum(
            1
            for item in node.body
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
        )

        # Check if abstract (has ABC in bases or abstractmethod decorators)
        is_abstract = any("ABC" in b or "Abstract" in b for b in bases)
        if not is_abstract:
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    for decorator in item.decorator_list:
                        if (
                            isinstance(decorator, ast.Name)
                            and decorator.id == "abstractmethod"
                        ):
                            is_abstract = True
                            break

        # Extract docstring
        class_docstring = None
        if (
            node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
            and isinstance(node.body[0].value.value, str)
        ):
            class_docstring = node.body[0].value.value

        info = ClassInfo(
            name=node.name,
            bases=bases,
            methods=methods,
            is_abstract=is_abstract,
            docstring=class_docstring,
            module_path=self.module_path,
        )
        self.class_info.append(info)

        self.generic_visit(node)

    def _get_base_name(self, node: ast.expr) -> Optional[str]:
        """Extract base class name from AST node."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            # Handle module.Class
            parts = []
            current = node
            while isinstance(current, ast.Attribute):
                parts.append(current.attr)
                current = current.value
            if isinstance(current, ast.Name):
                parts.append(current.id)
            return ".".join(reversed(parts))
        elif isinstance(node, ast.Subscript):
            # Handle Generic[T] style
            return self._get_base_name(node.value)
        return None

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.functions += 1
        # Don't count nested functions

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.functions += 1

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self._classify_import(alias.name)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module:
            self._classify_import(node.module)

    def _classify_import(self, module_name: str) -> None:
        self.imports.append(module_name)
        if module_name.startswith(self.root_package):
            self.internal_imports.append(module_name)
        else:
            self.external_imports.append(module_name)

    def visit_Module(self, node: ast.Module) -> None:
        # Extract module docstring
        if (
            node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
            and isinstance(node.body[0].value.value, str)
        ):
            self.docstring = node.body[0].value.value
        self.generic_visit(node)


def analyze_module(
    file_path: Path, root_package: str, module_import_path: str = ""
) -> ModuleMetrics:
    """Analyze a single Python module."""
    metrics = ModuleMetrics(path=file_path)

    try:
        content = file_path.read_text(encoding="utf-8")
        metrics.lines = len(content.splitlines())

        tree = ast.parse(content, filename=str(file_path))
        analyzer = ModuleAnalyzer(root_package, module_import_path)
        analyzer.visit(tree)

        metrics.classes = analyzer.classes
        metrics.functions = analyzer.functions
        metrics.imports = analyzer.imports
        metrics.internal_imports = analyzer.internal_imports
        metrics.external_imports = analyzer.external_imports
        metrics.docstring = analyzer.docstring
        metrics.class_info = analyzer.class_info

    except SyntaxError as e:
        print(f"  ⚠️  Syntax error in {file_path}: {e}", file=sys.stderr)
    except Exception as e:
        print(f"  ⚠️  Error analyzing {file_path}: {e}", file=sys.stderr)

    return metrics


# ============================================================================
# Complexity Analysis (using radon)
# ============================================================================


def analyze_complexity(
    file_path: Path, venv_python: Optional[Path] = None
) -> Optional[ComplexityMetrics]:
    """Analyze cyclomatic complexity using radon."""
    try:
        # Try to import radon
        from radon.complexity import cc_visit
        from radon.metrics import mi_visit

        content = file_path.read_text(encoding="utf-8")

        # Cyclomatic complexity
        cc_results = cc_visit(content)
        if not cc_results:
            return ComplexityMetrics()

        complexities = [r.complexity for r in cc_results]
        avg_cc = sum(complexities) / len(complexities) if complexities else 0
        max_cc = max(complexities) if complexities else 0

        # High complexity functions (CC > 10)
        high_cc = [(r.name, r.complexity) for r in cc_results if r.complexity > 10]

        # Maintainability Index
        mi_score = mi_visit(content, True)  # True = multi=True for better accuracy

        return ComplexityMetrics(
            cyclomatic_complexity=round(avg_cc, 2),
            max_complexity=max_cc,
            maintainability_index=round(mi_score, 2),
            functions_analyzed=len(cc_results),
            high_complexity_functions=high_cc,
        )

    except ImportError:
        return None  # radon not available
    except Exception as e:
        print(f"  ⚠️  Complexity analysis failed for {file_path}: {e}", file=sys.stderr)
        return None


def run_complexity_analysis(
    analysis: ArchitectureAnalysis, project_root: Path, verbose: bool = False
) -> None:
    """Run complexity analysis on all modules."""
    print("📊 Running complexity analysis (radon)...")

    total_cc = 0.0
    analyzed = 0
    high_complexity_modules: List[Tuple[str, float, int]] = []  # (path, avg_cc, max_cc)

    for module_path, metrics in analysis.all_modules.items():
        file_path = project_root / module_path.replace("/", os.sep)
        if file_path.exists():
            complexity = analyze_complexity(file_path)
            if complexity:
                metrics.complexity = complexity
                total_cc += complexity.cyclomatic_complexity
                analyzed += 1

                if complexity.max_complexity > 15:
                    high_complexity_modules.append(
                        (
                            module_path,
                            complexity.cyclomatic_complexity,
                            complexity.max_complexity,
                        )
                    )

                if verbose and complexity.high_complexity_functions:
                    print(f"  ⚠️  High complexity in {module_path}:")
                    for func, cc in complexity.high_complexity_functions[:3]:
                        print(f"      {func}: CC={cc}")

    # Store stats
    analysis.complexity_stats = {
        "modules_analyzed": analyzed,
        "average_complexity": round(total_cc / analyzed, 2) if analyzed else 0,
        "high_complexity_modules": sorted(
            high_complexity_modules, key=lambda x: x[2], reverse=True
        )[:20],
    }

    print(f"   Analyzed {analyzed} modules")
    print(f"   Average complexity: {analysis.complexity_stats['average_complexity']}")
    print(f"   High complexity modules: {len(high_complexity_modules)}")


# ============================================================================
# Test Coverage Analysis
# ============================================================================


def load_coverage_data(coverage_file: Path) -> Dict[str, float]:
    """Load coverage data from pytest-cov JSON report."""
    coverage_map: Dict[str, float] = {}

    try:
        data = json.loads(coverage_file.read_text())

        # Handle different coverage.json formats
        if "files" in data:
            # Standard coverage.py JSON format
            for file_path, file_data in data["files"].items():
                if "summary" in file_data:
                    coverage_map[file_path] = file_data["summary"].get(
                        "percent_covered", 0
                    )
                elif "executed_lines" in file_data and "missing_lines" in file_data:
                    executed = len(file_data["executed_lines"])
                    missing = len(file_data["missing_lines"])
                    total = executed + missing
                    coverage_map[file_path] = (
                        (executed / total * 100) if total > 0 else 0
                    )

    except Exception as e:
        print(f"  ⚠️  Error loading coverage data: {e}", file=sys.stderr)

    return coverage_map


def apply_coverage_data(
    analysis: ArchitectureAnalysis, coverage_map: Dict[str, float], package_name: str
) -> None:
    """Apply coverage data to module metrics."""
    matched = 0

    for module_path, metrics in analysis.all_modules.items():
        # Try to match coverage paths
        # Coverage paths might be absolute or relative
        for cov_path, percent in coverage_map.items():
            if module_path in cov_path or cov_path.endswith(
                module_path.replace("/", os.sep)
            ):
                metrics.coverage_percent = percent
                matched += 1
                break

    # Calculate package-level coverage
    total_covered = 0.0
    total_modules = 0
    for metrics in analysis.all_modules.values():
        if metrics.coverage_percent is not None:
            total_covered += metrics.coverage_percent
            total_modules += 1

    analysis.coverage_stats = {
        "modules_with_coverage": matched,
        "average_coverage": (
            round(total_covered / total_modules, 2) if total_modules else 0
        ),
        "total_files_in_report": len(coverage_map),
    }

    print(f"   Matched coverage for {matched} modules")
    print(f"   Average coverage: {analysis.coverage_stats['average_coverage']}%")


def analyze_package(
    package_path: Path,
    root_package: str,
    verbose: bool = False,
    import_prefix: str = "",
) -> PackageMetrics:
    """Recursively analyze a Python package."""
    package = PackageMetrics(path=package_path, name=package_path.name)

    current_prefix = (
        f"{import_prefix}.{package_path.name}" if import_prefix else root_package
    )

    if verbose:
        print(f"  Analyzing {package_path}...")

    for item in sorted(package_path.iterdir()):
        if item.name.startswith("__pycache__"):
            continue

        if item.is_file() and item.suffix == ".py":
            module_name = item.stem
            module_import_path = f"{current_prefix}.{module_name}"
            metrics = analyze_module(item, root_package, module_import_path)
            package.modules[module_name] = metrics

        elif item.is_dir() and (item / "__init__.py").exists():
            subpackage = analyze_package(item, root_package, verbose, current_prefix)
            package.subpackages[item.name] = subpackage

    return package


# ============================================================================
# Class Hierarchy Analysis
# ============================================================================


def build_class_hierarchy(analysis: ArchitectureAnalysis) -> None:
    """Build class hierarchy from all modules."""
    # Collect all classes with their full paths
    for module_path, metrics in analysis.all_modules.items():
        for cls in metrics.class_info:
            fqn = f"{cls.module_path}.{cls.name}" if cls.module_path else cls.name
            analysis.all_classes[fqn] = cls

    # Build inheritance edges
    for fqn, cls in analysis.all_classes.items():
        for base in cls.bases:
            # Try to resolve base class
            # First, check if it's a simple name that exists in our codebase
            resolved_base = None

            # Check for exact match
            for other_fqn in analysis.all_classes:
                if other_fqn.endswith(f".{base}") or other_fqn == base:
                    resolved_base = other_fqn
                    break

            # If not found, use the base name as-is (external class)
            if resolved_base is None:
                resolved_base = base

            analysis.inheritance_edges.append((fqn, resolved_base))

    print(f"   Found {len(analysis.all_classes)} classes")
    print(f"   Found {len(analysis.inheritance_edges)} inheritance relationships")


def generate_class_hierarchy_dot(
    analysis: ArchitectureAnalysis, package_filter: Optional[str] = None
) -> str:
    """Generate DOT graph for class hierarchy."""
    lines = [
        "digraph ClassHierarchy {",
        "    rankdir=BT;  // Bottom to top (children point to parents)",
        '    node [shape=box, style=filled, fontname="Helvetica", fontsize=10];',
        "    edge [arrowhead=empty];  // UML inheritance arrow",
        "",
        "    // Color scheme",
        "    // Abstract classes: light blue",
        "    // Concrete classes: light green",
        "    // External classes: light gray",
        "",
    ]

    # Track which classes to include
    included_classes = set()
    external_classes = set()

    for fqn, cls in analysis.all_classes.items():
        if package_filter and package_filter not in fqn:
            continue
        included_classes.add(fqn)

    # Add edges and track external classes
    edge_lines = []
    for child, parent in analysis.inheritance_edges:
        if child not in included_classes:
            continue

        # Skip generic type bases
        if parent in (
            "object",
            "ABC",
            "Generic",
            "Protocol",
            "TypedDict",
            "Enum",
            "Exception",
        ):
            continue

        if parent not in analysis.all_classes:
            external_classes.add(parent)

        # Sanitize node names for DOT
        child_id = child.replace(".", "_").replace("-", "_")
        parent_id = parent.replace(".", "_").replace("-", "_")
        edge_lines.append(f"    {child_id} -> {parent_id};")

    # Group classes by package
    packages: Dict[str, List[str]] = defaultdict(list)
    for fqn in included_classes:
        parts = fqn.split(".")
        if len(parts) >= 3:
            pkg = ".".join(parts[:3])  # traigent.package.subpackage
        else:
            pkg = parts[0] if parts else "unknown"
        packages[pkg].append(fqn)

    # Create subgraphs for packages
    for pkg_name, class_list in sorted(packages.items()):
        pkg_id = pkg_name.replace(".", "_")
        lines.append(f"    subgraph cluster_{pkg_id} {{")
        lines.append(f'        label="{pkg_name}";')
        lines.append("        style=dashed;")
        lines.append("        color=gray;")

        for fqn in class_list:
            cls = analysis.all_classes[fqn]
            node_id = fqn.replace(".", "_").replace("-", "_")
            short_name = cls.name

            # Color based on type
            if cls.is_abstract:
                color = "#bbdefb"  # Light blue for abstract
            else:
                color = "#c8e6c9"  # Light green for concrete

            label = f"{short_name}\\n({cls.methods} methods)"
            lines.append(f'        {node_id} [label="{label}" fillcolor="{color}"];')

        lines.append("    }")
        lines.append("")

    # Add external classes
    if external_classes:
        lines.append("    // External classes")
        for ext in sorted(external_classes):
            if ext in (
                "object",
                "ABC",
                "Generic",
                "Protocol",
                "TypedDict",
                "Enum",
                "Exception",
            ):
                continue
            node_id = ext.replace(".", "_").replace("-", "_")
            lines.append(
                f'    {node_id} [label="{ext}" fillcolor="#eeeeee" style="filled,dashed"];'
            )
        lines.append("")

    # Add edges
    lines.append("    // Inheritance edges")
    lines.extend(edge_lines)

    lines.append("}")
    return "\n".join(lines)


def generate_complexity_heatmap_dot(analysis: ArchitectureAnalysis) -> str:
    """Generate DOT graph with modules colored by complexity."""
    lines = [
        "digraph ComplexityHeatmap {",
        "    rankdir=TB;",
        '    node [shape=box, style=filled, fontname="Helvetica", fontsize=9];',
        "",
        "    // Complexity color scale:",
        "    // Green (0-5): Low complexity",
        "    // Yellow (5-10): Medium complexity",
        "    // Orange (10-15): High complexity",
        "    // Red (15+): Very high complexity",
        "",
    ]

    def get_complexity_color(cc: float) -> str:
        """Return color based on cyclomatic complexity."""
        if cc <= 5:
            return "#c8e6c9"  # Green
        elif cc <= 10:
            return "#fff9c4"  # Yellow
        elif cc <= 15:
            return "#ffcc80"  # Orange
        else:
            return "#ef9a9a"  # Red

    # Group by package
    packages: Dict[str, List[Tuple[str, ModuleMetrics]]] = defaultdict(list)
    for module_path, metrics in analysis.all_modules.items():
        parts = module_path.split("/")
        # Get the subpackage name (skip root package and files at root level)
        if len(parts) > 2:
            pkg = parts[1]  # e.g., "core" from "traigent/core/orchestrator.py"
        else:
            pkg = "root"  # Root-level files like __init__.py, _version.py
        packages[pkg].append((module_path, metrics))

    # Create subgraphs
    for pkg_name, modules in sorted(packages.items()):
        pkg_id = pkg_name.replace(".", "_").replace("-", "_")
        lines.append(f"    subgraph cluster_{pkg_id} {{")
        lines.append(f'        label="{pkg_name}";')
        lines.append("        style=rounded;")

        for module_path, metrics in modules:
            node_id = module_path.replace("/", "_").replace(".", "_").replace("-", "_")
            module_name = module_path.split("/")[-1].replace(".py", "")

            cc = metrics.avg_complexity
            max_cc = metrics.complexity.max_complexity if metrics.complexity else 0
            color = get_complexity_color(cc)

            label = f"{module_name}\\nCC: {cc:.1f} (max: {max_cc})"
            lines.append(f'        {node_id} [label="{label}" fillcolor="{color}"];')

        lines.append("    }")
        lines.append("")

    # Add legend
    lines.extend(
        [
            "    subgraph cluster_legend {",
            '        label="Complexity Legend";',
            "        style=dashed;",
            '        leg1 [label="Low (0-5)" fillcolor="#c8e6c9"];',
            '        leg2 [label="Medium (5-10)" fillcolor="#fff9c4"];',
            '        leg3 [label="High (10-15)" fillcolor="#ffcc80"];',
            '        leg4 [label="Critical (15+)" fillcolor="#ef9a9a"];',
            "    }",
        ]
    )

    lines.append("}")
    return "\n".join(lines)


def generate_coverage_map_dot(analysis: ArchitectureAnalysis) -> str:
    """Generate DOT graph with modules colored by test coverage."""
    lines = [
        "digraph CoverageMap {",
        "    rankdir=TB;",
        '    node [shape=box, style=filled, fontname="Helvetica", fontsize=9];',
        "",
        "    // Coverage color scale:",
        "    // Red (0-40%): Low coverage",
        "    // Orange (40-60%): Medium coverage",
        "    // Yellow (60-80%): Good coverage",
        "    // Green (80-100%): Excellent coverage",
        "    // Gray: No coverage data",
        "",
    ]

    def get_coverage_color(coverage: Optional[float]) -> str:
        """Return color based on coverage percentage."""
        if coverage is None:
            return "#e0e0e0"  # Gray - no data
        elif coverage < 40:
            return "#ef9a9a"  # Red
        elif coverage < 60:
            return "#ffcc80"  # Orange
        elif coverage < 80:
            return "#fff9c4"  # Yellow
        else:
            return "#c8e6c9"  # Green

    # Group by package
    packages: Dict[str, List[Tuple[str, ModuleMetrics]]] = defaultdict(list)
    for module_path, metrics in analysis.all_modules.items():
        parts = module_path.split("/")
        # Get the subpackage name (skip root package and files at root level)
        if len(parts) > 2:
            pkg = parts[1]  # e.g., "core" from "traigent/core/orchestrator.py"
        else:
            pkg = "root"  # Root-level files like __init__.py, _version.py
        packages[pkg].append((module_path, metrics))

    # Create subgraphs
    for pkg_name, modules in sorted(packages.items()):
        pkg_id = pkg_name.replace(".", "_").replace("-", "_")
        lines.append(f"    subgraph cluster_{pkg_id} {{")
        lines.append(f'        label="{pkg_name}";')
        lines.append("        style=rounded;")

        for module_path, metrics in modules:
            node_id = module_path.replace("/", "_").replace(".", "_").replace("-", "_")
            module_name = module_path.split("/")[-1].replace(".py", "")

            coverage = metrics.coverage_percent
            color = get_coverage_color(coverage)

            if coverage is not None:
                label = f"{module_name}\\n{coverage:.0f}%"
            else:
                label = f"{module_name}\\n(no data)"
            lines.append(f'        {node_id} [label="{label}" fillcolor="{color}"];')

        lines.append("    }")
        lines.append("")

    # Add legend
    lines.extend(
        [
            "    subgraph cluster_legend {",
            '        label="Coverage Legend";',
            "        style=dashed;",
            '        leg1 [label="Low (0-40%)" fillcolor="#ef9a9a"];',
            '        leg2 [label="Medium (40-60%)" fillcolor="#ffcc80"];',
            '        leg3 [label="Good (60-80%)" fillcolor="#fff9c4"];',
            '        leg4 [label="Excellent (80%+)" fillcolor="#c8e6c9"];',
            '        leg5 [label="No Data" fillcolor="#e0e0e0"];',
            "    }",
        ]
    )

    lines.append("}")
    return "\n".join(lines)


# ============================================================================
# Dependency Analysis
# ============================================================================


def extract_dependencies(analysis: ArchitectureAnalysis, root_package: str) -> None:
    """Extract dependency edges from all modules."""
    for module_path, metrics in analysis.all_modules.items():
        for imp in metrics.internal_imports:
            analysis.dependencies.append(
                DependencyEdge(source=module_path, target=imp, import_type="internal")
            )


def find_circular_dependencies(analysis: ArchitectureAnalysis) -> None:
    """Detect circular dependencies using DFS."""
    # Build adjacency list at package level
    graph: Dict[str, Set[str]] = defaultdict(set)

    for dep in analysis.dependencies:
        # Extract top-level package
        source_pkg = dep.source.split("/")[0] if "/" in dep.source else dep.source
        target_parts = dep.target.split(".")
        if len(target_parts) >= 2:
            target_pkg = target_parts[1]  # traigent.X -> X
            if source_pkg != target_pkg:
                graph[source_pkg].add(target_pkg)

    # Find cycles using DFS
    visited = set()
    rec_stack = set()
    cycles = []

    def dfs(node: str, path: List[str]) -> None:
        visited.add(node)
        rec_stack.add(node)
        path.append(node)

        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                dfs(neighbor, path)
            elif neighbor in rec_stack:
                # Found cycle
                cycle_start = path.index(neighbor)
                cycle = path[cycle_start:] + [neighbor]
                if cycle not in cycles:
                    cycles.append(cycle)

        path.pop()
        rec_stack.remove(node)

    for node in graph:
        if node not in visited:
            dfs(node, [])

    analysis.circular_dependencies = cycles


def find_hub_modules(analysis: ArchitectureAnalysis, threshold: int = 5) -> None:
    """Find modules with high fan-in (many dependents)."""
    fan_in: Dict[str, int] = defaultdict(int)

    for dep in analysis.dependencies:
        target_parts = dep.target.split(".")
        if len(target_parts) >= 2:
            target_module = ".".join(target_parts[:3])  # traigent.pkg.module
            fan_in[target_module] += 1

    # Sort by fan-in, descending
    sorted_modules = sorted(fan_in.items(), key=lambda x: x[1], reverse=True)
    analysis.hub_modules = [(m, c) for m, c in sorted_modules if c >= threshold]


def find_orphaned_modules(analysis: ArchitectureAnalysis) -> None:
    """Find modules that are not imported by anything."""
    imported = set()
    for dep in analysis.dependencies:
        imported.add(dep.target)

    all_modules = set()
    for module_path in analysis.all_modules:
        # Convert path to import style
        parts = module_path.replace("/", ".").replace(".py", "")
        all_modules.add(parts)

    # Modules that exist but are never imported (excluding __init__)
    orphaned = []
    for module in all_modules:
        if "__init__" in module:
            continue
        # Check if any import targets this module
        is_imported = any(
            dep.target.startswith(module) or module.endswith(dep.target.split(".")[-1])
            for dep in analysis.dependencies
        )
        if not is_imported and "test" not in module.lower():
            orphaned.append(module)

    analysis.orphaned_modules = orphaned


def define_layers() -> Dict[str, int]:
    """Define architectural layers (higher = more abstract)."""
    return {
        "api": 5,  # Public interface
        "cli": 5,  # CLI interface
        "core": 4,  # Core engine
        "cloud": 3,  # Cloud integration
        "optimizers": 3,  # Optimization algorithms
        "evaluators": 3,  # Evaluation
        "integrations": 2,  # Framework adapters
        "analytics": 2,  # Analytics
        "security": 2,  # Security
        "config": 2,  # Configuration
        "storage": 1,  # Storage
        "utils": 1,  # Utilities
        "telemetry": 1,  # Telemetry
    }


def find_layering_violations(analysis: ArchitectureAnalysis) -> None:
    """Find cases where lower layers depend on higher layers."""
    layers = define_layers()
    violations = []

    for dep in analysis.dependencies:
        source_parts = dep.source.split("/")
        target_parts = dep.target.split(".")

        if len(source_parts) >= 1 and len(target_parts) >= 2:
            source_layer = source_parts[0]
            target_layer = target_parts[1]

            source_level = layers.get(source_layer, 0)
            target_level = layers.get(target_layer, 0)

            # Lower layer depending on higher layer is a violation
            if source_level < target_level and source_level > 0:
                violations.append(
                    (
                        dep.source,
                        dep.target,
                        f"Layer {source_layer}({source_level}) → {target_layer}({target_level})",
                    )
                )

    analysis.layering_violations = violations


# ============================================================================
# Output Generators
# ============================================================================


def generate_package_tree(
    package: PackageMetrics, prefix: str = "", is_last: bool = True
) -> str:
    """Generate ASCII tree view of package structure."""
    lines = []
    connector = "└── " if is_last else "├── "

    # Package header with metrics
    metrics_str = f"[{package.total_files}f, {package.total_lines:,}L, {package.total_classes}C, {package.total_functions}F]"
    lines.append(f"{prefix}{connector}📦 {package.name}/ {metrics_str}")

    # Update prefix for children
    child_prefix = prefix + ("    " if is_last else "│   ")

    # List modules
    module_items = list(package.modules.items())
    subpkg_items = list(package.subpackages.items())
    all_items = len(module_items) + len(subpkg_items)

    for i, (name, metrics) in enumerate(sorted(module_items)):
        is_last_item = (i == all_items - 1) and not subpkg_items
        mod_connector = "└── " if is_last_item else "├── "

        # Module icon based on type
        if name == "__init__":
            icon = "📋"
        elif metrics.classes > 0:
            icon = "🏗️"
        else:
            icon = "📄"

        doc_indicator = "📝" if metrics.has_docstring else "  "
        lines.append(
            f"{child_prefix}{mod_connector}{icon} {name}.py "
            f"[{metrics.lines}L, {metrics.classes}C, {metrics.functions}F] {doc_indicator}"
        )

    # Recurse into subpackages
    for i, (name, subpkg) in enumerate(sorted(subpkg_items)):
        is_last_subpkg = i == len(subpkg_items) - 1
        lines.append(generate_package_tree(subpkg, child_prefix, is_last_subpkg))

    return "\n".join(lines)


def generate_dot_graph(analysis: ArchitectureAnalysis) -> str:
    """Generate Graphviz DOT format dependency graph."""
    lines = [
        "digraph TraigentArchitecture {",
        "    rankdir=TB;",
        '    node [shape=box, style=filled, fontname="Helvetica"];',
        "    edge [fontsize=10];",
        "",
        "    // Color scheme by layer",
        "    subgraph cluster_legend {",
        '        label="Layer Legend";',
        "        style=dashed;",
        '        l5 [label="API/CLI (L5)" fillcolor="#e8f5e9"];',
        '        l4 [label="Core (L4)" fillcolor="#fff3e0"];',
        '        l3 [label="Services (L3)" fillcolor="#e3f2fd"];',
        '        l2 [label="Adapters (L2)" fillcolor="#fce4ec"];',
        '        l1 [label="Utils (L1)" fillcolor="#f3e5f5"];',
        "    }",
        "",
    ]

    # Color map
    layer_colors = {
        5: "#e8f5e9",  # Green - API
        4: "#fff3e0",  # Orange - Core
        3: "#e3f2fd",  # Blue - Services
        2: "#fce4ec",  # Pink - Adapters
        1: "#f3e5f5",  # Purple - Utils
        0: "#ffffff",  # White - Unknown
    }

    layers = define_layers()

    # Collect unique packages
    packages = set()
    for dep in analysis.dependencies:
        source_pkg = dep.source.split("/")[0]
        target_parts = dep.target.split(".")
        if len(target_parts) >= 2:
            target_pkg = target_parts[1]
            packages.add(source_pkg)
            packages.add(target_pkg)

    # Define nodes with colors
    lines.append("    // Package nodes")
    for pkg in sorted(packages):
        level = layers.get(pkg, 0)
        color = layer_colors.get(level, "#ffffff")
        lines.append(f'    {pkg} [label="{pkg}" fillcolor="{color}"];')

    # Define edges (aggregate at package level)
    lines.append("")
    lines.append("    // Dependencies")
    edge_counts: Dict[Tuple[str, str], int] = defaultdict(int)

    for dep in analysis.dependencies:
        source_pkg = dep.source.split("/")[0]
        target_parts = dep.target.split(".")
        if len(target_parts) >= 2:
            target_pkg = target_parts[1]
            if source_pkg != target_pkg:
                edge_counts[(source_pkg, target_pkg)] += 1

    for (source, target), count in sorted(edge_counts.items()):
        # Thicker lines for more dependencies
        penwidth = min(1 + count * 0.3, 5)
        color = "#d32f2f" if count > 10 else "#1976d2"  # Red for heavy coupling
        lines.append(
            f'    {source} -> {target} [penwidth={penwidth:.1f}, color="{color}", label="{count}"];'
        )

    # Highlight circular dependencies
    if analysis.circular_dependencies:
        lines.append("")
        lines.append("    // Circular dependencies (red)")
        for cycle in analysis.circular_dependencies:
            for i in range(len(cycle) - 1):
                lines.append(
                    f'    {cycle[i]} -> {cycle[i+1]} [color="red", style="bold"];'
                )

    lines.append("}")
    return "\n".join(lines)


def generate_json_report(analysis: ArchitectureAnalysis) -> Dict:
    """Generate machine-readable JSON report."""

    def package_to_dict(pkg: PackageMetrics) -> Dict:
        return {
            "name": pkg.name,
            "total_files": pkg.total_files,
            "total_lines": pkg.total_lines,
            "total_classes": pkg.total_classes,
            "total_functions": pkg.total_functions,
            "modules": {
                name: {
                    "lines": m.lines,
                    "classes": m.classes,
                    "functions": m.functions,
                    "has_docstring": m.has_docstring,
                    "internal_imports": m.internal_imports,
                    "external_imports": m.external_imports[:10],  # Limit
                }
                for name, m in pkg.modules.items()
            },
            "subpackages": {
                name: package_to_dict(sub) for name, sub in pkg.subpackages.items()
            },
        }

    return {
        "summary": {
            "total_files": analysis.root_package.total_files,
            "total_lines": analysis.root_package.total_lines,
            "total_classes": analysis.root_package.total_classes,
            "total_functions": analysis.root_package.total_functions,
            "total_dependencies": len(analysis.dependencies),
            "circular_dependencies": len(analysis.circular_dependencies),
            "hub_modules": len(analysis.hub_modules),
            "layering_violations": len(analysis.layering_violations),
        },
        "package_structure": package_to_dict(analysis.root_package),
        "circular_dependencies": analysis.circular_dependencies,
        "hub_modules": [{"module": m, "fan_in": c} for m, c in analysis.hub_modules],
        "layering_violations": [
            {"source": s, "target": t, "violation": v}
            for s, t, v in analysis.layering_violations
        ],
    }


def generate_markdown_report(analysis: ArchitectureAnalysis) -> str:
    """Generate markdown architecture report with gap analysis."""
    lines = [
        "# Traigent Architecture Analysis Report",
        "",
        f"Generated: {__import__('datetime').datetime.now().isoformat()}",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Total Python Files | {analysis.root_package.total_files} |",
        f"| Total Lines of Code | {analysis.root_package.total_lines:,} |",
        f"| Total Classes | {analysis.root_package.total_classes} |",
        f"| Total Functions | {analysis.root_package.total_functions} |",
        f"| Internal Dependencies | {len(analysis.dependencies)} |",
        "",
        "## Package Structure",
        "",
        "```",
        generate_package_tree(analysis.root_package, "", True),
        "```",
        "",
        "### Legend",
        "- `[Nf, NL, NC, NF]` = Files, Lines, Classes, Functions",
        "- 📝 = Has module docstring",
        "- 📦 = Package, 🏗️ = Has classes, 📄 = Module only",
        "",
    ]

    # Architectural Issues
    lines.extend(
        [
            "## Architectural Analysis",
            "",
            "### 🔄 Circular Dependencies",
            "",
        ]
    )

    if analysis.circular_dependencies:
        lines.append(
            f"**Found {len(analysis.circular_dependencies)} circular dependency chains:**"
        )
        lines.append("")
        for cycle in analysis.circular_dependencies:
            lines.append(f"- `{' → '.join(cycle)}`")
        lines.append("")
        lines.append(
            "> ⚠️ **Action Required**: Break these cycles to improve modularity"
        )
    else:
        lines.append("✅ No circular dependencies detected at package level")

    lines.append("")

    # Hub modules
    lines.extend(
        [
            "### 🎯 Hub Modules (High Fan-In)",
            "",
            "Modules imported by many others (potential stability concerns):",
            "",
            "| Module | Dependents |",
            "|--------|------------|",
        ]
    )

    for module, count in analysis.hub_modules[:15]:
        lines.append(f"| `{module}` | {count} |")

    lines.append("")
    lines.append("> 💡 **Tip**: Hub modules should be stable and well-tested")
    lines.append("")

    # Layering violations
    lines.extend(
        [
            "### 📊 Layering Analysis",
            "",
            "Architectural layers (higher = more abstract):",
            "```",
            "L5: api, cli         (Public Interface)",
            "L4: core             (Core Engine)",
            "L3: cloud, optimizers, evaluators (Services)",
            "L2: integrations, analytics, security, config (Adapters)",
            "L1: utils, storage, telemetry (Foundation)",
            "```",
            "",
        ]
    )

    if analysis.layering_violations:
        lines.append(
            f"**Found {len(analysis.layering_violations)} potential layering violations:**"
        )
        lines.append("")
        lines.append("| Source | Target | Issue |")
        lines.append("|--------|--------|-------|")
        for source, target, violation in analysis.layering_violations[:20]:
            lines.append(f"| `{source}` | `{target}` | {violation} |")
        lines.append("")
        lines.append("> ⚠️ **Review**: Lower layers should not depend on higher layers")
    else:
        lines.append("✅ No obvious layering violations detected")

    lines.append("")

    # Recommendations
    lines.extend(
        [
            "## Recommendations",
            "",
            "### High Priority",
            "",
        ]
    )

    if analysis.circular_dependencies:
        lines.append(
            "1. **Break circular dependencies** - Use dependency injection or interfaces"
        )

    if len(analysis.hub_modules) > 10:
        lines.append(
            f"2. **Review hub modules** - {len(analysis.hub_modules)} modules have high fan-in"
        )

    if analysis.layering_violations:
        lines.append(
            f"3. **Fix layering violations** - {len(analysis.layering_violations)} cases of lower layers depending on higher"
        )

    lines.extend(
        [
            "",
            "### Suggested Next Steps",
            "",
            "1. Generate class hierarchy diagrams for core packages",
            "2. Add complexity analysis overlay (cyclomatic complexity)",
            "3. Add test coverage overlay",
            "4. Set up CI to track architecture drift",
            "",
            "---",
            "",
            "*Generated by Traigent Architecture Analyzer*",
        ]
    )

    return "\n".join(lines)


# ============================================================================
# Main Entry Point
# ============================================================================


def collect_all_modules(
    package: PackageMetrics, prefix: str = ""
) -> Dict[str, ModuleMetrics]:
    """Flatten package tree into module dictionary."""
    modules = {}

    current_prefix = f"{prefix}/{package.name}" if prefix else package.name

    for name, metrics in package.modules.items():
        module_path = f"{current_prefix}/{name}.py"
        modules[module_path] = metrics

    for name, subpkg in package.subpackages.items():
        modules.update(collect_all_modules(subpkg, current_prefix))

    return modules


def main():
    parser = argparse.ArgumentParser(
        description="Generate architecture diagrams for Traigent"
    )
    parser.add_argument(
        "--output-dir", default="tools/architecture/output", help="Output directory"
    )
    parser.add_argument(
        "--format", default="all", choices=["text", "dot", "json", "md", "all"]
    )
    parser.add_argument("--package", default="traigent", help="Package to analyze")
    parser.add_argument("--include-tests", action="store_true", help="Include tests/")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument(
        "--complexity",
        action="store_true",
        help="Run complexity analysis (requires radon)",
    )
    parser.add_argument("--coverage", type=str, help="Path to coverage JSON file")
    parser.add_argument(
        "--class-hierarchy",
        action="store_true",
        help="Generate class hierarchy diagrams",
    )
    parser.add_argument(
        "--all-features", action="store_true", help="Enable all analysis features"
    )

    args = parser.parse_args()

    # Enable all features if requested
    if args.all_features:
        args.complexity = True
        args.class_hierarchy = True

    # Find project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    package_path = project_root / args.package

    if not package_path.exists():
        print(f"❌ Package not found: {package_path}", file=sys.stderr)
        sys.exit(1)

    print(f"🔍 Analyzing {package_path}...")

    # Analyze package structure
    root_package = analyze_package(package_path, args.package, args.verbose)

    # Build analysis object
    analysis = ArchitectureAnalysis(root_package=root_package)
    analysis.all_modules = collect_all_modules(root_package)

    print(f"   Found {len(analysis.all_modules)} modules")

    # Extract and analyze dependencies
    print("📊 Analyzing dependencies...")
    extract_dependencies(analysis, args.package)
    print(f"   Found {len(analysis.dependencies)} internal dependencies")

    # Find issues
    print("🔎 Detecting architectural issues...")
    find_circular_dependencies(analysis)
    find_hub_modules(analysis)
    find_orphaned_modules(analysis)
    find_layering_violations(analysis)

    print(f"   Circular dependencies: {len(analysis.circular_dependencies)}")
    print(f"   Hub modules (fan-in ≥5): {len(analysis.hub_modules)}")
    print(f"   Layering violations: {len(analysis.layering_violations)}")

    # Build class hierarchy
    if args.class_hierarchy:
        print("🏗️  Building class hierarchy...")
        build_class_hierarchy(analysis)

    # Run complexity analysis
    if args.complexity:
        run_complexity_analysis(analysis, project_root, args.verbose)

    # Load coverage data
    if args.coverage:
        coverage_path = Path(args.coverage)
        if coverage_path.exists():
            print(f"📊 Loading coverage data from {coverage_path}...")
            coverage_map = load_coverage_data(coverage_path)
            apply_coverage_data(analysis, coverage_map, args.package)
        else:
            print(f"  ⚠️  Coverage file not found: {coverage_path}", file=sys.stderr)

    # Generate outputs
    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n📁 Writing outputs to {output_dir}/")

    formats = ["text", "dot", "json", "md"] if args.format == "all" else [args.format]

    if "text" in formats:
        output_file = output_dir / "package_structure.txt"
        tree = generate_package_tree(root_package, "", True)
        output_file.write_text(tree)
        print(f"   ✅ {output_file.name}")

    if "dot" in formats:
        # Module dependencies
        output_file = output_dir / "module_dependencies.dot"
        dot = generate_dot_graph(analysis)
        output_file.write_text(dot)
        print(f"   ✅ {output_file.name}")

        # Class hierarchy (if enabled)
        if args.class_hierarchy and analysis.all_classes:
            output_file = output_dir / "class_hierarchy.dot"
            dot = generate_class_hierarchy_dot(analysis)
            output_file.write_text(dot)
            print(f"   ✅ {output_file.name}")

            # Also generate focused diagrams for key packages
            for pkg in ["core", "optimizers", "evaluators", "cloud"]:
                output_file = output_dir / f"class_hierarchy_{pkg}.dot"
                dot = generate_class_hierarchy_dot(analysis, f"traigent.{pkg}")
                output_file.write_text(dot)
                print(f"   ✅ {output_file.name}")

        # Complexity heatmap (if enabled)
        if args.complexity and analysis.complexity_stats:
            output_file = output_dir / "complexity_heatmap.dot"
            dot = generate_complexity_heatmap_dot(analysis)
            output_file.write_text(dot)
            print(f"   ✅ {output_file.name}")

        # Coverage map (if enabled)
        if args.coverage and analysis.coverage_stats:
            output_file = output_dir / "coverage_map.dot"
            dot = generate_coverage_map_dot(analysis)
            output_file.write_text(dot)
            print(f"   ✅ {output_file.name}")

    if "json" in formats:
        output_file = output_dir / "architecture_data.json"
        data = generate_json_report(analysis)
        # Add new data to JSON
        if args.complexity:
            data["complexity"] = analysis.complexity_stats
        if args.coverage:
            data["coverage"] = analysis.coverage_stats
        if args.class_hierarchy:
            data["class_hierarchy"] = {
                "total_classes": len(analysis.all_classes),
                "inheritance_edges": len(analysis.inheritance_edges),
                "abstract_classes": sum(
                    1 for c in analysis.all_classes.values() if c.is_abstract
                ),
            }
        output_file.write_text(json.dumps(data, indent=2))
        print(f"   ✅ {output_file.name}")

    if "md" in formats:
        output_file = output_dir / "architecture_report.md"
        report = generate_markdown_report(analysis)

        # Append additional sections for new features
        additional_sections = []

        if args.class_hierarchy and analysis.all_classes:
            additional_sections.extend(
                [
                    "",
                    "## Class Hierarchy Analysis",
                    "",
                    f"- **Total Classes**: {len(analysis.all_classes)}",
                    f"- **Inheritance Relationships**: {len(analysis.inheritance_edges)}",
                    f"- **Abstract Classes**: {sum(1 for c in analysis.all_classes.values() if c.is_abstract)}",
                    "",
                    "### Top Classes by Method Count",
                    "",
                    "| Class | Package | Methods | Abstract |",
                    "|-------|---------|---------|----------|",
                ]
            )
            sorted_classes = sorted(
                analysis.all_classes.items(), key=lambda x: x[1].methods, reverse=True
            )[:15]
            for fqn, cls in sorted_classes:
                pkg = ".".join(fqn.split(".")[:-1])
                abstract = "✓" if cls.is_abstract else ""
                additional_sections.append(
                    f"| `{cls.name}` | {pkg} | {cls.methods} | {abstract} |"
                )

        if args.complexity and analysis.complexity_stats:
            additional_sections.extend(
                [
                    "",
                    "## Complexity Analysis",
                    "",
                    f"- **Modules Analyzed**: {analysis.complexity_stats.get('modules_analyzed', 0)}",
                    f"- **Average Complexity**: {analysis.complexity_stats.get('average_complexity', 0)}",
                    f"- **High Complexity Modules**: {len(analysis.complexity_stats.get('high_complexity_modules', []))}",
                    "",
                    "### High Complexity Modules (Max CC > 15)",
                    "",
                    "| Module | Avg CC | Max CC |",
                    "|--------|--------|--------|",
                ]
            )
            for mod, avg, max_cc in analysis.complexity_stats.get(
                "high_complexity_modules", []
            )[:15]:
                additional_sections.append(f"| `{mod}` | {avg:.1f} | {max_cc} |")

        if args.coverage and analysis.coverage_stats:
            additional_sections.extend(
                [
                    "",
                    "## Test Coverage Analysis",
                    "",
                    f"- **Modules with Coverage Data**: {analysis.coverage_stats.get('modules_with_coverage', 0)}",
                    f"- **Average Coverage**: {analysis.coverage_stats.get('average_coverage', 0)}%",
                    "",
                ]
            )

        if additional_sections:
            report += "\n" + "\n".join(additional_sections)

        output_file.write_text(report)
        print(f"   ✅ {output_file.name}")

    print("\n✨ Done!")

    # Quick summary
    print(f"\n📈 Quick Stats:")
    print(f"   Files: {root_package.total_files}")
    print(f"   Lines: {root_package.total_lines:,}")
    print(f"   Classes: {root_package.total_classes}")
    print(f"   Functions: {root_package.total_functions}")

    if args.class_hierarchy:
        print(f"   Inheritance relationships: {len(analysis.inheritance_edges)}")

    if args.complexity:
        print(
            f"   Avg complexity: {analysis.complexity_stats.get('average_complexity', 'N/A')}"
        )

    if args.coverage:
        print(
            f"   Avg coverage: {analysis.coverage_stats.get('average_coverage', 'N/A')}%"
        )

    # Print rendering hints
    print("\n💡 To render DOT files as images:")
    print(
        f"   dot -Tsvg {output_dir}/module_dependencies.dot -o {output_dir}/module_dependencies.svg"
    )
    print(
        f"   dot -Tpng {output_dir}/complexity_heatmap.dot -o {output_dir}/complexity_heatmap.png"
    )
    print("\n   Or use VS Code extension: Graphviz Interactive Preview")


if __name__ == "__main__":
    main()
