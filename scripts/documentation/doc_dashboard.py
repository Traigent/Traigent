#!/usr/bin/env python3
"""
Documentation Dashboard for Traigent SDK

Provides a visual overview of documentation health, coverage, and status.
Tracks metrics, identifies issues, and suggests improvements.

Usage:
    python scripts/doc_dashboard.py              # Show full dashboard
    python scripts/doc_dashboard.py --simple     # Simple text output
    python scripts/doc_dashboard.py --json       # JSON output for automation
"""

import argparse
import json
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict

from traigent.utils.secure_path import PathTraversalError, safe_write_text, validate_path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class DocumentationDashboard:
    """Generate documentation health dashboard."""

    def __init__(self, root_path: Path = PROJECT_ROOT):
        self.root_path = root_path
        self.metrics = {}
        self.issues = []
        self.recommendations = []

    def generate_dashboard(self) -> Dict:
        """Generate complete dashboard data."""
        print("📊 Generating Documentation Dashboard...\n")

        # Collect all metrics
        self.metrics["coverage"] = self._calculate_coverage()
        self.metrics["freshness"] = self._calculate_freshness()
        self.metrics["quality"] = self._assess_quality()
        self.metrics["consistency"] = self._check_consistency()
        self.metrics["completeness"] = self._check_completeness()

        # Calculate overall health score
        self.metrics["health_score"] = self._calculate_health_score()

        # Generate recommendations
        self._generate_recommendations()

        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metrics": self.metrics,
            "issues": self.issues,
            "recommendations": self.recommendations,
        }

    def _calculate_coverage(self) -> Dict:
        """Calculate documentation coverage metrics."""
        import ast

        coverage = {
            "public_apis": 0,
            "documented_apis": 0,
            "percentage": 0,
            "by_module": {},
        }

        traigent_path = self.root_path / "traigent"
        if not traigent_path.exists():
            return coverage

        for py_file in traigent_path.rglob("*.py"):
            if "__pycache__" in str(py_file):
                continue

            module_name = str(py_file.relative_to(traigent_path))[:-3]
            module_stats = {"total": 0, "documented": 0}

            try:
                content = py_file.read_text()
                tree = ast.parse(content)

                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                        if not node.name.startswith("_"):
                            coverage["public_apis"] += 1
                            module_stats["total"] += 1

                            if ast.get_docstring(node):
                                coverage["documented_apis"] += 1
                                module_stats["documented"] += 1
                            else:
                                self.issues.append(
                                    f"Missing docstring: {module_name}.{node.name}"
                                )

                if module_stats["total"] > 0:
                    module_stats["percentage"] = (
                        module_stats["documented"] / module_stats["total"]
                    ) * 100
                    coverage["by_module"][module_name] = module_stats

            except Exception as e:
                self.issues.append(f"Error parsing {py_file.name}: {e}")

        if coverage["public_apis"] > 0:
            coverage["percentage"] = (
                coverage["documented_apis"] / coverage["public_apis"]
            ) * 100

        return coverage

    def _calculate_freshness(self) -> Dict:
        """Calculate documentation freshness metrics."""
        freshness = {
            "avg_age_days": 0,
            "stale_files": [],
            "recently_updated": [],
            "never_updated": [],
        }

        ages = []
        current_time = datetime.now(timezone.utc)
        stale_threshold = timedelta(days=90)  # 3 months
        recent_threshold = timedelta(days=7)  # 1 week

        for md_file in self.root_path.rglob("*.md"):
            if any(
                part in str(md_file) for part in ["venv", "env", "node_modules", ".git"]
            ):
                continue

            # Get file modification time
            mtime = datetime.fromtimestamp(md_file.stat().st_mtime)
            age = current_time - mtime
            ages.append(age.days)

            rel_path = str(md_file.relative_to(self.root_path))

            if age > stale_threshold:
                freshness["stale_files"].append((rel_path, age.days))
                self.issues.append(
                    f"Stale documentation: {rel_path} ({age.days} days old)"
                )
            elif age < recent_threshold:
                freshness["recently_updated"].append((rel_path, age.days))

        if ages:
            freshness["avg_age_days"] = sum(ages) / len(ages)

        # Sort by age
        freshness["stale_files"].sort(key=lambda x: x[1], reverse=True)
        freshness["recently_updated"].sort(key=lambda x: x[1])

        return freshness

    def _assess_quality(self) -> Dict:
        """Assess documentation quality metrics."""
        quality = {
            "examples_found": 0,
            "examples_tested": 0,
            "broken_links": 0,
            "spelling_errors": 0,  # Would need spell checker
            "readability_score": 0,
            "average_doc_length": 0,
        }

        doc_lengths = []

        for md_file in self.root_path.rglob("*.md"):
            if any(part in str(md_file) for part in ["venv", "env", "node_modules"]):
                continue

            content = md_file.read_text()
            doc_lengths.append(len(content))

            # Count code examples
            python_blocks = re.findall(r"```python\n(.*?)\n```", content, re.DOTALL)
            quality["examples_found"] += len(python_blocks)

            # Check for broken internal links
            links = re.findall(r"\[([^\]]+)\]\(([^\)]+)\)", content)
            for _, link_url in links:
                if not link_url.startswith(("http://", "https://", "#", "mailto:")):
                    target = md_file.parent / link_url
                    if not target.exists():
                        quality["broken_links"] += 1

        if doc_lengths:
            quality["average_doc_length"] = sum(doc_lengths) / len(doc_lengths)

        # Simple readability estimate (would use proper algorithm in production)
        quality["readability_score"] = min(100, quality["average_doc_length"] / 100)

        return quality

    def _check_consistency(self) -> Dict:
        """Check documentation consistency."""
        consistency = {
            "version_consistent": True,
            "terminology_consistent": True,
            "format_consistent": True,
            "issues": [],
        }

        # Check version consistency
        versions = set()
        version_pattern = r'version[:\s=]+["\']?([0-9]+\.[0-9]+\.[0-9]+)'

        for file_path in [
            self.root_path / "pyproject.toml",
            self.root_path / "setup.py",
            self.root_path / "traigent" / "__init__.py",
        ]:
            if file_path.exists():
                content = file_path.read_text()
                matches = re.findall(version_pattern, content, re.IGNORECASE)
                versions.update(matches)

        if len(versions) > 1:
            consistency["version_consistent"] = False
            consistency["issues"].append(f"Inconsistent versions: {versions}")
            self.issues.append(f"Version inconsistency detected: {versions}")

        # Check terminology (simplified)
        terms_variations = {
            "Traigent": ["Traigent", "TRAIGENT", "traigent sdk"],
            "optimization": ["optimisation"],
        }

        for correct_term, variations in terms_variations.items():
            for md_file in self.root_path.glob("*.md"):
                content = md_file.read_text()
                for variation in variations:
                    if variation in content:
                        consistency["terminology_consistent"] = False
                        consistency["issues"].append(
                            f"Found '{variation}' instead of '{correct_term}' in {md_file.name}"
                        )

        return consistency

    def _check_completeness(self) -> Dict:
        """Check documentation completeness."""
        completeness = {"required_files": {}, "percentage": 0}

        required_docs = [
            "README.md",
            "CHANGELOG.md",
            "CONTRIBUTING.md",
            "LICENSE",
            "SECURITY.md",
            "INSTALLATION.md",
            "docs/CURRENT_STATUS.md",
        ]

        found = 0
        for doc in required_docs:
            doc_path = self.root_path / doc
            exists = doc_path.exists()
            completeness["required_files"][doc] = exists
            if exists:
                found += 1
            else:
                self.issues.append(f"Missing required documentation: {doc}")

        completeness["percentage"] = (found / len(required_docs)) * 100

        return completeness

    def _calculate_health_score(self) -> float:
        """Calculate overall documentation health score."""
        weights = {
            "coverage": 0.3,
            "freshness": 0.2,
            "quality": 0.2,
            "consistency": 0.15,
            "completeness": 0.15,
        }

        scores = {
            "coverage": self.metrics["coverage"]["percentage"],
            "freshness": max(0, 100 - (self.metrics["freshness"]["avg_age_days"] / 3)),
            "quality": min(
                100,
                self.metrics["quality"]["readability_score"]
                + (50 if self.metrics["quality"]["broken_links"] == 0 else 0),
            ),
            "consistency": (
                100 if self.metrics["consistency"]["version_consistent"] else 50
            ),
            "completeness": self.metrics["completeness"]["percentage"],
        }

        health_score = sum(scores[key] * weights[key] for key in weights)

        return round(health_score, 2)

    def _generate_recommendations(self) -> None:
        """Generate actionable recommendations."""
        # Coverage recommendations
        if self.metrics["coverage"]["percentage"] < 80:
            self.recommendations.append(
                {
                    "priority": "HIGH",
                    "category": "Coverage",
                    "action": f"Add docstrings to {self.metrics['coverage']['public_apis'] - self.metrics['coverage']['documented_apis']} undocumented APIs",
                    "impact": "Improves developer experience and API usability",
                }
            )

        # Freshness recommendations
        if self.metrics["freshness"]["stale_files"]:
            self.recommendations.append(
                {
                    "priority": "MEDIUM",
                    "category": "Freshness",
                    "action": f"Review and update {len(self.metrics['freshness']['stale_files'])} stale documentation files",
                    "impact": "Ensures documentation accuracy",
                }
            )

        # Quality recommendations
        if self.metrics["quality"]["broken_links"] > 0:
            self.recommendations.append(
                {
                    "priority": "HIGH",
                    "category": "Quality",
                    "action": f"Fix {self.metrics['quality']['broken_links']} broken links",
                    "impact": "Improves navigation and user experience",
                }
            )

        # Consistency recommendations
        if not self.metrics["consistency"]["version_consistent"]:
            self.recommendations.append(
                {
                    "priority": "HIGH",
                    "category": "Consistency",
                    "action": "Synchronize version numbers across all files",
                    "impact": "Prevents confusion about current version",
                }
            )

        # Completeness recommendations
        missing_files = [
            f
            for f, exists in self.metrics["completeness"]["required_files"].items()
            if not exists
        ]
        if missing_files:
            self.recommendations.append(
                {
                    "priority": "MEDIUM",
                    "category": "Completeness",
                    "action": f"Create missing documentation files: {', '.join(missing_files)}",
                    "impact": "Provides complete documentation set",
                }
            )

    def print_dashboard(self, simple: bool = False) -> None:
        """Print dashboard to console."""
        data = self.generate_dashboard()

        if simple:
            self._print_simple(data)
        else:
            self._print_detailed(data)

    def _print_simple(self, data: Dict) -> None:
        """Print simple text dashboard."""
        print("=" * 60)
        print("DOCUMENTATION DASHBOARD")
        print("=" * 60)
        print(f"Generated: {data['timestamp']}")
        print(f"Health Score: {data['metrics']['health_score']}/100")
        print()

        print("METRICS:")
        print(f"  Coverage: {data['metrics']['coverage']['percentage']:.1f}%")
        print(f"  Avg Age: {data['metrics']['freshness']['avg_age_days']:.0f} days")
        print(f"  Broken Links: {data['metrics']['quality']['broken_links']}")
        print(f"  Completeness: {data['metrics']['completeness']['percentage']:.1f}%")
        print()

        if data["issues"]:
            print(f"ISSUES ({len(data['issues'])}):")
            for issue in data["issues"][:5]:
                print(f"  - {issue}")
            if len(data["issues"]) > 5:
                print(f"  ... and {len(data['issues']) - 5} more")
        print()

        if data["recommendations"]:
            print("TOP RECOMMENDATIONS:")
            for rec in data["recommendations"][:3]:
                print(f"  [{rec['priority']}] {rec['action']}")

    def _print_detailed(self, data: Dict) -> None:
        """Print detailed dashboard with visual elements."""
        print("╔" + "═" * 58 + "╗")
        print("║" + " TRAIGENT DOCUMENTATION DASHBOARD ".center(58) + "║")
        print("╠" + "═" * 58 + "╣")

        # Health Score with visual bar
        score = data["metrics"]["health_score"]
        bar_length = int(score / 100 * 30)
        bar = "█" * bar_length + "░" * (30 - bar_length)
        color = "🟢" if score >= 80 else "🟡" if score >= 60 else "🔴"
        print(f"║ Health Score: {color} [{bar}] {score:.1f}/100".ljust(59) + "║")

        print("╠" + "═" * 58 + "╣")

        # Metrics
        print("║ METRICS".ljust(59) + "║")
        print("║" + "-" * 58 + "║")

        metrics_display = [
            (
                "Documentation Coverage",
                f"{data['metrics']['coverage']['percentage']:.1f}%",
                data["metrics"]["coverage"]["percentage"] >= 80,
            ),
            (
                "Average Freshness",
                f"{data['metrics']['freshness']['avg_age_days']:.0f} days",
                data["metrics"]["freshness"]["avg_age_days"] <= 30,
            ),
            (
                "Quality Score",
                f"{data['metrics']['quality']['readability_score']:.0f}/100",
                data["metrics"]["quality"]["readability_score"] >= 70,
            ),
            (
                "Broken Links",
                str(data["metrics"]["quality"]["broken_links"]),
                data["metrics"]["quality"]["broken_links"] == 0,
            ),
            (
                "Version Consistency",
                "✓" if data["metrics"]["consistency"]["version_consistent"] else "✗",
                data["metrics"]["consistency"]["version_consistent"],
            ),
            (
                "Completeness",
                f"{data['metrics']['completeness']['percentage']:.1f}%",
                data["metrics"]["completeness"]["percentage"] == 100,
            ),
        ]

        for name, value, is_good in metrics_display:
            status = "✅" if is_good else "⚠️"
            line = f"║ {status} {name}: {value}"
            print(line.ljust(59) + "║")

        print("╠" + "═" * 58 + "╣")

        # Top Issues
        if data["issues"]:
            print("║ TOP ISSUES".ljust(59) + "║")
            print("║" + "-" * 58 + "║")
            for issue in data["issues"][:3]:
                if len(issue) > 54:
                    issue = issue[:51] + "..."
                print(f"║ • {issue}".ljust(59) + "║")

        print("╠" + "═" * 58 + "╣")

        # Recommendations
        if data["recommendations"]:
            print("║ RECOMMENDATIONS".ljust(59) + "║")
            print("║" + "-" * 58 + "║")
            for rec in data["recommendations"][:3]:
                priority_icon = (
                    "🔴"
                    if rec["priority"] == "HIGH"
                    else "🟡" if rec["priority"] == "MEDIUM" else "🟢"
                )
                action = rec["action"]
                if len(action) > 50:
                    action = action[:47] + "..."
                print(f"║ {priority_icon} {action}".ljust(59) + "║")

        print("╚" + "═" * 58 + "╝")
        print()
        print(f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total Issues: {len(data['issues'])}")
        print(f"Total Recommendations: {len(data['recommendations'])}")

    def export_json(self, output_path: Path = None) -> None:
        """Export dashboard data as JSON."""
        data = self.generate_dashboard()

        if output_path is None:
            output_path = self.root_path / "docs" / "dashboard_report.json"

        try:
            output_path = validate_path(output_path, self.root_path)
        except (PathTraversalError, FileNotFoundError) as exc:
            raise ValueError(f"Invalid output path: {exc}") from exc

        output_path.parent.mkdir(parents=True, exist_ok=True)

        safe_write_text(
            output_path,
            json.dumps(data, indent=2, default=str),
            output_path.parent,
        )

        print(f"✅ Dashboard exported to {output_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Documentation Dashboard for Traigent")
    parser.add_argument("--simple", action="store_true", help="Simple text output")
    parser.add_argument("--json", action="store_true", help="Export as JSON")
    parser.add_argument("--output", help="Output file for JSON export")

    args = parser.parse_args()

    dashboard = DocumentationDashboard()

    if args.json:
        output_path = Path(args.output) if args.output else None
        dashboard.export_json(output_path)
    else:
        dashboard.print_dashboard(simple=args.simple)


if __name__ == "__main__":
    main()
