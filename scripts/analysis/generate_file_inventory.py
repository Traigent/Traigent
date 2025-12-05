#!/usr/bin/env python3
"""
Generate file inventory for PROJECT_CLEANUP_REVIEW.md
Creates a CSV file with all project files for systematic review
"""

import csv
import datetime
import os
from pathlib import Path
from typing import Dict, List

# Directories to exclude
EXCLUDE_DIRS = {
    "__pycache__",
    ".git",
    ".pytest_cache",
    "venv",
    "traigent_test_env",
    "node_modules",
    ".venv",
    "env",
    ".env",
    "htmlcov",
    ".coverage",
    "dist",
    "build",
    "*.egg-info",
    ".mypy_cache",
    ".ruff_cache",
    ".tox",
    ".nox",
    ".hypothesis",
}

# File extensions to categorize
FILE_CATEGORIES = {
    "Python": {".py", ".pyx", ".pyi"},
    "Documentation": {".md", ".rst", ".txt", ".adoc"},
    "Configuration": {".json", ".yaml", ".yml", ".toml", ".ini", ".cfg"},
    "Test": {"test_*.py", "*_test.py", "conftest.py"},
    "Requirements": {
        "requirements*.txt",
        "Pipfile",
        "poetry.lock",
        "setup.py",
        "setup.cfg",
        "pyproject.toml",
    },
    "Scripts": {".sh", ".bash", ".zsh", ".bat", ".ps1"},
    "Web": {".html", ".css", ".js", ".jsx", ".ts", ".tsx"},
    "Data": {".csv", ".jsonl", ".json", ".xml", ".sql"},
    "Image": {".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico"},
    "Notebook": {".ipynb"},
    "Cache": {".pyc", ".pyo", ".pyd", ".so", ".dylib", ".dll"},
    "IDE": {".idea", ".vscode", ".sublime-*"},
    "Other": set(),  # Catch-all
}


def should_exclude_path(path: Path) -> bool:
    """Check if path should be excluded"""
    parts = path.parts
    for part in parts:
        if part in EXCLUDE_DIRS:
            return True
        if part.startswith(".") and part not in {
            ".github",
            ".gitignore",
            ".gitattributes",
        }:
            return True
    return False


def categorize_file(file_path: Path) -> str:
    """Categorize file based on extension and name patterns"""
    name = file_path.name
    ext = file_path.suffix.lower()

    # Check for test files
    if name.startswith("test_") or name.endswith("_test.py") or name == "conftest.py":
        return "Test"

    # Check by extension
    for category, extensions in FILE_CATEGORIES.items():
        if ext in extensions:
            return category

    # Check for requirements files
    if "requirements" in name.lower() or name in {
        "Pipfile",
        "poetry.lock",
        "setup.py",
        "setup.cfg",
        "pyproject.toml",
    }:
        return "Requirements"

    # Default to Other
    return "Other"


def get_file_info(file_path: Path, project_root: Path) -> Dict[str, any]:
    """Get detailed information about a file"""
    try:
        stat = file_path.stat()
        relative_path = file_path.relative_to(project_root)

        return {
            "path": str(relative_path),
            "category": categorize_file(file_path),
            "size": stat.st_size,
            "modified": datetime.datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "extension": file_path.suffix.lower(),
            "status": "pending",
            "action": "",
            "notes": "",
        }
    except Exception:
        return None


def generate_inventory(project_root: Path) -> List[Dict[str, any]]:
    """Generate complete file inventory"""
    inventory = []

    for root, dirs, files in os.walk(project_root):
        root_path = Path(root)

        # Filter out excluded directories
        dirs[:] = [d for d in dirs if not should_exclude_path(root_path / d)]

        # Skip if current directory should be excluded
        if should_exclude_path(root_path):
            continue

        # Process files
        for file in files:
            file_path = root_path / file

            # Skip cache files
            if file.endswith((".pyc", ".pyo", ".pyd")):
                continue

            file_info = get_file_info(file_path, project_root)
            if file_info:
                inventory.append(file_info)

    return inventory


def generate_summary(inventory: List[Dict[str, any]]) -> Dict[str, any]:
    """Generate summary statistics"""
    summary = {
        "total_files": len(inventory),
        "by_category": {},
        "by_extension": {},
        "total_size": sum(f["size"] for f in inventory),
        "generated_at": datetime.datetime.now().isoformat(),
    }

    # Count by category
    for file in inventory:
        category = file["category"]
        summary["by_category"][category] = summary["by_category"].get(category, 0) + 1

        ext = file["extension"]
        if ext:
            summary["by_extension"][ext] = summary["by_extension"].get(ext, 0) + 1

    return summary


def write_csv(inventory: List[Dict[str, any]], output_path: Path):
    """Write inventory to CSV file"""
    if not inventory:
        return

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=inventory[0].keys())
        writer.writeheader()
        writer.writerows(inventory)


def write_summary_markdown(
    summary: Dict[str, any], inventory_stats: Dict[str, int], output_path: Path
):
    """Write summary as markdown"""
    with open(output_path, "w") as f:
        f.write("# File Inventory Summary\n\n")
        f.write(f"Generated: {summary['generated_at']}\n\n")
        f.write("## Statistics\n\n")
        f.write(f"- Total Files: {summary['total_files']:,}\n")
        f.write(f"- Total Size: {summary['total_size'] / (1024*1024):.2f} MB\n\n")

        f.write("## Files by Category\n\n")
        f.write("| Category | Count | Percentage |\n")
        f.write("|----------|-------|------------|\n")
        for category, count in sorted(
            summary["by_category"].items(), key=lambda x: x[1], reverse=True
        ):
            percentage = (count / summary["total_files"]) * 100
            f.write(f"| {category} | {count:,} | {percentage:.1f}% |\n")

        f.write("\n## Top File Extensions\n\n")
        f.write("| Extension | Count | Percentage |\n")
        f.write("|-----------|-------|------------|\n")
        for ext, count in sorted(
            summary["by_extension"].items(), key=lambda x: x[1], reverse=True
        )[:20]:
            percentage = (count / summary["total_files"]) * 100
            f.write(f"| {ext} | {count:,} | {percentage:.1f}% |\n")


def main():
    """Main execution"""
    project_root = Path(__file__).parent.parent  # Go up from scripts/ to project root

    print(f"Scanning project: {project_root}")
    print("This may take a moment...")

    # Generate inventory
    inventory = generate_inventory(project_root)
    print(f"Found {len(inventory):,} files")

    # Generate summary
    summary = generate_summary(inventory)

    # Create output directory
    output_dir = project_root / "project_review"
    output_dir.mkdir(exist_ok=True)

    # Write outputs
    csv_path = output_dir / "file_inventory.csv"
    summary_path = output_dir / "inventory_summary.md"

    write_csv(inventory, csv_path)
    print(f"Written CSV inventory to: {csv_path}")

    write_summary_markdown(summary, summary.get("by_category", {}), summary_path)
    print(f"Written summary to: {summary_path}")

    # Update PROJECT_CLEANUP_REVIEW.md
    review_path = project_root / "PROJECT_CLEANUP_REVIEW.md"
    if review_path.exists():
        with open(review_path, "a") as f:
            f.write("\n\n## File Inventory Generated\n\n")
            f.write(f"- Full inventory: `{csv_path.relative_to(project_root)}`\n")
            f.write(f"- Summary report: `{summary_path.relative_to(project_root)}`\n")
            f.write(f"- Total files to review: {len(inventory):,}\n")
            f.write(f"- Generated at: {summary['generated_at']}\n")
        print("Updated PROJECT_CLEANUP_REVIEW.md")


if __name__ == "__main__":
    main()
