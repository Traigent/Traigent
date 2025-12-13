#!/usr/bin/env python3
"""TODO Importer for Post-Release Recommendation Fixes.

Imports TODOs from POST_RELEASE_TODO.md into the fix tracking system.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class TodoItem:
    """A single TODO item from POST_RELEASE_TODO.md."""

    id: str
    title: str
    priority: str  # High, Medium, Low
    component: str
    location: str
    issue: str
    recommendation: str
    effort: str
    status: str = "Pending"
    owner: str = ""
    evidence: str = ""
    depends_on: list[str] = field(default_factory=list)  # Fix IDs this depends on
    started_at: str = ""
    completed_at: str = ""

    def to_markdown_row(self) -> str:
        """Convert to markdown table row."""
        return (
            f"| {self.id} | {self.title} | {self.priority} | "
            f"{self.component} | {self.status} | {self.owner} |"
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "title": self.title,
            "priority": self.priority,
            "component": self.component,
            "location": self.location,
            "issue": self.issue,
            "recommendation": self.recommendation,
            "effort": self.effort,
            "status": self.status,
            "owner": self.owner,
            "evidence": self.evidence,
            "depends_on": self.depends_on,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
        }


class TodoImporter:
    """Import TODOs from POST_RELEASE_TODO.md."""

    # Priority mapping
    PRIORITY_MAP = {
        "High": "P0",
        "Medium": "P1",
        "Low": "P2",
    }

    # Required sections for validation
    REQUIRED_SECTIONS = ["## High Priority", "## Medium Priority", "## Low Priority"]

    def __init__(self, source: str | Path) -> None:
        """Initialize importer.

        Args:
            source: Path to POST_RELEASE_TODO.md
        """
        self.source = Path(source)
        self.warnings: list[str] = []

    def validate_source_format(self) -> tuple[bool, list[str]]:
        """Validate POST_RELEASE_TODO.md adheres to expected format.

        Returns:
            Tuple of (is_valid, list of error messages)
        """
        if not self.source.exists():
            return False, [f"Source file not found: {self.source}"]

        content = self.source.read_text()
        errors = []

        # Check for required sections
        for section in self.REQUIRED_SECTIONS:
            if section not in content:
                errors.append(f"Missing required section: {section}")

        # Check for at least one item (### N. Title)
        if not re.search(r"^###\s+\d+\.\s+", content, re.MULTILINE):
            errors.append("No TODO items found (expected ### N. Title format)")

        return len(errors) == 0, errors

    def parse_source(self) -> list[TodoItem]:
        """Parse POST_RELEASE_TODO.md and extract TODO items.

        Returns:
            List of TodoItem objects
        """
        if not self.source.exists():
            raise FileNotFoundError(f"Source file not found: {self.source}")

        content = self.source.read_text()
        items: list[TodoItem] = []

        # Parse sections by priority
        current_priority = "Medium"  # Default
        current_item: dict[str, Any] = {}
        item_number = 0

        lines = content.split("\n")
        i = 0

        while i < len(lines):
            line = lines[i].strip()

            # Detect priority section
            if "## High Priority" in line:
                current_priority = "High"
            elif "## Medium Priority" in line:
                current_priority = "Medium"
            elif "## Low Priority" in line:
                current_priority = "Low"
            elif "## Accepted Risks" in line or "## Tracking" in line:
                # Stop parsing at these sections
                break

            # Detect item start (### N. Title)
            item_match = re.match(r"^###\s+(\d+)\.\s+(.+)$", line)
            if item_match:
                # Save previous item if exists
                if current_item:
                    items.append(self._create_item(current_item))

                item_number = int(item_match.group(1))
                current_item = {
                    "id": f"{item_number:03d}",
                    "title": item_match.group(2).strip(),
                    "priority": current_priority,
                    "component": "",
                    "location": "",
                    "issue": "",
                    "recommendation": "",
                    "effort": "",
                }

            # Parse item fields
            elif current_item:
                if line.startswith("- **Component**:"):
                    current_item["component"] = line.split(":", 1)[1].strip()
                elif line.startswith("- **Location**:"):
                    current_item["location"] = line.split(":", 1)[1].strip()
                elif line.startswith("- **Issue**:"):
                    current_item["issue"] = line.split(":", 1)[1].strip()
                elif line.startswith("- **Recommendation**:"):
                    # Recommendation might span multiple lines (code block)
                    rec_lines = [line.split(":", 1)[1].strip()]
                    i += 1
                    while i < len(lines) and not lines[i].strip().startswith("- **"):
                        if lines[i].strip().startswith("###"):
                            break  # Don't decrement, let outer loop handle
                        rec_lines.append(lines[i])
                        i += 1
                    # Adjust index since outer loop will increment
                    i -= 1
                    current_item["recommendation"] = "\n".join(rec_lines).strip()
                elif line.startswith("- **Effort**:"):
                    current_item["effort"] = line.split(":", 1)[1].strip()

            i += 1

        # Don't forget last item
        if current_item:
            items.append(self._create_item(current_item))

        return items

    def _create_item(self, data: dict[str, Any]) -> TodoItem:
        """Create TodoItem from parsed data."""
        return TodoItem(
            id=data.get("id", "000"),
            title=data.get("title", "Unknown"),
            priority=data.get("priority", "Medium"),
            component=data.get("component", ""),
            location=data.get("location", ""),
            issue=data.get("issue", ""),
            recommendation=data.get("recommendation", ""),
            effort=data.get("effort", ""),
        )

    def import_to_tracking(self, target: str | Path) -> Path:
        """Import TODOs to tracking file.

        Args:
            target: Path to TRACKING.md

        Returns:
            Path to created tracking file
        """
        items = self.parse_source()
        target_path = Path(target)

        # Group by priority
        high = [i for i in items if i.priority == "High"]
        medium = [i for i in items if i.priority == "Medium"]
        low = [i for i in items if i.priority == "Low"]

        # Generate tracking content
        content = self._generate_tracking_content(items, high, medium, low)

        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_text(content)

        return target_path

    def _generate_tracking_content(
        self,
        all_items: list[TodoItem],
        high: list[TodoItem],
        medium: list[TodoItem],
        low: list[TodoItem],
    ) -> str:
        """Generate TRACKING.md content."""
        timestamp = datetime.now().isoformat() + "Z"

        lines = [
            "# Post-Release Recommendation Fixes Tracking",
            "",
            f"**Source**: `{self.source}`",
            f"**Imported**: {timestamp}",
            f"**Total Items**: {len(all_items)}",
            "",
            "## Summary",
            "",
            "| Priority | Total | Pending | In Progress | Complete |",
            "|----------|-------|---------|-------------|----------|",
            f"| High (P0) | {len(high)} | {len(high)} | 0 | 0 |",
            f"| Medium (P1) | {len(medium)} | {len(medium)} | 0 | 0 |",
            f"| Low (P2) | {len(low)} | {len(low)} | 0 | 0 |",
            f"| **Total** | **{len(all_items)}** | **{len(all_items)}** | **0** | **0** |",
            "",
            "---",
            "",
            "## High Priority (P0)",
            "",
            "| ID | Title | Component | Status | Owner | Evidence |",
            "|----|-------|-----------|--------|-------|----------|",
        ]

        for item in high:
            lines.append(
                f"| {item.id} | {item.title} | {item.component} | "
                f"{item.status} | {item.owner} | {item.evidence} |"
            )

        lines.extend([
            "",
            "---",
            "",
            "## Medium Priority (P1)",
            "",
            "| ID | Title | Component | Status | Owner | Evidence |",
            "|----|-------|-----------|--------|-------|----------|",
        ])

        for item in medium:
            lines.append(
                f"| {item.id} | {item.title} | {item.component} | "
                f"{item.status} | {item.owner} | {item.evidence} |"
            )

        lines.extend([
            "",
            "---",
            "",
            "## Low Priority (P2)",
            "",
            "| ID | Title | Component | Status | Owner | Evidence |",
            "|----|-------|-----------|--------|-------|----------|",
        ])

        for item in low:
            lines.append(
                f"| {item.id} | {item.title} | {item.component} | "
                f"{item.status} | {item.owner} | {item.evidence} |"
            )

        lines.extend([
            "",
            "---",
            "",
            "## Item Details",
            "",
        ])

        for item in all_items:
            lines.extend([
                f"### {item.id}: {item.title}",
                "",
                f"- **Priority**: {item.priority}",
                f"- **Component**: {item.component}",
                f"- **Location**: {item.location}",
                f"- **Issue**: {item.issue}",
                f"- **Effort**: {item.effort}",
                f"- **Status**: {item.status}",
                "",
                "**Recommendation**:",
                item.recommendation if item.recommendation else "(See source file)",
                "",
                "**Evidence**:",
                "(To be filled when complete)",
                "",
                "---",
                "",
            ])

        return "\n".join(lines)


def main() -> None:
    """CLI entry point."""
    import sys

    if len(sys.argv) < 2:
        print("Usage: todo_importer.py <source> [target]")
        print("Example: todo_importer.py .release_review/v0.8.0/POST_RELEASE_TODO.md")
        sys.exit(1)

    source = sys.argv[1]
    target = (
        sys.argv[2]
        if len(sys.argv) > 2
        else ".post_release_recommendation_fixes/TRACKING.md"
    )

    importer = TodoImporter(source)

    print(f"Parsing: {source}")
    items = importer.parse_source()
    print(f"Found {len(items)} TODO items")

    for item in items:
        print(f"  [{item.priority}] {item.id}: {item.title}")

    print()
    confirm = input(f"Import to {target}? [y/N]: ").strip().lower()
    if confirm == "y":
        path = importer.import_to_tracking(target)
        print(f"Created: {path}")


if __name__ == "__main__":
    main()
