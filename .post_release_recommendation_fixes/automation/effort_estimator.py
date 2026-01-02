#!/usr/bin/env python3
"""Effort Estimator for Post-Release Recommendation Fixes.

Estimates implementation effort based on fix characteristics.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

from versioning import resolve_base_path, resolve_version


@dataclass
class EffortEstimate:
    """Effort estimate for a fix."""

    fix_id: str
    title: str
    priority: str
    estimated_hours: float
    complexity: str  # Low, Medium, High
    confidence: str  # Low, Medium, High
    factors: list[str]
    recommended_agent: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "fix_id": self.fix_id,
            "title": self.title,
            "priority": self.priority,
            "estimated_hours": self.estimated_hours,
            "complexity": self.complexity,
            "confidence": self.confidence,
            "factors": self.factors,
            "recommended_agent": self.recommended_agent,
        }

    def to_markdown_row(self) -> str:
        """Convert to markdown table row."""
        return (
            f"| {self.fix_id} | {self.title[:30]}... | {self.priority} | "
            f"{self.estimated_hours:.1f}h | {self.complexity} | "
            f"{self.recommended_agent} |"
        )


class EffortEstimator:
    """Estimate fix effort based on characteristics."""

    # Complexity factors and their weights
    COMPLEXITY_FACTORS = {
        # File patterns that indicate higher complexity
        "security": 1.5,
        "core": 1.3,
        "orchestrat": 1.4,
        "concurren": 1.5,
        "thread": 1.5,
        "async": 1.3,
        "lock": 1.4,
        "atomic": 1.3,
        # Lower complexity patterns
        "config": 0.8,
        "docs": 0.6,
        "test": 0.7,
        "readme": 0.5,
        "example": 0.6,
    }

    # Effort keywords in descriptions
    EFFORT_KEYWORDS = {
        "small": 0.5,
        "medium": 2.0,
        "large": 4.0,
        "trivial": 0.25,
        "15 min": 0.25,
        "30 min": 0.5,
        "1 hour": 1.0,
        "1-2 hour": 1.5,
        "2-3 hour": 2.5,
        "half day": 4.0,
        "full day": 8.0,
    }

    # Base effort by priority
    PRIORITY_BASE_EFFORT = {
        "High": 2.0,
        "Medium": 1.5,
        "Low": 1.0,
    }

    # Agent recommendations by complexity
    AGENT_BY_COMPLEXITY = {
        "High": "Claude Opus 4.5 or GPT-5.2",
        "Medium": "Claude Opus 4.5",
        "Low": "Gemini 3.0",
    }

    def __init__(self, tracking_path: str | Path | None = None, version: str | None = None) -> None:
        """Initialize estimator.

        Args:
            tracking_path: Path to TRACKING.md
            version: Release version override
        """
        if tracking_path:
            self.tracking_path = Path(tracking_path)
        else:
            base = resolve_base_path(".post_release_recommendation_fixes", resolve_version(version))
            self.tracking_path = base / "TRACKING.md"

    def estimate_fix(
        self,
        fix_id: str,
        title: str,
        priority: str,
        location: str,
        issue: str,
        effort_hint: str,
    ) -> EffortEstimate:
        """Estimate effort for a single fix.

        Args:
            fix_id: Fix ID
            title: Fix title
            priority: Priority (High/Medium/Low)
            location: File paths affected
            issue: Issue description
            effort_hint: Effort hint from original TODO

        Returns:
            EffortEstimate object
        """
        factors = []

        # Start with base effort for priority
        base_effort = self.PRIORITY_BASE_EFFORT.get(priority, 1.5)
        factors.append(f"Base effort for {priority} priority: {base_effort}h")

        # Parse explicit effort hint
        explicit_effort = self._parse_effort_hint(effort_hint)
        if explicit_effort:
            factors.append(f"Explicit effort hint: {explicit_effort}h")
            base_effort = explicit_effort

        # Calculate complexity multiplier from content
        multiplier, complexity_factors = self._calculate_complexity_multiplier(
            title, location, issue
        )
        factors.extend(complexity_factors)

        # Final estimate
        estimated_hours = base_effort * multiplier

        # Determine complexity level
        if estimated_hours < 1.0:
            complexity = "Low"
        elif estimated_hours < 3.0:
            complexity = "Medium"
        else:
            complexity = "High"

        # Determine confidence
        if explicit_effort:
            confidence = "High"
        elif multiplier == 1.0:
            confidence = "Low"
        else:
            confidence = "Medium"

        return EffortEstimate(
            fix_id=fix_id,
            title=title,
            priority=priority,
            estimated_hours=round(estimated_hours, 1),
            complexity=complexity,
            confidence=confidence,
            factors=factors,
            recommended_agent=self.AGENT_BY_COMPLEXITY[complexity],
        )

    def _parse_effort_hint(self, effort_hint: str) -> float | None:
        """Parse effort hint string to hours.

        Args:
            effort_hint: String like "Small (1-2 hours)"

        Returns:
            Hours as float or None if unparseable
        """
        effort_lower = effort_hint.lower()

        for keyword, hours in self.EFFORT_KEYWORDS.items():
            if keyword in effort_lower:
                return hours

        # Try to extract hours directly
        match = re.search(r"(\d+(?:\.\d+)?)\s*(?:hour|hr|h)", effort_lower)
        if match:
            return float(match.group(1))

        return None

    def _calculate_complexity_multiplier(
        self, title: str, location: str, issue: str
    ) -> tuple[float, list[str]]:
        """Calculate complexity multiplier from content.

        Returns:
            Tuple of (multiplier, list of factors)
        """
        combined = f"{title} {location} {issue}".lower()
        multiplier = 1.0
        factors = []

        for pattern, weight in self.COMPLEXITY_FACTORS.items():
            if pattern in combined:
                if weight > 1.0:
                    factors.append(f"+{(weight-1)*100:.0f}% for '{pattern}' pattern")
                else:
                    factors.append(f"-{(1-weight)*100:.0f}% for '{pattern}' pattern")
                multiplier *= weight

        # Count files affected (rough estimate)
        file_count = len(re.findall(r"[a-z_]+\.py", location))
        if file_count > 3:
            factors.append(f"+{(file_count-3)*10}% for {file_count} files")
            multiplier *= 1 + (file_count - 3) * 0.1

        return multiplier, factors

    def estimate_all(self) -> list[EffortEstimate]:
        """Estimate effort for all fixes in tracking file.

        Returns:
            List of EffortEstimate objects
        """
        if not self.tracking_path.exists():
            raise FileNotFoundError(f"Tracking file not found: {self.tracking_path}")

        content = self.tracking_path.read_text()
        estimates = []

        # Parse item details section
        in_details = False
        current_item: dict[str, str] = {}

        for line in content.split("\n"):
            if "## Item Details" in line:
                in_details = True
                continue

            if not in_details:
                continue

            # Parse item header
            match = re.match(r"^###\s+(\d+):\s+(.+)$", line)
            if match:
                # Process previous item
                if current_item:
                    estimates.append(self._estimate_from_dict(current_item))

                current_item = {
                    "id": match.group(1),
                    "title": match.group(2).strip(),
                }
                continue

            # Parse fields
            if current_item:
                if line.startswith("- **Priority**:"):
                    current_item["priority"] = line.split(":", 1)[1].strip()
                elif line.startswith("- **Location**:"):
                    current_item["location"] = line.split(":", 1)[1].strip()
                elif line.startswith("- **Issue**:"):
                    current_item["issue"] = line.split(":", 1)[1].strip()
                elif line.startswith("- **Effort**:"):
                    current_item["effort"] = line.split(":", 1)[1].strip()

        # Don't forget last item
        if current_item:
            estimates.append(self._estimate_from_dict(current_item))

        return estimates

    def _estimate_from_dict(self, item: dict[str, str]) -> EffortEstimate:
        """Create estimate from parsed item dictionary."""
        return self.estimate_fix(
            fix_id=item.get("id", "000"),
            title=item.get("title", "Unknown"),
            priority=item.get("priority", "Medium"),
            location=item.get("location", ""),
            issue=item.get("issue", ""),
            effort_hint=item.get("effort", ""),
        )

    def generate_report(self) -> str:
        """Generate effort estimation report.

        Returns:
            Markdown report
        """
        estimates = self.estimate_all()

        total_hours = sum(e.estimated_hours for e in estimates)

        lines = [
            "# Effort Estimation Report",
            "",
            f"**Total Fixes**: {len(estimates)}",
            f"**Total Estimated Effort**: {total_hours:.1f} hours",
            "",
            "## Summary by Priority",
            "",
            "| Priority | Fixes | Total Hours | Avg Hours |",
            "|----------|-------|-------------|-----------|",
        ]

        for priority in ["High", "Medium", "Low"]:
            p_estimates = [e for e in estimates if e.priority == priority]
            p_total = sum(e.estimated_hours for e in p_estimates)
            p_avg = p_total / len(p_estimates) if p_estimates else 0
            lines.append(
                f"| {priority} | {len(p_estimates)} | {p_total:.1f}h | {p_avg:.1f}h |"
            )

        lines.extend([
            "",
            "## All Estimates",
            "",
            "| ID | Title | Priority | Est. Hours | Complexity | Agent |",
            "|----|-------|----------|------------|------------|-------|",
        ])

        for e in sorted(estimates, key=lambda x: x.estimated_hours, reverse=True):
            title_short = e.title[:25] + "..." if len(e.title) > 25 else e.title
            lines.append(
                f"| {e.fix_id} | {title_short} | {e.priority} | "
                f"{e.estimated_hours:.1f}h | {e.complexity} | "
                f"{e.recommended_agent.split()[0]} |"
            )

        return "\n".join(lines)


def main() -> None:
    """CLI entry point."""
    import sys

    args = sys.argv[1:]
    version = None
    if "--version" in args:
        idx = args.index("--version")
        if idx + 1 >= len(args):
            print("Usage: effort_estimator.py --version <version> <command> [tracking_path]")
            sys.exit(1)
        version = args[idx + 1]
        del args[idx:idx + 2]

    if len(args) < 1:
        print("Usage: effort_estimator.py <command> [tracking_path]")
        print("Commands:")
        print("  report  - Generate effort estimation report")
        print("  total   - Show total estimated effort")
        sys.exit(1)

    command = args[0]
    tracking_path = args[1] if len(args) > 1 else None

    estimator = EffortEstimator(tracking_path, version=version)

    if command == "report":
        print(estimator.generate_report())

    elif command == "total":
        try:
            estimates = estimator.estimate_all()
            total = sum(e.estimated_hours for e in estimates)
            print(f"Total estimated effort: {total:.1f} hours")
            print(f"Fixes: {len(estimates)}")
        except FileNotFoundError as e:
            print(f"Error: {e}")
            sys.exit(1)

    else:
        print(f"Unknown command: {command}")
        sys.exit(1)


if __name__ == "__main__":
    main()
