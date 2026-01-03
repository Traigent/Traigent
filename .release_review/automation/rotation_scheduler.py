#!/usr/bin/env python3
"""Rotation Scheduler for Release Review Protocol.

Generates and manages model rotation schedules across releases.
Ensures different models review different component categories each round.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from itertools import cycle
from pathlib import Path
from typing import Any

from traigent.utils.secure_path import safe_read_text, safe_write_text, validate_path


@dataclass
class Assignment:
    """Single model assignment for a category."""

    category: str
    primary: str
    secondary: str
    spot_check: str

    def to_dict(self) -> dict[str, str]:
        """Convert to dictionary."""
        return {
            "category": self.category,
            "primary": self.primary,
            "secondary": self.secondary,
            "spot_check": self.spot_check,
        }


@dataclass
class RotationSchedule:
    """Complete rotation schedule for a release review."""

    round_number: int
    version: str
    generated_at: str
    assignments: list[Assignment] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "round_number": self.round_number,
            "version": self.version,
            "generated_at": self.generated_at,
            "assignments": [a.to_dict() for a in self.assignments],
        }

    def to_markdown(self) -> str:
        """Generate markdown table."""
        lines = [
            f"# Rotation Schedule: {self.version} (Round {self.round_number})",
            "",
            f"Generated: {self.generated_at}",
            "",
            "| Category | Primary | Secondary | Spot-Check |",
            "|----------|---------|-----------|------------|",
        ]

        for a in self.assignments:
            lines.append(f"| {a.category} | {a.primary} | {a.secondary} | {a.spot_check} |")

        return "\n".join(lines)

    def get_assignment(self, category: str) -> Assignment | None:
        """Get assignment for a specific category."""
        for a in self.assignments:
            if a.category == category:
                return a
        return None


class RotationScheduler:
    """Generate and manage model rotation schedules."""

    # Default component categories
    DEFAULT_CATEGORIES = [
        "Security/Core",
        "Integrations",
        "Packaging/CI",
        "Docs/Examples",
    ]

    # Default models (in capability order)
    DEFAULT_MODELS = [
        "Claude Opus 4.5",
        "GPT-5.2",
        "Gemini 3.0",
    ]

    # Model capability tiers
    TIER_1_MODELS = {"Claude Opus 4.5", "GPT-5.2"}
    TIER_2_MODELS = {"Gemini 3.0"}

    # Categories that require Tier 1 primary reviewer
    CRITICAL_CATEGORIES = {"Security/Core"}

    def __init__(
        self,
        models: list[str] | None = None,
        categories: list[str] | None = None,
        history_path: str | Path | None = None,
    ) -> None:
        """Initialize rotation scheduler.

        Args:
            models: List of model names. Defaults to DEFAULT_MODELS.
            categories: List of categories. Defaults to DEFAULT_CATEGORIES.
            history_path: Path to rotation history file.
        """
        self.models = models or self.DEFAULT_MODELS.copy()
        self.categories = categories or self.DEFAULT_CATEGORIES.copy()

        if history_path is None:
            history_path = Path(".release_review/rotation_history.json")
        self._base_dir = Path.cwd()
        self.history_path = validate_path(history_path, self._base_dir)

    def get_schedule(
        self,
        round_number: int,
        version: str = "unknown",
    ) -> RotationSchedule:
        """Generate schedule for a specific round.

        Uses Latin square rotation to ensure:
        - Each model gets each role (primary/secondary/spot-check)
        - Each model reviews each category across rounds
        - Tier 1 models handle critical categories

        Args:
            round_number: Round number (1-indexed)
            version: Release version string

        Returns:
            RotationSchedule for this round
        """
        n_models = len(self.models)
        n_categories = len(self.categories)

        # Validate
        if n_models < 3:
            raise ValueError("Need at least 3 models for rotation")

        assignments = []

        for cat_idx, category in enumerate(self.categories):
            # Calculate rotation offset based on round and category
            # This creates a Latin square pattern
            offset = (round_number - 1 + cat_idx) % n_models

            # Get rotated model indices
            primary_idx = offset % n_models
            secondary_idx = (offset + 1) % n_models
            spot_check_idx = (offset + 2) % n_models

            primary = self.models[primary_idx]
            secondary = self.models[secondary_idx]
            spot_check = self.models[spot_check_idx]

            # Enforce Tier 1 constraint for critical categories
            if category in self.CRITICAL_CATEGORIES:
                if primary not in self.TIER_1_MODELS:
                    # Swap primary with first Tier 1 model in the rotation
                    for i, model in enumerate([primary, secondary, spot_check]):
                        if model in self.TIER_1_MODELS:
                            if i == 1:
                                primary, secondary = secondary, primary
                            elif i == 2:
                                primary, spot_check = spot_check, primary
                            break

            assignments.append(Assignment(
                category=category,
                primary=primary,
                secondary=secondary,
                spot_check=spot_check,
            ))

        return RotationSchedule(
            round_number=round_number,
            version=version,
            generated_at=datetime.now().isoformat() + "Z",
            assignments=assignments,
        )

    def rotate_from(
        self,
        previous_version: str,
        new_version: str | None = None,
    ) -> RotationSchedule:
        """Generate rotation based on previous release.

        Args:
            previous_version: Previous release version
            new_version: Target release version for the new schedule

        Returns:
            New RotationSchedule with rotated assignments
        """
        history = self.load_history()

        # Find previous round number
        prev_round = 0
        for entry in history:
            if entry.get("version") == previous_version:
                prev_round = entry.get("round_number", 0)
                break

        # Generate next round
        next_round = prev_round + 1
        if new_version is None:
            raise ValueError("new_version is required for rotate_from")
        return self.get_schedule(next_round, new_version)

    def save_schedule(self, schedule: RotationSchedule) -> Path:
        """Save schedule to history.

        Args:
            schedule: Schedule to save

        Returns:
            Path to history file
        """
        history = self.load_history()
        history.append(schedule.to_dict())

        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        safe_write_text(
            self.history_path,
            json.dumps(history, indent=2),
            self._base_dir,
        )
        self.write_markdown(schedule)

        return self.history_path

    def write_markdown(self, schedule: RotationSchedule) -> Path:
        """Write or update the versioned rotation history markdown.

        Args:
            schedule: Schedule to render

        Returns:
            Path to the markdown file
        """
        version_dir = validate_path(
            Path(".release_review") / schedule.version,
            self._base_dir,
        )
        version_dir.mkdir(parents=True, exist_ok=True)
        md_path = validate_path(
            version_dir / "ROTATION_HISTORY.md",
            self._base_dir,
        )

        marker_start = "<!-- BEGIN AUTO-GENERATED ROTATION -->"
        marker_end = "<!-- END AUTO-GENERATED ROTATION -->"
        auto_block = schedule.to_markdown()

        if md_path.exists():
            content = safe_read_text(md_path, self._base_dir)
            if marker_start in content and marker_end in content:
                pre, rest = content.split(marker_start, 1)
                _, post = rest.split(marker_end, 1)
                new_content = (
                    pre
                    + marker_start
                    + "\n"
                    + auto_block
                    + "\n"
                    + marker_end
                    + post
                )
            else:
                new_content = (
                    content.rstrip()
                    + "\n\n"
                    + marker_start
                    + "\n"
                    + auto_block
                    + "\n"
                    + marker_end
                    + "\n"
                )
        else:
            new_content = (
                f"# Rotation History: {schedule.version}\n\n"
                f"{marker_start}\n{auto_block}\n{marker_end}\n\n"
                "## Component Mapping\n\n"
                "(Fill in or link the component mapping for this release.)\n"
            )

        safe_write_text(md_path, new_content, self._base_dir)
        return md_path

    def load_history(self) -> list[dict[str, Any]]:
        """Load rotation history.

        Returns:
            List of historical schedules
        """
        if not self.history_path.exists():
            return []

        try:
            return json.loads(safe_read_text(self.history_path, self._base_dir))
        except json.JSONDecodeError:
            return []

    def get_model_stats(self) -> dict[str, dict[str, int]]:
        """Analyze model assignment distribution across history.

        Returns:
            Stats per model showing how many times they had each role
        """
        history = self.load_history()
        stats: dict[str, dict[str, int]] = {
            model: {"primary": 0, "secondary": 0, "spot_check": 0}
            for model in self.models
        }

        for entry in history:
            for assignment in entry.get("assignments", []):
                primary = assignment.get("primary")
                secondary = assignment.get("secondary")
                spot_check = assignment.get("spot_check")

                if primary in stats:
                    stats[primary]["primary"] += 1
                if secondary in stats:
                    stats[secondary]["secondary"] += 1
                if spot_check in stats:
                    stats[spot_check]["spot_check"] += 1

        return stats

    def generate_comparison(
        self,
        schedule_a: RotationSchedule,
        schedule_b: RotationSchedule,
    ) -> str:
        """Generate comparison between two schedules.

        Args:
            schedule_a: First schedule
            schedule_b: Second schedule

        Returns:
            Markdown comparison
        """
        lines = [
            "# Schedule Comparison",
            "",
            f"Comparing Round {schedule_a.round_number} vs Round {schedule_b.round_number}",
            "",
            "| Category | Round A Primary | Round B Primary | Changed? |",
            "|----------|-----------------|-----------------|----------|",
        ]

        for cat in self.categories:
            a_assign = schedule_a.get_assignment(cat)
            b_assign = schedule_b.get_assignment(cat)

            a_primary = a_assign.primary if a_assign else "N/A"
            b_primary = b_assign.primary if b_assign else "N/A"
            changed = "Yes" if a_primary != b_primary else "No"

            lines.append(f"| {cat} | {a_primary} | {b_primary} | {changed} |")

        return "\n".join(lines)

    def map_components_to_categories(
        self,
        components: list[dict[str, Any]],
    ) -> dict[str, str]:
        """Map specific components to categories based on patterns.

        Args:
            components: List of component dicts with 'name' and 'scope' keys

        Returns:
            Mapping of component name to category
        """
        mapping = {}

        # Define patterns for categorization
        patterns = {
            "Security/Core": [
                "security", "core", "orchestr", "config", "invoker",
                "optimizer", "storage", "persist",
            ],
            "Integrations": [
                "integrat", "adapter", "plugin", "connect",
            ],
            "Packaging/CI": [
                "packag", "ci", "workflow", "test", "script", "tool",
                "pyproject", "manifest", "requirement",
            ],
            "Docs/Examples": [
                "doc", "example", "readme", "walkthrough", "use-case",
                "playground", "visual",
            ],
        }

        for component in components:
            name = component.get("name", "").lower()
            scope = component.get("scope", "").lower()
            combined = f"{name} {scope}"

            # Find matching category
            matched_category = "Docs/Examples"  # Default fallback
            for category, keywords in patterns.items():
                if any(kw in combined for kw in keywords):
                    matched_category = category
                    break

            mapping[component.get("name", "")] = matched_category

        return mapping


def main() -> None:
    """CLI entry point."""
    import sys

    scheduler = RotationScheduler()

    if len(sys.argv) < 2:
        print("Usage: rotation_scheduler.py <command> [args]")
        print("Commands:")
        print("  generate <round> [version]  - Generate schedule for round N")
        print("  rotate <prev_version> <new_version> - Rotate into a new version")
        print("  compare <round1> <round2>   - Compare two rounds")
        print("  stats                       - Show model assignment stats")
        print("  history                     - Show rotation history")
        sys.exit(1)

    command = sys.argv[1]

    if command == "generate":
        if len(sys.argv) < 3:
            print("Usage: rotation_scheduler.py generate <round> [version]")
            sys.exit(1)
        round_num = int(sys.argv[2])
        version = sys.argv[3] if len(sys.argv) > 3 else f"v{round_num}.0.0"

        schedule = scheduler.get_schedule(round_num, version)
        print(schedule.to_markdown())
        print()

        save = input("Save to history? [y/N]: ").strip().lower()
        if save == "y":
            path = scheduler.save_schedule(schedule)
            print(f"Saved to: {path}")

    elif command == "rotate":
        if len(sys.argv) < 4:
            print("Usage: rotation_scheduler.py rotate <prev_version> <new_version>")
            sys.exit(1)
        prev_version = sys.argv[2]
        new_version = sys.argv[3]

        schedule = scheduler.rotate_from(prev_version, new_version)
        print(schedule.to_markdown())
        print()

        save = input("Save to history? [y/N]: ").strip().lower()
        if save == "y":
            path = scheduler.save_schedule(schedule)
            print(f"Saved to: {path}")

    elif command == "compare":
        if len(sys.argv) < 4:
            print("Usage: rotation_scheduler.py compare <round1> <round2>")
            sys.exit(1)
        round1 = int(sys.argv[2])
        round2 = int(sys.argv[3])

        schedule1 = scheduler.get_schedule(round1, f"Round {round1}")
        schedule2 = scheduler.get_schedule(round2, f"Round {round2}")

        print(scheduler.generate_comparison(schedule1, schedule2))

    elif command == "stats":
        stats = scheduler.get_model_stats()
        if not any(sum(s.values()) > 0 for s in stats.values()):
            print("No history found. Generate and save some schedules first.")
        else:
            print("Model Assignment Statistics:")
            print()
            print("| Model | Primary | Secondary | Spot-Check |")
            print("|-------|---------|-----------|------------|")
            for model, counts in stats.items():
                print(
                    f"| {model} | {counts['primary']} | "
                    f"{counts['secondary']} | {counts['spot_check']} |"
                )

    elif command == "history":
        history = scheduler.load_history()
        if not history:
            print("No rotation history found.")
        else:
            print(f"Found {len(history)} historical schedules:")
            for entry in history:
                print(f"  - Round {entry.get('round_number')}: "
                      f"{entry.get('version')} ({entry.get('generated_at', 'unknown')})")

    else:
        print(f"Unknown command: {command}")
        sys.exit(1)


if __name__ == "__main__":
    main()
