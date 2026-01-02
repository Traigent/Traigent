#!/usr/bin/env python3
"""Metrics Tracker for Release Review Protocol.

Tracks agent effectiveness across releases.
Used for post-release analysis and continuous improvement.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from traigent.utils.secure_path import validate_path

@dataclass
class AgentMetrics:
    """Metrics for a single agent across reviews."""

    agent_id: str
    model: str
    components_reviewed: int = 0
    issues_found: int = 0
    issues_confirmed: int = 0  # Issues that were real (not false positives)
    issues_missed: int = 0  # Issues found by others that this agent missed
    fixes_applied: int = 0
    total_review_time_minutes: float = 0.0
    spot_check_matches: int = 0
    spot_check_mismatches: int = 0

    @property
    def precision(self) -> float:
        """Precision: confirmed issues / total issues found."""
        if self.issues_found == 0:
            return 0.0
        return self.issues_confirmed / self.issues_found

    @property
    def false_positive_rate(self) -> float:
        """Rate of false positives."""
        if self.issues_found == 0:
            return 0.0
        return (self.issues_found - self.issues_confirmed) / self.issues_found

    @property
    def avg_review_time(self) -> float:
        """Average time per component in minutes."""
        if self.components_reviewed == 0:
            return 0.0
        return self.total_review_time_minutes / self.components_reviewed

    @property
    def spot_check_accuracy(self) -> float:
        """Accuracy in spot checks."""
        total = self.spot_check_matches + self.spot_check_mismatches
        if total == 0:
            return 1.0  # No spot checks = assume good
        return self.spot_check_matches / total

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "agent_id": self.agent_id,
            "model": self.model,
            "components_reviewed": self.components_reviewed,
            "issues_found": self.issues_found,
            "issues_confirmed": self.issues_confirmed,
            "issues_missed": self.issues_missed,
            "fixes_applied": self.fixes_applied,
            "total_review_time_minutes": self.total_review_time_minutes,
            "spot_check_matches": self.spot_check_matches,
            "spot_check_mismatches": self.spot_check_mismatches,
            "precision": self.precision,
            "false_positive_rate": self.false_positive_rate,
            "avg_review_time": self.avg_review_time,
            "spot_check_accuracy": self.spot_check_accuracy,
        }


@dataclass
class ReleaseMetrics:
    """Metrics for an entire release review."""

    version: str
    started_at: str
    completed_at: str | None = None
    total_components: int = 0
    components_approved: int = 0
    total_issues_found: int = 0
    critical_issues: int = 0
    high_issues: int = 0
    medium_issues: int = 0
    low_issues: int = 0
    fixes_applied: int = 0
    blocking_conflicts: int = 0
    conflicts_resolved: int = 0
    human_escalations: int = 0
    agent_metrics: dict[str, AgentMetrics] = field(default_factory=dict)

    @property
    def completion_rate(self) -> float:
        """Completion rate of review."""
        if self.total_components == 0:
            return 0.0
        return self.components_approved / self.total_components

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "version": self.version,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "total_components": self.total_components,
            "components_approved": self.components_approved,
            "completion_rate": self.completion_rate,
            "issues": {
                "total": self.total_issues_found,
                "critical": self.critical_issues,
                "high": self.high_issues,
                "medium": self.medium_issues,
                "low": self.low_issues,
            },
            "fixes_applied": self.fixes_applied,
            "conflicts": {
                "blocking": self.blocking_conflicts,
                "resolved": self.conflicts_resolved,
            },
            "human_escalations": self.human_escalations,
            "agents": {
                agent_id: metrics.to_dict()
                for agent_id, metrics in self.agent_metrics.items()
            },
        }


class MetricsTracker:
    """Track and analyze release review metrics."""

    def __init__(self, base_path: str | Path | None = None) -> None:
        """Initialize metrics tracker.

        Args:
            base_path: Base path for metrics storage
        """
        if base_path is None:
            base_path = Path(".release_review/metrics")
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def get_metrics_file(self, version: str) -> Path:
        """Get path to metrics file for a version."""
        return self.base_path / f"{version}_metrics.json"

    def load_release(self, version: str) -> ReleaseMetrics | None:
        """Load metrics for a release.

        Args:
            version: Release version

        Returns:
            ReleaseMetrics or None if not found
        """
        metrics_file = self.get_metrics_file(version)
        if not metrics_file.exists():
            return None

        try:
            data = json.loads(metrics_file.read_text())
            release = ReleaseMetrics(
                version=data["version"],
                started_at=data["started_at"],
                completed_at=data.get("completed_at"),
                total_components=data.get("total_components", 0),
                components_approved=data.get("components_approved", 0),
                total_issues_found=data.get("issues", {}).get("total", 0),
                critical_issues=data.get("issues", {}).get("critical", 0),
                high_issues=data.get("issues", {}).get("high", 0),
                medium_issues=data.get("issues", {}).get("medium", 0),
                low_issues=data.get("issues", {}).get("low", 0),
                fixes_applied=data.get("fixes_applied", 0),
                blocking_conflicts=data.get("conflicts", {}).get("blocking", 0),
                conflicts_resolved=data.get("conflicts", {}).get("resolved", 0),
                human_escalations=data.get("human_escalations", 0),
            )

            # Load agent metrics
            for agent_id, agent_data in data.get("agents", {}).items():
                release.agent_metrics[agent_id] = AgentMetrics(
                    agent_id=agent_id,
                    model=agent_data.get("model", "unknown"),
                    components_reviewed=agent_data.get("components_reviewed", 0),
                    issues_found=agent_data.get("issues_found", 0),
                    issues_confirmed=agent_data.get("issues_confirmed", 0),
                    issues_missed=agent_data.get("issues_missed", 0),
                    fixes_applied=agent_data.get("fixes_applied", 0),
                    total_review_time_minutes=agent_data.get("total_review_time_minutes", 0),
                    spot_check_matches=agent_data.get("spot_check_matches", 0),
                    spot_check_mismatches=agent_data.get("spot_check_mismatches", 0),
                )

            return release
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error loading metrics: {e}")
            return None

    def save_release(self, release: ReleaseMetrics) -> Path:
        """Save metrics for a release.

        Args:
            release: ReleaseMetrics to save

        Returns:
            Path to saved file
        """
        metrics_file = self.get_metrics_file(release.version)
        metrics_file = validate_path(metrics_file, self.base_path, must_exist=False)
        metrics_file.write_text(json.dumps(release.to_dict(), indent=2))
        return metrics_file

    def create_release(self, version: str, total_components: int) -> ReleaseMetrics:
        """Create new release metrics.

        Args:
            version: Release version
            total_components: Total components to review

        Returns:
            New ReleaseMetrics instance
        """
        release = ReleaseMetrics(
            version=version,
            started_at=datetime.now().isoformat() + "Z",
            total_components=total_components,
        )
        self.save_release(release)
        return release

    def record_agent_review(
        self,
        version: str,
        agent_id: str,
        model: str,
        issues_found: int = 0,
        review_time_minutes: float = 0.0,
    ) -> None:
        """Record an agent's review of a component.

        Args:
            version: Release version
            agent_id: Agent identifier
            model: Model name
            issues_found: Number of issues found
            review_time_minutes: Time spent on review
        """
        release = self.load_release(version)
        if not release:
            raise ValueError(f"Release {version} not found")

        if agent_id not in release.agent_metrics:
            release.agent_metrics[agent_id] = AgentMetrics(
                agent_id=agent_id,
                model=model,
            )

        agent = release.agent_metrics[agent_id]
        agent.components_reviewed += 1
        agent.issues_found += issues_found
        agent.total_review_time_minutes += review_time_minutes

        release.total_issues_found += issues_found

        self.save_release(release)

    def record_spot_check(
        self,
        version: str,
        agent_id: str,
        matched: bool,
    ) -> None:
        """Record result of a spot check.

        Args:
            version: Release version
            agent_id: Agent identifier
            matched: Whether spot check matched agent's claims
        """
        release = self.load_release(version)
        if not release:
            raise ValueError(f"Release {version} not found")

        if agent_id not in release.agent_metrics:
            raise ValueError(f"Agent {agent_id} not found in release {version}")

        if matched:
            release.agent_metrics[agent_id].spot_check_matches += 1
        else:
            release.agent_metrics[agent_id].spot_check_mismatches += 1

        self.save_release(release)

    def complete_release(self, version: str) -> ReleaseMetrics:
        """Mark a release as complete.

        Args:
            version: Release version

        Returns:
            Updated ReleaseMetrics
        """
        release = self.load_release(version)
        if not release:
            raise ValueError(f"Release {version} not found")

        release.completed_at = datetime.now().isoformat() + "Z"
        self.save_release(release)
        return release

    def generate_summary(self, version: str) -> str:
        """Generate a human-readable summary.

        Args:
            version: Release version

        Returns:
            Markdown summary
        """
        release = self.load_release(version)
        if not release:
            return f"No metrics found for {version}"

        lines = [
            f"# Release Review Metrics: {release.version}",
            "",
            f"**Started**: {release.started_at}",
            f"**Completed**: {release.completed_at or 'In Progress'}",
            "",
            "## Overview",
            "",
            f"- Components: {release.components_approved}/{release.total_components} ({release.completion_rate:.0%})",
            f"- Issues Found: {release.total_issues_found}",
            f"  - Critical: {release.critical_issues}",
            f"  - High: {release.high_issues}",
            f"  - Medium: {release.medium_issues}",
            f"  - Low: {release.low_issues}",
            f"- Fixes Applied: {release.fixes_applied}",
            f"- Human Escalations: {release.human_escalations}",
            "",
            "## Agent Performance",
            "",
            "| Agent | Model | Components | Issues | Precision | Avg Time |",
            "|-------|-------|------------|--------|-----------|----------|",
        ]

        for agent_id, agent in release.agent_metrics.items():
            lines.append(
                f"| {agent_id} | {agent.model} | {agent.components_reviewed} | "
                f"{agent.issues_found} | {agent.precision:.0%} | {agent.avg_review_time:.1f}m |"
            )

        return "\n".join(lines)

    def list_releases(self) -> list[str]:
        """List all tracked releases.

        Returns:
            List of version strings
        """
        return sorted(
            f.stem.replace("_metrics", "")
            for f in self.base_path.glob("*_metrics.json")
        )


def main() -> None:
    """CLI entry point."""
    import sys

    tracker = MetricsTracker()

    if len(sys.argv) < 2:
        print("Usage: metrics.py <command> [args]")
        print("Commands:")
        print("  list                  - List all tracked releases")
        print("  show <version>        - Show metrics for a release")
        print("  summary <version>     - Generate summary report")
        print("  create <version> <n>  - Create new release with n components")
        sys.exit(1)

    command = sys.argv[1]

    if command == "list":
        releases = tracker.list_releases()
        if not releases:
            print("No releases tracked yet")
        else:
            print("Tracked releases:")
            for r in releases:
                print(f"  - {r}")

    elif command == "show":
        if len(sys.argv) < 3:
            print("Usage: metrics.py show <version>")
            sys.exit(1)
        version = sys.argv[2]
        release = tracker.load_release(version)
        if release:
            print(json.dumps(release.to_dict(), indent=2))
        else:
            print(f"No metrics found for {version}")

    elif command == "summary":
        if len(sys.argv) < 3:
            print("Usage: metrics.py summary <version>")
            sys.exit(1)
        version = sys.argv[2]
        print(tracker.generate_summary(version))

    elif command == "create":
        if len(sys.argv) < 4:
            print("Usage: metrics.py create <version> <total_components>")
            sys.exit(1)
        version = sys.argv[2]
        total = int(sys.argv[3])
        release = tracker.create_release(version, total)
        print(f"Created metrics for {version} with {total} components")
        print(f"Saved to: {tracker.get_metrics_file(version)}")

    else:
        print(f"Unknown command: {command}")
        sys.exit(1)


if __name__ == "__main__":
    main()
