#!/usr/bin/env python3
"""Progress Tracker for Post-Release Recommendation Fixes.

Tracks fix progress across sessions and generates reports.
"""

from __future__ import annotations

import fcntl
import json
import os
import re
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator


@dataclass
class FixProgress:
    """Progress record for a single fix."""

    id: str
    title: str
    priority: str
    status: str
    owner: str = ""
    branch: str = ""
    started_at: str = ""
    completed_at: str = ""
    commit_sha: str = ""
    tests_run: str = ""
    evidence: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "title": self.title,
            "priority": self.priority,
            "status": self.status,
            "owner": self.owner,
            "branch": self.branch,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "commit_sha": self.commit_sha,
            "tests_run": self.tests_run,
            "evidence": self.evidence,
        }


@dataclass
class SessionProgress:
    """Progress for a single fix session."""

    session_date: str
    started_at: str
    ended_at: str = ""
    fixes_targeted: list[str] = field(default_factory=list)
    fixes_completed: list[str] = field(default_factory=list)
    fixes_in_progress: list[str] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "session_date": self.session_date,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "fixes_targeted": self.fixes_targeted,
            "fixes_completed": self.fixes_completed,
            "fixes_in_progress": self.fixes_in_progress,
            "blockers": self.blockers,
        }


class ProgressTracker:
    """Track fix progress across sessions."""

    def __init__(
        self,
        base_path: str | Path = ".post_release_recommendation_fixes",
    ) -> None:
        """Initialize tracker.

        Args:
            base_path: Base path for fix workflow
        """
        self.base_path = Path(base_path)
        self.tracking_path = self.base_path / "TRACKING.md"
        self.history_path = self.base_path / "progress_history.json"
        self.lock_path = self.base_path / ".tracking.lock"

    @contextmanager
    def _file_lock(self, timeout: float = 30.0) -> Iterator[None]:
        """Acquire exclusive file lock to prevent concurrent modifications.

        Args:
            timeout: Maximum time to wait for lock (seconds)

        Yields:
            None when lock is acquired

        Raises:
            TimeoutError: If lock cannot be acquired within timeout
        """
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock_fd = os.open(str(self.lock_path), os.O_CREAT | os.O_RDWR)
        try:
            # Try to acquire lock with timeout
            import time

            start_time = time.time()
            while True:
                try:
                    fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if time.time() - start_time > timeout:
                        raise TimeoutError(
                            f"Could not acquire lock on {self.lock_path} "
                            f"within {timeout}s. Another process may be updating."
                        )
                    time.sleep(0.1)
            yield
        finally:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            os.close(lock_fd)

    def start_session(self, targeted_fixes: list[str]) -> SessionProgress:
        """Start a new fix session.

        Args:
            targeted_fixes: List of fix IDs targeted for this session

        Returns:
            SessionProgress object
        """
        session_date = datetime.now().strftime("%Y%m%d")
        started_at = datetime.now().isoformat() + "Z"

        session = SessionProgress(
            session_date=session_date,
            started_at=started_at,
            fixes_targeted=targeted_fixes,
        )

        # Create session directory
        session_dir = self.base_path / "sessions" / session_date
        session_dir.mkdir(parents=True, exist_ok=True)
        (session_dir / "artifacts").mkdir(exist_ok=True)

        # Write initial progress file
        self._write_session_progress(session)

        return session

    def update_fix_status(
        self,
        fix_id: str,
        status: str,
        owner: str = "",
        branch: str = "",
        evidence: str = "",
    ) -> None:
        """Update status of a fix in TRACKING.md.

        Args:
            fix_id: Fix ID (e.g., "001")
            status: New status (Pending, In Progress, Complete, Blocked, Reverted)
            owner: Agent/model working on the fix
            branch: Git branch for the fix
            evidence: Evidence of completion
        """
        if not self.tracking_path.exists():
            raise FileNotFoundError(f"Tracking file not found: {self.tracking_path}")

        with self._file_lock():
            content = self.tracking_path.read_text()
            lines = content.split("\n")
            updated_lines = []
            found = False

            # More robust pattern matching for fix ID
            fix_pattern = re.compile(rf"^\|\s*{re.escape(fix_id)}\s*\|")

            for line in lines:
                # Find and update the row for this fix ID
                if fix_pattern.match(line):
                    found = True
                    # Parse existing row
                    parts = line.split("|")
                    if len(parts) >= 7:
                        parts[4] = f" {status} "
                        if owner:
                            parts[5] = f" {owner} "
                        if evidence:
                            parts[6] = f" {evidence} "
                        line = "|".join(parts)
                updated_lines.append(line)

            if not found:
                raise ValueError(f"Fix ID {fix_id} not found in tracking file")

            self.tracking_path.write_text("\n".join(updated_lines))

    def mark_fix_complete(
        self,
        fix_id: str,
        commit_sha: str,
        tests_run: str,
        owner: str,
    ) -> None:
        """Mark a fix as complete with evidence.

        Args:
            fix_id: Fix ID
            commit_sha: Commit SHA of the fix
            tests_run: Test command and results
            owner: Agent that completed the fix
        """
        timestamp = datetime.now().isoformat() + "Z"
        evidence = f"Commit: {commit_sha[:7]} | Tests: {tests_run} | {timestamp}"

        self.update_fix_status(
            fix_id=fix_id,
            status="Complete",
            owner=owner,
            evidence=evidence,
        )

        # Update history
        self._add_to_history(fix_id, "Complete", evidence)

    def end_session(
        self,
        session: SessionProgress,
        completed: list[str],
        in_progress: list[str],
        blockers: list[str],
    ) -> None:
        """End a fix session and record progress.

        Args:
            session: SessionProgress object
            completed: List of completed fix IDs
            in_progress: List of in-progress fix IDs
            blockers: List of blocker descriptions
        """
        session.ended_at = datetime.now().isoformat() + "Z"
        session.fixes_completed = completed
        session.fixes_in_progress = in_progress
        session.blockers = blockers

        self._write_session_progress(session)
        self._add_session_to_history(session)

    def get_pending_fixes(self) -> list[str]:
        """Get list of pending fix IDs.

        Returns:
            List of fix IDs with status "Pending"
        """
        if not self.tracking_path.exists():
            return []

        content = self.tracking_path.read_text()
        pending = []

        for line in content.split("\n"):
            if "| Pending |" in line:
                match = re.match(r"\|\s*(\d+)\s*\|", line)
                if match:
                    pending.append(match.group(1))

        return pending

    def get_stats(self) -> dict[str, Any]:
        """Get overall progress statistics.

        Returns:
            Dictionary with stats
        """
        if not self.tracking_path.exists():
            return {"error": "Tracking file not found"}

        content = self.tracking_path.read_text()

        stats = {
            "total": 0,
            "pending": 0,
            "in_progress": 0,
            "complete": 0,
            "blocked": 0,
            "by_priority": {
                "High": {"total": 0, "complete": 0},
                "Medium": {"total": 0, "complete": 0},
                "Low": {"total": 0, "complete": 0},
            },
        }

        current_priority = "Medium"

        for line in content.split("\n"):
            if "## High Priority" in line:
                current_priority = "High"
            elif "## Medium Priority" in line:
                current_priority = "Medium"
            elif "## Low Priority" in line:
                current_priority = "Low"
            elif "## Item Details" in line:
                break

            # Count items in table rows
            if re.match(r"\|\s*\d+\s*\|", line):
                stats["total"] += 1
                stats["by_priority"][current_priority]["total"] += 1

                if "| Pending |" in line:
                    stats["pending"] += 1
                elif "| In Progress |" in line:
                    stats["in_progress"] += 1
                elif "| Complete |" in line:
                    stats["complete"] += 1
                    stats["by_priority"][current_priority]["complete"] += 1
                elif "| Blocked |" in line:
                    stats["blocked"] += 1

        return stats

    def generate_report(self) -> str:
        """Generate a progress report.

        Returns:
            Markdown report
        """
        stats = self.get_stats()
        timestamp = datetime.now().isoformat() + "Z"

        if "error" in stats:
            return f"# Progress Report\n\nError: {stats['error']}"

        total = stats["total"]
        complete = stats["complete"]
        pct = (complete / total * 100) if total > 0 else 0

        lines = [
            "# Fix Progress Report",
            "",
            f"**Generated**: {timestamp}",
            f"**Overall Progress**: {complete}/{total} ({pct:.1f}%)",
            "",
            "## Summary",
            "",
            "| Status | Count |",
            "|--------|-------|",
            f"| Pending | {stats['pending']} |",
            f"| In Progress | {stats['in_progress']} |",
            f"| Complete | {stats['complete']} |",
            f"| Blocked | {stats['blocked']} |",
            "",
            "## By Priority",
            "",
            "| Priority | Complete | Total | Progress |",
            "|----------|----------|-------|----------|",
        ]

        for priority in ["High", "Medium", "Low"]:
            p_stats = stats["by_priority"][priority]
            p_total = p_stats["total"]
            p_complete = p_stats["complete"]
            p_pct = (p_complete / p_total * 100) if p_total > 0 else 0
            lines.append(
                f"| {priority} | {p_complete} | {p_total} | {p_pct:.1f}% |"
            )

        return "\n".join(lines)

    def _write_session_progress(self, session: SessionProgress) -> None:
        """Write session progress to file."""
        session_dir = self.base_path / "sessions" / session.session_date
        progress_file = session_dir / "PROGRESS.md"

        lines = [
            f"# Session Progress: {session.session_date}",
            "",
            f"**Started**: {session.started_at}",
            f"**Ended**: {session.ended_at or 'In progress'}",
            "",
            "## Targeted Fixes",
            "",
        ]

        for fix_id in session.fixes_targeted:
            lines.append(f"- [ ] {fix_id}")

        lines.extend([
            "",
            "## Completed",
            "",
        ])

        for fix_id in session.fixes_completed:
            lines.append(f"- [x] {fix_id}")

        lines.extend([
            "",
            "## In Progress",
            "",
        ])

        for fix_id in session.fixes_in_progress:
            lines.append(f"- [ ] {fix_id}")

        if session.blockers:
            lines.extend([
                "",
                "## Blockers",
                "",
            ])
            for blocker in session.blockers:
                lines.append(f"- {blocker}")

        progress_file.write_text("\n".join(lines))

    def _add_to_history(self, fix_id: str, status: str, evidence: str) -> None:
        """Add entry to progress history."""
        history = self._load_history()

        entry = {
            "fix_id": fix_id,
            "status": status,
            "evidence": evidence,
            "timestamp": datetime.now().isoformat() + "Z",
        }

        if "fixes" not in history:
            history["fixes"] = []

        history["fixes"].append(entry)
        self._save_history(history)

    def _add_session_to_history(self, session: SessionProgress) -> None:
        """Add session to history."""
        history = self._load_history()

        if "sessions" not in history:
            history["sessions"] = []

        history["sessions"].append(session.to_dict())
        self._save_history(history)

    def _load_history(self) -> dict[str, Any]:
        """Load progress history."""
        if not self.history_path.exists():
            return {}

        try:
            return json.loads(self.history_path.read_text())
        except json.JSONDecodeError:
            return {}

    def _save_history(self, history: dict[str, Any]) -> None:
        """Save progress history."""
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        self.history_path.write_text(json.dumps(history, indent=2))


def main() -> None:
    """CLI entry point."""
    import sys

    tracker = ProgressTracker()

    if len(sys.argv) < 2:
        print("Usage: progress_tracker.py <command>")
        print("Commands:")
        print("  stats     - Show progress statistics")
        print("  report    - Generate full report")
        print("  pending   - List pending fixes")
        sys.exit(1)

    command = sys.argv[1]

    if command == "stats":
        stats = tracker.get_stats()
        if "error" in stats:
            print(f"Error: {stats['error']}")
        else:
            print(f"Total: {stats['total']}")
            print(f"  Pending:     {stats['pending']}")
            print(f"  In Progress: {stats['in_progress']}")
            print(f"  Complete:    {stats['complete']}")
            print(f"  Blocked:     {stats['blocked']}")

    elif command == "report":
        print(tracker.generate_report())

    elif command == "pending":
        pending = tracker.get_pending_fixes()
        if pending:
            print("Pending fixes:")
            for fix_id in pending:
                print(f"  - {fix_id}")
        else:
            print("No pending fixes found")

    else:
        print(f"Unknown command: {command}")
        sys.exit(1)


if __name__ == "__main__":
    main()
