#!/usr/bin/env python3
"""Progress Tracker for Post-Release Recommendation Fixes.

Tracks fix progress across sessions and generates reports.
"""

from __future__ import annotations

import fcntl
import json
import os
import re
import sys
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

from versioning import (
    ensure_within_base,
    read_tracking_version,
    resolve_base_path,
    resolve_version,
)


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
        version: str | None = None,
    ) -> None:
        """Initialize tracker.

        Args:
            base_path: Base path for fix workflow
            version: Release version override
        """
        self.root_path = Path(base_path)
        self.version = resolve_version(version)
        self.base_path = resolve_base_path(self.root_path, self.version)
        self.tracking_path = self.base_path / "TRACKING.md"
        self.history_path = self.base_path / "progress_history.json"
        self.lock_path = self.base_path / ".tracking.lock"
        self.tracking_version = read_tracking_version(
            self.tracking_path, self.base_path
        )

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

    def _ensure_within_base(self, path: Path) -> None:
        """Ensure file operations remain within the base path."""
        ensure_within_base(self.base_path, path)

    def _read_text(self, path: Path) -> str:
        """Read text from a path validated under the base path."""
        self._ensure_within_base(path)
        return path.read_text()

    def _write_text(self, path: Path, content: str) -> None:
        """Write text to a path validated under the base path."""
        self._ensure_within_base(path)
        path.write_text(content)

    def _build_evidence_json(
        self,
        commits: list[str],
        tests_command: str | None,
        tests_status: str,
        tests_passed: int | None,
        tests_total: int | None,
        models: str,
        reviewer: str,
        timestamp: str,
        followups: str = "None",
        accepted_risks: str = "None",
        fmt: str = "standard",
        legacy_summary: str | None = None,
    ) -> str:
        """Build machine-validated evidence JSON."""
        payload: dict[str, Any] = {
            "format": fmt,
            "commits": commits,
            "tests": {
                "command": tests_command,
                "status": tests_status,
                "passed": tests_passed,
                "total": tests_total,
            },
            "models": models,
            "reviewer": reviewer,
            "timestamp": timestamp,
            "followups": followups,
            "accepted_risks": accepted_risks,
        }
        if legacy_summary is not None:
            payload["legacy_summary"] = legacy_summary
        return json.dumps(payload, separators=(",", ":"))

    def _parse_tests_summary(self, summary: str) -> tuple[int | None, int | None]:
        """Parse pytest-style summary for pass/total counts."""
        counts: dict[str, int] = {}
        for match in re.finditer(
            r"(\d+)\s+(passed|failed|errors?|skipped|xfailed|xpassed|deselected)",
            summary,
            re.IGNORECASE,
        ):
            label = match.group(2).lower()
            if label.endswith("s"):
                label = label[:-1]
            counts[label] = counts.get(label, 0) + int(match.group(1))

        if not counts:
            return None, None

        passed = counts.get("passed", 0)
        total = sum(counts.values())
        return passed, total

    def _infer_tests_status(self, summary: str) -> str:
        """Infer PASS/FAIL from test summary text."""
        lowered = summary.lower()
        if "failed" in lowered or "error" in lowered:
            return "FAIL"
        if "passed" in lowered:
            return "PASS"
        return "UNKNOWN"

    def _update_item_details(
        self,
        lines: list[str],
        fix_id: str,
        status: str,
        evidence: str | None,
    ) -> None:
        """Update status/evidence in Item Details section."""
        in_item = False
        header_pattern = re.compile(rf"^###\s+{re.escape(fix_id)}\b")
        for idx, line in enumerate(lines):
            if header_pattern.match(line.strip()):
                in_item = True
                continue
            if in_item and line.strip().startswith("### "):
                break
            if in_item and line.startswith("- **Status**:"):
                lines[idx] = f"- **Status**: {status}"
                continue
            if in_item and line.strip() == "**Evidence**:" and evidence:
                if idx + 1 < len(lines):
                    next_line = lines[idx + 1].strip()
                    if next_line.startswith("(") or next_line.startswith("{"):
                        lines[idx + 1] = evidence

    def _split_table_row(
        self,
        line: str,
        expected_columns: int | None = None,
    ) -> list[str]:
        """Split a markdown table row, preserving pipes in the last column."""
        stripped = line.strip().strip("|")
        parts = [p.strip() for p in stripped.split("|")]
        if expected_columns and len(parts) > expected_columns:
            head = parts[: expected_columns - 1]
            tail = "|".join(parts[expected_columns - 1:]).strip()
            parts = head + [tail]
        return parts

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
            content = self._read_text(self.tracking_path)
            lines = content.split("\n")
            updated_lines: list[str] = []
            found = False
            current_columns: dict[str, int] | None = None

            for line in lines:
                if line.startswith("|") and "Evidence" in line and "Status" in line:
                    headers = self._split_table_row(line)
                    current_columns = {
                        name: idx for idx, name in enumerate(headers)
                    }
                    updated_lines.append(line)
                    continue

                if current_columns and line.startswith("|") and line.strip().startswith("|---"):
                    updated_lines.append(line)
                    continue

                if current_columns and line.startswith("|"):
                    cells = self._split_table_row(
                        line,
                        expected_columns=len(current_columns),
                    )
                    if cells and cells[0] == fix_id:
                        found = True
                        if "Status" in current_columns:
                            cells[current_columns["Status"]] = status
                        if owner and "Owner" in current_columns:
                            cells[current_columns["Owner"]] = owner
                        if evidence and "Evidence" in current_columns:
                            cells[current_columns["Evidence"]] = evidence
                        line = "| " + " | ".join(cells) + " |"
                    updated_lines.append(line)
                    continue

                if not line.startswith("|"):
                    current_columns = None

                updated_lines.append(line)

            if not found:
                raise ValueError(f"Fix ID {fix_id} not found in tracking file")

            self._update_item_details(updated_lines, fix_id, status, evidence or None)
            self._write_text(self.tracking_path, "\n".join(updated_lines))

    def mark_fix_complete(
        self,
        fix_id: str,
        commit_sha: str,
        tests_run: str,
        owner: str,
        reviewer: str = "captain",
        tests_command: str | None = None,
    ) -> None:
        """Mark a fix as complete with evidence.

        Args:
            fix_id: Fix ID
            commit_sha: Commit SHA of the fix
            tests_run: Test command and results
            owner: Agent that completed the fix
        """
        timestamp = datetime.now().isoformat() + "Z"
        tests_passed, tests_total = self._parse_tests_summary(tests_run)
        tests_status = self._infer_tests_status(tests_run)

        if tests_passed is not None and tests_total is not None:
            evidence = self._build_evidence_json(
                commits=[commit_sha],
                tests_command=tests_command or "UNKNOWN",
                tests_status=tests_status,
                tests_passed=tests_passed,
                tests_total=tests_total,
                models=owner or "UNKNOWN",
                reviewer=reviewer,
                timestamp=timestamp,
                fmt="standard",
            )
        else:
            evidence = self._build_evidence_json(
                commits=[commit_sha] if commit_sha else [],
                tests_command=None,
                tests_status="UNKNOWN",
                tests_passed=None,
                tests_total=None,
                models=owner or "UNKNOWN",
                reviewer=reviewer,
                timestamp=timestamp,
                fmt="legacy",
                legacy_summary=f"Tests: {tests_run}",
            )

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

        content = self._read_text(self.tracking_path)
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

        content = self._read_text(self.tracking_path)

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

        self._write_text(progress_file, "\n".join(lines))

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
            return json.loads(self._read_text(self.history_path))
        except json.JSONDecodeError:
            return {}

    def _save_history(self, history: dict[str, Any]) -> None:
        """Save progress history."""
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        self._write_text(self.history_path, json.dumps(history, indent=2))


def main() -> None:
    """CLI entry point."""
    import sys

    args = sys.argv[1:]
    version = None
    if "--version" in args:
        idx = args.index("--version")
        if idx + 1 >= len(args):
            print("Usage: progress_tracker.py --version <version> <command>")
            sys.exit(1)
        version = args[idx + 1]
        del args[idx:idx + 2]

    tracker = ProgressTracker(version=version)

    if len(args) < 1:
        print("Usage: progress_tracker.py <command>")
        print("Commands:")
        print("  stats     - Show progress statistics")
        print("  report    - Generate full report")
        print("  pending   - List pending fixes")
        sys.exit(1)

    command = args[0]

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
