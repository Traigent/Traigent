#!/usr/bin/env python3
"""Session Initializer for Post-Release Recommendation Fixes.

Automates session setup including directory creation, fix selection, and PROGRESS.md generation.
"""

from __future__ import annotations

import re
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

from versioning import read_tracking_version, resolve_base_path, resolve_version


class SessionInitializer:
    """Initialize fix sessions with proper structure and populated templates."""

    SCOPE_OPTIONS = ["high", "medium", "low", "all"]

    def __init__(
        self,
        base_path: str | Path = ".post_release_recommendation_fixes",
        version: str | None = None,
    ) -> None:
        """Initialize session initializer.

        Args:
            base_path: Base path for fix workflow
            version: Release version override
        """
        self.root_path = Path(base_path)
        self.version = resolve_version(version)
        self.base_path = resolve_base_path(self.root_path, self.version)
        self.tracking_path = self.base_path / "TRACKING.md"
        self.template_dir = self.root_path / "templates"
        self.tracking_version = read_tracking_version(self.tracking_path)

    def init_session(
        self,
        scope: str = "high",
        fix_ids: list[str] | None = None,
        time_box_hours: float = 4.0,
    ) -> Path:
        """Initialize a new fix session.

        Args:
            scope: Priority scope ("high", "medium", "low", "all")
            fix_ids: Specific fix IDs to target (overrides scope)
            time_box_hours: Time box for the session in hours

        Returns:
            Path to session directory
        """
        session_date = datetime.now().strftime("%Y%m%d")
        session_dir = self.base_path / "sessions" / session_date

        # Check if session already exists
        if session_dir.exists():
            # Add suffix for multiple sessions per day
            suffix = 1
            while (self.base_path / "sessions" / f"{session_date}_{suffix}").exists():
                suffix += 1
            session_dir = self.base_path / "sessions" / f"{session_date}_{suffix}"

        # Create session directory structure
        session_dir.mkdir(parents=True, exist_ok=True)
        (session_dir / "artifacts").mkdir(exist_ok=True)

        # Ensure versioned USER_QUESTIONS.md exists
        user_questions = self.base_path / "USER_QUESTIONS.md"
        template_questions = self.root_path / "USER_QUESTIONS.md"
        if not user_questions.exists() and template_questions.exists():
            shutil.copy(template_questions, user_questions)

        # Ensure versioned TEMPLATE exists (copied from shared templates)
        self._ensure_versioned_template()

        # Determine targeted fixes
        if fix_ids:
            targeted = fix_ids
        else:
            targeted = self._get_fixes_by_scope(scope)

        if not targeted:
            raise ValueError(f"No pending fixes found for scope: {scope}")

        # Generate PROGRESS.md from fixes
        self._generate_progress_file(session_dir, scope, targeted, time_box_hours)

        # Copy artifacts README
        artifacts_readme = self.template_dir / "artifacts" / "README.md"
        if artifacts_readme.exists():
            shutil.copy(artifacts_readme, session_dir / "artifacts" / "README.md")

        return session_dir

    def _ensure_versioned_template(self) -> None:
        """Ensure versioned TEMPLATE exists under the release workspace."""
        template_root = self.template_dir
        if not template_root.exists():
            return

        versioned_template = self.base_path / "sessions" / "TEMPLATE"
        versioned_template.mkdir(parents=True, exist_ok=True)

        progress_template = template_root / "PROGRESS.md"
        if progress_template.exists():
            shutil.copy(progress_template, versioned_template / "PROGRESS.md")

        artifacts_template = template_root / "artifacts" / "README.md"
        if artifacts_template.exists():
            artifacts_dir = versioned_template / "artifacts"
            artifacts_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy(artifacts_template, artifacts_dir / "README.md")

    def _get_fixes_by_scope(self, scope: str) -> list[str]:
        """Get pending fix IDs by priority scope.

        Args:
            scope: Priority scope

        Returns:
            List of fix IDs
        """
        if not self.tracking_path.exists():
            return []

        content = self.tracking_path.read_text()
        fixes = []
        current_priority = ""

        for line in content.split("\n"):
            # Track current section
            if "## High Priority" in line:
                current_priority = "high"
            elif "## Medium Priority" in line:
                current_priority = "medium"
            elif "## Low Priority" in line:
                current_priority = "low"
            elif "## Item Details" in line:
                break

            # Extract pending fixes from matching priority section
            if "| Pending |" in line:
                match = re.match(r"\|\s*(\d+)\s*\|", line)
                if match:
                    fix_id = match.group(1)
                    if scope == "all" or scope == current_priority:
                        fixes.append(fix_id)

        return fixes

    def _get_fix_details(self, fix_id: str) -> dict[str, str]:
        """Get details for a specific fix from TRACKING.md.

        Args:
            fix_id: Fix ID

        Returns:
            Dictionary with fix details
        """
        if not self.tracking_path.exists():
            return {}

        content = self.tracking_path.read_text()
        details: dict[str, str] = {"id": fix_id}

        # Find the item details section
        in_item = False
        for line in content.split("\n"):
            if line.strip().startswith(f"### {fix_id}:"):
                in_item = True
                # Extract title
                title_match = re.match(rf"###\s*{fix_id}:\s*(.+)$", line.strip())
                if title_match:
                    details["title"] = title_match.group(1)
            elif in_item:
                if line.strip().startswith("### "):
                    break  # Next item
                if line.startswith("- **Priority**:"):
                    details["priority"] = line.split(":", 1)[1].strip()
                elif line.startswith("- **Component**:"):
                    details["component"] = line.split(":", 1)[1].strip()
                elif line.startswith("- **Location**:"):
                    details["location"] = line.split(":", 1)[1].strip()
                elif line.startswith("- **Effort**:"):
                    details["effort"] = line.split(":", 1)[1].strip()

        return details

    def _generate_progress_file(
        self,
        session_dir: Path,
        scope: str,
        targeted: list[str],
        time_box_hours: float,
    ) -> None:
        """Generate PROGRESS.md for session.

        Args:
            session_dir: Session directory path
            scope: Scope description
            targeted: List of targeted fix IDs
            time_box_hours: Time box in hours
        """
        timestamp = datetime.now().isoformat() + "Z"
        session_id = session_dir.name

        # Get details for each fix
        fix_details = [self._get_fix_details(fix_id) for fix_id in targeted]
        metadata = self._read_tracking_metadata()
        release_version = (
            metadata.get("version")
            or self.tracking_version
            or self.version
            or "UNKNOWN"
        )
        source_todo = metadata.get("source", "UNKNOWN")
        release_review_tracking = (
            f".release_review/{release_version}/PRE_RELEASE_REVIEW_TRACKING.md"
            if release_version != "UNKNOWN"
            else "UNKNOWN"
        )

        lines = [
            f"# Session Progress: {session_id}",
            "",
            f"**Session ID**: {session_id}",
            f"**Release Version**: {release_version}",
            f"**Started**: {timestamp}",
            "**Ended**: (In progress)",
            f"**Tracking File**: [TRACKING.md](../../TRACKING.md)",
            f"**Source TODO**: {source_todo}",
            f"**Release Review Tracking**: {release_review_tracking}",
            "",
            "## Scope",
            "",
            f"- **Priority**: {scope.title()} priority fixes",
            f"- **Fixes targeted**: {len(targeted)} items",
            f"- **Time box**: {time_box_hours}h",
            "",
            "## Status Summary",
            "",
            f"- **Completed**: 0/{len(targeted)}",
            f"- **In Progress**: 0/{len(targeted)}",
            f"- **Blocked**: 0/{len(targeted)}",
            "",
            "## Fix Details",
            "",
        ]

        for details in fix_details:
            fix_id = details.get("id", "???")
            title = details.get("title", "Unknown")
            component = details.get("component", "Unknown")
            effort = details.get("effort", "Unknown")

            lines.extend([
                f"### {fix_id}: {title}",
                "",
                f"- **Status**: Pending",
                f"- **Component**: {component}",
                f"- **Effort**: {effort}",
                f"- **Owner**: (unassigned)",
                f"- **Branch**: (not created)",
                f"- **Started**: -",
                f"- **Evidence**: (pending JSON)",
                "",
            ])

        lines.extend([
            "---",
            "",
            "## Session Log",
            "",
            f"- {timestamp} - Session started, targeting {len(targeted)} fixes",
            "",
            "---",
            "",
            "## End of Session Notes",
            "",
            "(Fill this in when ending the session)",
            "",
            "### Decisions Made",
            "- (none yet)",
            "",
            "### Issues Encountered",
            "- (none yet)",
            "",
            "### Next Session Priorities",
            "1. (TBD)",
            "",
        ])

        progress_file = session_dir / "PROGRESS.md"
        progress_file.write_text("\n".join(lines))

    def get_latest_session(self) -> Path | None:
        """Get the most recent session directory.

        Returns:
            Path to latest session or None
        """
        sessions_dir = self.base_path / "sessions"
        if not sessions_dir.exists():
            return None

        sessions = [
            d for d in sessions_dir.iterdir()
            if d.is_dir() and d.name != "TEMPLATE"
        ]

        if not sessions:
            return None

        # Sort by name (date-based) descending
        return sorted(sessions, reverse=True)[0]

    def _read_tracking_metadata(self) -> dict[str, str]:
        """Read tracking metadata for version/source."""
        if not self.tracking_path.exists():
            return {}
        metadata: dict[str, str] = {}
        for line in self.tracking_path.read_text().splitlines():
            if line.startswith("**Release Version**:"):
                metadata["version"] = line.split(":", 1)[1].strip().strip("`")
            elif line.startswith("**Source**:"):
                metadata["source"] = line.split(":", 1)[1].strip()
        return metadata

    def resume_session(self) -> dict[str, Any]:
        """Get information for resuming the latest session.

        Returns:
            Dictionary with resume information
        """
        latest = self.get_latest_session()
        if not latest:
            return {"error": "No sessions found"}

        progress_file = latest / "PROGRESS.md"
        if not progress_file.exists():
            return {"error": f"No PROGRESS.md in {latest}"}

        content = progress_file.read_text()

        # Parse session state
        info: dict[str, Any] = {
            "session_dir": str(latest),
            "session_id": latest.name,
            "in_progress": [],
            "completed": [],
            "pending": [],
        }

        current_fix = None
        for line in content.split("\n"):
            # Parse fix status
            if line.strip().startswith("### ") and ":" in line:
                match = re.match(r"###\s*(\d+):", line.strip())
                if match:
                    current_fix = match.group(1)

            if current_fix and line.strip().startswith("- **Status**:"):
                status = line.split(":", 1)[1].strip()
                if status == "In Progress":
                    info["in_progress"].append(current_fix)
                elif status == "Complete":
                    info["completed"].append(current_fix)
                elif status == "Pending":
                    info["pending"].append(current_fix)
                current_fix = None

        return info


def main() -> None:
    """CLI entry point."""
    import sys

    args = sys.argv[1:]
    version = None
    if "--version" in args:
        idx = args.index("--version")
        if idx + 1 >= len(args):
            print("Usage: session_init.py --version <version> <command> [args]")
            sys.exit(1)
        version = args[idx + 1]
        del args[idx:idx + 2]

    if len(args) < 1:
        print("Usage: session_init.py <command> [args]")
        print("Commands:")
        print("  init [scope]     - Initialize new session (scope: high/medium/low/all)")
        print("  resume           - Get resume info for latest session")
        print("  latest           - Show latest session path")
        sys.exit(1)

    command = args[0]
    initializer = SessionInitializer(version=version)

    if command == "init":
        scope = args[1] if len(args) > 1 else "high"
        if scope not in SessionInitializer.SCOPE_OPTIONS:
            print(f"Invalid scope: {scope}")
            print(f"Options: {', '.join(SessionInitializer.SCOPE_OPTIONS)}")
            sys.exit(1)

        try:
            session_dir = initializer.init_session(scope=scope)
            print(f"Session initialized: {session_dir}")
            print(f"PROGRESS.md created with targeted fixes")
        except ValueError as e:
            print(f"Error: {e}")
            sys.exit(1)

    elif command == "resume":
        info = initializer.resume_session()
        if "error" in info:
            print(f"Error: {info['error']}")
            sys.exit(1)

        print(f"Session: {info['session_id']}")
        print(f"Directory: {info['session_dir']}")
        print(f"Completed: {len(info['completed'])} fixes")
        print(f"In Progress: {len(info['in_progress'])} fixes")
        print(f"Pending: {len(info['pending'])} fixes")

        if info["in_progress"]:
            print(f"\nResume with: {', '.join(info['in_progress'])}")
        elif info["pending"]:
            print(f"\nNext up: {info['pending'][0]}")

    elif command == "latest":
        latest = initializer.get_latest_session()
        if latest:
            print(latest)
        else:
            print("No sessions found")
            sys.exit(1)

    else:
        print(f"Unknown command: {command}")
        sys.exit(1)


if __name__ == "__main__":
    main()
