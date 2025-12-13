#!/usr/bin/env python3
"""Checkpoint Manager for Release Review Protocol.

Manages incremental progress saving for failure recovery.
Used by agents to save state during long reviews.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


class CheckpointManager:
    """Manage checkpoints for agent review sessions."""

    CHECKPOINT_MAX_AGE_HOURS = 1  # Checkpoints older than this are stale

    def __init__(
        self,
        component: str,
        agent_id: str,
        base_path: str | Path | None = None,
    ) -> None:
        """Initialize checkpoint manager.

        Args:
            component: Component being reviewed
            agent_id: Unique agent session identifier
            base_path: Base path for checkpoints. Defaults to .release_review/checkpoints/
        """
        self.component = component
        self.agent_id = agent_id

        if base_path is None:
            base_path = Path(".release_review/checkpoints")

        # Sanitize component name
        safe_component = component.replace("/", "_").replace("\\", "_")
        self.checkpoint_dir = Path(base_path)
        self.checkpoint_file = self.checkpoint_dir / f"{safe_component}_{agent_id}.json"

    def save(self, state: dict[str, Any]) -> Path:
        """Save checkpoint with auto-generated metadata.

        Args:
            state: Current state to save

        Returns:
            Path to checkpoint file
        """
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "version": "1.0",
            "component": self.component,
            "agent_id": self.agent_id,
            "timestamp": datetime.now().isoformat() + "Z",
            "state": state,
            "resumable": state.get("status") != "complete",
        }

        self.checkpoint_file.write_text(json.dumps(checkpoint, indent=2))
        return self.checkpoint_file

    def load(self) -> dict[str, Any] | None:
        """Load last checkpoint.

        Returns:
            Checkpoint data or None if not found
        """
        if not self.checkpoint_file.exists():
            return None

        try:
            return json.loads(self.checkpoint_file.read_text())
        except json.JSONDecodeError:
            return None

    def resume_or_start(self) -> dict[str, Any]:
        """Load checkpoint or return fresh state.

        Returns:
            State to continue from (either loaded or fresh)
        """
        checkpoint = self.load()

        if checkpoint:
            # Check if checkpoint is recent enough
            try:
                ts = checkpoint["timestamp"].rstrip("Z")
                checkpoint_time = datetime.fromisoformat(ts)
                age_hours = (datetime.now() - checkpoint_time).total_seconds() / 3600

                if age_hours < self.CHECKPOINT_MAX_AGE_HOURS and checkpoint.get("resumable"):
                    print(f"Resuming from checkpoint ({age_hours:.1f}h old)")
                    return checkpoint["state"]
                else:
                    print(f"Checkpoint too old ({age_hours:.1f}h) or not resumable, starting fresh")
            except (KeyError, ValueError):
                print("Invalid checkpoint format, starting fresh")

        # Fresh state
        return {
            "status": "starting",
            "files_reviewed": [],
            "issues_found": 0,
            "tests_run": [],
            "started_at": datetime.now().isoformat() + "Z",
        }

    def clear(self) -> bool:
        """Delete checkpoint file.

        Returns:
            True if deleted, False if not found
        """
        if self.checkpoint_file.exists():
            self.checkpoint_file.unlink()
            return True
        return False

    def get_age_hours(self) -> float | None:
        """Get age of checkpoint in hours.

        Returns:
            Age in hours or None if no checkpoint
        """
        checkpoint = self.load()
        if not checkpoint:
            return None

        try:
            ts = checkpoint["timestamp"].rstrip("Z")
            checkpoint_time = datetime.fromisoformat(ts)
            return (datetime.now() - checkpoint_time).total_seconds() / 3600
        except (KeyError, ValueError):
            return None

    @classmethod
    def list_all(cls, base_path: str | Path | None = None) -> list[dict[str, Any]]:
        """List all checkpoints.

        Args:
            base_path: Base path for checkpoints

        Returns:
            List of checkpoint summaries
        """
        if base_path is None:
            base_path = Path(".release_review/checkpoints")

        checkpoint_dir = Path(base_path)
        if not checkpoint_dir.exists():
            return []

        summaries = []
        for checkpoint_file in checkpoint_dir.glob("*.json"):
            try:
                data = json.loads(checkpoint_file.read_text())
                summaries.append({
                    "file": str(checkpoint_file),
                    "component": data.get("component"),
                    "agent_id": data.get("agent_id"),
                    "timestamp": data.get("timestamp"),
                    "resumable": data.get("resumable"),
                    "status": data.get("state", {}).get("status"),
                })
            except (json.JSONDecodeError, KeyError):
                summaries.append({
                    "file": str(checkpoint_file),
                    "error": "Invalid checkpoint file",
                })

        return summaries


def main() -> None:
    """CLI entry point."""
    import sys

    if len(sys.argv) < 2:
        print("Usage: checkpoint.py <command> [args]")
        print("Commands:")
        print("  list                     - List all checkpoints")
        print("  show <component> <agent> - Show checkpoint for component/agent")
        print("  clear <component> <agent> - Clear checkpoint")
        sys.exit(1)

    command = sys.argv[1]

    if command == "list":
        checkpoints = CheckpointManager.list_all()
        if not checkpoints:
            print("No checkpoints found")
        else:
            print(f"Found {len(checkpoints)} checkpoint(s):")
            for cp in checkpoints:
                if "error" in cp:
                    print(f"  ❌ {cp['file']}: {cp['error']}")
                else:
                    status_icon = "🔄" if cp["resumable"] else "✅"
                    print(f"  {status_icon} {cp['component']} ({cp['agent_id']})")
                    print(f"      Status: {cp['status']}, Time: {cp['timestamp']}")

    elif command == "show":
        if len(sys.argv) < 4:
            print("Usage: checkpoint.py show <component> <agent>")
            sys.exit(1)
        component, agent = sys.argv[2], sys.argv[3]
        manager = CheckpointManager(component, agent)
        checkpoint = manager.load()
        if checkpoint:
            print(json.dumps(checkpoint, indent=2))
        else:
            print(f"No checkpoint found for {component}/{agent}")

    elif command == "clear":
        if len(sys.argv) < 4:
            print("Usage: checkpoint.py clear <component> <agent>")
            sys.exit(1)
        component, agent = sys.argv[2], sys.argv[3]
        manager = CheckpointManager(component, agent)
        if manager.clear():
            print(f"Cleared checkpoint for {component}/{agent}")
        else:
            print(f"No checkpoint to clear for {component}/{agent}")

    else:
        print(f"Unknown command: {command}")
        sys.exit(1)


if __name__ == "__main__":
    main()
