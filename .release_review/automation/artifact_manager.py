#!/usr/bin/env python3
"""Artifact Manager for Release Review Protocol.

Auto-generates artifact paths and validates report structure.
Used by captain to assign paths to agents and validate outputs.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any


class ArtifactManager:
    """Manage artifact paths and validation for release reviews."""

    REQUIRED_SECTIONS = [
        "## Tests Executed",
        "## Issues Found",
        "## Evidence",
    ]

    REPORT_TEMPLATE = '''# Review Findings: {component}

**Agent**: {model}
**Agent ID**: {agent_id}
**Timestamp**: {timestamp}
**Scope**: {scope}

## Tests Executed

| Command | Exit Code | Passed | Failed | Duration |
|---------|-----------|--------|--------|----------|
| `pytest tests/...` | 0 | 0 | 0 | 0s |

## Issues Found

(None found / List issues here)

## Recommendations (Post-Release)

- [ ] (None / List recommendations)

## Evidence

- Terminal output hash: sha256:...
- Git state at review: (commit SHA)
- Environment: TRAIGENT_MOCK_MODE=true, Python 3.x
'''

    def __init__(
        self,
        release_version: str,
        base_path: str | Path | None = None,
    ) -> None:
        """Initialize artifact manager.

        Args:
            release_version: Version string (e.g., "v0.8.0")
            base_path: Base path for release review. Defaults to .release_review/

        Canonical structure:
            .release_review/<version>/
            ├── TRACE_LOG.md
            ├── AUDIT_TRAIL.md
            ├── POST_RELEASE_TODO.md
            └── artifacts/<component>/<model>/YYYYMMDD_<type>.md
        """
        self.release_version = release_version
        if base_path is None:
            base_path = Path(".release_review")
        self.release_dir = Path(base_path) / release_version
        self.base = self.release_dir / "artifacts"

    def get_component_path(
        self,
        component: str,
        model: str,
        report_type: str = "findings",
    ) -> Path:
        """Auto-generate standardized artifact path.

        Args:
            component: Component name (e.g., "core/orchestrator")
            model: Model name (e.g., "claude", "gpt5")
            report_type: Type of report (findings, fixes, recommendations)

        Returns:
            Path to the artifact file
        """
        # Sanitize component name for filesystem
        safe_component = component.replace("/", "_").replace("\\", "_")
        safe_model = model.lower().replace(" ", "_").replace(".", "")

        date = datetime.now().strftime("%Y%m%d")
        filename = f"{date}_{report_type}.md"

        path = self.base / safe_component / safe_model / filename
        return path

    def ensure_path(self, path: Path) -> Path:
        """Create parent directories if they don't exist.

        Args:
            path: Path to ensure exists

        Returns:
            The same path
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def generate_template(
        self,
        component: str,
        model: str,
        agent_id: str,
        scope: str,
    ) -> str:
        """Generate a findings report template.

        Args:
            component: Component name
            model: Model name
            agent_id: Unique agent session ID
            scope: File paths being reviewed

        Returns:
            Template string
        """
        return self.REPORT_TEMPLATE.format(
            component=component,
            model=model,
            agent_id=agent_id,
            timestamp=datetime.now().isoformat() + "Z",
            scope=scope,
        )

    def validate_report(self, path: Path) -> dict[str, Any]:
        """Validate report contains required sections.

        Args:
            path: Path to report file

        Returns:
            Validation result with missing sections
        """
        if not path.exists():
            return {
                "valid": False,
                "error": f"Report file not found: {path}",
                "missing_sections": self.REQUIRED_SECTIONS,
            }

        content = path.read_text()

        missing = []
        for section in self.REQUIRED_SECTIONS:
            if section not in content:
                missing.append(section)

        return {
            "valid": len(missing) == 0,
            "missing_sections": missing,
            "error": f"Missing sections: {missing}" if missing else None,
        }

    def list_artifacts(self, component: str | None = None) -> list[Path]:
        """List all artifacts for a component or all components.

        Args:
            component: Optional component name to filter by

        Returns:
            List of artifact paths
        """
        if not self.base.exists():
            return []

        if component:
            safe_component = component.replace("/", "_").replace("\\", "_")
            pattern = f"{safe_component}/**/*.md"
        else:
            pattern = "**/*.md"

        return sorted(self.base.glob(pattern))

    def get_trace_log_path(self) -> Path:
        """Get path to trace log for this release.

        Returns:
            Path to TRACE_LOG.md
        """
        return self.base.parent / "TRACE_LOG.md"


def main() -> None:
    """CLI entry point for testing."""
    import sys

    if len(sys.argv) < 2:
        print("Usage: artifact_manager.py <version> [component] [model]")
        print("Example: artifact_manager.py v0.8.0 core/orchestrator claude")
        sys.exit(1)

    version = sys.argv[1]
    manager = ArtifactManager(version)

    if len(sys.argv) >= 4:
        component = sys.argv[2]
        model = sys.argv[3]
        path = manager.get_component_path(component, model)
        print(f"Artifact path: {path}")
        manager.ensure_path(path)
        print(f"Directory created: {path.parent}")
    else:
        # List all artifacts
        artifacts = manager.list_artifacts()
        print(f"Artifacts for {version}:")
        for artifact in artifacts:
            print(f"  {artifact}")


if __name__ == "__main__":
    main()
