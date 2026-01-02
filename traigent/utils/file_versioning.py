"""File versioning and naming standardization for Traigent optimization logs.

This module provides consistent file naming conventions and versioning support
for all optimization run artifacts, ensuring reproducibility and traceability.
"""

# Traceability: CONC-Layer-Data CONC-Quality-Reliability CONC-Quality-Maintainability FUNC-STORAGE REQ-STOR-007 SYNC-StorageLogging

from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from traigent.utils.logging import get_logger
from traigent.utils.secure_path import safe_open, validate_path

logger = get_logger(__name__)

# File naming version
FILE_NAMING_VERSION = "2.0.0"

# Standard file name patterns with versioning
FILE_PATTERNS = {
    # Meta files (v2 adds version suffix)
    "session": "session_v{version}.json",
    "config": "config_v{version}.json",
    "objectives": "objectives_v{version}.json",
    "dataset": "dataset_v{version}.json",
    "environment": "environment_v{version}.json",
    "manifest": "manifest_v{version}.json",  # New in v2
    # Trial files
    "trials_stream": "trials_v{version}.jsonl",
    "trial_detail": "trial_{trial_id}_v{version}.json",
    # Metric files
    "metric_stream": "{metric_name}_over_time_v{version}.jsonl",
    "metrics_summary": "metrics_summary_v{version}.json",
    # Checkpoint files
    "checkpoint": "checkpoint_{trial_count:05d}_v{version}.json",
    "checkpoint_latest": "latest_checkpoint_v{version}.json",
    "trial_history": "trial_history_v{version}.json",
    # Artifact files
    "best_config": "best_config_v{version}.json",
    "weighted_results": "weighted_results_v{version}.json",
    "pareto_front": "pareto_front_v{version}.json",  # New in v2
    "results": "results_v{version}.json",  # General results file
    # Status files
    "status": "status_v{version}.json",
    "run_info": "run_info_v{version}.json",  # New in v2
}

# Legacy file patterns (for backward compatibility)
LEGACY_PATTERNS = {
    "session": "session.json",
    "config": "config.json",
    "objectives": "objectives.json",
    "dataset": "dataset.json",
    "environment": "environment.json",
    "trials_stream": "trials.jsonl",
    "trial_detail": "trial_{trial_id}.json",
    "metric_stream": "{metric_name}_over_time.jsonl",
    "metrics_summary": "objectives_summary.json",
    "checkpoint": "checkpoint_{trial_count:05d}.json",
    "checkpoint_latest": "latest_checkpoint.json",
    "trial_history": "trial_history.json",
    "best_config": "best_config.json",
    "weighted_results": "weighted_results.json",
    "status": "status.json",
}


class FileVersionManager:
    """Manages file versioning and naming conventions for optimization logs."""

    def __init__(self, version: str = "2", use_legacy: bool = False) -> None:
        """Initialize file version manager.

        Args:
            version: File format version to use ("1" for legacy, "2" for new)
            use_legacy: Force use of legacy naming (no version suffix)
        """
        self.version = version
        self.use_legacy = use_legacy
        self.patterns = LEGACY_PATTERNS if use_legacy else FILE_PATTERNS

    def get_filename(self, file_type: str, **kwargs) -> str:
        """Get standardized filename for a given file type.

        Args:
            file_type: Type of file (e.g., "session", "config", "checkpoint")
            **kwargs: Additional parameters for filename formatting

        Returns:
            Standardized filename

        Raises:
            ValueError: If file_type is not recognized
        """
        if file_type not in self.patterns:
            raise ValueError(f"Unknown file type: {file_type}")

        pattern = self.patterns[file_type]

        # Add version to kwargs if not legacy
        if not self.use_legacy:
            kwargs["version"] = self.version

        return pattern.format(**kwargs)

    def parse_filename(self, filename: str) -> tuple[str, dict[str, Any]]:
        """Parse a filename to extract type and metadata.

        Args:
            filename: Filename to parse

        Returns:
            Tuple of (file_type, metadata_dict)
        """
        filename = Path(filename).name  # Get just the filename

        # Try to match against patterns
        for file_type, pattern in self.patterns.items():
            # Convert pattern to regex
            regex_pattern = pattern
            regex_pattern = regex_pattern.replace("{version}", r"(?P<version>\d+)")
            regex_pattern = regex_pattern.replace("{trial_id}", r"(?P<trial_id>[^_]+)")
            regex_pattern = regex_pattern.replace(
                "{trial_count:05d}", r"(?P<trial_count>\d{5})"
            )
            regex_pattern = regex_pattern.replace(
                "{metric_name}", r"(?P<metric_name>[^_]+)"
            )
            regex_pattern = "^" + regex_pattern + "$"

            match = re.match(regex_pattern, filename)
            if match:
                return file_type, match.groupdict()

        # Try legacy patterns
        for file_type, pattern in LEGACY_PATTERNS.items():
            regex_pattern = pattern
            regex_pattern = regex_pattern.replace("{trial_id}", r"(?P<trial_id>[^_]+)")
            regex_pattern = regex_pattern.replace(
                "{trial_count:05d}", r"(?P<trial_count>\d{5})"
            )
            regex_pattern = regex_pattern.replace(
                "{metric_name}", r"(?P<metric_name>[^_]+)"
            )
            regex_pattern = "^" + regex_pattern + "$"

            match = re.match(regex_pattern, filename)
            if match:
                metadata = match.groupdict()
                metadata["version"] = "1"  # Legacy version
                return file_type, metadata

        return "unknown", {}

    def create_manifest(self, run_path: Path) -> dict[str, Any]:
        """Create a manifest file documenting all files in a run.

        Args:
            run_path: Path to the run directory

        Returns:
            Manifest dictionary
        """
        manifest: dict[str, Any] = {
            "file_naming_version": FILE_NAMING_VERSION,
            "created_at": datetime.now(UTC).isoformat(),
            "files": {},
        }

        # Scan all files in the run directory
        base_path = run_path.resolve()

        for relative_path, absolute_path in self._iter_run_files(base_path):
            try:
                stats = absolute_path.stat()
            except OSError as exc:
                logger.warning("Skipping %s due to stat error: %s", absolute_path, exc)
                continue

            file_info = {
                "path": str(relative_path),
                "size": stats.st_size,
                "modified": datetime.fromtimestamp(stats.st_mtime).isoformat(),
            }

            file_type, metadata = self.parse_filename(absolute_path.name)
            file_info["type"] = file_type
            file_info["metadata"] = metadata

            if absolute_path.suffix == ".json":
                try:
                    with safe_open(absolute_path, base_path, mode="rb") as f:
                        file_info["sha256"] = hashlib.sha256(f.read()).hexdigest()
                except OSError as exc:
                    logger.warning(
                        "Failed to compute checksum for %s: %s", absolute_path, exc
                    )

            manifest["files"][str(relative_path)] = file_info

        return manifest

    def validate_manifest(
        self, run_path: Path, manifest: dict[str, Any]
    ) -> dict[str, Any]:
        """Validate files against a manifest.

        Args:
            run_path: Path to the run directory
            manifest: Manifest to validate against

        Returns:
            Validation report with any discrepancies
        """
        missing_files: list[str] = []
        extra_files: list[str] = []
        checksum_mismatches: list[dict[str, Any]] = []
        size_mismatches: list[dict[str, Any]] = []

        report = {
            "valid": True,
            "missing_files": missing_files,
            "extra_files": extra_files,
            "checksum_mismatches": checksum_mismatches,
            "size_mismatches": size_mismatches,
        }

        manifest_files = set(manifest.get("files", {}).keys())
        actual_files = set()

        # Check existing files
        base_path = run_path.resolve()

        for relative_path, absolute_path in self._iter_run_files(base_path):
            relative_str = str(relative_path)
            actual_files.add(relative_str)

            if relative_str in manifest_files:
                file_info = manifest["files"][relative_str]

                try:
                    stats = absolute_path.stat()
                except OSError as exc:
                    logger.warning(
                        "Skipping %s due to stat error: %s", absolute_path, exc
                    )
                    continue

                actual_size = stats.st_size
                if actual_size != file_info.get("size", actual_size):
                    size_mismatches.append(
                        {
                            "file": relative_str,
                            "expected": file_info.get("size"),
                            "actual": actual_size,
                        }
                    )
                    report["valid"] = False

                if "sha256" in file_info and absolute_path.suffix == ".json":
                    try:
                        validated_path = validate_path(
                            absolute_path, base_path, must_exist=True
                        )
                        with safe_open(validated_path, base_path, mode="rb") as f:
                            actual_checksum = hashlib.sha256(f.read()).hexdigest()
                    except OSError as exc:
                        logger.warning(
                            "Failed to compute checksum for %s: %s", absolute_path, exc
                        )
                        report["valid"] = False
                        continue

                    if actual_checksum != file_info["sha256"]:
                        checksum_mismatches.append(
                            {
                                "file": relative_str,
                                "expected": file_info["sha256"],
                                "actual": actual_checksum,
                            }
                        )
                        report["valid"] = False

        # Check for missing and extra files
        missing_files[:] = sorted(manifest_files - actual_files)
        extra_files[:] = sorted(actual_files - manifest_files)

        if report["missing_files"] or report["extra_files"]:
            report["valid"] = False

        return report

    def _iter_run_files(self, base_path: Path):
        """Yield (relative_path, absolute_path) for files under base_path, skipping symlinks."""
        if not base_path.exists():
            return

        for root, dirs, files in os.walk(base_path, topdown=True, followlinks=False):
            root_path = Path(root)

            for dirname in list(dirs):
                dir_path = root_path / dirname
                try:
                    if dir_path.is_symlink():
                        logger.warning(
                            "Skipping symlinked directory during manifest scan: %s",
                            dir_path,
                        )
                        dirs.remove(dirname)
                except OSError as exc:
                    logger.warning(
                        "Skipping directory %s due to access error: %s",
                        dir_path,
                        exc,
                    )
                    dirs.remove(dirname)

            for name in files:
                file_path = root_path / name
                try:
                    if file_path.is_symlink():
                        logger.warning(
                            "Skipping symlinked file during manifest scan: %s",
                            file_path,
                        )
                        continue

                    resolved_path = file_path.resolve()
                    try:
                        relative_path = resolved_path.relative_to(base_path)
                    except ValueError:
                        logger.warning(
                            "Skipping file outside of base path during manifest scan: %s",
                            file_path,
                        )
                        continue
                except OSError as exc:
                    logger.warning(
                        "Skipping file %s due to access error: %s", file_path, exc
                    )
                    continue

                yield relative_path, resolved_path


class RunVersionInfo:
    """Manages version information for optimization runs."""

    def __init__(self, run_path: Path) -> None:
        """Initialize run version info.

        Args:
            run_path: Path to the run directory
        """
        self.run_path = run_path
        self.version_file = run_path / "meta" / "version_info.json"

    def create_version_info(
        self,
        traigent_version: str,
        file_naming_version: str = FILE_NAMING_VERSION,
        custom_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create version information for the run.

        Args:
            traigent_version: Traigent SDK version
            file_naming_version: File naming convention version
            custom_metadata: Additional metadata to include

        Returns:
            Version info dictionary
        """
        import platform
        import sys

        version_info = {
            "traigent_version": traigent_version,
            "file_naming_version": file_naming_version,
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "platform": {
                "system": platform.system(),
                "release": platform.release(),
                "machine": platform.machine(),
                "python_implementation": platform.python_implementation(),
            },
            "created_at": datetime.now(UTC).isoformat(),
            "custom_metadata": custom_metadata or {},
        }

        # Save version info
        self.version_file.parent.mkdir(parents=True, exist_ok=True)
        validated_version_file = validate_path(
            self.version_file, self.version_file.parent, must_exist=False
        )
        with safe_open(validated_version_file, self.version_file.parent, mode="w") as f:
            json.dump(version_info, f, indent=2)

        return version_info

    def load_version_info(self) -> dict[str, Any] | None:
        """Load version information from the run.

        Returns:
            Version info dictionary or None if not found
        """
        if not self.version_file.exists():
            return None

        validated_version_file = validate_path(
            self.version_file, self.version_file.parent, must_exist=True
        )
        with safe_open(validated_version_file, self.version_file.parent, mode="r") as f:
            return cast(dict[str, Any] | None, json.load(f))

    def check_compatibility(self, current_version: str) -> dict[str, Any]:
        """Check compatibility between current and run versions.

        Args:
            current_version: Current Traigent version

        Returns:
            Compatibility report
        """
        version_info = self.load_version_info()
        if not version_info:
            return {
                "compatible": True,
                "warnings": ["No version info found, assuming compatible"],
                "run_version": None,
                "current_version": current_version,
            }

        run_version = version_info.get("traigent_version", "unknown")

        # Simple version comparison (could be enhanced with semantic versioning)
        major_run = run_version.split(".")[0] if "." in run_version else "0"
        major_current = current_version.split(".")[0] if "." in current_version else "0"

        compatible = major_run == major_current
        warnings = []

        if not compatible:
            warnings.append(
                f"Major version mismatch: run={run_version}, current={current_version}"
            )

        # Check file naming version
        file_version = version_info.get("file_naming_version", "1.0.0")
        if file_version != FILE_NAMING_VERSION:
            warnings.append(
                f"File naming version mismatch: run={file_version}, current={FILE_NAMING_VERSION}"
            )

        return {
            "compatible": compatible,
            "warnings": warnings,
            "run_version": run_version,
            "current_version": current_version,
            "file_naming_version": file_version,
        }


def migrate_legacy_files(run_path: Path, dry_run: bool = True) -> dict[str, Any]:
    """Migrate legacy file names to new versioned format.

    Args:
        run_path: Path to the run directory
        dry_run: If True, only report what would be done

    Returns:
        Migration report
    """
    report: dict[str, Any] = {
        "files_to_migrate": [],
        "files_migrated": [],
        "errors": [],
    }

    version_manager = FileVersionManager(version="2")

    # Map legacy patterns to new patterns
    migration_map = {
        "session.json": version_manager.get_filename("session"),
        "config.json": version_manager.get_filename("config"),
        "objectives.json": version_manager.get_filename("objectives"),
        "dataset.json": version_manager.get_filename("dataset"),
        "environment.json": version_manager.get_filename("environment"),
        "trials.jsonl": version_manager.get_filename("trials_stream"),
        "objectives_summary.json": version_manager.get_filename("metrics_summary"),
        "latest_checkpoint.json": version_manager.get_filename("checkpoint_latest"),
        "trial_history.json": version_manager.get_filename("trial_history"),
        "best_config.json": version_manager.get_filename("best_config"),
        "weighted_results.json": version_manager.get_filename("weighted_results"),
        "status.json": version_manager.get_filename("status"),
    }

    # Find and migrate files
    for old_name, new_name in migration_map.items():
        old_paths = list(run_path.rglob(old_name))
        for old_path in old_paths:
            new_path = old_path.parent / new_name

            migration_info = {
                "old_path": str(old_path),
                "new_path": str(new_path),
                "old_name": old_name,
                "new_name": new_name,
            }

            report["files_to_migrate"].append(migration_info)

            if not dry_run:
                try:
                    # Create a copy with new name (preserve original)
                    import shutil

                    shutil.copy2(old_path, new_path)
                    report["files_migrated"].append(migration_info)
                    logger.info(f"Migrated {old_name} to {new_name}")
                except Exception as e:
                    error_info = migration_info.copy()
                    error_info["error"] = str(e)
                    report["errors"].append(error_info)
                    logger.error(f"Failed to migrate {old_name}: {e}")

    # Handle special cases (trial files with IDs)
    trial_pattern = re.compile(r"trial_([^_]+)\.json")
    for trial_file in run_path.rglob("trial_*.json"):
        match = trial_pattern.match(trial_file.name)
        if match and "_v" not in trial_file.name:  # Not already versioned
            trial_id = match.group(1)
            new_name = version_manager.get_filename("trial_detail", trial_id=trial_id)
            new_path = trial_file.parent / new_name

            migration_info = {
                "old_path": str(trial_file),
                "new_path": str(new_path),
                "old_name": trial_file.name,
                "new_name": new_name,
            }

            report["files_to_migrate"].append(migration_info)

            if not dry_run:
                try:
                    import shutil

                    shutil.copy2(trial_file, new_path)
                    report["files_migrated"].append(migration_info)
                    logger.info(f"Migrated {trial_file.name} to {new_name}")
                except Exception as e:
                    error_info = migration_info.copy()
                    error_info["error"] = str(e)
                    report["errors"].append(error_info)
                    logger.error(f"Failed to migrate {trial_file.name}: {e}")

    return report
