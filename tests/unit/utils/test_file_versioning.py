"""Tests for file versioning and standardization system."""

import json
import os
import tempfile
from pathlib import Path

import pytest

from traigent.utils.file_versioning import FileVersionManager, RunVersionInfo


class TestFileVersionManager:
    """Test FileVersionManager functionality."""

    def test_initialization(self):
        """Test FileVersionManager initialization."""
        # Test default initialization
        manager = FileVersionManager()
        assert manager.version == "2"
        assert manager.use_legacy is False

        # Test with custom version
        manager = FileVersionManager(version="3")
        assert manager.version == "3"

        # Test legacy mode
        manager = FileVersionManager(use_legacy=True)
        assert manager.use_legacy is True

    def test_get_filename(self):
        """Test filename generation."""
        manager = FileVersionManager(version="2")

        # Test versioned filenames
        assert manager.get_filename("session") == "session_v2.json"
        assert manager.get_filename("config") == "config_v2.json"
        assert manager.get_filename("objectives") == "objectives_v2.json"
        assert manager.get_filename("best_config") == "best_config_v2.json"
        assert (
            manager.get_filename("checkpoint", trial_count=5)
            == "checkpoint_00005_v2.json"
        )
        assert (
            manager.get_filename("trial_detail", trial_id="abc123")
            == "trial_abc123_v2.json"
        )

        # Test legacy filenames
        legacy_manager = FileVersionManager(use_legacy=True)
        assert legacy_manager.get_filename("session") == "session.json"
        assert legacy_manager.get_filename("config") == "config.json"
        assert (
            legacy_manager.get_filename("checkpoint", trial_count=5)
            == "checkpoint_00005.json"
        )

    def test_get_pattern(self):
        """Test filename pattern retrieval."""
        manager = FileVersionManager()

        # Test pattern retrieval - patterns is a dict now
        assert "session" in manager.patterns
        assert "checkpoint" in manager.patterns

        # Test unknown pattern
        assert "unknown" not in manager.patterns

    def test_parse_filename(self):
        """Test filename parsing."""
        manager = FileVersionManager()

        # Test versioned filename parsing
        file_type, metadata = manager.parse_filename("session_v2.json")
        assert file_type == "session"
        assert metadata.get("version") == "2"

        file_type, metadata = manager.parse_filename("checkpoint_00005_v2.json")
        assert file_type == "checkpoint"
        assert metadata.get("version") == "2"
        assert metadata.get("trial_count") == "00005"

        file_type, metadata = manager.parse_filename("trial_abc123_v2.json")
        assert file_type == "trial_detail"
        assert metadata.get("version") == "2"
        assert metadata.get("trial_id") == "abc123"

        # Test legacy filename parsing
        file_type, metadata = manager.parse_filename("session.json")
        assert file_type == "session"
        assert metadata.get("version") == "1"

        # Test unknown filename
        file_type, metadata = manager.parse_filename("unknown_file.txt")
        assert file_type == "unknown"

    def test_create_manifest(self):
        """Test manifest creation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = FileVersionManager(version="2")
            tmpdir_path = Path(tmpdir)

            # Create test files
            file1 = tmpdir_path / "session_v2.json"
            file1.write_text('{"test": "data1"}')

            file2 = tmpdir_path / "config_v2.json"
            file2.write_text('{"test": "data2"}')

            # Create manifest
            manifest = manager.create_manifest(tmpdir_path)

            # Check manifest structure
            assert "file_naming_version" in manifest
            assert "created_at" in manifest
            assert "files" in manifest
            assert len(manifest["files"]) == 2

            # Check file entries
            for _rel_path, file_info in manifest["files"].items():
                assert "path" in file_info
                assert "size" in file_info
                assert "modified" in file_info
                assert "type" in file_info
                assert "metadata" in file_info
                assert "sha256" in file_info  # JSON files should have checksums

    def test_validate_manifest(self):
        """Test manifest validation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = FileVersionManager(version="2")
            tmpdir_path = Path(tmpdir)

            # Create test file and manifest
            file1 = tmpdir_path / "session_v2.json"
            file1.write_text('{"test": "data"}')
            manifest = manager.create_manifest(tmpdir_path)

            # Validate should pass
            report = manager.validate_manifest(tmpdir_path, manifest)
            assert report["valid"] is True
            assert len(report["missing_files"]) == 0
            assert len(report["checksum_mismatches"]) == 0

            # Modify file
            file1.write_text('{"test": "modified"}')

            # Validation should fail
            report = manager.validate_manifest(tmpdir_path, manifest)
            assert report["valid"] is False
            assert len(report["checksum_mismatches"]) > 0

    def test_create_manifest_skips_symlinks(self, caplog):
        """Ensure manifest creation ignores symlinks outside the run directory."""
        if not hasattr(os, "symlink"):
            pytest.skip("symlink support not available on this platform")

        with (
            tempfile.TemporaryDirectory() as run_dir,
            tempfile.TemporaryDirectory() as outside_dir,
        ):
            manager = FileVersionManager(version="2")
            run_path = Path(run_dir)
            outside_path = Path(outside_dir) / "secret.json"
            outside_path.write_text('{"secret": true}')

            safe_file = run_path / "session_v2.json"
            safe_file.write_text('{"ok": true}')

            symlink_path = run_path / "leak.json"
            try:
                symlink_path.symlink_to(outside_path)
            except OSError as exc:
                pytest.skip(f"symlink creation not permitted on this platform: {exc}")

            caplog.set_level("WARNING")
            manifest = manager.create_manifest(run_path)

            assert "leak.json" not in manifest["files"]
            assert "session_v2.json" in manifest["files"]
            assert any(
                "Skipping symlinked file" in record.message for record in caplog.records
            )

            report = manager.validate_manifest(run_path, manifest)
            assert report["valid"] is True
            assert "leak.json" not in report["extra_files"]


class TestRunVersionInfo:
    """Test RunVersionInfo functionality."""

    def test_initialization(self):
        """Test RunVersionInfo initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            run_path = Path(tmpdir)
            info = RunVersionInfo(run_path)

            assert info.run_path == run_path
            assert info.version_file == run_path / "meta" / "version_info.json"

    def test_create_version_info(self):
        """Test creating version info."""
        with tempfile.TemporaryDirectory() as tmpdir:
            run_path = Path(tmpdir)
            info = RunVersionInfo(run_path)

            version_data = info.create_version_info(
                traigent_version="1.0.0",
                file_naming_version="2.0.0",
                custom_metadata={"experiment": "test"},
            )

            assert version_data["traigent_version"] == "1.0.0"
            assert version_data["file_naming_version"] == "2.0.0"
            assert "python_version" in version_data
            assert "platform" in version_data
            assert version_data["custom_metadata"]["experiment"] == "test"

            # Check file was created
            assert info.version_file.exists()

    def test_load_version_info(self):
        """Test loading version info."""
        with tempfile.TemporaryDirectory() as tmpdir:
            run_path = Path(tmpdir)
            info = RunVersionInfo(run_path)

            # Should return None if no file
            assert info.load_version_info() is None

            # Create version info
            info.create_version_info("1.0.0")

            # Should load successfully
            loaded = info.load_version_info()
            assert loaded is not None
            assert loaded["traigent_version"] == "1.0.0"

    def test_check_compatibility(self):
        """Test compatibility checking."""
        with tempfile.TemporaryDirectory() as tmpdir:
            run_path = Path(tmpdir)
            info = RunVersionInfo(run_path)

            # No version info should be compatible with warning
            report = info.check_compatibility("1.0.0")
            assert report["compatible"] is True
            assert len(report["warnings"]) > 0

            # Create version info
            info.create_version_info("1.0.0")

            # Same major version should be compatible
            report = info.check_compatibility("1.5.0")
            assert report["compatible"] is True

            # Different major version should be incompatible
            report = info.check_compatibility("2.0.0")
            assert report["compatible"] is False
            assert len(report["warnings"]) > 0


class TestIntegration:
    """Integration tests for file versioning system."""

    def test_complete_workflow(self):
        """Test complete versioning workflow."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Initialize manager
            manager = FileVersionManager(version="2")

            # Create some files
            session_file = tmpdir_path / manager.get_filename("session")
            session_file.write_text('{"session": "data"}')

            config_file = tmpdir_path / manager.get_filename("config")
            config_file.write_text('{"config": "data"}')

            # Create version info
            version_info = RunVersionInfo(tmpdir_path)
            version_info.create_version_info("1.0.0")

            # Create manifest
            manifest = manager.create_manifest(tmpdir_path)

            # Validate everything
            report = manager.validate_manifest(tmpdir_path, manifest)
            assert report["valid"] is True
            assert version_info.version_file.exists()
            assert len(list(tmpdir_path.glob("*_v2.json"))) >= 2

    def test_legacy_migration_workflow(self):
        """Test migrating legacy files to versioned format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Create legacy files
            legacy_files = [
                ("session.json", {"type": "session"}),
                ("config.json", {"type": "config"}),
                ("objectives.json", {"type": "objectives"}),
            ]

            for filename, data in legacy_files:
                with open(tmpdir_path / filename, "w") as f:
                    json.dump(data, f)

            # Use the standalone migration function
            from traigent.utils.file_versioning import migrate_legacy_files

            # Dry run first
            report = migrate_legacy_files(tmpdir_path, dry_run=True)
            assert len(report["files_to_migrate"]) == 3
            assert len(report["files_migrated"]) == 0

            # Actual migration
            report = migrate_legacy_files(tmpdir_path, dry_run=False)
            assert len(report["files_migrated"]) == 3
            assert len(report["errors"]) == 0

            # Check new files exist
            for migrated_info in report["files_migrated"]:
                new_path = Path(migrated_info["new_path"])
                assert new_path.exists()
                assert "_v2" in new_path.name

    def test_multi_version_compatibility(self):
        """Test handling multiple versions simultaneously."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Create files with different versions
            v1_manager = FileVersionManager(version="1")
            v2_manager = FileVersionManager(version="2")
            v3_manager = FileVersionManager(version="3")

            # Create files for each version
            for manager in [v1_manager, v2_manager, v3_manager]:
                filename = manager.get_filename("session")
                filepath = tmpdir_path / filename
                filepath.write_text(f'{{"version": "{manager.version}"}}')

            # All versions should be created
            all_files = list(tmpdir_path.glob("session*.json"))
            assert len(all_files) == 3

            # Test parsing different versions
            for filepath in all_files:
                file_type, metadata = v2_manager.parse_filename(filepath.name)
                assert file_type == "session"
                assert "version" in metadata
