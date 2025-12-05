"""Tests for reproducibility metadata collection."""

import copy
import json
import random
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from traigent.utils.reproducibility import (
    ReproducibilityMetadata,
    ensure_reproducibility,
)


class TestReproducibilityMetadata:
    """Test reproducibility metadata collection."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_initialization(self, temp_dir):
        """Test metadata collector initialization."""
        collector = ReproducibilityMetadata(temp_dir)

        assert collector.run_path == temp_dir
        assert isinstance(collector.metadata, dict)
        assert "timestamp" in collector.metadata
        assert "environment" in collector.metadata
        assert "python" in collector.metadata
        assert "dependencies" in collector.metadata
        assert "random_state" in collector.metadata
        assert "hardware" in collector.metadata
        assert "git" in collector.metadata
        assert "traigent" in collector.metadata

    def test_collect_environment(self, temp_dir):
        """Test environment metadata collection."""
        collector = ReproducibilityMetadata(temp_dir)
        env_info = collector._collect_environment()

        assert "variables" in env_info
        assert "platform" in env_info
        assert "hostname" in env_info
        assert "user" in env_info
        assert "cwd" in env_info

        # Check that sensitive vars are not included
        assert "AWS_SECRET_ACCESS_KEY" not in env_info["variables"]
        assert "OPENAI_API_KEY" not in env_info["variables"]

    def test_collect_python_info(self, temp_dir):
        """Test Python information collection."""
        collector = ReproducibilityMetadata(temp_dir)
        python_info = collector._collect_python_info()

        assert "version" in python_info
        assert "version_info" in python_info
        assert "executable" in python_info
        assert "implementation" in python_info
        assert "compiler" in python_info

        # Check version_info structure
        version_info = python_info["version_info"]
        assert "major" in version_info
        assert "minor" in version_info
        assert "micro" in version_info

    def test_collect_dependencies(self, temp_dir):
        """Test dependency collection."""
        collector = ReproducibilityMetadata(temp_dir)
        deps = collector._collect_dependencies()

        assert isinstance(deps, dict)
        # At least numpy should be installed in test environment
        assert "numpy" in deps or len(deps) > 0

    def test_collect_random_state(self, temp_dir):
        """Test random state collection."""
        collector = ReproducibilityMetadata(temp_dir)
        random_info = collector._collect_random_state()

        assert "python_random_state" in random_info
        assert "numpy_random_state" in random_info
        assert "seeds_captured" in random_info

        # Should successfully capture seeds in test environment
        assert random_info["seeds_captured"] is True
        assert isinstance(random_info["python_random_state"], dict)
        assert isinstance(random_info["numpy_random_state"], dict)

    def test_random_state_round_trip(self, temp_dir):
        """Random state capture should serialize and restore correctly."""
        collector = ReproducibilityMetadata(temp_dir)

        # Set deterministic seeds
        random.seed(1234)
        np.random.seed(5678)

        state = collector._collect_random_state()

        # Change RNG states
        random.random()
        np.random.rand()

        # Restore
        collector.restore_random_state(state)

        # Verify reproducibility
        random_values = [random.random() for _ in range(3)]
        np_values = np.random.rand(3).tolist()

        # Reset again from same state to compare
        collector.restore_random_state(state)
        assert random_values == [random.random() for _ in range(3)]
        assert np.allclose(np_values, np.random.rand(3))

    def test_collect_hardware_info(self, temp_dir):
        """Test hardware information collection."""
        collector = ReproducibilityMetadata(temp_dir)
        hardware = collector._collect_hardware_info()

        assert "processor" in hardware
        assert "architecture" in hardware
        assert "cpu_count" in hardware
        assert "memory" in hardware
        assert "gpu" in hardware

        # CPU count should be available
        assert hardware["cpu_count"] is not None
        assert hardware["cpu_count"] > 0

    @patch("subprocess.run")
    def test_collect_git_info(self, mock_run, temp_dir):
        """Test git information collection."""
        # Mock git commands - need enough for both init and test calls
        mock_run.side_effect = [
            # First call during initialization
            MagicMock(returncode=0, stdout=""),  # git rev-parse --git-dir
            MagicMock(returncode=0, stdout="abc123def456\n"),  # git rev-parse HEAD
            MagicMock(returncode=0, stdout="main\n"),  # git rev-parse --abbrev-ref HEAD
            MagicMock(returncode=0, stdout=""),  # git status --porcelain
            MagicMock(
                returncode=0, stdout="https://github.com/test/repo.git\n"
            ),  # git remote get-url origin
            # Second call during test
            MagicMock(returncode=0, stdout=""),  # git rev-parse --git-dir
            MagicMock(returncode=0, stdout="abc123def456\n"),  # git rev-parse HEAD
            MagicMock(returncode=0, stdout="main\n"),  # git rev-parse --abbrev-ref HEAD
            MagicMock(returncode=0, stdout=""),  # git status --porcelain
            MagicMock(
                returncode=0, stdout="https://github.com/test/repo.git\n"
            ),  # git remote get-url origin
        ]

        collector = ReproducibilityMetadata(temp_dir)
        git_info = collector._collect_git_info()

        assert git_info["available"] is True
        assert git_info["commit"] == "abc123def456"
        assert git_info["branch"] == "main"
        assert git_info["dirty"] is False
        assert git_info["remote"] == "https://github.com/test/repo.git"

    def test_collect_traigent_info(self, temp_dir):
        """Test TraiGent information collection."""
        collector = ReproducibilityMetadata(temp_dir)
        traigent_info = collector._collect_traigent_info()

        assert "version" in traigent_info
        assert "config_path" in traigent_info
        assert "cache_dir" in traigent_info

    def test_compute_dataset_checksum(self, temp_dir):
        """Test dataset checksum computation."""
        collector = ReproducibilityMetadata(temp_dir)

        # Create test dataset file
        dataset_path = temp_dir / "test_dataset.jsonl"
        dataset_path.write_text('{"input": "test", "output": "result"}\n')

        checksum = collector.compute_dataset_checksum(dataset_path)

        assert isinstance(checksum, str)
        assert len(checksum) == 64  # SHA256 produces 64 hex characters

        # Same content should produce same checksum
        checksum2 = collector.compute_dataset_checksum(dataset_path)
        assert checksum == checksum2

        # Different content should produce different checksum
        dataset_path.write_text('{"input": "different", "output": "data"}\n')
        checksum3 = collector.compute_dataset_checksum(dataset_path)
        assert checksum != checksum3

    def test_add_dataset_info(self, temp_dir):
        """Test adding dataset information."""
        collector = ReproducibilityMetadata(temp_dir)

        # Create test dataset
        dataset_path = temp_dir / "dataset.jsonl"
        dataset_path.write_text('{"test": "data"}\n')

        collector.add_dataset_info(dataset_path)

        assert "dataset" in collector.metadata
        dataset_info = collector.metadata["dataset"]
        assert dataset_info["path"] == str(dataset_path)
        assert dataset_info["exists"] is True
        assert dataset_info["size_bytes"] > 0
        assert dataset_info["checksum"] is not None
        assert dataset_info["modified"] is not None

    def test_add_custom_metadata(self, temp_dir):
        """Test adding custom metadata."""
        collector = ReproducibilityMetadata(temp_dir)

        collector.add_custom_metadata("experiment_type", "hyperparameter_tuning")
        collector.add_custom_metadata("model_family", "transformer")

        assert "custom" in collector.metadata
        assert (
            collector.metadata["custom"]["experiment_type"] == "hyperparameter_tuning"
        )
        assert collector.metadata["custom"]["model_family"] == "transformer"

    def test_save_and_load(self, temp_dir):
        """Test saving and loading metadata."""
        collector = ReproducibilityMetadata(temp_dir)

        # Add some custom data
        collector.add_custom_metadata("test_key", "test_value")

        # Save metadata
        metadata_path = collector.save()
        assert metadata_path.exists()
        assert metadata_path.name == "reproducibility.json"

        # Load metadata
        loaded = ReproducibilityMetadata.load(metadata_path)

        assert isinstance(loaded, dict)
        assert "timestamp" in loaded
        assert "environment" in loaded
        assert "custom" in loaded
        assert loaded["custom"]["test_key"] == "test_value"

    def test_validate_environment(self, temp_dir):
        """Test environment validation."""
        collector1 = ReproducibilityMetadata(temp_dir)
        metadata1 = collector1.metadata

        # Create slightly different metadata
        collector2 = ReproducibilityMetadata(temp_dir)
        metadata2 = copy.deepcopy(collector2.metadata)

        # Test compatible environment
        report = collector1.validate_environment(metadata1)
        assert report["compatible"] is True
        assert len(report["errors"]) == 0

        # Test Python major version mismatch
        metadata2["python"]["version_info"]["major"] = 2
        report = collector1.validate_environment(metadata2)
        assert report["compatible"] is False
        assert len(report["errors"]) > 0
        assert "Python major version mismatch" in report["errors"][0]

        # Test dependency version mismatch
        metadata3 = copy.deepcopy(collector1.metadata)  # Start fresh from collector1
        # Only test if numpy is actually installed
        if "numpy" in metadata3["dependencies"]:
            metadata3["dependencies"]["numpy"] = "0.0.1"
            report = collector1.validate_environment(metadata3)
            assert report["compatible"] is True  # Warnings only
            assert len(report["warnings"]) > 0
            assert "numpy version mismatch" in report["warnings"][0]
        else:
            # If numpy is not installed, test with traigent instead
            if "traigent" in metadata3["dependencies"]:
                metadata3["dependencies"]["traigent"] = "0.0.1"
                report = collector1.validate_environment(metadata3)
                # traigent is a critical package, might cause incompatibility
                assert len(report["warnings"]) > 0

    def test_ensure_reproducibility(self, temp_dir):
        """Test the convenience function."""
        # Create test dataset
        dataset_path = temp_dir / "test.jsonl"
        dataset_path.write_text('{"test": "data"}\n')

        # Custom metadata
        custom = {
            "experiment_name": "test_exp",
            "researcher": "test_user",
        }

        # Ensure reproducibility
        metadata_path = ensure_reproducibility(
            run_path=temp_dir,
            dataset_path=dataset_path,
            custom_metadata=custom,
        )

        assert metadata_path.exists()

        # Load and verify
        with open(metadata_path) as f:
            metadata = json.load(f)

        assert "dataset" in metadata
        assert metadata["dataset"]["path"] == str(dataset_path)
        assert "custom" in metadata
        assert metadata["custom"]["experiment_name"] == "test_exp"
        assert metadata["custom"]["researcher"] == "test_user"

    def test_serialization_of_complex_objects(self, temp_dir):
        """Test that complex objects are properly serialized."""
        collector = ReproducibilityMetadata(temp_dir)

        # The random state might contain tuples and other non-serializable objects
        # This test ensures they are converted properly
        metadata_path = collector.save()

        # Should be able to load without errors
        with open(metadata_path) as f:
            loaded = json.load(f)

        assert isinstance(loaded, dict)
        # Random state should be converted to strings if it was captured
        if loaded["random_state"]["seeds_captured"]:
            python_state = loaded["random_state"]["python_random_state"]
            numpy_state = loaded["random_state"]["numpy_random_state"]

            assert isinstance(python_state, dict)
            assert set(python_state.keys()) == {"version", "state", "gauss"}
            assert isinstance(python_state["state"], list)

            assert isinstance(numpy_state, dict)
            assert set(numpy_state.keys()) == {
                "version",
                "keys",
                "pos",
                "has_gauss",
                "cached_gaussian",
            }
