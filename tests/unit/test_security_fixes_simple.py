"""Simplified test suite for security vulnerability fixes."""

import gzip
import hashlib
import json
import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from traigent.cloud.backend_client import BackendIntegratedClient


class TestSecurityFixes:
    """Test that critical security vulnerabilities have been fixed."""

    def test_backend_client_uses_sha256_not_md5(self):
        """Verify that backend client uses SHA256 instead of MD5 for hashing."""
        client = BackendIntegratedClient(base_url="http://test.local")

        # Test the trial ID generation method
        session_id = "test_session"
        config = {"param1": "value1", "param2": 42}
        metadata = {"test": "data"}

        trial_id = client._generate_trial_id(session_id, config, metadata)

        # Verify the trial_id format follows the expected pattern
        assert trial_id.startswith("trial_"), (
            f"Trial ID should start with 'trial_', got: {trial_id}"
        )

        # Verify it's using SHA256 by checking the implementation logic
        # The generate_trial_hash function creates hash from session_id:config:dataset_name
        sorted_config = json.dumps(config, sort_keys=True)
        hash_input = f"{session_id}:{sorted_config}:"  # empty dataset_name
        expected_hash = hashlib.sha256(hash_input.encode()).hexdigest()[:12]
        expected_trial_id = f"trial_{expected_hash}"

        assert trial_id == expected_trial_id, (
            f"Expected {expected_trial_id}, got {trial_id}"
        )

        legacy_md5_trial_id = "trial_f05e53e940fe"
        assert trial_id != legacy_md5_trial_id, "Should not use MD5 hash"

    def test_persistence_secure_pickle_loading(self):
        """Test that persistence module handles pickle securely."""
        from traigent.api.types import OptimizationResult, OptimizationStatus
        from traigent.utils.persistence import PersistenceManager

        with tempfile.TemporaryDirectory() as tmpdir:
            persistence = PersistenceManager(tmpdir)

            # Create a mock optimization result
            result = OptimizationResult(
                trials=[],
                best_config={"param": "value"},
                best_score=0.95,
                optimization_id="test_opt",
                duration=100.0,
                convergence_info={},
                status=OptimizationStatus.COMPLETED,
                objectives=["accuracy"],
                algorithm="random",
                timestamp=datetime.now(),
            )

            # Save the result - should create JSON file
            save_path = persistence.save_result(result, "test_result")

            # Verify JSON file was created (secure format)
            result_dir = Path(save_path)
            json_file = result_dir / "trials.json.gz"
            assert json_file.exists(), "Should save as JSON for security"

            # Verify JSON content is valid and not pickled
            with gzip.open(json_file, "rt") as f:
                data = json.load(f)
                assert isinstance(data, list), "Should be JSON list, not pickled object"

    def test_sha256_in_subset_selection(self):
        """Verify subset selection uses SHA256 for hashing."""
        # Read the source file to verify SHA256 is used
        source_file = Path("traigent/cloud/subset_selection.py")
        if source_file.exists():
            content = source_file.read_text()

            # Check that SHA256 is used, not MD5
            assert "hashlib.sha256" in content, "Should use SHA256 for hashing"
            assert "hashlib.md5" not in content, "Should not use MD5"

    def test_sha256_in_encryption(self):
        """Verify encryption module uses SHA256 for hashing."""
        # Read the source file to verify SHA256 is used
        source_file = Path("traigent/security/encryption.py")
        if source_file.exists():
            content = source_file.read_text()

            # Check that SHA256 is used where hashing is needed
            assert "hashlib.sha256" in content or "SHA256" in content, (
                "Should use SHA256"
            )
            assert "hashlib.md5" not in content, "Should not use MD5"

    def test_random_seeding_in_optimizer(self):
        """Test that random optimizer uses proper seeding."""
        from traigent.core.types import ConfigurationSpace, Parameter, ParameterType
        from traigent.optimizers.random import RandomSearchOptimizer

        config_space = ConfigurationSpace(
            parameters=[
                Parameter(
                    name="test_param", type=ParameterType.FLOAT, bounds=(0.0, 1.0)
                )
            ]
        )

        # Create optimizer with seed
        optimizer = RandomSearchOptimizer(
            config_space=config_space, objectives=["accuracy"], random_seed=42
        )

        # Should have its own Random instance (not using global random)
        assert hasattr(optimizer, "_random")

        # Generate configs - should be reproducible with same seed
        config1 = optimizer.suggest_next_trial([])

        # Create another optimizer with same seed
        optimizer2 = RandomSearchOptimizer(
            config_space=config_space, objectives=["accuracy"], random_seed=42
        )

        config2 = optimizer2.suggest_next_trial([])

        # Should generate same config with same seed (reproducible)
        assert config1 == config2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
