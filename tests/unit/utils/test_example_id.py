"""Unit tests for stable example ID generation."""

import pytest

from traigent.utils.example_id import compute_dataset_hash, generate_stable_example_id


class TestComputeDatasetHash:
    """Test dataset hash computation."""

    def test_deterministic_hash(self):
        """Hash should be deterministic for same input."""
        dataset_name = "customer_support_qa"
        hash1 = compute_dataset_hash(dataset_name)
        hash2 = compute_dataset_hash(dataset_name)
        assert hash1 == hash2

    def test_hash_length(self):
        """Hash should be exactly 12 characters."""
        dataset_name = "test_dataset"
        dataset_hash = compute_dataset_hash(dataset_name)
        assert len(dataset_hash) == 12

    def test_hash_format(self):
        """Hash should be hexadecimal."""
        dataset_name = "test_dataset"
        dataset_hash = compute_dataset_hash(dataset_name)
        # All characters should be valid hex digits
        assert all(c in "0123456789abcdef" for c in dataset_hash)

    def test_different_names_different_hashes(self):
        """Different dataset names should produce different hashes."""
        hash1 = compute_dataset_hash("dataset_a")
        hash2 = compute_dataset_hash("dataset_b")
        assert hash1 != hash2


class TestGenerateStableExampleId:
    """Test stable example ID generation."""

    def test_deterministic_id(self):
        """ID should be deterministic for same inputs."""
        dataset_hash = "a3f4b2c891de"
        example_index = 42
        id1 = generate_stable_example_id(dataset_hash, example_index)
        id2 = generate_stable_example_id(dataset_hash, example_index)
        assert id1 == id2

    def test_id_format(self):
        """ID should follow format: ex_{hash}_{index}."""
        dataset_hash = "a3f4b2c891de"
        example_index = 0
        example_id = generate_stable_example_id(dataset_hash, example_index)
        assert example_id.startswith("ex_")
        assert example_id.endswith("_0")
        parts = example_id.split("_")
        assert len(parts) == 3
        assert parts[0] == "ex"
        assert len(parts[1]) == 8  # Hash component is 8 chars
        assert parts[2] == "0"

    def test_different_indices_different_ids(self):
        """Different indices should produce different IDs."""
        dataset_hash = "a3f4b2c891de"
        id1 = generate_stable_example_id(dataset_hash, 0)
        id2 = generate_stable_example_id(dataset_hash, 1)
        assert id1 != id2
        # But they should share the same prefix pattern
        assert id1.split("_")[0] == id2.split("_")[0] == "ex"

    def test_different_datasets_different_ids(self):
        """Same index but different datasets should produce different IDs."""
        hash1 = compute_dataset_hash("dataset_a")
        hash2 = compute_dataset_hash("dataset_b")
        id1 = generate_stable_example_id(hash1, 0)
        id2 = generate_stable_example_id(hash2, 0)
        assert id1 != id2

    def test_large_index(self):
        """Should handle large example indices."""
        dataset_hash = "a3f4b2c891de"
        example_index = 9999
        example_id = generate_stable_example_id(dataset_hash, example_index)
        assert example_id.endswith("_9999")
        assert example_id.startswith("ex_")


class TestEndToEndExampleIdGeneration:
    """Test complete workflow from dataset name to example IDs."""

    def test_complete_workflow(self):
        """Test generating IDs for multiple examples in a dataset."""
        dataset_name = "my_evaluation_set"
        dataset_hash = compute_dataset_hash(dataset_name)

        # Generate IDs for 5 examples
        example_ids = [generate_stable_example_id(dataset_hash, i) for i in range(5)]

        # All IDs should be unique
        assert len(example_ids) == len(set(example_ids))

        # All IDs should follow the correct format
        for example_id in example_ids:
            assert example_id.startswith("ex_")
            parts = example_id.split("_")
            assert len(parts) == 3

    def test_stability_across_runs(self):
        """IDs should be stable across multiple 'runs' with same dataset."""
        dataset_name = "stable_dataset"

        # Simulate first run
        hash1 = compute_dataset_hash(dataset_name)
        ids_run1 = [generate_stable_example_id(hash1, i) for i in range(10)]

        # Simulate second run (same dataset)
        hash2 = compute_dataset_hash(dataset_name)
        ids_run2 = [generate_stable_example_id(hash2, i) for i in range(10)]

        # IDs should be identical
        assert ids_run1 == ids_run2
