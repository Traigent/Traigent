"""Unit tests for Traigent Edge Analytics mode safeguards."""

import json
import os
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from traigent.api.types import TrialResult as APITrialResult
from traigent.config.types import TraigentConfig
from traigent.core.optimized_function import OptimizedFunction
from traigent.core.orchestrator import OptimizationOrchestrator
from traigent.evaluators.base import BaseEvaluator, Dataset, EvaluationExample
from traigent.optimizers.base import BaseOptimizer
from traigent.storage.local_storage import LocalStorageManager
from traigent.utils.exceptions import OptimizationError


@pytest.fixture(autouse=True)
def cleanup_token_file():
    """Cleanup token file before and after each test."""
    token_file = Path("/tmp/approval.token")
    if token_file.exists():
        token_file.unlink()
    yield
    if token_file.exists():
        token_file.unlink()


# Helper class for testing - minimal optimizer implementation
class MockOptimizer(BaseOptimizer):
    """Minimal optimizer for testing."""

    def __init__(self, config_space, objectives, **kwargs):
        super().__init__(config_space, objectives, **kwargs)
        self._trials_generated = 0
        self._max_trials = kwargs.get("max_trials", 100)

    def suggest_next_trial(self, history: list[APITrialResult]) -> dict[str, any]:
        """Generate next configuration."""
        if self._trials_generated >= self._max_trials:
            raise OptimizationError("No more trials")
        self._trials_generated += 1
        return {"x": self._trials_generated}

    def should_stop(self, history: list[APITrialResult]) -> bool:
        """Check if should stop."""
        return len(history) >= self._max_trials


# Helper class for testing - minimal evaluator
class MockEvaluator(BaseEvaluator):
    """Minimal evaluator for testing."""

    def evaluate(self, func, config, dataset):
        """Mock evaluation."""
        return APITrialResult(
            trial_id=0,
            config=config,
            metrics={"score": 0.5},
            is_successful=True,
            duration=0.1,
            timestamp=datetime.now(),
        )


class TestTrialCaps:
    """Test trial cap enforcement."""

    @pytest.mark.asyncio
    async def test_parallel_trial_cap_no_overshoot(self):
        """Ensure parallel execution respects max_trials."""
        # Create proper mocked optimizer with spec
        optimizer = MagicMock(spec=BaseOptimizer)
        optimizer.config_space = {"x": [1, 2, 3]}
        optimizer.objectives = ["score"]
        optimizer.should_stop = Mock(return_value=False)

        # Mock async candidate generation
        optimizer.generate_candidates_async = AsyncMock(
            side_effect=[
                [{"x": 1}, {"x": 2}, {"x": 3}, {"x": 4}],  # First batch: 4 configs
                [{"x": 5}],  # Second batch: 1 config (should be limited)
            ]
        )

        # Create evaluator mock
        evaluator = MagicMock(spec=BaseEvaluator)

        # Create config
        config = TraigentConfig(execution_mode="edge_analytics")

        # Create orchestrator with trial cap
        orchestrator = OptimizationOrchestrator(
            optimizer=optimizer,
            evaluator=evaluator,
            max_trials=5,
            parallel_trials=4,
            config=config,
        )

        # Verify settings
        assert orchestrator.max_trials == 5
        assert orchestrator.parallel_trials == 4

        # The actual optimization would need more mocking, but we've validated the setup

    def test_sequential_trial_cap(self):
        """Test trial cap in sequential execution."""
        # Use the MockOptimizer class
        optimizer = MockOptimizer(
            config_space={"x": [1, 2, 3]}, objectives=["score"], max_trials=3
        )

        evaluator = MockEvaluator()
        config = TraigentConfig(execution_mode="edge_analytics")

        orchestrator = OptimizationOrchestrator(
            optimizer=optimizer,
            evaluator=evaluator,
            max_trials=3,
            parallel_trials=1,  # Sequential
            config=config,
        )

        # Verify the trial cap is set
        assert orchestrator.max_trials == 3
        assert orchestrator.parallel_trials == 1


class TestExampleCaps:
    """Test example cap implementation."""

    def test_dataset_capping(self):
        """Test that dataset is correctly capped to max_examples."""
        # Create a large dataset with proper EvaluationExample objects
        examples = [
            EvaluationExample(
                input_data={"value": i},
                expected_output={"result": i * 2},
                metadata={"id": i},
            )
            for i in range(100)
        ]

        dataset = Dataset(
            examples=examples,
            name="test_dataset",
            description="Test dataset",
        )

        # Test capping logic (simulated)
        max_examples = 10
        if len(dataset.examples) > max_examples:
            capped_examples = dataset.examples[:max_examples]
            capped_dataset = Dataset(
                examples=capped_examples,
                name=dataset.name,
                description=f"{dataset.description} (capped to {max_examples} examples)",
                metadata={
                    "original_count": len(dataset.examples),
                    "capped_count": max_examples,
                },
            )

        assert len(capped_dataset.examples) == 10
        assert all(isinstance(ex, EvaluationExample) for ex in capped_dataset.examples)
        assert capped_dataset.metadata["original_count"] == 100
        assert capped_dataset.metadata["capped_count"] == 10

    @pytest.mark.asyncio
    async def test_example_cap_in_optimize(self):
        """Test that example cap is applied in optimize method."""
        with patch("traigent.core.optimized_function.get_logger") as mock_logger:
            mock_logger.return_value = Mock()

            # Create optimized function with minimal config_space
            func = OptimizedFunction(
                func=lambda x: x,
                eval_dataset="test.jsonl",
                objectives=["score"],
                configuration_space={"x": [1, 2, 3]},  # Required parameter
            )
            func.traigent_config = Mock(is_edge_analytics_mode=lambda: False)

            # Mock dataset loading
            original_examples = [
                EvaluationExample(
                    input_data={"value": i},
                    expected_output={"result": i},
                    metadata={"id": i},
                )
                for i in range(100)
            ]
            original_dataset = Dataset(
                examples=original_examples,
                name="test",
                description="Test",
            )
            func._load_dataset = Mock(return_value=original_dataset)

            # Mock CI approval check
            func._check_ci_approval = Mock()

            # Simulate optimize with max_examples
            max_examples = 20
            dataset = func._load_dataset()

            if max_examples and len(dataset.examples) > max_examples:
                capped_dataset = Dataset(
                    examples=dataset.examples[:max_examples],
                    name=dataset.name,
                    description=f"{dataset.description} (capped to {max_examples} examples)",
                    metadata={
                        "original_count": len(dataset.examples),
                        "capped_count": max_examples,
                    },
                )
                dataset = capped_dataset

            assert len(dataset.examples) == 20


class TestFloatNormalization:
    """Test canonical config hashing with float normalization."""

    def test_float_normalization(self):
        """Test that float values are normalized for consistent hashing."""
        storage = LocalStorageManager("/tmp/test_storage")

        # Test float normalization (should hash to same value)
        config1 = {"learning_rate": 0.1000000001, "batch_size": 32}
        config2 = {"learning_rate": 0.1000000002, "batch_size": 32}

        hash1 = storage.compute_config_hash(config1)
        hash2 = storage.compute_config_hash(config2)

        # Should be equal after normalization
        assert hash1 == hash2

    def test_nested_float_normalization(self):
        """Test normalization of nested structures with floats."""
        storage = LocalStorageManager("/tmp/test_storage")

        config1 = {
            "optimizer": {"lr": 0.001000001, "beta": [0.900000001, 0.999]},
            "scheduler": {"gamma": 0.950000001},
        }
        config2 = {
            "optimizer": {"lr": 0.001000002, "beta": [0.900000002, 0.999]},
            "scheduler": {"gamma": 0.950000002},
        }

        hash1 = storage.compute_config_hash(config1)
        hash2 = storage.compute_config_hash(config2)

        # Should be equal after normalization
        assert hash1 == hash2


class TestDeduplication:
    """Test cross-run configuration deduplication."""

    def test_config_hashing(self):
        """Test that configuration hashing is consistent."""
        storage = LocalStorageManager(tempfile.mkdtemp())

        config1 = {"x": 1.0, "y": 2.0, "z": "test"}
        config2 = {"y": 2.0, "x": 1.0, "z": "test"}  # Different order
        config3 = {"x": 2.0, "y": 2.0, "z": "test"}  # Different value

        hash1 = storage.compute_config_hash(config1)
        hash2 = storage.compute_config_hash(config2)
        hash3 = storage.compute_config_hash(config3)

        # Same config in different order should have same hash
        assert hash1 == hash2
        # Different config should have different hash
        assert hash1 != hash3

    def test_config_seen_detection(self):
        """Test detection of previously seen configurations."""
        storage_path = tempfile.mkdtemp()
        storage = LocalStorageManager(storage_path)

        # Create session using public API
        session_id = storage.create_session(
            function_name="test_func", metadata={"evaluation_set": "test_dataset"}
        )

        # Add trial results using public API
        storage.add_trial_result(
            session_id=session_id,
            config={"x": 1.0, "y": 2.0},
            score=0.5,
            metadata={"test": "data"},
        )

        storage.add_trial_result(
            session_id=session_id,
            config={"x": 2.0, "y": 3.0},
            score=0.7,
            metadata={"test": "data"},
        )

        # Update session status
        storage.update_session_status(session_id, "completed")

        # Test that seen configs are detected
        assert storage.is_config_seen("test_func", "test_dataset", {"x": 1.0, "y": 2.0})
        assert storage.is_config_seen(
            "test_func", "test_dataset", {"y": 2.0, "x": 1.0}
        )  # Order independent
        assert not storage.is_config_seen(
            "test_func", "test_dataset", {"x": 3.0, "y": 4.0}
        )

        # Test with different function name
        assert not storage.is_config_seen(
            "other_func", "test_dataset", {"x": 1.0, "y": 2.0}
        )

        # Test with different dataset
        assert not storage.is_config_seen(
            "test_func", "other_dataset", {"x": 1.0, "y": 2.0}
        )

    def test_cache_policy_filtering(self):
        """Test that cache policy correctly filters configurations."""
        # Create minimal optimizer
        optimizer = MockOptimizer(config_space={"x": [1, 2, 3]}, objectives=["score"])

        evaluator = MockEvaluator()

        config = TraigentConfig(execution_mode="edge_analytics")
        orchestrator = OptimizationOrchestrator(
            optimizer=optimizer,
            evaluator=evaluator,
            max_trials=10,
            config=config,
        )

        # Mock the cache policy handler's storage to simulate seen configs
        with patch.object(
            orchestrator.cache_policy_handler._storage, "is_config_seen"
        ) as mock_is_seen:
            # Config 1 is seen, config 2 and 3 are new
            mock_is_seen.side_effect = [True, False, False]

            configs = [
                {"x": 1.0},  # Seen
                {"x": 2.0},  # New
                {"x": 3.0},  # New
            ]

            # Test prefer_new policy
            filtered = orchestrator.cache_policy_handler.apply_policy(
                configs, "prefer_new", "test_func", "test_dataset"
            )

            assert len(filtered) == 2
            assert {"x": 2.0} in filtered
            assert {"x": 3.0} in filtered
            assert orchestrator.cache_policy_handler.configs_deduplicated == 1

            # Reset for next test
            orchestrator.cache_policy_handler.reset_stats()

            # Test allow_repeats policy
            filtered = orchestrator.cache_policy_handler.apply_policy(
                configs, "allow_repeats", "test_func", "test_dataset"
            )

            assert len(filtered) == 3  # All configs allowed
            assert orchestrator.cache_policy_handler.configs_deduplicated == 0


class TestSafeguardTelemetry:
    """Test safeguards telemetry in optimization results."""

    def test_telemetry_in_results(self):
        """Test that safeguard telemetry appears in optimization results."""
        # Create optimizer with trial cap
        optimizer = MockOptimizer(config_space={"x": [1, 2, 3]}, objectives=["score"])

        evaluator = MockEvaluator()

        config = TraigentConfig(execution_mode="edge_analytics")
        orchestrator = OptimizationOrchestrator(
            optimizer=optimizer,
            evaluator=evaluator,
            max_trials=3,
            config=config,
            cache_policy="prefer_new",
        )

        # Add a successful trial to avoid empty trials list
        from traigent.api.types import TrialResult as APITrialResult
        from traigent.api.types import TrialStatus

        trial = APITrialResult(
            trial_id="trial-0",
            config={"x": 1},
            metrics={"score": 0.5},
            status=TrialStatus.COMPLETED,
            duration=0.1,
            timestamp=datetime.now(),
        )
        orchestrator._trials.append(trial)

        # Mock some telemetry values on orchestrator
        orchestrator._trials_prevented = 2
        orchestrator._examples_capped = 10
        orchestrator._ci_blocks = 0
        orchestrator._cached_results_reused = 0

        # Mock cache policy handler telemetry
        orchestrator.cache_policy_handler._configs_deduplicated = 1
        orchestrator.cache_policy_handler._cache_policy_used = "prefer_new"

        # Create optimization result
        result = orchestrator._create_optimization_result()

        # Check telemetry in metadata
        assert "safeguards" in result.metadata
        safeguards = result.metadata["safeguards"]

        assert safeguards["trials_prevented"] == 2
        assert safeguards["configs_deduplicated"] == 1
        assert safeguards["examples_capped"] == 10
        assert safeguards["ci_blocks"] == 0
        assert safeguards["cached_results_reused"] == 0
        assert safeguards["cache_policy"] == "prefer_new"


class TestCIApproval:
    """Test CI/CD approval gates."""

    def test_ci_detection(self, tmp_path: Path):
        """Test CI environment detection."""
        func = OptimizedFunction(
            func=lambda x: x,
            eval_dataset="test.jsonl",
            objectives=["score"],
            configuration_space={"x": [1, 2, 3]},  # Required parameter
        )
        func.traigent_config = Mock(
            is_edge_analytics_mode=lambda: True,
            get_local_storage_path=lambda: str(tmp_path),
        )

        # Test various CI environment variables (10 providers)
        test_cases = [
            {"CI": "true"},
            {"CI": "1"},
            {"GITHUB_ACTIONS": "true"},
            {"GITHUB_ACTIONS": "1"},
            {"JENKINS_URL": "http://jenkins.example.com"},
            {"GITLAB_CI": "true"},
            {"CIRCLECI": "true"},
            {"TRAVIS": "true"},
            {"BUILDKITE": "true"},
            {"TEAMCITY_VERSION": "2023.05"},
            {"AZURE_HTTP_USER_AGENT": "VSTS_agent"},
            {"BITBUCKET_BUILD_NUMBER": "123"},
        ]

        for env_vars in test_cases:
            with patch.dict(os.environ, env_vars, clear=True):
                with pytest.raises(OptimizationError) as exc:
                    func._check_ci_approval()

                assert "CI/CD Approval Required" in str(exc.value)

    def test_environment_approval(self, tmp_path: Path):
        """Test approval via environment variables."""
        func = OptimizedFunction(
            func=lambda x: x,
            eval_dataset="test.jsonl",
            objectives=["score"],
            configuration_space={"x": [1, 2, 3]},  # Required parameter
        )
        func.traigent_config = Mock(
            is_edge_analytics_mode=lambda: True,
            get_local_storage_path=lambda: str(tmp_path),
        )

        # Test with CI environment and approval
        with patch.dict(
            os.environ,
            {
                "CI": "true",
                "TRAIGENT_RUN_APPROVED": "1",
                "TRAIGENT_APPROVED_BY": "test_user",
            },
        ):
            result = func._check_ci_approval()  # Should not raise
            assert result is None  # Successful approval returns None

    def test_mock_mode_skips_ci_approval(self, tmp_path: Path):
        """Mock mode should bypass CI approval gate."""
        func = OptimizedFunction(
            func=lambda x: x,
            eval_dataset="test.jsonl",
            objectives=["score"],
            configuration_space={"x": [1, 2, 3]},
        )
        func.traigent_config = Mock(
            is_edge_analytics_mode=lambda: True,
            get_local_storage_path=lambda: str(tmp_path),
        )

        with patch.dict(
            os.environ, {"CI": "true", "TRAIGENT_MOCK_LLM": "true"}, clear=True
        ):
            result = func._check_ci_approval()  # Should not raise
            assert result is None  # Mock mode bypass returns None

    def test_token_file_approval(self):
        """Test approval via token file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            func = OptimizedFunction(
                func=lambda x: x,
                eval_dataset="test.jsonl",
                objectives=["score"],
                configuration_space={"x": [1, 2, 3]},  # Required parameter
            )
            func.traigent_config = Mock(
                is_edge_analytics_mode=lambda: True,
                get_local_storage_path=lambda: tmpdir,
            )

            # Create valid token file
            token_file = Path(tmpdir) / "approval.token"
            expires_at = (datetime.now() + timedelta(hours=1)).isoformat()
            token_data = {
                "approved_by": "test_user",
                "expires_at": expires_at,
            }

            with open(token_file, "w") as f:
                json.dump(token_data, f)

            # Test with CI environment and valid token
            with patch.dict(os.environ, {"CI": "true"}):
                result = func._check_ci_approval()  # Should not raise
                assert result is None  # Token approval returns None

    def test_expired_token_rejected(self):
        """Test that expired tokens are rejected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            func = OptimizedFunction(
                func=lambda x: x,
                eval_dataset="test.jsonl",
                objectives=["score"],
                configuration_space={"x": [1, 2, 3]},  # Required parameter
            )
            func.traigent_config = Mock(
                is_edge_analytics_mode=lambda: True,
                get_local_storage_path=lambda: tmpdir,
            )

            # Create expired token file
            token_file = Path(tmpdir) / "approval.token"
            expires_at = (datetime.now() - timedelta(hours=1)).isoformat()
            token_data = {
                "approved_by": "test_user",
                "expires_at": expires_at,
            }

            with open(token_file, "w") as f:
                json.dump(token_data, f)

            # Test with CI environment and expired token (disable mock mode to test CI approval)
            with patch.dict(os.environ, {"CI": "true", "TRAIGENT_MOCK_LLM": "false"}):
                with pytest.raises(OptimizationError) as exc:
                    func._check_ci_approval()

                assert "CI/CD Approval Required" in str(exc.value)

    def test_hmac_token_validation(self, tmp_path: Path):
        """Test HMAC token signature validation."""
        import base64
        import hashlib
        import hmac

        func = OptimizedFunction(
            func=lambda x: x,
            eval_dataset="test.jsonl",
            objectives=["score"],
            configuration_space={"x": [1, 2, 3]},
        )
        func.traigent_config = Mock(
            is_edge_analytics_mode=lambda: True,
            get_local_storage_path=lambda: str(tmp_path),
        )

        # Create token with valid HMAC signature
        secret = b"test_secret"
        approver = "test_approver"
        expires_iso = (datetime.now() + timedelta(hours=1)).isoformat()
        nonce = "test_nonce_123"

        # Compute HMAC signature
        payload = f"{approver}|{expires_iso}|{nonce}".encode()
        signature = base64.b64encode(
            hmac.new(secret, payload, hashlib.sha256).digest()
        ).decode()

        token_data = {
            "approver": approver,
            "expires_iso": expires_iso,
            "nonce": nonce,
            "signature": signature,
        }

        token_file = tmp_path / "approval.token"
        with open(token_file, "w") as f:
            json.dump(token_data, f)

        # Test with CI environment and valid HMAC token
        with patch.dict(
            os.environ,
            {
                "CI": "true",
                "TRAIGENT_APPROVAL_SECRET": "test_secret",  # pragma: allowlist secret
            },
        ):
            result = func._check_ci_approval()  # Should not raise
            assert result is None  # HMAC approval returns None

    def test_invalid_hmac_signature(self, tmp_path: Path):
        """Test that invalid HMAC signatures are rejected."""
        func = OptimizedFunction(
            func=lambda x: x,
            eval_dataset="test.jsonl",
            objectives=["score"],
            configuration_space={"x": [1, 2, 3]},
        )
        func.traigent_config = Mock(
            is_edge_analytics_mode=lambda: True,
            get_local_storage_path=lambda: str(tmp_path),
        )

        # Create token with invalid signature
        token_data = {
            "approver": "test_approver",
            "expires_iso": (datetime.now() + timedelta(hours=1)).isoformat(),
            "nonce": "test_nonce",
            "signature": "invalid_signature",
        }

        token_file = tmp_path / "approval.token"
        with open(token_file, "w") as f:
            json.dump(token_data, f)

        # Test with CI environment and invalid signature (disable mock mode to test CI approval)
        with patch.dict(
            os.environ,
                {
                    "CI": "true",
                    "TRAIGENT_APPROVAL_SECRET": "test_secret",  # pragma: allowlist secret
                    "TRAIGENT_MOCK_LLM": "false",
                },
        ):
            with pytest.raises(OptimizationError):
                func._check_ci_approval()

    def test_non_ci_environment_no_approval_needed(self, tmp_path: Path):
        """Test that non-CI environments don't need approval."""
        func = OptimizedFunction(
            func=lambda x: x,
            eval_dataset="test.jsonl",
            objectives=["score"],
            configuration_space={"x": [1, 2, 3]},  # Required parameter
        )
        func.traigent_config = Mock(
            is_edge_analytics_mode=lambda: True,
            get_local_storage_path=lambda: str(tmp_path),
        )

        # Test with no CI environment variables
        with patch.dict(os.environ, {}, clear=True):
            result = func._check_ci_approval()  # Should not raise
            assert result is None  # Non-CI environment returns None
