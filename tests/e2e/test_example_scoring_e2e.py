"""End-to-end integration tests for example scoring system.

Tests the complete workflow:
1. Content scores computed during optimization
2. Stable IDs and scores injected into backend metadata
3. Scores retrievable via ExampleInsightsClient
"""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from traigent.api.types import TrialResult, TrialStatus
from traigent.config.types import TraigentConfig
from traigent.evaluators.base import EvaluationExample

try:
    from traigent.metrics.content_scoring import ContentScorer

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    ContentScorer = None  # type: ignore

try:
    from traigent.analytics.example_insights import ExampleInsightsClient

    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False
    ExampleInsightsClient = None  # type: ignore


@pytest.mark.skipif(not SKLEARN_AVAILABLE, reason="scikit-learn not installed")
class TestContentScoringIntegration:
    """Test content scoring integration in optimization pipeline."""

    def test_content_scorer_computes_scores_for_dataset(self):
        """ContentScorer should compute scores for a dataset of examples."""
        # Create a simple dataset
        examples = [
            EvaluationExample(
                input_data={"text": "Hello world"}, expected_output="greeting"
            ),
            EvaluationExample(
                input_data={"text": "Goodbye world"}, expected_output="farewell"
            ),
            EvaluationExample(
                input_data={"text": "Hello there"}, expected_output="greeting"
            ),
        ]

        # Extract inputs
        example_inputs = [str(ex.input_data) for ex in examples]

        # Compute scores
        scorer = ContentScorer()
        uniqueness = scorer.compute_uniqueness_scores(example_inputs)
        novelty = scorer.compute_novelty_scores(example_inputs)

        # Verify scores computed for all examples
        assert len(uniqueness) == 3
        assert len(novelty) == 3

        # All scores should be in valid range
        for i in range(3):
            assert 0.0 <= uniqueness[i] <= 1.0
            assert 0.0 <= novelty[i] <= 1.0

    def test_stable_ids_generated_for_dataset(self):
        """Stable IDs should be deterministic and unique."""
        from traigent.utils.example_id import (
            compute_dataset_hash,
            generate_stable_example_id,
        )

        dataset_name = "test_dataset"
        dataset_hash = compute_dataset_hash(dataset_name)

        # Generate IDs for 5 examples
        ids = [generate_stable_example_id(dataset_hash, i) for i in range(5)]

        # All IDs should be unique
        assert len(ids) == len(set(ids))

        # All IDs should follow format
        for example_id in ids:
            assert example_id.startswith("ex_")
            assert "_" in example_id

        # IDs should be stable across runs
        ids_run2 = [generate_stable_example_id(dataset_hash, i) for i in range(5)]
        assert ids == ids_run2

    def test_metadata_builder_includes_content_scores(self):
        """build_backend_metadata should include content scores in measures."""
        from traigent.core.metadata_helpers import build_backend_metadata

        # Create mock trial result with example results
        trial_result = TrialResult(
            trial_id="trial_1",
            config={"model": "gpt-4"},
            metrics={"accuracy": 0.8},
            status=TrialStatus.COMPLETED,
            duration=1.5,
            timestamp=datetime.now(),
        )

        # Add example results
        example_results = []
        for i in range(3):
            example = MagicMock()
            example.metrics = {"score": 0.9 + i * 0.05, "latency": 0.5}
            example.execution_time = 0.5
            example_results.append(example)

        trial_result.metadata = {"example_results": example_results}

        # Pre-computed content scores
        content_scores = {
            "uniqueness": {0: 0.7, 1: 0.8, 2: 0.6},
            "novelty": {0: 0.5, 1: 0.6, 2: 0.4},
        }

        # Build metadata
        config = TraigentConfig(execution_mode="edge_analytics")
        metadata = build_backend_metadata(
            trial_result,
            "accuracy",
            config,
            dataset_name="test_dataset",
            content_scores=content_scores,
        )

        # Verify measures include content scores
        assert "measures" in metadata
        measures = metadata["measures"]
        assert len(measures) == 3

        for i, measure in enumerate(measures):
            # Should have stable ID
            assert "example_id" in measure
            assert measure["example_id"].startswith("ex_")

            # Should have metrics dict (nested format)
            assert "metrics" in measure
            metrics = measure["metrics"]

            # Should have content scores in metrics
            assert "content_uniqueness" in metrics
            assert "content_novelty" in metrics
            assert metrics["content_uniqueness"] == content_scores["uniqueness"][i]
            assert metrics["content_novelty"] == content_scores["novelty"][i]

    def test_metadata_builder_validates_measures(self):
        """Metadata builder should validate measures against MeasuresDict constraints."""
        from traigent.core.metadata_helpers import build_backend_metadata

        # Create trial result with invalid metrics (string value)
        trial_result = TrialResult(
            trial_id="trial_1",
            config={"model": "gpt-4"},
            metrics={"accuracy": 0.8},
            status=TrialStatus.COMPLETED,
            duration=1.5,
            timestamp=datetime.now(),
        )

        example = MagicMock()
        example.metrics = {
            "score": 0.9,
            "invalid_metric": "string_value",  # String values not allowed
        }
        example.execution_time = 0.5

        trial_result.metadata = {"example_results": [example]}

        config = TraigentConfig(execution_mode="edge_analytics")

        # Should not raise - string metrics are filtered out
        metadata = build_backend_metadata(
            trial_result, "accuracy", config, dataset_name="test_dataset"
        )

        # Verify string metric was excluded (check in nested metrics dict)
        measures = metadata["measures"]
        metrics = measures[0]["metrics"]
        assert "invalid_metric" not in metrics
        assert "score" in metrics


@pytest.mark.skipif(not HTTPX_AVAILABLE, reason="httpx not installed")
class TestExampleInsightsClient:
    """Test ExampleInsightsClient for retrieving backend analytics."""

    @pytest.mark.asyncio
    async def test_client_initialization(self):
        """Client should initialize with proper configuration."""
        client = ExampleInsightsClient(
            backend_url="https://api.traigent.ai", api_key="test_key"
        )

        assert client.backend_url == "https://api.traigent.ai"
        assert client.api_key == "test_key"
        assert client.timeout == 30.0

        await client.close()

    @pytest.mark.asyncio
    async def test_compute_scores_triggers_backend_job(self):
        """compute_scores should trigger async backend job."""
        with patch("traigent.analytics.example_insights.httpx") as mock_httpx:
            # Mock async client
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.json.return_value = {
                "status": "accepted",
                "job_id": "job_123",
                "poll_url": "/jobs/job_123",
            }
            mock_response.raise_for_status = MagicMock()

            # Make post method actually async
            async def mock_post(*args, **kwargs):
                return mock_response

            mock_client.post = mock_post
            mock_client.aclose = AsyncMock()  # Mock close method

            # Mock httpx.AsyncClient() constructor to return our mock
            mock_httpx.AsyncClient.return_value = mock_client

            client = ExampleInsightsClient(api_key="test_key")
            result = await client.compute_scores("run_123")

            assert result["status"] == "accepted"
            assert result["job_id"] == "job_123"

            await client.close()

    @pytest.mark.asyncio
    async def test_get_example_scores_with_immediate_response(self):
        """get_example_scores should return scores if immediately available."""
        with patch("traigent.analytics.example_insights.httpx") as mock_httpx:
            # Mock async client
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.json.return_value = {
                "ex_abc123_0": {
                    "example_id": "ex_abc123_0",
                    "content_uniqueness": 0.8,
                    "content_novelty": 0.6,
                    "composite_score": 0.75,
                }
            }
            mock_response.raise_for_status = MagicMock()

            # Make get method actually async
            async def mock_get(*args, **kwargs):
                return mock_response

            mock_client.get = mock_get
            mock_client.aclose = AsyncMock()  # Mock close method

            # Mock httpx.AsyncClient() constructor to return our mock
            mock_httpx.AsyncClient.return_value = mock_client

            client = ExampleInsightsClient(api_key="test_key")
            scores = await client.get_example_scores("run_123")

            assert "ex_abc123_0" in scores
            assert scores["ex_abc123_0"]["content_uniqueness"] == 0.8

            await client.close()

    @pytest.mark.asyncio
    async def test_get_example_scores_with_polling(self):
        """get_example_scores should poll until scores are ready."""
        # Need to import httpx to create proper exception
        try:
            import httpx

            HTTPX_AVAILABLE_LOCAL = True
        except ImportError:
            HTTPX_AVAILABLE_LOCAL = False

        if not HTTPX_AVAILABLE_LOCAL:
            pytest.skip("httpx not available for HTTPStatusError testing")

        with patch("traigent.analytics.example_insights.httpx") as mock_httpx:
            # Expose real HTTPStatusError for exception handling
            mock_httpx.HTTPStatusError = httpx.HTTPStatusError

            # Mock async client
            mock_client = AsyncMock()

            # Prepare successful response
            mock_response_200 = MagicMock()
            mock_response_200.json.return_value = {
                "ex_abc123_0": {"content_uniqueness": 0.8}
            }
            mock_response_200.raise_for_status = MagicMock()

            # Prepare 404 response
            mock_response_404 = MagicMock()
            mock_response_404.status_code = 404

            # Configure mock to return 404 then 200
            call_count = 0

            async def mock_get(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    # First call: raise 404
                    mock_request = MagicMock()
                    raise httpx.HTTPStatusError(
                        "Not found", request=mock_request, response=mock_response_404
                    )
                # Second call: return data
                return mock_response_200

            mock_client.get = mock_get
            mock_client.aclose = AsyncMock()  # Mock close method

            # Mock httpx.AsyncClient() constructor to return our mock
            mock_httpx.AsyncClient.return_value = mock_client

            client = ExampleInsightsClient(api_key="test_key")

            # Should poll and eventually succeed
            scores = await client.get_example_scores("run_123", poll_interval=0.1)

            assert "ex_abc123_0" in scores
            assert call_count == 2  # Polled twice

            await client.close()

    @pytest.mark.asyncio
    async def test_get_example_scores_timeout(self):
        """get_example_scores should timeout if scores not ready."""
        # Need to import httpx to create proper exception
        try:
            import httpx

            HTTPX_AVAILABLE_LOCAL = True
        except ImportError:
            HTTPX_AVAILABLE_LOCAL = False

        if not HTTPX_AVAILABLE_LOCAL:
            pytest.skip("httpx not available for HTTPStatusError testing")

        with patch("traigent.analytics.example_insights.httpx") as mock_httpx:
            # Expose real HTTPStatusError for exception handling
            mock_httpx.HTTPStatusError = httpx.HTTPStatusError

            # Mock async client that always returns 404
            mock_client = AsyncMock()

            mock_response_404 = MagicMock()
            mock_response_404.status_code = 404

            async def mock_get(*args, **kwargs):
                # Always raise 404
                mock_request = MagicMock()
                raise httpx.HTTPStatusError(
                    "Not found", request=mock_request, response=mock_response_404
                )

            mock_client.get = mock_get
            mock_client.aclose = AsyncMock()  # Mock close method

            # Mock httpx.AsyncClient() constructor to return our mock
            mock_httpx.AsyncClient.return_value = mock_client

            client = ExampleInsightsClient(api_key="test_key")

            # Should timeout after 1 second
            with pytest.raises(TimeoutError, match="Scores not ready after"):
                await client.get_example_scores(
                    "run_123", timeout=1.0, poll_interval=0.1
                )

            await client.close()

    @pytest.mark.asyncio
    async def test_get_dataset_quality(self):
        """get_dataset_quality should retrieve dataset-level metrics."""
        with patch("traigent.analytics.example_insights.httpx") as mock_httpx:
            # Mock async client
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.json.return_value = {
                "dataset_quality": 0.85,
                "coverage_score": 0.9,
                "diversity_score": 0.8,
                "top_informative_ids": ["ex_abc123_0", "ex_abc123_1"],
            }
            mock_response.raise_for_status = MagicMock()

            # Make get method actually async
            async def mock_get(*args, **kwargs):
                return mock_response

            mock_client.get = mock_get
            mock_client.aclose = AsyncMock()  # Mock close method

            # Mock httpx.AsyncClient() constructor to return our mock
            mock_httpx.AsyncClient.return_value = mock_client

            client = ExampleInsightsClient(api_key="test_key")
            quality = await client.get_dataset_quality("run_123")

            assert quality["dataset_quality"] == 0.85
            assert "top_informative_ids" in quality

            await client.close()

    @pytest.mark.asyncio
    async def test_get_job_status(self):
        """get_job_status should check async job progress."""
        with patch("traigent.analytics.example_insights.httpx") as mock_httpx:
            # Mock async client
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.json.return_value = {
                "status": "completed",
                "result": {"num_examples": 10},
            }
            mock_response.raise_for_status = MagicMock()

            # Make get method actually async
            async def mock_get(*args, **kwargs):
                return mock_response

            mock_client.get = mock_get
            mock_client.aclose = AsyncMock()  # Mock close method

            # Mock httpx.AsyncClient() constructor to return our mock
            mock_httpx.AsyncClient.return_value = mock_client

            client = ExampleInsightsClient(api_key="test_key")
            status = await client.get_job_status("job_123")

            assert status["status"] == "completed"
            assert "result" in status

            await client.close()

    @pytest.mark.asyncio
    async def test_context_manager_usage(self):
        """Client should work as async context manager."""
        with patch("traigent.analytics.example_insights.httpx"):
            async with ExampleInsightsClient(api_key="test_key") as client:
                assert client is not None
                assert client.api_key == "test_key"

            # Client should be closed after context exit
            assert client._client is None


@pytest.mark.skipif(
    not SKLEARN_AVAILABLE or not HTTPX_AVAILABLE,
    reason="sklearn or httpx not installed",
)
class TestEndToEndWorkflow:
    """Test complete end-to-end workflow from optimization to score retrieval."""

    @pytest.mark.asyncio
    async def test_complete_example_scoring_workflow(self):
        """Test full workflow: compute scores -> submit trial -> retrieve analytics."""
        from traigent.core.metadata_helpers import build_backend_metadata
        from traigent.metrics.content_scoring import ContentScorer
        from traigent.utils.example_id import (
            compute_dataset_hash,
            generate_stable_example_id,
        )

        # Step 1: Create dataset
        examples = [
            EvaluationExample(
                input_data={"text": "Python programming"}, expected_output="code"
            ),
            EvaluationExample(
                input_data={"text": "JavaScript web dev"}, expected_output="code"
            ),
            EvaluationExample(
                input_data={"text": "Machine learning AI"}, expected_output="ml"
            ),
        ]

        # Step 2: Compute content scores
        example_inputs = [str(ex.input_data) for ex in examples]
        scorer = ContentScorer()
        uniqueness = scorer.compute_uniqueness_scores(example_inputs)
        novelty = scorer.compute_novelty_scores(example_inputs)

        content_scores = {"uniqueness": uniqueness, "novelty": novelty}

        # Verify scores computed
        assert len(content_scores["uniqueness"]) == 3
        assert len(content_scores["novelty"]) == 3

        # Step 3: Create trial result with example results
        trial_result = TrialResult(
            trial_id="trial_e2e",
            config={"model": "gpt-4"},
            metrics={"accuracy": 0.9},
            status=TrialStatus.COMPLETED,
            duration=2.0,
            timestamp=datetime.now(),
        )

        example_results = []
        for _i in range(3):
            example = MagicMock()
            example.metrics = {"score": 0.9, "latency": 0.5}
            example.execution_time = 0.5
            example_results.append(example)

        trial_result.metadata = {"example_results": example_results}

        # Step 4: Build backend metadata with scores
        config = TraigentConfig(execution_mode="edge_analytics")
        metadata = build_backend_metadata(
            trial_result,
            "accuracy",
            config,
            dataset_name="e2e_test_dataset",
            content_scores=content_scores,
        )

        # Verify metadata structure
        assert "measures" in metadata
        measures = metadata["measures"]
        assert len(measures) == 3

        # Step 5: Verify stable IDs and scores in measures
        dataset_hash = compute_dataset_hash("e2e_test_dataset")

        for i, measure in enumerate(measures):
            expected_id = generate_stable_example_id(dataset_hash, i)

            assert measure["example_id"] == expected_id
            # Metrics are now nested in 'metrics' dict
            metrics = measure["metrics"]
            assert metrics["content_uniqueness"] == uniqueness[i]
            assert metrics["content_novelty"] == novelty[i]
            assert 0.0 <= metrics["content_uniqueness"] <= 1.0
            assert 0.0 <= metrics["content_novelty"] <= 1.0

        # Step 6: Simulate backend retrieval (with mocked HTTP client)
        with patch("traigent.analytics.example_insights.httpx") as mock_httpx:
            mock_client = AsyncMock()
            mock_response = MagicMock()

            # Mock backend response with computed scores
            backend_scores = {}
            for i in range(3):
                example_id = generate_stable_example_id(dataset_hash, i)
                backend_scores[example_id] = {
                    "example_id": example_id,
                    "content_uniqueness": uniqueness[i],
                    "content_novelty": novelty[i],
                    "composite_score": (uniqueness[i] + novelty[i]) / 2,
                }

            mock_response.json.return_value = backend_scores
            mock_response.raise_for_status = MagicMock()

            # Make get method actually async
            async def mock_get(*args, **kwargs):
                return mock_response

            mock_client.get = mock_get
            mock_client.aclose = AsyncMock()  # Mock close method

            # Mock httpx.AsyncClient() constructor to return our mock
            mock_httpx.AsyncClient.return_value = mock_client

            # Retrieve scores via client
            client = ExampleInsightsClient(api_key="test_key")
            retrieved_scores = await client.get_example_scores("run_e2e")

            # Verify retrieved scores match
            for i in range(3):
                example_id = generate_stable_example_id(dataset_hash, i)
                assert example_id in retrieved_scores
                assert (
                    retrieved_scores[example_id]["content_uniqueness"] == uniqueness[i]
                )
                assert retrieved_scores[example_id]["content_novelty"] == novelty[i]

            await client.close()
