"""Client for retrieving example-level insights and dataset quality metrics from backend.

This module provides async-first methods to retrieve:
- Example-level scores (uniqueness, novelty, informativeness, etc.)
- Dataset-level quality metrics (coverage, diversity, efficiency)
- Scoring job status (for async computation polling)

Usage:
    >>> from traigent.analytics import ExampleInsightsClient
    >>>
    >>> # Initialize client (requires backend authentication)
    >>> client = ExampleInsightsClient(backend_url="https://portal.traigent.ai")
    >>>
    >>> # Trigger scoring computation (async job)
    >>> job_result = await client.compute_scores(experiment_run_id="run_123")
    >>> print(f"Job ID: {job_result['job_id']}")
    >>>
    >>> # Poll for completion (use asyncio.timeout for custom deadline)
    >>> async with asyncio.timeout(60):
    ...     scores = await client.get_example_scores(
    ...         experiment_run_id="run_123",
    ...     )
    >>>
    >>> # Retrieve dataset quality metrics
    >>> quality = await client.get_dataset_quality(experiment_run_id="run_123")
    >>> print(f"Dataset Quality: {quality['dataset_quality']}")
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, cast

from traigent.config.backend_config import DEFAULT_LOCAL_URL
from traigent.utils.logging import get_logger

logger = get_logger(__name__)

try:
    import httpx

    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False
    httpx = None  # type: ignore


class ExampleInsightsClient:
    """Client for retrieving example-level insights from backend.

    Thread Safety: Safe for concurrent use (httpx.AsyncClient is thread-safe).
    """

    def __init__(
        self,
        backend_url: str = DEFAULT_LOCAL_URL,
        api_key: str | None = None,
        timeout: float = 30.0,
    ):
        """Initialize ExampleInsightsClient.

        Args:
            backend_url: Backend API base URL
            api_key: API key for authentication (None uses env var TRAIGENT_API_KEY)
            timeout: Default timeout for HTTP requests in seconds

        Raises:
            ImportError: If httpx is not installed
        """
        if not HTTPX_AVAILABLE:
            raise ImportError(
                "httpx is required for ExampleInsightsClient. "
                "Install with: pip install traigent[analytics]"
            )

        self.backend_url = backend_url.rstrip("/")
        self.timeout = timeout

        # Import here to avoid circular dependency
        from traigent.config.backend_config import get_no_credentials_hint
        from traigent.utils.env_config import get_api_key

        self.api_key = api_key or get_api_key("traigent")
        if not self.api_key:
            logger.warning(
                "No API key found for ExampleInsightsClient. %s",
                get_no_credentials_hint(),
            )

        self._client: httpx.AsyncClient | None = None

    def _get_client(self) -> httpx.AsyncClient:
        """Get or create async HTTP client.

        Returns:
            Configured httpx.AsyncClient instance
        """
        if self._client is None:
            headers = {}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"

            self._client = httpx.AsyncClient(
                base_url=self.backend_url,
                headers=headers,
                timeout=self.timeout,
            )

        return self._client

    async def close(self) -> None:
        """Close HTTP client and release resources."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self) -> ExampleInsightsClient:
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.close()

    async def _poll_endpoint(
        self,
        path: str,
        error_label: str,
        poll_interval: float,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Poll a GET endpoint until it returns 200 or timeout is reached.

        Args:
            path: API path to poll
            error_label: Human-readable label for timeout error messages
            poll_interval: Seconds between attempts
            params: Optional query parameters

        Returns:
            Parsed JSON response

        Raises:
            httpx.HTTPStatusError: For non-404 HTTP errors
            TimeoutError: If endpoint still returns 404 after ``self.timeout``
        """
        client = self._get_client()
        timeout = self.timeout
        start_time = time.monotonic()

        while True:
            try:
                response = await client.get(path, params=params or {})
                response.raise_for_status()
                return cast(dict[str, Any], response.json())
            except httpx.HTTPStatusError as e:
                if e.response.status_code != 404:
                    raise
                elapsed = time.monotonic() - start_time
                if elapsed >= timeout:
                    raise TimeoutError(
                        f"{error_label} not ready after {timeout}s. "
                        f"Check job status or increase timeout."
                    ) from e
                logger.debug(
                    "%s not ready, polling again in %.1fs (elapsed: %.1fs/%ds)",
                    error_label,
                    poll_interval,
                    elapsed,
                    timeout,
                )
                await asyncio.sleep(poll_interval)

    async def compute_scores(
        self,
        experiment_run_id: str,
    ) -> dict[str, Any]:
        """Trigger async scoring computation for an experiment run.

        Returns immediately with job information. Poll /jobs/{job_id} or use
        get_example_scores() with timeout to wait for completion.

        Args:
            experiment_run_id: Experiment run ID to compute scores for

        Returns:
            Dict with keys:
            - status: "accepted"
            - job_id: Job identifier for polling
            - poll_url: URL to check job status

        Raises:
            httpx.HTTPError: If request fails
        """
        client = self._get_client()

        response = await client.post(
            f"/analytics/example-scoring/{experiment_run_id}/compute"
        )
        response.raise_for_status()

        return cast(dict[str, Any], response.json())

    async def get_example_scores(
        self,
        experiment_run_id: str,
        example_ids: list[str] | None = None,
        poll_interval: float = 2.0,
    ) -> dict[str, dict[str, Any]]:
        """Retrieve computed example scores.

        If scores are not yet computed, this will wait (polling) until they are
        ready or ``self.timeout`` is reached.  Callers needing a custom timeout
        should use ``asyncio.timeout()`` around the call.

        Args:
            experiment_run_id: Experiment run ID to get scores for
            example_ids: Optional list of example IDs to filter (None = all)
            poll_interval: Time between polling attempts in seconds

        Returns:
            Dict mapping example_id -> score_dict with keys:
            - example_id: Stable example identifier
            - content_uniqueness: Uniqueness score (0-1)
            - content_novelty: Novelty score (0-1)
            - informativeness: Informativeness score (0-1)
            - consistency: Consistency score (0-1)
            - difficulty: Difficulty score (0-1)
            - discriminative_power: Discriminative power score (0-1)
            - cost_efficiency: Cost efficiency score (0-1)
            - error_sensitivity: Error sensitivity score (0-1)
            - predictive_value: Predictive value score (0-1)
            - ambiguity: Ambiguity score (0-1)
            - composite_score: Weighted composite score (0-1)
            - sample_count: Number of trials contributing to score

        Raises:
            httpx.HTTPError: If request fails
            TimeoutError: If scores not ready within timeout
        """
        params: dict[str, Any] = {}
        if example_ids:
            params["example_ids"] = example_ids

        return cast(
            dict[str, dict[str, Any]],
            await self._poll_endpoint(
                f"/analytics/example-scoring/{experiment_run_id}/scores",
                "Scores",
                poll_interval,
                params=params,
            ),
        )

    async def get_dataset_quality(
        self,
        experiment_run_id: str,
        poll_interval: float = 2.0,
    ) -> dict[str, Any]:
        """Retrieve dataset-level quality metrics.

        If quality metrics are not yet computed, this will wait (polling) until
        they are ready or ``self.timeout`` is reached.  Callers needing a custom
        timeout should use ``asyncio.timeout()`` around the call.

        Args:
            experiment_run_id: Experiment run ID to get quality for
            poll_interval: Time between polling attempts in seconds

        Returns:
            Dict with keys:
            - dataset_quality: Overall quality score (0-1)
            - coverage_score: Coverage score (0-1)
            - diversity_score: Diversity score (0-1)
            - efficiency_score: Efficiency score (0-1)
            - top_informative_ids: List of top informative example IDs (max 20)
            - top_difficult_ids: List of top difficult example IDs (max 20)
            - low_value_ids: List of low-value example IDs (max 20)
            - redundant_pairs: List of redundant example ID pairs (max 50)
            - score_distributions: Score distribution statistics
            - recommendations: List of improvement recommendations

        Raises:
            httpx.HTTPError: If request fails
            TimeoutError: If metrics not ready within timeout
        """
        return await self._poll_endpoint(
            f"/analytics/example-scoring/{experiment_run_id}/dataset-quality",
            "Quality metrics",
            poll_interval,
        )

    async def get_job_status(self, job_id: str) -> dict[str, Any]:
        """Get status of a scoring computation job.

        Args:
            job_id: Job identifier from compute_scores() response

        Returns:
            Dict with keys:
            - status: "pending" | "running" | "completed" | "failed"
            - result: Computation result (if completed)
            - error: Error message (if failed)

        Raises:
            httpx.HTTPError: If request fails
        """
        client = self._get_client()

        response = await client.get(f"/jobs/{job_id}")
        response.raise_for_status()

        return cast(dict[str, Any], response.json())
