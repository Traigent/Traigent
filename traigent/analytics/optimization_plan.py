"""Client for retrieving backend-provided pre-run optimization plans.

This module provides an async-first thin client for:
``POST /api/v1/optimization/plan``.

The backend owns all planning intelligence. The SDK only sends the request
context, validates the response has the required top-level contract keys, and
returns the allowlisted payload unchanged.
"""

from __future__ import annotations

from collections.abc import Sequence
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


_REQUIRED_RESPONSE_KEYS = {
    "schema_version",
    "phase",
    "plan",
    "steps",
    "evidence_level",
    "caveat",
    "advisory",
}


def _build_auth_headers(credential: str | None) -> dict[str, str]:
    """Return optimization-plan auth headers for an API-key credential."""
    from traigent.cloud.auth import _build_api_key_auth_headers

    # OptimizationPlanClient accepts API-key credentials only. Reuse the same
    # X-API-Key-only construction as AuthManager._get_api_key_headers
    # (cloud/auth.py) and do not infer JWT mode from string shape.
    return _build_api_key_auth_headers(credential)


class OptimizationPlanClient:
    """Client for retrieving allowlisted pre-run optimization plans.

    Thread Safety: Safe for concurrent use (httpx.AsyncClient is thread-safe).
    """

    def __init__(
        self,
        backend_url: str = DEFAULT_LOCAL_URL,
        api_key: str | None = None,
        timeout: float = 30.0,
    ):
        """Initialize OptimizationPlanClient.

        Args:
            backend_url: Backend API base URL. The backend must expose the
                optimization plan endpoint.
            api_key: API key for authentication (None uses env var
                TRAIGENT_API_KEY)
            timeout: Default timeout for HTTP requests in seconds

        Raises:
            ImportError: If httpx is not installed
        """
        if not HTTPX_AVAILABLE:
            raise ImportError(
                "httpx is required for OptimizationPlanClient. "
                "Install with: pip install traigent[hybrid]"
            )

        self.backend_url = backend_url.rstrip("/")
        self.timeout = timeout

        # Import here to avoid circular dependency
        from traigent.config.backend_config import get_no_credentials_hint
        from traigent.utils.env_config import get_api_key

        self.api_key = get_api_key("traigent") if api_key is None else api_key
        if not self.api_key:
            logger.warning(
                "No API key found for OptimizationPlanClient. %s",
                get_no_credentials_hint(),
            )

        self._client: httpx.AsyncClient | None = None

    def _get_client(self) -> httpx.AsyncClient:
        """Get or create async HTTP client.

        Returns:
            Configured httpx.AsyncClient instance
        """
        from traigent.utils.env_config import raise_if_backend_offline

        raise_if_backend_offline("OptimizationPlanClient request")
        if self._client is None:
            headers: dict[str, str] = {}
            if self.api_key:
                headers.update(_build_auth_headers(self.api_key))

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

    async def __aenter__(self) -> OptimizationPlanClient:
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.close()

    async def get_optimization_plan(
        self,
        *,
        task_description: str,
        dataset_size: int,
        dataset_has_holdout: bool,
        objectives: Sequence[str],
        max_trials: int,
        cost_limit_usd: float,
        task_type: str | None = None,
        agent_shape: str | None = None,
        weights: dict[str, float] | None = None,
        offline: bool | None = None,
    ) -> dict[str, Any]:
        """Retrieve a backend-provided pre-run optimization plan.

        The backend must expose ``POST /api/v1/optimization/plan`` and returns
        the bare allowlisted optimization plan contract payload.

        Args:
            task_description: Natural-language description of the optimization
                task.
            dataset_size: Number of available examples.
            dataset_has_holdout: Whether the dataset already has a holdout split.
            objectives: Objective metric names to plan against.
            max_trials: Maximum trial count requested for the planned run.
            cost_limit_usd: Maximum approved spend for the planned run, in USD.
            task_type: Optional safe task category.
            agent_shape: Optional safe label for the agent shape.
            weights: Optional objective-name to relative-weight map.
            offline: Optional request for a fully offline plan.

        Returns:
            Dict with the backend optimization plan contract payload.

        Raises:
            httpx.HTTPError: If request fails.
            ValueError: If the backend returns a non-object JSON payload or an
                object missing required optimization-plan contract keys.
        """
        body: dict[str, Any] = {
            "task_description": task_description,
            "dataset": {
                "size": dataset_size,
                "has_holdout": dataset_has_holdout,
            },
            "objectives": list(objectives),
            "budget": {
                "max_trials": max_trials,
                "cost_limit_usd": cost_limit_usd,
            },
        }
        if task_type is not None:
            body["task_type"] = task_type
        if agent_shape is not None:
            body["agent_shape"] = agent_shape
        if weights is not None:
            body["weights"] = dict(weights)
        if offline is not None:
            body["offline"] = offline

        client = self._get_client()
        response = await client.post("/api/v1/optimization/plan", json=body)
        response.raise_for_status()

        payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError(
                "Malformed optimization-plan response: expected a JSON object "
                "matching the optimization-plan contract."
            )

        missing = sorted(_REQUIRED_RESPONSE_KEYS - payload.keys())
        if missing:
            raise ValueError(
                "Malformed optimization-plan response: missing required key(s): "
                f"{', '.join(missing)}."
            )

        return cast(dict[str, Any], payload)
