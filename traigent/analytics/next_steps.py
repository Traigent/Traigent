"""Client for retrieving client-safe next-step recommendations.

This module provides an async-first client for the backend next-steps endpoint:
``GET /api/v1/analytics/experiments/{experiment_run_id}/next-steps``.

The endpoint is available only on backend versions that include the next-steps
feature. Older backends should fail truthfully, especially with a 404 response.

Returned recommendations are category-level, templated advice for what to try
next. They expose safe labels, rationale text, action templates, and coarse
evidence levels only. Proprietary tuning signals, signal values, formulas, and
rankings are not returned to clients, following the platform's IP discipline.

Usage:
    >>> from traigent.analytics import NextStepsClient
    >>>
    >>> client = NextStepsClient(backend_url="https://portal.traigent.ai")
    >>> payload = await client.get_next_steps("run_123")
    >>> print(payload["caveat"])
"""

from __future__ import annotations

from typing import Any, cast

from traigent.cloud.auth import _build_api_key_auth_headers
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
    "experiment_run_id",
    "caveat",
    "summary",
    "next_steps",
}  # matches the contract's required set (next_steps_schema.json)


class NextStepsClient:
    """Client for retrieving category-level next-step recommendations.

    Thread Safety: Safe for concurrent use (httpx.AsyncClient is thread-safe).
    """

    def __init__(
        self,
        backend_url: str = DEFAULT_LOCAL_URL,
        api_key: str | None = None,
        timeout: float = 30.0,
    ):
        """Initialize NextStepsClient.

        Args:
            backend_url: Backend API base URL. The backend must expose the
                next-steps endpoint.
            api_key: API key for authentication (None uses env var
                TRAIGENT_API_KEY)
            timeout: Default timeout for HTTP requests in seconds

        Raises:
            ImportError: If httpx is not installed
        """
        if not HTTPX_AVAILABLE:
            raise ImportError(
                "httpx is required for NextStepsClient. "
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
                "No API key found for NextStepsClient. %s",
                get_no_credentials_hint(),
            )

        self._client: httpx.AsyncClient | None = None

    def _get_client(self) -> httpx.AsyncClient:
        """Get or create async HTTP client.

        Returns:
            Configured httpx.AsyncClient instance
        """
        from traigent.utils.env_config import raise_if_backend_offline

        raise_if_backend_offline("NextStepsClient request")
        if self._client is None:
            headers: dict[str, str] = {}
            if self.api_key:
                headers.update(_build_api_key_auth_headers(self.api_key))

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

    async def __aenter__(self) -> NextStepsClient:
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.close()

    async def get_next_steps(self, experiment_run_id: str) -> dict[str, Any]:
        """Retrieve category-level next-step recommendations for an experiment.

        The backend must include the next-steps feature and expose:
        ``GET /api/v1/analytics/experiments/{experiment_run_id}/next-steps``.

        Args:
            experiment_run_id: Experiment run ID to retrieve recommendations for

        Returns:
            Dict with the backend next-steps contract payload. The response
            includes a ``caveat`` field that callers should display near the
            recommendations.

        Raises:
            httpx.HTTPError: If request fails. A 404 response is raised as
                httpx.HTTPStatusError with a message noting that the backend may
                predate the next-steps feature.
            ValueError: If the backend returns a JSON object missing required
                next-steps contract keys.
        """
        client = self._get_client()

        try:
            response = await client.get(
                f"/api/v1/analytics/experiments/{experiment_run_id}/next-steps"
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                raise httpx.HTTPStatusError(
                    "Next steps endpoint returned 404. The backend may predate "
                    "the next-steps feature; use a backend version that exposes "
                    "GET /api/v1/analytics/experiments/{experiment_run_id}/next-steps.",
                    request=exc.request,
                    response=exc.response,
                ) from exc
            raise

        payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError(
                "Malformed next-steps response: expected a JSON object matching "
                "the next-steps contract."
            )

        missing = sorted(_REQUIRED_RESPONSE_KEYS - payload.keys())
        if missing:
            raise ValueError(
                "Malformed next-steps response: missing required key(s): "
                f"{', '.join(missing)}."
            )

        return cast(dict[str, Any], payload)
