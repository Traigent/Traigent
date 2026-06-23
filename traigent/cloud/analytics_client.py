"""Read-only client for backend optimization-results analytics.

This module powers the ``client.analytics`` read namespace and the
``traigent-analytics-mcp`` server. It is a **thin authenticated read client**:
the backend owns all analytics intelligence, so the SDK only

* sends the request with the user's existing credentials,
* validates that the response is the platform success envelope, unwraps ``data``
  (and, for the frozen v0 contracts, validates that the required payload keys
  are present), and
* returns the allowlisted payload unchanged.

It deliberately reuses the SDK's existing credential plumbing
(:func:`traigent.utils.env_config.get_api_key`,
:meth:`traigent.cloud.credential_manager.CredentialManager.get_auth_headers`,
:meth:`traigent.config.backend_config.BackendConfig.get_backend_url`) rather
than adding a second auth path. Tenancy is owned by the backend and derived
from the authenticated principal; this client never sends a caller-supplied
``tenant_id``.

Today's wired endpoints (Wave-1):

* ``GET /api/v1/analytics/runs/{run_id}/report``
* ``GET /api/v1/analytics/dashboards/optimization-overview``
* ``POST /api/v1/optimization-comparisons``
* ``GET /api/v1/analytics/runs/{run_id}/decision-payload``
"""

# Traceability: CONC-Layer-Infra CONC-Security FUNC-CLOUD-HYBRID FUNC-ANALYTICS REQ-CLOUD-009

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast
from urllib.parse import quote

from traigent.utils.logging import get_logger

logger = get_logger(__name__)

try:
    import httpx

    HTTPX_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only when httpx is absent
    HTTPX_AVAILABLE = False
    httpx = None  # type: ignore[assignment]


SUPPORTED_DECISION_INTENTS: tuple[str, ...] = ("iterate", "deploy", "debug", "report")
_DEFAULT_DECISION_INTENT = "iterate"

# Frozen v0 contract keys the SDK asserts before returning a payload. The
# backend may add fields; the SDK only fails closed when a required key is
# missing (a malformed/partial response must never look like success).
_DECISION_PAYLOAD_REQUIRED_KEYS = frozenset(
    {
        "run_id",
        "project_id",
        "intent",
        "headline",
        "confidence",
        "recommended_action",
        "evidence",
        "drilldowns",
        "warnings",
    }
)


class AnalyticsClientError(RuntimeError):
    """Raised when the analytics backend returns a malformed response.

    Transport errors (connection failures, non-2xx status) are surfaced as the
    underlying :class:`httpx.HTTPError` so callers can distinguish "the backend
    said no" from "the backend lied about the shape".
    """


def _require_object(payload: Any, *, what: str) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise AnalyticsClientError(
            f"Malformed {what} response: expected a JSON object."
        )
    return cast(dict[str, Any], payload)


def _unwrap_success_data(payload: Any, *, what: str) -> dict[str, Any]:
    envelope = _require_object(payload, what=what)
    if envelope.get("success") is not True or "data" not in envelope:
        raise AnalyticsClientError(
            f"Malformed {what} response: expected Traigent success envelope with data."
        )
    return _require_object(envelope["data"], what=what)


def _require_keys(
    payload: dict[str, Any], required: frozenset[str], *, what: str
) -> None:
    missing = sorted(required - payload.keys())
    if missing:
        raise AnalyticsClientError(
            f"Malformed {what} response: missing required key(s): {', '.join(missing)}."
        )


def _quote_segment(value: str, *, field: str) -> str:
    """URL-encode a path segment, rejecting empty values up front.

    The MCP tool layer requires an explicit ``project_id`` / ``run_id`` (no
    implicit "latest run"), so an empty identifier here is a programming error
    rather than user input — surface it loudly instead of building a request
    against ``.../projects//overview``.
    """
    text = (value or "").strip()
    if not text:
        raise ValueError(f"{field} must be a non-empty string.")
    return quote(text, safe="")


def normalize_decision_intent(intent: str | None = None) -> str:
    """Return a supported decision-payload intent or raise ``ValueError``."""
    normalized = (intent or "").strip() or _DEFAULT_DECISION_INTENT
    if normalized not in SUPPORTED_DECISION_INTENTS:
        allowed = ", ".join(SUPPORTED_DECISION_INTENTS)
        raise ValueError(f"intent must be one of: {allowed}.")
    return normalized


class BackendAnalyticsClient:
    """Async-first read client for backend optimization-results analytics.

    Thread Safety: safe for concurrent use (``httpx.AsyncClient`` is
    thread-safe).
    """

    def __init__(
        self,
        backend_url: str | None = None,
        api_key: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        """Initialize the analytics read client.

        Args:
            backend_url: Backend origin URL. Defaults to the SDK's resolved
                backend URL (env / stored CLI credentials / cloud default).
            api_key: Explicit API key. When ``None`` the SDK's existing
                credential resolution is used (``TRAIGENT_API_KEY`` /
                stored CLI credentials / dev-mode key).
            timeout: Per-request timeout in seconds.

        Raises:
            ImportError: If ``httpx`` is not installed.
        """
        if not HTTPX_AVAILABLE:
            raise ImportError(
                "httpx is required for BackendAnalyticsClient. "
                "Install with: pip install 'traigent[hybrid]'"
            )

        # Import lazily to avoid import cycles through the cloud package.
        from traigent.config.backend_config import (
            BackendConfig,
            get_no_credentials_hint,
        )

        resolved_url = backend_url or BackendConfig.get_backend_url()
        self.backend_url = resolved_url.rstrip("/")
        self.timeout = timeout

        self.api_key = api_key if api_key is not None else self._resolve_api_key()
        if not self.api_key:
            logger.warning(
                "No API key found for BackendAnalyticsClient. %s",
                get_no_credentials_hint(),
            )

        self._client: httpx.AsyncClient | None = None

    @staticmethod
    def _resolve_api_key() -> str | None:
        """Resolve the API key through the SDK's existing credential path."""
        from traigent.cloud.credential_manager import CredentialManager

        # CredentialManager.get_api_key() honors env vars, stored CLI
        # credentials, and (only in explicit dev mode) TRAIGENT_DEV_API_KEY.
        return cast("str | None", CredentialManager.get_api_key())

    def _auth_headers(self) -> dict[str, str]:
        """Build auth headers using the canonical credential helper.

        Reuses :meth:`CredentialManager.get_auth_headers`, which correctly
        routes API keys to ``X-API-Key`` and JWTs to ``Authorization: Bearer``
        (an API key sent as a bearer token is rejected by the backend).
        """
        if not self.api_key:
            return {}
        if "." in self.api_key and len(self.api_key.split(".")) == 3:
            return {"Authorization": f"Bearer {self.api_key}"}
        return {"X-API-Key": self.api_key}

    def _get_client(self) -> httpx.AsyncClient:
        """Get or create the async HTTP client, failing closed when offline."""
        from traigent.utils.env_config import raise_if_backend_offline

        raise_if_backend_offline("BackendAnalyticsClient request")
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.backend_url,
                headers=self._auth_headers(),
                timeout=self.timeout,
            )
        return self._client

    async def close(self) -> None:
        """Close the HTTP client and release resources."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self) -> BackendAnalyticsClient:
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.close()

    @staticmethod
    def _project_headers(project_id: str) -> dict[str, str]:
        return {"X-Project-Id": _require_non_empty(project_id, field="project_id")}

    async def _get_json(
        self,
        path: str,
        *,
        what: str,
        headers: dict[str, str] | None = None,
        params: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        client = self._get_client()
        response = await client.get(path, headers=headers, params=params)
        response.raise_for_status()
        return _unwrap_success_data(response.json(), what=what)

    # === Read methods (the client.analytics surface) ===

    async def get_run_report(self, project_id: str, run_id: str) -> dict[str, Any]:
        """Return the backend's full analytics report for one run.

        Args:
            project_id: Owning project identifier (explicit; no implicit
                "latest").
            run_id: Experiment-run identifier.

        Returns:
            The backend report payload (returned unchanged).
        """
        path = f"/api/v1/analytics/runs/{_quote_segment(run_id, field='run_id')}/report"
        return await self._get_json(
            path,
            what="run report",
            headers=self._project_headers(project_id),
        )

    async def get_project_overview(self, project_id: str) -> dict[str, Any]:
        """Return the backend's cross-run overview for a project.

        Args:
            project_id: Project identifier (explicit).

        Returns:
            The backend overview payload (returned unchanged).
        """
        path = "/api/v1/analytics/dashboards/optimization-overview"
        return await self._get_json(
            path,
            what="project overview",
            headers=self._project_headers(project_id),
        )

    async def compare_runs(
        self, project_id: str, run_ids: Sequence[str]
    ) -> dict[str, Any]:
        """Compare two or more runs within a project.

        Args:
            project_id: Project identifier (explicit).
            run_ids: Run identifiers to compare (at least two).

        Returns:
            The backend comparison payload (returned unchanged).
        """
        cleaned = [str(run_id).strip() for run_id in run_ids]
        cleaned = [run_id for run_id in cleaned if run_id]
        if len(cleaned) < 2:
            raise ValueError("compare_runs requires at least two run_ids.")

        path = "/api/v1/optimization-comparisons"
        client = self._get_client()
        response = await client.post(
            path,
            json={"run_ids": cleaned},
            headers=self._project_headers(project_id),
        )
        response.raise_for_status()
        return _unwrap_success_data(response.json(), what="run comparison")

    async def get_run_decision_brief(
        self,
        project_id: str,
        run_id: str,
        intent: str = _DEFAULT_DECISION_INTENT,
    ) -> dict[str, Any]:
        """Return the backend's decision brief (decision_payload v0) for a run.

        Endpoint: ``GET /api/v1/analytics/runs/{run_id}/decision-payload``.

        The frozen v0 ``decision_payload`` contract is asserted before the
        payload is returned: a partial/malformed brief must never be presented
        as a confident recommendation.

        Args:
            project_id: Project identifier (explicit). Sent via
                ``X-Project-Id`` so the backend can scope/validate against the
                run's owning project.
            run_id: Experiment-run identifier.
            intent: One of ``iterate`` / ``deploy`` / ``debug`` / ``report``.

        Returns:
            The decision_payload v0 dict (returned unchanged on success).
        """
        normalized_intent = normalize_decision_intent(intent)
        path = (
            "/api/v1/analytics/runs/"
            f"{_quote_segment(run_id, field='run_id')}/decision-payload"
        )
        payload = await self._get_json(
            path,
            what="decision brief",
            headers=self._project_headers(project_id),
            params={"intent": normalized_intent},
        )
        _require_keys(payload, _DECISION_PAYLOAD_REQUIRED_KEYS, what="decision brief")
        return payload


def _require_non_empty(value: str, *, field: str) -> str:
    text = (value or "").strip()
    if not text:
        raise ValueError(f"{field} must be a non-empty string.")
    return text
