"""Sync-friendly Agent Arena client."""

from __future__ import annotations

import json
from typing import Any, cast

import requests as _requests  # type: ignore[import-untyped]

from traigent.arena.config import ArenaConfig
from traigent.arena.dtos import (
    ArenaInvokeResult,
    ArenaLeaderboard,
    ArenaProviderSource,
    ArenaRun,
)
from traigent.utils.exceptions import (
    AuthenticationError,
    ClientError,
    TraigentConnectionError,
)


class ArenaClient:
    """Explicit client for Agent Arena provider configuration and brokered invoke."""

    def __init__(
        self,
        config: ArenaConfig | None = None,
        *,
        request_sender=None,
    ) -> None:
        self.config = config or ArenaConfig()
        self._request_sender_override = request_sender
        self._session: _requests.Session | None = None

    def _get_session(self) -> _requests.Session:
        if self._session is None:
            self._session = _requests.Session()
            self._session.headers.update(self.config.build_headers())
        return self._session

    def close(self) -> None:
        """Close the underlying HTTP session."""
        if self._session is not None:
            self._session.close()
            self._session = None

    def list_provider_sources(
        self, *, page: int = 1, per_page: int = 50
    ) -> list[ArenaProviderSource]:
        payload = self._request_json(
            "GET", f"/provider-sources?page={max(1, page)}&per_page={max(1, per_page)}"
        )
        data = self._unwrap_items(payload, "provider source list")
        return [
            ArenaProviderSource.from_dict(item)
            for item in data
            if isinstance(item, dict)
        ]

    def create_provider_source(
        self,
        *,
        provider: str,
        kind: str,
        label: str | None = None,
        credentials: dict[str, Any] | None = None,
        sponsor_grant: dict[str, Any] | None = None,
        allowed_models: list[str] | None = None,
        attribution_payload: dict[str, Any] | None = None,
        config_metadata: dict[str, Any] | None = None,
        persist: bool = True,
        session_scope_id: str | None = None,
        optimization_token_budget: int | None = None,
    ) -> ArenaProviderSource:
        payload: dict[str, Any] = {
            "provider": provider,
            "kind": kind,
            "persist": persist,
        }
        if label is not None:
            payload["label"] = label
        if credentials is not None:
            payload["credentials"] = dict(credentials)
        if sponsor_grant is not None:
            payload["sponsor_grant"] = dict(sponsor_grant)
        if allowed_models is not None:
            payload["allowed_models"] = list(allowed_models)
        if attribution_payload is not None:
            payload["attribution_payload"] = dict(attribution_payload)
        if config_metadata is not None:
            payload["config_metadata"] = dict(config_metadata)
        if session_scope_id is not None:
            payload["session_scope_id"] = session_scope_id
        if optimization_token_budget is not None:
            payload["optimization_token_budget"] = optimization_token_budget

        response = self._request_json("POST", "/provider-sources", payload)
        return ArenaProviderSource.from_dict(
            self._unwrap_data(response, "provider source create")
        )

    def update_provider_source(
        self,
        provider_source_id: str,
        *,
        label: str | None = None,
        allowed_models: list[str] | None = None,
        attribution_payload: dict[str, Any] | None = None,
        config_metadata: dict[str, Any] | None = None,
        credentials: dict[str, Any] | None = None,
        sponsor_grant: dict[str, Any] | None = None,
        status: str | None = None,
        optimization_token_budget: int | None = None,
    ) -> ArenaProviderSource:
        payload: dict[str, Any] = {}
        if label is not None:
            payload["label"] = label
        if allowed_models is not None:
            payload["allowed_models"] = list(allowed_models)
        if attribution_payload is not None:
            payload["attribution_payload"] = dict(attribution_payload)
        if config_metadata is not None:
            payload["config_metadata"] = dict(config_metadata)
        if credentials is not None:
            payload["credentials"] = dict(credentials)
        if sponsor_grant is not None:
            payload["sponsor_grant"] = dict(sponsor_grant)
        if status is not None:
            payload["status"] = status
        if optimization_token_budget is not None:
            payload["optimization_token_budget"] = optimization_token_budget

        response = self._request_json(
            "PUT",
            f"/provider-sources/{provider_source_id}",
            payload,
        )
        return ArenaProviderSource.from_dict(
            self._unwrap_data(response, "provider source update")
        )

    def delete_provider_source(self, provider_source_id: str) -> ArenaProviderSource:
        response = self._request_json(
            "DELETE", f"/provider-sources/{provider_source_id}"
        )
        return ArenaProviderSource.from_dict(
            self._unwrap_data(response, "provider source delete")
        )

    def accept_provider_consent(self, provider_source_id: str) -> dict[str, Any]:
        response = self._request_json(
            "POST",
            f"/provider-sources/{provider_source_id}/consent",
        )
        return cast(dict[str, Any], self._unwrap_data(response, "provider consent"))

    def list_runs(self, *, page: int = 1, per_page: int = 50) -> list[ArenaRun]:
        response = self._request_json(
            "GET", f"/runs?page={max(1, page)}&per_page={max(1, per_page)}"
        )
        data = self._unwrap_items(response, "arena run list")
        return [ArenaRun.from_dict(item) for item in data if isinstance(item, dict)]

    def create_run(
        self,
        *,
        function_name: str,
        providers: list[dict[str, Any]] | None = None,
        provider_source_ids: list[str] | None = None,
        configuration_space: dict[str, Any],
        objectives: list[Any],
        dataset_metadata: dict[str, Any] | None = None,
        max_trials: int = 10,
        budget: dict[str, Any] | None = None,
        constraints: dict[str, Any] | None = None,
        default_config: dict[str, Any] | None = None,
        promotion_policy: dict[str, Any] | None = None,
        optimization_strategy: dict[str, Any] | None = None,
        ranking_weights: dict[str, float] | None = None,
        dataset_ref: dict[str, Any] | None = None,
        agent_ref: dict[str, Any] | None = None,
        prompt_ref: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        name: str | None = None,
    ) -> ArenaRun:
        provider_payload = list(providers or [])
        if not provider_payload and provider_source_ids:
            provider_payload = [
                {"provider_source_id": provider_source_id}
                for provider_source_id in provider_source_ids
            ]
        payload: dict[str, Any] = {
            "function_name": function_name,
            "providers": provider_payload,
            "configuration_space": configuration_space,
            "objectives": objectives,
            "max_trials": max_trials,
        }
        if dataset_metadata is not None:
            payload["dataset_metadata"] = dataset_metadata
        if budget is not None:
            payload["budget"] = budget
        if constraints is not None:
            payload["constraints"] = constraints
        if default_config is not None:
            payload["default_config"] = default_config
        if promotion_policy is not None:
            payload["promotion_policy"] = promotion_policy
        if optimization_strategy is not None:
            payload["optimization_strategy"] = optimization_strategy
        if ranking_weights is not None:
            payload["ranking_weights"] = ranking_weights
        if dataset_ref is not None:
            payload["dataset_ref"] = dataset_ref
        if agent_ref is not None:
            payload["agent_ref"] = agent_ref
        if prompt_ref is not None:
            payload["prompt_ref"] = prompt_ref
        if metadata is not None:
            payload["metadata"] = metadata
        if name is not None:
            payload["name"] = name

        response = self._request_json("POST", "/runs", payload)
        return ArenaRun.from_dict(self._unwrap_data(response, "arena run create"))

    def get_run(self, run_id: str) -> ArenaRun:
        response = self._request_json("GET", f"/runs/{run_id}")
        return ArenaRun.from_dict(self._unwrap_data(response, "arena run"))

    def get_leaderboard(self, run_id: str) -> ArenaLeaderboard:
        response = self._request_json("GET", f"/runs/{run_id}/leaderboard")
        return ArenaLeaderboard.from_dict(
            self._unwrap_data(response, "arena leaderboard")
        )

    def invoke(
        self,
        *,
        run_id: str,
        provider_source_id: str,
        model: str,
        prompt: str | None = None,
        messages: list[dict[str, str]] | None = None,
        provider: str | None = None,
        operation_type: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: str | None = None,
        metadata: dict[str, Any] | None = None,
        evaluation_scores: dict[str, float] | None = None,
        session_id: str | None = None,
        trial_id: str | None = None,
        trace_id: str | None = None,
        idempotency_key: str | None = None,
        include_raw: bool = False,
    ) -> ArenaInvokeResult:
        payload: dict[str, Any] = {
            "run_id": run_id,
            "provider_source_id": provider_source_id,
            "model": model,
            "include_raw": include_raw,
        }
        if prompt is not None:
            payload["prompt"] = prompt
        if messages is not None:
            payload["messages"] = messages
        if provider is not None:
            payload["provider"] = provider
        if operation_type is not None:
            payload["operation_type"] = operation_type
        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if response_format is not None:
            payload["response_format"] = response_format
        if metadata is not None:
            payload["metadata"] = metadata
        if evaluation_scores is not None:
            payload["evaluation_scores"] = evaluation_scores
        if session_id is not None:
            payload["session_id"] = session_id
        if trial_id is not None:
            payload["trial_id"] = trial_id
        if trace_id is not None:
            payload["trace_id"] = trace_id
        if idempotency_key is not None:
            payload["idempotency_key"] = idempotency_key

        response = self._request_json("POST", "/invoke", payload)
        return ArenaInvokeResult.from_dict(self._unwrap_data(response, "arena invoke"))

    def _request_json(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if self._request_sender_override is not None:
            return cast(
                dict[str, Any], self._request_sender_override(method, path, payload)
            )
        return self._request_json_sync(method, path, payload)

    def _request_json_sync(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        url = f"{self.config.backend_origin}{self.config.api_path}{path}"
        session = self._get_session()

        try:
            response = session.request(
                method=method.upper(),
                url=url,
                json=payload,
                timeout=self.config.request_timeout,
            )
        except _requests.ConnectionError as exc:
            raise TraigentConnectionError(
                f"Could not reach Agent Arena backend: {exc}"
            ) from exc
        except _requests.Timeout as exc:
            raise TraigentConnectionError(
                f"Agent Arena backend request timed out after {self.config.request_timeout}s"
            ) from exc

        if response.status_code >= 400:
            error_payload = self._decode_json_payload(response.text)
            message = (
                error_payload.get("error")
                or error_payload.get("message")
                or response.text
                or f"HTTP {response.status_code}"
            )
            if response.status_code in {401, 403}:
                raise AuthenticationError(str(message))
            raise ClientError(str(message))

        return self._decode_json_payload(response.text)

    @staticmethod
    def _decode_json_payload(raw_payload: str) -> dict[str, Any]:
        if not raw_payload:
            return {}
        try:
            return cast(dict[str, Any], json.loads(raw_payload))
        except json.JSONDecodeError as exc:
            raise ClientError("Arena backend returned invalid JSON") from exc

    @staticmethod
    def _unwrap_data(payload: dict[str, Any], label: str) -> Any:
        if not isinstance(payload, dict):
            raise ClientError(f"Invalid {label} response from Arena backend")
        if payload.get("success") is False:
            message = (
                payload.get("error")
                or payload.get("message")
                or f"{label} request failed"
            )
            raise ClientError(str(message))
        if "data" not in payload:
            raise ClientError(f"Arena backend did not include data for {label}")
        return payload["data"]

    @classmethod
    def _unwrap_items(cls, payload: dict[str, Any], label: str) -> list[Any]:
        data = cls._unwrap_data(payload, label)
        if not isinstance(data, dict):
            raise ClientError(
                f"Arena backend returned invalid paginated data for {label}"
            )
        items = data.get("items")
        if not isinstance(items, list):
            raise ClientError(f"Arena backend did not include items for {label}")
        return items
