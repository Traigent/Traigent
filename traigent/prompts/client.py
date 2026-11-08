"""Sync-friendly prompt management client."""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any, NoReturn, cast
from urllib import error, request
from urllib.parse import quote, urlencode

from traigent.cloud.url_security import open_no_redirect
from traigent.prompts.config import PromptManagementConfig
from traigent.prompts.dtos import (
    ChatPromptMessage,
    PromptAnalytics,
    PromptDetail,
    PromptListResponse,
    PromptPlaygroundResult,
    PromptSummary,
    PromptType,
    ResolvedPrompt,
)
from traigent.prompts.errors import error_from_problem
from traigent.utils.env_config import raise_if_backend_offline
from traigent.utils.exceptions import (
    AuthenticationError,
    ClientError,
    RateLimitError,
    ServiceUnavailableError,
    TraigentConnectionError,
)

PROBLEM_CONTENT_TYPE = "application/problem+json"


class PromptManagementClient:
    """Client for versioned prompt registry operations in one project.

    Every request goes to ``/api/v1/projects/{project_id}/prompts``; the
    project comes from ``PromptManagementConfig.project_id`` or the
    ``TRAIGENT_PROJECT_ID`` environment variable.

    Failures raise subclasses of :class:`~traigent.prompts.errors.PromptApiError`
    (all of them :class:`~traigent.utils.exceptions.ClientError`). Authentication
    and rate-limit failures raised before a request reaches the prompts service
    raise :class:`~traigent.utils.exceptions.AuthenticationError` and
    :class:`~traigent.utils.exceptions.RateLimitError`.

    The client is intentionally lightweight and does not coordinate mutable state
    across threads. Treat instances as thread-local or create separate clients
    per worker when using the SDK from multi-threaded code.
    """

    def __init__(
        self,
        config: PromptManagementConfig | None = None,
        *,
        request_sender=None,
    ) -> None:
        self.config = config or PromptManagementConfig()
        self._request_sender_override = request_sender

    def list_prompts(
        self,
        *,
        limit: int = 20,
        cursor: str | None = None,
        search: str | None = None,
        prompt_type: PromptType | str | None = None,
        label: str | None = None,
    ) -> PromptListResponse:
        """Return one page of prompts in the configured project, newest first.

        Args:
            limit: Page size, 1 to 100. Values outside that range are rejected
                with :class:`~traigent.prompts.errors.PromptInvalidRequestError`.
            cursor: The ``pagination.next_cursor`` of the previous page. Pass the
                same filters again; a cursor is bound to the project and filters
                it was issued for.
            search, prompt_type, label: Optional filters.

        There is no total count: use ``response.pagination.has_more``, or
        :meth:`iter_prompts` to walk every page.
        """
        if isinstance(prompt_type, str):
            prompt_type = PromptType(prompt_type)
        query = self._build_query_string(
            limit=limit,
            cursor=cursor,
            search=search,
            prompt_type=prompt_type.value if prompt_type else None,
            label=label,
        )
        payload = self._request_json("GET", query or "")
        return PromptListResponse.from_dict(self._unwrap_data(payload, "prompt list"))

    def iter_prompts(
        self,
        *,
        limit: int = 100,
        search: str | None = None,
        prompt_type: PromptType | str | None = None,
        label: str | None = None,
    ) -> Iterator[PromptSummary]:
        """Iterate over *all* prompts across pages, fetching lazily.

        Follows ``pagination.next_cursor`` until the backend reports no more
        pages. Filters are forwarded unchanged on every page request.

        Parameters
        ----------
        limit:
            Items per page (1 to 100). Defaults to 100 to minimise round-trips.
        search, prompt_type, label:
            Same filter semantics as :meth:`list_prompts`.
        """
        cursor: str | None = None
        while True:
            page = self.list_prompts(
                limit=limit,
                cursor=cursor,
                search=search,
                prompt_type=prompt_type,
                label=label,
            )
            yield from page.items
            cursor = page.pagination.next_cursor
            if not page.pagination.has_more or cursor is None:
                return

    def get_prompt(self, name: str) -> PromptDetail:
        payload = self._request_json("GET", f"/{self._quote_name(name)}")
        return PromptDetail.from_dict(self._unwrap_data(payload, "prompt detail"))

    def delete_prompt(self, name: str) -> None:
        """Delete a prompt and all of its versions."""
        self._request_json("DELETE", f"/{self._quote_name(name)}")

    def get_prompt_version(self, name: str, version: int) -> ResolvedPrompt:
        """Return one specific version of a prompt (``version`` is 1 or more)."""
        payload = self._request_json(
            "GET", f"/{self._quote_name(name)}/versions/{int(version)}"
        )
        return ResolvedPrompt.from_dict(self._unwrap_data(payload, "prompt version"))

    def resolve_prompt(
        self,
        name: str,
        *,
        version: int | None = None,
        label: str | None = "latest",
    ) -> ResolvedPrompt:
        query = self._build_query_string(version=version, label=label)
        payload = self._request_json(
            "GET",
            f"/{self._quote_name(name)}/resolve{query}",
        )
        return ResolvedPrompt.from_dict(self._unwrap_data(payload, "resolved prompt"))

    def get_prompt_analytics(
        self,
        name: str,
        *,
        recent_limit: int = 20,
        recent_page: int = 1,
    ) -> PromptAnalytics:
        query = self._build_query_string(
            recent_limit=recent_limit, recent_page=recent_page
        )
        payload = self._request_json(
            "GET",
            f"/{self._quote_name(name)}/analytics{query}",
        )
        return PromptAnalytics.from_dict(self._unwrap_data(payload, "prompt analytics"))

    def run_playground(
        self,
        name: str,
        *,
        version: int | None = None,
        label: str | None = "latest",
        prompt_text: str | None = None,
        chat_messages: list[ChatPromptMessage | dict] | None = None,
        variables: dict[str, Any] | None = None,
        provider: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        dry_run: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> PromptPlaygroundResult:
        payload: dict[str, Any] = {
            "variables": dict(variables or {}),
            "dry_run": dry_run,
            "metadata": dict(metadata or {}),
        }
        if version is not None:
            payload["version"] = version
        elif label is not None:
            payload["label"] = label
        if prompt_text is not None:
            payload["prompt_text"] = prompt_text
        if chat_messages is not None:
            payload["chat_messages"] = [
                self._serialize_message(message) for message in chat_messages
            ]
        if provider is not None:
            payload["provider"] = provider
        if model is not None:
            payload["model"] = model
        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        response = self._request_json(
            "POST",
            f"/{self._quote_name(name)}/playground/run",
            payload,
        )
        return PromptPlaygroundResult.from_dict(
            self._unwrap_data(response, "prompt playground")
        )

    def create_text_prompt(
        self,
        name: str,
        *,
        prompt_text: str,
        description: str | None = None,
        config: dict | None = None,
        commit_message: str | None = None,
        labels: list[str] | None = None,
        tags: list[str] | None = None,
    ) -> PromptDetail:
        payload = self._request_json(
            "POST",
            "",
            {
                "name": name,
                "prompt_type": PromptType.TEXT.value,
                "description": description,
                "prompt_text": prompt_text,
                "config": config or {},
                "commit_message": commit_message,
                "labels": labels or [],
                "tags": tags or [],
            },
        )
        return PromptDetail.from_dict(self._unwrap_data(payload, "prompt create"))

    def create_chat_prompt(
        self,
        name: str,
        *,
        chat_messages: list[ChatPromptMessage | dict],
        description: str | None = None,
        config: dict | None = None,
        commit_message: str | None = None,
        labels: list[str] | None = None,
        tags: list[str] | None = None,
    ) -> PromptDetail:
        payload = self._request_json(
            "POST",
            "",
            {
                "name": name,
                "prompt_type": PromptType.CHAT.value,
                "description": description,
                "chat_messages": [
                    self._serialize_message(message) for message in chat_messages
                ],
                "config": config or {},
                "commit_message": commit_message,
                "labels": labels or [],
                "tags": tags or [],
            },
        )
        return PromptDetail.from_dict(self._unwrap_data(payload, "prompt create"))

    def create_prompt_version(
        self,
        name: str,
        *,
        prompt_text: str | None = None,
        chat_messages: list[ChatPromptMessage | dict] | None = None,
        config: dict | None = None,
        commit_message: str | None = None,
        labels: list[str] | None = None,
    ) -> PromptDetail:
        payload: dict[str, object] = {
            "config": config or {},
            "commit_message": commit_message,
            "labels": labels or [],
        }
        if prompt_text is not None:
            payload["prompt_text"] = prompt_text
        if chat_messages is not None:
            payload["chat_messages"] = [
                self._serialize_message(message) for message in chat_messages
            ]

        response = self._request_json(
            "POST",
            f"/{self._quote_name(name)}/versions",
            payload,
        )
        return PromptDetail.from_dict(self._unwrap_data(response, "prompt version"))

    def update_prompt_labels(self, name: str, labels: dict[str, int]) -> PromptDetail:
        payload = self._request_json(
            "PATCH",
            f"/{self._quote_name(name)}/labels",
            {"labels": labels},
        )
        return PromptDetail.from_dict(self._unwrap_data(payload, "prompt labels"))

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
        self, method: str, path: str, payload: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        raise_if_backend_offline("PromptManagementClient request")
        encoded_payload = None
        if payload is not None:
            encoded_payload = json.dumps(payload).encode("utf-8")

        http_request = request.Request(
            f"{self.config.backend_origin}{self.config.api_path}{path}",
            data=encoded_payload,
            headers=self.config.build_headers(),
            method=method,
        )
        try:
            with open_no_redirect(  # nosec B310 - no-redirect opener, caller-configured endpoint
                http_request, timeout=self.config.request_timeout
            ) as response:
                body = response.read().decode("utf-8") if response else ""
                return json.loads(body) if body else {}
        except error.HTTPError as exc:
            try:
                body = exc.read().decode("utf-8") if exc.fp else ""
            finally:
                exc.close()
            self._raise_for_response(exc, body)
        except error.URLError as exc:
            raise TraigentConnectionError(
                f"Failed to connect to prompt backend at {self.config.backend_origin}"
            ) from exc

    def _raise_for_response(self, exc: error.HTTPError, body: str) -> NoReturn:
        """Raise the SDK exception that matches a failed response."""
        headers = exc.headers
        content_type = (headers.get("Content-Type") if headers else None) or ""
        if content_type.split(";")[0].strip().lower() == PROBLEM_CONTENT_TYPE:
            try:
                problem = json.loads(body)
            except ValueError:
                problem = None
            if isinstance(problem, dict):
                raise error_from_problem(exc.code, problem) from exc
        # Failures raised by shared layers before the prompts service keep a
        # different shape; map them by status.
        if exc.code in {401, 403}:
            raise AuthenticationError(
                f"Prompt request rejected with status {exc.code}"
            ) from exc
        if exc.code == 429:
            retry_after = None
            raw = headers.get("Retry-After") if headers else None
            try:
                retry_after = float(raw) if raw is not None else None
            except ValueError:
                retry_after = None
            raise RateLimitError(
                "Prompt request was rate limited", retry_after=retry_after
            ) from exc
        if exc.code == 503:
            raise ServiceUnavailableError(
                "Prompt service is temporarily unavailable", status_code=exc.code
            ) from exc
        raise ClientError(
            f"Prompt request failed with status {exc.code}",
            status_code=exc.code,
            details={"body": body},
        ) from exc

    def _unwrap_data(self, payload: dict, label: str) -> dict:
        data = payload.get("data")
        if not isinstance(data, dict):
            raise ClientError(f"Unexpected response structure for {label}")
        return data

    def _build_query_string(self, **params) -> str:
        serialized = {
            key: str(value)
            for key, value in params.items()
            if value is not None and value != ""
        }
        if not serialized:
            return ""
        return "?" + urlencode(serialized)

    def _quote_name(self, name: str) -> str:
        return quote(name, safe="")

    def _serialize_message(self, message: ChatPromptMessage | dict) -> dict:
        if isinstance(message, ChatPromptMessage):
            return message.to_dict()
        return {
            "role": str(message.get("role", "")),
            "content": str(message.get("content", "")),
            **(
                {"name": message.get("name")} if message.get("name") is not None else {}
            ),
        }
