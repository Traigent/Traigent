"""Sync-friendly prompt management client."""

from __future__ import annotations

import json
from typing import Any, cast
from urllib import error, request
from urllib.parse import quote, urlencode

from traigent.prompts.config import PromptManagementConfig
from traigent.prompts.dtos import (
    ChatPromptMessage,
    PromptAnalytics,
    PromptDetail,
    PromptListResponse,
    PromptType,
    ResolvedPrompt,
)
from traigent.utils.exceptions import (
    AuthenticationError,
    ClientError,
    TraigentConnectionError,
)


class PromptManagementClient:
    """Client for versioned prompt registry operations.

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
        page: int = 1,
        per_page: int = 20,
        search: str | None = None,
        prompt_type: PromptType | str | None = None,
        label: str | None = None,
    ) -> PromptListResponse:
        if isinstance(prompt_type, str):
            prompt_type = PromptType(prompt_type)
        query = self._build_query_string(
            page=page,
            per_page=per_page,
            search=search,
            prompt_type=prompt_type.value if prompt_type else None,
            label=label,
        )
        payload = self._request_json("GET", query or "")
        return PromptListResponse.from_dict(self._unwrap_data(payload, "prompt list"))

    def get_prompt(self, name: str) -> PromptDetail:
        payload = self._request_json("GET", f"/{self._quote_name(name)}")
        return PromptDetail.from_dict(self._unwrap_data(payload, "prompt detail"))

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
            with request.urlopen(  # nosec B310 - backend_origin is caller-configured API endpoint
                http_request, timeout=self.config.request_timeout
            ) as response:
                status_code = getattr(response, "status", 200)
                body = response.read().decode("utf-8") if response else ""
                parsed = json.loads(body) if body else {}
                if status_code >= 400:
                    raise ClientError(
                        f"Prompt request failed with status {status_code}",
                        status_code=status_code,
                        details={"body": body},
                    )
                return parsed
        except error.HTTPError as exc:
            try:
                body = exc.read().decode("utf-8") if exc.fp else ""
            finally:
                exc.close()
            if exc.code in {401, 403}:
                raise AuthenticationError(
                    f"Prompt request rejected with status {exc.code}"
                ) from exc
            raise ClientError(
                f"Prompt request failed with status {exc.code}",
                status_code=exc.code,
                details={"body": body},
            ) from exc
        except error.URLError as exc:
            raise TraigentConnectionError(
                f"Failed to connect to prompt backend at {self.config.backend_origin}"
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
        return quote(name, safe="/")

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
