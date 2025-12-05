"""Azure OpenAI chat client wrapper for paper experiments.

Uses the official OpenAI Python client configured for Azure endpoints.
Provides a tiny surface with optional mock mode for CI: set AZURE_OPENAI_MOCK=true.
"""

# Traceability: CONC-Layer-Integration CONC-Quality-Compatibility CONC-Quality-Reliability FUNC-INTEGRATIONS REQ-INT-008 SYNC-IntegrationHook

from __future__ import annotations

import os
from collections.abc import AsyncGenerator, Generator, Iterable, Mapping
from dataclasses import dataclass
from typing import Any


def _coerce_messages(prompt_or_messages: str | Iterable[str] | list[Mapping[str, Any]]):
    if isinstance(prompt_or_messages, str):
        return [{"role": "user", "content": prompt_or_messages}]
    if (
        isinstance(prompt_or_messages, list)
        and prompt_or_messages
        and isinstance(prompt_or_messages[0], Mapping)
    ):
        return prompt_or_messages
    return [{"role": "user", "content": p} for p in prompt_or_messages]


@dataclass
class AzureOpenAIChatResponse:
    text: str
    raw: Mapping[str, Any]
    usage: Mapping[str, Any] | None = None


class AzureOpenAIChatClient:
    """Thin wrapper around OpenAI client configured for Azure."""

    def __init__(
        self,
        *,
        endpoint: str | None = None,
        api_key: str | None = None,
        api_version: str | None = None,
    ) -> None:
        self._endpoint = endpoint or os.getenv("AZURE_OPENAI_ENDPOINT") or ""
        self._api_key = api_key or os.getenv("AZURE_OPENAI_API_KEY") or ""
        self._api_version = (
            api_version or os.getenv("AZURE_OPENAI_API_VERSION") or "2024-02-15-preview"
        )

    def invoke(
        self,
        *,
        deployment: str,
        messages: list[Mapping[str, Any]] | str | Iterable[str],
        max_tokens: int = 512,
        temperature: float = 0.5,
        top_p: float | None = None,
        extra_params: Mapping[str, Any] | None = None,
    ) -> AzureOpenAIChatResponse:
        """Perform a non-streaming chat invocation using an Azure OpenAI deployment."""

        if os.getenv("AZURE_OPENAI_MOCK", "").strip().lower() == "true":
            msgs = _coerce_messages(messages)
            last = msgs[-1]["content"] if msgs else ""
            data: dict[str, Any] = {
                "mock": True,
                "choices": [
                    {"message": {"content": f"[MOCK_AZURE:{deployment}] {last}"}}
                ],
                "usage": {"prompt_tokens": 0, "completion_tokens": min(max_tokens, 32)},
            }
            return AzureOpenAIChatResponse(
                text=data["choices"][0]["message"]["content"],
                raw=data,
                usage=data.get("usage"),
            )

        try:
            from openai import OpenAI
        except Exception as exc:  # pragma: no cover - optional dependency
            raise ImportError("Install 'openai' for Azure OpenAI usage.") from exc

        base_url = f"{self._endpoint.rstrip('/')}/openai/deployments/{deployment}"
        client = OpenAI(
            api_key=self._api_key,
            base_url=base_url,
            default_query={"api-version": self._api_version},
        )

        payload: dict[str, Any] = {
            "messages": _coerce_messages(messages),
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
        }
        if top_p is not None:
            payload["top_p"] = float(top_p)
        if extra_params:
            payload.update(dict(extra_params))

        response = client.chat.completions.create(**payload)
        usage = getattr(response, "usage", None)
        try:
            usage_payload = usage.model_dump() if usage is not None else None
        except Exception:
            usage_payload = dict(usage) if usage is not None else None
        text = response.choices[0].message.content.strip()
        return AzureOpenAIChatResponse(text=text, raw=response, usage=usage_payload)

    async def ainvoke(
        self,
        *,
        deployment: str,
        messages: list[Mapping[str, Any]] | str | Iterable[str],
        max_tokens: int = 512,
        temperature: float = 0.5,
        top_p: float | None = None,
        extra_params: Mapping[str, Any] | None = None,
    ) -> AzureOpenAIChatResponse:
        """Perform an async non-streaming chat invocation using Azure OpenAI."""

        if os.getenv("AZURE_OPENAI_MOCK", "").strip().lower() == "true":
            msgs = _coerce_messages(messages)
            last = msgs[-1]["content"] if msgs else ""
            data: dict[str, Any] = {
                "mock": True,
                "choices": [
                    {"message": {"content": f"[MOCK_AZURE:{deployment}] {last}"}}
                ],
                "usage": {"prompt_tokens": 0, "completion_tokens": min(max_tokens, 32)},
            }
            return AzureOpenAIChatResponse(
                text=data["choices"][0]["message"]["content"],
                raw=data,
                usage=data.get("usage"),
            )

        try:
            from openai import AsyncOpenAI
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("Install 'openai' for Azure OpenAI async usage.") from exc

        base_url = f"{self._endpoint.rstrip('/')}/openai/deployments/{deployment}"
        client = AsyncOpenAI(
            api_key=self._api_key,
            base_url=base_url,
            default_query={"api-version": self._api_version},
        )

        payload: dict[str, Any] = {
            "messages": _coerce_messages(messages),
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
        }
        if top_p is not None:
            payload["top_p"] = float(top_p)
        if extra_params:
            payload.update(dict(extra_params))

        response = await client.chat.completions.create(**payload)
        usage = getattr(response, "usage", None)
        try:
            usage_payload = usage.model_dump() if usage is not None else None
        except Exception:
            usage_payload = dict(usage) if usage is not None else None
        text = response.choices[0].message.content.strip()
        return AzureOpenAIChatResponse(text=text, raw=response, usage=usage_payload)

    def invoke_stream(
        self,
        *,
        deployment: str,
        messages: list[Mapping[str, Any]] | str | Iterable[str],
        max_tokens: int = 512,
        temperature: float = 0.5,
        top_p: float | None = None,
        extra_params: Mapping[str, Any] | None = None,
    ) -> Generator[str, None, AzureOpenAIChatResponse]:
        """Perform a sync streaming chat invocation; yields text chunks and returns final response."""

        if os.getenv("AZURE_OPENAI_MOCK", "").strip().lower() == "true":
            msgs = _coerce_messages(messages)
            last = msgs[-1]["content"] if msgs else ""
            chunks = [
                f"[MOCK_AZURE:{deployment}] ",
                last[: len(last) // 2],
                last[len(last) // 2 :],
            ]
            full_text = "".join(chunks)
            for chunk in chunks:
                yield chunk

            data: dict[str, Any] = {
                "mock": True,
                "choices": [{"message": {"content": full_text}}],
                "usage": {"prompt_tokens": 0, "completion_tokens": min(max_tokens, 32)},
            }
            return AzureOpenAIChatResponse(
                text=full_text, raw=data, usage=data.get("usage")
            )

        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "Install 'openai' for Azure OpenAI streaming usage."
            ) from exc

        base_url = f"{self._endpoint.rstrip('/')}/openai/deployments/{deployment}"
        client = OpenAI(
            api_key=self._api_key,
            base_url=base_url,
            default_query={"api-version": self._api_version},
        )

        payload: dict[str, Any] = {
            "messages": _coerce_messages(messages),
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
            "stream": True,
        }
        if top_p is not None:
            payload["top_p"] = float(top_p)
        if extra_params:
            payload.update(dict(extra_params))

        response = client.chat.completions.create(**payload)
        full_text_parts = []
        for chunk in response:
            if (
                chunk.choices
                and chunk.choices[0].delta
                and chunk.choices[0].delta.content
            ):
                text_chunk = chunk.choices[0].delta.content
                full_text_parts.append(text_chunk)
                yield text_chunk

        full_text = "".join(full_text_parts)
        return AzureOpenAIChatResponse(
            text=full_text, raw={"streamed": True}, usage=None
        )

    async def ainvoke_stream(
        self,
        *,
        deployment: str,
        messages: list[Mapping[str, Any]] | str | Iterable[str],
        max_tokens: int = 512,
        temperature: float = 0.5,
        top_p: float | None = None,
        extra_params: Mapping[str, Any] | None = None,
    ) -> AsyncGenerator[str, None]:
        """Perform an async streaming chat invocation; yields text chunks."""

        if os.getenv("AZURE_OPENAI_MOCK", "").strip().lower() == "true":
            msgs = _coerce_messages(messages)
            last = msgs[-1]["content"] if msgs else ""
            chunks = [
                f"[MOCK_AZURE:{deployment}] ",
                last[: len(last) // 2],
                last[len(last) // 2 :],
            ]
            for chunk in chunks:
                yield chunk
            return

        try:
            from openai import AsyncOpenAI
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "Install 'openai' for Azure OpenAI async streaming usage."
            ) from exc

        base_url = f"{self._endpoint.rstrip('/')}/openai/deployments/{deployment}"
        client = AsyncOpenAI(
            api_key=self._api_key,
            base_url=base_url,
            default_query={"api-version": self._api_version},
        )

        payload: dict[str, Any] = {
            "messages": _coerce_messages(messages),
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
            "stream": True,
        }
        if top_p is not None:
            payload["top_p"] = float(top_p)
        if extra_params:
            payload.update(dict(extra_params))

        response = await client.chat.completions.create(**payload)
        async for chunk in response:
            if (
                chunk.choices
                and chunk.choices[0].delta
                and chunk.choices[0].delta.content
            ):
                yield chunk.choices[0].delta.content
