"""Google Generative AI (Gemini) chat client wrapper for paper experiments.

Provides a tiny, mockable surface compatible with our pipelines.
Set GEMINI_MOCK=true to bypass SDK/network and get deterministic echoes.
"""

# Traceability: CONC-Layer-Integration CONC-Quality-Compatibility CONC-Quality-Reliability FUNC-INTEGRATIONS REQ-INT-008 SYNC-IntegrationHook

from __future__ import annotations

import os
from collections.abc import AsyncGenerator, Generator, Iterable, Mapping
from dataclasses import dataclass
from typing import Any, cast


def _coerce_messages(prompt_or_messages: str | Iterable[str] | list[Mapping[str, Any]]):
    if isinstance(prompt_or_messages, str):
        return [prompt_or_messages]
    if (
        isinstance(prompt_or_messages, list)
        and prompt_or_messages
        and isinstance(prompt_or_messages[0], Mapping)
    ):
        # Extract text content from role/content mappings
        texts: list[str] = []
        for m in prompt_or_messages:
            content = m.get("content")
            if isinstance(content, str):
                texts.append(content)
            elif isinstance(content, list):
                texts.extend(
                    str(part.get("text", ""))
                    for part in content
                    if isinstance(part, Mapping)
                )
        return texts
    return list(prompt_or_messages)


@dataclass
class GeminiChatResponse:
    text: str
    raw: Mapping[str, Any]
    usage: Mapping[str, Any] | None = None


class GeminiChatClient:
    """Thin wrapper around google-generativeai for Gemini chat."""

    def __init__(self, *, api_key: str | None = None) -> None:
        self._api_key = api_key or os.getenv("GOOGLE_API_KEY") or ""

    def invoke(
        self,
        *,
        model: str,
        messages: list[Mapping[str, Any]] | str | Iterable[str],
        max_tokens: int = 512,
        temperature: float = 0.5,
        extra_params: Mapping[str, Any] | None = None,
    ) -> GeminiChatResponse:
        if os.getenv("GEMINI_MOCK", "").strip().lower() == "true":
            texts = _coerce_messages(messages)
            last = texts[-1] if texts else ""
            data: dict[str, Any] = {
                "mock": True,
                "candidates": [
                    {"content": {"parts": [{"text": f"[MOCK_GEMINI:{model}] {last}"}]}}
                ],
                "usageMetadata": {
                    "promptTokenCount": 0,
                    "candidatesTokenCount": min(max_tokens, 32),
                },
            }
            text = data["candidates"][0]["content"]["parts"][0]["text"]
            return GeminiChatResponse(
                text=text,
                raw=data,
                usage=cast("Mapping[str, Any] | None", data.get("usageMetadata")),
            )

        try:
            import google.generativeai as genai
        except Exception as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "Install 'google-generativeai' for Gemini usage."
            ) from exc

        genai.configure(api_key=self._api_key)
        client = genai.GenerativeModel(model)
        kwargs: dict[str, Any] = {"temperature": float(temperature)}
        if extra_params:
            kwargs.update(dict(extra_params))
        # Join content for basic chats
        texts = _coerce_messages(messages)
        prompt = "\n\n".join(texts)
        response = client.generate_content(prompt, **kwargs)

        try:
            text = response.text or ""
        except Exception:
            text = ""
        usage = getattr(response, "usage_metadata", None)
        try:
            usage_payload = usage.to_dict() if usage is not None else None
        except Exception:
            usage_payload = None
        return GeminiChatResponse(text=text.strip(), raw=response, usage=usage_payload)

    def invoke_stream(
        self,
        *,
        model: str,
        messages: list[Mapping[str, Any]] | str | Iterable[str],
        max_tokens: int = 512,
        temperature: float = 0.5,
        extra_params: Mapping[str, Any] | None = None,
    ) -> Generator[str, None, GeminiChatResponse]:
        if os.getenv("GEMINI_MOCK", "").strip().lower() == "true":
            texts = _coerce_messages(messages)
            last = texts[-1] if texts else ""
            chunks = [
                f"[MOCK_GEMINI:{model}] ",
                last[: max(1, len(last) // 2)],
                last[max(1, len(last) // 2) :],
            ]
            full_text = "".join(chunks)
            for chunk in chunks:
                yield chunk

            data = {
                "mock": True,
                "candidates": [{"content": {"parts": [{"text": full_text}]}}],
                "usageMetadata": {
                    "promptTokenCount": 0,
                    "candidatesTokenCount": min(max_tokens, 32),
                },
            }
            return GeminiChatResponse(
                text=full_text,
                raw=data,
                usage=cast("Mapping[str, Any] | None", data.get("usageMetadata")),
            )

        try:
            import google.generativeai as genai
        except Exception as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "Install 'google-generativeai' for Gemini usage."
            ) from exc

        genai.configure(api_key=self._api_key)
        client = genai.GenerativeModel(model)
        kwargs: dict[str, Any] = {"temperature": float(temperature), "stream": True}
        if extra_params:
            kwargs.update(dict(extra_params))

        texts = _coerce_messages(messages)
        prompt = "\n\n".join(texts)
        response = client.generate_content(prompt, **kwargs)

        full_text_parts = []
        for chunk in response:
            text_chunk = chunk.text
            if text_chunk:
                full_text_parts.append(text_chunk)
                yield text_chunk

        full_text = "".join(full_text_parts)
        usage = getattr(response, "usage_metadata", None)
        try:
            usage_payload = usage.to_dict() if usage is not None else None
        except Exception:
            usage_payload = None

        return GeminiChatResponse(text=full_text, raw=response, usage=usage_payload)

    async def ainvoke(
        self,
        *,
        model: str,
        messages: list[Mapping[str, Any]] | str | Iterable[str],
        max_tokens: int = 512,
        temperature: float = 0.5,
        extra_params: Mapping[str, Any] | None = None,
    ) -> GeminiChatResponse:
        """Perform an async non-streaming chat invocation using Gemini."""

        if os.getenv("GEMINI_MOCK", "").strip().lower() == "true":
            texts = _coerce_messages(messages)
            last = texts[-1] if texts else ""
            data: dict[str, Any] = {
                "mock": True,
                "candidates": [
                    {"content": {"parts": [{"text": f"[MOCK_GEMINI:{model}] {last}"}]}}
                ],
                "usageMetadata": {
                    "promptTokenCount": 0,
                    "candidatesTokenCount": min(max_tokens, 32),
                },
            }
            text = data["candidates"][0]["content"]["parts"][0]["text"]
            return GeminiChatResponse(
                text=text,
                raw=data,
                usage=cast("Mapping[str, Any] | None", data.get("usageMetadata")),
            )

        try:
            import google.generativeai as genai
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "Install 'google-generativeai' for Gemini async usage."
            ) from exc

        genai.configure(api_key=self._api_key)
        client = genai.GenerativeModel(model)
        kwargs: dict[str, Any] = {"temperature": float(temperature)}
        if extra_params:
            kwargs.update(dict(extra_params))

        texts = _coerce_messages(messages)
        prompt = "\n\n".join(texts)
        response = await client.generate_content_async(prompt, **kwargs)

        try:
            text = response.text or ""
        except Exception:
            text = ""
        usage = getattr(response, "usage_metadata", None)
        try:
            usage_payload = usage.to_dict() if usage is not None else None
        except Exception:
            usage_payload = None
        return GeminiChatResponse(text=text.strip(), raw=response, usage=usage_payload)

    async def ainvoke_stream(
        self,
        *,
        model: str,
        messages: list[Mapping[str, Any]] | str | Iterable[str],
        max_tokens: int = 512,
        temperature: float = 0.5,
        extra_params: Mapping[str, Any] | None = None,
    ) -> AsyncGenerator[str, None]:
        """Perform an async streaming chat invocation; yields text chunks."""

        if os.getenv("GEMINI_MOCK", "").strip().lower() == "true":
            texts = _coerce_messages(messages)
            last = texts[-1] if texts else ""
            chunks = [
                f"[MOCK_GEMINI:{model}] ",
                last[: max(1, len(last) // 2)],
                last[max(1, len(last) // 2) :],
            ]
            for chunk in chunks:
                yield chunk
            return

        try:
            import google.generativeai as genai
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "Install 'google-generativeai' for Gemini async streaming usage."
            ) from exc

        genai.configure(api_key=self._api_key)
        client = genai.GenerativeModel(model)
        kwargs: dict[str, Any] = {"temperature": float(temperature), "stream": True}
        if extra_params:
            kwargs.update(dict(extra_params))

        texts = _coerce_messages(messages)
        prompt = "\n\n".join(texts)
        response = await client.generate_content_async(prompt, **kwargs)

        async for chunk in response:
            text_chunk = chunk.text
            if text_chunk:
                yield text_chunk
