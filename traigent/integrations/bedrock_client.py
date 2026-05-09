"""AWS Bedrock chat client wrapper used by paper experiments.

This minimal adapter focuses on Anthropic Claude chat models exposed via
`bedrock-runtime`. It provides a small, dependency-light surface area that
mirrors the shape used by our pipelines and evaluators.
# Traceability: CONC-Layer-Integration CONC-Quality-Compatibility CONC-Quality-Reliability FUNC-INTEGRATIONS REQ-INT-008 SYNC-IntegrationHook

Notes
-----
* Imports are optional; a helpful ImportError is raised when boto3 is missing.
* Both non-streaming and streaming interfaces are supported.
* Payload uses Anthropic Messages API format for Bedrock.
"""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import AsyncGenerator, Generator, Iterable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass


def _require_boto3():  # pragma: no cover - import guard
    try:
        import boto3
    except Exception as exc:  # noqa: BLE001 - want to capture all import failures
        raise ImportError(
            "boto3 is required for AWS Bedrock usage. Install with: \n"
            "  pip install boto3 botocore\n"
            "and ensure AWS credentials/region are configured."
        ) from exc
    return boto3


def _default_anthropic_version() -> str:
    # As of 2024-2025 this is the documented Bedrock Anthropic version identifier.
    return "bedrock-2023-05-31"


def _coerce_user_messages(
    prompt_or_messages: str | Iterable[str] | list[Mapping[str, Any]],
):
    if isinstance(prompt_or_messages, str):
        return [
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt_or_messages}],
            }
        ]
    if (
        isinstance(prompt_or_messages, list)
        and prompt_or_messages
        and isinstance(prompt_or_messages[0], Mapping)
    ):
        return prompt_or_messages
    # Iterable of strings -> one message per string
    return [
        {"role": "user", "content": [{"type": "text", "text": p}]}
        for p in prompt_or_messages
    ]


def _extract_text_from_messages_response(body: Mapping[str, Any]) -> str:
    # Anthropic messages API usually returns { content: [{type: "text", text: "..."}, ...] }
    pieces: list[str] = []
    for block in body.get("content", []) or []:
        if isinstance(block, Mapping) and block.get("type") == "text":
            text = str(block.get("text", ""))
            if text:
                pieces.append(text)
    return "".join(pieces).strip()


@dataclass
class BedrockChatResponse:
    text: str
    raw: Mapping[str, Any]
    usage: Mapping[str, Any] | None = None


class BedrockChatClient:
    """Thin wrapper around `boto3` Bedrock Runtime for Claude chat models."""

    def __init__(
        self,
        *,
        region_name: str | None = None,
        profile_name: str | None = None,
        client: Any | None = None,
    ) -> None:
        self._client = client
        self._region_name = region_name
        self._profile_name = profile_name

    def _ensure_client(self):  # pragma: no cover - network client guard
        if self._client is not None:
            return self._client
        boto3 = _require_boto3()
        if self._profile_name:
            session = boto3.session.Session(profile_name=self._profile_name)
        else:
            session = boto3.session.Session()
        self._client = session.client(
            "bedrock-runtime",
            region_name=self._region_name,
        )
        return self._client

    def invoke(
        self,
        *,
        model_id: str,
        messages: list[Mapping[str, Any]] | str | Iterable[str],
        max_tokens: int = 512,
        temperature: float = 0.5,
        top_p: float | None = None,
        extra_params: Mapping[str, Any] | None = None,
    ) -> BedrockChatResponse:
        """Perform a non-streaming chat invocation.

        Returns normalized `BedrockChatResponse` with text and raw payload.
        """

        client = self._ensure_client()

        if str(model_id).startswith("ai21."):
            return self._invoke_ai21(
                model_id=model_id,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            )

        payload: dict[str, Any] = {
            "anthropic_version": _default_anthropic_version(),
            "max_tokens": int(max_tokens),
            "messages": _coerce_user_messages(messages),
            "temperature": float(temperature),
        }
        if top_p is not None:
            payload["top_p"] = float(top_p)
        if extra_params:
            payload.update(dict(extra_params))

        resp = client.invoke_model(
            modelId=model_id,
            accept="application/json",
            contentType="application/json",
            body=json.dumps(payload),
        )

        # `body` can be streaming-like; ensure we read and parse JSON
        raw_body = resp.get("body")
        if hasattr(raw_body, "read"):
            data = json.loads(raw_body.read())
        else:
            data = json.loads(raw_body)

        text = _extract_text_from_messages_response(data)
        usage = data.get("usage") if isinstance(data, Mapping) else None
        return BedrockChatResponse(text=text, raw=data, usage=usage)

    def _invoke_ai21(
        self,
        *,
        model_id: str,
        messages: list[Mapping[str, Any]] | str | Iterable[str],
        max_tokens: int,
        temperature: float,
        top_p: float | None,
    ) -> BedrockChatResponse:
        """Invoke an AI21 Jamba family model via Bedrock."""

        if isinstance(messages, str):
            request_messages = [{"role": "user", "content": messages}]
        else:
            coalesced = _coerce_user_messages(messages)
            request_messages = []
            for item in coalesced:
                role = str(item.get("role", "user"))
                content = item.get("content", "")
                if isinstance(content, list):
                    text = "\n".join(
                        str(block.get("text", ""))
                        for block in content
                        if isinstance(block, Mapping)
                    )
                else:
                    text = str(content)
                request_messages.append({"role": role, "content": text})

        client = self._ensure_client()
        payload = {
            "messages": request_messages,
        }

        resp = client.invoke_model(
            modelId=model_id,
            accept="application/json",
            contentType="application/json",
            body=json.dumps(payload),
        )
        raw_body = resp.get("body")
        if hasattr(raw_body, "read"):
            data = json.loads(raw_body.read())
        else:
            data = json.loads(raw_body)

        text = ""
        if isinstance(data, Mapping):
            choices = data.get("choices") or []
            if choices:
                message = (
                    choices[0].get("message") if isinstance(choices[0], Mapping) else {}
                )
                if isinstance(message, Mapping):
                    text = str(message.get("content", "")).strip()
            if not text:
                text = str(data.get("outputText", "")).strip()
            usage = data.get("usage") or data.get("tokenUsage")
        else:
            text = str(data)
            usage = None
        return BedrockChatResponse(text=text, raw=data, usage=usage)

    def invoke_stream(
        self,
        *,
        model_id: str,
        messages: list[Mapping[str, Any]] | str | Iterable[str],
        max_tokens: int = 512,
        temperature: float = 0.5,
        top_p: float | None = None,
        extra_params: Mapping[str, Any] | None = None,
    ) -> Generator[str, None, BedrockChatResponse]:  # pragma: no cover - streaming path
        """Stream chat invocation; yields text chunks and returns final response."""

        client = self._ensure_client()

        payload: dict[str, Any] = {
            "anthropic_version": _default_anthropic_version(),
            "max_tokens": int(max_tokens),
            "messages": _coerce_user_messages(messages),
            "temperature": float(temperature),
        }
        if top_p is not None:
            payload["top_p"] = float(top_p)
        if extra_params:
            payload.update(dict(extra_params))

        final_text_parts: list[str] = []
        resp = client.invoke_model_with_response_stream(
            modelId=model_id,
            accept="application/json",
            contentType="application/json",
            body=json.dumps(payload),
        )

        stream = resp.get("body")
        if stream is not None:
            for event in stream:
                chunk = event.get("chunk") if isinstance(event, Mapping) else None
                if not chunk:
                    continue
                try:
                    data = json.loads(chunk.get("bytes").decode("utf-8"))
                except Exception:  # noqa: BLE001 - defensive
                    continue
                # Extract incremental text if present
                text_piece = _extract_text_from_messages_response(data)
                if text_piece:
                    final_text_parts.append(text_piece)
                    yield text_piece

        final = "".join(final_text_parts)
        return BedrockChatResponse(text=final, raw={"streamed": True}, usage=None)

    async def ainvoke(
        self,
        *,
        model_id: str,
        messages: list[Mapping[str, Any]] | str | Iterable[str],
        max_tokens: int = 512,
        temperature: float = 0.5,
        top_p: float | None = None,
        extra_params: Mapping[str, Any] | None = None,
    ) -> BedrockChatResponse:
        """Perform an async non-streaming chat invocation.

        Uses aioboto3 for true async or falls back to thread-pool executor.
        Returns normalized `BedrockChatResponse` with text and raw payload.
        """
        # Try aioboto3 for true async
        try:
            import aioboto3  # noqa: F401 - used for availability check and async method

            return await self._ainvoke_aioboto3(
                model_id=model_id,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                extra_params=extra_params,
            )
        except ImportError:
            # Fallback to running sync invoke in executor
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None,
                lambda: self.invoke(
                    model_id=model_id,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    extra_params=extra_params,
                ),
            )

    async def _ainvoke_aioboto3(
        self,
        *,
        model_id: str,
        messages: list[Mapping[str, Any]] | str | Iterable[str],
        max_tokens: int,
        temperature: float,
        top_p: float | None,
        extra_params: Mapping[str, Any] | None,
    ) -> BedrockChatResponse:
        """Internal async implementation using aioboto3."""
        import aioboto3

        payload: dict[str, Any] = {
            "anthropic_version": _default_anthropic_version(),
            "max_tokens": int(max_tokens),
            "messages": _coerce_user_messages(messages),
            "temperature": float(temperature),
        }
        if top_p is not None:
            payload["top_p"] = float(top_p)
        if extra_params:
            payload.update(dict(extra_params))

        session = (
            aioboto3.Session(profile_name=self._profile_name)
            if self._profile_name
            else aioboto3.Session()
        )
        async with session.client(
            "bedrock-runtime", region_name=self._region_name
        ) as client:
            resp = await client.invoke_model(
                modelId=model_id,
                accept="application/json",
                contentType="application/json",
                body=json.dumps(payload),
            )
            raw_body = resp.get("body")
            if hasattr(raw_body, "read"):
                body_bytes = await raw_body.read()
                data = json.loads(body_bytes)
            else:
                data = json.loads(raw_body)

        text = _extract_text_from_messages_response(data)
        usage = data.get("usage") if isinstance(data, Mapping) else None
        return BedrockChatResponse(text=text, raw=data, usage=usage)

    async def ainvoke_stream(
        self,
        *,
        model_id: str,
        messages: list[Mapping[str, Any]] | str | Iterable[str],
        max_tokens: int = 512,
        temperature: float = 0.5,
        top_p: float | None = None,
        extra_params: Mapping[str, Any] | None = None,
    ) -> AsyncGenerator[str, None]:
        """Async streaming chat invocation; yields text chunks.

        Uses aioboto3 for true async streaming or falls back to sync streaming
        wrapped in an executor.
        """
        # Try aioboto3 for true async streaming
        try:
            import aioboto3  # noqa: F401 - used for availability check and async stream

            async for chunk in self._ainvoke_stream_aioboto3(
                model_id=model_id,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                extra_params=extra_params,
            ):
                yield chunk
        except ImportError:
            # Fallback: run sync streaming in executor and yield chunks
            loop = asyncio.get_event_loop()
            sync_gen = self.invoke_stream(
                model_id=model_id,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                extra_params=extra_params,
            )

            # Collect chunks from sync generator in executor
            def _collect_chunks():
                chunks = []
                try:
                    for chunk in sync_gen:
                        chunks.append(chunk)
                except StopIteration:
                    pass
                return chunks

            chunks = await loop.run_in_executor(None, _collect_chunks)
            for chunk in chunks:
                yield chunk

    async def _ainvoke_stream_aioboto3(
        self,
        *,
        model_id: str,
        messages: list[Mapping[str, Any]] | str | Iterable[str],
        max_tokens: int,
        temperature: float,
        top_p: float | None,
        extra_params: Mapping[str, Any] | None,
    ) -> AsyncGenerator[str, None]:
        """Internal async streaming implementation using aioboto3."""
        import aioboto3

        payload: dict[str, Any] = {
            "anthropic_version": _default_anthropic_version(),
            "max_tokens": int(max_tokens),
            "messages": _coerce_user_messages(messages),
            "temperature": float(temperature),
        }
        if top_p is not None:
            payload["top_p"] = float(top_p)
        if extra_params:
            payload.update(dict(extra_params))

        session = (
            aioboto3.Session(profile_name=self._profile_name)
            if self._profile_name
            else aioboto3.Session()
        )
        async with session.client(
            "bedrock-runtime", region_name=self._region_name
        ) as client:
            resp = await client.invoke_model_with_response_stream(
                modelId=model_id,
                accept="application/json",
                contentType="application/json",
                body=json.dumps(payload),
            )
            stream = resp.get("body")
            if stream is not None:
                async for event in stream:
                    chunk = event.get("chunk") if isinstance(event, Mapping) else None
                    if not chunk:
                        continue
                    try:
                        data = json.loads(chunk.get("bytes").decode("utf-8"))
                    except Exception:  # noqa: BLE001 - defensive
                        continue
                    text_piece = _extract_text_from_messages_response(data)
                    if text_piece:
                        yield text_piece


def resolve_default_bedrock_model_id(model_hint: str | None) -> str:
    """Return a Bedrock model id for a given model hint.

    Allows environment override via `BEDROCK_MODEL_ID`. Fallback mapping supports
    common Claude variants used in our experiments.
    """

    override = os.getenv("BEDROCK_MODEL_ID")
    if override:
        return override

    hint = (model_hint or "").lower()
    # Update these defaults as account access evolves
    if "sonnet" in hint:
        return "anthropic.claude-3-sonnet-20240229-v1:0"
    if "haiku" in hint and "4.5" in hint:
        # Map to latest Haiku release; update ID as AWS publishes newer revisions.
        return "anthropic.claude-3-5-haiku-20241022-v1:0"
    if "haiku" in hint and "3.5" in hint:
        return "anthropic.claude-3-5-haiku-20241022-v1:0"
    if "haiku" in hint:
        return "anthropic.claude-3-haiku-20240307-v1:0"
    if "opus" in hint:
        return "anthropic.claude-3-opus-20240229-v1:0"
    if "jamba" in hint or "ai21" in hint:
        if "mini" in hint:
            return "ai21.jamba-1-5-mini-v1:0"
        if "large" in hint or "instruct" in hint:
            return "ai21.jamba-1-5-large-v1:0"
        # Default to the mini tier when size is unspecified to favour lower cost.
        return "ai21.jamba-1-5-mini-v1:0"
    # Sensible default for Claude 3 family
    return "anthropic.claude-3-sonnet-20240229-v1:0"
