"""LangChain response interceptor for capturing token metadata.

This module provides utilities to capture LangChain response metadata
that would otherwise be lost when functions return only strings.
"""

# Traceability: CONC-Layer-Integration CONC-Quality-Observability CONC-Quality-Compatibility FUNC-INTEGRATIONS REQ-INT-008 SYNC-IntegrationHook

import threading
import time
from contextlib import contextmanager
from typing import Any, cast

from traigent.utils.logging import get_logger

logger = get_logger(__name__)


class LangChainMetadataCapture:
    """Thread-safe storage for LangChain response metadata."""

    def __init__(self) -> None:
        self._storage = threading.local()
        self._lock = threading.Lock()
        # Keep all responses in order
        self._all_responses: list[Any] = []
        self._response_lock = threading.Lock()
        # Map correlation keys (e.g., example_id) to responses
        self._by_key: dict[str, Any] = {}
        self._by_key_lock = threading.Lock()
        # Current correlation key in thread local
        self._key_local = threading.local()

    def set_last_response(self, response: Any) -> None:
        """Store the last LangChain response."""
        with self._lock:
            if not hasattr(self._storage, "responses"):
                self._storage.responses = []
            self._storage.responses.append(response)
            logger.debug("Captured LangChain response with metadata")

        # Also store in global list for batch processing
        with self._response_lock:
            self._all_responses.append(response)

        # Also store by correlation key if present
        key = getattr(self._key_local, "current_key", None)
        if key is not None:
            with self._by_key_lock:
                self._by_key[key] = response

    def get_last_response(self) -> Any | None:
        """Get and clear the last LangChain response."""
        with self._lock:
            if hasattr(self._storage, "responses") and self._storage.responses:
                response = self._storage.responses.pop()
                logger.debug("Retrieved captured LangChain response")
                return response
            return None

    def get_all_responses(self) -> list[Any]:
        """Get all captured responses (for batch processing)."""
        with self._response_lock:
            return self._all_responses.copy()

    def clear(self) -> None:
        """Clear all stored responses."""
        with self._lock:
            if hasattr(self._storage, "responses"):
                self._storage.responses.clear()
        with self._response_lock:
            self._all_responses.clear()
        with self._by_key_lock:
            self._by_key.clear()
        if hasattr(self._key_local, "current_key"):
            self._key_local.current_key = None

    # Key management
    def set_current_key(self, key: Any) -> None:
        self._key_local.current_key = key

    def clear_current_key(self) -> None:
        self._key_local.current_key = None

    def get_by_key(self, key: Any) -> Any:
        with self._by_key_lock:
            return self._by_key.get(key)


# Global instance for metadata capture
_metadata_capture = LangChainMetadataCapture()


def capture_langchain_response(response: Any) -> Any:
    """Capture a LangChain response for metadata extraction.

    This should be called immediately after getting a response from LangChain
    but before returning just the string content.
    """
    _metadata_capture.set_last_response(response)
    return response


def get_captured_response() -> Any | None:
    """Retrieve the last captured LangChain response."""
    return _metadata_capture.get_last_response()


def get_all_captured_responses() -> list[Any]:
    """Get all captured responses for batch processing."""
    return _metadata_capture.get_all_responses()


def get_captured_response_by_key(key: Any) -> Any | None:
    """Get captured response by correlation key (e.g., example_id)."""
    return _metadata_capture.get_by_key(key)


def clear_captured_responses() -> None:
    """Clear all captured responses."""
    _metadata_capture.clear()


@contextmanager
def langchain_metadata_context():
    """Context manager for capturing LangChain metadata within a scope."""
    try:
        clear_captured_responses()
        yield _metadata_capture
    finally:
        clear_captured_responses()


@contextmanager
def capture_key(key: Any):
    """Context manager to associate subsequent captured responses with a key."""
    try:
        _metadata_capture.set_current_key(key)
        yield
    finally:
        _metadata_capture.clear_current_key()


def _create_stream_wrapper(original_meth: Any) -> Any:
    def stream_wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        start_time = time.perf_counter()
        last = None
        for chunk in original_meth(self, *args, **kwargs):
            last = chunk
            yield chunk
        if last is not None:
            response_time_ms = (time.perf_counter() - start_time) * 1000
            if not hasattr(last, "response_metadata"):
                last.response_metadata = {}
            last.response_metadata["response_time_ms"] = response_time_ms
            capture_langchain_response(last)

    return stream_wrapper


def _create_astream_wrapper(original_meth: Any) -> Any:
    async def astream_wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        start_time = time.perf_counter()
        last = None
        async for chunk in original_meth(self, *args, **kwargs):
            last = chunk
            yield chunk
        if last is not None:
            response_time_ms = (time.perf_counter() - start_time) * 1000
            if not hasattr(last, "response_metadata"):
                last.response_metadata = {}
            last.response_metadata["response_time_ms"] = response_time_ms
            capture_langchain_response(last)

    return astream_wrapper


def _extract_bedrock_model_name(instance: Any) -> str:
    return str(
        getattr(instance, "model_id", None)
        or getattr(instance, "model", None)
        or getattr(instance, "model_name", None)
        or "mock-model"
    )


def _normalize_langchain_usage_metadata(response: Any) -> None:
    """Normalize provider usage aliases into LangChain usage_metadata keys."""
    usage = getattr(response, "usage_metadata", None)
    if not isinstance(usage, dict):
        metadata = getattr(response, "response_metadata", None)
        if isinstance(metadata, dict):
            candidate = metadata.get("usage") or metadata.get("token_usage")
            usage = candidate if isinstance(candidate, dict) else None
    if not isinstance(usage, dict):
        return

    input_tokens = (
        usage.get("input_tokens")
        if usage.get("input_tokens") is not None
        else usage.get("prompt_tokens")
    )
    if input_tokens is None:
        input_tokens = usage.get("inputTokens")
    output_tokens = (
        usage.get("output_tokens")
        if usage.get("output_tokens") is not None
        else usage.get("completion_tokens")
    )
    if output_tokens is None:
        output_tokens = usage.get("outputTokens")
    total_tokens = (
        usage.get("total_tokens")
        if usage.get("total_tokens") is not None
        else usage.get("totalTokens")
    )

    normalized: dict[str, int] = {}
    try:
        if input_tokens is not None:
            normalized["input_tokens"] = int(input_tokens)
        if output_tokens is not None:
            normalized["output_tokens"] = int(output_tokens)
        if total_tokens is not None:
            normalized["total_tokens"] = int(total_tokens)
        elif normalized:
            normalized["total_tokens"] = normalized.get(
                "input_tokens", 0
            ) + normalized.get("output_tokens", 0)
    except (TypeError, ValueError):
        return

    if normalized:
        response.usage_metadata = normalized


def _create_bedrock_invoke_wrapper(original_invoke: Any) -> Any:
    def invoke_with_capture_bedrock(self: Any, *args: Any, **kwargs: Any) -> Any:
        """Wrapped invoke method that captures Bedrock responses with timing."""
        from traigent.integrations.utils.mock_adapter import MockAdapter

        model_name = _extract_bedrock_model_name(self)
        if MockAdapter.is_mock_enabled("bedrock"):
            from langchain_core.messages import AIMessage

            mock_data = MockAdapter.get_mock_response("anthropic", model=model_name)
            content = mock_data["content"][0]["text"]
            input_tokens = int(mock_data["usage"]["input_tokens"])
            output_tokens = int(mock_data["usage"]["output_tokens"])
            response = AIMessage(
                content=content,
                response_metadata={
                    "model_name": mock_data["model"],
                    "provider": "bedrock",
                    "stop_reason": mock_data["stop_reason"],
                    "response_time_ms": 0.0,
                },
                usage_metadata={
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": input_tokens + output_tokens,
                },
            )
            capture_langchain_response(response)
            return response

        start_time = time.perf_counter()
        response = original_invoke(self, *args, **kwargs)
        response_time_ms = (time.perf_counter() - start_time) * 1000

        if not hasattr(response, "response_metadata"):
            response.response_metadata = {}
        response.response_metadata["response_time_ms"] = response_time_ms
        if "model_name" not in response.response_metadata:
            response.response_metadata["model_name"] = model_name
        _normalize_langchain_usage_metadata(response)

        capture_langchain_response(response)
        logger.debug(
            "Captured Bedrock LangChain invoke usage: %s, response_time_ms: %.2f",
            getattr(response, "usage_metadata", None),
            response_time_ms,
        )
        return response

    return invoke_with_capture_bedrock


def patch_langchain_for_metadata_capture() -> bool:
    """Monkey-patch LangChain to automatically capture response metadata.

    This patches both ChatAnthropic and ChatOpenAI invoke methods to capture responses.
    """
    patched_any = False

    # Patch ChatAnthropic
    try:
        from langchain_anthropic import ChatAnthropic

        # Store original method
        if getattr(ChatAnthropic, "_traigent_patched_invoke", False) is False:
            original_invoke = ChatAnthropic.invoke

            def invoke_with_capture_anthropic(
                self: Any, *args: Any, **kwargs: Any
            ) -> Any:
                """Wrapped invoke method that captures response with timing."""
                # Check if mock mode is enabled — return mock response without API call
                from traigent.integrations.utils.mock_adapter import MockAdapter

                if MockAdapter.is_mock_enabled("anthropic"):
                    from langchain_core.messages import AIMessage

                    model_name = getattr(self, "model", "mock-model")
                    mock_data = MockAdapter.get_mock_response(
                        "anthropic", model=model_name
                    )
                    content = mock_data["content"][0]["text"]
                    response = AIMessage(
                        content=content,
                        response_metadata={
                            "model_name": mock_data["model"],
                            "stop_reason": mock_data["stop_reason"],
                            "response_time_ms": 0.0,
                        },
                        usage_metadata={
                            "input_tokens": mock_data["usage"]["input_tokens"],
                            "output_tokens": mock_data["usage"]["output_tokens"],
                            "total_tokens": mock_data["usage"]["input_tokens"]
                            + mock_data["usage"]["output_tokens"],
                        },
                    )
                    capture_langchain_response(response)
                    return response

                start_time = time.perf_counter()
                response = original_invoke(self, *args, **kwargs)
                response_time_ms = (time.perf_counter() - start_time) * 1000

                # Inject timing into response metadata
                if not hasattr(response, "response_metadata"):
                    response.response_metadata = {}
                response.response_metadata["response_time_ms"] = response_time_ms

                capture_langchain_response(response)
                logger.debug(
                    f"Captured ChatAnthropic invoke usage: {getattr(response, 'usage_metadata', None)}, "
                    f"response_time_ms: {response_time_ms:.2f}"
                )
                return response

            ChatAnthropic.invoke = invoke_with_capture_anthropic
            ChatAnthropic._traigent_patched_invoke = True
            logger.info("✅ Patched ChatAnthropic.invoke for metadata capture")
            patched_any = True

        # Patch stream methods if available
        for meth_name, flag in [
            ("stream", "_traigent_patched_stream"),
            ("astream", "_traigent_patched_astream"),
        ]:
            if hasattr(ChatAnthropic, meth_name) and not getattr(
                ChatAnthropic, flag, False
            ):
                original_meth = getattr(ChatAnthropic, meth_name)

                if meth_name == "stream":
                    setattr(
                        ChatAnthropic, meth_name, _create_stream_wrapper(original_meth)
                    )
                else:
                    setattr(
                        ChatAnthropic, meth_name, _create_astream_wrapper(original_meth)
                    )
                setattr(ChatAnthropic, flag, True)
                logger.info(
                    f"✅ Patched ChatAnthropic.{meth_name} for metadata capture"
                )

    except ImportError:
        logger.debug("ChatAnthropic not available, skipping")
    except Exception as e:
        logger.error(f"Failed to patch ChatAnthropic: {e}")

    # Patch ChatOpenAI
    try:
        from langchain_openai import ChatOpenAI

        if getattr(ChatOpenAI, "_traigent_patched_invoke", False) is False:
            original_invoke_openai = ChatOpenAI.invoke

            def invoke_with_capture_openai(self: Any, *args: Any, **kwargs: Any) -> Any:
                """Wrapped invoke method that captures response with timing."""
                # Check if mock mode is enabled — return mock response without API call
                from traigent.integrations.utils.mock_adapter import MockAdapter

                if MockAdapter.is_mock_enabled("openai"):
                    from langchain_core.messages import AIMessage

                    model_name = getattr(self, "model_name", None) or getattr(
                        self, "model", "mock-model"
                    )
                    mock_data = MockAdapter.get_mock_response(
                        "openai", model=model_name
                    )
                    content = mock_data["choices"][0]["message"]["content"]
                    response = AIMessage(
                        content=content,
                        response_metadata={
                            "model_name": mock_data["model"],
                            "finish_reason": mock_data["choices"][0]["finish_reason"],
                            "response_time_ms": 0.0,
                        },
                        usage_metadata={
                            "input_tokens": mock_data["usage"]["prompt_tokens"],
                            "output_tokens": mock_data["usage"]["completion_tokens"],
                            "total_tokens": mock_data["usage"]["total_tokens"],
                        },
                    )
                    capture_langchain_response(response)
                    return response

                start_time = time.perf_counter()
                response = original_invoke_openai(self, *args, **kwargs)
                response_time_ms = (time.perf_counter() - start_time) * 1000

                # Inject timing into response metadata
                if not hasattr(response, "response_metadata"):
                    response.response_metadata = {}
                response.response_metadata["response_time_ms"] = response_time_ms

                capture_langchain_response(response)
                logger.debug(
                    f"Captured ChatOpenAI invoke usage: {getattr(response, 'usage_metadata', None)}, "
                    f"response_time_ms: {response_time_ms:.2f}"
                )
                return response

            ChatOpenAI.invoke = invoke_with_capture_openai
            ChatOpenAI._traigent_patched_invoke = True
            logger.info("✅ Patched ChatOpenAI.invoke for metadata capture")
            patched_any = True

        for meth_name, flag in [
            ("stream", "_traigent_patched_stream"),
            ("astream", "_traigent_patched_astream"),
        ]:
            if hasattr(ChatOpenAI, meth_name) and not getattr(ChatOpenAI, flag, False):
                original_meth = getattr(ChatOpenAI, meth_name)
                if meth_name == "stream":
                    setattr(
                        ChatOpenAI, meth_name, _create_stream_wrapper(original_meth)
                    )
                else:
                    setattr(
                        ChatOpenAI, meth_name, _create_astream_wrapper(original_meth)
                    )
                setattr(ChatOpenAI, flag, True)
                logger.info(f"✅ Patched ChatOpenAI.{meth_name} for metadata capture")

    except ImportError:
        logger.debug("ChatOpenAI not available, skipping")
    except Exception as e:
        logger.error(f"Failed to patch ChatOpenAI: {e}")

    # Patch ChatBedrock / ChatBedrockConverse
    try:
        import langchain_aws

        bedrock_classes: list[tuple[str, Any]] = []
        for name in ("ChatBedrock", "ChatBedrockConverse"):
            chat_cls = getattr(langchain_aws, name, None)
            if chat_cls is not None:
                bedrock_classes.append((name, cast(Any, chat_cls)))

        if not bedrock_classes:
            logger.debug("LangChain AWS Bedrock chat models not available, skipping")

        for class_name, chat_cls in bedrock_classes:
            if getattr(chat_cls, "_traigent_patched_invoke", False) is False:
                original_invoke_bedrock = chat_cls.invoke
                chat_cls.invoke = _create_bedrock_invoke_wrapper(
                    original_invoke_bedrock
                )
                chat_cls._traigent_patched_invoke = True
                logger.info(f"✅ Patched {class_name}.invoke for metadata capture")
                patched_any = True

            for meth_name, flag in [
                ("stream", "_traigent_patched_stream"),
                ("astream", "_traigent_patched_astream"),
            ]:
                if hasattr(chat_cls, meth_name) and not getattr(chat_cls, flag, False):
                    original_meth = getattr(chat_cls, meth_name)
                    if meth_name == "stream":
                        setattr(
                            chat_cls,
                            meth_name,
                            _create_stream_wrapper(original_meth),
                        )
                    else:
                        setattr(
                            chat_cls,
                            meth_name,
                            _create_astream_wrapper(original_meth),
                        )
                    setattr(chat_cls, flag, True)
                    logger.info(
                        f"✅ Patched {class_name}.{meth_name} for metadata capture"
                    )

    except ImportError:
        logger.debug("langchain_aws not available, skipping Bedrock chat models")
    except Exception as e:
        logger.error(f"Failed to patch LangChain AWS Bedrock models: {e}")

    if not patched_any:
        logger.debug("No LangChain models could be patched for metadata capture")
        return False

    return True
