"""HuggingFace platform executor implementation.

This module provides integration with HuggingFace models, supporting
both the Inference API (cloud) and Edge Analytics model execution via transformers.
"""

# Traceability: CONC-Layer-Integration CONC-Quality-Compatibility FUNC-AGENTS FUNC-INTEGRATIONS REQ-AGNT-013 REQ-INT-008 SYNC-OptimizationFlow

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from typing import Any, cast

try:
    from huggingface_hub import AsyncInferenceClient

    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False

try:
    import torch
    from transformers import pipeline

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

from traigent.agents.platforms.base_platform import BasePlatformExecutor
from traigent.agents.platforms.parameter_mapping import ParameterMapper
from traigent.utils.exceptions import (
    AgentExecutionError,
    PlatformCapabilityError,
)
from traigent.utils.logging import get_logger

logger = get_logger(__name__)


class HuggingFaceAgentExecutor(BasePlatformExecutor):
    """Executor for HuggingFace models."""

    # Popular models available on HuggingFace
    SUPPORTED_MODELS = [
        # Open source LLMs
        "meta-llama/Llama-2-7b-chat-hf",
        "meta-llama/Llama-2-13b-chat-hf",
        "meta-llama/Llama-2-70b-chat-hf",
        "mistralai/Mistral-7B-Instruct-v0.2",
        "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "google/flan-t5-base",
        "google/flan-t5-large",
        "google/flan-t5-xl",
        "google/flan-t5-xxl",
        "EleutherAI/gpt-neo-2.7B",
        "EleutherAI/gpt-j-6B",
        "bigscience/bloom-560m",
        "bigscience/bloom-1b7",
        "bigscience/bloom-3b",
        "tiiuae/falcon-7b-instruct",
        "tiiuae/falcon-40b-instruct",
        # Allow any model from HF hub
        "*",  # Wildcard to accept any model
    ]

    def __init__(
        self,
        api_key: str | None = None,
        use_local: bool = False,
        device: str | None = None,
        **kwargs,
    ) -> None:
        """Initialize HuggingFace executor.

        Args:
            api_key: HuggingFace API key (or use HUGGINGFACE_API_KEY env var)
            use_local: Whether to use Edge Analytics model execution
            device: Device for local execution (cuda, cpu, mps)
            **kwargs: Additional configuration
        """
        super().__init__(**kwargs)

        self.use_local = use_local
        self.device = device
        self.local_pipeline = None
        self.model_id: str | None = None

        if use_local:
            if not TRANSFORMERS_AVAILABLE:
                raise ImportError(
                    "transformers package not installed. "
                    "Install with: pip install transformers torch"
                )
            # Device setup for local execution
            if device is None:
                if torch.cuda.is_available():
                    self.device = "cuda"
                elif torch.backends.mps.is_available():
                    self.device = "mps"
                else:
                    self.device = "cpu"
            logger.info(f"Using device: {self.device}")
        else:
            if not HF_HUB_AVAILABLE:
                raise ImportError(
                    "huggingface_hub package not installed. "
                    "Install with: pip install huggingface_hub"
                )

            # Get API key for Inference API
            self.api_key = api_key or os.getenv("HUGGINGFACE_API_KEY")
            if not self.api_key:
                logger.warning(
                    "No HuggingFace API key found. "
                    "Some models may have rate limits without authentication."
                )

            # Initialize async client
            self.client = AsyncInferenceClient(token=self.api_key)

        self.mapper = ParameterMapper("huggingface")

        logger.info(f"Initialized HuggingFace executor (local={use_local})")

    # ===== Core Implementation =====

    async def complete(self, prompt: str, **kwargs) -> str:
        """Generate completion using HuggingFace.

        Args:
            prompt: The input prompt
            **kwargs: Additional parameters (unified + hf-specific)

        Returns:
            The completion response
        """
        try:
            # Extract and translate parameters
            unified_params = self._extract_unified_params(kwargs)
            hf_kwargs = kwargs.get("huggingface_kwargs", {})

            # Translate to HuggingFace format
            params = self.mapper.to_platform_params(unified_params, hf_kwargs)

            # Get model ID
            model_id = params.pop("model_id", "mistralai/Mistral-7B-Instruct-v0.2")

            if self.use_local:
                return await self._complete_local(prompt, model_id, params)
            else:
                return await self._complete_api(prompt, model_id, params)

        except Exception as e:
            logger.error(f"HuggingFace completion failed: {e}")
            raise AgentExecutionError(f"HuggingFace completion failed: {e}") from None

    async def _complete_api(
        self, prompt: str, model_id: str, params: dict[str, Any]
    ) -> str:
        """Complete using HuggingFace Inference API."""
        # Prepare parameters
        api_params: dict[str, Any] = {"inputs": prompt, "parameters": {}}

        # Map parameters
        param_mapping = {
            "max_new_tokens": "max_new_tokens",
            "temperature": "temperature",
            "top_p": "top_p",
            "top_k": "top_k",
            "stop": "stop_sequences",
            "seed": "seed",
            "do_sample": "do_sample",
            "repetition_penalty": "repetition_penalty",
        }

        for hf_key, api_key in param_mapping.items():
            if hf_key in params:
                api_params["parameters"][api_key] = params[hf_key]

        # Default do_sample if temperature is set
        if (
            "temperature" in api_params["parameters"]
            and "do_sample" not in api_params["parameters"]
        ):
            api_params["parameters"]["do_sample"] = True

        # Make API call
        response = await self.client.text_generation(
            prompt, model=model_id, **api_params["parameters"]
        )

        return cast(str, response)

    async def _complete_local(
        self, prompt: str, model_id: str, params: dict[str, Any]
    ) -> str:
        """Complete using local transformers pipeline."""
        # Load model if needed
        if self.local_pipeline is None or self.model_id != model_id:
            logger.info(f"Loading model: {model_id}")
            self.local_pipeline = pipeline(
                "text-generation",
                model=model_id,
                device=self.device,
                torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
            )
            self.model_id = model_id

        # Generate
        assert self.local_pipeline is not None  # Ensured by the above if block
        result = self.local_pipeline(prompt, **params)

        # Extract generated text
        return cast(str, result[0]["generated_text"][len(prompt) :])

    async def stream(self, prompt: str, **kwargs) -> AsyncIterator[str]:
        """Stream completion using HuggingFace.

        Note: Streaming is only supported with Inference API, not local execution.

        Args:
            prompt: The input prompt
            **kwargs: Additional parameters

        Yields:
            Completion chunks as they arrive
        """
        if self.use_local:
            raise PlatformCapabilityError(
                "HuggingFace local execution does not support streaming. "
                "Use use_local=False for streaming support."
            )

        try:
            # Extract and translate parameters
            unified_params = self._extract_unified_params(kwargs)
            hf_kwargs = kwargs.get("huggingface_kwargs", {})

            # Translate to HuggingFace format
            params = self.mapper.to_platform_params(unified_params, hf_kwargs)

            # Get model ID
            model_id = params.pop("model_id", "mistralai/Mistral-7B-Instruct-v0.2")

            # Prepare streaming parameters
            stream_params = {}
            param_mapping = {
                "max_new_tokens": "max_new_tokens",
                "temperature": "temperature",
                "top_p": "top_p",
                "top_k": "top_k",
                "stop": "stop_sequences",
                "seed": "seed",
                "do_sample": "do_sample",
            }

            for hf_key, api_key in param_mapping.items():
                if hf_key in params:
                    stream_params[api_key] = params[hf_key]

            # Stream from API
            async for token in self.client.text_generation(
                prompt, model=model_id, stream=True, **stream_params
            ):
                yield token

        except Exception as e:
            logger.error(f"HuggingFace streaming failed: {e}")
            raise AgentExecutionError(f"HuggingFace streaming failed: {e}") from None

    # ===== Capability Support =====

    def supports_streaming(self) -> bool:
        """HuggingFace supports streaming via Inference API."""
        return not self.use_local

    def supports_batch(self) -> bool:
        """HuggingFace supports batch via local pipeline."""
        return self.use_local

    def supports_tools(self) -> bool:
        """HuggingFace models don't natively support tools."""
        return False

    async def batch_complete(self, prompts: list[str], **kwargs) -> list[str]:
        """Batch completion using HuggingFace.

        Only supported in Edge Analytics mode.

        Args:
            prompts: List of input prompts
            **kwargs: Additional parameters

        Returns:
            List of completion responses
        """
        if not self.use_local:
            raise PlatformCapabilityError(
                "HuggingFace Inference API does not support batch execution. "
                "Use use_local=True for batch support."
            )

        try:
            # Extract and translate parameters
            unified_params = self._extract_unified_params(kwargs)
            hf_kwargs = kwargs.get("huggingface_kwargs", {})

            # Translate to HuggingFace format
            params = self.mapper.to_platform_params(unified_params, hf_kwargs)

            # Get model ID
            model_id = params.pop("model_id", "mistralai/Mistral-7B-Instruct-v0.2")

            # Load model if needed
            if self.local_pipeline is None or self.model_id != model_id:
                logger.info(f"Loading model: {model_id}")
                self.local_pipeline = pipeline(
                    "text-generation",
                    model=model_id,
                    device=self.device,
                    torch_dtype=(
                        torch.float16 if self.device != "cpu" else torch.float32
                    ),
                )
                self.model_id = model_id

            # Batch generate
            assert self.local_pipeline is not None  # Ensured by the above if block
            results = self.local_pipeline(prompts, **params)

            # Extract generated texts
            completions = []
            for i, result in enumerate(results):
                generated = result[0]["generated_text"][len(prompts[i]) :]
                completions.append(generated)

            return completions

        except Exception as e:
            logger.error(f"HuggingFace batch completion failed: {e}")
            raise AgentExecutionError(
                f"HuggingFace batch completion failed: {e}"
            ) from None

    # ===== Model Support =====

    def get_supported_models(self) -> list[str]:
        """Get list of supported HuggingFace models."""
        # Return common models, but note that any model is supported
        return [m for m in self.SUPPORTED_MODELS if m != "*"]

    def validate_model(self, model: str) -> bool:
        """Validate if a model is supported.

        For HuggingFace, we accept any model ID as they have thousands of models.
        """
        # Accept any model that looks like a valid HF model ID
        return "/" in model or model in self.SUPPORTED_MODELS

    # ===== Cost Estimation =====

    def estimate_cost(self, input_tokens: int, output_tokens: int, model: str) -> float:
        """Estimate cost for HuggingFace completion.

        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            model: Model identifier

        Returns:
            Estimated cost in USD
        """
        if self.use_local:
            # Local execution has no API costs
            return 0.0
        else:
            # Inference API pricing varies by model
            # Using approximate pricing for common models
            if "llama-2-70b" in model.lower():
                # Larger models cost more
                return (input_tokens + output_tokens) / 1000 * 0.001
            elif "llama-2-13b" in model.lower() or "mixtral" in model.lower():
                return (input_tokens + output_tokens) / 1000 * 0.0005
            else:
                # Smaller models or default pricing
                return (input_tokens + output_tokens) / 1000 * 0.0002

    # ===== Parameter Handling =====

    def _extract_unified_params(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Extract unified parameters from kwargs.

        Args:
            kwargs: All parameters passed

        Returns:
            Dictionary of unified parameters
        """
        unified = {}

        # List of known unified parameters
        unified_keys = [
            "model",
            "temperature",
            "max_tokens",
            "top_p",
            "top_k",
            "stop_sequences",
            "seed",
            "stream",
        ]

        for key in unified_keys:
            if key in kwargs:
                unified[key] = kwargs[key]

        return unified

    async def _cleanup(self) -> None:
        """Cleanup resources."""
        if self.local_pipeline is not None:
            # Clear model from memory
            del self.local_pipeline
            self.local_pipeline = None
            self.model_id = None

            # Clear GPU cache if using CUDA
            if self.device == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()
