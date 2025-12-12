"""⚠️  EXPERIMENTAL: Simple base platform executor - NOT FOR PRODUCTION.

🚨 WARNING: This is a NAIVE, simplified implementation for local testing only.
This is NOT the real TraiGent cloud implementation and does NOT represent
TraiGent's proprietary IP.

This module provides a basic base class for experimental platform executors
used during development and testing while the OptiGen backend is being built.

Real TraiGent cloud execution happens in the OptiGen backend (proprietary).
"""

# Traceability: CONC-Layer-Integration CONC-Quality-Compatibility FUNC-AGENTS FUNC-INTEGRATIONS REQ-AGNT-013 REQ-INT-008 SYNC-OptimizationFlow

from __future__ import annotations

from abc import abstractmethod
from collections.abc import AsyncIterator
from typing import Any

from traigent.agents.executor import (
    AgentExecutor,
    CostEstimate,
    PlatformConfigValidationResult,
)
from traigent.cloud.models import AgentSpecification
from traigent.utils.exceptions import AgentExecutionError, PlatformCapabilityError
from traigent.utils.logging import get_logger

logger = get_logger(__name__)


class BasePlatformExecutor(AgentExecutor):
    """⚠️  EXPERIMENTAL: Simple base class for naive platform testing.

    🚨 WARNING: This is NOT the real TraiGent cloud architecture!

    This is a simplified base class for local testing and development.
    Real platform execution happens in the OptiGen backend.

    All experimental executors should inherit from this class.
    """

    def __init__(
        self, platform_config: dict[str, Any] | None = None, **kwargs: Any
    ) -> None:
        """Initialize the platform executor.

        Args:
            platform_config: Optional explicit platform configuration.
            **kwargs: Additional platform-specific configuration (merged into platform_config).
        """
        merged_config = dict(platform_config or {})
        merged_config.update(kwargs)
        super().__init__(platform_config=merged_config)
        self._platform_name = self.__class__.__name__.replace("Executor", "")

    async def _platform_initialize(self) -> None:
        """Experimental executors have no shared async initialization."""
        return None

    @staticmethod
    def _normalize_platform_id(value: str) -> str:
        return value.lower().replace("-", "").replace("_", "")

    @staticmethod
    def _format_prompt(template: str | None, input_data: dict[str, Any]) -> str:
        template_str = template or ""

        class _SafeDict(dict[str, str]):
            def __missing__(self, key: str) -> str:
                return "{" + key + "}"

        try:
            return template_str.format_map(
                _SafeDict({k: str(v) for k, v in input_data.items()})
            )
        except Exception:
            prompt = template_str
            for key, value in input_data.items():
                prompt = prompt.replace(f"{{{key}}}", str(value))
            return prompt

    async def _execute_agent(
        self,
        agent_spec: AgentSpecification,
        input_data: dict[str, Any],
        config: dict[str, Any],
    ) -> dict[str, Any]:
        """Bridge AgentExecutor contract to experimental executor API."""
        prompt = self._format_prompt(agent_spec.prompt_template, input_data)

        model = str(config.get("model", "")) if config.get("model") is not None else ""
        max_tokens = int(config.get("max_tokens", 1000) or 1000)
        prompt_tokens = int(len(prompt.split()) * 1.3)
        completion_tokens = max(1, int(max_tokens * 0.7))
        tokens_used = prompt_tokens + completion_tokens

        output: Any
        try:
            if agent_spec.custom_tools and self.supports_tools():
                tools = [
                    {"name": name, "description": "", "parameters": {}}
                    for name in agent_spec.custom_tools
                ]
                output = await self.complete_with_tools(prompt, tools, **config)
            else:
                output = await self.complete(prompt, **config)
        except Exception as exc:
            raise AgentExecutionError(
                f"{self._platform_name} execution failed: {exc}"
            ) from exc

        cost = self.estimate_completion_cost(prompt_tokens, completion_tokens, model)

        return {
            "output": output,
            "tokens_used": tokens_used,
            "cost": cost,
            "metadata": {"platform": self._platform_name, "model": model},
        }

    def _validate_platform_spec(self, agent_spec: AgentSpecification) -> None:
        """Best-effort validation for experimental executors."""
        if not agent_spec.agent_platform:
            return

        actual = self._normalize_platform_id(str(agent_spec.agent_platform))
        if actual == "test":
            return

        expected = self._normalize_platform_id(self._platform_name.replace("Agent", ""))
        if actual != expected:
            raise ValueError(f"Invalid platform: {agent_spec.agent_platform}") from None

    def _get_platform_capabilities(self) -> list[str]:
        capabilities = ["complete"]
        if self.supports_streaming():
            capabilities.append("streaming")
        if self.supports_batch():
            capabilities.append("batch")
        if self.supports_tools():
            capabilities.append("tools")
        return capabilities

    async def _validate_platform_config(
        self, config: dict[str, Any]
    ) -> PlatformConfigValidationResult:
        errors: list[str] = []
        warnings: list[str] = []

        model = config.get("model")
        if model is not None and not isinstance(model, str):
            errors.append("model must be a string")
        elif isinstance(model, str) and model and not self.validate_model(model):
            warnings.append(f"Model '{model}' may not be supported")

        return {"valid": len(errors) == 0, "errors": errors, "warnings": warnings}

    async def estimate_cost(
        self,
        agent_spec: AgentSpecification,
        input_data: dict[str, Any],
        config_overrides: dict[str, Any] | None = None,
    ) -> CostEstimate:
        """Estimate execution cost using executor pricing tables when available."""
        config = self._merge_configurations(
            agent_spec.model_parameters or {}, config_overrides
        )
        model = str(config.get("model", "")) if config.get("model") is not None else ""

        prompt = self._format_prompt(agent_spec.prompt_template, input_data)
        prompt_tokens = int(len(prompt.split()) * 1.3)
        max_tokens = int(config.get("max_tokens", 1000) or 1000)
        completion_tokens = max(1, int(max_tokens * 0.7))

        total_tokens = prompt_tokens + completion_tokens
        cost = self.estimate_completion_cost(prompt_tokens, completion_tokens, model)

        return {
            "estimated_cost": float(cost),
            "estimated_tokens": int(total_tokens),
            "confidence": 0.2,
        }

    # ===== Core Capabilities (All platforms must implement) =====

    @abstractmethod
    async def complete(self, prompt: str, **kwargs) -> str:
        """Standard completion - all platforms must support this.

        Args:
            prompt: The input prompt
            **kwargs: Platform-specific parameters

        Returns:
            The completion response
        """
        raise NotImplementedError("Platform must implement complete()")

    # ===== Optional Capabilities (Platforms may support) =====

    def stream(self, prompt: str, **kwargs) -> AsyncIterator[str]:
        """Streaming completion - raise exception if not supported.

        Args:
            prompt: The input prompt
            **kwargs: Platform-specific parameters

        Yields:
            Completion chunks as they arrive

        Raises:
            PlatformCapabilityError: If streaming is not supported
        """
        if not self.supports_streaming():
            raise PlatformCapabilityError(
                f"{self._platform_name} does not support streaming completions"
            )
        # Platforms that support streaming should override this method
        raise NotImplementedError(f"{self._platform_name} must implement stream()")

    async def batch_complete(self, prompts: list[str], **kwargs) -> list[str]:
        """Batch completion - raise exception if not supported.

        Args:
            prompts: List of input prompts
            **kwargs: Platform-specific parameters

        Returns:
            List of completion responses

        Raises:
            PlatformCapabilityError: If batch execution is not supported
        """
        if not self.supports_batch():
            raise PlatformCapabilityError(
                f"{self._platform_name} does not support batch execution"
            )
        # Default implementation: sequential execution
        # Platforms can override for true batch support
        results = []
        for prompt in prompts:
            result = await self.complete(prompt, **kwargs)
            results.append(result)
        return results

    async def complete_with_tools(
        self, prompt: str, tools: list[dict[str, Any]], **kwargs
    ) -> dict[str, Any]:
        """Completion with tool/function calling - raise exception if not supported.

        Args:
            prompt: The input prompt
            tools: List of tool definitions
            **kwargs: Platform-specific parameters

        Returns:
            Response including potential tool calls

        Raises:
            PlatformCapabilityError: If tools are not supported
        """
        if not self.supports_tools():
            raise PlatformCapabilityError(
                f"{self._platform_name} does not support tool/function calling"
            )
        # Platforms that support tools should override this method
        raise NotImplementedError(
            f"{self._platform_name} must implement complete_with_tools()"
        )

    # ===== Capability Detection =====

    def supports_streaming(self) -> bool:
        """Check if platform supports streaming.

        Returns:
            True if streaming is supported
        """
        return False  # Override in platforms that support it

    def supports_batch(self) -> bool:
        """Check if platform supports batch execution.

        Returns:
            True if batch execution is supported
        """
        return False  # Override in platforms that support it

    def supports_tools(self) -> bool:
        """Check if platform supports tool/function calling.

        Returns:
            True if tools are supported
        """
        return False  # Override in platforms that support it

    def get_supported_models(self) -> list[str]:
        """Get list of supported models for this platform.

        Returns:
            List of model identifiers
        """
        return []  # Override in platform implementations

    # ===== Parameter Handling =====

    def translate_parameters(
        self,
        unified_params: dict[str, Any],
        platform_kwargs: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Translate unified parameters to platform-specific format.

        Args:
            unified_params: Parameters in TraiGent's unified format
            platform_kwargs: Additional platform-specific parameters

        Returns:
            Platform-specific parameter dictionary
        """
        # Base implementation: pass through
        # Platform implementations should override this
        result = unified_params.copy()
        if platform_kwargs:
            result.update(platform_kwargs)
        return result

    # ===== Cost Estimation =====

    def estimate_completion_cost(
        self, input_tokens: int, output_tokens: int, model: str
    ) -> float:
        """Estimate cost for a completion.

        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            model: Model identifier

        Returns:
            Estimated cost in USD
        """
        # Override in platforms with known pricing
        return 0.0

    # ===== Validation =====

    def validate_model(self, model: str) -> bool:
        """Validate if a model is supported.

        Args:
            model: Model identifier

        Returns:
            True if model is supported
        """
        return model in self.get_supported_models()

    def validate_parameters(self, params: dict[str, Any]) -> dict[str, Any]:
        """Validate and clean parameters.

        Args:
            params: Parameters to validate

        Returns:
            Validated parameters

        Raises:
            ValueError: If parameters are invalid
        """
        # Base implementation: pass through
        # Platform implementations should add validation
        return params
