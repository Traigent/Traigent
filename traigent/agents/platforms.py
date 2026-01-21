"""Platform-specific agent executors.

This module provides concrete implementations of agent executors
for different AI platforms like LangChain, OpenAI, etc.
Integrated with unified authentication system for secure credential management.
"""

# Traceability: CONC-Layer-Core CONC-Quality-Reliability FUNC-AGENTS REQ-AGNT-013

from __future__ import annotations

import importlib.util
from typing import Any, cast

from traigent.agents.executor import (
    AgentExecutor,
    CostEstimate,
    PlatformConfigValidationResult,
)
from traigent.cloud.auth import (
    AuthCredentials,
    AuthMode,
    get_auth_manager,
)
from traigent.cloud.models import AgentSpecification
from traigent.utils.exceptions import AgentExecutionError
from traigent.utils.logging import get_logger
from traigent.utils.validation import CoreValidators, validate_or_raise

logger = get_logger(__name__)

# Common model alias fixes
MODEL_ALIASES: dict[str, str] = {
    "o4-mini": "gpt-4o-mini",
}


class LangChainAgentExecutor(AgentExecutor):
    """Executor for LangChain-based agents."""

    async def _platform_initialize(self) -> None:
        """Initialize LangChain components."""
        # Check if langchain is available (legacy or modern split packages)
        self._langchain_available = (
            importlib.util.find_spec("langchain") is not None
            or importlib.util.find_spec("langchain_openai") is not None
        )
        if self._langchain_available:
            logger.info("LangChain initialized successfully")
        else:
            logger.warning(
                "LangChain not available - install with: pip install langchain langchain-openai"
            )

    async def _execute_agent(
        self,
        agent_spec: AgentSpecification,
        input_data: dict[str, Any],
        config: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute LangChain agent."""
        if not self._langchain_available:
            raise AgentExecutionError("LangChain not available")

        chat_openai_cls, human_message_cls, modern_lc = (
            self._import_langchain_components()
        )
        llm, model_name = self._build_langchain_llm(chat_openai_cls, modern_lc, config)

        prompt_text = self._format_prompt(agent_spec.prompt_template, input_data)

        if agent_spec.agent_type == "conversational":
            return await self._execute_langchain_conversation(
                llm, human_message_cls, model_name, prompt_text
            )
        elif agent_spec.agent_type == "task":
            if agent_spec.custom_tools:
                return await self._execute_with_tools(
                    llm, prompt_text, agent_spec.custom_tools
                )
            return await self._execute_langchain_task(
                llm, human_message_cls, prompt_text
            )
        else:
            raise ValueError(f"Unsupported agent type: {agent_spec.agent_type}")

    def _import_langchain_components(self):
        """Import LC components with modern-first strategy."""
        try:
            from langchain_openai import ChatOpenAI

            modern_lc = True
        except ImportError:
            from langchain.chat_models import ChatOpenAI

            modern_lc = False
        try:
            from langchain_core.messages import HumanMessage
        except ImportError:
            from langchain.schema import HumanMessage
        return ChatOpenAI, HumanMessage, modern_lc

    def _build_langchain_llm(
        self, chat_openai_cls, modern_lc: bool, config: dict[str, Any]
    ):
        model_name = config.get("model", "gpt-4o-mini")
        temperature = config.get("temperature", 0.7)
        max_tokens = int(config.get("max_tokens", 1000))
        llm_kwargs = {
            ("model" if modern_lc else "model_name"): model_name,
            "temperature": temperature,
            "model_kwargs": {"max_tokens": max_tokens},
            **self._extract_llm_kwargs(config),
        }
        llm = chat_openai_cls(**llm_kwargs)
        return llm, model_name

    async def _execute_langchain_conversation(
        self, llm, human_message_cls, model_name: str, prompt_text: str
    ) -> dict[str, Any]:
        messages = [human_message_cls(content=prompt_text)]
        if hasattr(llm, "ainvoke"):
            ai_message = await llm.ainvoke(messages)
            output = getattr(ai_message, "content", None) or str(ai_message)
            usage = getattr(ai_message, "usage_metadata", None)
            meta: dict[str, Any] = {"model": model_name}
            if usage and isinstance(usage, dict):
                # Surface LangChain usage metadata for downstream metrics capture
                meta["usage_metadata"] = usage
            # Some LC variants expose response_metadata as well
            resp_meta = getattr(ai_message, "response_metadata", None)
            if resp_meta and isinstance(resp_meta, dict):
                meta["response_metadata"] = resp_meta
            tokens_used = None
            if isinstance(usage, dict):
                tokens_used = usage.get("total_tokens")
            return {"output": output, "tokens_used": tokens_used, "metadata": meta}
        response = await llm.agenerate([messages])
        output = response.generations[0][0].text
        tokens_used = None
        if hasattr(response, "llm_output") and response.llm_output:
            tokens_used = response.llm_output.get("token_usage", {}).get("total_tokens")
        return {
            "output": output,
            "tokens_used": tokens_used,
            "metadata": {
                "model": model_name,
                "prompt_tokens": (
                    response.llm_output.get("token_usage", {}).get("prompt_tokens")
                    if hasattr(response, "llm_output") and response.llm_output
                    else None
                ),
                "completion_tokens": (
                    response.llm_output.get("token_usage", {}).get("completion_tokens")
                    if hasattr(response, "llm_output") and response.llm_output
                    else None
                ),
            },
        }

    async def _execute_langchain_task(
        self, llm, human_message_cls, prompt_text: str
    ) -> dict[str, Any]:
        if hasattr(llm, "ainvoke"):
            ai_message = await llm.ainvoke([human_message_cls(content=prompt_text)])
            meta: dict[str, Any] = {}
            usage = getattr(ai_message, "usage_metadata", None)
            if usage and isinstance(usage, dict):
                meta["usage_metadata"] = usage
            resp_meta = getattr(ai_message, "response_metadata", None)
            if resp_meta and isinstance(resp_meta, dict):
                meta["response_metadata"] = resp_meta
            tokens_used = usage.get("total_tokens") if isinstance(usage, dict) else None
            return {
                "output": getattr(ai_message, "content", None) or str(ai_message),
                "tokens_used": tokens_used,
                "metadata": meta,
            }
        response = await llm.agenerate([[human_message_cls(content=prompt_text)]])
        return {
            "output": response.generations[0][0].text,
            "tokens_used": (
                response.llm_output.get("token_usage", {}).get("total_tokens")
                if hasattr(response, "llm_output") and response.llm_output
                else None
            ),
        }

    async def _execute_with_tools(
        self, llm: Any, prompt: str, tools: list[str]
    ) -> dict[str, Any]:
        """Execute agent with tools."""
        # Simplified tool execution - in production, integrate actual tools
        logger.info(f"Executing with tools: {tools}")

        # For now, just execute without tools
        try:
            from langchain_core.messages import HumanMessage
        except Exception:
            from langchain.schema import HumanMessage

        if hasattr(llm, "ainvoke"):
            ai_message = await llm.ainvoke([HumanMessage(content=prompt)])
            return {
                "output": getattr(ai_message, "content", None) or str(ai_message),
                "tokens_used": None,
                "metadata": {"tools_available": tools},
            }
        else:
            response = await llm.agenerate([[HumanMessage(content=prompt)]])
            return {
                "output": response.generations[0][0].text,
                "tokens_used": (
                    response.llm_output.get("token_usage", {}).get("total_tokens")
                    if hasattr(response, "llm_output") and response.llm_output
                    else None
                ),
                "metadata": {"tools_available": tools},
            }

    def _validate_platform_spec(self, agent_spec: AgentSpecification) -> None:
        """Validate LangChain-specific requirements."""
        if agent_spec.agent_platform != "langchain":
            raise ValueError(f"Invalid platform: {agent_spec.agent_platform}") from None

        # Check for required model parameters
        if (
            agent_spec.model_parameters is None
            or "model" not in agent_spec.model_parameters
        ):
            raise ValueError("LangChain agents require 'model' parameter")

    async def _validate_platform_config(
        self, config: dict[str, Any]
    ) -> PlatformConfigValidationResult:
        """Validate LangChain configuration using unified validators."""
        from traigent.utils.validation import ValidationResult

        overall_result = ValidationResult()

        # Validate model
        valid_models = [
            "gpt-4o-mini",
            "GPT-4o",
            "gpt-4-turbo",
            "claude-2",
            "claude-instant",
        ]
        model = config.get("model")
        if model:
            model_result = CoreValidators.validate_choices(model, "model", valid_models)
            if not model_result.is_valid:
                # Convert to warning instead of error for unsupported models
                overall_result.add_warning(
                    "model", f"Model '{model}' may not be supported"
                )

        # Validate temperature
        temp = config.get("temperature", 0.7)
        temp_result = CoreValidators.validate_number(temp, "temperature", 0.0, 2.0)
        if not temp_result.is_valid:
            for error in temp_result.errors:
                overall_result.add_error(error.field, error.message)

        # Validate max_tokens
        max_tokens = config.get("max_tokens", 1000)
        tokens_result = CoreValidators.validate_positive_int(max_tokens, "max_tokens")
        if not tokens_result.is_valid:
            for error in tokens_result.errors:
                overall_result.add_error(error.field, error.message)

        return {
            "valid": overall_result.is_valid,
            "errors": [error.message for error in overall_result.errors],
            "warnings": [warning.message for warning in overall_result.warnings],
        }

    def _get_platform_capabilities(self) -> list[str]:
        """Get LangChain capabilities."""
        return ["conversational", "task", "tools", "memory", "streaming", "async"]

    def _extract_llm_kwargs(self, config: dict[str, Any]) -> dict[str, Any]:
        """Extract additional LLM kwargs from config with validation."""
        llm_params = {}

        # Extract and validate known parameters
        for param in [
            "top_p",
            "frequency_penalty",
            "presence_penalty",
            "stop_sequences",
        ]:
            if param in config:
                value = config[param]

                # Validate numeric parameters
                if param in ["top_p", "frequency_penalty", "presence_penalty"]:
                    try:
                        validate_or_raise(
                            CoreValidators.validate_number(value, param, 0.0, 2.0)
                        )
                    except ValueError as e:
                        logger.warning(f"Invalid {param} value: {e}")
                        continue

                llm_params[param] = value

        return llm_params

    def _format_prompt(self, template: str | None, input_data: dict[str, Any]) -> str:
        """Format prompt template with input data."""
        template_str = template or ""
        # Safer template formatting that leaves unknown placeholders intact

        class _SafeDict(dict[str, str]):
            def __missing__(self, key: str) -> str:
                return "{" + key + "}"

        try:
            return template_str.format_map(
                _SafeDict({k: str(v) for k, v in input_data.items()})
            )
        except Exception:
            # Fallback to naive replacement if formatting fails
            logger.debug(
                "format_map failed, using naive replacement for prompt template"
            )
            prompt = template_str
            for key, value in input_data.items():
                prompt = prompt.replace(f"{{{key}}}", str(value))
            return prompt


class OpenAIAgentExecutor(AgentExecutor):
    """Executor for OpenAI SDK-based agents."""

    def __init__(self, platform_config: dict[str, Any] | None = None) -> None:
        super().__init__(platform_config=platform_config)
        self._openai_available: bool = False
        self._openai_use_v1: bool = False
        self._openai_client: Any | None = None
        self.auth_manager: Any | None = None

    async def _platform_cleanup(self) -> None:
        """Cleanup OpenAI client and auth resources."""
        # Close OpenAI client if it has a close method
        if self._openai_client is not None:
            if hasattr(self._openai_client, "close"):
                try:
                    await self._openai_client.close()
                    logger.debug("OpenAI client closed successfully")
                except Exception as e:
                    logger.warning(f"Error closing OpenAI client: {e}")

            self._openai_client = None

        # Clear auth manager reference
        self.auth_manager = None

        # Reset availability flags
        self._openai_available = False
        self._openai_use_v1 = False

    async def _platform_initialize(self) -> None:
        """Initialize OpenAI components with unified auth integration."""
        try:
            # Initialize unified auth manager
            self.auth_manager = get_auth_manager()

            # Prepare default headers via unified auth (may be empty)
            default_headers = await self._get_authenticated_headers()

            # Prefer modern OpenAI SDK v1 client
            try:
                from openai import AsyncOpenAI

                api_key = await self._get_platform_api_key("OPENAI_API_KEY")
                self._openai_client = AsyncOpenAI(
                    api_key=api_key, default_headers=default_headers
                )
                self._openai_use_v1 = True
                self._openai_available = True
                logger.info("OpenAI SDK initialized successfully")
                return
            except Exception:
                # Fallback to legacy SDK usage
                logger.debug(
                    "AsyncOpenAI initialization failed, falling back to legacy SDK"
                )
                import openai

                api_key = await self._get_platform_api_key("OPENAI_API_KEY")
                if api_key:
                    openai.api_key = api_key
                    logger.info("OpenAI SDK initialized with authenticated API key")
                else:
                    logger.warning(
                        "OpenAI SDK initialized without API key - authentication may fail"
                    )
                self._openai_client = None
                self._openai_use_v1 = False
                self._openai_available = True
                logger.info("OpenAI SDK initialized successfully")
        except ImportError:
            self._openai_available = False
            self._openai_use_v1 = False
            self._openai_client = None
            logger.warning(
                "OpenAI SDK not available - install with: pip install openai"
            )

    async def _get_platform_api_key(self, env_var_name: str) -> str | None:
        """Get API key through unified auth system."""
        try:
            # First check platform config for direct key
            if "api_key" in self.platform_config:
                api_key = self.platform_config["api_key"]

                # Validate key format
                validate_or_raise(
                    CoreValidators.validate_string(api_key, "api_key", min_length=1)
                )

                # Create auth credentials for this API key
                auth_credentials = AuthCredentials(
                    mode=AuthMode.API_KEY,
                    api_key=api_key,
                    metadata={"platform": "openai", "source": "platform_config"},
                )

                # Authenticate to validate and enable rate limiting
                if self.auth_manager is None:
                    return cast(str | None, api_key)

                auth_result = await self.auth_manager.authenticate(auth_credentials)

                if auth_result.success:
                    return cast(str | None, api_key)
                logger.warning(
                    f"API key validation failed: {auth_result.error_message}"
                )

            # Fallback to environment variable through unified auth
            import os

            env_api_key = os.getenv(env_var_name)
            if env_api_key:
                validate_or_raise(
                    CoreValidators.validate_string(
                        env_api_key, "env_api_key", min_length=1
                    )
                )

                auth_credentials = AuthCredentials(
                    mode=AuthMode.API_KEY,
                    api_key=env_api_key,
                    metadata={"platform": "openai", "source": "environment"},
                )

                if self.auth_manager is None:
                    return env_api_key

                auth_result = await self.auth_manager.authenticate(auth_credentials)

                if auth_result.success:
                    return env_api_key
                logger.warning(
                    f"Environment API key validation failed: {auth_result.error_message}"
                )

            return None

        except Exception as e:
            logger.error(f"Failed to get platform API key: {e}")
            return None

    async def _execute_agent(
        self,
        agent_spec: AgentSpecification,
        input_data: dict[str, Any],
        config: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute OpenAI agent."""
        if not self._openai_available:
            raise AgentExecutionError("OpenAI SDK not available") from None

        # Extract configuration
        model = str(config.get("model", "gpt-4o-mini"))
        temperature = float(config.get("temperature", 0.7))
        max_tokens = int(config.get("max_tokens", 1000))

        # Format prompt
        prompt = self._format_prompt(agent_spec.prompt_template, input_data)

        # Prepare messages based on agent configuration
        messages = self._prepare_messages(agent_spec, prompt, input_data)

        try:
            # Make API call (prefer v1 client)
            if (
                getattr(self, "_openai_use_v1", False)
                and getattr(self, "_openai_client", None) is not None
            ):
                client = cast(Any, self._openai_client)
                response = await client.chat.completions.create(
                    model=model,
                    messages=cast(Any, messages),
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **self._extract_api_kwargs(config),
                )
            else:
                import openai

                response = await cast(Any, openai).ChatCompletion.acreate(
                    model=model,
                    messages=cast(Any, messages),
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **self._extract_api_kwargs(config),
                )

            # Extract response
            output = response.choices[0].message.content
            tokens_used = response.usage.total_tokens

            # Calculate cost
            cost = self._calculate_cost(model, response.usage)

            return {
                "output": output,
                "tokens_used": tokens_used,
                "cost": cost,
                "metadata": {
                    "model": model,
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "finish_reason": getattr(
                        response.choices[0], "finish_reason", None
                    ),
                },
            }

        except Exception as e:
            # Check if it's an authentication error and log appropriately
            if "authentication" in str(e).lower() or "api_key" in str(e).lower():
                logger.error(f"OpenAI authentication failed: {e}")
                raise AgentExecutionError(f"OpenAI authentication error: {e}") from e
            else:
                raise AgentExecutionError(f"OpenAI API error: {e}") from e

    async def _get_authenticated_headers(self) -> dict[str, str]:
        """Get authenticated headers for OpenAI API calls."""
        try:
            # Get headers from unified auth manager
            if self.auth_manager is None:
                return {}
            headers = await self.auth_manager.get_auth_headers(target="cloud")
            return cast(dict[str, str], headers)
        except Exception as e:
            logger.warning(f"Failed to get authenticated headers: {e}")
            return {}

    def _prepare_messages(
        self,
        agent_spec: AgentSpecification,
        prompt: str,
        input_data: dict[str, Any] | None = None,
    ) -> list[dict[str, str]]:
        """Prepare messages for OpenAI API."""
        messages = []

        # Add system message if persona or guidelines are specified
        if agent_spec.persona or agent_spec.guidelines:
            system_content = ""

            if agent_spec.persona:
                system_content += f"You are {agent_spec.persona}. "

            if agent_spec.guidelines:
                if isinstance(agent_spec.guidelines, list):
                    system_content += "Follow these guidelines: " + "; ".join(
                        agent_spec.guidelines
                    )
                else:
                    system_content += agent_spec.guidelines

            messages.append({"role": "system", "content": system_content.strip()})

        # Add user message
        messages.append({"role": "user", "content": prompt})

        # Optionally append an auxiliary user message from input_data to utilize the parameter
        if isinstance(input_data, dict) and input_data.get("aux_user_message"):
            messages.append(
                {"role": "user", "content": str(input_data["aux_user_message"])}
            )

        return messages

    def _validate_platform_spec(self, agent_spec: AgentSpecification) -> None:
        """Validate OpenAI-specific requirements with unified validation."""
        # Validate platform type
        if agent_spec.agent_platform != "openai":
            raise ValueError(f"Invalid platform: {agent_spec.agent_platform}") from None

        # Validate required parameters using common validators
        if (
            agent_spec.model_parameters is None
            or "model" not in agent_spec.model_parameters
        ):
            raise ValueError("OpenAI agents require 'model' parameter")

        model = agent_spec.model_parameters["model"]
        validate_or_raise(CoreValidators.validate_string(model, "model", min_length=1))

        # Validate model is in supported list (deduplicated)
        supported_models = [
            "gpt-4o-mini",
            "gpt-4o",
            "gpt-4-turbo",
            "GPT-4o",
        ]
        validate_or_raise(
            CoreValidators.validate_choices(model, "model", supported_models)
        )

    async def _validate_platform_config(
        self, config: dict[str, Any]
    ) -> PlatformConfigValidationResult:
        """Validate OpenAI configuration."""
        errors = []
        warnings = []

        # Check model (deduplicated list; unknowns produce warnings only)
        valid_models = {
            "gpt-4o-mini",
            "gpt-4o-mini-16k",
            "GPT-4o",
            "gpt-4-32k",
            "gpt-4-turbo",
            "gpt-4-1106-preview",
        }

        model = config.get("model")
        if model and model not in valid_models:
            warnings.append(f"Model '{model}' may be deprecated or unavailable")

        # Validate parameters
        temp = config.get("temperature", 0.7)
        if not 0 <= temp <= 2:
            errors.append("Temperature must be between 0 and 2")

        top_p = config.get("top_p", 1.0)
        if not 0 <= top_p <= 1:
            errors.append("top_p must be between 0 and 1")

        return {"valid": len(errors) == 0, "errors": errors, "warnings": warnings}

    def _get_platform_capabilities(self) -> list[str]:
        """Get OpenAI capabilities."""
        return [
            "chat",
            "completion",
            "function_calling",
            "streaming",
            "async",
            "vision",
        ]

    def _extract_api_kwargs(self, config: dict[str, Any]) -> dict[str, Any]:
        """Extract additional API kwargs."""
        api_params = {}

        # Known OpenAI parameters
        for param in [
            "top_p",
            "n",
            "stop",
            "presence_penalty",
            "frequency_penalty",
            "logit_bias",
        ]:
            if param in config:
                api_params[param] = config[param]

        return api_params

    def _format_prompt(self, template: str | None, input_data: dict[str, Any]) -> str:
        """Format prompt template with input data."""

        # Safer template formatting that leaves unknown placeholders intact
        class _SafeDict(dict[str, str]):
            def __missing__(self, key: str) -> str:
                return "{" + key + "}"

        try:
            return (template or "").format_map(
                _SafeDict({k: str(v) for k, v in input_data.items()})
            )
        except Exception:
            # Fallback to naive replacement if formatting fails
            logger.debug(
                "format_map failed in OpenAI executor, using naive replacement"
            )
            prompt = template or ""
            for key, value in input_data.items():
                prompt = prompt.replace(f"{{{key}}}", str(value))
            return prompt

    def _calculate_cost(self, model: str, usage: Any) -> float:
        """Calculate cost based on token usage with robust fallbacks."""
        model_mapped = MODEL_ALIASES.get(model, model)
        # Deterministic per-1k token pricing to match current expectations

        per_1k_rates = {
    "gpt-4o-mini": {"prompt": 0.00015, "completion": 0.0006},  #
    "gpt-4o": {"prompt": 0.0025, "completion": 0.01},          #
    "gpt-4-turbo": {"prompt": 0.01, "completion": 0.03},       #
    "gpt-5-nano": {"prompt": 0.00005, "completion": 0.0004},   #
    "gpt-4.1-nano": {"prompt": 0.0001, "completion": 0.0004},  #
    "gpt-5.1": {"prompt": 0.00125, "completion": 0.01},        #
    "gpt-5.2": {"prompt": 0.00175, "completion": 0.014},       #
}

        rates = per_1k_rates.get(model_mapped.lower(), per_1k_rates["gpt-4o-mini"])
        prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
        completion_tokens = getattr(usage, "completion_tokens", 0) or 0
        cost = (prompt_tokens / 1000) * rates["prompt"] + (
            completion_tokens / 1000
        ) * rates["completion"]
        return float(cost)

    async def estimate_cost(
        self,
        agent_spec: AgentSpecification,
        input_data: dict[str, Any],
        config_overrides: dict[str, Any] | None = None,
    ) -> CostEstimate:
        """Estimate execution cost using tokencost library."""
        config = self._merge_configurations(
            agent_spec.model_parameters or {}, config_overrides
        )
        model = config.get("model", "gpt-4o-mini")

        # Format the actual prompt that will be sent
        prompt = self._format_prompt(agent_spec.prompt_template, input_data)

        try:
            from tokencost import calculate_completion_cost, calculate_prompt_cost

            # Convert prompt string to message format for tokencost
            prompt_messages = [{"role": "user", "content": prompt}]

            # Calculate input cost
            input_cost = float(calculate_prompt_cost(prompt_messages, model))

            # Estimate output cost based on max_tokens configuration
            max_tokens = int(config.get("max_tokens", 1000))
            # Create sample output text based on estimated length
            estimated_output = "x" * max(1, int(max_tokens * 0.7))
            output_cost = float(calculate_completion_cost(estimated_output, model))

            total_cost = float(input_cost + output_cost)

            return {
                "estimated_cost": float(total_cost),
                "estimated_input_cost": float(input_cost),
                "estimated_output_cost": float(output_cost),
                "confidence": 0.8,  # Higher confidence with tokencost
            }

        except ImportError:
            # Fallback to old estimation method
            estimated_prompt_tokens = int(len(prompt.split()) * 1.3)
            estimated_completion_tokens = int(int(config.get("max_tokens", 1000)) * 0.7)

            class MockUsage:
                def __init__(self, prompt: int, completion: int) -> None:
                    self.prompt_tokens = prompt
                    self.completion_tokens = completion
                    self.total_tokens = prompt + completion

            usage = MockUsage(estimated_prompt_tokens, estimated_completion_tokens)
            estimated_cost = float(self._calculate_cost(model, usage))

            return {
                "estimated_cost": estimated_cost,
                "estimated_tokens": usage.total_tokens,
                "confidence": 0.7,  # Medium confidence in estimate
            }
        except Exception:
            # If all cost calculation fails, return minimal estimates
            logger.debug("Cost estimation failed, returning minimal estimates")
            return {
                "estimated_cost": 0.001,  # Minimal estimate
                "confidence": 0.1,  # Low confidence
            }


class PlatformRegistry:
    """Registry for agent executor platforms."""

    _executors: dict[str, type[AgentExecutor]] = {
        "langchain": LangChainAgentExecutor,
        "openai": OpenAIAgentExecutor,
    }

    @classmethod
    def register_executor(
        cls, platform: str, executor_class: type[AgentExecutor]
    ) -> None:
        """Register a new executor platform.

        Args:
            platform: Platform name
            executor_class: Executor class
        """
        cls._executors[platform.lower()] = executor_class

    @classmethod
    def get_executor(cls, platform: str) -> type[AgentExecutor]:
        """Get executor class for platform.

        Args:
            platform: Platform name

        Returns:
            Executor class

        Raises:
            ValueError: If platform not found
        """
        executor = cls._executors.get(platform.lower())
        if not executor:
            raise ValueError(
                f"Unknown platform: {platform}. Available: {list(cls._executors.keys())}"
            )
        return executor

    @classmethod
    def list_platforms(cls) -> list[str]:
        """Get list of available platforms."""
        return list(cls._executors.keys())


def get_executor_for_platform(
    platform: str, platform_config: dict[str, Any] | None = None
) -> AgentExecutor:
    """Get executor instance for a platform.

    Args:
        platform: Platform name
        platform_config: Platform-specific configuration

    Returns:
        Executor instance
    """
    executor_class = PlatformRegistry.get_executor(platform)
    return executor_class(platform_config)
