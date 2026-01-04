"""Agent Specification Generator for Traigent SDK.

This module provides utilities to generate backend agent specifications
from SDK optimized functions, enabling seamless integration between
function-level optimization and agent-based execution.
"""

# Traceability: CONC-Layer-Core CONC-Quality-Usability FUNC-AGENTS REQ-AGNT-013

from __future__ import annotations

import ast
import inspect
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Protocol

from traigent.config.types import TraigentConfig
from traigent.core.constants import DEFAULT_MODEL
from traigent.utils.exceptions import FeatureNotAvailableError
from traigent.utils.logging import get_logger

# Cloud models - required at runtime for agent specification generation
try:
    from traigent.cloud.models import AgentSpecification

    _CLOUD_MODELS_AVAILABLE = True
except ModuleNotFoundError as err:
    # Check .name to distinguish missing cloud vs broken transitive dependency
    if err.name and err.name.startswith("traigent.cloud"):
        _CLOUD_MODELS_AVAILABLE = False
    else:
        raise  # Re-raise for broken dependencies like missing pydantic
    if TYPE_CHECKING:
        from traigent.cloud.models import AgentSpecification

logger = get_logger(__name__)


def _require_cloud_models() -> None:
    """Raise FeatureNotAvailableError if cloud models are not available."""
    if not _CLOUD_MODELS_AVAILABLE:
        raise FeatureNotAvailableError(
            "Agent specification generation",
            plugin_name="traigent-cloud",
            install_hint="pip install traigent[cloud]",
        )


class PromptTemplateBuilder(Protocol):
    """Protocol for prompt template building strategies."""

    def build_template(
        self, function_analysis: FunctionAnalysis, config: dict[str, Any]
    ) -> str:
        """Build a prompt template from function analysis and config."""
        ...


class PlatformInferenceStrategy(Protocol):
    """Protocol for platform inference strategies."""

    def infer_platform(self, config: dict[str, Any]) -> str:
        """Infer the target platform from configuration."""
        ...


@dataclass
class FunctionAnalysis:
    """Analysis of an optimized function."""

    name: str
    signature: str
    docstring: str | None
    parameters: list[str]
    return_annotation: str | None
    imports: list[str]
    dependencies: list[str]
    complexity_score: float
    inferred_domain: str
    suggested_agent_type: str


class DefaultPromptTemplateBuilder:
    """Default implementation of prompt template building strategy."""

    def build_template(
        self,
        function_analysis: FunctionAnalysis,
        _config: dict[
            str, Any
        ],  # Required by protocol but unused in default implementation
    ) -> str:
        """Build a prompt template using the default strategy."""
        function_name = function_analysis.name
        docstring = function_analysis.docstring or ""

        # Base template structure
        template_parts = []

        # System instruction
        if function_analysis.inferred_domain == "conversational":
            template_parts.append(
                f"You are a helpful AI assistant specialized in {function_name} tasks."
            )
        elif function_analysis.inferred_domain == "analytical":
            template_parts.append(
                f"You are an analytical AI that performs {function_name} analysis."
            )
        elif function_analysis.inferred_domain == "content_generation":
            template_parts.append(
                f"You are a content generation AI specialized in {function_name} tasks."
            )
        else:
            template_parts.append(
                f"You are an AI assistant that processes {function_name} requests."
            )

        # Add function description if available
        if docstring:
            clean_docstring = docstring.strip().split("\n")[0]  # First line only
            template_parts.append(f"\nTask Description: {clean_docstring}")

        # Input section
        template_parts.append("\nInput: {input}")

        # Processing instructions
        if function_analysis.parameters:
            if len(function_analysis.parameters) == 1:
                template_parts.append(
                    "\nProcess the input according to the requirements and provide an appropriate response."
                )
            else:
                param_list = ", ".join(function_analysis.parameters)
                template_parts.append(f"\nConsider these aspects: {param_list}")

        # Output format
        if function_analysis.return_annotation:
            template_parts.append(
                f"\nProvide your response as: {function_analysis.return_annotation}"
            )
        else:
            template_parts.append("\nResponse:")

        return "".join(template_parts)


class DefaultPlatformInferenceStrategy:
    """Default implementation of platform inference strategy."""

    def __init__(self) -> None:
        """Initialize with platform mappings."""
        self._platform_mappings: dict[str, Any] = {
            "openai": "openai",
            "anthropic": "anthropic",
            "langchain": "langchain",
            "llama": "huggingface",
            "default": "openai",
        }

    def infer_platform(self, config: dict[str, Any]) -> str:
        """Infer platform using the default strategy."""
        # Check configuration space for platform hints
        config_space = config.get("configuration_space", {})

        if "model" in config_space:
            models = config_space["model"]
            if isinstance(models, list):
                first_model = models[0] if models else ""
            else:
                first_model = str(models)

            if "gpt" in first_model.lower():
                return "openai"
            elif "claude" in first_model.lower():
                return "anthropic"
            elif "llama" in first_model.lower():
                return "huggingface"

        # Check for framework overrides
        if config.get("auto_override_frameworks"):
            return "langchain"  # Default for framework override

        return "openai"  # Default platform


class SpecificationGenerator:
    """Generator for creating agent specifications from optimized functions."""

    def __init__(
        self,
        prompt_builder: PromptTemplateBuilder | None = None,
        platform_strategy: PlatformInferenceStrategy | None = None,
    ) -> None:
        """Initialize the specification generator.

        Args:
            prompt_builder: Strategy for building prompt templates
            platform_strategy: Strategy for inferring target platforms
        """
        self._prompt_builder = prompt_builder or DefaultPromptTemplateBuilder()
        self._platform_strategy = (
            platform_strategy or DefaultPlatformInferenceStrategy()
        )

        self._domain_patterns: dict[str, Any] = {
            "conversational": [
                "chat",
                "conversation",
                "dialogue",
                "response",
                "reply",
                "message",
                "customer",
                "support",
                "assistant",
                "help",
            ],
            "analytical": [
                "analyze",
                "analysis",
                "classify",
                "classification",
                "predict",
                "prediction",
                "score",
                "rating",
                "sentiment",
                "evaluation",
                "assessment",
            ],
            "content_generation": [
                "generate",
                "creation",
                "write",
                "writing",
                "compose",
                "draft",
                "summary",
                "summarize",
                "translation",
                "translate",
            ],
            "task_automation": [
                "process",
                "processing",
                "transformation",
                "extract",
                "extraction",
                "parsing",
                "automation",
                "workflow",
            ],
        }

        self._platform_mappings: dict[str, Any] = {
            "openai": "openai",
            "anthropic": "anthropic",
            "langchain": "langchain",
            "llama": "huggingface",
            "default": "openai",
        }

    def from_optimized_function(
        self,
        func: Callable[..., Any],
        optimization_config: dict[str, Any] | None = None,
        agent_name: str | None = None,
        agent_platform: str | None = None,
    ) -> AgentSpecification:
        """Generate agent specification from optimized function.

        Args:
            func: Decorated function with optimization
            optimization_config: Optimization configuration from decorator
            agent_name: Optional custom agent name
            agent_platform: Optional target platform

        Returns:
            Complete agent specification

        Raises:
            FeatureNotAvailableError: If cloud models are not installed
        """
        _require_cloud_models()
        logger.info(f"Generating agent specification for function {func.__name__}")

        # Analyze the function
        analysis = self._analyze_function(func)

        # Extract optimization configuration
        config = self._extract_optimization_config(func, optimization_config)

        # Generate agent components
        agent_id = str(uuid.uuid4())
        name = agent_name or self._generate_agent_name(analysis)
        platform = agent_platform or self._platform_strategy.infer_platform(config)
        prompt_template = self._prompt_builder.build_template(analysis, config)
        model_parameters = self._extract_model_parameters(config)

        # Generate agent specification
        agent_spec = AgentSpecification(
            id=agent_id,
            name=name,
            agent_type=analysis.suggested_agent_type,
            agent_platform=platform,
            prompt_template=prompt_template,
            model_parameters=model_parameters,
            reasoning=self._generate_reasoning_instructions(analysis),
            style=self._infer_style(analysis),
            tone=self._infer_tone(analysis),
            format=self._infer_format(analysis),
            persona=self._generate_persona(analysis),
            guidelines=self._generate_guidelines(analysis),
            response_validation=True,
            custom_tools=self._extract_custom_tools(func),
            metadata={
                "generated_from": "sdk_optimized_function",
                "function_name": func.__name__,
                "function_signature": analysis.signature,
                "domain": analysis.inferred_domain,
                "complexity_score": analysis.complexity_score,
                "generation_timestamp": datetime.now(UTC).isoformat(),
            },
        )

        logger.info(
            f"Generated agent specification: {agent_spec.name} ({agent_spec.agent_type})"
        )
        return agent_spec

    def from_function_signature(
        self,
        function_name: str,
        function_signature: str,
        docstring: str | None = None,
        configuration_space: dict[str, Any] | None = None,
        objectives: list[str] | None = None,
    ) -> AgentSpecification:
        """Generate agent specification from function signature and metadata.

        Args:
            function_name: Name of the function
            function_signature: Function signature string
            docstring: Optional function docstring
            configuration_space: Parameter configuration space
            objectives: Optimization objectives

        Returns:
            Agent specification

        Raises:
            FeatureNotAvailableError: If cloud models are not installed
        """
        _require_cloud_models()
        logger.info(f"Generating agent specification from signature: {function_name}")

        # Create minimal analysis
        analysis = FunctionAnalysis(
            name=function_name,
            signature=function_signature,
            docstring=docstring,
            parameters=self._extract_parameters_from_signature(function_signature),
            return_annotation=self._extract_return_annotation(function_signature),
            imports=[],
            dependencies=[],
            complexity_score=0.5,  # Default medium complexity
            inferred_domain=self._infer_domain_from_name(function_name),
            suggested_agent_type=self._suggest_agent_type_from_domain(
                self._infer_domain_from_name(function_name)
            ),
        )

        # Generate configuration
        config = {
            "configuration_space": configuration_space or {},
            "objectives": objectives or ["accuracy"],
            "auto_override_frameworks": True,
        }

        return self._create_agent_specification_from_analysis(
            analysis, config, function_name
        )

    def update_agent_specification(
        self, agent_spec: AgentSpecification, optimization_results: dict[str, Any]
    ) -> AgentSpecification:
        """Update agent specification with optimization results.

        Args:
            agent_spec: Existing agent specification
            optimization_results: Results from optimization

        Returns:
            Updated agent specification
        """
        logger.info(
            f"Updating agent specification {agent_spec.name} with optimization results"
        )

        # Update model parameters with best configuration
        if "best_config" in optimization_results:
            best_config = optimization_results["best_config"]
            updated_params = (agent_spec.model_parameters or {}).copy()

            # Update with optimized parameters
            for key, value in best_config.items():
                if key in [
                    "model",
                    "temperature",
                    "max_tokens",
                    "top_p",
                    "frequency_penalty",
                    "presence_penalty",
                ]:
                    updated_params[key] = value

            agent_spec.model_parameters = updated_params

        # Update metadata with optimization results
        agent_spec.metadata.update(
            {
                "optimization_results": optimization_results,
                "optimized": True,
                "last_optimization": datetime.now(UTC).isoformat(),
            }
        )

        return agent_spec

    def _analyze_function(self, func: Callable[..., Any]) -> FunctionAnalysis:
        """Analyze function to extract characteristics."""
        name = func.__name__
        signature = str(inspect.signature(func))
        docstring = inspect.getdoc(func)

        # Get function source for deeper analysis
        try:
            source = inspect.getsource(func)
            imports, dependencies = self._extract_dependencies(source)
            complexity_score = self._calculate_complexity(source)
        except (OSError, TypeError):
            # Can't get source, use defaults
            imports, dependencies = [], []
            complexity_score = 0.5

        # Infer domain and agent type
        inferred_domain = self._infer_domain(name, docstring)
        suggested_agent_type = self._suggest_agent_type_from_domain(inferred_domain)

        return FunctionAnalysis(
            name=name,
            signature=signature,
            docstring=docstring,
            parameters=list(inspect.signature(func).parameters.keys()),
            return_annotation=self._get_return_annotation(func),
            imports=imports,
            dependencies=dependencies,
            complexity_score=complexity_score,
            inferred_domain=inferred_domain,
            suggested_agent_type=suggested_agent_type,
        )

    def _extract_optimization_config(
        self, func: Callable[..., Any], optimization_config: dict[str, Any] | None
    ) -> dict[str, Any]:
        """Extract optimization configuration from function or provided config."""
        config = dict(optimization_config or {})

        # Try to get config from function attributes (set by decorator)
        traigent_config = getattr(func, "_traigent_config", None)
        if isinstance(traigent_config, TraigentConfig):
            config.update(traigent_config.to_dict())
        elif isinstance(traigent_config, dict):
            config.update(traigent_config)

        # Try to get from other function attributes
        for attr in ["configuration_space", "objectives", "algorithm"]:
            value = getattr(func, f"_traigent_{attr}", None)
            if value is not None:
                config[attr] = value

        return config

    def _generate_agent_name(self, analysis: FunctionAnalysis) -> str:
        """Generate appropriate agent name from function analysis."""
        name = analysis.name

        # Convert snake_case or camelCase to Title Case
        if "_" in name:
            words = name.split("_")
        else:
            # Simple camelCase split
            words = []
            current_word = ""
            for char in name:
                if char.isupper() and current_word:
                    words.append(current_word)
                    current_word = char.lower()
                else:
                    current_word += char.lower()
            if current_word:
                words.append(current_word)

        # Create human-readable name
        title_words = [word.capitalize() for word in words]
        base_name = " ".join(title_words)

        # Add domain suffix if appropriate
        domain_suffixes = {
            "conversational": "Assistant",
            "analytical": "Analyzer",
            "content_generation": "Generator",
            "task_automation": "Processor",
        }

        suffix = domain_suffixes.get(analysis.inferred_domain, "Agent")
        if not base_name.endswith(suffix):
            base_name += f" {suffix}"

        return base_name

    # Backward compatibility methods for tests
    def _infer_platform(self, config: dict[str, Any]) -> str:
        """Backward compatibility method for tests."""
        return self._platform_strategy.infer_platform(config)

    def _generate_prompt_template(
        self, analysis: FunctionAnalysis, config: dict[str, Any]
    ) -> str:
        """Backward compatibility method for tests."""
        return self._prompt_builder.build_template(analysis, config)

    def _extract_model_parameters(self, config: dict[str, Any]) -> dict[str, Any]:
        """Extract model parameters from configuration.

        Uses shared constants from traigent.core.constants for default values.
        For parameter ranges (tuples), uses midpoint heuristic as a simple strategy.

        Args:
            config: Configuration dictionary containing configuration_space

        Returns:
            Dictionary of model parameters with defaults and overrides applied

        Note:
            Midpoint heuristic: For range parameters like (min, max), uses (min + max) / 2.
            This is a simple heuristic that works well for most optimization scenarios
            but may not be optimal for all use cases.
        """
        config_space = config.get("configuration_space", {})

        # Default parameters (maintaining backward compatibility with tests)
        # Note: These defaults are specific to agent generation and may differ from global constants
        model_params = {
            "model": DEFAULT_MODEL,  # From traigent.core.constants
            "temperature": 0.7,  # Default for agent generation
            "max_tokens": 150,  # Specific default for agent generation
            "top_p": 1.0,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
        }

        # Override with configuration space values
        for param, value in config_space.items():
            if param in model_params:
                if isinstance(value, list):
                    # Use first value from list
                    model_params[param] = value[0] if value else model_params[param]
                elif isinstance(value, tuple) and len(value) == 2:
                    # Midpoint heuristic for ranges: (min + max) / 2
                    # This provides a reasonable default for optimization ranges
                    min_val, max_val = value
                    model_params[param] = (min_val + max_val) / 2
                    logger.debug(
                        f"Applied midpoint heuristic to {param}: ({min_val}, {max_val}) -> {model_params[param]}"
                    )
                else:
                    # Direct value assignment
                    model_params[param] = value

        return model_params

    def _generate_reasoning_instructions(
        self, analysis: FunctionAnalysis
    ) -> str | None:
        """Generate reasoning instructions based on function analysis."""
        if analysis.complexity_score > 0.7:
            return (
                "Think step by step and break down complex problems into smaller parts."
            )
        elif analysis.inferred_domain == "analytical":
            return "Analyze the input carefully and provide evidence-based conclusions."
        else:
            return None

    def _infer_style(self, analysis: FunctionAnalysis) -> str | None:
        """Infer response style from function analysis."""
        style_mapping = {
            "conversational": "conversational and helpful",
            "analytical": "analytical and precise",
            "content_generation": "creative and engaging",
            "task_automation": "direct and efficient",
        }
        return style_mapping.get(analysis.inferred_domain)

    def _infer_tone(self, analysis: FunctionAnalysis) -> str | None:
        """Infer response tone from function analysis."""
        if "customer" in analysis.name.lower() or "support" in analysis.name.lower():
            return "professional and empathetic"
        elif "analysis" in analysis.name.lower():
            return "objective and informative"
        else:
            return "friendly and professional"

    def _infer_format(self, analysis: FunctionAnalysis) -> str | None:
        """Infer response format from function analysis."""
        if analysis.return_annotation:
            if (
                "dict" in analysis.return_annotation.lower()
                or "json" in analysis.return_annotation.lower()
            ):
                return "structured JSON"
            elif "list" in analysis.return_annotation.lower():
                return "list format"
            elif "str" in analysis.return_annotation.lower():
                return "plain text"

        # Infer from domain
        if analysis.inferred_domain == "analytical":
            return "structured analysis with key points"
        else:
            return "clear and concise text"

    def _generate_persona(self, analysis: FunctionAnalysis) -> str | None:
        """Generate agent persona from function analysis."""
        persona_templates = {
            "conversational": "a helpful and knowledgeable assistant",
            "analytical": "an expert analyst with attention to detail",
            "content_generation": "a creative and skilled writer",
            "task_automation": "an efficient and reliable processor",
        }
        return persona_templates.get(analysis.inferred_domain)

    def _generate_guidelines(self, analysis: FunctionAnalysis) -> list[str] | None:
        """Generate guidelines based on function analysis."""
        guidelines = [
            "Always provide accurate and helpful responses",
            "Stay focused on the specific task requirements",
        ]

        # Add domain-specific guidelines
        if analysis.inferred_domain == "conversational":
            guidelines.extend(
                [
                    "Be empathetic and understanding",
                    "Ask clarifying questions when needed",
                ]
            )
        elif analysis.inferred_domain == "analytical":
            guidelines.extend(
                ["Support conclusions with evidence", "Be objective and unbiased"]
            )
        elif analysis.inferred_domain == "content_generation":
            guidelines.extend(
                [
                    "Be creative while staying relevant",
                    "Maintain consistency in style and tone",
                ]
            )

        return guidelines

    def _extract_custom_tools(self, func: Callable[..., Any]) -> list[str] | None:
        """Extract custom tools from function if any.

        This is a best-effort heuristic that attempts to infer tools from function source.
        Results should be treated as suggestions and validated by the caller.
        """
        # Check for tool attributes set by decorators (highest priority)
        try:
            tools_attr = inspect.getattr_static(func, "_traigent_tools")
        except (AttributeError, RecursionError):
            tools_attr = None

        if tools_attr is not None:
            if isinstance(tools_attr, (list, tuple, set)):
                tools = [str(tool) for tool in tools_attr]
            else:
                tools = [str(tools_attr)]
            logger.debug(f"Found explicit tools from decorator: {tools}")
            return tools

        # Try to infer from function source (best-effort heuristic)
        try:
            source = inspect.getsource(func)
            tools = []

            # Look for common tool patterns (fragile heuristics)
            if "requests." in source or "urllib" in source:
                tools.append("web_search")
                logger.debug("Detected web_search tool from source pattern")
            if "json." in source or "JSON" in source:
                tools.append("json_parser")
                logger.debug("Detected json_parser tool from source pattern")
            if "datetime" in source or "time." in source:
                tools.append("datetime_utils")
                logger.debug("Detected datetime_utils tool from source pattern")

            if tools:
                unique_tools = list(dict.fromkeys(tools))
                logger.debug(f"Inferred tools from source analysis: {unique_tools}")
                return unique_tools
            logger.debug("No tools inferred from source analysis")
            return None
        except (OSError, TypeError) as e:
            logger.debug(f"Could not analyze function source for tools: {e}")
            return None

    def _infer_domain(self, name: str, docstring: str | None) -> str:
        """Infer function domain from name and docstring."""
        text_to_analyze = f"{name} {docstring or ''}".lower()

        domain_scores = {}
        for domain, patterns in self._domain_patterns.items():
            score = sum(1 for pattern in patterns if pattern in text_to_analyze)
            domain_scores[domain] = score

        # Return domain with highest score, default to conversational
        if domain_scores:
            return max(domain_scores, key=lambda domain: domain_scores[domain])
        return "conversational"

    def _infer_domain_from_name(self, name: str) -> str:
        """Infer domain from function name only."""
        return self._infer_domain(name, None)

    def _suggest_agent_type_from_domain(self, domain: str) -> str:
        """Suggest agent type based on domain."""
        type_mapping = {
            "conversational": "conversational",
            "analytical": "analytical",
            "content_generation": "content_generation",
            "task_automation": "task_automation",
        }
        return type_mapping.get(domain, "conversational")

    def _extract_dependencies(self, source: str) -> tuple[list[str], list[str]]:
        """Extract imports and dependencies from source code."""
        try:
            tree = ast.parse(source)
            imports = []
            dependencies = []

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)
                        dependencies.append(alias.name.split(".")[0])
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imports.append(node.module)
                    dependencies.append(node.module.split(".")[0])

            return list(set(imports)), list(set(dependencies))
        except (ValueError, SyntaxError, TypeError):
            return [], []

    def _calculate_complexity(self, source: str) -> float:
        """Calculate function complexity score (0.0 to 1.0)."""
        try:
            lines = source.split("\n")
            non_empty_lines = [line for line in lines if line.strip()]

            # Simple complexity metrics
            line_count = len(non_empty_lines)
            if_count = sum(1 for line in non_empty_lines if "if " in line)
            loop_count = sum(
                1
                for line in non_empty_lines
                if any(keyword in line for keyword in ["for ", "while "])
            )
            try_count = sum(1 for line in non_empty_lines if "try:" in line)

            # Normalize to 0-1 scale
            complexity = min(
                1.0, (line_count + if_count * 2 + loop_count * 3 + try_count * 2) / 50.0
            )
            return complexity
        except (ValueError, SyntaxError, TypeError):
            return 0.5  # Default medium complexity

    def _get_return_annotation(self, func: Callable[..., Any]) -> str | None:
        """Get return type annotation as string."""
        sig = inspect.signature(func)
        if sig.return_annotation != inspect.Signature.empty:
            return str(sig.return_annotation)
        return None

    def _extract_parameters_from_signature(self, signature: str) -> list[str]:
        """Extract parameter names from signature string."""
        try:
            # Simple parsing of signature string
            if "(" not in signature or ")" not in signature:
                return []

            params_str = signature[signature.find("(") + 1 : signature.rfind(")")]
            if not params_str.strip():
                return []

            # Build a temporary function definition to leverage Python's parser
            function_stub = f"def _temp({params_str}):\n    pass"
            module = ast.parse(function_stub)
            func_def = next(
                node for node in module.body if isinstance(node, ast.FunctionDef)
            )

            params: list[str] = []

            def append_arg(arg: ast.arg) -> None:
                if arg.arg != "self":
                    params.append(arg.arg)

            for arg in func_def.args.posonlyargs:
                append_arg(arg)
            for arg in func_def.args.args:
                append_arg(arg)
            if func_def.args.vararg:
                append_arg(func_def.args.vararg)
            for arg in func_def.args.kwonlyargs:
                append_arg(arg)
            if func_def.args.kwarg:
                append_arg(func_def.args.kwarg)

            return params
        except (ValueError, SyntaxError, TypeError, AttributeError, StopIteration):
            pass
        return []

    def _extract_return_annotation(self, signature: str) -> str | None:
        """Extract return annotation from signature string."""
        try:
            if "->" in signature:
                return signature.split("->")[-1].strip()
        except (ValueError, TypeError, AttributeError):
            pass
        return None

    def _create_agent_specification_from_analysis(
        self,
        analysis: FunctionAnalysis,
        config: dict[str, Any],
        function_name: str | None = None,
    ) -> AgentSpecification:
        """Create agent specification from analysis and config."""
        agent_id = str(uuid.uuid4())
        name = self._generate_agent_name(analysis)
        platform = self._platform_strategy.infer_platform(config)
        prompt_template = self._prompt_builder.build_template(analysis, config)
        model_parameters = self._extract_model_parameters(config)

        return AgentSpecification(
            id=agent_id,
            name=name,
            agent_type=analysis.suggested_agent_type,
            agent_platform=platform,
            prompt_template=prompt_template,
            model_parameters=model_parameters,
            reasoning=self._generate_reasoning_instructions(analysis),
            style=self._infer_style(analysis),
            tone=self._infer_tone(analysis),
            format=self._infer_format(analysis),
            persona=self._generate_persona(analysis),
            guidelines=self._generate_guidelines(analysis),
            response_validation=True,
            metadata={
                "generated_from": "sdk_function_signature",
                "function_name": analysis.name,
                "function_signature": analysis.signature,
                "domain": analysis.inferred_domain,
                "complexity_score": analysis.complexity_score,
            },
        )


# Global generator instance
generator = SpecificationGenerator()
