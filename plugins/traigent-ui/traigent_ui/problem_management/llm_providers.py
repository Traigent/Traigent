"""
LLM Provider Abstraction for Example Generation.

This module provides a unified interface for generating examples using different
LLM providers including OpenAI, Anthropic, and the built-in Claude Code generator.
"""

import json
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from .example_generator import ExampleGenerator

# Try to import Claude Code SDK
try:
    from claude_code_sdk import ClaudeCodeOptions, query

    CLAUDE_CODE_SDK_AVAILABLE = True
except ImportError:
    CLAUDE_CODE_SDK_AVAILABLE = False


@dataclass
class GenerationResult:
    """Result of example generation."""

    success: bool
    examples: List[Dict[str, Any]]
    provider: str
    token_usage: Optional[Dict[str, int]] = None
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    async def generate_examples(
        self,
        prompt: str,
        count: int,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> GenerationResult:
        """Generate examples using this provider."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this provider is available (API keys, etc.)."""
        pass

    def validate_example(self, example: Dict[str, Any]) -> bool:
        """Validate that an example has the required structure."""
        required_fields = ["input_data", "expected_output"]
        return all(field in example for field in required_fields)


class TemplateGeneratorProvider(LLMProvider):
    """Provider using the built-in template-based example generator."""

    def __init__(self):
        super().__init__("Template Generator")
        self.generator = ExampleGenerator()

    def is_available(self) -> bool:
        """Template generator is always available."""
        return True

    async def generate_examples(
        self,
        prompt: str,
        count: int,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> GenerationResult:
        """Generate examples using the built-in generator."""
        try:
            if progress_callback:
                progress_callback("Analyzing prompt for domain and type...", 0.1)

            # Parse the prompt to extract domain and problem type
            domain, problem_type, description = self._parse_prompt(prompt)

            if progress_callback:
                progress_callback(
                    f"Generating {count} examples with Claude Code...", 0.3
                )

            # Generate examples using the existing generator
            generated_examples = await self.generator.generate_examples(
                problem_type=problem_type,
                domain=domain,
                description=description,
                count=count,
                difficulty="Advanced",
            )

            if progress_callback:
                progress_callback("Converting examples to standard format...", 0.8)

            # Convert to standard format
            examples = []
            for gen_ex in generated_examples:
                example = {
                    "input_data": gen_ex.input_data,
                    "expected_output": gen_ex.expected_output,
                    "difficulty": gen_ex.difficulty,
                    "metadata": gen_ex.metadata,
                }
                examples.append(example)

            if progress_callback:
                progress_callback("Example generation complete!", 1.0)

            return GenerationResult(
                success=True,
                examples=examples,
                provider=self.name,
                metadata={"method": "built_in_generator"},
            )

        except Exception as e:
            return GenerationResult(
                success=False, examples=[], provider=self.name, error=str(e)
            )

    def _parse_prompt(self, prompt: str) -> tuple[str, str, str]:
        """Parse prompt to extract domain, type, and description."""
        # Simple parsing - in practice this could be more sophisticated
        prompt_lower = prompt.lower()

        # Extract domain
        domains = [
            "customer_service",
            "technical",
            "medical",
            "legal",
            "financial",
            "educational",
        ]
        domain = "general"
        for d in domains:
            if d.replace("_", " ") in prompt_lower or d in prompt_lower:
                domain = d
                break

        # Extract problem type
        if "classif" in prompt_lower or "categor" in prompt_lower:
            problem_type = "classification"
        elif "generat" in prompt_lower or "creat" in prompt_lower:
            problem_type = "generation"
        elif "analy" in prompt_lower:
            problem_type = "analysis"
        elif "extract" in prompt_lower:
            problem_type = "extraction"
        else:
            problem_type = "classification"  # Default

        # Use the prompt as description
        description = prompt

        return domain, problem_type, description


class ClaudeCodeSDKProvider(LLMProvider):
    """Provider using the official Claude Code SDK."""

    def __init__(self):
        super().__init__("Claude Code SDK")

    def is_available(self) -> bool:
        """Check if Claude Code SDK is available."""
        return CLAUDE_CODE_SDK_AVAILABLE

    async def generate_examples(
        self,
        prompt: str,
        count: int,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> GenerationResult:
        """Generate examples using Claude Code SDK."""
        try:
            if not self.is_available():
                raise ValueError(
                    "Claude Code SDK not installed. Run: pip install claude-code-sdk"
                )

            if progress_callback:
                progress_callback("Initializing Claude Code SDK...", 0.1)

            if progress_callback:
                progress_callback(
                    f"Generating {count} examples with Claude Code SDK...", 0.3
                )

            # Build the generation prompt
            generation_prompt = self._build_claude_code_prompt(prompt, count)

            # Use Claude Code SDK to generate examples
            messages = []
            async for message in query(
                prompt=generation_prompt,
                options=ClaudeCodeOptions(
                    system_prompt="You are an expert at generating diverse, high-quality training examples for LLM optimization problems. Always respond with valid JSON only.",
                    max_turns=1,  # We only need one response
                ),
            ):
                messages.append(message)

            # Extract text from assistant messages
            response_text = ""
            for msg in messages:
                if hasattr(msg, "type") and msg.type == "assistant":
                    if hasattr(msg, "message") and hasattr(msg.message, "content"):
                        content = msg.message.content
                        if isinstance(content, str):
                            response_text += content
                        elif isinstance(content, list):
                            for block in content:
                                if hasattr(block, "type") and block.type == "text":
                                    if hasattr(block, "text"):
                                        response_text += block.text

            if progress_callback:
                progress_callback("Parsing generated examples...", 0.8)

            # Parse the response
            examples = self._parse_claude_code_response(response_text)

            # Validate examples
            valid_examples = [ex for ex in examples if self.validate_example(ex)]

            if progress_callback:
                progress_callback(
                    f"Generated {len(valid_examples)} examples successfully!", 1.0
                )

            return GenerationResult(
                success=True,
                examples=valid_examples,
                provider=self.name,
                metadata={"sdk_version": "claude-code-sdk"},
            )

        except Exception as e:
            return GenerationResult(
                success=False, examples=[], provider=self.name, error=str(e)
            )

    def _build_claude_code_prompt(self, base_prompt: str, count: int) -> str:
        """Build the prompt for Claude Code SDK."""
        return f"""
{base_prompt}

Please generate {count} diverse and unique examples following this exact JSON format:

[
  {{
    "input_data": {{"query": "example input text"}},
    "expected_output": "expected category or response",
    "difficulty": "easy|medium|hard|very_hard|expert",
    "metadata": {{"reasoning": "brief explanation"}}
  }}
]

Requirements:
- Generate exactly {count} examples
- Ensure each example is unique and diverse
- Cover different scenarios and edge cases
- Mix difficulty levels appropriately
- Follow the exact JSON structure above
- Make examples realistic and practical
- Respond with ONLY the JSON array, no additional text
"""

    def _parse_claude_code_response(self, response) -> List[Dict[str, Any]]:
        """Parse Claude Code SDK response to extract examples."""
        # Extract text from response (adjust based on actual SDK response format)
        if hasattr(response, "text"):
            response_text = response.text
        elif isinstance(response, str):
            response_text = response
        else:
            response_text = str(response)

        # Use the same parsing logic as other providers
        return self._parse_json_response(response_text)

    def _parse_json_response(self, response: str) -> List[Dict[str, Any]]:
        """Parse JSON response to extract examples."""
        try:
            # Clean the response
            response = response.strip()

            # Extract JSON from the response
            json_match = re.search(r"\[.*\]", response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                examples = json.loads(json_str)
                return examples if isinstance(examples, list) else []
            else:
                # Try to parse the entire response as JSON
                return json.loads(response)

        except json.JSONDecodeError:
            # Fallback: try to extract individual JSON objects
            examples = []
            json_objects = re.findall(r"\{[^{}]*\}", response)
            for obj_str in json_objects:
                try:
                    obj = json.loads(obj_str)
                    if self.validate_example(obj):
                        examples.append(obj)
                except json.JSONDecodeError:
                    continue
            return examples


class OpenAIProvider(LLMProvider):
    """Provider using OpenAI API."""

    def __init__(self, model: str = "gpt-4o-mini"):
        super().__init__(f"OpenAI {model}")
        self.model = model
        self._client = None

    def is_available(self) -> bool:
        """Check if OpenAI API key is available."""
        api_key = os.environ.get("OPENAI_API_KEY", "")
        return bool(api_key and api_key.startswith("sk-"))

    async def generate_examples(
        self,
        prompt: str,
        count: int,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> GenerationResult:
        """Generate examples using OpenAI API."""
        try:
            if not self.is_available():
                raise ValueError("OpenAI API key not found or invalid")

            # Import OpenAI here to avoid dependency issues
            try:
                from openai import AsyncOpenAI
            except ImportError as e:
                raise ImportError(
                    "OpenAI library not installed. Run: pip install openai"
                ) from e

            if progress_callback:
                progress_callback("Connecting to OpenAI API...", 0.1)

            # Initialize client
            if not self._client:
                self._client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

            if progress_callback:
                progress_callback(
                    f"Generating {count} examples with {self.model}...", 0.3
                )

            # Create the generation prompt
            generation_prompt = self._build_openai_prompt(prompt, count)

            # Call OpenAI API
            response = await self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert at generating diverse, high-quality training examples for LLM optimization problems.",
                    },
                    {"role": "user", "content": generation_prompt},
                ],
                temperature=0.8,
                max_tokens=4000,
            )

            if progress_callback:
                progress_callback("Parsing generated examples...", 0.8)

            # Parse the response
            examples = self._parse_openai_response(response.choices[0].message.content)

            # Validate examples
            valid_examples = [ex for ex in examples if self.validate_example(ex)]

            if progress_callback:
                progress_callback(
                    f"Generated {len(valid_examples)} valid examples!", 1.0
                )

            return GenerationResult(
                success=True,
                examples=valid_examples,
                provider=self.name,
                token_usage={
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                },
                metadata={"model": self.model},
            )

        except Exception as e:
            return GenerationResult(
                success=False, examples=[], provider=self.name, error=str(e)
            )

    def _build_openai_prompt(self, base_prompt: str, count: int) -> str:
        """Build the prompt for OpenAI."""
        return f"""
{base_prompt}

Please generate {count} diverse examples following this exact JSON format:

[
  {{
    "input_data": {{"query": "example input text"}},
    "expected_output": "expected category or response",
    "difficulty": "easy|medium|hard",
    "metadata": {{"reasoning": "brief explanation"}}
  }}
]

Requirements:
- Generate exactly {count} examples
- Ensure diversity in patterns, difficulty, and content
- Follow the exact JSON structure above
- Make examples realistic and challenging
- Include a mix of difficulty levels
- Respond with ONLY the JSON array, no additional text
"""

    def _parse_openai_response(self, response: str) -> List[Dict[str, Any]]:
        """Parse OpenAI response to extract examples."""
        try:
            # Clean the response
            response = response.strip()

            # Extract JSON from the response
            json_match = re.search(r"\[.*\]", response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                examples = json.loads(json_str)
                return examples if isinstance(examples, list) else []
            else:
                # Try to parse the entire response as JSON
                return json.loads(response)

        except json.JSONDecodeError:
            # Fallback: try to extract individual JSON objects
            examples = []
            json_objects = re.findall(r"\{[^{}]*\}", response)
            for obj_str in json_objects:
                try:
                    obj = json.loads(obj_str)
                    if self.validate_example(obj):
                        examples.append(obj)
                except json.JSONDecodeError:
                    continue
            return examples


class AnthropicProvider(LLMProvider):
    """Provider using Anthropic Claude API."""

    def __init__(self, model: str = "claude-sonnet-4-0"):
        super().__init__(f"Anthropic {model}")
        self.model = model
        self._client = None

    def is_available(self) -> bool:
        """Check if Anthropic API key is available."""
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        return bool(api_key)

    async def generate_examples(
        self,
        prompt: str,
        count: int,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> GenerationResult:
        """Generate examples using Anthropic API."""
        try:
            if not self.is_available():
                raise ValueError("Anthropic API key not found")

            # Import Anthropic here to avoid dependency issues
            try:
                from anthropic import AsyncAnthropic
            except ImportError as e:
                raise ImportError(
                    "Anthropic library not installed. Run: pip install anthropic"
                ) from e

            if progress_callback:
                progress_callback("Connecting to Anthropic API...", 0.1)

            # Initialize client
            if not self._client:
                self._client = AsyncAnthropic(
                    api_key=os.environ.get("ANTHROPIC_API_KEY")
                )

            if progress_callback:
                progress_callback(
                    f"Generating {count} examples with {self.model}...", 0.3
                )

            # Create the generation prompt
            generation_prompt = self._build_anthropic_prompt(prompt, count)

            # Call Anthropic API
            response = await self._client.messages.create(
                model=self.model,
                max_tokens=4000,
                temperature=0.8,
                system="You are an expert at generating diverse, high-quality training examples for LLM optimization problems.",
                messages=[{"role": "user", "content": generation_prompt}],
            )

            if progress_callback:
                progress_callback("Parsing generated examples...", 0.8)

            # Parse the response
            examples = self._parse_anthropic_response(response.content[0].text)

            # Validate examples
            valid_examples = [ex for ex in examples if self.validate_example(ex)]

            if progress_callback:
                progress_callback(
                    f"Generated {len(valid_examples)} valid examples!", 1.0
                )

            return GenerationResult(
                success=True,
                examples=valid_examples,
                provider=self.name,
                token_usage={
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                },
                metadata={"model": self.model},
            )

        except Exception as e:
            return GenerationResult(
                success=False, examples=[], provider=self.name, error=str(e)
            )

    def _build_anthropic_prompt(self, base_prompt: str, count: int) -> str:
        """Build the prompt for Anthropic."""
        return f"""
{base_prompt}

Please generate {count} diverse examples following this exact JSON format:

[
  {{
    "input_data": {{"query": "example input text"}},
    "expected_output": "expected category or response",
    "difficulty": "easy|medium|hard",
    "metadata": {{"reasoning": "brief explanation"}}
  }}
]

Requirements:
- Generate exactly {count} examples
- Ensure diversity in patterns, difficulty, and content
- Follow the exact JSON structure above
- Make examples realistic and challenging
- Include a mix of difficulty levels
- Respond with ONLY the JSON array, no additional text
"""

    def _parse_anthropic_response(self, response: str) -> List[Dict[str, Any]]:
        """Parse Anthropic response to extract examples."""
        # Use the same parsing logic as OpenAI
        return OpenAIProvider._parse_openai_response(self, response)


class ClaudeAPIProvider(LLMProvider):
    """Provider using Claude API via Anthropic SDK (optimized for example generation)."""

    def __init__(self):
        super().__init__("Claude API")
        self.model = "claude-3-5-haiku-20241022"  # Fast model for example generation
        self._client = None

    def is_available(self) -> bool:
        """Check if Anthropic API key is available."""
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        return bool(api_key)

    async def generate_examples(
        self,
        prompt: str,
        count: int,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> GenerationResult:
        """Generate examples using Claude API with optimized prompting."""
        try:
            if not self.is_available():
                raise ValueError("Anthropic API key not found")

            # Import Anthropic here to avoid dependency issues
            try:
                from anthropic import AsyncAnthropic
            except ImportError as e:
                raise ImportError(
                    "Anthropic library not installed. Run: pip install anthropic"
                ) from e

            if progress_callback:
                progress_callback("Connecting to Claude API...", 0.1)

            # Initialize client
            if not self._client:
                self._client = AsyncAnthropic(
                    api_key=os.environ.get("ANTHROPIC_API_KEY")
                )

            if progress_callback:
                progress_callback(
                    f"Generating {count} diverse examples with Claude...", 0.3
                )

            # Create the generation prompt with Claude-specific optimizations
            generation_prompt = self._build_claude_prompt(prompt, count)

            # Call Claude API
            response = await self._client.messages.create(
                model=self.model,
                max_tokens=4000,
                temperature=0.9,  # Higher temperature for more diversity
                messages=[{"role": "user", "content": generation_prompt}],
            )

            if progress_callback:
                progress_callback("Parsing and validating examples...", 0.8)

            # Parse the response
            examples = self._parse_claude_response(response.content[0].text)

            # Validate examples
            valid_examples = [ex for ex in examples if self.validate_example(ex)]

            if progress_callback:
                progress_callback(
                    f"Generated {len(valid_examples)} unique examples!", 1.0
                )

            return GenerationResult(
                success=True,
                examples=valid_examples,
                provider=self.name,
                token_usage={
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                    "total_tokens": response.usage.input_tokens
                    + response.usage.output_tokens,
                },
                metadata={"model": self.model},
            )

        except Exception as e:
            return GenerationResult(
                success=False, examples=[], provider=self.name, error=str(e)
            )

    def _build_claude_prompt(self, base_prompt: str, count: int) -> str:
        """Build an optimized prompt for Claude to generate diverse examples."""
        return f"""
{base_prompt}

Your task is to generate {count} diverse and unique examples.

IMPORTANT REQUIREMENTS:
1. Each example MUST be completely different from others - no duplicates or near-duplicates
2. Vary the writing style, sentence structure, and vocabulary
3. Cover different scenarios and edge cases
4. Mix difficulty levels appropriately
5. Use realistic, practical examples

Generate exactly {count} examples in this JSON format:

[
  {{
    "input_data": {{"query": "unique example text here"}},
    "expected_output": "appropriate category or response",
    "difficulty": "easy|medium|hard|very_hard|expert",
    "metadata": {{"reasoning": "explanation of why this classification"}}
  }}
]

Remember:
- Each example should test different aspects of the problem
- Include edge cases and ambiguous scenarios for harder difficulties
- Ensure proper JSON formatting
- Output ONLY the JSON array, no other text
"""

    def _parse_claude_response(self, response: str) -> List[Dict[str, Any]]:
        """Parse Claude response to extract examples."""
        # Use the same parsing logic as OpenAI
        return OpenAIProvider._parse_openai_response(self, response)


class LLMProviderManager:
    """Manager for all available LLM providers."""

    def __init__(self):
        self.providers = {
            "Claude Code SDK": ClaudeCodeSDKProvider(),  # Default - uses your Claude Code Max account
            "Template Generator": TemplateGeneratorProvider(),
            "Claude API": ClaudeAPIProvider(),
            "OpenAI GPT-3.5-turbo": OpenAIProvider("gpt-3.5-turbo"),
            "OpenAI GPT-4o-mini": OpenAIProvider("gpt-4o-mini"),
            "OpenAI GPT-4o": OpenAIProvider("gpt-4o"),
            "Anthropic Claude-3-sonnet": AnthropicProvider("claude-sonnet-4-0"),
            "Anthropic Claude-3-opus": AnthropicProvider("claude-3-opus-20240229"),
        }

    def get_available_providers(self) -> List[str]:
        """Get list of available provider names."""
        return [
            name for name, provider in self.providers.items() if provider.is_available()
        ]

    def get_provider(self, name: str) -> Optional[LLMProvider]:
        """Get a provider by name."""
        return self.providers.get(name)

    async def generate_examples(
        self,
        provider_name: str,
        prompt: str,
        count: int,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> GenerationResult:
        """Generate examples using the specified provider."""
        provider = self.get_provider(provider_name)
        if not provider:
            return GenerationResult(
                success=False,
                examples=[],
                provider=provider_name,
                error=f"Provider '{provider_name}' not found",
            )

        if not provider.is_available():
            return GenerationResult(
                success=False,
                examples=[],
                provider=provider_name,
                error=f"Provider '{provider_name}' is not available (check API keys)",
            )

        return await provider.generate_examples(prompt, count, progress_callback)


# Backwards-compatible helper for code that imported directly from this module
def get_available_providers() -> List[str]:
    """Return names of providers that are currently usable (API keys present)."""
    return LLMProviderManager().get_available_providers()
