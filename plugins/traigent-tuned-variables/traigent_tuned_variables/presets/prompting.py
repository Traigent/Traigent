"""Prompting strategy presets.

Provides pre-configured parameter ranges for prompt engineering optimization
including prompting strategies, context formats, and output formats.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from traigent.api.parameter_ranges import Choices, IntRange


class PromptingPresets:
    """Pre-configured parameter ranges for prompt engineering optimization.

    These presets encode domain knowledge about effective prompting strategies
    and formatting options.
    """

    @staticmethod
    def strategy() -> Choices:
        """Pre-configured prompting strategies.

        Includes research-backed strategies:
        - direct: Minimal instruction, baseline
        - chain_of_thought: Step-by-step reasoning (Wei et al. 2022)
        - react: Thought/Action/Observation pattern (Yao et al. 2022)
        - self_consistency: Multiple reasoning paths with voting

        Returns:
            Choices instance configured for strategy selection
        """
        from traigent.api.parameter_ranges import Choices

        return Choices(
            ["direct", "chain_of_thought", "react", "self_consistency"],
            default="direct",
            name="prompting_strategy",
        )

    @staticmethod
    def context_format() -> Choices:
        """Pre-configured context formatting options.

        Includes common formatting styles:
        - bullet: Bullet point list
        - numbered: Numbered list with sections
        - xml: XML-tagged sections (modern models prefer)
        - markdown: Markdown formatted

        Returns:
            Choices instance configured for format selection
        """
        from traigent.api.parameter_ranges import Choices

        return Choices(
            ["bullet", "numbered", "xml", "markdown"],
            default="bullet",
            name="context_format",
        )

    @staticmethod
    def output_format() -> Choices:
        """Pre-configured output format instructions.

        Includes common output styles:
        - concise: Brief, to-the-point responses
        - detailed: Comprehensive explanations
        - structured: Organized with headers/sections

        Returns:
            Choices instance configured for output format selection
        """
        from traigent.api.parameter_ranges import Choices

        return Choices(
            ["concise", "detailed", "structured"],
            default="concise",
            name="output_format",
        )

    @staticmethod
    def persona() -> Choices:
        """Pre-configured system personas.

        Includes common persona styles:
        - helpful_assistant: General-purpose helpful assistant
        - expert_advisor: Domain expert providing detailed guidance
        - friendly_helper: Approachable, simple explanations
        - professional: Formal, business-appropriate

        Returns:
            Choices instance configured for persona selection
        """
        from traigent.api.parameter_ranges import Choices

        return Choices(
            ["helpful_assistant", "expert_advisor", "friendly_helper", "professional"],
            default="helpful_assistant",
            name="persona",
        )

    @staticmethod
    def few_shot_count(*, max_examples: int = 10) -> IntRange:
        """Number of few-shot examples to include.

        Args:
            max_examples: Maximum number of examples

        Returns:
            IntRange instance configured for few-shot count
        """
        from traigent.api.parameter_ranges import IntRange

        return IntRange(0, max_examples, default=3, name="few_shot_count")

    @staticmethod
    def few_shot_strategy() -> Choices:
        """Few-shot example selection strategy.

        Includes common selection strategies:
        - random: Random selection from pool
        - semantic_knn: K-nearest neighbors by embedding similarity
        - mmr: Maximal Marginal Relevance (balance relevance + diversity)
        - curriculum: Easy-to-hard ordering

        Returns:
            Choices instance configured for selection strategy
        """
        from traigent.api.parameter_ranges import Choices

        return Choices(
            ["random", "semantic_knn", "mmr", "curriculum"],
            default="random",
            name="few_shot_strategy",
        )
