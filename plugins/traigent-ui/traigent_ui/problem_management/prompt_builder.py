"""
Prompt Builder for Example Generation.

This module constructs effective prompts for generating additional examples
for existing problems using various LLM providers.
"""

import json
from typing import Any, Dict, List, Optional


class PromptBuilder:
    """Builds prompts for example generation."""

    def __init__(self):
        self.max_example_context = (
            5  # Number of existing examples to include as context
        )

    def build_generation_prompt(
        self,
        problem_instance,
        existing_examples: List[Any],
        custom_instructions: str,
        target_count: int,
    ) -> str:
        """
        Build a comprehensive prompt for example generation.

        Args:
            problem_instance: Instance of the problem class
            existing_examples: List of existing examples
            custom_instructions: User-provided custom instructions
            target_count: Number of examples to generate

        Returns:
            Complete generation prompt
        """
        # Extract problem metadata
        problem_info = self._extract_problem_info(problem_instance)

        # Get representative examples
        sample_examples = self._select_sample_examples(existing_examples)

        # Build the prompt
        prompt_parts = [
            self._build_problem_description(problem_info),
            self._build_example_context(sample_examples, problem_info),
            self._build_generation_requirements(target_count, problem_info),
            self._build_custom_instructions(custom_instructions),
            self._build_output_format_specification(problem_info),
            self._build_quality_guidelines(),
        ]

        return "\n\n".join(filter(None, prompt_parts))

    def _extract_problem_info(self, problem_instance) -> Dict[str, Any]:
        """Extract relevant information from the problem instance."""
        info = {
            "name": problem_instance.__class__.__name__,
            "description": "Unknown",
            "domain": "general",
            "categories": [],
            "difficulty_level": "medium",
            "input_structure": {},
            "output_type": "string",
        }

        # Try to get config information
        if hasattr(problem_instance, "config"):
            config = problem_instance.config
            info["description"] = getattr(config, "description", info["description"])
            info["difficulty_level"] = getattr(
                config, "difficulty_level", info["difficulty_level"]
            )

        # Try to get docstring
        if hasattr(problem_instance, "__doc__") and problem_instance.__doc__:
            info["description"] = problem_instance.__doc__.strip()

        # Try to get categories (for classification problems)
        if hasattr(problem_instance, "CATEGORIES"):
            info["categories"] = problem_instance.CATEGORIES
        elif hasattr(problem_instance, "categories"):
            info["categories"] = problem_instance.categories

        # Infer domain from class name or description
        info["domain"] = self._infer_domain(info["name"], info["description"])

        return info

    def _infer_domain(self, class_name: str, description: str) -> str:
        """Infer the domain from class name and description."""
        text = (class_name + " " + description).lower()

        domain_keywords = {
            "customer_service": ["customer", "support", "service", "ticket", "inquiry"],
            "technical": ["technical", "code", "software", "bug", "api", "system"],
            "medical": ["medical", "health", "patient", "diagnosis", "treatment"],
            "legal": ["legal", "law", "contract", "compliance", "regulation"],
            "financial": ["financial", "finance", "payment", "billing", "cost"],
            "educational": ["educational", "education", "student", "course", "lesson"],
        }

        for domain, keywords in domain_keywords.items():
            if any(keyword in text for keyword in keywords):
                return domain

        return "general"

    def _select_sample_examples(self, examples: List[Any]) -> List[Dict[str, Any]]:
        """Select representative examples to include in the prompt."""
        if not examples:
            return []

        # Convert examples to dictionaries
        sample_examples = []

        # Try different ways to extract example data
        for example in examples[: self.max_example_context]:
            if hasattr(example, "input_data") and hasattr(example, "expected_output"):
                # EvaluationExample format
                sample_examples.append(
                    {
                        "input_data": example.input_data,
                        "expected_output": example.expected_output,
                        "metadata": getattr(example, "metadata", {}),
                    }
                )
            elif isinstance(example, dict):
                # Dictionary format
                sample_examples.append(example)
            else:
                # Try to convert to dict
                try:
                    if hasattr(example, "__dict__"):
                        sample_examples.append(example.__dict__)
                except Exception:
                    continue

        return sample_examples

    def _build_problem_description(self, problem_info: Dict[str, Any]) -> str:
        """Build the problem description section."""
        description = f"**Problem**: {problem_info['name']}\n"
        description += f"**Domain**: {problem_info['domain']}\n"
        description += f"**Description**: {problem_info['description']}\n"

        if problem_info["categories"]:
            categories_str = ", ".join(problem_info["categories"])
            description += f"**Categories**: {categories_str}\n"

        description += f"**Difficulty Level**: {problem_info['difficulty_level']}"

        return description

    def _build_example_context(
        self, sample_examples: List[Dict[str, Any]], problem_info: Dict[str, Any]
    ) -> str:
        """Build the example context section."""
        if not sample_examples:
            return "No existing examples available for reference."

        context = f"**Existing Examples** (showing {len(sample_examples)} representative examples):\n\n"

        for i, example in enumerate(sample_examples, 1):
            context += f"Example {i}:\n"
            context += f"```json\n{json.dumps(example, indent=2)}\n```\n\n"

        return context.rstrip()

    def _build_generation_requirements(
        self, target_count: int, problem_info: Dict[str, Any]
    ) -> str:
        """Build the generation requirements section."""
        requirements = "**Generation Requirements**:\n"
        requirements += f"- Generate exactly {target_count} new examples\n"
        requirements += "- Follow the same format and structure as existing examples\n"
        requirements += (
            f"- Ensure examples are relevant to the {problem_info['domain']} domain\n"
        )
        requirements += (
            f"- Match the {problem_info['difficulty_level']} difficulty level\n"
        )

        if problem_info["categories"]:
            requirements += f"- Distribute examples across these categories: {', '.join(problem_info['categories'])}\n"

        requirements += "- Make examples diverse and non-repetitive\n"
        requirements += "- Ensure examples are realistic and practical\n"
        requirements += "- Include a variety of edge cases and scenarios"

        return requirements

    def _build_custom_instructions(self, custom_instructions: str) -> Optional[str]:
        """Build the custom instructions section."""
        if not custom_instructions or not custom_instructions.strip():
            return None

        return f"**Custom Instructions**:\n{custom_instructions.strip()}"

    def _build_output_format_specification(self, problem_info: Dict[str, Any]) -> str:
        """Build the output format specification."""
        format_spec = "**Output Format**:\n"
        format_spec += "Please provide your response as a JSON array of examples. Each example should have:\n"
        format_spec += "```json\n"
        format_spec += "[\n"
        format_spec += "  {\n"
        format_spec += '    "input_data": {"query": "example input text"},\n'
        format_spec += '    "expected_output": "expected response",\n'
        format_spec += '    "difficulty": "easy|medium|hard",\n'
        format_spec += '    "metadata": {"reasoning": "brief explanation"}\n'
        format_spec += "  }\n"
        format_spec += "]\n"
        format_spec += "```\n\n"
        format_spec += "**Important**: \n"
        format_spec += "- Respond with ONLY the JSON array\n"
        format_spec += "- No additional text or explanations\n"
        format_spec += "- Ensure valid JSON format\n"
        format_spec += "- Match the exact structure shown above"

        return format_spec

    def _build_quality_guidelines(self) -> str:
        """Build quality guidelines for generation."""
        guidelines = "**Quality Guidelines**:\n"
        guidelines += (
            "1. **Diversity**: Vary sentence structure, length, and complexity\n"
        )
        guidelines += "2. **Realism**: Use realistic scenarios and language\n"
        guidelines += "3. **Difficulty**: Match the specified difficulty level\n"
        guidelines += "4. **Clarity**: Ensure clear, unambiguous examples\n"
        guidelines += (
            "5. **Edge Cases**: Include some challenging or unusual scenarios\n"
        )
        guidelines += (
            "6. **Balance**: Distribute across different sub-categories if applicable\n"
        )
        guidelines += "7. **Consistency**: Maintain consistent format and style with existing examples"

        return guidelines


def build_prompt_for_problem(
    problem_instance, count: int, custom_instructions: str = ""
) -> str:
    """
    Convenience function to build a prompt for a specific problem.

    Args:
        problem_instance: Instance of the problem class
        count: Number of examples to generate
        custom_instructions: Custom instructions from user

    Returns:
        Complete generation prompt
    """
    # Get existing examples
    existing_examples = []
    if hasattr(problem_instance, "get_dataset"):
        try:
            dataset = problem_instance.get_dataset()
            if hasattr(dataset, "examples"):
                existing_examples = dataset.examples
        except Exception:
            pass
    elif hasattr(problem_instance, "create_dataset"):
        try:
            dataset = problem_instance.create_dataset()
            if hasattr(dataset, "examples"):
                existing_examples = dataset.examples
        except Exception:
            pass

    # Build prompt
    builder = PromptBuilder()
    return builder.build_generation_prompt(
        problem_instance=problem_instance,
        existing_examples=existing_examples,
        custom_instructions=custom_instructions,
        target_count=count,
    )
