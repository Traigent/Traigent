"""
Problem Management Package for LangChain Optimization Problems.

This package provides intelligent tools for creating, modifying, and managing
LangChain optimization problems using Claude Code SDK.

Core Components:
- ProblemIntelligence: Uses Claude to understand and analyze problems
- CodeGenerator: Generates Python code for problem modules
- ExampleGenerator: Creates realistic examples across domains
- ProblemAnalyzer: Analyzes problem quality and suggests improvements
- DomainKnowledge: Domain-specific patterns and knowledge
"""

from .code_generator import CodeGenerator
from .domain_knowledge import DomainKnowledge
from .enhanced_example_generation import (
    generate_examples_for_problem,
    get_available_providers,
    save_examples_to_problem,
)
from .example_generator import ExampleGenerator
from .intelligence import ProblemIntelligence
from .llm_providers import LLMProviderManager
from .problem_analyzer import ProblemAnalyzer
from .prompt_builder import PromptBuilder, build_prompt_for_problem
from .smart_problem_analyzer import SmartProblemAnalyzer

__all__ = [
    "ProblemIntelligence",
    "CodeGenerator",
    "ExampleGenerator",
    "ProblemAnalyzer",
    "SmartProblemAnalyzer",
    "DomainKnowledge",
    "LLMProviderManager",
    "PromptBuilder",
    "build_prompt_for_problem",
    "generate_examples_for_problem",
    "get_available_providers",
    "save_examples_to_problem",
]
