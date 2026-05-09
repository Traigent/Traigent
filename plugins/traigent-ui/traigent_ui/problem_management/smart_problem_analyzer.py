"""
Smart Problem Analyzer - LLM-Powered Problem Specification Generation.

This module uses Claude Code SDK to analyze user descriptions and generate
structured problem specifications that are contextually relevant and appropriate
for the described task.
"""

import asyncio
import json
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

try:
    from claude_code_sdk import ClaudeCodeOptions, query

    CLAUDE_SDK_AVAILABLE = True
except ImportError:
    print("Warning: claude_code_sdk not available. Using mock implementation.")
    CLAUDE_SDK_AVAILABLE = False

    # Define mock ClaudeCodeOptions class
    class ClaudeCodeOptions:
        def __init__(self, system_prompt=None, permission_mode=None, **kwargs):
            self.system_prompt = system_prompt
            self.permission_mode = permission_mode
            for k, v in kwargs.items():
                setattr(self, k, v)

    async def query(prompt: str, options=None):
        """Mock implementation for when Claude Code SDK is not available."""
        yield f"Mock response for: {prompt[:100]}..."


def extract_text_from_claude_messages(messages):
    """Extract text content from Claude SDK messages.

    Args:
        messages: List of Message objects from Claude SDK

    Returns:
        str: Concatenated text content from assistant messages
    """
    response_text = ""
    for msg in messages:
        # Check if this is an assistant message with content
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
                        elif isinstance(block, str):
                            response_text += block
    return response_text


from .example_validator import ExampleValidator  # noqa: E402
from .prompt_templates import PromptTemplates  # noqa: E402

# Import problem types system
try:
    # Use absolute import - we add parent to path in traigent_control_center.py
    from problem_generation.constrained_generators import get_constrained_generator
    from problem_generation.improved_problem_classifier import (
        ImprovedProblemClassifier as ProblemTypeClassifier,
    )
    from problem_generation.problem_types import (
        ClassificationProblem,
        CodeGenerationProblem,
        InformationExtractionProblem,
        ProblemType,
        QuestionAnsweringProblem,
        RankingRetrievalProblem,
        ReasoningProblem,
        SequenceGenerationProblem,
        SummarizationProblem,
        TranslationTransformationProblem,
        get_problem_type,
    )
except ImportError as e:
    print(f"Warning: Problem types system not available. Using fallback. Error: {e}")
    ProblemType = None
    ProblemTypeClassifier = None
    get_constrained_generator = None


@dataclass
class SmartProblemSpecification:
    """Enhanced problem specification with LLM-generated insights."""

    name: str
    description: str
    domain: str
    problem_type: str
    difficulty_level: str

    # LLM-generated content
    contextual_examples: List[Dict[str, Any]]
    example_patterns: List[str]
    evaluation_metrics: List[str]
    technical_requirements: List[str]

    # Structure definitions
    input_structure: Dict[str, Any]
    output_structure: Dict[str, Any]

    # Metadata
    user_intent: str
    confidence_score: float
    reasoning: str


class SmartProblemAnalyzer:
    """
    Intelligent problem analyzer using Claude Code SDK.

    Analyzes natural language descriptions and generates structured
    problem specifications with contextually relevant examples and patterns.
    """

    def __init__(self):
        """Initialize the smart analyzer."""
        # Problem type instances cache
        self._problem_type_cache = {}

        # Initialize problem type classifier if available
        self.classifier = ProblemTypeClassifier() if ProblemTypeClassifier else None

        self.domain_expertise = {
            "educational": {
                "math": ["arithmetic", "word_problems", "geometry", "algebra"],
                "science": ["physics", "chemistry", "biology", "earth_science"],
                "language": [
                    "reading_comprehension",
                    "grammar",
                    "writing",
                    "vocabulary",
                ],
                "social_studies": ["history", "geography", "civics", "economics"],
            },
            "business": {
                "finance": ["budgeting", "investment", "accounting", "risk_analysis"],
                "marketing": ["campaigns", "customer_analysis", "brand_strategy"],
                "operations": [
                    "supply_chain",
                    "quality_control",
                    "process_optimization",
                ],
                "hr": ["recruitment", "performance_review", "training"],
            },
            "technical": {
                "software": ["debugging", "code_review", "architecture", "testing"],
                "data": ["analysis", "visualization", "cleaning", "modeling"],
                "security": ["threat_analysis", "compliance", "incident_response"],
                "infrastructure": ["monitoring", "deployment", "scaling"],
            },
            "customer_service": {
                "support": ["issue_resolution", "escalation", "satisfaction"],
                "sales": ["lead_qualification", "objection_handling", "closing"],
                "retention": ["churn_prevention", "upselling", "feedback"],
            },
            "legal": {
                "contracts": ["review", "drafting", "negotiation", "compliance"],
                "litigation": ["discovery", "brief_writing", "case_analysis"],
                "regulatory": ["compliance", "reporting", "risk_assessment"],
            },
            "medical": {
                "diagnosis": ["symptom_analysis", "differential_diagnosis", "testing"],
                "treatment": ["care_plans", "medication", "monitoring"],
                "research": ["clinical_trials", "data_analysis", "publication"],
            },
        }

        # Updated mapping to match 8 standardized problem types
        self.problem_type_mapping = {
            # Action words → problem types
            "classify": "classification",
            "categorize": "classification",
            "sort": "classification",
            "identify": "classification",
            "generate": "generation",
            "create": "generation",
            "write": "generation",
            "compose": "generation",
            "produce": "generation",
            "extract": "information_extraction",
            "find": "information_extraction",
            "locate": "information_extraction",
            "parse": "information_extraction",
            "answer": "question_answering",
            "respond": "question_answering",
            "explain": "question_answering",
            "summarize": "summarization",
            "condense": "summarization",
            "brief": "summarization",
            "abstract": "summarization",
            "rank": "ranking_retrieval",
            "search": "ranking_retrieval",
            "retrieve": "ranking_retrieval",
            "recommend": "ranking_retrieval",
            "translate": "translation_transformation",
            "transform": "translation_transformation",
            "convert": "translation_transformation",
            "rewrite": "translation_transformation",
            "solve": "reasoning",
            "reason": "reasoning",
            "deduce": "reasoning",
            "calculate": "reasoning",
        }

        # Map problem types to their structured implementations
        self.structured_problem_map = {
            "classification": ClassificationProblem,
            "generation": SequenceGenerationProblem,
            "information_extraction": InformationExtractionProblem,
            "question_answering": QuestionAnsweringProblem,
            "summarization": SummarizationProblem,
            "ranking_retrieval": RankingRetrievalProblem,
            "translation_transformation": TranslationTransformationProblem,
            "reasoning": ReasoningProblem,
            "code_generation": CodeGenerationProblem,
        }

    async def analyze_and_generate_spec(
        self,
        user_description: str,
        target_examples: int = 100,
        generation_mode: str = "smart_single",
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> SmartProblemSpecification:
        """
        Analyze user description and generate a complete problem specification.

        Args:
            user_description: Natural language description from user
            target_examples: Number of examples to generate
            generation_mode: Generation strategy
            progress_callback: Optional callback for progress updates

        Returns:
            Complete smart problem specification
        """
        # Update progress
        if progress_callback:
            progress_callback(10, "Analyzing user intent...")

        # Step 1: Use the new classifier if available
        if self.classifier:
            classification_result = self.classifier.classify(user_description)
            intent_analysis = {
                "intent": f"Create {classification_result.problem_type} problem",
                "domain": self._extract_domain_from_description(user_description),
                "problem_type": classification_result.problem_type,
                "confidence": classification_result.confidence,
                "reasoning": classification_result.reasoning,
                "suggested_metrics": classification_result.suggested_metrics,
                "detected_keywords": classification_result.detected_keywords,
            }
        else:
            # Fallback to original analysis
            intent_analysis = await self._analyze_user_intent(user_description)

        if progress_callback:
            progress_callback(
                30, f"Identified problem type: {intent_analysis['problem_type']}"
            )

        # Step 2: Generate problem specification
        problem_spec = await self._generate_problem_specification(
            user_description, intent_analysis, target_examples, progress_callback
        )

        if progress_callback:
            progress_callback(50, "Creating problem structure...")

        # Step 3: Get structured problem type if available
        structured_problem = self._get_structured_problem_type(
            problem_spec["problem_type"], problem_spec
        )

        # Step 4: Generate contextual examples with proper input-output pairs
        if progress_callback:
            progress_callback(60, "Generating constrained examples...")

        contextual_examples = await self._generate_contextual_examples(
            problem_spec, target_examples, structured_problem, progress_callback
        )

        if progress_callback:
            progress_callback(90, "Finalizing specification...")

        # Use suggested metrics from classifier if available
        evaluation_metrics = intent_analysis.get(
            "suggested_metrics", []
        ) or problem_spec.get("evaluation_metrics", ["accuracy", "relevance"])

        # Step 5: Create final specification
        return SmartProblemSpecification(
            name=problem_spec["name"],
            description=problem_spec["description"],
            domain=problem_spec["domain"],
            problem_type=problem_spec["problem_type"],
            difficulty_level=problem_spec["difficulty_level"],
            contextual_examples=contextual_examples,
            example_patterns=problem_spec["example_patterns"],
            evaluation_metrics=evaluation_metrics,
            technical_requirements=problem_spec["technical_requirements"],
            input_structure=problem_spec.get("input_structure", {"text": "input text"}),
            output_structure=problem_spec.get(
                "output_structure", {"result": "output result"}
            ),
            user_intent=intent_analysis["intent"],
            confidence_score=intent_analysis["confidence"],
            reasoning=intent_analysis["reasoning"],
        )

    def _extract_domain_from_description(self, description: str) -> str:
        """Extract domain from description using keyword matching."""
        description_lower = description.lower()

        # Special cases for technical domains
        technical_keywords = [
            "sql",
            "database",
            "query",
            "code",
            "programming",
            "algorithm",
            "function",
            "api",
        ]
        if any(keyword in description_lower for keyword in technical_keywords):
            return "technical"

        # Check each domain and its keywords
        for domain, subdomains in self.domain_expertise.items():
            # Check domain name itself
            if domain in description_lower:
                return domain

            # Check subdomain keywords
            for subdomain, keywords in subdomains.items():
                if subdomain in description_lower:
                    return domain
                for keyword in keywords:
                    if keyword in description_lower:
                        return domain

        # Default domain
        return "general"

    async def _analyze_user_intent(self, description: str) -> Dict[str, Any]:
        """Analyze user intent from description using Claude Code SDK."""

        analysis_prompt = f"""
Analyze this user description for creating an LLM optimization problem:

"{description}"

Respond with a simple JSON object containing these exact keys:

```json
{{
    "intent": "What the user wants to accomplish",
    "domain": "educational, business, technical, customer_service, legal, medical, or general",
    "problem_type": "classification, generation, question_answering, analysis, extraction, or reasoning",
    "confidence": 0.85
}}
```

Examples:
- "elementary school math problems" → domain: educational, type: question_answering
- "customer support classification" → domain: customer_service, type: classification
- "legal document analysis" → domain: legal, type: analysis

Keep the response simple and focused on these four key fields.
"""

        try:
            messages = []
            async for message in query(
                prompt=analysis_prompt,
                options=ClaudeCodeOptions(
                    system_prompt="You are an expert at analyzing problem descriptions and understanding user intent for LLM task creation. Provide accurate, detailed analysis in valid JSON format.",
                    max_turns=1,
                ),
            ):
                messages.append(message)

            # Extract text using helper function
            response_text = extract_text_from_claude_messages(messages)

            # Use robust JSON extraction
            expected_keys = ["intent", "domain", "problem_type", "confidence"]
            analysis = self._extract_json_robustly(response_text, expected_keys)

            if analysis:
                # Ensure all required fields are present with defaults
                analysis.setdefault("intent", f"Analyze {description}")
                analysis.setdefault("domain", "general")
                analysis.setdefault("problem_type", "classification")
                analysis.setdefault("confidence", 0.7)

                # Add missing fields for compatibility
                analysis.setdefault("subdomain", "")
                analysis.setdefault("key_entities", description.split()[:5])
                analysis.setdefault("difficulty_indicators", ["medium"])
                analysis.setdefault("context_clues", [description])
                analysis.setdefault(
                    "example_requirements",
                    [f"{analysis['domain']} {analysis['problem_type']} examples"],
                )
                analysis.setdefault(
                    "reasoning",
                    f"Extracted from Claude Code SDK response: {response_text[:100]}...",
                )

                return analysis

            # Fallback analysis
            print("Using fallback analysis due to parsing issues")
            return self._fallback_intent_analysis(description)

        except Exception as e:
            print(f"Error in intent analysis: {e}")
            return self._fallback_intent_analysis(description)

    def _extract_json_robustly(
        self, response_text: str, expected_keys: List[str]
    ) -> Optional[Dict[str, Any]]:
        """
        Robust JSON extraction with multiple strategies.

        Args:
            response_text: The raw response text
            expected_keys: List of keys we expect in the JSON

        Returns:
            Parsed JSON dict or None if all strategies fail
        """
        if not response_text.strip():
            return None

        strategies = [
            self._extract_from_code_blocks,
            self._extract_from_json_pattern,
            self._extract_key_value_pairs,
            self._extract_from_lines,
        ]

        for strategy in strategies:
            try:
                result = strategy(response_text, expected_keys)
                if result and all(key in result for key in expected_keys):
                    return result
            except Exception as e:
                print(f"Strategy {strategy.__name__} failed: {e}")
                continue

        return None

    def _extract_from_code_blocks(
        self, text: str, expected_keys: List[str]
    ) -> Optional[Dict[str, Any]]:
        """Extract JSON from markdown code blocks."""
        # Look for ```json blocks
        json_blocks = re.findall(
            r"```json\s*\n(.*?)\n```", text, re.DOTALL | re.IGNORECASE
        )
        if not json_blocks:
            # Look for ``` blocks that might contain JSON
            json_blocks = re.findall(r"```\s*\n(\{.*?\})\s*\n```", text, re.DOTALL)

        for block in json_blocks:
            try:
                # Clean up the block
                cleaned_block = block.strip()
                # Fix common JSON issues
                cleaned_block = re.sub(
                    r"(\w+):", r'"\1":', cleaned_block
                )  # Add quotes to keys
                cleaned_block = re.sub(
                    r",\s*}", "}", cleaned_block
                )  # Remove trailing commas

                result = json.loads(cleaned_block)
                if isinstance(result, dict):
                    return result
            except json.JSONDecodeError:
                continue

        return None

    def _extract_from_json_pattern(
        self, text: str, expected_keys: List[str]
    ) -> Optional[Dict[str, Any]]:
        """Extract JSON using regex patterns."""
        # Look for JSON-like patterns
        json_patterns = [
            r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}",  # Simple nested objects
            r"\{.*?\}",  # Simple objects
        ]

        for pattern in json_patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            for match in matches:
                try:
                    # Clean up the match
                    cleaned = match.strip()
                    # Fix common formatting issues
                    cleaned = re.sub(r"(\w+):", r'"\1":', cleaned)  # Add quotes to keys
                    cleaned = re.sub(r",\s*}", "}", cleaned)  # Remove trailing commas

                    result = json.loads(cleaned)
                    if (
                        isinstance(result, dict)
                        and len(result) >= len(expected_keys) // 2
                    ):
                        return result
                except json.JSONDecodeError:
                    continue

        return None

    def _extract_key_value_pairs(
        self, text: str, expected_keys: List[str]
    ) -> Optional[Dict[str, Any]]:
        """Extract key-value pairs from structured text."""
        result = {}

        # Look for key: value patterns
        for key in expected_keys:
            patterns = [
                rf'"{key}"\s*:\s*"([^"]*)"',  # "key": "value"
                rf'"{key}"\s*:\s*([^,\n\}}]+)',  # "key": value
                rf'{key}\s*:\s*"([^"]*)"',  # key: "value"
                rf"{key}\s*:\s*([^,\n\}}]+)",  # key: value
            ]

            for pattern in patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    value = match.group(1).strip().strip('"').strip("'")
                    result[key] = value
                    break

        return result if result else None

    def _extract_from_lines(
        self, text: str, expected_keys: List[str]
    ) -> Optional[Dict[str, Any]]:
        """Extract information from line-by-line analysis."""
        result = {}
        lines = text.split("\n")

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Look for key-value patterns in each line
            for key in expected_keys:
                if key.lower() in line.lower():
                    # Try to extract the value
                    value_patterns = [
                        rf"{key}\s*[:\-=]\s*(.+)",
                        rf"{key.title()}\s*[:\-=]\s*(.+)",
                        rf"{key.upper()}\s*[:\-=]\s*(.+)",
                    ]

                    for pattern in value_patterns:
                        match = re.search(pattern, line, re.IGNORECASE)
                        if match:
                            value = match.group(1).strip().strip('"').strip("'")
                            result[key] = value
                            break

        return result if result else None

    def _parse_response_alternative(
        self, response_text: str
    ) -> Optional[Dict[str, Any]]:
        """Alternative parsing method for when JSON parsing fails."""
        try:
            # Extract key information using regex patterns
            analysis = {}

            # Look for intent patterns
            intent_match = re.search(
                r'intent["\s]*[:=]["\s]*([^",\n]+)', response_text, re.IGNORECASE
            )
            analysis["intent"] = (
                intent_match.group(1).strip()
                if intent_match
                else "Analyze user description"
            )

            # Look for domain patterns
            domain_match = re.search(
                r'domain["\s]*[:=]["\s]*([^",\n]+)', response_text, re.IGNORECASE
            )
            analysis["domain"] = (
                domain_match.group(1).strip().lower() if domain_match else "general"
            )

            # Look for problem type patterns
            type_match = re.search(
                r'problem_type["\s]*[:=]["\s]*([^",\n]+)', response_text, re.IGNORECASE
            )
            analysis["problem_type"] = (
                type_match.group(1).strip().lower() if type_match else "classification"
            )

            # Look for confidence patterns
            conf_match = re.search(
                r'confidence["\s]*[:=]["\s]*([0-9.]+)', response_text, re.IGNORECASE
            )
            analysis["confidence"] = float(conf_match.group(1)) if conf_match else 0.7

            # Set defaults for other fields
            analysis["subdomain"] = ""
            analysis["key_entities"] = []
            analysis["difficulty_indicators"] = ["medium"]
            analysis["context_clues"] = [response_text[:100]]
            analysis["example_requirements"] = ["relevant examples"]
            analysis["reasoning"] = f"Parsed from response: {response_text[:100]}..."

            return analysis if analysis["intent"] else None

        except Exception as e:
            print(f"Alternative parsing error: {e}")
            return None

    async def _generate_problem_specification(
        self,
        description: str,
        intent_analysis: Dict[str, Any],
        target_examples: int,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> Dict[str, Any]:
        """Generate detailed problem specification using Claude Code SDK."""

        spec_prompt = f"""
Create a problem specification for: "{description}"

Based on analysis:
- Domain: {intent_analysis.get('domain', 'general')}
- Type: {intent_analysis.get('problem_type', 'classification')}

Respond with a simple JSON object:

```json
{{
    "name": "descriptive_problem_name",
    "description": "Clear description of what this problem does",
    "domain": "{intent_analysis.get('domain', 'general')}",
    "problem_type": "{intent_analysis.get('problem_type', 'classification')}",
    "difficulty_level": "medium"
}}
```

Make the name descriptive and the description specific to the user's request.
For "elementary school math problems" → name: "elementary_math_word_problems"
"""

        try:
            messages = []
            async for message in query(
                prompt=spec_prompt,
                options=ClaudeCodeOptions(
                    system_prompt="You are an expert at creating detailed, accurate problem specifications for LLM tasks. Generate comprehensive, contextually appropriate specifications in valid JSON format.",
                    max_turns=1,
                ),
            ):
                messages.append(message)

            # Extract text using helper function
            response_text = extract_text_from_claude_messages(messages)

            # Use robust JSON extraction
            expected_keys = [
                "name",
                "description",
                "domain",
                "problem_type",
                "difficulty_level",
            ]
            spec = self._extract_json_robustly(response_text, expected_keys)

            if spec:
                # Ensure all required fields are present with defaults
                spec.setdefault(
                    "name",
                    f"{intent_analysis.get('domain', 'general')}_{intent_analysis.get('problem_type', 'problem')}",
                )
                spec.setdefault(
                    "description",
                    f"A {intent_analysis.get('problem_type', 'classification')} problem: {description}",
                )
                spec.setdefault("domain", intent_analysis.get("domain", "general"))
                spec.setdefault(
                    "problem_type",
                    intent_analysis.get("problem_type", "classification"),
                )
                spec.setdefault("difficulty_level", "medium")

                # Add additional fields for compatibility
                spec.setdefault(
                    "example_patterns",
                    [
                        f"Basic {spec['problem_type']} examples",
                        f"Intermediate {spec['problem_type']} examples",
                        f"Advanced {spec['problem_type']} examples",
                    ],
                )
                spec.setdefault(
                    "evaluation_metrics", ["accuracy", "relevance", "quality"]
                )
                spec.setdefault(
                    "technical_requirements", [f"{spec['problem_type']}_capability"]
                )
                spec.setdefault("input_structure", {"text": "input text"})
                spec.setdefault("output_structure", {"result": "output result"})

                return spec

            # Fallback specification
            print("Using fallback specification due to parsing issues")
            return self._fallback_problem_specification(description, intent_analysis)

        except Exception as e:
            print(f"Error in specification generation: {e}")
            return self._fallback_problem_specification(description, intent_analysis)

    def _parse_specification_alternative(
        self, response_text: str, intent_analysis: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Alternative parsing for problem specifications."""
        try:
            spec = {}

            # Extract name
            name_match = re.search(
                r'name["\s]*[:=]["\s]*([^",\n]+)', response_text, re.IGNORECASE
            )
            spec["name"] = (
                name_match.group(1).strip()
                if name_match
                else f"{intent_analysis.get('domain', 'general')}_{intent_analysis.get('problem_type', 'problem')}"
            )

            # Extract description
            desc_match = re.search(
                r'description["\s]*[:=]["\s]*([^",\n]+)', response_text, re.IGNORECASE
            )
            spec["description"] = (
                desc_match.group(1).strip()
                if desc_match
                else f"A {intent_analysis.get('problem_type', 'classification')} problem for {intent_analysis.get('domain', 'general')} domain"
            )

            # Use intent analysis for core fields
            spec["domain"] = intent_analysis.get("domain", "general")
            spec["problem_type"] = intent_analysis.get("problem_type", "classification")
            spec["difficulty_level"] = "medium"

            # Default patterns and metrics
            spec["example_patterns"] = [
                f"Basic {spec['problem_type']} examples",
                f"Intermediate {spec['problem_type']} examples",
                f"Advanced {spec['problem_type']} examples",
            ]
            spec["evaluation_metrics"] = ["accuracy", "relevance", "quality"]
            spec["technical_requirements"] = [f"{spec['problem_type']}_capability"]
            spec["input_structure"] = {"text": "input text"}
            spec["output_structure"] = {"result": "output result"}

            return spec

        except Exception as e:
            print(f"Alternative specification parsing error: {e}")
            return None

    def _get_domain_categories(self, domain: str) -> List[str]:
        """Get default categories for a domain."""
        domain_categories = {
            "customer_service": [
                "billing_issue",
                "technical_support",
                "account_support",
                "order_inquiry",
                "general_question",
            ],
            "medical": [
                "routine_checkup",
                "specialist_referral",
                "prescription_request",
                "test_results",
                "emergency",
            ],
            "legal": [
                "contract_review",
                "compliance_check",
                "litigation_support",
                "regulatory_filing",
                "general_inquiry",
            ],
            "educational": [
                "correct",
                "incorrect",
                "partially_correct",
                "needs_review",
            ],
            "technical": [
                "bug",
                "feature_request",
                "documentation",
                "performance",
                "security",
            ],
            "financial": ["investment", "loan", "accounting", "tax", "general"],
            "business": ["strategy", "operations", "marketing", "hr", "finance"],
            "general": ["category_a", "category_b", "category_c", "other"],
        }

        return domain_categories.get(domain, domain_categories["general"])

    def _get_structured_problem_type(
        self, problem_type: str, problem_spec: Dict[str, Any]
    ) -> Optional[ProblemType]:
        """Get structured problem type instance if available."""
        if ProblemType is None:
            return None

        # Get the problem type class
        problem_class = self.structured_problem_map.get(problem_type)
        if not problem_class:
            return None

        # Check cache first
        cache_key = f"{problem_type}_{problem_spec.get('domain', 'general')}"
        if cache_key in self._problem_type_cache:
            return self._problem_type_cache[cache_key]

        try:
            # Create problem type instance based on type and spec
            if problem_type == "classification":
                # Determine number of classes from domain
                domain_classes = self._get_domain_categories(
                    problem_spec.get("domain", "general")
                )
                problem = problem_class(
                    num_classes=len(domain_classes),
                    class_names=domain_classes,
                    multi_label=False,  # Could be enhanced based on spec
                )
            elif problem_type == "generation":
                problem = problem_class(
                    min_length=10, max_length=200, constrained=False
                )
            elif problem_type == "information_extraction":
                problem = problem_class(
                    extraction_type=(
                        "entities"
                        if "entity" in problem_spec.get("description", "")
                        else "slots"
                    )
                )
            elif problem_type == "question_answering":
                problem = problem_class(qa_type="open", with_context=True)
            elif problem_type == "summarization":
                problem = problem_class(summary_type="abstractive", max_length=100)
            elif problem_type == "ranking_retrieval":
                problem = problem_class(task_type="ranking", top_k=5)
            elif problem_type == "translation_transformation":
                problem = problem_class(
                    transformation_type="style_transfer",
                    source_format="informal",
                    target_format="formal",
                )
            elif problem_type == "reasoning":
                problem = problem_class(reasoning_type="logical", requires_steps=True)
            elif problem_type == "code_generation":
                # Detect if it's SQL based on description
                if any(
                    word in problem_spec.get("description", "").lower()
                    for word in ["sql", "query", "database"]
                ):
                    problem = problem_class(target_language="sql", code_type="query")
                else:
                    problem = problem_class(
                        target_language="python", code_type="function"
                    )
            else:
                # Try to get from registry
                problem = get_problem_type(problem_type)

            self._problem_type_cache[cache_key] = problem
            return problem

        except Exception as e:
            print(f"Could not create structured problem type: {e}")
            return None

    async def _generate_contextual_examples(
        self,
        problem_spec: Dict[str, Any],
        target_examples: int,
        structured_problem: Optional[ProblemType] = None,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> List[Dict[str, Any]]:
        """Generate contextual examples using constrained generators or Claude Code SDK."""

        # Use constrained generators if available
        if structured_problem and get_constrained_generator:
            return await self._generate_constrained_examples(
                problem_spec, target_examples, structured_problem, progress_callback
            )

        # Fallback to original implementation
        # Use the full target_examples count
        sample_size = target_examples

        # Use new prompt templates for better generation
        problem_type = problem_spec.get("problem_type", "general")
        domain = problem_spec.get("domain", "general")
        description = problem_spec.get("description", "")

        # Initialize validator
        validator = ExampleValidator()

        # Retry logic variables
        max_retries = 3
        retry_count = 0
        all_valid_examples = []

        while retry_count < max_retries and len(all_valid_examples) < sample_size:
            # Calculate how many more examples we need
            needed_examples = sample_size - len(all_valid_examples)

            # Get appropriate prompt based on problem type
            if problem_type == "classification" and structured_problem:
                categories = getattr(structured_problem, "class_names", None)
                examples_prompt = PromptTemplates.get_classification_prompt(
                    description, needed_examples, domain, categories
                )
            elif problem_type == "reasoning":
                reasoning_type = (
                    getattr(structured_problem, "reasoning_type", "general")
                    if structured_problem
                    else "general"
                )
                examples_prompt = PromptTemplates.get_reasoning_prompt(
                    description, needed_examples, domain, reasoning_type
                )
            else:
                # Use type-specific prompt
                examples_prompt = PromptTemplates.get_prompt_for_type(
                    problem_type, description, needed_examples, domain
                )

            # Add retry-specific instructions if this is a retry
            if retry_count > 0:
                examples_prompt += "\n\nIMPORTANT: Previous generation attempt produced insufficient valid examples. Please ensure:\n"
                examples_prompt += (
                    "1. Each example has complete input and output data\n"
                )
                examples_prompt += (
                    "2. Output is NOT 'To be determined' or placeholder text\n"
                )
                examples_prompt += "3. Examples are properly formatted as JSON\n"
                examples_prompt += (
                    f"4. Generate exactly {needed_examples} valid examples\n"
                )

            try:
                messages = []
                async for message in query(
                    prompt=examples_prompt,
                    options=ClaudeCodeOptions(
                        system_prompt="You are an expert at creating diverse, high-quality examples for LLM training tasks. Generate contextually appropriate, realistic examples in valid JSON format. Never use placeholders like 'To be determined' - always provide complete, valid outputs.",
                        max_turns=1,
                    ),
                ):
                    messages.append(message)

                # Extract text using helper function
                response_text = extract_text_from_claude_messages(messages)

                # Parse examples - now expecting JSON format
                examples = []
                if response_text.strip():
                    # First try to parse as JSON objects
                    try:
                        # Improved JSON extraction with multiple strategies
                        json_objects = self._extract_json_examples(response_text)

                        example_count = len(all_valid_examples)
                        for json_obj in json_objects:
                            if len(examples) >= needed_examples:
                                break

                            try:
                                # Extract input and output
                                input_data = json_obj.get("input", {})
                                output_data = json_obj.get("output", "")

                                # Ensure input_data is properly formatted
                                if isinstance(input_data, str):
                                    input_data = {"text": input_data}
                                elif not isinstance(input_data, dict):
                                    input_data = {"text": str(input_data)}

                                # Create the example
                                example = {
                                    "id": example_count + len(examples) + 1,
                                    "input_data": input_data,
                                    "expected_output": output_data,
                                    "difficulty": "medium",
                                    "reasoning": f'Example for {problem_spec.get("problem_type", "problem")} task',
                                    "metadata": {
                                        "category": problem_spec.get(
                                            "domain", "general"
                                        ),
                                        "pattern": f"example_{example_count + len(examples) + 1}",
                                        "domain_specific_info": f"Generated for {problem_spec.get('description', 'problem')}",
                                    },
                                }

                                # Validate the example
                                is_valid, issues = validator.validate_example(
                                    example, problem_spec.get("problem_type", "general")
                                )

                                if is_valid:
                                    examples.append(example)
                                else:
                                    print(
                                        f"⚠️ Skipping invalid example: {', '.join(issues)}"
                                    )
                            except Exception as e:
                                print(f"Error processing example: {e}")
                                continue

                    except Exception as e:
                        print(f"JSON parsing failed: {e}")

                    # Add valid examples to our collection
                    all_valid_examples.extend(examples)

                    if len(all_valid_examples) >= sample_size:
                        print(
                            f"✅ Successfully generated {len(all_valid_examples)} valid examples"
                        )
                        return all_valid_examples[:sample_size]
                    elif len(examples) > 0:
                        print(
                            f"Generated {len(examples)} valid examples in attempt {retry_count + 1}, total: {len(all_valid_examples)}/{sample_size}"
                        )
                    else:
                        print(
                            f"⚠️ No valid examples generated in attempt {retry_count + 1}"
                        )

            except Exception as e:
                print(f"Error in example generation attempt {retry_count + 1}: {e}")

            retry_count += 1

            # Brief pause before retry
            if retry_count < max_retries and len(all_valid_examples) < sample_size:
                print(f"Retrying... (attempt {retry_count + 1}/{max_retries})")
                await asyncio.sleep(1)

        # If we still don't have enough examples after retries, use fallback
        if len(all_valid_examples) < sample_size:
            print(
                f"⚠️ Only generated {len(all_valid_examples)}/{sample_size} valid examples after {max_retries} attempts"
            )
            print("Using fallback examples to fill the gap")

            # Generate fallback examples for the remaining count
            fallback_count = sample_size - len(all_valid_examples)
            fallback_examples = self._fallback_contextual_examples(
                problem_spec, fallback_count
            )

            # Adjust IDs for fallback examples
            for i, example in enumerate(fallback_examples):
                example["id"] = len(all_valid_examples) + i + 1

            all_valid_examples.extend(fallback_examples)

        return all_valid_examples[:sample_size]

    def _parse_examples_alternative(
        self, response_text: str, problem_spec: Dict[str, Any], sample_size: int
    ) -> List[Dict[str, Any]]:
        """Alternative parsing for examples when JSON fails."""
        try:
            examples = []

            # Look for example-like patterns in the text
            lines = response_text.split("\n")
            current_example = {}
            example_count = 0

            for line in lines:
                line = line.strip()
                if not line:
                    continue

                # Look for input patterns
                if any(
                    keyword in line.lower()
                    for keyword in ["input:", "question:", "text:", "problem:"]
                ):
                    if current_example and "input_data" in current_example:
                        # Save previous example
                        if example_count < sample_size:
                            examples.append(current_example)
                            example_count += 1
                    # Start new example
                    input_text = re.sub(r"^[^:]*:", "", line).strip()
                    current_example = {
                        "id": example_count + 1,
                        "input_data": {"text": input_text},
                        "expected_output": "",
                        "difficulty": "medium",
                        "reasoning": "Generated from response text",
                        "metadata": {
                            "category": problem_spec.get("domain", "general"),
                            "pattern": "extracted_from_text",
                            "domain_specific_info": f"Extracted example for {problem_spec.get('problem_type', 'problem')}",
                        },
                    }

                # Look for output patterns
                elif any(
                    keyword in line.lower()
                    for keyword in ["output:", "answer:", "result:", "response:"]
                ):
                    if current_example:
                        output_text = re.sub(r"^[^:]*:", "", line).strip()
                        current_example["expected_output"] = output_text

            # Add final example if exists
            if (
                current_example
                and "input_data" in current_example
                and example_count < sample_size
            ):
                examples.append(current_example)

            # If we couldn't extract enough examples, generate some based on the text
            while len(examples) < min(sample_size, 3):
                examples.append(
                    {
                        "id": len(examples) + 1,
                        "input_data": {
                            "text": f"Example {len(examples) + 1} for {problem_spec.get('name', 'problem')}"
                        },
                        "expected_output": f"Sample output {len(examples) + 1}",
                        "difficulty": "medium",
                        "reasoning": "Generated example based on problem specification",
                        "metadata": {
                            "category": problem_spec.get("domain", "general"),
                            "pattern": "generated_from_spec",
                            "domain_specific_info": f"Auto-generated for {problem_spec.get('problem_type', 'problem')}",
                        },
                    }
                )

            return examples

        except Exception as e:
            print(f"Alternative examples parsing error: {e}")
            return []

    async def _generate_constrained_examples(
        self,
        problem_spec: Dict[str, Any],
        target_examples: int,
        structured_problem: ProblemType,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> List[Dict[str, Any]]:
        """Generate examples using constrained generators."""
        domain = problem_spec.get("domain", "general")
        difficulty_level = problem_spec.get("difficulty_level", "medium")
        problem_type = problem_spec.get("problem_type", "general")

        # Initialize validator
        validator = ExampleValidator()

        # Get appropriate generator
        generator_kwargs = {
            "context": {
                "problem_description": problem_spec.get("description", ""),
                "user_description": problem_spec.get(
                    "original_description", problem_spec.get("description", "")
                ),
                "domain": domain,
                "difficulty": difficulty_level,
            }
        }

        # Add specific kwargs for classification
        if isinstance(structured_problem, ClassificationProblem):
            generator_kwargs["categories"] = structured_problem.class_names

        # Get the constrained generator
        generator = get_constrained_generator(structured_problem, **generator_kwargs)

        # Generate examples in batches for progress updates
        valid_examples = []
        batch_size = 10
        total_batches = (target_examples + batch_size - 1) // batch_size
        max_retries = 2

        for batch_idx in range(total_batches):
            batch_start = len(valid_examples)
            batch_end = min(batch_start + batch_size, target_examples)
            batch_count = batch_end - batch_start

            if batch_count <= 0:
                break

            if progress_callback:
                progress = 60 + (batch_idx / total_batches) * 30  # 60-90%
                progress_callback(
                    progress, f"Generating examples {batch_start + 1}-{batch_end}..."
                )

            # Generate batch with retry logic
            batch_attempts = 0
            while batch_attempts < max_retries and len(valid_examples) < batch_end:
                # Generate batch
                constrained_examples = generator.generate_batch(
                    count=batch_count, domain=domain, difficulty=difficulty_level
                )

                # Convert and validate
                for i, constrained_example in enumerate(constrained_examples):
                    if len(valid_examples) >= target_examples:
                        break

                    example_dict = constrained_example.to_dict()

                    # Ensure proper format
                    example = {
                        "id": len(valid_examples) + 1,
                        "input_data": example_dict.get("input_data", {"text": "input"}),
                        "expected_output": example_dict.get(
                            "expected_output", "output"
                        ),
                        "difficulty": difficulty_level,
                        "reasoning": f"Generated using {structured_problem.__class__.__name__} constraints",
                        "metadata": example_dict.get("metadata", {}),
                    }

                    # Add domain info to metadata
                    example["metadata"]["category"] = domain
                    example["metadata"]["problem_type"] = problem_type

                    # Validate the example
                    is_valid, issues = validator.validate_example(example, problem_type)

                    if is_valid:
                        valid_examples.append(example)
                    else:
                        print(
                            f"⚠️ Skipping invalid constrained example: {', '.join(issues)}"
                        )

                batch_attempts += 1

                # Check if we need more examples
                if len(valid_examples) >= batch_end:
                    break
                else:
                    print(
                        f"Generated {len(valid_examples)}/{batch_end} valid examples, retrying batch..."
                    )

        # If we don't have enough valid examples, use fallback
        if len(valid_examples) < target_examples:
            print(
                f"⚠️ Only generated {len(valid_examples)}/{target_examples} valid constrained examples"
            )
            print("Using fallback examples to fill the gap")

            # Generate fallback examples for the remaining count
            fallback_count = target_examples - len(valid_examples)
            fallback_examples = self._fallback_contextual_examples(
                problem_spec, fallback_count
            )

            # Adjust IDs for fallback examples
            for i, example in enumerate(fallback_examples):
                example["id"] = len(valid_examples) + i + 1

            valid_examples.extend(fallback_examples)

        return valid_examples[:target_examples]

    def _fallback_intent_analysis(self, description: str) -> Dict[str, Any]:
        """Fallback intent analysis when Claude Code SDK is not available."""
        # Simple keyword-based analysis
        description_lower = description.lower()

        # Enhanced domain detection with specific keywords
        domain = "general"
        domain_keywords = {
            "educational": [
                "school",
                "student",
                "grade",
                "education",
                "learning",
                "teaching",
                "homework",
                "math",
                "science",
                "history",
                "geography",
                "elementary",
                "high school",
                "university",
                "college",
                "academic",
            ],
            "customer_service": [
                "customer",
                "support",
                "ticket",
                "service",
                "help",
                "complaint",
                "billing",
                "account",
                "issue",
                "inquiry",
            ],
            "legal": [
                "legal",
                "law",
                "contract",
                "court",
                "lawyer",
                "attorney",
                "compliance",
                "regulation",
                "statute",
                "litigation",
            ],
            "medical": [
                "medical",
                "health",
                "doctor",
                "patient",
                "diagnosis",
                "treatment",
                "symptom",
                "medicine",
                "clinical",
                "hospital",
            ],
            "technical": [
                "code",
                "software",
                "programming",
                "debug",
                "system",
                "technical",
                "computer",
                "technology",
                "development",
                "engineering",
            ],
            "financial": [
                "financial",
                "money",
                "bank",
                "investment",
                "budget",
                "accounting",
                "finance",
                "loan",
                "credit",
                "payment",
            ],
            "business": [
                "business",
                "company",
                "management",
                "marketing",
                "sales",
                "strategy",
                "operations",
                "corporate",
                "enterprise",
            ],
        }

        # Find the domain with the most keyword matches
        best_domain = "general"
        best_score = 0

        for dom, keywords in domain_keywords.items():
            score = sum(1 for keyword in keywords if keyword in description_lower)
            if score > best_score:
                best_score = score
                best_domain = dom

        # Also check the original domain expertise mapping
        for dom, subdomains in self.domain_expertise.items():
            if dom in description_lower or any(
                sub in description_lower for subs in subdomains.values() for sub in subs
            ):
                if best_score == 0:  # Only use if no keyword matches found
                    best_domain = dom
                break

        domain = best_domain

        # Enhanced problem type detection
        problem_type = "classification"

        # Check for code-related tasks first (more specific)
        if any(
            word in description_lower
            for word in ["code", "coding", "program", "function", "algorithm", "script"]
        ):
            problem_type = "code_generation"
        elif any(
            word in description_lower
            for word in ["question", "answer", "solve", "problem", "quiz", "test"]
        ):
            problem_type = "question_answering"
        elif any(
            word in description_lower
            for word in ["generate", "create", "write", "compose", "produce"]
        ):
            problem_type = "generation"
        elif any(
            word in description_lower
            for word in [
                "analyze",
                "analysis",
                "review",
                "evaluate",
                "assess",
                "examine",
            ]
        ):
            problem_type = "analysis"
        elif any(
            word in description_lower
            for word in ["classify", "categorize", "sort", "group", "label"]
        ):
            problem_type = "classification"
        elif any(
            word in description_lower
            for word in ["extract", "find", "identify", "locate", "detect"]
        ):
            problem_type = "extraction"
        else:
            # Check original mapping
            for action, ptype in self.problem_type_mapping.items():
                if action in description_lower:
                    problem_type = ptype
                    break

        return {
            "intent": f"Create {problem_type} problem for {domain} domain",
            "domain": domain,
            "subdomain": "",
            "problem_type": problem_type,
            "key_entities": description.split()[:5],
            "difficulty_indicators": ["medium"],
            "context_clues": [description],
            "example_requirements": [f"{domain} {problem_type} examples"],
            "confidence": 0.6,
            "reasoning": "Fallback analysis based on keyword matching",
        }

    def _fallback_problem_specification(
        self, description: str, intent_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Fallback problem specification when Claude Code SDK is not available."""
        domain = intent_analysis.get("domain", "general")
        problem_type = intent_analysis.get("problem_type", "classification")

        return {
            "name": f"{domain}_{problem_type}_problem",
            "description": f"A {problem_type} problem for {domain} domain: {description}",
            "domain": domain,
            "problem_type": problem_type,
            "difficulty_level": "medium",
            "example_patterns": [
                f"Basic {problem_type} examples",
                f"Intermediate {problem_type} examples",
                f"Advanced {problem_type} examples",
            ],
            "evaluation_metrics": ["accuracy", "precision", "recall"],
            "technical_requirements": [f"{problem_type}_capability"],
            "input_structure": {"text": "input text"},
            "output_structure": {"result": "output result"},
        }

    def _extract_json_examples(self, text: str) -> List[Dict[str, Any]]:
        """Extract JSON examples from text with multiple strategies."""
        json_objects = []

        # Strategy 1: Try to fix common JSON formatting issues
        cleaned_text = self._clean_json_text(text)

        # Strategy 2: Use json.JSONDecoder for streaming parse
        decoder = json.JSONDecoder()
        idx = 0
        while idx < len(cleaned_text):
            cleaned_text = cleaned_text[idx:].lstrip()
            if not cleaned_text:
                break
            try:
                obj, end_idx = decoder.raw_decode(cleaned_text)
                if isinstance(obj, dict) and "input" in obj and "output" in obj:
                    json_objects.append(obj)
                idx += end_idx
            except json.JSONDecodeError:
                # Move forward and try again
                idx += 1

        # Strategy 3: Fallback to regex if decoder fails
        if not json_objects:
            # Look for JSON objects with better regex
            json_pattern = r"\{[^{}]*(?:\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}[^{}]*)*\}"
            potential_jsons = re.findall(json_pattern, cleaned_text, re.DOTALL)

            for json_str in potential_jsons:
                try:
                    obj = json.loads(json_str)
                    if isinstance(obj, dict) and "input" in obj and "output" in obj:
                        json_objects.append(obj)
                except json.JSONDecodeError:
                    continue

        return json_objects

    def _clean_json_text(self, text: str) -> str:
        """Clean text to fix common JSON formatting issues."""
        # Remove markdown code blocks
        text = re.sub(r"```(?:json)?\s*\n?", "", text)
        text = re.sub(r"```\s*$", "", text)

        # Fix single quotes to double quotes (carefully)
        # Only replace single quotes that are likely string delimiters
        text = re.sub(r"(?<=[{\s,])\'([^']+)\'(?=[:},\s])", r'"\1"', text)

        # Remove trailing commas before closing braces/brackets
        text = re.sub(r",\s*}", "}", text)
        text = re.sub(r",\s*]", "]", text)

        # Fix unquoted keys (simple cases)
        text = re.sub(r"(?<=[{\s,])(\w+)(?=\s*:)", r'"\1"', text)

        # Remove any text before first { or after last }
        first_brace = text.find("{")
        last_brace = text.rfind("}")
        if first_brace >= 0 and last_brace > first_brace:
            text = text[first_brace : last_brace + 1]

        return text.strip()

    def _fallback_contextual_examples(
        self, problem_spec: Dict[str, Any], sample_size: int
    ) -> List[Dict[str, Any]]:
        """Generate meaningful fallback examples when LLM generation fails."""
        examples = []
        problem_type = problem_spec.get("problem_type", "general")
        domain = problem_spec.get("domain", "general")

        # High-quality type-specific fallback templates
        fallback_templates = self._get_fallback_templates(problem_type, domain)

        # Generate examples from templates
        for i in range(min(sample_size, len(fallback_templates))):
            template = fallback_templates[i % len(fallback_templates)]
            example_id = i + 1

            # Create example from template
            example_data = self._instantiate_template(template, example_id, domain)

            # Build the example
            example = {
                "id": example_id,
                "input_data": example_data["input"],
                "expected_output": example_data["output"],
                "difficulty": example_data.get("difficulty", "medium"),
                "reasoning": example_data.get(
                    "reasoning", f"High-quality fallback example for {problem_type}"
                ),
                "metadata": {
                    "category": domain,
                    "pattern": f"fallback_{problem_type}_{example_id}",
                    "domain_specific_info": f"Template-based example for {problem_spec.get('description', problem_type)}",
                    "is_fallback": True,
                    "template_quality": "high",
                },
            }
            examples.append(example)

        # Log that we're using fallback examples
        print(
            f"⚠️ Using {len(examples)} high-quality fallback examples for {problem_type} problem in {domain} domain"
        )

        return examples

    def _get_fallback_templates(
        self, problem_type: str, domain: str
    ) -> List[Dict[str, Any]]:
        """Get high-quality fallback templates for a problem type."""
        templates = {
            "classification": [
                {
                    "input": {
                        "text": "The new smartphone has excellent battery life and a stunning display, but the camera quality is disappointing."
                    },
                    "output": "mixed_review",
                    "difficulty": "easy",
                },
                {
                    "input": {
                        "text": "I need to cancel my subscription immediately. This service is not working as advertised."
                    },
                    "output": "cancellation_request",
                    "difficulty": "easy",
                },
                {
                    "input": {
                        "text": "Can you help me understand the new features in the latest software update?"
                    },
                    "output": "feature_inquiry",
                    "difficulty": "easy",
                },
                {
                    "input": {
                        "text": "The package was damaged during shipping and I need a replacement sent urgently."
                    },
                    "output": "shipping_issue",
                    "difficulty": "medium",
                },
                {
                    "input": {
                        "text": "Your team provided exceptional service. I'm extremely satisfied with the outcome."
                    },
                    "output": "positive_feedback",
                    "difficulty": "easy",
                },
            ],
            "question_answering": [
                {
                    "input": {
                        "question": "What is machine learning?",
                        "context": "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed. It uses algorithms to analyze data, identify patterns, and make decisions with minimal human intervention.",
                    },
                    "output": "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed.",
                    "difficulty": "easy",
                },
                {
                    "input": {
                        "question": "When was the company founded?",
                        "context": "TechCorp was established in 2010 by a group of engineers in Silicon Valley. The company started with just 5 employees and has now grown to over 5000 employees worldwide.",
                    },
                    "output": "The company was founded in 2010.",
                    "difficulty": "easy",
                },
                {
                    "input": {
                        "question": "What are the main benefits mentioned?",
                        "context": "Our new product offers three key benefits: improved efficiency by reducing processing time by 50%, enhanced security with military-grade encryption, and cost savings of up to 30% compared to traditional solutions.",
                    },
                    "output": "The main benefits are improved efficiency (50% faster processing), enhanced security with military-grade encryption, and cost savings of up to 30%.",
                    "difficulty": "medium",
                },
            ],
            "reasoning": [
                {
                    "input": {
                        "problem": "If all birds can fly, and penguins are birds, can penguins fly?"
                    },
                    "output": {
                        "steps": [
                            "The premise states 'all birds can fly'",
                            "Penguins are classified as birds",
                            "However, the premise is false - not all birds can fly",
                            "Penguins are an example of flightless birds",
                        ],
                        "answer": "No, penguins cannot fly despite being birds",
                        "explanation": "The initial premise is incorrect; not all birds can fly, and penguins are flightless birds.",
                    },
                    "difficulty": "medium",
                },
                {
                    "input": {
                        "problem": "A car travels 120 miles in 2 hours. What is its average speed?"
                    },
                    "output": {
                        "steps": [
                            "Average speed = Total distance / Total time",
                            "Total distance = 120 miles",
                            "Total time = 2 hours",
                            "Average speed = 120 miles / 2 hours = 60 mph",
                        ],
                        "answer": "60 mph",
                        "explanation": "The average speed is calculated by dividing the total distance by the total time.",
                    },
                    "difficulty": "easy",
                },
            ],
            "generation": [
                {
                    "input": {
                        "text": "Write a brief product description for a reusable water bottle"
                    },
                    "output": "Stay hydrated in style with our eco-friendly reusable water bottle. Made from durable stainless steel, this 24oz bottle keeps drinks cold for 24 hours or hot for 12. The leak-proof design and wide mouth make it perfect for any adventure, while the sleek design fits in most cup holders. Available in multiple colors to match your personal style.",
                    "difficulty": "medium",
                },
                {
                    "input": {"text": "Create a welcome message for new users"},
                    "output": "Welcome to our platform! We're excited to have you join our community. To get started, take a moment to complete your profile and explore our features. If you need any help along the way, our support team is here to assist you. Let's begin your journey!",
                    "difficulty": "easy",
                },
            ],
            "information_extraction": [
                {
                    "input": {
                        "text": "Meeting scheduled for March 15, 2024 at 2:00 PM in Conference Room A. Attendees: John Smith (Manager), Sarah Johnson (Developer), Mike Chen (Designer)."
                    },
                    "output": {
                        "date": "March 15, 2024",
                        "time": "2:00 PM",
                        "location": "Conference Room A",
                        "attendees": [
                            "John Smith (Manager)",
                            "Sarah Johnson (Developer)",
                            "Mike Chen (Designer)",
                        ],
                    },
                    "difficulty": "easy",
                },
                {
                    "input": {
                        "text": "Order #12345 placed by customer@email.com on January 10th. Total: $156.99. Shipping to 123 Main St, New York, NY 10001."
                    },
                    "output": {
                        "order_id": "12345",
                        "email": "customer@email.com",
                        "date": "January 10th",
                        "total": "$156.99",
                        "address": "123 Main St, New York, NY 10001",
                    },
                    "difficulty": "medium",
                },
            ],
            "summarization": [
                {
                    "input": {
                        "text": "The quarterly financial report shows significant growth across all departments. Revenue increased by 25% compared to last quarter, reaching $10 million. The marketing department successfully launched three new campaigns, resulting in a 40% increase in lead generation. Operations improved efficiency by 15% through process automation. The company also hired 50 new employees to support expansion plans."
                    },
                    "output": "Q4 showed 25% revenue growth to $10M, 40% more leads from marketing campaigns, 15% efficiency gains, and 50 new hires.",
                    "difficulty": "medium",
                }
            ],
            "analysis": [
                {
                    "input": {
                        "text": "The stock market showed mixed results today with tech stocks leading gains while energy sector declined. The S&P 500 rose 0.5% while crude oil prices fell 2%."
                    },
                    "output": "Mixed market performance with sector divergence. Tech stocks drove S&P 500 up 0.5%, contrasting with 2% decline in energy sector linked to falling oil prices. Suggests rotation from cyclical to growth stocks.",
                    "difficulty": "medium",
                },
                {
                    "input": {
                        "text": "Customer reviews show 85% satisfaction rate, with main complaints about shipping delays (30%) and product packaging (15%). Positive feedback highlights product quality (60%) and customer service (40%)."
                    },
                    "output": "Overall strong satisfaction at 85%. Key improvement areas: shipping logistics (30% of complaints) and packaging (15%). Strengths include product quality and customer service, representing competitive advantages.",
                    "difficulty": "easy",
                },
            ],
            "extraction": [
                {
                    "input": {
                        "text": "Contact John Smith at john.smith@email.com or call (555) 123-4567. Meeting scheduled for Monday at 3 PM."
                    },
                    "output": {
                        "name": "John Smith",
                        "email": "john.smith@email.com",
                        "phone": "(555) 123-4567",
                        "meeting_day": "Monday",
                        "meeting_time": "3 PM",
                    },
                    "difficulty": "easy",
                }
            ],
            "code_generation": [
                {
                    "input": {"description": "Write a function that reverses a string"},
                    "output": "function reverseString(str) {\n    return str.split('').reverse().join('');\n}",
                    "difficulty": "easy",
                },
                {
                    "input": {
                        "description": "Create a function to check if a number is prime"
                    },
                    "output": "function isPrime(n) {\n    if (n <= 1) return false;\n    if (n <= 3) return true;\n    if (n % 2 === 0 || n % 3 === 0) return false;\n    for (let i = 5; i * i <= n; i += 6) {\n        if (n % i === 0 || n % (i + 2) === 0) return false;\n    }\n    return true;\n}",
                    "difficulty": "medium",
                },
                {
                    "input": {
                        "description": "Write a function to find the factorial of a number"
                    },
                    "output": "function factorial(n) {\n    if (n < 0) return undefined;\n    if (n === 0 || n === 1) return 1;\n    let result = 1;\n    for (let i = 2; i <= n; i++) {\n        result *= i;\n    }\n    return result;\n}",
                    "difficulty": "easy",
                },
            ],
        }

        # Get templates for the problem type, or use generic if not found
        type_templates = templates.get(problem_type, [])

        # If no templates for this type, create generic ones
        if not type_templates:
            type_templates = [
                {
                    "input": {
                        "text": f"Sample input for {problem_type} in {domain} domain"
                    },
                    "output": f"Sample output for {problem_type}",
                    "difficulty": "medium",
                }
            ] * 3

        return type_templates

    def _instantiate_template(
        self, template: Dict[str, Any], example_id: int, domain: str
    ) -> Dict[str, Any]:
        """Instantiate a template with specific values."""
        # Deep copy the template
        import copy

        instance = copy.deepcopy(template)

        # Add any domain-specific modifications if needed
        # For now, just return the template as-is
        return instance

    def validate_specification(
        self, spec: SmartProblemSpecification
    ) -> Tuple[bool, List[str]]:
        """
        Validate a generated problem specification.

        Args:
            spec: Generated problem specification

        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []

        # Check required fields
        if not spec.name or len(spec.name) < 3:
            issues.append("Problem name is too short or missing")

        if not spec.description or len(spec.description) < 10:
            issues.append("Problem description is too short or missing")

        if spec.domain not in self.domain_expertise and spec.domain != "general":
            issues.append(f"Unknown domain: {spec.domain}")

        if not spec.contextual_examples:
            issues.append("No contextual examples generated")

        if spec.confidence_score < 0.5:
            issues.append("Low confidence in analysis - may need manual review")

        # Check example quality
        if spec.contextual_examples:
            for i, example in enumerate(spec.contextual_examples[:3]):  # Check first 3
                if not example.get("input_data"):
                    issues.append(f"Example {i+1} missing input_data")
                if not example.get("expected_output"):
                    issues.append(f"Example {i+1} missing expected_output")

        return len(issues) == 0, issues
