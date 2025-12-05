"""
Constrained Example Generators for TraiGent SDK.

This module provides type-specific example generators that follow strict
templates and constraints for each problem type. This ensures consistency
and validity of generated examples.
"""

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .problem_types import (
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
)


@dataclass
class ConstrainedExample:
    """A constrained example following problem type template."""

    id: int
    input_data: Dict[str, Any]
    expected_output: Any
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "id": self.id,
            "input_data": self.input_data,
            "expected_output": self.expected_output,
            "metadata": self.metadata,
        }


class ConstrainedExampleGenerator(ABC):
    """Abstract base class for constrained example generators."""

    def __init__(self, problem_type: ProblemType):
        """Initialize with problem type."""
        self.problem_type = problem_type
        self.input_constraints = problem_type.get_input_constraints()
        self.output_constraints = problem_type.get_output_constraints()

    @abstractmethod
    def generate_example(
        self,
        example_id: int,
        domain: str,
        difficulty: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> ConstrainedExample:
        """Generate a single constrained example."""
        pass

    @abstractmethod
    def generate_batch(
        self,
        count: int,
        domain: str,
        difficulty: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[ConstrainedExample]:
        """Generate a batch of constrained examples."""
        pass

    def validate_example(self, example: ConstrainedExample) -> Tuple[bool, List[str]]:
        """Validate an example against problem type constraints."""
        issues = []

        # Validate input
        if not self.input_constraints.validate(example.input_data):
            issues.append("Input data fails validation constraints")

        # Validate output
        if not self.output_constraints.validate(example.expected_output):
            issues.append("Expected output fails validation constraints")

        # Check required fields
        for field in self.input_constraints.required_fields:
            if field not in example.input_data:
                issues.append(f"Missing required input field: {field}")

        return len(issues) == 0, issues


class ClassificationExampleGenerator(ConstrainedExampleGenerator):
    """Generator for classification problem examples."""

    def __init__(
        self,
        problem_type: ClassificationProblem,
        categories: List[str],
        context: Optional[Dict[str, Any]] = None,
    ):
        """Initialize with classification problem and categories."""
        # Update problem type with provided categories
        problem_type.class_names = categories
        super().__init__(problem_type)
        self.categories = categories
        self.problem = problem_type
        self.context = context or {}

        # Domain-specific templates
        self.templates = {
            "customer_service": {
                "easy": [
                    ("My order hasn't arrived yet", "shipping_inquiry"),
                    ("I want to return this item", "return_request"),
                    ("How do I reset my password?", "account_support"),
                ],
                "medium": [
                    ("I was charged twice for my order", "billing_issue"),
                    ("The product page won't load properly", "technical_support"),
                    ("Can I change my delivery address?", "order_modification"),
                ],
                "hard": [
                    (
                        "The app crashes when I try to apply my discount code at checkout",
                        "technical_support",
                    ),
                    (
                        "I ordered last week but now the price dropped, can you help?",
                        "billing_issue",
                    ),
                ],
            },
            "medical": {
                "easy": [
                    ("Annual checkup appointment needed", "routine_checkup"),
                    ("Prescription refill request", "prescription_request"),
                    ("Need flu shot appointment", "routine_checkup"),
                ],
                "medium": [
                    ("Persistent headaches for two weeks", "specialist_referral"),
                    ("Blood test results need review", "test_results"),
                    ("Medication causing side effects", "prescription_request"),
                ],
                "hard": [
                    (
                        "Multiple symptoms: fatigue, joint pain, cognitive issues",
                        "specialist_referral",
                    ),
                    ("Severe chest pain and shortness of breath", "emergency"),
                ],
            },
            "educational": {
                "easy": [
                    ("How do plants make food?", "science_concept"),
                    ("What is 5 + 3?", "math_problem"),
                    ("Name the capital of France", "geography_fact"),
                ],
                "medium": [
                    ("Explain why the sky is blue", "science_explanation"),
                    (
                        "If a train travels 60 miles in 2 hours, what is its speed?",
                        "math_word_problem",
                    ),
                    ("What caused the American Revolution?", "history_analysis"),
                ],
                "hard": [
                    (
                        "Compare and contrast photosynthesis and cellular respiration",
                        "science_comparison",
                    ),
                    (
                        "A store offers 25% off, then an additional 10% off. What's the total discount?",
                        "complex_math",
                    ),
                    (
                        "Analyze the impact of the Industrial Revolution on society",
                        "historical_analysis",
                    ),
                ],
            },
        }

    def generate_example(
        self,
        example_id: int,
        domain: str,
        difficulty: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> ConstrainedExample:
        """Generate a single classification example."""

        # Merge with instance context if provided
        full_context = {**self.context, **(context or {})}

        # Get domain templates or use context-aware generation
        domain_templates = self.templates.get(domain, {})
        difficulty_templates = domain_templates.get(difficulty, [])

        if difficulty_templates:
            text, category = random.choice(difficulty_templates)
            # Ensure category is in allowed categories
            if category not in self.categories:
                category = random.choice(self.categories)
        else:
            # Context-aware generation for generic/general domain
            if full_context.get("problem_description"):
                # Generate based on problem description
                text, category = self._generate_from_context(
                    full_context["problem_description"], self.categories, difficulty
                )
            else:
                # Improved generic template
                category = random.choice(self.categories)
                text = self._generate_generic_text(domain, category, difficulty)

        # Create constrained example
        example = ConstrainedExample(
            id=example_id,
            input_data={"text": text},
            expected_output=category,
            metadata={
                "domain": domain,
                "difficulty": difficulty,
                "category": category,
                "problem_type": "classification",
            },
        )

        # Validate
        is_valid, issues = self.validate_example(example)
        if not is_valid:
            print(f"Warning: Generated invalid example: {issues}")

        return example

    def _generate_from_context(
        self, description: str, categories: List[str], difficulty: str
    ) -> Tuple[str, str]:
        """Generate example from problem description."""
        # For "how stuff works" type problems
        if "how" in description.lower() and "work" in description.lower():
            examples = {
                "easy": [
                    (
                        "How does a bicycle work?",
                        categories[0] if len(categories) > 0 else "mechanism",
                    ),
                    (
                        "How does photosynthesis work?",
                        categories[1] if len(categories) > 1 else "biological_process",
                    ),
                    (
                        "How does email work?",
                        categories[2] if len(categories) > 2 else "technology",
                    ),
                ],
                "medium": [
                    (
                        "How does a combustion engine convert fuel into motion?",
                        categories[0] if len(categories) > 0 else "mechanism",
                    ),
                    (
                        "How does the immune system identify and destroy pathogens?",
                        categories[1] if len(categories) > 1 else "biological_process",
                    ),
                    (
                        "How does encryption protect data during transmission?",
                        categories[2] if len(categories) > 2 else "technology",
                    ),
                ],
                "hard": [
                    (
                        "How does quantum entanglement enable secure communication?",
                        categories[0] if len(categories) > 0 else "physics",
                    ),
                    (
                        "How does CRISPR-Cas9 precisely edit genetic sequences?",
                        categories[1] if len(categories) > 1 else "biotechnology",
                    ),
                    (
                        "How does machine learning optimize neural network architectures?",
                        categories[2] if len(categories) > 2 else "ai_technology",
                    ),
                ],
            }
            return random.choice(examples.get(difficulty, examples["medium"]))

        # For weight loss problems
        elif "weight" in description.lower() or "lose" in description.lower():
            examples = {
                "easy": [
                    (
                        "What's the best diet for weight loss?",
                        categories[0] if len(categories) > 0 else "diet_advice",
                    ),
                    (
                        "How many calories should I eat to lose weight?",
                        categories[1] if len(categories) > 1 else "calorie_guidance",
                    ),
                    (
                        "Is walking good for weight loss?",
                        categories[2] if len(categories) > 2 else "exercise_advice",
                    ),
                ],
                "medium": [
                    (
                        "How can I overcome weight loss plateaus?",
                        categories[0] if len(categories) > 0 else "troubleshooting",
                    ),
                    (
                        "What's the role of metabolism in weight loss?",
                        categories[1] if len(categories) > 1 else "science_explanation",
                    ),
                    (
                        "How do I balance cardio and strength training for weight loss?",
                        categories[2] if len(categories) > 2 else "exercise_planning",
                    ),
                ],
                "hard": [
                    (
                        "How do hormones affect weight loss in women over 40?",
                        categories[0] if len(categories) > 0 else "specialized_advice",
                    ),
                    (
                        "What's the relationship between insulin resistance and weight loss?",
                        categories[1] if len(categories) > 1 else "medical_explanation",
                    ),
                    (
                        "How can I maintain muscle mass during aggressive weight loss?",
                        categories[2] if len(categories) > 2 else "advanced_strategy",
                    ),
                ],
            }
            return random.choice(examples.get(difficulty, examples["medium"]))

        # Default contextual generation
        return self._generate_generic_text(
            "contextual", random.choice(categories), difficulty
        ), random.choice(categories)

    def _generate_generic_text(
        self, domain: str, category: str, difficulty: str
    ) -> str:
        """Generate more meaningful generic text."""
        templates = {
            "easy": [
                f"A simple question about {category} in the {domain} field",
                f"Basic inquiry regarding {category}",
                f"Elementary {domain} question about {category}",
            ],
            "medium": [
                f"A detailed question about {category} requiring {domain} knowledge",
                f"Intermediate analysis of {category} concepts",
                f"Practical application of {category} in {domain} context",
            ],
            "hard": [
                f"Complex {domain} scenario involving {category} analysis",
                f"Advanced {category} problem requiring expert knowledge",
                f"Multi-faceted {domain} challenge related to {category}",
            ],
        }
        return random.choice(templates.get(difficulty, templates["medium"]))

    def generate_batch(
        self,
        count: int,
        domain: str,
        difficulty: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[ConstrainedExample]:
        """Generate a batch of classification examples."""
        examples = []
        for i in range(count):
            example = self.generate_example(i + 1, domain, difficulty, context)
            examples.append(example)
        return examples


class GenerationExampleGenerator(ConstrainedExampleGenerator):
    """Generator for generation problem examples."""

    def __init__(self, problem_type: SequenceGenerationProblem):
        """Initialize with generation problem."""
        super().__init__(problem_type)
        self.problem = problem_type

        # Templates for different domains
        self.templates = {
            "marketing": {
                "easy": [
                    {
                        "prompt": "Write a simple product description for a blue t-shirt",
                        "output": "Classic blue t-shirt made from soft cotton. Comfortable fit for everyday wear. Available in sizes S-XL.",
                    },
                    {
                        "prompt": "Create a short social media post about a sale",
                        "output": "🎉 Flash Sale Alert! Get 20% off everything today only. Shop now and save big! Limited time offer.",
                    },
                ],
                "medium": [
                    {
                        "prompt": "Generate an email subject line for a product launch",
                        "output": "Introducing Our Game-Changing Innovation - Be First to Experience the Future",
                    },
                    {
                        "prompt": "Write a customer testimonial for a fitness app",
                        "output": "This app transformed my fitness journey! The personalized workouts and progress tracking kept me motivated. Lost 15 pounds in 2 months!",
                    },
                ],
                "hard": [
                    {
                        "prompt": "Create a comprehensive marketing strategy outline for a new eco-friendly product line",
                        "output": "1. Target Audience: Environmentally conscious millennials and Gen Z\n2. Key Messages: Sustainability without compromise, carbon-neutral production\n3. Channels: Instagram influencers, eco-blogs, sustainable living podcasts\n4. Campaign Timeline: 3-month pre-launch, 6-month post-launch\n5. KPIs: Brand awareness lift, conversion rate, customer lifetime value",
                    }
                ],
            },
            "technical": {
                "easy": [
                    {
                        "prompt": "Write a simple function description",
                        "output": "This function calculates the sum of two numbers and returns the result.",
                    }
                ],
                "medium": [
                    {
                        "prompt": "Generate API documentation for a user endpoint",
                        "output": "GET /api/users/{id} - Retrieves user information by ID. Returns: {id, name, email, created_at}. Status: 200 OK or 404 Not Found.",
                    }
                ],
                "hard": [
                    {
                        "prompt": "Create a technical design document outline for a microservices migration",
                        "output": "1. Current Architecture Analysis\n2. Service Decomposition Strategy\n3. Data Management and Consistency\n4. Inter-service Communication\n5. Security and Authentication\n6. Deployment and Orchestration\n7. Monitoring and Observability\n8. Migration Phases and Rollback Plan",
                    }
                ],
            },
        }

    def generate_example(
        self,
        example_id: int,
        domain: str,
        difficulty: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> ConstrainedExample:
        """Generate a single generation example."""

        # Get templates
        domain_templates = self.templates.get(domain, {})
        difficulty_templates = domain_templates.get(difficulty, [])

        if difficulty_templates:
            template = random.choice(difficulty_templates)
            prompt = template["prompt"]
            expected_output = template["output"]
        else:
            # Generic template
            prompt = f"Generate {domain} content with {difficulty} complexity"
            expected_output = f"Generated {domain} content appropriate for {difficulty} level with relevant details and proper structure."

        # Add constraints if specified
        constraints = []
        if self.problem.min_length:
            constraints.append(f"minimum {self.problem.min_length} words")
        if self.problem.max_length:
            constraints.append(f"maximum {self.problem.max_length} words")

        input_data = {"prompt": prompt, "constraints": constraints}

        example = ConstrainedExample(
            id=example_id,
            input_data=input_data,
            expected_output=expected_output,
            metadata={
                "domain": domain,
                "difficulty": difficulty,
                "problem_type": "generation",
                "length": len(expected_output.split()),
            },
        )

        return example

    def generate_batch(
        self,
        count: int,
        domain: str,
        difficulty: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[ConstrainedExample]:
        """Generate a batch of generation examples."""
        examples = []
        for i in range(count):
            example = self.generate_example(i + 1, domain, difficulty, context)
            examples.append(example)
        return examples


class InformationExtractionExampleGenerator(ConstrainedExampleGenerator):
    """Generator for information extraction problem examples."""

    def __init__(
        self,
        problem_type: InformationExtractionProblem,
        schema: Optional[Dict[str, Any]] = None,
    ):
        """Initialize with extraction problem and optional schema."""
        super().__init__(problem_type)
        self.problem = problem_type
        self.schema = schema or {}

        self.templates = {
            "business": {
                "easy": [
                    {
                        "text": "John Smith is the CEO of TechCorp, founded in 2020 in San Francisco.",
                        "output": {
                            "entities": [
                                {
                                    "type": "person",
                                    "value": "John Smith",
                                    "role": "CEO",
                                },
                                {"type": "company", "value": "TechCorp"},
                                {"type": "date", "value": "2020"},
                                {"type": "location", "value": "San Francisco"},
                            ]
                        },
                    }
                ],
                "medium": [
                    {
                        "text": "The Q3 2024 revenue was $2.5M, up 15% YoY. Operating expenses totaled $1.8M with EBITDA of $700K.",
                        "output": {
                            "financial_metrics": {
                                "period": "Q3 2024",
                                "revenue": 2500000,
                                "revenue_growth": 0.15,
                                "operating_expenses": 1800000,
                                "ebitda": 700000,
                            }
                        },
                    }
                ],
            },
            "legal": {
                "easy": [
                    {
                        "text": "This Agreement is between ABC Corp (Buyer) and XYZ Ltd (Seller), dated January 1, 2024.",
                        "output": {
                            "parties": [
                                {"name": "ABC Corp", "role": "Buyer"},
                                {"name": "XYZ Ltd", "role": "Seller"},
                            ],
                            "date": "2024-01-01",
                            "document_type": "Agreement",
                        },
                    }
                ]
            },
        }

    def generate_example(
        self,
        example_id: int,
        domain: str,
        difficulty: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> ConstrainedExample:
        """Generate a single extraction example."""

        domain_templates = self.templates.get(domain, {})
        difficulty_templates = domain_templates.get(difficulty, [])

        if difficulty_templates:
            template = random.choice(difficulty_templates)
            text = template["text"]
            expected_output = template["output"]
        else:
            # Generic template
            text = f"Extract information from this {domain} document with {difficulty} complexity."
            expected_output = {
                "entities": [{"type": "generic", "value": f"{domain}_entity"}],
                "metadata": {"difficulty": difficulty},
            }

        input_data = {"text": text, "schema": self.schema}

        example = ConstrainedExample(
            id=example_id,
            input_data=input_data,
            expected_output=expected_output,
            metadata={
                "domain": domain,
                "difficulty": difficulty,
                "problem_type": "information_extraction",
                "extraction_type": self.problem.extraction_type,
            },
        )

        return example

    def generate_batch(
        self,
        count: int,
        domain: str,
        difficulty: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[ConstrainedExample]:
        """Generate a batch of extraction examples."""
        examples = []
        for i in range(count):
            example = self.generate_example(i + 1, domain, difficulty, context)
            examples.append(example)
        return examples


class QuestionAnsweringExampleGenerator(ConstrainedExampleGenerator):
    """Generator for question answering problem examples."""

    def __init__(
        self,
        problem_type: QuestionAnsweringProblem,
        context: Optional[Dict[str, Any]] = None,
    ):
        """Initialize with QA problem."""
        super().__init__(problem_type)
        self.problem = problem_type
        self.context = context or {}

        self.templates = {
            "educational": {
                "easy": [
                    {
                        "question": "What is the capital of France?",
                        "context": "France is a country in Western Europe. Its capital city is Paris, known for the Eiffel Tower.",
                        "answer": "Paris",
                    },
                    {
                        "question": "How many days are in a week?",
                        "context": "A week is a time unit equal to seven days.",
                        "answer": "Seven",
                    },
                ],
                "medium": [
                    {
                        "question": "What causes seasons on Earth?",
                        "context": "Earth's axis is tilted at 23.5 degrees. As Earth orbits the Sun, different parts receive varying amounts of sunlight throughout the year.",
                        "answer": "Seasons are caused by Earth's tilted axis as it orbits the Sun, resulting in varying amounts of sunlight reaching different parts of Earth throughout the year.",
                    }
                ],
            },
            "technical": {
                "easy": [
                    {
                        "question": "What does API stand for?",
                        "context": "APIs are mechanisms that enable two software components to communicate. API stands for Application Programming Interface.",
                        "answer": "Application Programming Interface",
                    }
                ],
                "medium": [
                    {
                        "question": "What is the difference between POST and GET requests?",
                        "context": "HTTP methods include GET and POST. GET requests retrieve data from a server, while POST requests send data to a server to create or update resources.",
                        "answer": "GET requests retrieve data from a server, while POST requests send data to create or update resources on the server.",
                    }
                ],
            },
        }

    def generate_example(
        self,
        example_id: int,
        domain: str,
        difficulty: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> ConstrainedExample:
        """Generate a single QA example."""

        # Merge with instance context
        full_context = {**self.context, **(context or {})}

        domain_templates = self.templates.get(domain, {})
        difficulty_templates = domain_templates.get(difficulty, [])

        if difficulty_templates:
            template = random.choice(difficulty_templates)
            question = template["question"]
            qa_context = (
                template.get("context", "") if self.problem.with_context else ""
            )
            answer = template["answer"]
        else:
            # Context-aware generation for generic/general domain
            if full_context.get("problem_description"):
                # Generate based on problem description
                question, qa_context, answer = self._generate_from_context(
                    full_context["problem_description"], domain, difficulty
                )
            else:
                # Generic template
                question = f"What is a {domain} question with {difficulty} difficulty?"
                qa_context = (
                    f"This is context about {domain}."
                    if self.problem.with_context
                    else ""
                )
                answer = f"This is a {difficulty} answer about {domain}."

        input_data = {"question": question}
        if self.problem.with_context and qa_context:
            input_data["context"] = qa_context

        example = ConstrainedExample(
            id=example_id,
            input_data=input_data,
            expected_output=answer,
            metadata={
                "domain": domain,
                "difficulty": difficulty,
                "problem_type": "question_answering",
                "qa_type": self.problem.qa_type,
                "has_context": bool(qa_context),
            },
        )

        return example

    def _generate_from_context(
        self, description: str, domain: str, difficulty: str
    ) -> Tuple[str, str, str]:
        """Generate QA example from problem description."""
        # For "how stuff works" type problems
        if "how" in description.lower() and "work" in description.lower():
            examples = {
                "easy": [
                    (
                        "How does a bicycle work?",
                        "A bicycle is a two-wheeled vehicle powered by pedaling.",
                        "A bicycle works by converting the rider's pedaling motion into wheel rotation through a chain and gear system.",
                    ),
                    (
                        "How does photosynthesis work?",
                        "Photosynthesis is the process plants use to make food from sunlight.",
                        "Photosynthesis works by using chlorophyll to capture sunlight and convert carbon dioxide and water into glucose and oxygen.",
                    ),
                ],
                "medium": [
                    (
                        "How does a combustion engine work?",
                        "A combustion engine is a type of engine that burns fuel to create motion.",
                        "A combustion engine works through a four-stroke cycle: intake (air/fuel mixture enters), compression (mixture is compressed), combustion (spark ignites mixture), and exhaust (burnt gases expelled).",
                    ),
                    (
                        "How does the internet work?",
                        "The internet is a global network of interconnected computers.",
                        "The internet works by routing data packets between computers using protocols like TCP/IP, with routers directing traffic through the most efficient paths.",
                    ),
                ],
                "hard": [
                    (
                        "How does quantum computing work?",
                        "Quantum computing uses quantum mechanical phenomena to process information.",
                        "Quantum computing works by manipulating quantum bits (qubits) that can exist in superposition states, allowing parallel processing of multiple possibilities simultaneously through quantum gates and entanglement.",
                    ),
                    (
                        "How does CRISPR gene editing work?",
                        "CRISPR is a technology for editing genetic sequences.",
                        "CRISPR works by using a guide RNA to direct the Cas9 enzyme to specific DNA sequences, where it creates precise cuts allowing for deletion, insertion, or modification of genetic material.",
                    ),
                ],
            }
            result = random.choice(examples.get(difficulty, examples["medium"]))
            return result

        # Default generation
        question = f"Explain the key concepts related to: {description}"
        context = (
            f"This relates to {domain} domain concepts."
            if self.problem.with_context
            else ""
        )
        answer = f"The key concepts involve understanding the fundamental aspects of {description} in the {domain} field."
        return question, context, answer

    def generate_batch(
        self,
        count: int,
        domain: str,
        difficulty: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[ConstrainedExample]:
        """Generate a batch of QA examples."""
        examples = []
        for i in range(count):
            example = self.generate_example(i + 1, domain, difficulty, context)
            examples.append(example)
        return examples


class SummarizationExampleGenerator(ConstrainedExampleGenerator):
    """Generator for summarization problem examples."""

    def __init__(self, problem_type: SummarizationProblem):
        """Initialize with summarization problem."""
        super().__init__(problem_type)
        self.problem = problem_type

        self.templates = {
            "news": {
                "easy": [
                    {
                        "document": "The city council met yesterday to discuss the new park project. The project will cost $2 million and create a 10-acre green space in downtown. Construction is expected to begin in June and complete by December. Local residents expressed strong support for the initiative.",
                        "summary": "City council approved a $2 million, 10-acre downtown park project starting in June, with strong community support.",
                    }
                ],
                "medium": [
                    {
                        "document": "Climate scientists released a comprehensive report showing global temperatures have risen 1.1°C since pre-industrial times. The report, compiled by 234 scientists from 66 countries, analyzed data from multiple sources including satellite measurements, weather stations, and ocean buoys. Key findings indicate accelerating ice melt, rising sea levels, and increased frequency of extreme weather events. The scientists warn that without immediate action, temperatures could rise 3°C by 2100, causing catastrophic impacts.",
                        "summary": "A major scientific report by 234 international scientists confirms 1.1°C global warming, with accelerating ice melt and extreme weather. Without immediate action, temperatures may rise 3°C by 2100, causing catastrophic impacts.",
                    }
                ],
            }
        }

    def generate_example(
        self,
        example_id: int,
        domain: str,
        difficulty: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> ConstrainedExample:
        """Generate a single summarization example."""

        domain_templates = self.templates.get(domain, {})
        difficulty_templates = domain_templates.get(difficulty, [])

        if difficulty_templates:
            template = random.choice(difficulty_templates)
            document = template["document"]
            summary = template["summary"]
        else:
            # Generic template
            document = (
                f"This is a {difficulty} {domain} document that needs to be summarized. "
                * 10
            )
            summary = (
                f"Summary of {difficulty} {domain} document with key points preserved."
            )

        input_data = {"document": document}

        example = ConstrainedExample(
            id=example_id,
            input_data=input_data,
            expected_output=summary,
            metadata={
                "domain": domain,
                "difficulty": difficulty,
                "problem_type": "summarization",
                "compression_ratio": len(summary.split()) / len(document.split()),
                "summary_type": self.problem.summary_type,
            },
        )

        return example

    def generate_batch(
        self,
        count: int,
        domain: str,
        difficulty: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[ConstrainedExample]:
        """Generate a batch of summarization examples."""
        examples = []
        for i in range(count):
            example = self.generate_example(i + 1, domain, difficulty, context)
            examples.append(example)
        return examples


class RankingRetrievalExampleGenerator(ConstrainedExampleGenerator):
    """Generator for ranking/retrieval problem examples."""

    def __init__(self, problem_type: RankingRetrievalProblem):
        """Initialize with ranking problem."""
        super().__init__(problem_type)
        self.problem = problem_type

        self.templates = {
            "search": {
                "easy": [
                    {
                        "query": "best pizza restaurants",
                        "candidates": [
                            "Joe's Pizza - Classic NY style",
                            "Pizza Palace - Gourmet toppings",
                            "Quick Slice - Fast delivery",
                            "Mama's Kitchen - Traditional Italian",
                            "The Pizza Box - Budget friendly",
                        ],
                        "ranked": [
                            {
                                "item": "Mama's Kitchen - Traditional Italian",
                                "score": 0.95,
                                "rank": 1,
                            },
                            {
                                "item": "Joe's Pizza - Classic NY style",
                                "score": 0.88,
                                "rank": 2,
                            },
                            {
                                "item": "Pizza Palace - Gourmet toppings",
                                "score": 0.82,
                                "rank": 3,
                            },
                        ],
                    }
                ]
            }
        }

    def generate_example(
        self,
        example_id: int,
        domain: str,
        difficulty: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> ConstrainedExample:
        """Generate a single ranking example."""

        domain_templates = self.templates.get(domain, {})
        difficulty_templates = domain_templates.get(difficulty, [])

        if difficulty_templates:
            template = random.choice(difficulty_templates)
            query = template["query"]
            candidates = template["candidates"]
            ranked = template["ranked"]
        else:
            # Generic template
            query = f"Find {domain} items with {difficulty} relevance"
            candidates = [f"{domain} item {i}" for i in range(1, 6)]
            ranked = [
                {"item": candidates[i], "score": 0.9 - i * 0.1, "rank": i + 1}
                for i in range(min(3, len(candidates)))
            ]

        input_data = {"query": query, "candidates": candidates}

        example = ConstrainedExample(
            id=example_id,
            input_data=input_data,
            expected_output=ranked[: self.problem.top_k],
            metadata={
                "domain": domain,
                "difficulty": difficulty,
                "problem_type": "ranking_retrieval",
                "task_type": self.problem.task_type,
                "num_candidates": len(candidates),
            },
        )

        return example

    def generate_batch(
        self,
        count: int,
        domain: str,
        difficulty: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[ConstrainedExample]:
        """Generate a batch of ranking examples."""
        examples = []
        for i in range(count):
            example = self.generate_example(i + 1, domain, difficulty, context)
            examples.append(example)
        return examples


class TranslationTransformationExampleGenerator(ConstrainedExampleGenerator):
    """Generator for translation/transformation problem examples."""

    def __init__(self, problem_type: TranslationTransformationProblem):
        """Initialize with transformation problem."""
        super().__init__(problem_type)
        self.problem = problem_type

        self.templates = {
            "style_transfer": {
                "easy": [
                    {
                        "text": "The experiment failed.",
                        "target_style": "formal",
                        "output": "The experimental procedure did not yield the anticipated results.",
                    },
                    {
                        "text": "Please send me the report ASAP.",
                        "target_style": "polite",
                        "output": "I would greatly appreciate it if you could send me the report at your earliest convenience.",
                    },
                ]
            }
        }

    def generate_example(
        self,
        example_id: int,
        domain: str,
        difficulty: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> ConstrainedExample:
        """Generate a single transformation example."""

        if self.problem.transformation_type == "style_transfer":
            domain_templates = self.templates.get("style_transfer", {})
        else:
            domain_templates = {}

        difficulty_templates = domain_templates.get(difficulty, [])

        if difficulty_templates:
            template = random.choice(difficulty_templates)
            text = template["text"]
            output = template["output"]
            input_data = {
                "text": text,
                "target_style": template.get(
                    "target_style", self.problem.target_format
                ),
            }
        else:
            # Generic template
            text = f"Transform this {domain} text with {difficulty} complexity"
            output = f"Transformed {domain} text in {self.problem.target_format or 'target'} format"
            input_data = {"text": text}

        if self.problem.transformation_type == "translation":
            input_data["source_lang"] = self.problem.source_format or "English"
            input_data["target_lang"] = self.problem.target_format or "Spanish"

        example = ConstrainedExample(
            id=example_id,
            input_data=input_data,
            expected_output=output,
            metadata={
                "domain": domain,
                "difficulty": difficulty,
                "problem_type": "translation_transformation",
                "transformation_type": self.problem.transformation_type,
            },
        )

        return example

    def generate_batch(
        self,
        count: int,
        domain: str,
        difficulty: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[ConstrainedExample]:
        """Generate a batch of transformation examples."""
        examples = []
        for i in range(count):
            example = self.generate_example(i + 1, domain, difficulty, context)
            examples.append(example)
        return examples


class ReasoningExampleGenerator(ConstrainedExampleGenerator):
    """Generator for reasoning problem examples."""

    def __init__(self, problem_type: ReasoningProblem):
        """Initialize with reasoning problem."""
        super().__init__(problem_type)
        self.problem = problem_type

        self.templates = {
            "mathematical": {
                "easy": [
                    {
                        "problem": "If John has 5 apples and gives 2 to Mary, how many apples does John have left?",
                        "answer": {
                            "steps": [
                                "John starts with 5 apples",
                                "He gives 2 to Mary",
                                "5 - 2 = 3",
                            ],
                            "answer": "3",
                        },
                    }
                ],
                "medium": [
                    {
                        "problem": "A train travels 60 miles in 1.5 hours. What is its average speed?",
                        "answer": {
                            "steps": [
                                "Speed = Distance / Time",
                                "Distance = 60 miles",
                                "Time = 1.5 hours",
                                "Speed = 60 / 1.5 = 40 mph",
                            ],
                            "answer": "40 mph",
                        },
                    }
                ],
            },
            "logical": {
                "easy": [
                    {
                        "problem": "All cats are animals. Fluffy is a cat. What can we conclude about Fluffy?",
                        "answer": {
                            "steps": [
                                "Given: All cats are animals",
                                "Given: Fluffy is a cat",
                                "Therefore: Fluffy is an animal",
                            ],
                            "answer": "Fluffy is an animal",
                        },
                    }
                ]
            },
            "technical": {
                "easy": [
                    {
                        "problem": "Given the assembly code:\nMOV AX, 10\nMOV BX, 5\nADD AX, BX\nWhat is the final value in AX?",
                        "answer": {
                            "steps": [
                                "MOV AX, 10 - Load 10 into register AX",
                                "MOV BX, 5 - Load 5 into register BX",
                                "ADD AX, BX - Add BX to AX: AX = 10 + 5 = 15",
                            ],
                            "answer": "15",
                            "explanation": "The ADD instruction adds the source operand (BX) to the destination operand (AX) and stores the result in AX",
                        },
                    },
                    {
                        "problem": "Trace this code:\nMOV CX, 3\nDEC CX\nDEC CX\nWhat is the value in CX?",
                        "answer": {
                            "steps": [
                                "MOV CX, 3 - CX = 3",
                                "DEC CX - Decrement CX: CX = 2",
                                "DEC CX - Decrement CX: CX = 1",
                            ],
                            "answer": "1",
                            "explanation": "DEC decrements the operand by 1",
                        },
                    },
                ],
                "medium": [
                    {
                        "problem": "Analyze this code:\nMOV AX, 8\nMOV BX, 3\nMUL BX\nWhat are the values in AX and DX after execution?",
                        "answer": {
                            "steps": [
                                "MOV AX, 8 - AX = 8",
                                "MOV BX, 3 - BX = 3",
                                "MUL BX - Multiply AX by BX: 8 × 3 = 24",
                                "Result stored in DX:AX pair",
                            ],
                            "answer": "AX = 24, DX = 0",
                            "explanation": "MUL performs unsigned multiplication. For 16-bit operands, the 32-bit result is stored in DX:AX. Since 24 fits in AX, DX remains 0",
                        },
                    },
                    {
                        "problem": "Debug this loop:\nMOV CX, 5\nLOOP_START:\nDEC AX\nLOOP LOOP_START\nIf AX starts at 10, what is its final value?",
                        "answer": {
                            "steps": [
                                "Initial: AX = 10, CX = 5",
                                "Iteration 1: DEC AX (AX = 9), LOOP decrements CX to 4",
                                "Iteration 2: DEC AX (AX = 8), LOOP decrements CX to 3",
                                "Iteration 3: DEC AX (AX = 7), LOOP decrements CX to 2",
                                "Iteration 4: DEC AX (AX = 6), LOOP decrements CX to 1",
                                "Iteration 5: DEC AX (AX = 5), LOOP decrements CX to 0, exit",
                            ],
                            "answer": "5",
                            "explanation": "LOOP decrements CX and jumps if CX ≠ 0. The loop executes 5 times",
                        },
                    },
                ],
                "hard": [
                    {
                        "problem": "Find the bug:\nMOV SI, 0x100\nMOV DI, 0x200\nMOV CX, 10\nREP MOVSB\nWhat's wrong if DS and ES point to different segments?",
                        "answer": {
                            "steps": [
                                "MOVSB moves byte from [DS:SI] to [ES:DI]",
                                "REP MOVSB repeats CX times",
                                "If DS ≠ ES, copying between different segments",
                                "No bug if intended, but often ES should equal DS for local copy",
                            ],
                            "answer": "Potential issue: copying between different segments",
                            "explanation": "REP MOVSB copies from DS:SI to ES:DI. If segments differ unintentionally, data may be copied to wrong location",
                        },
                    },
                    {
                        "problem": "Stack trace:\nPUSH AX\nPUSH BX\nPOP CX\nPOP DX\nIf AX=10, BX=20, what are CX and DX?",
                        "answer": {
                            "steps": [
                                "PUSH AX - Stack: [10] (top)",
                                "PUSH BX - Stack: [20, 10] (20 on top)",
                                "POP CX - CX = 20, Stack: [10]",
                                "POP DX - DX = 10, Stack: []",
                            ],
                            "answer": "CX = 20, DX = 10",
                            "explanation": "Stack is LIFO (Last In First Out). BX was pushed last, so it's popped first into CX",
                        },
                    },
                ],
            },
            "analytical": {
                "easy": [
                    {
                        "problem": "A function returns 0 on success and -1 on error. It returned -1. What can we conclude?",
                        "answer": {
                            "steps": [
                                "Function returns 0 for success",
                                "Function returns -1 for error",
                                "Function returned -1",
                                "Therefore: An error occurred",
                            ],
                            "answer": "An error occurred",
                            "explanation": "Based on the return value convention, -1 indicates an error condition",
                        },
                    }
                ]
            },
        }

    def generate_example(
        self,
        example_id: int,
        domain: str,
        difficulty: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> ConstrainedExample:
        """Generate a single reasoning example."""

        # First check if we have domain-specific templates (e.g., technical for assembly)
        if domain == "technical" and "assembly" in str(context).lower():
            reasoning_type = "technical"
        else:
            reasoning_type = self.problem.reasoning_type

        reasoning_templates = self.templates.get(reasoning_type, {})
        difficulty_templates = reasoning_templates.get(difficulty, [])

        if difficulty_templates:
            template = random.choice(difficulty_templates)
            problem = template["problem"]
            answer = template["answer"]
        else:
            # Better generic templates based on domain
            if domain == "technical":
                problem = f"Debug this code snippet and identify the issue (difficulty: {difficulty})"
                if self.problem.requires_steps:
                    answer = {
                        "steps": [
                            "Analyze the code structure",
                            "Identify potential issues",
                            "Provide solution",
                        ],
                        "answer": "Code analysis complete",
                        "explanation": "Technical debugging example",
                    }
                else:
                    answer = "Technical analysis result"
            elif domain == "mathematical":
                problem = f"Solve this {difficulty} mathematical problem"
                if self.problem.requires_steps:
                    answer = {
                        "steps": [
                            "Set up the equation",
                            "Apply the formula",
                            "Calculate the result",
                        ],
                        "answer": "Mathematical solution",
                        "explanation": "Step-by-step mathematical reasoning",
                    }
                else:
                    answer = "Mathematical result"
            else:
                problem = (
                    f"Solve this {reasoning_type} problem with {difficulty} difficulty"
                )
                if self.problem.requires_steps:
                    answer = {
                        "steps": [f"Step {i}" for i in range(1, 4)],
                        "answer": f"Solution to {difficulty} problem",
                        "explanation": "Detailed reasoning explanation",
                    }
                else:
                    answer = f"Solution to {difficulty} {reasoning_type} problem"

        input_data = {"problem": problem}

        example = ConstrainedExample(
            id=example_id,
            input_data=input_data,
            expected_output=answer,
            metadata={
                "domain": domain,
                "difficulty": difficulty,
                "problem_type": "reasoning",
                "reasoning_type": reasoning_type,
                "requires_steps": self.problem.requires_steps,
            },
        )

        return example

    def generate_batch(
        self,
        count: int,
        domain: str,
        difficulty: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[ConstrainedExample]:
        """Generate a batch of reasoning examples."""
        examples = []
        for i in range(count):
            example = self.generate_example(i + 1, domain, difficulty, context)
            examples.append(example)
        return examples


class CodeGenerationExampleGenerator(ConstrainedExampleGenerator):
    """Generator for code generation problem examples."""

    def __init__(
        self,
        problem_type: CodeGenerationProblem,
        context: Optional[Dict[str, Any]] = None,
    ):
        """Initialize with code generation problem."""
        super().__init__(problem_type)
        self.problem = problem_type
        self.context = context or {}

        # SQL templates
        self.sql_templates = {
            "easy": [
                {
                    "description": "Get all users from the users table",
                    "code": "SELECT * FROM users;",
                    "schema": "users(id, name, email, created_at)",
                },
                {
                    "description": "Count the number of orders",
                    "code": "SELECT COUNT(*) FROM orders;",
                    "schema": "orders(id, user_id, total, status)",
                },
            ],
            "medium": [
                {
                    "description": "Find all users who made orders in the last 30 days",
                    "code": "SELECT DISTINCT u.* FROM users u JOIN orders o ON u.id = o.user_id WHERE o.created_at >= DATE_SUB(CURRENT_DATE, INTERVAL 30 DAY);",
                    "schema": "users(id, name, email), orders(id, user_id, created_at, total)",
                },
                {
                    "description": "Get the total revenue by product category",
                    "code": "SELECT c.name, SUM(oi.quantity * oi.price) as total_revenue FROM categories c JOIN products p ON c.id = p.category_id JOIN order_items oi ON p.id = oi.product_id GROUP BY c.id, c.name;",
                    "schema": "categories(id, name), products(id, category_id, name), order_items(product_id, quantity, price)",
                },
            ],
            "hard": [
                {
                    "description": "Find customers with declining order frequency",
                    "code": "WITH monthly_orders AS (SELECT user_id, DATE_TRUNC('month', created_at) as month, COUNT(*) as order_count FROM orders GROUP BY user_id, month) SELECT user_id FROM monthly_orders m1 JOIN monthly_orders m2 ON m1.user_id = m2.user_id AND m1.month = m2.month - INTERVAL '1 month' WHERE m2.order_count < m1.order_count * 0.8;",
                    "schema": "orders(id, user_id, created_at, total)",
                }
            ],
        }

        # Python templates
        self.python_templates = {
            "easy": [
                {
                    "description": "Calculate the factorial of a number",
                    "code": "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n - 1)",
                },
                {
                    "description": "Check if a string is a palindrome",
                    "code": "def is_palindrome(s):\n    s = s.lower().replace(' ', '')\n    return s == s[::-1]",
                },
            ],
            "medium": [
                {
                    "description": "Find all prime numbers up to n",
                    "code": "def find_primes(n):\n    primes = []\n    for num in range(2, n + 1):\n        is_prime = True\n        for i in range(2, int(num ** 0.5) + 1):\n            if num % i == 0:\n                is_prime = False\n                break\n        if is_prime:\n            primes.append(num)\n    return primes",
                }
            ],
        }

    def generate_example(
        self,
        example_id: int,
        domain: str,
        difficulty: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> ConstrainedExample:
        """Generate a single code generation example."""

        # Merge contexts
        full_context = {**self.context, **(context or {})}

        # Choose templates based on language
        if self.problem.target_language.lower() in ["sql", "mysql", "postgresql"]:
            templates = self.sql_templates.get(difficulty, self.sql_templates["medium"])
        else:
            templates = self.python_templates.get(
                difficulty, self.python_templates["medium"]
            )

        if templates and not full_context.get("problem_description"):
            # Use predefined template
            template = random.choice(templates)
            input_data = {"description": template["description"]}
            if "schema" in template:
                input_data["context"] = f"Database schema: {template['schema']}"
            expected_output = template["code"]
        else:
            # Context-aware generation
            if full_context.get("problem_description"):
                input_data, expected_output = self._generate_from_context(
                    full_context["problem_description"],
                    self.problem.target_language,
                    difficulty,
                )
            else:
                # Generic template
                input_data = {
                    "description": f"Generate {self.problem.target_language} {self.problem.code_type} for {domain} with {difficulty} complexity"
                }
                expected_output = f"# {self.problem.target_language} code here"

        example = ConstrainedExample(
            id=example_id,
            input_data=input_data,
            expected_output=expected_output,
            metadata={
                "domain": domain,
                "difficulty": difficulty,
                "problem_type": "code_generation",
                "target_language": self.problem.target_language,
                "code_type": self.problem.code_type,
            },
        )

        return example

    def _generate_from_context(
        self, description: str, language: str, difficulty: str
    ) -> Tuple[Dict[str, Any], str]:
        """Generate code example from problem description."""
        description_lower = description.lower()

        # Text to SQL specific examples
        if "sql" in description_lower or "query" in description_lower:
            if difficulty == "easy":
                return (
                    {
                        "description": "Get all products with price greater than 50",
                        "context": "Database schema: products(id, name, price, category)",
                    },
                    "SELECT * FROM products WHERE price > 50;",
                )
            elif difficulty == "medium":
                return (
                    {
                        "description": "Find customers who have placed more than 5 orders",
                        "context": "Database schema: customers(id, name, email), orders(id, customer_id, total, date)",
                    },
                    "SELECT c.*, COUNT(o.id) as order_count FROM customers c JOIN orders o ON c.id = o.customer_id GROUP BY c.id HAVING COUNT(o.id) > 5;",
                )
            else:  # hard
                return (
                    {
                        "description": "Get the top 10 products by revenue in each category",
                        "context": "Database schema: products(id, name, category_id), order_items(product_id, quantity, price), categories(id, name)",
                    },
                    "WITH product_revenue AS (SELECT p.id, p.name, p.category_id, SUM(oi.quantity * oi.price) as revenue FROM products p JOIN order_items oi ON p.id = oi.product_id GROUP BY p.id, p.name, p.category_id), ranked_products AS (SELECT *, ROW_NUMBER() OVER (PARTITION BY category_id ORDER BY revenue DESC) as rank FROM product_revenue) SELECT rp.*, c.name as category_name FROM ranked_products rp JOIN categories c ON rp.category_id = c.id WHERE rank <= 10;",
                )

        # Generic code generation
        return (
            {"description": description},
            f"# Generated {language} code for: {description}",
        )

    def generate_batch(
        self,
        count: int,
        domain: str,
        difficulty: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[ConstrainedExample]:
        """Generate a batch of code generation examples."""
        examples = []
        for i in range(count):
            example = self.generate_example(i + 1, domain, difficulty, context)
            examples.append(example)
        return examples


# Factory function to get the right generator
def get_constrained_generator(
    problem_type: ProblemType, **kwargs
) -> ConstrainedExampleGenerator:
    """
    Get the appropriate constrained generator for a problem type.

    Args:
        problem_type: The problem type instance
        **kwargs: Additional arguments for specific generators

    Returns:
        Appropriate ConstrainedExampleGenerator instance
    """
    # Extract common context
    context = kwargs.get("context", {})

    if isinstance(problem_type, ClassificationProblem):
        categories = kwargs.get(
            "categories", ["category_a", "category_b", "category_c"]
        )
        return ClassificationExampleGenerator(problem_type, categories, context)

    elif isinstance(problem_type, SequenceGenerationProblem):
        return GenerationExampleGenerator(problem_type)

    elif isinstance(problem_type, InformationExtractionProblem):
        schema = kwargs.get("schema", {})
        return InformationExtractionExampleGenerator(problem_type, schema)

    elif isinstance(problem_type, QuestionAnsweringProblem):
        return QuestionAnsweringExampleGenerator(problem_type, context)

    elif isinstance(problem_type, SummarizationProblem):
        return SummarizationExampleGenerator(problem_type)

    elif isinstance(problem_type, RankingRetrievalProblem):
        return RankingRetrievalExampleGenerator(problem_type)

    elif isinstance(problem_type, TranslationTransformationProblem):
        return TranslationTransformationExampleGenerator(problem_type)

    elif isinstance(problem_type, ReasoningProblem):
        return ReasoningExampleGenerator(problem_type)

    elif isinstance(problem_type, CodeGenerationProblem):
        return CodeGenerationExampleGenerator(problem_type, context)

    else:
        raise ValueError(
            f"No generator available for problem type: {type(problem_type)}"
        )
