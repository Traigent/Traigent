"""
Dataset generation for few-shot example selection optimization.

This module creates diverse task datasets for testing different example selection strategies
including classification, generation, and complex reasoning tasks.
"""

# ruff: noqa: E501
# Long lines are intentional - they contain realistic example outputs (emails, summaries)

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List


class TaskType(Enum):
    """Types of tasks for few-shot learning."""

    CLASSIFICATION = "classification"
    GENERATION = "generation"
    EXTRACTION = "extraction"
    REASONING = "reasoning"
    TRANSLATION = "translation"


@dataclass
class FewShotExample:
    """A single few-shot example."""

    input: str
    output: str
    metadata: Dict[str, Any]
    difficulty: float  # 0.0 (easy) to 1.0 (hard)
    domain: str


@dataclass
class FewShotTask:
    """A task with query and candidate examples."""

    query: str
    expected_output: str
    task_type: TaskType
    domain: str
    candidate_examples: List[FewShotExample]
    metadata: Dict[str, Any]


def generate_classification_examples() -> List[FewShotExample]:
    """Generate sentiment classification examples."""
    examples = [
        # Easy positive examples
        FewShotExample(
            input="This product exceeded all my expectations! Absolutely fantastic!",
            output="positive",
            metadata={"confidence": 0.95, "keywords": ["exceeded", "fantastic"]},
            difficulty=0.1,
            domain="product_review",
        ),
        FewShotExample(
            input="I'm so happy with this purchase. Great quality and fast shipping.",
            output="positive",
            metadata={"confidence": 0.9, "keywords": ["happy", "great"]},
            difficulty=0.2,
            domain="product_review",
        ),
        # Easy negative examples
        FewShotExample(
            input="Terrible experience. The product broke after one day.",
            output="negative",
            metadata={"confidence": 0.95, "keywords": ["terrible", "broke"]},
            difficulty=0.1,
            domain="product_review",
        ),
        FewShotExample(
            input="Complete waste of money. Would not recommend to anyone.",
            output="negative",
            metadata={"confidence": 0.93, "keywords": ["waste", "not recommend"]},
            difficulty=0.15,
            domain="product_review",
        ),
        # Medium difficulty - mixed signals
        FewShotExample(
            input="The product works but the customer service was horrible.",
            output="negative",
            metadata={"confidence": 0.7, "keywords": ["works", "horrible"]},
            difficulty=0.5,
            domain="product_review",
        ),
        FewShotExample(
            input="Not bad but definitely overpriced for what you get.",
            output="negative",
            metadata={"confidence": 0.65, "keywords": ["not bad", "overpriced"]},
            difficulty=0.6,
            domain="product_review",
        ),
        # Hard examples - subtle sentiment
        FewShotExample(
            input="It does what it says, nothing more, nothing less.",
            output="neutral",
            metadata={"confidence": 0.55, "keywords": ["does what it says"]},
            difficulty=0.8,
            domain="product_review",
        ),
        FewShotExample(
            input="I guess it's okay if you don't have high expectations.",
            output="negative",
            metadata={"confidence": 0.6, "keywords": ["okay", "don't have high"]},
            difficulty=0.85,
            domain="product_review",
        ),
        # Domain variety
        FewShotExample(
            input="The movie had stunning visuals but the plot was confusing.",
            output="neutral",
            metadata={"confidence": 0.6, "keywords": ["stunning", "confusing"]},
            difficulty=0.7,
            domain="movie_review",
        ),
        FewShotExample(
            input="Best restaurant in town! The chef is a genius.",
            output="positive",
            metadata={"confidence": 0.95, "keywords": ["best", "genius"]},
            difficulty=0.1,
            domain="restaurant_review",
        ),
    ]

    return examples


def generate_generation_examples() -> List[FewShotExample]:
    """Generate text generation examples (email writing)."""
    examples = [
        # Formal business emails
        FewShotExample(
            input="Write a follow-up email after a job interview for a software engineer position.",
            output="Subject: Thank You - Software Engineer Interview\n\nDear [Hiring Manager],\n\nThank you for taking the time to meet with me yesterday to discuss the Software Engineer position. I enjoyed learning about your team's innovative projects and the company's commitment to technical excellence.\n\nOur discussion reinforced my enthusiasm for this opportunity, and I'm excited about the possibility of contributing to your team.\n\nPlease let me know if you need any additional information. I look forward to hearing from you.\n\nBest regards,\n[Your Name]",
            metadata={"tone": "formal", "length": "medium"},
            difficulty=0.3,
            domain="business_email",
        ),
        # Casual emails
        FewShotExample(
            input="Write an email to cancel dinner plans with a friend.",
            output="Hey [Friend],\n\nI'm so sorry but I need to cancel our dinner plans tonight. Something urgent came up at work that I need to handle.\n\nCan we reschedule for next week? Maybe Thursday or Friday? Let me know what works for you.\n\nReally sorry about the short notice!\n\n[Your Name]",
            metadata={"tone": "casual", "length": "short"},
            difficulty=0.2,
            domain="personal_email",
        ),
        # Customer service
        FewShotExample(
            input="Write a response to a customer complaint about a delayed shipment.",
            output="Dear [Customer Name],\n\nI sincerely apologize for the delay in your shipment. I understand how frustrating this must be.\n\nI've looked into your order (#[ORDER_NUMBER]) and can confirm it will arrive by [DATE]. As an apology for this inconvenience, I've applied a 15% discount to your next purchase.\n\nIf you have any other concerns, please don't hesitate to reach out.\n\nSincerely,\n[Customer Service Team]",
            metadata={"tone": "apologetic", "length": "medium"},
            difficulty=0.4,
            domain="customer_service",
        ),
        # Technical communication
        FewShotExample(
            input="Write an email explaining a bug fix to the development team.",
            output="Team,\n\nI've identified and fixed the authentication bug we discussed in standup.\n\n**Issue**: JWT tokens were expiring 1 hour early due to timezone mismatch\n**Fix**: Updated token validation to use UTC timestamps\n**Testing**: All auth tests passing, deployed to staging\n\nPR #234 is ready for review. Please test your features against staging.\n\nThanks,\n[Your Name]",
            metadata={"tone": "technical", "length": "short"},
            difficulty=0.5,
            domain="technical_email",
        ),
        # Complex scenarios
        FewShotExample(
            input="Write a diplomatic email declining a business partnership proposal.",
            output="Dear [Partner Name],\n\nThank you for considering us for this partnership opportunity. We've carefully reviewed your proposal and appreciate the time and effort you've invested.\n\nAfter thorough consideration, we've decided that this partnership doesn't align with our current strategic priorities. However, we value the relationship we've built and would welcome the opportunity to explore future collaborations that might be a better fit.\n\nWe wish you success with this venture and hope to stay in touch.\n\nBest regards,\n[Your Name]",
            metadata={"tone": "diplomatic", "length": "medium"},
            difficulty=0.8,
            domain="business_email",
        ),
    ]

    return examples


def generate_reasoning_examples() -> List[FewShotExample]:
    """Generate reasoning and problem-solving examples."""
    examples = [
        # Math word problems
        FewShotExample(
            input="A store offers a 20% discount on a $50 item. If tax is 8%, what's the final price?",
            output="Step 1: Calculate discount: $50 × 0.20 = $10\nStep 2: Price after discount: $50 - $10 = $40\nStep 3: Calculate tax: $40 × 0.08 = $3.20\nStep 4: Final price: $40 + $3.20 = $43.20",
            metadata={"type": "math", "steps": 4},
            difficulty=0.3,
            domain="math_reasoning",
        ),
        # Logic puzzles
        FewShotExample(
            input="If all roses are flowers and some flowers fade quickly, can we conclude that some roses fade quickly?",
            output="No, we cannot conclude that some roses fade quickly. While all roses are flowers, the statement only tells us that 'some flowers' fade quickly. These quick-fading flowers might not include any roses. This is a classic logical fallacy - we cannot make this inference without additional information.",
            metadata={"type": "logic", "fallacy": "hasty_generalization"},
            difficulty=0.6,
            domain="logic_reasoning",
        ),
        # Causal reasoning
        FewShotExample(
            input="Sales increased 30% after we launched the new website. Can we attribute the increase to the website?",
            output="Not necessarily. While the timing suggests a connection, we need to consider:\n1. Correlation doesn't imply causation\n2. Other factors: seasonal trends, marketing campaigns, competitor issues\n3. Need controlled testing (A/B test) to establish causation\n4. Should analyze traffic sources and conversion rates\nConclusion: The website may have contributed, but we need more data to confirm causation.",
            metadata={"type": "causal", "factors": 4},
            difficulty=0.7,
            domain="business_reasoning",
        ),
        # Code debugging
        FewShotExample(
            input="This function returns undefined instead of the sum: function add(a, b) { a + b }",
            output="The function is missing a return statement. In JavaScript, without an explicit return, functions return undefined. Fix: function add(a, b) { return a + b; }",
            metadata={"type": "debugging", "language": "javascript"},
            difficulty=0.2,
            domain="code_reasoning",
        ),
    ]

    return examples


def generate_baseline_strategies() -> Dict[str, Any]:
    """Generate baseline selection strategies for comparison."""
    return {
        "random_selection": {
            "n_examples": 3,
            "selection_method": "random",
            "order": "random",
        },
        "all_examples": {
            "n_examples": 10,
            "selection_method": "all_available",
            "order": "difficulty_ascending",
        },
        "manual_curation": {
            "n_examples": 5,
            "selection_method": "expert_selected",
            "order": "mixed_difficulty",
        },
    }


def generate_evaluation_tasks(num_tasks: int = 50) -> List[FewShotTask]:
    """Generate evaluation tasks for testing selection strategies."""
    tasks = []

    # Get all example pools
    classification_examples = generate_classification_examples()
    generation_examples = generate_generation_examples()
    reasoning_examples = generate_reasoning_examples()

    # Generate classification tasks
    for i in range(num_tasks // 3):
        tasks.append(
            FewShotTask(
                query="The interface is intuitive but the performance is disappointing.",
                expected_output="negative",
                task_type=TaskType.CLASSIFICATION,
                domain="product_review",
                candidate_examples=classification_examples,
                metadata={"task_id": f"class_{i}", "difficulty": 0.6},
            )
        )

    # Generate generation tasks
    for i in range(num_tasks // 3):
        tasks.append(
            FewShotTask(
                query="Write an email requesting a meeting to discuss project timeline changes.",
                expected_output="Subject: Meeting Request - Project Timeline Discussion\n\nDear Team,\n\nI'd like to schedule a meeting to discuss some necessary adjustments to our project timeline.\n\nCould we meet this week to review the changes and ensure alignment?\n\nPlease let me know your availability.\n\nBest regards,\n[Your Name]",
                task_type=TaskType.GENERATION,
                domain="business_email",
                candidate_examples=generation_examples,
                metadata={"task_id": f"gen_{i}", "expected_tone": "professional"},
            )
        )

    # Generate reasoning tasks
    for i in range(num_tasks - 2 * (num_tasks // 3)):
        tasks.append(
            FewShotTask(
                query="If we increase prices by 10% and lose 5% of customers, what happens to revenue?",
                expected_output="Revenue increases by approximately 4.5%. Here's why:\nOriginal: 100 customers × $100 = $10,000\nAfter change: 95 customers × $110 = $10,450\nIncrease: $450 / $10,000 = 4.5%",
                task_type=TaskType.REASONING,
                domain="business_reasoning",
                candidate_examples=reasoning_examples,
                metadata={"task_id": f"reason_{i}", "requires": "calculation"},
            )
        )

    return tasks


def create_task_variations(
    base_task: FewShotTask, num_variations: int = 5
) -> List[FewShotTask]:
    """Create variations of a task to test robustness."""
    variations = []

    # Create variations with different phrasings
    for i in range(num_variations):
        variation = FewShotTask(
            query=f"{base_task.query} (Variation {i+1})",
            expected_output=base_task.expected_output,
            task_type=base_task.task_type,
            domain=base_task.domain,
            candidate_examples=base_task.candidate_examples,
            metadata={**base_task.metadata, "variation_id": i},
        )
        variations.append(variation)

    return variations


def analyze_example_diversity(examples: List[FewShotExample]) -> Dict[str, float]:
    """Analyze diversity metrics of example set."""
    if not examples:
        return {}

    # Calculate diversity metrics
    difficulties = [e.difficulty for e in examples]
    domains = [e.domain for e in examples]

    return {
        "difficulty_variance": sum(
            (d - sum(difficulties) / len(difficulties)) ** 2 for d in difficulties
        )
        / len(difficulties),
        "difficulty_range": max(difficulties) - min(difficulties),
        "domain_diversity": len(set(domains)) / len(domains),
        "avg_difficulty": sum(difficulties) / len(difficulties),
        "example_count": len(examples),
    }
