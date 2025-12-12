#!/usr/bin/env python3
"""
Evaluator for Customer Support Agent

This evaluator scores support responses on:
1. Resolution Accuracy - Did the response address the customer's issue correctly?
2. Tone Quality - Empathy, clarity, professionalism
3. Escalation Accuracy - Correct escalation decisions (precision, recall, F1)

Supports two modes:
- MOCK MODE (default): Uses heuristic rules for fast, free evaluation
- REAL MODE: Uses actual LLM calls for agent and LLM-as-judge evaluation

Usage:
  Mock mode: python evaluator.py  (default, uses heuristics)
  Real mode: TRAIGENT_MOCK_MODE=false python evaluator.py  (requires OPENAI_API_KEY)
"""

import json
import os
import re
from dataclasses import dataclass
from typing import Any

# ============================================================================
# PROMPTS FOR REAL LLM MODE
# ============================================================================

# Prompt for the support agent to respond to customers
AGENT_PROMPT = """You are a customer support agent for ShopEasy, an e-commerce company.

CUSTOMER MESSAGE:
{query}

CUSTOMER CONTEXT:
- Customer Tier: {customer_tier}
- Sentiment: {sentiment}
- Order Status: {order_status}

Respond helpfully and professionally. If the customer is very upset or demanding a supervisor,
you should escalate (set should_escalate=true).

Respond in EXACTLY this JSON format:
{{"response": "your response here", "should_escalate": true/false, "resolution_type": "refund|replacement|information|escalated"}}
"""

# Prompt for LLM-as-judge to evaluate support responses
JUDGE_PROMPT = """You are evaluating a customer support response.

CUSTOMER QUERY: {query}
CUSTOMER SENTIMENT: {sentiment}

SUPPORT RESPONSE: {response}

Evaluate the response:
1. RESOLUTION (1-5): Did it address the customer's actual problem?
2. EMPATHY (1-5): Did it acknowledge customer feelings appropriately?
3. CLARITY (1-5): Were next steps clearly explained?
4. PROFESSIONALISM (1-5): Appropriate tone and language?

Respond in EXACTLY this format:
RESOLUTION: [1-5]
EMPATHY: [1-5]
CLARITY: [1-5]
PROFESSIONALISM: [1-5]
"""


@dataclass
class SupportEvaluationResult:
    """Result of support response evaluation."""

    resolution_accuracy: float  # 0-1 scale
    tone_quality: float  # 0-1 scale
    empathy_score: float  # 0-1 scale
    clarity_score: float  # 0-1 scale
    professionalism_score: float  # 0-1 scale
    escalation_correct: bool
    escalation_precision: float  # For aggregate metrics
    escalation_recall: float  # For aggregate metrics
    overall_score: float


class SupportEvaluator:
    """
    Evaluator for customer support agent responses.

    Uses LLM-as-judge for resolution accuracy and tone quality,
    plus deterministic evaluation for escalation decisions.
    """

    def __init__(self):
        """Initialize the evaluator."""
        self.weights = {
            "resolution_accuracy": 0.4,
            "tone_quality": 0.35,
            "escalation_accuracy": 0.25,
        }

        # Tone quality rubric for LLM-as-judge
        self.tone_rubric = """
Rate the customer support response on the following criteria (1-5 scale each):

1. EMPATHY (1-5):
   1 = Dismissive, no acknowledgment of customer feelings
   2 = Minimal acknowledgment, feels robotic
   3 = Adequate empathy, acknowledges concern
   4 = Good empathy, validates customer experience
   5 = Exceptional empathy, truly connects with customer

2. CLARITY (1-5):
   1 = Confusing, unclear what to do next
   2 = Somewhat clear but missing key information
   3 = Adequately clear, basic understanding possible
   4 = Clear and well-structured response
   5 = Crystal clear, all steps/info perfectly explained

3. PROFESSIONALISM (1-5):
   1 = Unprofessional, inappropriate language/tone
   2 = Marginally professional, some issues
   3 = Professional but generic
   4 = Highly professional, brand-appropriate
   5 = Exemplary professionalism, perfect brand voice

Provide scores in this exact format:
EMPATHY: [score]
CLARITY: [score]
PROFESSIONALISM: [score]
"""

        # Resolution accuracy rubric
        self.resolution_rubric = """
Evaluate if the support response correctly addresses the customer's issue.

Customer Query: {query}
Customer Context: {context}
Support Response: {response}
Expected Resolution Type: {expected_resolution}

Rate the resolution accuracy (1-5):
1 = Completely wrong or irrelevant response
2 = Partially addresses issue but major gaps
3 = Addresses main issue but could be more complete
4 = Good resolution, covers key points
5 = Excellent resolution, comprehensive and accurate

Also evaluate:
- Did the response match the expected resolution type? (yes/no)
- Were company policies correctly applied? (yes/no)
- Were next steps clearly communicated? (yes/no)

Provide in this exact format:
ACCURACY_SCORE: [1-5]
RESOLUTION_MATCH: [yes/no]
POLICY_CORRECT: [yes/no]
NEXT_STEPS_CLEAR: [yes/no]
"""

    def __call__(
        self,
        prediction: dict[str, Any] | str,
        expected: dict[str, Any] | str | None,
        input_data: dict[str, Any],
    ) -> dict[str, float]:
        """
        Evaluate a support response.

        Args:
            prediction: The generated response (dict or string)
            expected: Expected response data (may contain gold_response, should_escalate)
            input_data: Input data containing query, customer_context

        Returns:
            Dictionary of metric scores
        """
        # Parse prediction
        if isinstance(prediction, str):
            response = prediction
            predicted_escalate = False
            resolution_type = "unknown"
        else:
            response = prediction.get("response", "")
            predicted_escalate = prediction.get("should_escalate", False)
            resolution_type = prediction.get("resolution_type", "unknown")

        # Parse expected
        if isinstance(expected, str):
            expected_response = expected
            expected_escalate = False
            expected_resolution = "unknown"
        elif expected:
            expected_response = expected.get("gold_response", "")
            expected_escalate = expected.get("should_escalate", False)
            expected_resolution = expected.get("resolution_type", "unknown")
        else:
            expected_response = ""
            expected_escalate = input_data.get("should_escalate", False)
            expected_resolution = input_data.get("resolution_type", "unknown")

        # Get query and context
        query = input_data.get("query", "")
        customer_context = input_data.get("customer_context", {})

        # Evaluate resolution accuracy
        resolution_result = self._evaluate_resolution(
            response=response,
            query=query,
            context=customer_context,
            expected_resolution=expected_resolution,
        )

        # Evaluate tone quality
        tone_result = self._evaluate_tone(response)

        # Evaluate escalation accuracy
        escalation_correct = predicted_escalate == expected_escalate
        escalation_score = 1.0 if escalation_correct else 0.0

        # Calculate overall score
        overall = (
            self.weights["resolution_accuracy"] * resolution_result["accuracy"]
            + self.weights["tone_quality"] * tone_result["overall"]
            + self.weights["escalation_accuracy"] * escalation_score
        )

        return {
            "resolution_accuracy": resolution_result["accuracy"],
            "resolution_match": resolution_result["resolution_match"],
            "policy_correct": resolution_result["policy_correct"],
            "tone_quality": tone_result["overall"],
            "empathy_score": tone_result["empathy"],
            "clarity_score": tone_result["clarity"],
            "professionalism_score": tone_result["professionalism"],
            "escalation_accuracy": escalation_score,
            "escalation_correct": float(escalation_correct),
            "predicted_escalate": float(predicted_escalate),
            "expected_escalate": float(expected_escalate),
            "overall": overall,
        }

    def _evaluate_resolution(
        self,
        response: str,
        query: str,
        context: dict[str, Any],
        expected_resolution: str,
    ) -> dict[str, float]:
        """
        Evaluate resolution accuracy using LLM-as-judge.

        Returns:
            Dictionary with accuracy and sub-metrics
        """
        if not response.strip():
            return {
                "accuracy": 0.0,
                "resolution_match": 0.0,
                "policy_correct": 0.0,
                "next_steps_clear": 0.0,
            }

        # Try LLM evaluation
        try:
            from langchain_openai import ChatOpenAI

            mock_mode = os.environ.get("TRAIGENT_MOCK_MODE", "true").lower() == "true"
            if mock_mode:
                raise ImportError("Mock mode enabled")

            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)

            prompt = self.resolution_rubric.format(
                query=query,
                context=json.dumps(context),
                response=response,
                expected_resolution=expected_resolution,
            )

            result = llm.invoke(prompt)
            return self._parse_resolution_scores(
                result.content, response, expected_resolution
            )

        except (ImportError, Exception):
            # Fallback to heuristic evaluation
            return self._heuristic_resolution_eval(response, query, expected_resolution)

    def _parse_resolution_scores(
        self,
        llm_response: str,
        response: str,
        expected_resolution: str,
    ) -> dict[str, float]:
        """Parse LLM judge response for resolution metrics."""
        # Extract accuracy score
        accuracy_match = re.search(r"ACCURACY_SCORE:\s*(\d)", llm_response)
        accuracy = int(accuracy_match.group(1)) / 5.0 if accuracy_match else 0.6

        # Extract resolution match
        resolution_match = re.search(
            r"RESOLUTION_MATCH:\s*(yes|no)", llm_response.lower()
        )
        resolution_score = (
            1.0 if resolution_match and resolution_match.group(1) == "yes" else 0.0
        )

        # Extract policy correct
        policy_match = re.search(r"POLICY_CORRECT:\s*(yes|no)", llm_response.lower())
        policy_score = 1.0 if policy_match and policy_match.group(1) == "yes" else 0.5

        # Extract next steps
        next_steps_match = re.search(
            r"NEXT_STEPS_CLEAR:\s*(yes|no)", llm_response.lower()
        )
        next_steps_score = (
            1.0 if next_steps_match and next_steps_match.group(1) == "yes" else 0.5
        )

        return {
            "accuracy": accuracy,
            "resolution_match": resolution_score,
            "policy_correct": policy_score,
            "next_steps_clear": next_steps_score,
        }

    def _heuristic_resolution_eval(
        self,
        response: str,
        query: str,
        expected_resolution: str,
    ) -> dict[str, float]:
        """Heuristic evaluation of resolution accuracy."""
        response_lower = response.lower()
        query_lower = query.lower()

        score = 0.5  # Start neutral

        # Check if response addresses the query topic
        if "refund" in query_lower and any(
            w in response_lower for w in ["refund", "credited", "money back"]
        ):
            score += 0.2
        elif "track" in query_lower and any(
            w in response_lower for w in ["track", "shipping", "delivery", "transit"]
        ):
            score += 0.2
        elif "cancel" in query_lower and any(
            w in response_lower for w in ["cancel", "cancelled", "cancellation"]
        ):
            score += 0.2
        elif "damaged" in query_lower and any(
            w in response_lower for w in ["replacement", "refund", "sorry", "apologize"]
        ):
            score += 0.2

        # Check for clear next steps
        if any(
            w in response_lower for w in ["will be", "within", "please", "let me know"]
        ):
            score += 0.1

        # Check for acknowledgment
        if any(
            w in response_lower
            for w in ["understand", "apologize", "sorry", "thank you"]
        ):
            score += 0.1

        # Check resolution type match
        actual_resolution = self._detect_resolution_type(response)
        resolution_match = 1.0 if actual_resolution == expected_resolution else 0.5

        return {
            "accuracy": min(1.0, score),
            "resolution_match": resolution_match,
            "policy_correct": 0.7,  # Assume mostly correct without LLM
            "next_steps_clear": 0.7,
        }

    def _detect_resolution_type(self, response: str) -> str:
        """Detect the resolution type from response content."""
        response_lower = response.lower()

        if any(
            w in response_lower for w in ["escalat", "supervisor", "senior", "manager"]
        ):
            return "escalated"
        if any(w in response_lower for w in ["refund", "credited", "money back"]):
            return "refund"
        if any(w in response_lower for w in ["replacement", "replace", "new item"]):
            return "replacement"
        if any(w in response_lower for w in ["cancel", "cancelled"]):
            return "cancellation"
        if any(w in response_lower for w in ["track", "shipping", "status", "transit"]):
            return "information"
        if any(w in response_lower for w in ["update", "change", "modify"]):
            return "order_modification"

        return "resolved"

    def _evaluate_tone(self, response: str) -> dict[str, float]:
        """
        Evaluate tone quality using LLM-as-judge.

        Returns:
            Dictionary with empathy, clarity, professionalism, and overall scores
        """
        if not response.strip():
            return {
                "empathy": 0.0,
                "clarity": 0.0,
                "professionalism": 0.0,
                "overall": 0.0,
            }

        # Try LLM evaluation
        try:
            from langchain_openai import ChatOpenAI

            mock_mode = os.environ.get("TRAIGENT_MOCK_MODE", "true").lower() == "true"
            if mock_mode:
                raise ImportError("Mock mode enabled")

            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)

            prompt = f"{self.tone_rubric}\n\nCustomer Support Response:\n{response}"
            result = llm.invoke(prompt)
            return self._parse_tone_scores(result.content)

        except (ImportError, Exception):
            # Fallback to heuristic evaluation
            return self._heuristic_tone_eval(response)

    def _parse_tone_scores(self, llm_response: str) -> dict[str, float]:
        """Parse LLM judge response for tone scores."""
        # Extract scores using regex
        empathy_match = re.search(r"EMPATHY:\s*(\d)", llm_response)
        clarity_match = re.search(r"CLARITY:\s*(\d)", llm_response)
        professionalism_match = re.search(r"PROFESSIONALISM:\s*(\d)", llm_response)

        empathy = int(empathy_match.group(1)) / 5.0 if empathy_match else 0.6
        clarity = int(clarity_match.group(1)) / 5.0 if clarity_match else 0.6
        professionalism = (
            int(professionalism_match.group(1)) / 5.0 if professionalism_match else 0.6
        )

        # Weighted average for overall tone
        overall = empathy * 0.35 + clarity * 0.35 + professionalism * 0.30

        return {
            "empathy": empathy,
            "clarity": clarity,
            "professionalism": professionalism,
            "overall": overall,
        }

    def _heuristic_tone_eval(self, response: str) -> dict[str, float]:
        """Heuristic evaluation of tone quality."""
        response_lower = response.lower()

        # Empathy indicators
        empathy_words = [
            "understand",
            "sorry",
            "apologize",
            "appreciate",
            "frustrating",
            "inconvenience",
            "concern",
            "help",
        ]
        empathy_count = sum(1 for w in empathy_words if w in response_lower)
        empathy_score = min(1.0, 0.4 + empathy_count * 0.12)

        # Clarity indicators
        # Check for structure and clear communication
        has_greeting = any(
            w in response_lower for w in ["hello", "hi", "thank you", "thanks"]
        )
        has_next_steps = any(
            w in response_lower for w in ["will", "please", "let me", "can"]
        )
        has_closing = any(
            w in response_lower for w in ["anything else", "further", "additional"]
        )
        clarity_score = (
            0.5 + (has_greeting * 0.15) + (has_next_steps * 0.2) + (has_closing * 0.15)
        )

        # Professionalism indicators
        unprofessional_words = ["dude", "bro", "lol", "omg", "whatever", "idk"]
        professional_words = ["pleased", "assist", "ensure", "priority", "committed"]
        unprofessional_count = sum(
            1 for w in unprofessional_words if w in response_lower
        )
        professional_count = sum(1 for w in professional_words if w in response_lower)
        professionalism_score = min(
            1.0, 0.6 + professional_count * 0.1 - unprofessional_count * 0.2
        )

        # Overall weighted average
        overall = (
            empathy_score * 0.35 + clarity_score * 0.35 + professionalism_score * 0.30
        )

        return {
            "empathy": empathy_score,
            "clarity": clarity_score,
            "professionalism": professionalism_score,
            "overall": overall,
        }


def is_mock_mode() -> bool:
    """Check if running in mock mode."""
    return os.environ.get("TRAIGENT_MOCK_MODE", "true").lower() == "true"


def run_optimization(num_configs: int = 5, num_examples: int = 10):
    """Run optimization testing different support agent configurations."""
    from openai import OpenAI

    client = OpenAI()
    dataset = load_dataset()[:num_examples]
    evaluator = SupportEvaluator()

    configs = [
        {
            "name": "baseline",
            "temperature": 0.3,
            "instruction": "Be helpful and professional.",
        },
        {
            "name": "empathetic",
            "temperature": 0.4,
            "instruction": "Lead with empathy. Acknowledge feelings first.",
        },
        {
            "name": "efficient",
            "temperature": 0.2,
            "instruction": "Be concise and solution-focused.",
        },
        {
            "name": "thorough",
            "temperature": 0.3,
            "instruction": "Be detailed. Explain all options thoroughly.",
        },
        {
            "name": "cautious",
            "temperature": 0.1,
            "instruction": "Escalate if customer is upset or asks for manager.",
        },
    ][:num_configs]

    print("\n" + "=" * 70)
    print("OPTIMIZATION RUN: Testing Different Support Agent Configs")
    print("=" * 70)
    print(
        f"\nConfigs: {num_configs}, Examples: {num_examples}, Total calls: {num_configs * num_examples}"
    )

    results = []
    for config in configs:
        print(f"\n--- Config: {config['name']} (temp={config['temperature']}) ---")
        scores = []

        for i, entry in enumerate(dataset):
            input_data = entry.get("input", {})
            query = input_data.get("query", "")
            sentiment = input_data.get("sentiment", "neutral")
            should_escalate = entry.get("should_escalate", False)
            expected = entry.get("expected_resolution", "")

            prompt = f"""{config['instruction']}

Customer: {query}
Sentiment: {sentiment}

Respond helpfully. Return JSON: {{"response": "...", "should_escalate": true/false}}"""

            try:
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=config["temperature"],
                )
                content = response.choices[0].message.content
                json_match = re.search(r"\{.*\}", content, re.DOTALL)
                pred = (
                    json.loads(json_match.group())
                    if json_match
                    else {"response": content, "should_escalate": False}
                )

                result = evaluator(
                    pred,
                    expected,
                    {
                        "should_escalate": should_escalate,
                        "expected_resolution": expected,
                        "sentiment": sentiment,
                    },
                )
                scores.append(result)
                print(
                    f"  [{i+1}/{num_examples}] tone={result['tone_quality']:.2f} esc={'✓' if result['escalation_accuracy']==1 else '✗'}"
                )
            except Exception as e:
                print(f"  [{i+1}/{num_examples}] Error: {e}")
                scores.append(
                    {"tone_quality": 0, "escalation_accuracy": 0, "overall": 0}
                )

        avg_tone = sum(s["tone_quality"] for s in scores) / len(scores)
        avg_esc = sum(s["escalation_accuracy"] for s in scores) / len(scores)
        results.append(
            {
                "config": config["name"],
                "temp": config["temperature"],
                "tone": avg_tone,
                "escalation": avg_esc,
                "overall": (avg_tone + avg_esc) / 2,
            }
        )

    print("\n" + "=" * 70)
    print("RESULTS TABLE")
    print("=" * 70)
    print(
        f"\n{'Config':<12} {'Temp':<6} {'Tone':<10} {'Escalation':<12} {'Overall':<10}"
    )
    print("-" * 50)
    for r in sorted(results, key=lambda x: x["overall"], reverse=True):
        print(
            f"{r['config']:<12} {r['temp']:<6.1f} {r['tone']:.3f}      {r['escalation']*100:>5.0f}%       {r['overall']:.3f}"
        )

    best = max(results, key=lambda x: x["overall"])
    print("-" * 50)
    print(f"🏆 BEST: {best['config']} (score={best['overall']:.3f})")
    print("=" * 70)
    return results


def respond_with_llm(query: str, context: dict) -> dict:
    """Generate a support response using LLM (real mode only)."""
    try:
        from openai import OpenAI

        client = OpenAI()
        prompt = AGENT_PROMPT.format(
            query=query,
            customer_tier=context.get("customer_tier", "standard"),
            sentiment=context.get("sentiment", "neutral"),
            order_status=context.get("order_status", "unknown"),
        )

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )

        # Parse JSON response
        content = response.choices[0].message.content
        json_match = re.search(r"\{.*\}", content, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        return {
            "response": content,
            "should_escalate": False,
            "resolution_type": "resolved",
        }
    except Exception as e:
        return {
            "response": f"Error: {e}",
            "should_escalate": False,
            "resolution_type": "error",
        }


def load_dataset() -> list[dict]:
    """Load the support tickets dataset."""
    from pathlib import Path

    dataset_path = Path(__file__).parent.parent / "datasets" / "support_tickets.jsonl"
    if not dataset_path.exists():
        return []

    entries = []
    with open(dataset_path) as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))
    return entries


def print_score_bar(label: str, score: float, max_score: float = 1.0, width: int = 20):
    """Print a visual score bar."""
    normalized = min(score / max_score, 1.0)
    filled = int(normalized * width)
    bar = "█" * filled + "░" * (width - filled)
    pct = score * 100 if max_score == 1.0 else score
    print(f"  {label:<20} {bar} {pct:.0f}%")


def demo_evaluator():
    """Demo the Customer Support Agent evaluator with clear input/output examples."""
    mock_mode = is_mock_mode()

    print("=" * 70)
    print("CUSTOMER SUPPORT AGENT - Evaluator Demo")
    print("=" * 70)
    print(
        f"\nMODE: {'MOCK (heuristic rules)' if mock_mode else 'REAL (using OpenAI API)'}"
    )

    print(
        """
WHAT THIS AGENT DOES:
  A customer support chatbot that responds to customer inquiries.
  Given a customer message and context (tier, order status, sentiment),
  it generates a helpful response and decides if escalation is needed.

HOW IT'S EVALUATED:"""
    )
    if mock_mode:
        print("  MOCK MODE: Using heuristic rules (fast, free, no API needed)")
    else:
        print("  REAL MODE: Using LLM for response generation + evaluation")
    print()

    # Load and show dataset info
    dataset = load_dataset()
    print(f"DATASET: {len(dataset)} support scenarios in support_tickets.jsonl")

    if dataset:
        escalation_count = sum(1 for e in dataset if e.get("should_escalate", False))
        print(f"  - {len(dataset) - escalation_count} can be resolved by the bot")
        print(f"  - {escalation_count} require escalation to a human")

        print("\n" + "-" * 70)
        print("SAMPLE DATA (first 2 entries):")
        print("-" * 70)
        for i, entry in enumerate(dataset[:2]):
            input_data = entry.get("input", {})
            query = input_data.get("query", "")
            ctx = input_data.get("customer_context", {})
            output = entry.get("output", "")
            escalate = entry.get("should_escalate", False)

            print(f"\n[Entry {i+1}]")
            print("  INPUT (customer message):")
            print(f'    "{query[:70]}..."' if len(query) > 70 else f'    "{query}"')
            print(f"    Customer tier: {ctx.get('customer_tier', 'N/A')}")
            print(f"    Sentiment: {ctx.get('sentiment', 'N/A')}")
            print("\n  OUTPUT (expected response):")
            preview = output[:100].replace("\n", " ")
            print(f'    "{preview}..."')
            print(f"    Should escalate: {'Yes' if escalate else 'No'}")

    evaluator = SupportEvaluator()

    print("\n" + "=" * 70)
    print("HOW SCORING WORKS:")
    print("=" * 70)
    print(
        """
The evaluator measures:

  - Resolution Accuracy: Did the response address what the customer asked?
                         (e.g., if they asked for refund, did we offer one?)

  - Tone Quality:        Was the response appropriately empathetic and clear?
                         Broken down into:
                         - Empathy: Did we acknowledge their frustration?
                         - Clarity: Were next steps clearly explained?
                         - Professionalism: Appropriate language and tone?

  - Escalation Accuracy: When customer demanded a supervisor or was very
                         upset, did the agent correctly escalate?
                         (Missing an escalation = very bad for CSAT!)
"""
    )

    print("=" * 70)
    print("EVALUATION EXAMPLES:")
    print("=" * 70)

    # Test case 1: Good response
    good_response = """I sincerely apologize for the inconvenience with your damaged laptop.

I completely understand how frustrating this must be. As a Gold member, your satisfaction is our priority.

I'd be happy to process a full refund immediately - credited within 5-7 business days. Or I can arrange express replacement shipping at no cost.

Please let me know which works best. Is there anything else I can help with?"""

    print("\n[GREAT RESPONSE] - Empathetic, solves the problem")
    print("-" * 70)
    print('Customer: "I received a damaged laptop and want a refund"')
    print("\nAgent Response:")
    for line in good_response.strip().split("\n"):
        print(f"  {line}")

    result = evaluator(
        prediction={"response": good_response, "should_escalate": False},
        expected={"should_escalate": False, "resolution_type": "refund"},
        input_data={
            "query": "I received a damaged laptop and want a refund",
            "customer_context": {"customer_tier": "gold", "sentiment": "negative"},
        },
    )
    print("\nScores:")
    print_score_bar("Resolution", result["resolution_accuracy"])
    print_score_bar("Tone Quality", result["tone_quality"])
    print(f"    Empathy:      {result['empathy_score']:.0%}")
    print(f"    Clarity:      {result['clarity_score']:.0%}")
    print(f"    Professional: {result['professionalism_score']:.0%}")
    print_score_bar("Escalation", result["escalation_accuracy"])
    print(f"  {'─' * 42}")
    print_score_bar("OVERALL", result["overall"])

    # Test case 2: Poor response
    bad_response = "Ok, we'll look into it. Check back later."

    print("\n[BAD RESPONSE] - Dismissive, doesn't help")
    print("-" * 70)
    print('Customer: "I received a damaged laptop and want a refund"')
    print(f'\nAgent Response:\n  "{bad_response}"')

    result = evaluator(
        prediction={"response": bad_response, "should_escalate": False},
        expected={"should_escalate": False, "resolution_type": "refund"},
        input_data={
            "query": "I received a damaged laptop and want a refund",
            "customer_context": {"customer_tier": "gold", "sentiment": "negative"},
        },
    )
    print("\nScores:")
    print_score_bar("Resolution", result["resolution_accuracy"])
    print("    ^ Didn't address the refund request!")
    print_score_bar("Tone Quality", result["tone_quality"])
    print("    ^ Cold, no empathy, no clear next steps")
    print_score_bar("Escalation", result["escalation_accuracy"])
    print(f"  {'─' * 42}")
    print_score_bar("OVERALL", result["overall"])

    # Test case 3: Missed escalation
    print("\n[MISSED ESCALATION] - Should have transferred to supervisor")
    print("-" * 70)
    print('Customer: "This is unacceptable! I want to speak to your supervisor!"')
    print(
        '\nAgent Response:\n  "Our standard procedure will resolve this within 24 hours."'
    )

    result = evaluator(
        prediction={
            "response": "I understand. Our standard procedure will resolve this within 24 hours.",
            "should_escalate": False,
        },
        expected={"should_escalate": True},
        input_data={
            "query": "This is unacceptable! I want to speak to your supervisor!",
            "customer_context": {
                "customer_tier": "platinum",
                "sentiment": "very_negative",
            },
        },
    )
    print("\nScores:")
    print_score_bar("Resolution", result["resolution_accuracy"])
    print_score_bar("Tone Quality", result["tone_quality"])
    print_score_bar("Escalation", result["escalation_accuracy"])
    print("    ^ Customer asked for supervisor - must escalate!")
    print(f"  {'─' * 42}")
    print_score_bar("OVERALL", result["overall"])

    # Test case 4: False escalation (new example)
    print("\n[FALSE ESCALATION] - Escalated when not needed")
    print("-" * 70)
    print('Customer: "Can you tell me my order status?"')
    print('\nAgent Response:\n  "Let me transfer you to a supervisor for that."')

    result = evaluator(
        prediction={
            "response": "Let me transfer you to a supervisor for that.",
            "should_escalate": True,
        },
        expected={"should_escalate": False},
        input_data={
            "query": "Can you tell me my order status?",
            "customer_context": {"customer_tier": "standard", "sentiment": "neutral"},
        },
    )
    print("\nScores:")
    print_score_bar("Resolution", result["resolution_accuracy"])
    print("    ^ Didn't answer the simple question!")
    print_score_bar("Tone Quality", result["tone_quality"])
    print_score_bar("Escalation", result["escalation_accuracy"])
    print("    ^ Simple query - no escalation needed!")
    print(f"  {'─' * 42}")
    print_score_bar("OVERALL", result["overall"])

    # In real mode, run optimization
    if not mock_mode:
        run_optimization(num_configs=5, num_examples=10)

    print("\n" + "=" * 70)
    print("HOW TO RUN:")
    print("  Mock mode (heuristics): python evaluator.py  (default)")
    print(
        "  Real mode (LLM+optimize): TRAIGENT_MOCK_MODE=false OPENAI_API_KEY=sk-... python evaluator.py"
    )
    print("=" * 70)


if __name__ == "__main__":
    demo_evaluator()
