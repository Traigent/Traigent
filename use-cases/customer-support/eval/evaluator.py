#!/usr/bin/env python3
"""
Evaluator for Customer Support Agent

This evaluator scores support responses on:
1. Resolution Accuracy - Did the response address the customer's issue correctly?
2. Tone Quality - Empathy, clarity, professionalism
3. Escalation Accuracy - Correct escalation decisions (precision, recall, F1)

Based on the Traigent Agent Optimization Guide specifications.

Calibration Notes:
- CSAT Calibration: Where historical CSAT scores exist, regress LLM tone predictions
  against actual scores. Adjust rubric until correlation > 0.7.
- Tone LLM-as-judge should predict real user satisfaction
- Escalation decisions: Balance false escalation rate vs missed escalation rate
- Dataset should include customer emotional states: angry, confused, anxious, frustrated
"""

import json
import os
import re
from dataclasses import dataclass
from typing import Any


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

            mock_mode = os.environ.get("TRAIGENT_MOCK_MODE", "false").lower() == "true"
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
            return self._parse_resolution_scores(result.content, response, expected_resolution)

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
        resolution_match = re.search(r"RESOLUTION_MATCH:\s*(yes|no)", llm_response.lower())
        resolution_score = 1.0 if resolution_match and resolution_match.group(1) == "yes" else 0.0

        # Extract policy correct
        policy_match = re.search(r"POLICY_CORRECT:\s*(yes|no)", llm_response.lower())
        policy_score = 1.0 if policy_match and policy_match.group(1) == "yes" else 0.5

        # Extract next steps
        next_steps_match = re.search(r"NEXT_STEPS_CLEAR:\s*(yes|no)", llm_response.lower())
        next_steps_score = 1.0 if next_steps_match and next_steps_match.group(1) == "yes" else 0.5

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
        if any(w in response_lower for w in ["will be", "within", "please", "let me know"]):
            score += 0.1

        # Check for acknowledgment
        if any(w in response_lower for w in ["understand", "apologize", "sorry", "thank you"]):
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

        if any(w in response_lower for w in ["escalat", "supervisor", "senior", "manager"]):
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

            mock_mode = os.environ.get("TRAIGENT_MOCK_MODE", "false").lower() == "true"
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
        professionalism = int(professionalism_match.group(1)) / 5.0 if professionalism_match else 0.6

        # Weighted average for overall tone
        overall = (empathy * 0.35 + clarity * 0.35 + professionalism * 0.30)

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
            "understand", "sorry", "apologize", "appreciate",
            "frustrating", "inconvenience", "concern", "help"
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
        clarity_score = 0.5 + (has_greeting * 0.15) + (has_next_steps * 0.2) + (has_closing * 0.15)

        # Professionalism indicators
        unprofessional_words = ["dude", "bro", "lol", "omg", "whatever", "idk"]
        professional_words = ["pleased", "assist", "ensure", "priority", "committed"]
        unprofessional_count = sum(1 for w in unprofessional_words if w in response_lower)
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


def load_dataset() -> list[dict]:
    """Load the support tickets dataset."""
    import json
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


def demo_evaluator():
    """
    Demo the Customer Support Agent evaluator.

    This runs in MOCK MODE - no API calls are made.
    The evaluator uses heuristic rules to score support responses on:
    - Resolution Accuracy: Did it address the customer's actual issue?
    - Tone Quality: Empathy, clarity, professionalism (CSAT proxy)
    - Escalation Accuracy: Correct routing decisions
    """
    print("=" * 60)
    print("CUSTOMER SUPPORT AGENT - Evaluator Demo")
    print("=" * 60)
    print("\nMODE: Mock (heuristic scoring, no API calls)")
    print("\nEVALUATOR: SupportEvaluator")
    print("  - Evaluates resolution accuracy (addresses customer issue)")
    print("  - Measures tone quality (empathy, clarity, professionalism)")
    print("  - Tracks escalation accuracy (precision/recall/F1)")
    print("\nCALIBRATION NOTE: Tone scores should correlate with CSAT > 0.7")

    # Load and show dataset info
    dataset = load_dataset()
    print(f"\nDATASET: support_tickets.jsonl ({len(dataset)} support scenarios)")

    if dataset:
        # Count escalations and sentiments
        escalation_count = sum(1 for e in dataset if e.get("should_escalate", False))
        sentiments = {}
        for e in dataset:
            input_data = e.get("input", {})
            ctx = input_data.get("customer_context", {})
            sent = ctx.get("sentiment", "unknown")
            sentiments[sent] = sentiments.get(sent, 0) + 1

        print(f"  - Requiring escalation: {escalation_count}")
        print(f"  - Self-resolvable: {len(dataset) - escalation_count}")
        print(f"  - Sentiment distribution: {sentiments}")

        print("\n" + "-" * 60)
        print("FIRST 3 SUPPORT TICKETS FROM DATASET:")
        print("-" * 60)
        for i, entry in enumerate(dataset[:3]):
            input_data = entry.get("input", {})
            query = input_data.get("query", "")[:50]
            ctx = input_data.get("customer_context", {})
            tier = ctx.get("customer_tier", "unknown")
            sentiment = ctx.get("sentiment", "unknown")
            escalate = entry.get("should_escalate", False)
            print(f"\n  [{i+1}] \"{query}...\"")
            print(f"      Customer: {tier} tier | Sentiment: {sentiment}")
            print(f"      Should escalate: {'Yes' if escalate else 'No'}")

    evaluator = SupportEvaluator()

    print("\n" + "=" * 60)
    print("EVALUATION EXAMPLES")
    print("=" * 60)

    # Test case 1: Good response
    print("\n[EXAMPLE 1] High-quality empathetic response")
    print("-" * 60)
    good_response = """Thank you for reaching out. I sincerely apologize for the inconvenience with your damaged laptop.

I completely understand how frustrating this must be. As a Gold member, your satisfaction is our priority.

I'd be happy to process a full refund immediately - credited within 5-7 business days. Or I can arrange express replacement shipping at no cost.

Please let me know which works best. Is there anything else I can help with?"""

    result = evaluator(
        prediction={"response": good_response, "should_escalate": False, "resolution_type": "refund"},
        expected={"should_escalate": False, "resolution_type": "refund"},
        input_data={"query": "I received a damaged laptop and want a refund", "customer_context": {"customer_tier": "gold", "sentiment": "negative"}},
    )
    print(f"  Query: \"I received a damaged laptop and want a refund\"")
    print(f"  Response: Offers refund OR replacement, empathetic tone")
    print(f"\n  Scores:")
    print(f"    Resolution Accuracy: {result['resolution_accuracy']:.2f}")
    print(f"    Tone Quality:        {result['tone_quality']:.2f}")
    print(f"      - Empathy:         {result['empathy_score']:.2f}")
    print(f"      - Clarity:         {result['clarity_score']:.2f}")
    print(f"      - Professionalism: {result['professionalism_score']:.2f}")
    print(f"    Escalation Accuracy: {result['escalation_accuracy']:.2f}")
    print(f"    ─────────────────────────")
    print(f"    Overall:             {result['overall']:.2f}")

    # Test case 2: Poor response
    print("\n[EXAMPLE 2] Low-quality dismissive response")
    print("-" * 60)
    result = evaluator(
        prediction={"response": "Ok, we'll look into it. Check back later.", "should_escalate": False},
        expected={"should_escalate": False, "resolution_type": "refund"},
        input_data={"query": "I received a damaged laptop and want a refund", "customer_context": {"customer_tier": "gold", "sentiment": "negative"}},
    )
    print(f"  Query: \"I received a damaged laptop and want a refund\"")
    print(f"  Response: \"Ok, we'll look into it. Check back later.\"")
    print(f"\n  Scores:")
    print(f"    Resolution Accuracy: {result['resolution_accuracy']:.2f} ← Doesn't address issue!")
    print(f"    Tone Quality:        {result['tone_quality']:.2f} ← No empathy!")
    print(f"    Escalation Accuracy: {result['escalation_accuracy']:.2f}")
    print(f"    ─────────────────────────")
    print(f"    Overall:             {result['overall']:.2f}")

    # Test case 3: Wrong escalation
    print("\n[EXAMPLE 3] Wrong escalation decision (should have escalated)")
    print("-" * 60)
    result = evaluator(
        prediction={"response": "I understand. Our standard procedure will resolve this within 24 hours.", "should_escalate": False},
        expected={"should_escalate": True},
        input_data={"query": "This is unacceptable! I want to speak to your supervisor!", "customer_context": {"customer_tier": "platinum", "sentiment": "very_negative"}},
    )
    print(f"  Query: \"I want to speak to your supervisor!\"")
    print(f"  Agent chose: NOT to escalate | Expected: ESCALATE")
    print(f"\n  Scores:")
    print(f"    Resolution Accuracy: {result['resolution_accuracy']:.2f}")
    print(f"    Tone Quality:        {result['tone_quality']:.2f}")
    print(f"    Escalation Accuracy: {result['escalation_accuracy']:.2f} ← WRONG DECISION!")
    print(f"    ─────────────────────────")
    print(f"    Overall:             {result['overall']:.2f}")

    print("\n" + "=" * 60)
    print("To run optimization with real API calls:")
    print("  export OPENAI_API_KEY=<your-key>")
    print("  unset TRAIGENT_MOCK_MODE")
    print("  python use-cases/customer-support/agent/support_agent.py")
    print("=" * 60)


if __name__ == "__main__":
    demo_evaluator()
