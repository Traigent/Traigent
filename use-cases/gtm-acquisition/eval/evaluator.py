#!/usr/bin/env python3
"""
LLM-as-Judge Evaluator for GTM & Acquisition Agent

This evaluator scores outbound sales messages on:
1. Message Quality Score (ICP fit, personalization, value proposition, tone)
2. Compliance Pass Rate (spam indicators, banned phrases, professional tone)

Based on the Traigent Agent Optimization Guide specifications.

Calibration Notes:
- LLM-as-judge scores should be calibrated against human raters
- Target Cohen's κ > 0.7 for inter-rater reliability
- Calibration set: Have sales leadership score 50+ messages
- Adjust rubric anchors until LLM-human agreement meets threshold
"""

import re
from dataclasses import dataclass
from typing import Any

# Banned phrases that violate compliance
BANNED_PHRASES = [
    "act now",
    "limited time",
    "urgent",
    "don't miss out",
    "free money",
    "guaranteed",
    "no risk",
    "100% satisfaction",
    "click here",
    "buy now",
    "discount",
    "offer expires",
    "once in a lifetime",
    "winner",
    "congratulations you've won",
]

# Spam trigger words
SPAM_TRIGGERS = [
    "!!!",
    "???",
    "$$$",
    "FREE",
    "URGENT",
    "ACT NOW",
    "LIMITED",
    "EXCLUSIVE OFFER",
    "EARN EXTRA CASH",
    "DOUBLE YOUR",
]


@dataclass
class MessageQualityResult:
    """Result of message quality evaluation."""

    icp_fit: float  # 1-5 scale
    personalization: float  # 1-5 scale
    value_proposition: float  # 1-5 scale
    tone_appropriateness: float  # 1-5 scale
    overall_quality: float  # Weighted average
    compliance_passed: bool
    compliance_issues: list[str]


class MessageQualityEvaluator:
    """
    LLM-as-judge evaluator for GTM outbound messages.

    Evaluates messages on quality dimensions and compliance checks.
    """

    def __init__(self):
        """Initialize the evaluator."""
        self.quality_weights = {
            "icp_fit": 0.3,
            "personalization": 0.3,
            "value_proposition": 0.25,
            "tone_appropriateness": 0.15,
        }

    def __call__(
        self,
        prediction: str,
        expected: str | None,
        input_data: dict[str, Any],
    ) -> dict[str, float]:
        """
        Evaluate a generated message.

        Args:
            prediction: The generated outbound message
            expected: The gold standard message (if available)
            input_data: The input data containing lead info

        Returns:
            Dictionary of metric scores
        """
        result = self.evaluate_message(prediction, input_data)

        return {
            "message_quality": result.overall_quality / 5.0,  # Normalize to 0-1
            "compliance": 1.0 if result.compliance_passed else 0.0,
            "icp_fit": result.icp_fit / 5.0,
            "personalization": result.personalization / 5.0,
            "value_proposition": result.value_proposition / 5.0,
            "tone": result.tone_appropriateness / 5.0,
        }

    def evaluate_message(
        self,
        message: str,
        input_data: dict[str, Any],
    ) -> MessageQualityResult:
        """
        Evaluate a message comprehensively.

        Args:
            message: The generated message
            input_data: Input data containing lead information

        Returns:
            MessageQualityResult with all scores
        """
        lead = input_data.get("lead", {})

        # Evaluate each dimension
        icp_fit = self._evaluate_icp_fit(message, lead)
        personalization = self._evaluate_personalization(message, lead)
        value_prop = self._evaluate_value_proposition(message, input_data)
        tone = self._evaluate_tone(message)

        # Calculate overall quality
        overall = (
            self.quality_weights["icp_fit"] * icp_fit
            + self.quality_weights["personalization"] * personalization
            + self.quality_weights["value_proposition"] * value_prop
            + self.quality_weights["tone_appropriateness"] * tone
        )

        # Check compliance
        compliance_passed, issues = self._check_compliance(message)

        return MessageQualityResult(
            icp_fit=icp_fit,
            personalization=personalization,
            value_proposition=value_prop,
            tone_appropriateness=tone,
            overall_quality=overall,
            compliance_passed=compliance_passed,
            compliance_issues=issues,
        )

    def _evaluate_icp_fit(self, message: str, lead: dict) -> float:
        """
        Evaluate how well the message addresses the lead's ICP characteristics.

        Scoring rubric:
        5: Directly addresses industry, company size, and role challenges
        4: Addresses most ICP characteristics
        3: Generic industry reference
        2: Minimal ICP relevance
        1: No ICP fit
        """
        score = 1.0
        message_lower = message.lower()

        # Check if industry is mentioned or addressed
        industry = lead.get("industry", "").lower()
        if industry and industry in message_lower:
            score += 1.0

        # Check if role/title is acknowledged
        title = lead.get("title", "").lower()
        title_keywords = ["vp", "director", "head", "cto", "ceo", "manager", "lead"]
        if any(kw in title.lower() for kw in title_keywords):
            if any(kw in message_lower for kw in ["leader", "executive", "team"]):
                score += 0.5

        # Check if company size challenges are addressed
        company_size = lead.get("company_size", "")
        if company_size:
            if "scaling" in message_lower or "growth" in message_lower:
                score += 1.0

        # Check if pain points are addressed
        pain_points = lead.get("pain_points", [])
        pain_points_addressed = sum(
            1 for pp in pain_points if pp.lower() in message_lower
        )
        if pain_points_addressed > 0:
            score += min(pain_points_addressed * 0.5, 1.5)

        return min(score, 5.0)

    def _evaluate_personalization(self, message: str, lead: dict) -> float:
        """
        Evaluate personalization depth.

        Scoring rubric:
        5: Multiple personalized elements (name, company, news, specific context)
        4: Good personalization with 2-3 elements
        3: Basic personalization (name + company)
        2: Only name mentioned
        1: No personalization
        """
        score = 1.0
        message_lower = message.lower()

        # Check for name usage
        name = lead.get("name", "")
        if name:
            first_name = name.split()[0].lower()
            if first_name in message_lower:
                score += 1.0

        # Check for company mention
        company = lead.get("company", "").lower()
        if company and company in message_lower:
            score += 1.0

        # Check for recent news reference
        recent_news = lead.get("recent_news", "").lower()
        if recent_news:
            # Check for keywords from the news
            news_keywords = recent_news.split()[:3]  # First 3 words
            if any(kw.lower() in message_lower for kw in news_keywords if len(kw) > 3):
                score += 1.0

        # Check for specific pain point mention
        pain_points = lead.get("pain_points", [])
        if any(pp.lower() in message_lower for pp in pain_points):
            score += 1.0

        return min(score, 5.0)

    def _evaluate_value_proposition(
        self, message: str, input_data: dict
    ) -> float:
        """
        Evaluate clarity and relevance of value proposition.

        Scoring rubric:
        5: Clear, specific value prop tied to lead's needs
        4: Clear value prop with some relevance
        3: Generic but present value prop
        2: Vague value mention
        1: No value proposition
        """
        score = 1.0
        message_lower = message.lower()

        # Check for product mention
        product = input_data.get("product", "").lower()
        if product:
            product_keywords = product.split()[:2]  # First 2 words
            if any(kw.lower() in message_lower for kw in product_keywords if len(kw) > 2):
                score += 1.0

        # Check for benefit language
        benefit_keywords = [
            "help",
            "reduce",
            "improve",
            "increase",
            "save",
            "faster",
            "better",
            "automate",
            "streamline",
            "optimize",
        ]
        benefits_found = sum(1 for kw in benefit_keywords if kw in message_lower)
        score += min(benefits_found * 0.5, 1.5)

        # Check for specific metrics or outcomes
        metrics_patterns = [
            r"\d+%",
            r"\d+x",
            "roi",
            "cost",
            "time",
            "efficiency",
        ]
        metrics_found = sum(
            1 for pattern in metrics_patterns if re.search(pattern, message_lower)
        )
        score += min(metrics_found * 0.5, 1.0)

        # Check for call-to-action
        cta_keywords = ["call", "chat", "conversation", "discuss", "explore", "demo"]
        if any(kw in message_lower for kw in cta_keywords):
            score += 0.5

        return min(score, 5.0)

    def _evaluate_tone(self, message: str) -> float:
        """
        Evaluate tone appropriateness.

        Scoring rubric:
        5: Perfect professional tone with appropriate warmth
        4: Professional with minor issues
        3: Acceptable but generic
        2: Too casual or too formal
        1: Inappropriate tone
        """
        score = 3.0  # Start at acceptable

        message_lower = message.lower()

        # Professional greeting
        if any(
            greeting in message_lower
            for greeting in ["hi ", "hello ", "dear "]
        ):
            score += 0.5

        # Professional sign-off
        if any(
            signoff in message_lower
            for signoff in ["best", "regards", "sincerely", "cheers"]
        ):
            score += 0.5

        # Penalize overly casual language
        casual_markers = ["yo", "hey!", "sup", "gonna", "wanna", "lol", "omg"]
        if any(marker in message_lower for marker in casual_markers):
            score -= 1.0

        # Penalize aggressive language
        aggressive_markers = ["you must", "you need to", "don't miss", "act fast"]
        if any(marker in message_lower for marker in aggressive_markers):
            score -= 1.0

        # Reward empathetic language
        empathetic_markers = ["understand", "imagine", "know", "appreciate"]
        if any(marker in message_lower for marker in empathetic_markers):
            score += 0.5

        return max(min(score, 5.0), 1.0)

    def _check_compliance(self, message: str) -> tuple[bool, list[str]]:
        """
        Check message for compliance issues.

        Returns:
            Tuple of (passed, list of issues)
        """
        issues = []
        message_lower = message.lower()

        # Check for banned phrases
        for phrase in BANNED_PHRASES:
            if phrase in message_lower:
                issues.append(f"Banned phrase detected: '{phrase}'")

        # Check for spam triggers
        for trigger in SPAM_TRIGGERS:
            if trigger in message.upper():
                issues.append(f"Spam trigger detected: '{trigger}'")

        # Check message length (too short or too long)
        word_count = len(message.split())
        if word_count < 30:
            issues.append(f"Message too short ({word_count} words)")
        elif word_count > 200:
            issues.append(f"Message too long ({word_count} words)")

        # Check for excessive punctuation
        if message.count("!") > 2:
            issues.append("Excessive exclamation marks")
        if message.count("?") > 3:
            issues.append("Excessive question marks")

        # Check for all caps words (excluding common acronyms)
        words = message.split()
        all_caps_words = [
            w for w in words
            if w.isupper() and len(w) > 2 and w not in ["CEO", "CTO", "VP", "AI", "ML", "API", "SaaS", "ROI"]
        ]
        if len(all_caps_words) > 1:
            issues.append("Excessive capitalization")

        return len(issues) == 0, issues


def load_dataset() -> list[dict]:
    """Load the leads dataset."""
    import json
    from pathlib import Path

    dataset_path = Path(__file__).parent.parent / "datasets" / "leads_dataset.jsonl"
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
    Demo the GTM & Acquisition evaluator.

    This runs in MOCK MODE - no API calls are made.
    The evaluator uses heuristic rules to score messages on:
    - ICP Fit (1-5): How well message addresses lead's characteristics
    - Personalization (1-5): Use of name, company, news, pain points
    - Value Proposition (1-5): Clarity of benefits and outcomes
    - Tone (1-5): Professional yet warm communication style
    - Compliance: Checks for spam triggers and banned phrases
    """
    print("=" * 60)
    print("GTM & ACQUISITION AGENT - Evaluator Demo")
    print("=" * 60)
    print("\nMODE: Mock (heuristic-based scoring, no API calls)")
    print("\nEVALUATOR: MessageQualityEvaluator")
    print("  - Scores outbound sales messages on quality dimensions")
    print("  - Checks compliance against spam/banned phrase rules")
    print("  - Calibration target: Cohen's κ > 0.7 vs human raters")

    # Load and show dataset info
    dataset = load_dataset()
    print(f"\nDATASET: leads_dataset.jsonl ({len(dataset)} lead profiles)")

    if dataset:
        print("\n" + "-" * 60)
        print("FIRST 3 LEADS FROM DATASET:")
        print("-" * 60)
        for i, entry in enumerate(dataset[:3]):
            input_data = entry.get("input", {})
            lead = input_data.get("lead", {})
            print(f"\n  [{i+1}] {lead.get('name', 'N/A')} - {lead.get('title', 'N/A')}")
            print(f"      Company: {lead.get('company', 'N/A')} ({lead.get('industry', 'N/A')})")
            print(f"      Pain Points: {', '.join(lead.get('pain_points', [])[:2])}")

    # Initialize evaluator
    evaluator = MessageQualityEvaluator()

    print("\n" + "=" * 60)
    print("EVALUATION EXAMPLES")
    print("=" * 60)

    # Sample input
    input_data = {
        "lead": {
            "name": "Sarah Chen",
            "title": "VP of Engineering",
            "company": "TechCorp",
            "industry": "SaaS",
            "company_size": "200-500",
            "recent_news": "Just raised Series B",
            "pain_points": ["scaling infrastructure", "engineering hiring"],
        },
        "product": "AI-powered DevOps platform",
    }

    # Good message example
    good_message = """Hi Sarah,

Congratulations on TechCorp's Series B! As you scale your engineering team, infrastructure complexity often becomes a bottleneck.

Our AI-powered DevOps platform helps engineering leaders like yourself automate deployment pipelines and reduce incident response time by 60%. Companies at your stage typically see ROI within 3 months.

Would you be open to a 15-minute call to see if we might be a fit?

Best,
Alex"""

    # Bad message example
    bad_message = """URGENT!!! ACT NOW!!!

Don't miss this LIMITED TIME OFFER!!! Our product is AMAZING and will GUARANTEE you success!!!

BUY NOW and get FREE bonus!!!

Click here to learn more!!!"""

    print("\n[EXAMPLE 1] Well-crafted personalized message")
    print("-" * 60)
    print(f"Lead: {input_data['lead']['name']} @ {input_data['lead']['company']}")
    print(f"Message preview: \"{good_message[:80]}...\"")
    print("\nScores:")
    good_result = evaluator.evaluate_message(good_message, input_data)
    print(f"  ICP Fit:          {good_result.icp_fit}/5")
    print(f"  Personalization:  {good_result.personalization}/5")
    print(f"  Value Proposition:{good_result.value_proposition}/5")
    print(f"  Tone:             {good_result.tone_appropriateness}/5")
    print(f"  ─────────────────────────")
    print(f"  Overall Quality:  {good_result.overall_quality:.2f}/5")
    print(f"  Compliance:       {'✓ PASSED' if good_result.compliance_passed else '✗ FAILED'}")

    print("\n[EXAMPLE 2] Spammy non-compliant message")
    print("-" * 60)
    print(f"Lead: {input_data['lead']['name']} @ {input_data['lead']['company']}")
    print(f"Message preview: \"{bad_message[:60]}...\"")
    print("\nScores:")
    bad_result = evaluator.evaluate_message(bad_message, input_data)
    print(f"  ICP Fit:          {bad_result.icp_fit}/5")
    print(f"  Personalization:  {bad_result.personalization}/5")
    print(f"  Value Proposition:{bad_result.value_proposition}/5")
    print(f"  Tone:             {bad_result.tone_appropriateness}/5")
    print(f"  ─────────────────────────")
    print(f"  Overall Quality:  {bad_result.overall_quality:.2f}/5")
    print(f"  Compliance:       {'✓ PASSED' if bad_result.compliance_passed else '✗ FAILED'}")
    if bad_result.compliance_issues:
        print(f"  Issues ({len(bad_result.compliance_issues)}):")
        for issue in bad_result.compliance_issues[:5]:
            print(f"    - {issue}")
        if len(bad_result.compliance_issues) > 5:
            print(f"    ... and {len(bad_result.compliance_issues) - 5} more")

    print("\n" + "=" * 60)
    print("To run optimization with real API calls:")
    print("  export OPENAI_API_KEY=<your-key>")
    print("  unset TRAIGENT_MOCK_MODE")
    print("  python use-cases/gtm-acquisition/agent/gtm_agent.py")
    print("=" * 60)


if __name__ == "__main__":
    demo_evaluator()
