#!/usr/bin/env python3
"""
LLM-as-Judge Evaluator for GTM & Acquisition Agent

This evaluator scores outbound sales messages on:
1. Message Quality Score (ICP fit, personalization, value proposition, tone)
2. Compliance Pass Rate (spam indicators, banned phrases, professional tone)

Supports two modes:
- MOCK MODE (default): Uses heuristic rules for fast, free evaluation
- REAL MODE: Uses actual LLM calls for agent and LLM-as-judge evaluation

Usage:
  Mock mode: python evaluator.py  (default, uses heuristics)
  Real mode: TRAIGENT_MOCK_LLM=false python evaluator.py  (requires OPENAI_API_KEY)
"""

import os
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

# ============================================================================
# PROMPTS FOR REAL LLM MODE
# ============================================================================

# Prompt for the SDR agent to generate outbound messages
AGENT_PROMPT = """You are an SDR (Sales Development Rep) writing a personalized outreach email.

LEAD INFORMATION:
- Name: {name}
- Title: {title}
- Company: {company} ({industry})
- Company Size: {company_size}
- Pain Points: {pain_points}
- Recent News: {recent_news}

PRODUCT: {product}

Write a short, personalized outreach email (50-150 words) that:
1. Opens with something specific to their situation (news, pain point, or industry)
2. Briefly mentions how our product helps with their specific challenges
3. Ends with a soft call-to-action (suggest a brief call, not a hard sell)

Keep it conversational and professional. No spam phrases like "ACT NOW" or "LIMITED TIME".
"""

# Prompt for LLM-as-judge to score messages
JUDGE_PROMPT = """You are evaluating an SDR outreach email for quality.

LEAD CONTEXT:
- Name: {name}
- Title: {title}
- Company: {company} ({industry})
- Pain Points: {pain_points}

MESSAGE TO EVALUATE:
{message}

Score each dimension from 1-5:

ICP_FIT: How well does it address their industry/role/challenges? (1=generic, 5=highly specific)
PERSONALIZATION: Does it use their name, company, news, pain points? (1=none, 5=deeply personalized)
VALUE_PROPOSITION: Is the benefit clear and relevant? (1=vague, 5=compelling and specific)
TONE: Professional but warm? (1=spammy/pushy, 5=perfect professional warmth)

Respond in EXACTLY this format:
ICP_FIT: [1-5]
PERSONALIZATION: [1-5]
VALUE_PROPOSITION: [1-5]
TONE: [1-5]
"""


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

    def is_mock_mode(self) -> bool:
        """Check if running in mock mode."""
        return os.environ.get("TRAIGENT_MOCK_LLM", "true").lower() == "true"

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

        # Use LLM-as-judge in real mode, heuristics in mock mode
        if not self.is_mock_mode():
            llm_scores = self._llm_judge_evaluate(message, lead, input_data)
            if llm_scores:
                icp_fit = llm_scores.get("icp_fit", 3.0)
                personalization = llm_scores.get("personalization", 3.0)
                value_prop = llm_scores.get("value_proposition", 3.0)
                tone = llm_scores.get("tone", 3.0)
            else:
                # Fallback to heuristics if LLM fails
                icp_fit = self._evaluate_icp_fit(message, lead)
                personalization = self._evaluate_personalization(message, lead)
                value_prop = self._evaluate_value_proposition(message, input_data)
                tone = self._evaluate_tone(message)
        else:
            # Mock mode: use heuristics
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

    def _llm_judge_evaluate(
        self, message: str, lead: dict, input_data: dict
    ) -> dict[str, float] | None:
        """Use LLM-as-judge to evaluate message quality."""
        try:
            from openai import OpenAI

            client = OpenAI()
            prompt = JUDGE_PROMPT.format(
                name=lead.get("name", "Unknown"),
                title=lead.get("title", "Unknown"),
                company=lead.get("company", "Unknown"),
                industry=lead.get("industry", "Unknown"),
                pain_points=", ".join(lead.get("pain_points", [])),
                message=message,
            )

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )

            # Parse scores from response using robust regex
            content = response.choices[0].message.content
            scores = {}
            for key in ["icp_fit", "personalization", "value_proposition", "tone"]:
                # Match patterns like "ICP_FIT: 4" or "ICP_FIT: 4/5" or "ICP_FIT: [4]"
                pattern = rf"{key.upper()}\s*:\s*\[?(\d(?:\.\d)?)"
                match = re.search(pattern, content, re.IGNORECASE)
                if match:
                    scores[key] = float(match.group(1))

            return scores if scores else None
        except Exception as e:
            print(f"  [LLM Judge Error: {e}]")
            return None

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

    def _evaluate_value_proposition(self, message: str, input_data: dict) -> float:
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
            if any(
                kw.lower() in message_lower for kw in product_keywords if len(kw) > 2
            ):
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
        if any(greeting in message_lower for greeting in ["hi ", "hello ", "dear "]):
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
            w
            for w in words
            if w.isupper()
            and len(w) > 2
            and w not in ["CEO", "CTO", "VP", "AI", "ML", "API", "SaaS", "ROI"]
        ]
        if len(all_caps_words) > 1:
            issues.append("Excessive capitalization")

        return len(issues) == 0, issues


def generate_message_with_llm(input_data: dict) -> str:
    """Generate an outbound message using LLM (real mode only)."""
    try:
        from openai import OpenAI

        client = OpenAI()
        lead = input_data.get("lead", {})
        prompt = AGENT_PROMPT.format(
            name=lead.get("name", "Unknown"),
            title=lead.get("title", "Unknown"),
            company=lead.get("company", "Unknown"),
            industry=lead.get("industry", "Unknown"),
            company_size=lead.get("company_size", "Unknown"),
            pain_points=", ".join(lead.get("pain_points", [])),
            recent_news=lead.get("recent_news", "N/A"),
            product=input_data.get("product", "our product"),
        )

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"[Error generating message: {e}]"


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


def run_optimization(num_configs: int = 5, num_examples: int = 10):
    """
    Run a mini optimization loop testing different prompt configurations.

    Args:
        num_configs: Number of configurations to test
        num_examples: Number of examples to evaluate per config
    """
    from openai import OpenAI

    client = OpenAI()
    dataset = load_dataset()[:num_examples]
    evaluator = MessageQualityEvaluator()

    # Define different prompt configurations to test
    configs = [
        {
            "name": "baseline",
            "temperature": 0.7,
            "system_prompt": "You are an SDR writing outreach emails.",
        },
        {
            "name": "formal",
            "temperature": 0.3,
            "system_prompt": "You are a professional sales representative. Write formal, polished emails.",
        },
        {
            "name": "casual",
            "temperature": 0.9,
            "system_prompt": "You are a friendly SDR. Write conversational, warm emails.",
        },
        {
            "name": "data-driven",
            "temperature": 0.5,
            "system_prompt": "You are an SDR focused on metrics and ROI. Include specific numbers.",
        },
        {
            "name": "empathy-first",
            "temperature": 0.6,
            "system_prompt": "You are an SDR who leads with empathy. Focus on pain points.",
        },
    ][:num_configs]

    print("\n" + "=" * 70)
    print("OPTIMIZATION RUN: Testing Different Prompt Configurations")
    print("=" * 70)
    print(f"\nConfigurations: {num_configs}")
    print(f"Examples per config: {num_examples}")
    print(f"Total LLM calls: {num_configs * num_examples}")

    results = []

    for config in configs:
        print(f"\n--- Config: {config['name']} (temp={config['temperature']}) ---")
        config_scores = []

        for i, entry in enumerate(dataset):
            input_data = entry.get("input", {})
            lead = input_data.get("lead", {})

            prompt = f"""{config['system_prompt']}

Write a personalized outreach email for:
- Name: {lead.get('name', 'Unknown')}
- Title: {lead.get('title', 'Unknown')}
- Company: {lead.get('company', 'Unknown')} ({lead.get('industry', 'Unknown')})
- Pain Points: {', '.join(lead.get('pain_points', []))}
- Recent News: {lead.get('recent_news', 'N/A')}

Product: {input_data.get('product', 'our product')}

Keep it under 150 words. End with a soft call-to-action."""

            try:
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=config["temperature"],
                    max_tokens=300,
                )
                message = response.choices[0].message.content
                scores = evaluator(message, None, input_data)
                config_scores.append(scores)
                status = "✓" if scores["compliance"] == 1.0 else "✗"
                print(
                    f"  [{i+1}/{num_examples}] quality={scores['message_quality']:.2f} {status}"
                )
            except Exception as e:
                print(f"  [{i+1}/{num_examples}] Error: {e}")
                config_scores.append({"message_quality": 0, "compliance": 0})

        avg_quality = sum(s["message_quality"] for s in config_scores) / len(
            config_scores
        )
        avg_compliance = sum(s["compliance"] for s in config_scores) / len(
            config_scores
        )
        results.append(
            {
                "config": config["name"],
                "temp": config["temperature"],
                "quality": avg_quality,
                "compliance": avg_compliance,
                "overall": (avg_quality + avg_compliance) / 2,
            }
        )

    # Print results table
    print("\n" + "=" * 70)
    print("RESULTS TABLE")
    print("=" * 70)
    print(
        f"\n{'Config':<15} {'Temp':<6} {'Quality':<10} {'Compliance':<12} {'Overall':<10}"
    )
    print("-" * 55)
    for r in sorted(results, key=lambda x: x["overall"], reverse=True):
        print(
            f"{r['config']:<15} {r['temp']:<6.1f} {r['quality']:.3f}      {r['compliance']*100:>5.0f}%       {r['overall']:.3f}"
        )

    best = max(results, key=lambda x: x["overall"])
    print("-" * 55)
    print(f"🏆 BEST: {best['config']} (score={best['overall']:.3f})")
    print("=" * 70)
    return results


def print_score_bar(label: str, score: float, max_score: float = 5.0, width: int = 20):
    """Print a visual score bar."""
    normalized = score / max_score
    filled = int(normalized * width)
    bar = "█" * filled + "░" * (width - filled)
    print(f"  {label:<18} {bar} {score:.1f}/{max_score:.0f}")


def demo_evaluator():
    """Demo the GTM & Acquisition evaluator with clear input/output examples."""
    mock_mode = os.environ.get("TRAIGENT_MOCK_LLM", "true").lower() == "true"

    print("=" * 70)
    print("GTM & ACQUISITION AGENT - Evaluator Demo")
    print("=" * 70)
    print(
        f"\nMODE: {'MOCK (heuristic rules)' if mock_mode else 'REAL (using OpenAI API)'}"
    )

    print(
        """
WHAT THIS AGENT DOES:
  An SDR (Sales Development Rep) agent that writes personalized outreach
  emails to sales leads. Given info about a lead, it generates a message.

HOW IT'S EVALUATED:"""
    )
    if mock_mode:
        print("  MOCK MODE: Using rule-based heuristics (fast, free, no API needed)")
    else:
        print("  REAL MODE: Using LLM-as-judge to score messages (requires API key)")
    print()

    # Load and show dataset info
    dataset = load_dataset()
    print(f"DATASET: {len(dataset)} lead profiles in leads_dataset.jsonl")

    if dataset:
        print("\n" + "-" * 70)
        print("SAMPLE DATA (first 2 entries):")
        print("-" * 70)
        for i, entry in enumerate(dataset[:2]):
            input_data = entry.get("input", {})
            lead = input_data.get("lead", {})
            output = entry.get("output", "")

            print(f"\n[Entry {i+1}]")
            print("  INPUT (lead info):")
            print(f"    Name:        {lead.get('name', 'N/A')}")
            print(f"    Title:       {lead.get('title', 'N/A')}")
            print(
                f"    Company:     {lead.get('company', 'N/A')} ({lead.get('industry', 'N/A')})"
            )
            print(f"    Pain Points: {', '.join(lead.get('pain_points', []))}")
            print(f"    Recent News: {lead.get('recent_news', 'N/A')}")
            print("\n  OUTPUT (expected message):")
            # Show first 150 chars of expected output
            preview = output[:150].replace("\n", " ")
            print(f'    "{preview}..."')

    # Initialize evaluator
    evaluator = MessageQualityEvaluator()

    print("\n" + "=" * 70)
    print("HOW SCORING WORKS:")
    print("=" * 70)
    print(
        """
The evaluator scores messages on these dimensions (1-5 scale each):

  - ICP Fit:          Does the message address the lead's industry/role/challenges?
  - Personalization:  Does it use the lead's name, company, news, pain points?
  - Value Proposition: Is the benefit clear? Are there specific outcomes/metrics?
  - Tone:             Is it professional but warm? Not too pushy or too formal?

Plus a compliance check (pass/fail):
  - No spam phrases like "ACT NOW", "LIMITED TIME", "GUARANTEED"
  - No excessive punctuation (!!!) or ALL CAPS
  - Appropriate message length (30-200 words)
"""
    )

    # Sample input for evaluation
    sample_input = {
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

    print("=" * 70)
    print("EVALUATION EXAMPLES:")
    print("=" * 70)

    # Example 1: Good message
    print("\n[GOOD MESSAGE] - Personalized, professional, compliant")
    print("-" * 70)
    print("Message:")
    for line in good_message.strip().split("\n"):
        print(f"  {line}")
    good_result = evaluator.evaluate_message(good_message, sample_input)
    print("\nScores:")
    print_score_bar("ICP Fit", good_result.icp_fit)
    print_score_bar("Personalization", good_result.personalization)
    print_score_bar("Value Proposition", good_result.value_proposition)
    print_score_bar("Tone", good_result.tone_appropriateness)
    print(f"  {'─' * 42}")
    print_score_bar("OVERALL QUALITY", good_result.overall_quality)
    print(
        f"  Compliance:        {'✓ PASSED' if good_result.compliance_passed else '✗ FAILED'}"
    )

    # Example 2: Borderline message (high quality but compliance issue)
    borderline_message = """Hi Sarah,

Congratulations on TechCorp's Series B! I know scaling infrastructure while hiring is a challenge.

Our platform has helped similar SaaS companies reduce deployment time by 60%. I'd love to offer you an exclusive discount to try it out.

Quick call this week?

Best,
Alex"""

    print("\n[BORDERLINE] - Good quality but compliance issue")
    print("-" * 70)
    print("Message:")
    for line in borderline_message.strip().split("\n"):
        print(f"  {line}")
    borderline_result = evaluator.evaluate_message(borderline_message, sample_input)
    print("\nScores:")
    print_score_bar("ICP Fit", borderline_result.icp_fit)
    print_score_bar("Personalization", borderline_result.personalization)
    print_score_bar("Value Proposition", borderline_result.value_proposition)
    print_score_bar("Tone", borderline_result.tone_appropriateness)
    print(f"  {'─' * 42}")
    print_score_bar("OVERALL QUALITY", borderline_result.overall_quality)
    print(
        f"  Compliance:        {'✓ PASSED' if borderline_result.compliance_passed else '✗ FAILED'}"
    )
    if borderline_result.compliance_issues:
        print(f"  Issue: {borderline_result.compliance_issues[0]}")

    # Example 3: Bad message
    print("\n[BAD MESSAGE] - Spammy, generic, non-compliant")
    print("-" * 70)
    print("Message:")
    for line in bad_message.strip().split("\n"):
        print(f"  {line}")
    bad_result = evaluator.evaluate_message(bad_message, sample_input)
    print("\nScores:")
    print_score_bar("ICP Fit", bad_result.icp_fit)
    print_score_bar("Personalization", bad_result.personalization)
    print_score_bar("Value Proposition", bad_result.value_proposition)
    print_score_bar("Tone", bad_result.tone_appropriateness)
    print(f"  {'─' * 42}")
    print_score_bar("OVERALL QUALITY", bad_result.overall_quality)
    print(
        f"  Compliance:        {'✓ PASSED' if bad_result.compliance_passed else '✗ FAILED'}"
    )
    if bad_result.compliance_issues:
        print(f"\n  Why it failed ({len(bad_result.compliance_issues)} issues):")
        for issue in bad_result.compliance_issues[:3]:
            print(f"    - {issue}")
        if len(bad_result.compliance_issues) > 3:
            print(f"    ... and {len(bad_result.compliance_issues) - 3} more")

    # In real mode, run optimization
    if not mock_mode:
        run_optimization(num_configs=5, num_examples=10)

    print("\n" + "=" * 70)
    print("HOW TO RUN:")
    print("  Mock mode (heuristics): python evaluator.py  (default)")
    print(
        "  Real mode (LLM+optimize): TRAIGENT_MOCK_LLM=false OPENAI_API_KEY=sk-... python evaluator.py"
    )
    print("=" * 70)


if __name__ == "__main__":
    demo_evaluator()
