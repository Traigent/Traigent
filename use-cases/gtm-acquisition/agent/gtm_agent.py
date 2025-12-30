#!/usr/bin/env python3
"""
GTM & Acquisition Agent - Outbound Message Generator

This agent generates personalized outbound sales messages for leads based on
ICP (Ideal Customer Profile) criteria. It optimizes for message quality and
compliance.

Usage:
    export TRAIGENT_MOCK_MODE=true
    python use-cases/gtm-acquisition/agent/gtm_agent.py
"""

import asyncio
import os
import sys
from pathlib import Path
from typing import Any

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Import evaluator from sibling directory
import importlib.util

import traigent
from traigent.api.decorators import EvaluationOptions, ExecutionOptions

_evaluator_path = Path(__file__).parent.parent / "eval" / "evaluator.py"
_spec = importlib.util.spec_from_file_location("gtm_evaluator", _evaluator_path)
_evaluator_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_evaluator_module)
MessageQualityEvaluator = _evaluator_module.MessageQualityEvaluator


def format_lead_context(lead: dict[str, Any]) -> str:
    """Format lead information for the prompt."""
    return f"""
Lead Information:
- Name: {lead.get('name', 'Unknown')}
- Title: {lead.get('title', 'Unknown')}
- Company: {lead.get('company', 'Unknown')}
- Industry: {lead.get('industry', 'Unknown')}
- Company Size: {lead.get('company_size', 'Unknown')}
- Recent News: {lead.get('recent_news', 'None available')}
- Pain Points: {', '.join(lead.get('pain_points', []))}
"""


MESSAGE_GENERATION_PROMPT = """You are an expert SDR (Sales Development Representative) writing a personalized outbound message.

{lead_context}

Product Being Sold: {product}
Sender: {sender_name}

Personalization Style: {personalization_depth}
Tone: {tone}

Guidelines:
1. Start with a personalized hook based on recent news or their role
2. Connect their pain points to the product value
3. Keep it concise (under 150 words)
4. Include a soft call-to-action
5. Be professional but {tone}
6. Sign off with the sender's first name

Write the outbound message:"""


@traigent.optimize(
    configuration_space={
        "model": ["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o"],
        "temperature": [0.3, 0.5, 0.7, 0.9],
        "personalization_depth": ["basic", "moderate", "deep"],
        "tone": ["professional", "friendly", "consultative"],
    },
    objectives=["message_quality", "compliance", "cost"],
    evaluation=EvaluationOptions(
        eval_dataset="use-cases/gtm-acquisition/datasets/leads_dataset.jsonl",
        # MessageQualityEvaluator has scoring_function interface: (prediction, expected, input_data) -> dict
        scoring_function=MessageQualityEvaluator(),
    ),
    execution=ExecutionOptions(execution_mode="edge_analytics"),
)
def gtm_outreach_agent(
    lead: dict[str, Any],
    product: str,
    sender_name: str,
) -> str:
    """
    Generate a personalized outbound sales message for a lead.

    Args:
        lead: Lead information dictionary containing name, title, company, etc.
        product: The product being sold
        sender_name: Name of the sender (SDR)

    Returns:
        Personalized outbound message string
    """
    # Get current configuration (works during optimization and after apply_best_config)
    config = traigent.get_config()

    # Extract tuned variables with defaults
    model = config.get("model", "gpt-4o-mini")
    temperature = config.get("temperature", 0.5)
    personalization_depth = config.get("personalization_depth", "moderate")
    tone = config.get("tone", "professional")

    # Format the lead context
    lead_context = format_lead_context(lead)

    # Build the prompt
    prompt = MESSAGE_GENERATION_PROMPT.format(
        lead_context=lead_context,
        product=product,
        sender_name=sender_name,
        personalization_depth=personalization_depth,
        tone=tone,
    )

    # Use LangChain for LLM call (Traigent intercepts and optimizes)
    try:
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(
            model=model,
            temperature=temperature,
        )
        response = llm.invoke(prompt)
        return response.content
    except ImportError:
        # Fallback for mock mode without LangChain
        return f"""Hi {lead.get('name', 'there').split()[0]},

Congratulations on {lead.get('recent_news', 'your recent achievements')}! As {lead.get('title', 'a leader')} at {lead.get('company', 'your company')}, I imagine {lead.get('pain_points', ['scaling'])[0] if lead.get('pain_points') else 'growth'} is top of mind.

{product} helps companies in {lead.get('industry', 'your space')} tackle exactly these challenges. Would you be open to a 15-minute call to explore?

Best,
{sender_name.split()[0] if sender_name else 'Alex'}"""


async def run_optimization():
    """Run the GTM agent optimization."""
    print("=" * 60)
    print("GTM & Acquisition Agent - Traigent Optimization")
    print("=" * 60)

    # Check if mock mode is enabled
    mock_mode = os.environ.get("TRAIGENT_MOCK_MODE", "false").lower() == "true"
    print(f"\nMock Mode: {'Enabled' if mock_mode else 'Disabled'}")

    if not mock_mode:
        print("\nWARNING: Running without mock mode will incur API costs!")
        print("Set TRAIGENT_MOCK_MODE=true for testing.\n")

    print("\nStarting optimization...")
    print("Configuration Space:")
    print("  - Models: gpt-3.5-turbo, gpt-4o-mini, gpt-4o")
    print("  - Temperature: 0.3, 0.5, 0.7, 0.9")
    print("  - Personalization: basic, moderate, deep")
    print("  - Tone: professional, friendly, consultative")
    print("\nObjectives: message_quality, compliance, cost")
    print("-" * 60)

    # Run optimization
    results = await gtm_outreach_agent.optimize(
        algorithm="random",
        max_trials=20,
    )

    # Display results
    print("\n" + "=" * 60)
    print("OPTIMIZATION RESULTS")
    print("=" * 60)
    print("\nBest Configuration:")
    for key, value in results.best_config.items():
        print(f"  {key}: {value}")
    print(f"\nBest Score: {results.best_score:.4f}")

    # Apply best config
    gtm_outreach_agent.apply_best_config(results)
    print("\nBest configuration applied!")

    # Test with a sample lead
    print("\n" + "-" * 60)
    print("Testing optimized agent with sample lead...")
    print("-" * 60)

    sample_lead = {
        "name": "Test User",
        "title": "VP of Engineering",
        "company": "TestCorp",
        "industry": "SaaS",
        "company_size": "100-200",
        "recent_news": "Just launched new product",
        "pain_points": ["scaling", "developer productivity"],
    }

    message = gtm_outreach_agent(
        lead=sample_lead,
        product="AI DevOps Platform",
        sender_name="Demo SDR",
    )

    print(f"\nGenerated Message:\n{message}")

    return results


def main():
    """Main entry point."""
    asyncio.run(run_optimization())


if __name__ == "__main__":
    main()
