"""Generate Bazak child-age-prompting evaluation dataset using GPT-4o.

Creates 50-100 test cases for evaluating whether a travel booking agent
correctly asks for children's ages when relevant.

Usage:
    python examples/child_age_demo/generate_dataset.py
"""

import json
import re
from pathlib import Path

from openai import OpenAI

OUTPUT_PATH = Path(__file__).parent / "bazak_child_age_dataset.jsonl"

CATEGORIES = [
    {
        "name": "explicit_children",
        "expected_behavior": "should_ask_ages",
        "count": 15,
        "description": (
            "Queries where children are explicitly mentioned. "
            "Use varied phrasing: 'my kids', 'my children', 'my daughter', "
            "'my son', 'the little ones', 'my boy/girl'. "
            "Vary child count (1-4). "
            "Vary travel types: flights, vacations, cruises, hotels, packages. "
            "Vary destinations: domestic US, Europe, Asia, beach, city, theme parks."
        ),
        "context_template": {"has_children": True},
        "examples": [
            "I need a flight to Paris for me and my two kids",
            "Looking for a vacation package to Cancun for myself and my three children",
            "Can you book flights to Orlando for me, my husband, and our daughter?",
        ],
    },
    {
        "name": "family_ambiguous",
        "expected_behavior": "should_ask_ages",
        "count": 10,
        "description": (
            "Queries that mention 'family' or imply children may be present "
            "without stating it explicitly. The agent SHOULD ask about children's ages "
            "because family travel commonly involves kids. "
            "Examples: 'family vacation', 'traveling with family', 'family trip', "
            "'the whole family', 'family getaway'."
        ),
        "context_template": {"has_children": None},
        "examples": [
            "We're planning a family vacation to Hawaii",
            "I'd like to book a family trip to Disney World",
            "Traveling with the whole family to London next summer",
        ],
    },
    {
        "name": "infants_toddlers",
        "expected_behavior": "should_ask_ages",
        "count": 8,
        "description": (
            "Queries mentioning very young children: babies, infants, toddlers. "
            "These are important because infant pricing (0-2) differs from child pricing (2-12). "
            "Vary between 'baby', 'infant', 'toddler', 'newborn', '1-year-old'."
        ),
        "context_template": {"has_children": True},
        "examples": [
            "I need to fly to Miami with my baby",
            "Booking a cruise for two adults and a toddler",
            "Can I bring my infant on the flight to Tokyo?",
        ],
    },
    {
        "name": "teens",
        "expected_behavior": "should_ask_ages",
        "count": 7,
        "description": (
            "Queries mentioning teenagers. Age matters because some airlines "
            "charge child rates up to 12, others up to 16. "
            "Use: 'teenager', 'teen', 'my 13-year-old', 'high schooler'."
        ),
        "context_template": {"has_children": True},
        "examples": [
            "Flying to New York with my teenager",
            "Need a hotel for me and my two teens in Barcelona",
        ],
    },
    {
        "name": "ages_already_provided",
        "expected_behavior": "should_ask_ages",
        "count": 5,
        "description": (
            "Queries where the user already provides some age info but the agent "
            "should still confirm/acknowledge ages for correct pricing. "
            "E.g., 'my 5-year-old and 8-year-old' — agent should confirm these ages "
            "and ask if there are additional children."
        ),
        "context_template": {"has_children": True},
        "examples": [
            "Book flights to Rome for me, my wife, my 5-year-old and 8-year-old",
            "Family trip to Bali — I have a 3-year-old and a 10-year-old",
        ],
    },
    {
        "name": "solo_travel",
        "expected_behavior": "should_not_ask_ages",
        "count": 8,
        "description": (
            "Solo traveler queries with no mention of children or family. "
            "Business trips, personal trips, adventures. "
            "Clear single-adult context."
        ),
        "context_template": {"has_children": False},
        "examples": [
            "Book me a flight to London next Tuesday",
            "I need a one-way ticket to San Francisco",
            "Looking for a solo backpacking trip through Southeast Asia",
        ],
    },
    {
        "name": "couples",
        "expected_behavior": "should_not_ask_ages",
        "count": 8,
        "description": (
            "Couple/partner travel with no children mentioned. "
            "Honeymoons, anniversaries, romantic getaways, 'me and my wife/husband/partner'. "
            "These should NOT trigger age questions."
        ),
        "context_template": {"has_children": False},
        "examples": [
            "Anniversary trip to Italy for me and my wife",
            "Honeymoon in the Maldives for two",
            "Romantic getaway to Napa Valley for me and my partner",
        ],
    },
    {
        "name": "business_travel",
        "expected_behavior": "should_not_ask_ages",
        "count": 6,
        "description": (
            "Business/corporate travel. Conferences, meetings, team trips. "
            "Clearly adult-only professional context."
        ),
        "context_template": {"has_children": False},
        "examples": [
            "I need flights for a business trip to Chicago next week",
            "Book a hotel near the convention center in Las Vegas for our team of 4",
        ],
    },
    {
        "name": "adult_groups",
        "expected_behavior": "should_not_ask_ages",
        "count": 6,
        "description": (
            "Groups of adults traveling together. Friends trips, reunions, "
            "bachelor/bachelorette parties, 'girls trip', 'guys weekend'. "
            "No children implied."
        ),
        "context_template": {"has_children": False},
        "examples": [
            "Planning a girls trip to Miami for 6 of us",
            "Bachelor party in Vegas for 8 guys",
            "College reunion trip to Cabo for 5 friends",
        ],
    },
    {
        "name": "adult_children_mentioned",
        "expected_behavior": "should_not_ask_ages",
        "count": 5,
        "description": (
            "Queries that mention 'son', 'daughter', or family members who are "
            "clearly adults (age 18+). The agent should NOT ask for ages. "
            "E.g., 'my son who's 25', 'my adult daughter', 'my grown kids'."
        ),
        "context_template": {"has_children": False},
        "examples": [
            "Traveling with my son — he's 25 and I'm 55",
            "Flight to Denver for me and my adult daughter",
            "Family reunion — all of us are over 30",
        ],
    },
    {
        "name": "edge_negative",
        "expected_behavior": "should_not_ask_ages",
        "count": 5,
        "description": (
            "Edge cases that should NOT trigger age questions: "
            "traveling with pets (not children), pregnant traveler (no child yet), "
            "'young adults' who are 18+, explicit 'two adults' phrasing, "
            "mentioning 'kids' in a non-literal way ('we're just big kids at heart')."
        ),
        "context_template": {"has_children": False},
        "examples": [
            "Flying to Portland with my dog",
            "I'm pregnant — can I fly to Hawaii at 7 months?",
            "Two adults to Rome, no kids this time",
        ],
    },
]

SYSTEM_PROMPT = """\
You are a dataset generator for evaluating travel booking AI agents.

Generate realistic, diverse travel booking queries that a customer might type
or say to a travel booking agent. Each query should sound natural and varied —
avoid templated or repetitive phrasing.

Rules:
- Each query should be 1-2 sentences, written as a customer would naturally ask
- Vary destinations, travel types, group sizes, and phrasing
- Do not repeat the example queries given to you
- Include realistic details (dates, destinations, purposes) but keep queries concise
- Return ONLY a JSON array of objects, no other text
"""


def generate_category(client: OpenAI, category: dict) -> list[dict]:
    """Generate test cases for a single category."""
    user_prompt = f"""\
Generate exactly {category["count"]} travel booking queries for category: {category["name"]}

Description: {category["description"]}

Expected behavior: {category["expected_behavior"]}

Example queries (do NOT reuse these, generate new ones):
{json.dumps(category["examples"], indent=2)}

Return a JSON array of objects with this exact structure:
[
  {{
    "query": "the customer's travel booking query",
    "child_count": <number or null if unknown/not applicable>
  }}
]

Return ONLY the JSON array, no markdown, no explanation."""

    response = client.chat.completions.create(
        model="gpt-4o",
        temperature=0.7,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )

    content = response.choices[0].message.content.strip()
    # Strip markdown code fences if present
    content = re.sub(r"^```(?:json)?\s*", "", content)
    content = re.sub(r"\s*```$", "", content)

    items = json.loads(content)
    if not isinstance(items, list):
        raise ValueError(f"Expected list, got {type(items)}")

    results = []
    for item in items:
        context = dict(category["context_template"])
        if "child_count" in item and item["child_count"] is not None:
            context["child_count"] = item["child_count"]
        context["category"] = category["name"]

        results.append(
            {
                "query": item["query"],
                "expected_behavior": category["expected_behavior"],
                "context": context,
            }
        )

    return results


def deduplicate(cases: list[dict]) -> list[dict]:
    """Remove near-duplicate queries (exact match after lowering)."""
    seen = set()
    unique = []
    for case in cases:
        key = case["query"].strip().lower()
        if key not in seen:
            seen.add(key)
            unique.append(case)
    return unique


def main():
    client = OpenAI()
    all_cases = []

    print(f"Generating dataset across {len(CATEGORIES)} categories...")
    for cat in CATEGORIES:
        print(f"  [{cat['expected_behavior']}] {cat['name']} ({cat['count']} cases)...", end=" ")
        try:
            cases = generate_category(client, cat)
            all_cases.extend(cases)
            print(f"got {len(cases)}")
        except Exception as e:
            print(f"ERROR: {e}")

    # Deduplicate
    before = len(all_cases)
    all_cases = deduplicate(all_cases)
    if before != len(all_cases):
        print(f"  Removed {before - len(all_cases)} duplicates")

    # Assign sequential IDs
    for i, case in enumerate(all_cases, 1):
        case["input_id"] = f"case_{i:03d}"

    # Reorder keys for readability
    ordered = []
    for case in all_cases:
        ordered.append(
            {
                "input_id": case["input_id"],
                "query": case["query"],
                "expected_behavior": case["expected_behavior"],
                "context": case["context"],
            }
        )

    # Write JSONL
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        for case in ordered:
            f.write(json.dumps(case) + "\n")

    # Summary
    positive = sum(1 for c in ordered if c["expected_behavior"] == "should_ask_ages")
    negative = len(ordered) - positive
    print(f"\nDataset written to {OUTPUT_PATH}")
    print(f"Total: {len(ordered)} cases")
    print(f"  Positive (should_ask_ages): {positive} ({100*positive/len(ordered):.0f}%)")
    print(f"  Negative (should_not_ask_ages): {negative} ({100*negative/len(ordered):.0f}%)")

    # Category breakdown
    print("\nCategory breakdown:")
    from collections import Counter
    cats = Counter(c["context"]["category"] for c in ordered)
    for cat, count in cats.most_common():
        behavior = next(c["expected_behavior"] for c in ordered if c["context"]["category"] == cat)
        marker = "+" if behavior == "should_ask_ages" else "-"
        print(f"  [{marker}] {cat}: {count}")


if __name__ == "__main__":
    main()
