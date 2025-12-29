#!/usr/bin/env python
"""
TraiGent Quickstart: Customer Support RAG with FAISS and Real LLM Calls

This example demonstrates production-ready RAG (Retrieval Augmented Generation)
optimization with FAISS vector store, OpenAI embeddings, and document chunking:

- Document chunking for large texts (RecursiveCharacterTextSplitter)
- Semantic similarity search with FAISS
- OpenAI embeddings (text-embedding-3-small)
- Chunk size and overlap optimization (chunk_size, chunk_overlap parameters)
- RAG retrieval depth optimization (k parameter)
- Model and temperature tuning
- Multi-objective optimization (accuracy, cost, latency)

Run with:
    export OPENAI_API_KEY="your-key-here"
    pip install faiss-cpu  # if not already installed
    python examples/quickstart/02_customer_support_rag_real_llm.py
"""

import asyncio
import csv
import json
import os
import sys
from pathlib import Path

# Check for API key first
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    print("ERROR: OPENAI_API_KEY environment variable not set.")
    print("Please run: export OPENAI_API_KEY='your-key-here'")
    sys.exit(1)

# Check for FAISS and text splitter
try:
    from langchain_community.vectorstores import FAISS
    from langchain_core.documents import Document
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError as e:
    print(f"ERROR: Missing dependency: {e}")
    print("Please install: pip install faiss-cpu langchain-community langchain-openai langchain-text-splitters")
    sys.exit(1)

# Set results folder to local directory
os.environ.setdefault(
    "TRAIGENT_RESULTS_FOLDER", str(Path(__file__).parent / ".traigent_results")
)

import traigent
from traigent.api.decorators import (
    EvaluationOptions,
    ExecutionOptions,
    InjectionOptions,
)

# Create a simple RAG dataset for customer support
RAG_DATASET_PATH = Path(__file__).parent / "rag_feedback.jsonl"

# Create the dataset if it doesn't exist
if not RAG_DATASET_PATH.exists():
    rag_data = [
        '{"input": {"query": "What is your return policy?"}, "output": "Returns accepted within 30 days"}',
        '{"input": {"query": "Do you offer free shipping?"}, "output": "Free shipping on orders over $50"}',
        '{"input": {"query": "How can I track my order?"}, "output": "Use the tracking link in your confirmation email"}',
        '{"input": {"query": "What payment methods do you accept?"}, "output": "We accept credit cards, PayPal, and Apple Pay"}',
        '{"input": {"query": "How do I contact support?"}, "output": "Email support@example.com or call 1-800-SUPPORT"}',
        '{"input": {"query": "What are your business hours?"}, "output": "Monday-Friday 9am-5pm EST"}',
        '{"input": {"query": "Do gift cards expire?"}, "output": "Gift cards never expire"}',
        '{"input": {"query": "Do you price match?"}, "output": "Price match guarantee within 14 days of purchase"}',
    ]
    RAG_DATASET_PATH.write_text("\n".join(rag_data) + "\n")


# =============================================================================
# KNOWLEDGE BASE - Long-form documents that will be chunked
# =============================================================================
# In production, you would load these from files, databases, or web scraping.
# Each document represents a full page/article that needs to be chunked.

KNOWLEDGE_BASE_DOCUMENTS = [
    {
        "source": "returns_policy.md",
        "category": "returns",
        "content": """# Return Policy

## Overview
We want you to be completely satisfied with your purchase. If you're not happy with your order, we offer a comprehensive return policy to make things right.

## Return Window
Returns are accepted within 30 days of the original purchase date. Items must be unused and in their original packaging with all tags attached. After 30 days, we cannot accept returns but may offer store credit on a case-by-case basis.

## How to Initiate a Return
1. Log into your account and go to Order History
2. Select the order containing the item you wish to return
3. Click "Request Return" and select a reason
4. Print the prepaid shipping label (free for defective items)
5. Pack the item securely and drop it off at any carrier location

## Refund Processing
Once we receive your return, please allow 5-7 business days for inspection and processing. Refunds are issued to the original payment method. Credit card refunds may take an additional 3-5 business days to appear on your statement.

## Exceptions
The following items cannot be returned:
- Personalized or custom items
- Perishable goods
- Intimate apparel and swimwear (for hygiene reasons)
- Items marked as "Final Sale"

## Defective Items
For defective or damaged items, we offer free return shipping. Contact support within 48 hours of delivery with photos of the damage. We'll send a replacement immediately or issue a full refund including original shipping costs.
"""
    },
    {
        "source": "shipping_info.md",
        "category": "shipping",
        "content": """# Shipping Information

## Free Shipping
We offer FREE standard shipping on all orders over $50 within the continental United States. Orders under $50 have a flat rate shipping fee of $5.99.

## Shipping Methods and Timeframes

### Standard Shipping (5-7 business days)
- Free on orders over $50
- $5.99 for orders under $50
- Delivered by USPS or FedEx Ground

### Express Shipping (2-3 business days)
- $12.99 flat rate
- Delivered by FedEx Express or UPS 2nd Day Air
- Order by 2pm EST for same-day processing

### Overnight Shipping (1 business day)
- $24.99 flat rate
- Delivered by FedEx Priority Overnight
- Order by 12pm EST for next-day delivery
- Not available for PO Boxes

## Order Tracking
Track your order using the tracking link sent in your confirmation email. You can also check your order status on our website by logging into your account and viewing Order History. Tracking information is typically available within 24 hours of shipment.

## International Shipping
We currently ship to Canada, UK, and EU countries. International shipping rates are calculated at checkout based on destination and package weight. Please note that customers are responsible for any customs duties or import taxes.

## Shipping Restrictions
Some items cannot be shipped to certain locations due to carrier restrictions or local regulations. Hazardous materials, lithium batteries, and oversized items may have additional shipping requirements.
"""
    },
    {
        "source": "payment_methods.md",
        "category": "payments",
        "content": """# Payment Methods

## Accepted Payment Types
We accept a wide variety of payment methods to make your shopping experience convenient and secure.

### Credit and Debit Cards
- Visa
- Mastercard
- American Express
- Discover

All card transactions are processed through our secure payment gateway with 256-bit SSL encryption. We never store your full card number on our servers.

### Digital Wallets
- PayPal
- Apple Pay
- Google Pay
- Shop Pay

Digital wallet payments offer an extra layer of security since your card details are never shared with us directly.

### Buy Now, Pay Later
- Affirm (split into 4 payments)
- Klarna (pay in 30 days or 4 installments)
- Afterpay (4 interest-free payments)

Buy now, pay later options are available at checkout for orders over $35. Subject to approval. See provider terms for details.

### Gift Cards
Gift cards can be used as full or partial payment. Enter your gift card code at checkout. Any remaining balance stays on your card for future purchases. Gift cards never expire and can be used online or in-store.

## Payment Security
All transactions are secure and encrypted. We are PCI DSS compliant and regularly undergo security audits. We use fraud detection systems to protect both you and our business. If you notice any suspicious activity, contact us immediately.

## Billing Issues
If you have questions about a charge or need to dispute a transaction, please contact our support team with your order number. We'll investigate and respond within 24 hours.
"""
    },
    {
        "source": "contact_support.md",
        "category": "support",
        "content": """# Contact Support

## How to Reach Us
Our customer support team is here to help! We offer multiple ways to get in touch.

### Email Support
- Email: support@example.com
- Response time: Within 24 hours (usually faster)
- Available: 24/7

For fastest resolution, please include your order number and a detailed description of your issue.

### Phone Support
- Phone: 1-800-SUPPORT (1-800-787-7678)
- Hours: Monday-Friday, 9am-5pm EST
- Average wait time: Under 5 minutes

### Live Chat
- Available on our website
- Hours: Monday-Friday, 9am-8pm EST
- Saturday: 10am-4pm EST
- Instant connection with a support agent

### Social Media
- Twitter: @examplestore (DMs open)
- Facebook: facebook.com/examplestore
- Instagram: @examplestore

Social media messages are typically answered within 2-4 hours during business hours.

## Before Contacting Support
To help us resolve your issue quickly, please have the following ready:
1. Your order number
2. Email address used for the order
3. Description of the issue
4. Photos (if applicable, for damaged items)

## FAQ
Many common questions are answered in our FAQ section. Check there first - you might find an instant answer!

## Escalation
If you're not satisfied with the initial response, ask to speak with a supervisor. We take customer satisfaction seriously and will work to find a resolution.
"""
    },
    {
        "source": "store_hours.md",
        "category": "hours",
        "content": """# Business Hours

## Customer Support Hours

### Phone and Live Chat Support
- Monday through Friday: 9:00 AM - 5:00 PM EST
- Saturday: 10:00 AM - 4:00 PM EST (Live Chat only)
- Sunday: Closed

### Email Support
- Available 24/7
- Responses within 24 hours
- Weekend emails answered Monday morning

## Holiday Schedule
We observe the following holidays when phone support is closed:
- New Year's Day
- Memorial Day
- Independence Day (July 4th)
- Labor Day
- Thanksgiving Day
- Christmas Day

Email support remains available during holidays with delayed response times.

## Order Processing Hours
- Orders placed before 2:00 PM EST ship same day
- Orders after 2:00 PM EST ship next business day
- Weekend orders ship Monday

## Warehouse Hours
Our fulfillment center operates Monday-Saturday to ensure quick shipping. During peak seasons (Black Friday, holiday shopping), we extend hours to handle increased volume.

## Time Zone
All times listed are in Eastern Standard Time (EST) / Eastern Daylight Time (EDT). Please adjust for your local time zone.
"""
    },
    {
        "source": "gift_cards.md",
        "category": "gift_cards",
        "content": """# Gift Cards

## About Our Gift Cards
Give the gift of choice! Our gift cards are the perfect present for any occasion.

## Gift Card Options
- $25, $50, $75, $100, $150, $200, or custom amount
- Physical cards ship free with beautiful packaging
- E-gift cards delivered instantly via email
- Corporate bulk orders available (10+ cards)

## Key Features
- **Never Expire**: Our gift cards have no expiration date
- **No Fees**: No activation fees, maintenance fees, or dormancy fees
- **Flexible Use**: Can be used online or in-store
- **Stackable**: Use multiple gift cards on a single order
- **Partial Use**: Remaining balance stays on your card

## How to Use
1. Add items to your cart
2. At checkout, enter your gift card code
3. Click "Apply"
4. The balance will be deducted from your order total

## Check Your Balance
Visit giftcards.example.com or call 1-800-SUPPORT to check your gift card balance. You'll need the 16-digit card number and 4-digit PIN (located on the back of physical cards or in your email for e-gift cards).

## Lost or Stolen Cards
Report lost or stolen cards immediately to customer support. We can freeze the card and transfer the balance to a new card with proof of purchase. Physical cards cannot be replaced without the original receipt.

## Refunds to Gift Cards
Returns from gift card purchases are refunded back to a gift card. We cannot convert gift card balances to cash except where required by law.
"""
    },
    {
        "source": "price_matching.md",
        "category": "pricing",
        "content": """# Price Match Guarantee

## Our Promise
We're committed to offering competitive prices. If you find a lower price on an identical item from an authorized retailer, we'll match it!

## Price Match Policy

### Eligibility Requirements
- Request must be made within 14 days of purchase
- Item must be identical (same brand, model, size, color)
- Competitor must be an authorized retailer
- Item must be in stock at the competitor
- Price must be publicly advertised (not members-only)

### What We Match
- Regular prices
- Sale prices
- Promotional prices (with valid promo code)

### What We Don't Match
- Marketplace sellers (Amazon third-party, eBay, etc.)
- Auction sites
- Outlet or clearance stores
- Wholesale clubs (Costco, Sam's Club)
- Pricing errors or misprints
- Bundle deals or BOGO offers
- Coupons or rebates

## How to Request a Price Match

### Before Purchase
1. Contact us with a link to the competitor's listing
2. We'll verify the price and eligibility
3. If approved, we'll adjust the price before you check out

### After Purchase (within 14 days)
1. Email support@example.com with:
   - Your order number
   - Link to competitor's lower price
   - Screenshot showing the price and date
2. We'll verify and issue a refund for the difference

## Post-Purchase Price Protection
If the price drops on our own website within 14 days of your purchase, we'll refund the difference. Just contact support with your order number.

## Limitations
Price matching is limited to one claim per item per customer. We reserve the right to refuse price matching in cases of fraud or abuse.
"""
    },
]


def chunk_documents(
    documents: list[dict],
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list[Document]:
    """Split long documents into smaller chunks for better retrieval.

    Args:
        documents: List of document dicts with 'content', 'source', 'category'
        chunk_size: Maximum characters per chunk (default 500)
        chunk_overlap: Overlap between chunks to preserve context (default 50)

    Returns:
        List of LangChain Document objects with metadata
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", ", ", " ", ""],
    )

    all_chunks = []
    for doc in documents:
        chunks = text_splitter.split_text(doc["content"])
        for i, chunk in enumerate(chunks):
            all_chunks.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "source": doc["source"],
                        "category": doc["category"],
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                    }
                )
            )

    return all_chunks


# Cache for vector stores by (chunk_size, chunk_overlap) tuple
_vector_store_cache: dict[tuple[int, int], FAISS] = {}


def create_vector_store(chunk_size: int = 500, chunk_overlap: int = 50) -> FAISS:
    """Create FAISS vector store with OpenAI embeddings.

    Args:
        chunk_size: Size of document chunks (affects retrieval quality)
        chunk_overlap: Overlap between chunks (sliding window for context preservation)

    Uses text-embedding-3-small for cost-effective semantic embeddings.
    Cost: ~$0.00002 per 1K tokens (very cheap for small knowledge bases).
    """
    print(f"Creating FAISS vector store (chunk_size={chunk_size}, overlap={chunk_overlap})...")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # Chunk documents
    documents = chunk_documents(
        KNOWLEDGE_BASE_DOCUMENTS,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    vector_store = FAISS.from_documents(documents, embeddings)
    print(f"  Created vector store with {len(documents)} chunks from {len(KNOWLEDGE_BASE_DOCUMENTS)} documents")
    return vector_store


def get_vector_store(chunk_size: int = 500, chunk_overlap: int = 50) -> FAISS:
    """Get or create vector store for given chunk_size and chunk_overlap."""
    cache_key = (chunk_size, chunk_overlap)
    if cache_key not in _vector_store_cache:
        _vector_store_cache[cache_key] = create_vector_store(chunk_size, chunk_overlap)
    return _vector_store_cache[cache_key]


def semantic_retriever(vector_store: FAISS, query: str, k: int = 3) -> list[str]:
    """Retrieve documents using semantic similarity search.

    Unlike keyword matching, this finds semantically similar content
    even if the exact words don't match.
    """
    docs = vector_store.similarity_search(query, k=k)
    return [doc.page_content for doc in docs]


def rag_accuracy_scorer(output: str, expected: str) -> float:
    """Score RAG response quality.

    Checks if key information from expected output appears in actual output.

    Args:
        output: The actual LLM output (TraiGent passes this automatically)
        expected: The expected output from dataset (TraiGent passes this automatically)
    """
    if not output or not expected:
        return 0.0

    actual_lower = output.lower()
    expected_lower = expected.lower()

    # Extract key terms from expected output
    key_terms = [word for word in expected_lower.split() if len(word) > 3]
    if not key_terms:
        return 1.0 if expected_lower in actual_lower else 0.0

    # Score based on key term matches
    matches = sum(1 for term in key_terms if term in actual_lower)
    return matches / len(key_terms)


# =============================================================================
# Configuration Space - including chunk_size and chunk_overlap for RAG optimization
# =============================================================================
CONFIGURATION_SPACE = {
    "model": ["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o"],
    "temperature": [0.1, 0.3, 0.5, 0.7],
    "k": [2, 3, 5],  # Number of chunks to retrieve
    "chunk_size": [300, 500, 800],  # Characters per chunk
    "chunk_overlap": [25, 50, 100],  # Sliding window overlap (context preservation)
}

DEFAULT_CONFIG = {
    "model": "gpt-3.5-turbo",
    "temperature": 0.3,
    "k": 3,
    "chunk_size": 500,
    "chunk_overlap": 50,
}

# Constraints as descriptive strings for printing
CONSTRAINTS_DESCRIPTIONS = [
    "GPT-4o: temperature <= 0.3 (expensive model, keep focused)",
    "GPT-3.5-turbo: k >= 3 (needs more context)",
    "chunk_overlap < chunk_size (overlap must be smaller than chunk)",
]


@traigent.optimize(
    configuration_space=CONFIGURATION_SPACE,
    default_config=DEFAULT_CONFIG,
    objectives=["accuracy", "cost", "latency"],
    constraints=[
        # Don't use high temperature with GPT-4o (expensive + unpredictable)
        lambda cfg: cfg["temperature"] <= 0.3 if cfg["model"] == "gpt-4o" else True,
        # GPT-3.5-turbo needs more context documents
        lambda cfg: cfg["k"] >= 3 if cfg["model"] == "gpt-3.5-turbo" else True,
        # Overlap must be smaller than chunk size
        lambda cfg: cfg["chunk_overlap"] < cfg["chunk_size"],
    ],
    metric_functions={"accuracy": rag_accuracy_scorer},
    evaluation=EvaluationOptions(eval_dataset=str(RAG_DATASET_PATH)),
    execution=ExecutionOptions(
        execution_mode="edge_analytics",
        minimal_logging=False,
        reps_per_trial=2,
        reps_aggregation="mean",
    ),
    injection=InjectionOptions(
        auto_override_frameworks=True,
    ),
    max_trials=6,
    timeout=600,
    cost_limit=5.00,
    cost_approved=True,
)
def customer_support_agent(
    query: str,
    model: str = "gpt-3.5-turbo",
    temperature: float = 0.3,
    k: int = 3,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> str:
    """Answer customer questions using RAG with real LLM calls.

    This function uses:
    - Document chunking with configurable chunk_size and chunk_overlap
    - FAISS for semantic similarity search
    - OpenAI embeddings for document vectors
    - LangChain ChatOpenAI for LLM responses

    TraiGent automatically injects optimized parameters:
    - model: Which LLM to use
    - temperature: Creativity vs consistency
    - k: How many chunks to retrieve
    - chunk_size: Size of document chunks (affects retrieval precision)
    - chunk_overlap: Sliding window overlap (preserves context at boundaries)
    """
    # Get vector store for this chunk_size/overlap and retrieve relevant chunks
    vector_store = get_vector_store(chunk_size, chunk_overlap)
    docs = semantic_retriever(vector_store, query, k=k)
    context = "\n".join(f"- {doc}" for doc in docs)

    # Build prompt with retrieved context
    prompt = f"""You are a helpful customer support agent. Answer the customer's question
based ONLY on the following knowledge base information.

Knowledge Base:
{context}

Customer Question: {query}

Provide a concise, helpful answer based on the information above. If the information
doesn't fully answer the question, say what you can based on what's available."""

    # Make real LLM call
    llm = ChatOpenAI(
        model=model,
        temperature=temperature,
        max_tokens=150,
    )
    response = llm.invoke(prompt)
    return str(response.content)


def save_results_to_csv(results, dataset_path: Path, output_path: Path) -> None:
    """Save optimization results to a CSV file."""
    questions = []
    expected_answers = []
    with open(dataset_path) as f:
        for line in f:
            data = json.loads(line)
            questions.append(data["input"]["query"])
            expected_answers.append(data["output"])

    headers = ["Query", "Expected"]
    trial_configs = []

    for i, trial in enumerate(results.trials, 1):
        config = getattr(trial, "config", getattr(trial, "configuration", {}))
        config_str = f"T{i}: {config.get('model', 'N/A')}, k={config.get('k', 'N/A')}, chunk={config.get('chunk_size', 'N/A')}"
        headers.append(f"{config_str} Answer")
        headers.append(f"{config_str} Pass")
        trial_configs.append(config)

    rows = []
    for q_idx, (question, expected) in enumerate(zip(questions, expected_answers)):
        question_clean = question.replace("\n", " | ").replace("\r", "")
        expected_clean = expected.replace("\n", " | ").replace("\r", "")
        rows.append([question_clean, expected_clean])

    trial_pass_counts = []

    for trial in results.trials:
        example_results = trial.metadata.get("example_results", [])

        results_by_id = {}
        for ex_result in example_results:
            ex_idx = int(ex_result.example_id.split("_")[1])
            results_by_id[ex_idx] = ex_result

        pass_count = 0
        total_count = 0
        for q_idx in range(len(questions)):
            ex_result = results_by_id.get(q_idx)
            if ex_result:
                answer = str(ex_result.actual_output) if ex_result.actual_output else ""
                answer = answer.replace("\n", " | ").replace("\r", "")
                metrics = getattr(ex_result, "metrics", {}) or {}
                score = metrics.get("accuracy", metrics.get("score", 0))
                passed = score >= 0.5 if isinstance(score, (int, float)) else False
                rows[q_idx].append(answer)
                rows[q_idx].append("PASS" if passed else "FAIL")
                total_count += 1
                if passed:
                    pass_count += 1
            else:
                rows[q_idx].append("N/A")
                rows[q_idx].append("")

        trial_pass_counts.append((pass_count, total_count))

    summary_row = ["SUMMARY", "Pass Rate"]
    for pass_count, total_count in trial_pass_counts:
        if total_count > 0:
            ratio = pass_count / total_count
            summary_row.append(f"{pass_count}/{total_count}")
            summary_row.append(f"{ratio:.1%}")
        else:
            summary_row.append("N/A")
            summary_row.append("N/A")

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerow(headers)
        writer.writerows(rows)
        writer.writerow(summary_row)

    print(f"Results saved to: {output_path}")


def print_model_summary(results) -> None:
    """Print accuracy breakdown by model, k value, chunk_size, and chunk_overlap."""
    model_results = {}
    k_results = {}
    chunk_size_results = {}
    chunk_overlap_results = {}
    total_cost = 0.0

    for trial in results.trials:
        config = getattr(trial, "config", getattr(trial, "configuration", {}))
        model = config.get("model", "unknown")
        k_val = config.get("k", "unknown")
        chunk_size_val = config.get("chunk_size", "unknown")
        chunk_overlap_val = config.get("chunk_overlap", "unknown")
        example_results = trial.metadata.get("example_results", [])

        trial_cost = trial.metadata.get("total_example_cost", 0)
        if trial_cost:
            total_cost += float(trial_cost)

        for ex_result in example_results:
            metrics = getattr(ex_result, "metrics", {}) or {}
            score = metrics.get("accuracy", metrics.get("score", 0))
            passed = score >= 0.5 if isinstance(score, (int, float)) else False

            if model not in model_results:
                model_results[model] = []
            model_results[model].append(passed)

            if k_val not in k_results:
                k_results[k_val] = []
            k_results[k_val].append(passed)

            if chunk_size_val not in chunk_size_results:
                chunk_size_results[chunk_size_val] = []
            chunk_size_results[chunk_size_val].append(passed)

            if chunk_overlap_val not in chunk_overlap_results:
                chunk_overlap_results[chunk_overlap_val] = []
            chunk_overlap_results[chunk_overlap_val].append(passed)

    print()
    print("=" * 60)
    print("Results Summary")
    print("=" * 60)
    print()

    print("Accuracy by Model:")
    for model in sorted(model_results.keys()):
        passes = model_results[model]
        accuracy = sum(passes) / len(passes) * 100 if passes else 0
        print(f"  {model}: {accuracy:.1f}% ({sum(passes)}/{len(passes)})")

    print()
    print("Accuracy by Retrieval Depth (k):")
    for k_val in sorted(k_results.keys()):
        passes = k_results[k_val]
        accuracy = sum(passes) / len(passes) * 100 if passes else 0
        print(f"  k={k_val}: {accuracy:.1f}% ({sum(passes)}/{len(passes)})")

    print()
    print("Accuracy by Chunk Size:")
    for chunk_val in sorted(chunk_size_results.keys()):
        passes = chunk_size_results[chunk_val]
        accuracy = sum(passes) / len(passes) * 100 if passes else 0
        print(f"  chunk_size={chunk_val}: {accuracy:.1f}% ({sum(passes)}/{len(passes)})")

    print()
    print("Accuracy by Chunk Overlap:")
    for overlap_val in sorted(chunk_overlap_results.keys()):
        passes = chunk_overlap_results[overlap_val]
        accuracy = sum(passes) / len(passes) * 100 if passes else 0
        print(f"  chunk_overlap={overlap_val}: {accuracy:.1f}% ({sum(passes)}/{len(passes)})")

    print()
    print(f"Total Cost: ${total_cost:.4f}")
    print()


async def main():
    print("=" * 60)
    print("TraiGent Quickstart: Customer Support RAG (FAISS + Chunking)")
    print("=" * 60)
    print()

    # Show knowledge base info
    total_chars = sum(len(doc["content"]) for doc in KNOWLEDGE_BASE_DOCUMENTS)
    print(f"Knowledge Base: {len(KNOWLEDGE_BASE_DOCUMENTS)} documents, {total_chars:,} total characters")
    for i, doc in enumerate(KNOWLEDGE_BASE_DOCUMENTS[:3], 1):
        print(f"  {i}. [{doc['category']}] {doc['source']} ({len(doc['content']):,} chars)")
    print(f"  ... and {len(KNOWLEDGE_BASE_DOCUMENTS) - 3} more documents")
    print()

    print(f"Dataset: {RAG_DATASET_PATH}")
    print("Using FAISS vector store with OpenAI embeddings + document chunking")
    print()

    print("Configuration Space:")
    print(f"  - Models: {', '.join(CONFIGURATION_SPACE['model'])}")
    print(f"  - Temperature: {', '.join(str(t) for t in CONFIGURATION_SPACE['temperature'])}")
    print(f"  - Retrieval Depth (k): {', '.join(str(k) for k in CONFIGURATION_SPACE['k'])}")
    print(f"  - Chunk Size: {', '.join(str(c) for c in CONFIGURATION_SPACE['chunk_size'])}")
    print(f"  - Chunk Overlap: {', '.join(str(o) for o in CONFIGURATION_SPACE['chunk_overlap'])}")
    print()

    print("Constraints Applied:")
    for constraint in CONSTRAINTS_DESCRIPTIONS:
        print(f"  - {constraint}")
    print()

    # Show chunking example
    print("Document Chunking Example (chunk_size=500):")
    example_chunks = chunk_documents(KNOWLEDGE_BASE_DOCUMENTS[:1], chunk_size=500)
    print(f"  '{KNOWLEDGE_BASE_DOCUMENTS[0]['source']}' ({len(KNOWLEDGE_BASE_DOCUMENTS[0]['content']):,} chars) -> {len(example_chunks)} chunks")
    for i, chunk in enumerate(example_chunks[:2]):
        preview = chunk.page_content[:80].replace("\n", " ")
        print(f"    Chunk {i+1}: \"{preview}...\"")
    if len(example_chunks) > 2:
        print(f"    ... and {len(example_chunks) - 2} more chunks")
    print()

    # Initialize default vector store
    vector_store = get_vector_store(chunk_size=500, chunk_overlap=50)
    print()

    # Test semantic retriever
    print("Testing semantic retriever:")
    test_query = "How do I get my money back?"
    retrieved = semantic_retriever(vector_store, test_query, k=2)
    print(f"  Query: '{test_query}'")
    print(f"  Retrieved (semantic match):")
    for doc in retrieved:
        preview = doc[:80].replace("\n", " ")
        print(f"    - \"{preview}...\"")
    print()

    # Run optimization
    print("Starting RAG optimization...")
    print("-" * 40)
    results = await customer_support_agent.optimize()

    print()
    print("=" * 60)
    print("Optimization Complete!")
    print("=" * 60)
    print()
    print(f"Best Score: {results.best_score}")
    print(f"Best Configuration: {results.best_config}")
    print()

    if hasattr(results, "trials") and results.trials:
        print("All Trials:")
        print("-" * 40)
        for i, trial in enumerate(results.trials, 1):
            score = getattr(trial, "score", None) or getattr(trial, "metrics", {}).get(
                "score", "N/A"
            )
            config = getattr(trial, "config", getattr(trial, "configuration", {}))
            print(f"  Trial {i}: {config} -> score={score}")

    # Save results
    csv_output_path = Path(__file__).parent / "results" / "rag_optimization_results.csv"
    csv_output_path.parent.mkdir(parents=True, exist_ok=True)
    save_results_to_csv(results, RAG_DATASET_PATH, csv_output_path)

    best_config_path = Path(__file__).parent / "results" / "rag_best_config.json"
    with open(best_config_path, "w") as f:
        json.dump({
            "best_score": results.best_score,
            "best_config": results.best_config,
        }, f, indent=2)
    print(f"Best config saved to: {best_config_path}")

    print_model_summary(results)

    # Demo with optimized config
    print("=" * 60)
    print("Using Optimized Configuration")
    print("=" * 60)
    print()

    best_config = customer_support_agent.get_best_config()
    if best_config:
        print(f"Applying best config: {best_config}")
        customer_support_agent.set_config(best_config)

        test_questions = [
            "What is your return policy?",
            "Do you offer free shipping?",
            "How do I contact support?",
        ]
        for question in test_questions:
            print(f"\nQ: {question}")
            answer = customer_support_agent(question)
            print(f"A: {answer}")

    print()
    print("RAG optimization complete with FAISS, chunking, and real LLM calls!")


if __name__ == "__main__":
    asyncio.run(main())
