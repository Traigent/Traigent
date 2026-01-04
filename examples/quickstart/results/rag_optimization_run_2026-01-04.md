# RAG Optimization Run - January 4, 2026

## Overview

Ran TraiGent optimization on the Customer Support RAG example with FAISS vector store, OpenAI embeddings, and document chunking. This run includes a newly expanded dataset with **multi-source questions** designed to test RAG retrieval across multiple documents.

---

## Dataset Enhancement: Multi-Source Questions

### Motivation

Standard RAG datasets typically test single-document retrieval. We added 9 new questions (questions 20-28) that require synthesizing information from 2+ knowledge base documents to test whether:

1. The RAG system can retrieve relevant chunks from multiple documents
2. The LLM can synthesize information across different sources
3. Higher `k` values (more retrieved chunks) improve performance

### New Multi-Source Questions Added

| # | Question | Required Documents |
|---|----------|-------------------|
| 20 | If I return an item I paid for with a gift card, how do I get my money back? | returns_policy.md + gift_cards.md |
| 21 | I ordered at 3pm on a weekday - when will it ship and how long until it arrives? | store_hours.md + shipping_info.md |
| 22 | If I find a lower price after buying, how do I request a refund and how long until I get it? | price_matching.md + returns_policy.md |
| 23 | Can I use Klarna to buy a $25 gift card? | payment_methods.md + gift_cards.md |
| 24 | If my item arrives damaged, who do I contact and within what timeframe? | returns_policy.md + contact_support.md |
| 25 | I want to buy a physical gift card for someone in Canada - can I ship it there? | gift_cards.md + shipping_info.md |
| 26 | If I order at 1pm on Saturday and choose overnight shipping, when will it arrive? | store_hours.md + shipping_info.md |
| 27 | I paid with PayPal and returned an item - when will I see the refund? | payment_methods.md + returns_policy.md |
| 28 | What's the total cost to ship a $40 order with express shipping? | shipping_info.md (multiple facts) |

### Dataset Composition
- **Single-source questions**: 19 (questions 1-19)
- **Multi-source questions**: 9 (questions 20-28)
- **Total questions**: 28

---

## Setup

### Knowledge Base
- **Documents**: 7 files, 9,830 total characters
- **Source**: `examples/quickstart/knowledge_base/`
- **Topics**: Gift cards, store hours, payment methods, returns, shipping, support, pricing

### Configuration Space
| Parameter | Values |
|-----------|--------|
| Model | gpt-3.5-turbo, gpt-4o-mini, gpt-4o |
| Temperature | 0.1, 0.3, 0.5, 0.7 |
| Retrieval Depth (k) | 2, 3, 5 |
| Chunk Size | 300, 500, 800 |
| Chunk Overlap | 25, 50, 100 |

### Constraints
1. GPT-4o: temperature <= 0.3 (expensive model, keep focused)
2. GPT-3.5-turbo: k >= 3 (needs more context)
3. chunk_overlap < chunk_size (must be smaller)

---

## Results Summary

### Best Configuration
```json
{
  "model": "gpt-4o-mini",
  "temperature": 0.7,
  "k": 3,
  "chunk_size": 500,
  "chunk_overlap": 25
}
```

**Best Score**: 92.86% accuracy (26/28 questions passed)

### All Trials

| Trial | Model | Temp | k | Chunk Size | Overlap | Pass Rate | Score |
|-------|-------|------|---|------------|---------|-----------|-------|
| 1 | gpt-3.5-turbo | 0.3 | 3 | 500 | 50 | 25/28 | 89.3% |
| 2 | gpt-4o-mini | 0.7 | 3 | 500 | 25 | 26/28 | **92.9%** |
| 3 | gpt-4o-mini | 0.5 | 3 | 300 | 50 | 26/28 | 92.9% |
| 4 | gpt-3.5-turbo | 0.3 | 5 | 500 | 25 | 26/28 | 92.9% |
| 5 | gpt-4o-mini | 0.3 | 5 | 800 | 50 | 26/28 | 92.9% |
| 6 | gpt-4o | 0.3 | 2 | 500 | 25 | 25/28 | 89.3% |
| 7 | gpt-4o-mini | 0.5 | 5 | 300 | 50 | 26/28 | 92.9% |
| 8 | gpt-4o-mini | 0.3 | 3 | 300 | 50 | 26/28 | 92.9% |
| 9 | gpt-4o | 0.7 | 3 | 800 | 50 | N/A | N/A (constraint violation) |
| 10 | gpt-4o-mini | 0.1 | 3 | 800 | 25 | 26/28 | 92.9% |

---

## Per-Question Analysis

### Questions That FAILED Across All Trials

| # | Question | Expected Answer | Issue |
|---|----------|-----------------|-------|
| 22 | If I find a lower price after buying, how do I request a refund and how long until I get it? | "Email support within 14 days, refund processed in 5-7 business days" | **All models failed** - They confused price match refund with return refund. Said "email support" but not specifically about price matching process. |
| 23 | Can I use Klarna to buy a $25 gift card? | "No, buy now pay later requires orders over $35" | **All models failed** - Models incorrectly said "Yes" because they found Klarna info and $35 minimum, but didn't synthesize that $25 < $35 means NO. |

### Questions With Mixed Results

| # | Question | Expected | Pass Rate | Notes |
|---|----------|----------|-----------|-------|
| 25 | Physical gift card to Canada - shipping cost? | "Free domestic, international calculated at checkout" | 7/9 | Trials 1 & 6 (with k=2,3) failed - didn't retrieve international shipping info |

### All Single-Source Questions (1-19): 100% Pass Rate
Every trial passed all 19 single-source questions.

### Multi-Source Questions Performance (20-28)

| Question | Type | T1 | T2 | T3 | T4 | T5 | T6 | T7 | T8 | T10 |
|----------|------|----|----|----|----|----|----|----|----|-----|
| Q20 (gift card return) | 2-doc | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Q21 (3pm order + shipping) | 2-doc | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Q22 (price match + refund time) | 2-doc | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| Q23 (Klarna + $25 gift card) | 2-doc | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| Q24 (damaged + contact) | 2-doc | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Q25 (gift card + Canada) | 2-doc | ✗ | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ | ✓ |
| Q26 (Saturday + overnight) | 2-doc | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Q27 (PayPal + return refund) | 2-doc | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Q28 (express shipping cost) | 1-doc | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

**Multi-source success rate**: 7/9 questions reliably passed, 2/9 consistently failed.

---

## Accuracy by Parameter

### By Model
| Model | Accuracy | Samples |
|-------|----------|---------|
| gpt-3.5-turbo | 91.1% | 51/56 |
| gpt-4o | 89.3% | 25/28 |
| gpt-4o-mini | **92.9%** | 156/168 |

### By Retrieval Depth (k)
| k | Accuracy | Samples |
|---|----------|---------|
| 2 | 89.3% | 25/28 |
| 3 | 92.1% | 129/140 |
| 5 | **92.9%** | 78/84 |

**Observation**: Higher k correlates with better accuracy, supporting the multi-source hypothesis.

### By Chunk Size
| Chunk Size | Accuracy | Samples |
|------------|----------|---------|
| 300 | **92.9%** | 78/84 |
| 500 | 91.1% | 102/112 |
| 800 | **92.9%** | 52/56 |

### By Chunk Overlap
| Overlap | Accuracy | Samples |
|---------|----------|---------|
| 25 | 92.0% | 103/112 |
| 50 | 92.1% | 129/140 |

---

## Cost

**Total Cost**: $0.0327 USD

- Initial cost limit: $2.00
- Raised to: $25.20
- Actual spend: ~$0.03 (very efficient)

---

## Key Findings

### 1. Multi-Source Questions Reveal RAG Limitations
- 2 out of 9 multi-source questions **consistently failed across ALL configurations**
- These require logical reasoning (Q23: $25 < $35 minimum) or process disambiguation (Q22: price match vs return)
- This is not a retrieval problem - it's an LLM reasoning problem

### 2. gpt-4o-mini is Best Value
- Highest accuracy (92.9%) at lowest cost
- Outperformed gpt-4o (89.3%) - likely due to gpt-4o being tested with k=2

### 3. Higher k Helps (Slightly)
- k=5: 92.9% vs k=2: 89.3%
- Validates hypothesis that more context helps multi-source questions

### 4. Chunk Parameters Have Minimal Impact
- All chunk sizes achieved similar accuracy
- Chunk overlap 25 vs 50 nearly identical

### 5. gpt-4o Constraint Issue
- Trial 9 (gpt-4o, temp=0.7) violated constraint and returned N/A
- This is a known bug: #27 (constraint-rejected configs consume trial slots)

---

## Failure Analysis

### Q22: Price Match Refund Process
**Expected**: "Email support within 14 days, refund processed in 5-7 business days"
**Actual (all models)**: Confused price match with return process

**Root Cause**: The price_matching.md doc says "email support@example.com" but the returns_policy.md refund timing (5-7 days) is for returns, not price matches. The expected answer may be incorrect or the knowledge base is ambiguous.

### Q23: Klarna for $25 Gift Card
**Expected**: "No, buy now pay later requires orders over $35"
**Actual (all models)**: Said "Yes, you can use Klarna"

**Root Cause**: Models retrieved both facts ($35 minimum, Klarna available) but failed to reason that $25 < $35. This is an LLM reasoning failure, not a retrieval failure.

---

## Recommendations

### For Production
```python
config = {
    "model": "gpt-4o-mini",
    "temperature": 0.3,  # More consistent than 0.7
    "k": 5,              # Better for multi-source questions
    "chunk_size": 500,   # Middle ground
    "chunk_overlap": 50  # Standard overlap
}
```

### For Future Work
1. **Fix Q22/Q23 expected answers** or add explicit reasoning examples
2. **Test k=7 or k=10** to see if multi-source accuracy continues improving
3. **Add reasoning prompts** for questions requiring numerical comparisons
4. **Track per-question accuracy** separately for single vs multi-source

---

## Files Generated
- `results/rag_optimization_results.csv` - Full trial-by-trial results with all answers
- `results/rag_best_config.json` - Best configuration JSON
- `rag_feedback.jsonl` - Updated dataset with 28 questions
- `rag_feedback_sources.md` - Documentation of source documents per question
