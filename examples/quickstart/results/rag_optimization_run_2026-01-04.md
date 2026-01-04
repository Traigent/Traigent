# RAG Optimization Run - January 4, 2026 (Updated)

## Overview

Ran TraiGent optimization on the Customer Support RAG example with an **all multi-source dataset** (30 challenging questions requiring synthesis across 2-3 documents).

---

## Dataset: All Multi-Source Questions

The dataset was redesigned to contain **only challenging multi-source questions**. All 19 simple single-source questions were removed and replaced with questions requiring:
- Synthesis of information from 2-3 documents
- Logical reasoning (e.g., $25 < $35 minimum = NO)
- Exception handling (defective overrides non-returnable)
- Multi-step calculations (weekend → Monday ship → Tuesday arrive)

**Total Questions**: 30 (all multi-source)

---

## Results Summary

### Best Configuration
```json
{
  "model": "gpt-4o-mini",
  "temperature": 0.3,
  "k": 3,
  "chunk_size": 800,
  "chunk_overlap": 50
}
```

**Best Score**: 80.0% accuracy (24/30 questions)

### All Trials

| Trial | Model | k | Chunk | Score | Pass Rate |
|-------|-------|---|-------|-------|-----------|
| T1 | gpt-3.5-turbo | 3 | 500 | 76.7% | 23/30 |
| T2 | gpt-4o-mini | 2 | 300 | 73.3% | 22/30 |
| T3 | gpt-4o-mini | 2 | 300 | 76.7% | 23/30 |
| T4 | gpt-4o-mini | 3 | 300 | 76.7% | 23/30 |
| T5 | gpt-3.5-turbo | 5 | 800 | 76.7% | 23/30 |
| T6 | gpt-4o-mini | 3 | 800 | **80.0%** | **24/30** |
| T7 | gpt-3.5-turbo | 2 | 500 | N/A | (constraint) |
| T8 | gpt-4o | 3 | 500 | 73.3% | 22/30 |
| T9 | gpt-4o-mini | 2 | 300 | 73.3% | 22/30 |
| T10 | gpt-3.5-turbo | 3 | 800 | 73.3% | 22/30 |

---

## Per-Question Performance Matrix

| # | Question Summary | Sources | T1 | T2 | T3 | T4 | T5 | T6 | T8 | T9 | T10 |
|---|------------------|---------|----|----|----|----|----|----|----|----|-----|
| 1 | Gift card return refund | returns + gift_cards | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| 2 | 3pm weekday + standard shipping | hours + shipping | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| 3 | Price match refund timing | price_match + returns | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| 4 | Klarna for $25 gift card | payments + gift_cards | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| 5 | Damaged item contact | returns + support | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| 6 | Physical gift card to Canada | gift_cards + shipping | ✓ | ✗ | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ |
| 7 | Saturday 1pm + overnight | hours + shipping | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| 8 | PayPal return refund | payments + returns | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| 9 | $40 + express shipping cost | shipping | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| 10 | $100 gift card + Afterpay | gift_cards + payments | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ |
| 11 | Defective swimwear return | returns (exception) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| 12 | 11am EST + overnight | hours + shipping | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| 13 | Multiple gift cards + Apple Pay | gift_cards + payments | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| 14 | Affirm return refund | payments + returns | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| 15 | Cheapest 2-3 day shipping $45 | shipping | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| 16 | Lost gift card replacement | gift_cards + support | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| 17 | Friday 3pm + express | hours + shipping | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| 18 | Damaged personalized item | returns (exception) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| 19 | E-gift card to UK | gift_cards + shipping | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| 20 | Payment methods under $35 | payments | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ |
| 21 | Overnight at 1pm EST | shipping | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| 22 | Corporate bulk to Canada | gift_cards + shipping | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| 23 | 10 day credit card refund | returns | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| 24 | Google Pay + Final Sale return | payments + returns | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| 25 | $60 + standard shipping | shipping | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| 26 | $50 e-gift card + Klarna | gift_cards + payments | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| 27 | Overnight to PO Box Canada | shipping | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| 28 | 25-day return refund type | returns | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| 29 | Track international UK order | shipping | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| 30 | Split refund (gift card + Apple Pay) | gift_cards + payments + returns | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| **Total** | | | **23** | **22** | **23** | **23** | **23** | **24** | **22** | **22** | **22** |

**Legend**: ✓ = Pass, ✗ = Fail, T7 omitted (constraint violation)

---

## Failure Analysis by Question

### Consistently Failing Questions (6 questions - 0% pass rate)

| # | Question | Expected Answer | Actual Behavior | Root Cause |
|---|----------|-----------------|-----------------|------------|
| 3 | Price match refund timing | "Email support within 14 days, 5-7 days refund" | Confused with return refund process | Multi-source synthesis failure |
| 4 | Klarna for $25 gift card | "No, requires $35 minimum" | All said "Yes" | **Numerical reasoning failure** ($25 < $35) |
| 21 | Overnight at 1pm EST | "No, cutoff is 12pm" | All said "Yes" | **Numerical reasoning failure** (1pm > 12pm) |
| 23 | 10 day credit card refund | "Contact support if over 12 days total" | Said 5-7 days abnormal | Incomplete reasoning (missed 3-5 day credit card delay) |
| 27 | Overnight to PO Box Canada | "No overnight for PO Box + no international overnight" | Only mentioned PO Box restriction | Multi-constraint synthesis failure |
| 30 | Split refund (gift card + Apple Pay) | "Gift card → gift card, Apple Pay → original payment" | Only mentioned gift card portion | **3-source synthesis failure** |

### Partially Failing Questions (3 questions)

| # | Question | Pass Rate | Issue |
|---|----------|-----------|-------|
| 6 | Physical gift card to Canada | 7/9 (78%) | T2, T8 missed international shipping rates |
| 10 | $100 gift card + Afterpay | 6/9 (67%) | T8, T9, T10 said Afterpay unavailable for gift cards |
| 20 | Payment methods under $35 | 1/9 (11%) | Most trials missed digital wallets or gift cards |

---

## Failure Pattern Classification

### Pattern 1: Numerical Reasoning Failures (2 questions)
- **Q4**: $25 < $35 minimum → All models said YES
- **Q21**: 1pm > 12pm cutoff → All models said YES

**Analysis**: Models retrieve the facts ($35 minimum, 12pm cutoff) but fail to compare against the query values ($25, 1pm).

### Pattern 2: Multi-Source Synthesis Failures (3 questions)
- **Q3**: Price match process (doc A) + refund timing (doc B) → Conflated with return process
- **Q27**: PO Box restriction + international restriction → Only mentioned one
- **Q30**: 3-way split (gift card + Apple Pay + refund policy) → Only addressed gift card

**Analysis**: Models retrieve from one source but miss or ignore the second/third required source.

### Pattern 3: Incomplete Reasoning (1 question)
- **Q23**: 5-7 days processing + 3-5 days credit card = 12 days total → Models only cited 5-7 days

**Analysis**: Models retrieve facts but don't perform the addition to get total expected time.

---

## Accuracy by Parameter

### By Model
| Model | Avg Accuracy | Best | Notes |
|-------|-------------|------|-------|
| gpt-3.5-turbo | 75.6% | 76.7% | Consistent but lower ceiling |
| gpt-4o | 73.3% | 73.3% | Only 1 trial, underperformed |
| gpt-4o-mini | **76.1%** | **80.0%** | Best overall, highest ceiling |

### By Retrieval Depth (k)
| k | Trials | Best Accuracy | Notes |
|---|--------|---------------|-------|
| 2 | T2, T3, T9 | 76.7% | Lower - fewer chunks hurts multi-source |
| 3 | T1, T4, T6, T8, T10 | **80.0%** | Best - optimal balance |
| 5 | T5 | 76.7% | No improvement over k=3 |

### By Chunk Size
| Chunk Size | Best Accuracy | Notes |
|------------|---------------|-------|
| 300 | 76.7% | Smaller chunks fragment context |
| 500 | 76.7% | Middle ground |
| 800 | **80.0%** | Best - keeps related facts together |

---

## Key Findings

### 1. Multi-Source Questions Are Hard
- Previous dataset (with simple questions): **92.9%** best score
- New all-multi-source dataset: **80.0%** best score
- **12.9% accuracy drop** when testing only challenging questions

### 2. Numerical Reasoning is the Biggest Gap
- Q4 and Q21 require simple math ($25 < $35, 1pm > 12pm)
- **100% failure rate** across all configurations
- This is an LLM reasoning failure, not retrieval

### 3. 3-Source Questions Are Hardest
- Q30 (gift card + Apple Pay + returns) failed 100%
- Models can handle 2-source but struggle with 3-source

### 4. k=3 with Large Chunks is Optimal
- k=3 retrieves enough context for 2-source questions
- chunk_size=800 keeps related facts together in single chunks
- Best config: gpt-4o-mini, k=3, chunk=800

### 5. gpt-4o Underperforms
- gpt-4o (73.3%) < gpt-4o-mini (80.0%)
- May be due to gpt-4o being more verbose and missing key details

---

## Comparison: Previous vs Current Dataset

| Metric | Previous (28 Q, mixed) | Current (30 Q, all multi-source) |
|--------|----------------------|----------------------------------|
| Best Score | 92.9% | 80.0% |
| Worst Score | 89.3% | 73.3% |
| Score Range | 3.6% | 6.7% |
| Consistently Failing Qs | 2 | 6 |
| Numerical Reasoning Failures | 1 (Q4) | 2 (Q4, Q21) |

**Conclusion**: The harder dataset better differentiates configurations and reveals systematic model limitations.

---

## Recommendations

### For Production RAG
```python
config = {
    "model": "gpt-4o-mini",
    "temperature": 0.3,
    "k": 3,
    "chunk_size": 800,
    "chunk_overlap": 50
}
```

### To Improve Accuracy on Failing Questions
1. **Add explicit comparison prompts** for numerical questions (Q4, Q21)
2. **Increase k to 5-7** for known 3-source questions
3. **Add "check all constraints" instruction** in prompts for compound questions
4. **Consider chain-of-thought prompting** for reasoning questions

---

## Files

- `rag_feedback.jsonl` - 30 multi-source questions
- `rag_feedback_sources.md` - Source documentation per question
- `results/rag_optimization_results.csv` - Full trial results with all answers
- `results/rag_best_config.json` - Best configuration
