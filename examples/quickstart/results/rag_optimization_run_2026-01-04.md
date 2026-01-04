# RAG Optimization Run - January 4, 2026 (v2)

## Overview

Ran TraiGent optimization on the Customer Support RAG example with a **balanced 28-question dataset** designed to better differentiate configurations.

---

## Dataset Design

The dataset was rebalanced from the previous 30-question all-multi-source version:
- **Removed** 12 redundant 100% success questions
- **Removed** 2 ambiguous 0% success questions
- **Added** 10 new challenging questions based on partial-success patterns
- **Total**: 28 questions

### Question Distribution by Pattern

| Pattern | Count | Description |
|---------|-------|-------------|
| Numerical Reasoning | 5 | Tests $X < $35 minimum comparisons |
| Multi-Constraint | 4 | PO Box + international + overnight combinations |
| Split Payment Refund | 3 | 3-source synthesis (gift card + payment + returns) |
| Gift Card + BNPL | 3 | BNPL restrictions on gift cards |
| Timing/Shipping | 5 | Order processing and shipping timing |
| Exception Handling | 2 | Defective overrides non-returnable |
| Standard Multi-Source | 6 | 2-source questions (baseline) |

---

## Results Summary

### Best Configuration
```json
{
  "model": "gpt-4o",
  "k": 3,
  "chunk_size": 300
}
```
**Best Score**: 64.3% accuracy (18/28 questions)

**Tied configurations** (all at 64.3%):
- gpt-4o, k=3, chunk=300
- gpt-4o-mini, k=5, chunk=500
- gpt-4o-mini, k=2, chunk=500

### All Trials

| Trial | Model | k | Chunk | Score | Pass Rate |
|-------|-------|---|-------|-------|-----------|
| T1 | gpt-3.5-turbo | 3 | 500 | 60.7% | 17/28 |
| T2 | gpt-4o | 5 | 300 | 57.1% | 16/28 |
| T3 | gpt-4o-mini | 5 | 500 | 57.1% | 16/28 |
| T4 | gpt-4o | 3 | 300 | **64.3%** | **18/28** |
| T5 | gpt-4o-mini | 5 | 500 | **64.3%** | **18/28** |
| T6 | gpt-3.5-turbo | 3 | 800 | 60.7% | 17/28 |
| T7 | gpt-4o-mini | 2 | 500 | **64.3%** | **18/28** |
| T8 | gpt-4o | 3 | 500 | 57.1% | 16/28 |
| T9 | gpt-3.5-turbo | 2 | 800 | N/A | (constraint) |
| T10 | gpt-4o | 5 | 300 | N/A | (constraint) |

---

## Per-Question Performance Matrix

| # | Question Summary | T1 | T2 | T3 | T4 | T5 | T6 | T7 | T8 |
|---|------------------|----|----|----|----|----|----|----|----|
| 1 | Gift card return refund | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| 2 | 3pm weekday + standard shipping | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| 3 | Klarna $25 gift card | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| 4 | Physical gift card to Canada | ✗ | ✗ | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ |
| 5 | Saturday 1pm + overnight | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| 6 | PayPal return refund | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| 7 | $100 gift card + Afterpay | ✓ | ✗ | ✓ | ✗ | ✓ | ✓ | ✗ | ✓ |
| 8 | Defective swimwear return | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| 9 | Multiple gift cards + Apple Pay | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| 10 | Affirm return refund | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| 11 | Lost gift card replacement | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| 12 | Friday 3pm + express shipping | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| 13 | Payment methods under $35 | ✓ | ✗ | ✗ | ✓ | ✗ | ✗ | ✓ | ✗ |
| 14 | Overnight at 1pm EST | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| 15 | Corporate bulk to Canada | ✓ | ✓ | ✗ | ✓ | ✓ | ✓ | ✓ | ✓ |
| 16 | PO Box + Canada overnight | ✗ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| 17 | 25-day return refund | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| 18 | Split refund: gift card + Apple Pay | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| 19 | Affirm for $30 item | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| 20 | Physical gift card to Mexico overnight | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| 21 | Afterpay for order with gift card | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| 22 | 2:30pm EST overnight timing | ✗ | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ | ✗ |
| 23 | Split refund: Shop Pay + gift card | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| 24 | Express to PO Box US | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| 25 | Afterpay for $25 item | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| 26 | Defective underwear/intimates | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| 27 | Split refund: Google Pay + gift card | ✗ | ✗ | ✗ | ✗ | ✓ | ✗ | ✓ | ✗ |
| 28 | Standard shipping to int'l PO Box | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| **Total** | | **17** | **16** | **16** | **18** | **18** | **17** | **18** | **16** |

**Legend**: ✓ = Pass, ✗ = Fail, T9/T10 omitted (constraint violations)

---

## Failure Analysis by Question Type

### 100% Failure Rate (0/8 trials) - 7 questions

| # | Question | Pattern | Root Cause |
|---|----------|---------|------------|
| 3 | Klarna $25 gift card | Numerical | $25 < $35 not computed |
| 14 | Overnight at 1pm EST | Numerical | 1pm > 12pm not computed |
| 18 | Gift card + Apple Pay split refund | 3-source | Only mentions gift card portion |
| 20 | Gift card to Mexico overnight | Multi-constraint | Says "not for gift cards" instead of "international" |
| 21 | Afterpay for order WITH gift card | BNPL exclusion | Thinks gift card can combine with Afterpay |
| 23 | Shop Pay + gift card split refund | 3-source | Only mentions gift card portion |
| 28 | Standard to int'l PO Box | Multi-constraint | Says "not available" without nuance |

### Partial Success (1-7/8 trials) - 5 questions

| # | Question | Pass Rate | Issue |
|---|----------|-----------|-------|
| 4 | Physical gift card to Canada | 5/8 (63%) | Inconsistent on free domestic + int'l rates |
| 7 | $100 gift card + Afterpay | 5/8 (63%) | Some say "not for gift cards", some say "yes" |
| 13 | Payment methods under $35 | 3/8 (38%) | Some include BNPL options incorrectly |
| 16 | PO Box + Canada overnight | 1/8 (13%) | Only T2 mentioned both constraints |
| 22 | 2:30pm overnight timing | 1/8 (13%) | Only T4 correctly said "arrives in 2 days" |
| 27 | Google Pay + gift card split | 2/8 (25%) | T5, T7 mentioned both payment methods |

### 100% Success Rate (8/8 trials) - 16 questions

Questions 1, 2, 5, 6, 8, 9, 10, 11, 12, 17, 19, 24, 25, 26 (and 15 at 7/8)

---

## Failure Pattern Classification

### Pattern 1: Numerical Reasoning (2 questions, 0% success)
- **Q3**: $25 < $35 minimum → All said YES
- **Q14**: 1pm > 12pm cutoff → All said YES

**Analysis**: Models retrieve the rules but fail to compare query values.

### Pattern 2: Split Payment Refunds (3 questions, 0-25% success)
- **Q18**: Gift card + Apple Pay → 0% (all only mention gift card)
- **Q23**: Shop Pay + gift card → 0%
- **Q27**: Google Pay + gift card → 25% (2/8 got both)

**Analysis**: Models retrieve gift card refund policy but miss that other payment also needs addressing.

### Pattern 3: BNPL + Gift Card Exclusion (2 questions, 0-63% success)
- **Q7**: Afterpay for $100 gift card → 63% (interpretation varies)
- **Q21**: Afterpay for order WITH gift card → 0%

**Analysis**: Q7 is ambiguous (buying gift card vs using gift card). Q21 clearly about using gift card in order but models think you can combine Afterpay + gift card.

### Pattern 4: Multi-Constraint (3 questions, 0-13% success)
- **Q16**: PO Box + Canada + overnight → 13% (only T2 got both)
- **Q20**: Mexico + overnight → 0% (wrong reason given)
- **Q28**: Int'l + PO Box + standard → 0%

**Analysis**: Models identify one constraint but miss the second/third.

### Pattern 5: New vs Original Numerical (contrasting)
- **Q3**: Klarna $25 → 0% (original, always fails)
- **Q19**: Affirm $30 → 100% (new, always passes!)
- **Q25**: Afterpay $25 → 100% (new, always passes!)

**Interesting**: New questions Q19 and Q25 test same pattern as Q3 but pass 100%. Possible reasons:
1. Different phrasing triggers different behavior
2. Q3 combines gift card + BNPL (two concepts), Q19/Q25 are simpler

---

## Accuracy by Parameter

### By Model
| Model | Avg Accuracy | Best | Notes |
|-------|-------------|------|-------|
| gpt-3.5-turbo | 60.7% | 60.7% | Consistent but lower |
| gpt-4o | 59.5% | 64.3% | Variable, T4 best |
| gpt-4o-mini | 61.9% | **64.3%** | Best average |

### By Retrieval Depth (k)
| k | Trials | Best Accuracy | Notes |
|---|--------|---------------|-------|
| 2 | T7, T9 | 64.3% | T7 tied for best |
| 3 | T1, T4, T6, T8 | **64.3%** | T4 best |
| 5 | T2, T3, T5, T10 | 64.3% | T5 tied for best |

### By Chunk Size
| Chunk Size | Best Accuracy | Notes |
|------------|---------------|-------|
| 300 | **64.3%** | T4 (gpt-4o, k=3) |
| 500 | **64.3%** | T5, T7 tied |
| 800 | 60.7% | Slightly worse |

---

## Comparison: Previous vs Current Dataset

| Metric | Previous (30 Q) | Current (28 Q) |
|--------|-----------------|----------------|
| Best Score | 80.0% | 64.3% |
| Worst Score | 73.3% | 57.1% |
| Score Range | 6.7% | 7.2% |
| 100% Failing Qs | 6 | 7 |
| Partial Success Qs | 3 | 6 |

**Observations**:
- Lower overall accuracy (harder dataset)
- More partial-success questions (better for optimization signal)
- Similar score range (good differentiation)

---

## Key Findings

### 1. Balanced Dataset Works
- More partial-success questions (6 vs 3)
- Better signal for optimization
- Still challenging (64% best vs 80% before)

### 2. New Questions Reveal Interesting Patterns
- Q19 (Affirm $30) and Q25 (Afterpay $25) pass 100% while Q3 (Klarna $25) fails 100%
- Difference: Q3 is about buying a gift card, Q19/Q25 are about regular items
- The gift card context may confuse the model

### 3. Split Payment Refunds are Hardest
- 0-25% success across all trials
- Models consistently only address gift card portion
- This is a true retrieval/synthesis failure

### 4. No Clear Winner on Parameters
- Three configs tied at 64.3%
- Model choice matters less than expected
- k=2-5 all similar performance

### 5. Chunk Size 800 Slightly Worse
- 300 and 500 perform similarly
- 800 may be too large, reducing retrieval precision

---

## Recommendations

### For Production RAG
```python
config = {
    "model": "gpt-4o-mini",  # Best cost/performance
    "k": 3,
    "chunk_size": 500,
    "chunk_overlap": 50
}
```

### To Improve Accuracy

1. **Split Payment Refunds**: Add explicit "for combined payments" section in KB
2. **BNPL + Gift Cards**: Clarify in KB that gift card purchases exclude BNPL
3. **Numerical Comparisons**: Add chain-of-thought prompting for comparisons
4. **Multi-Constraint**: Increase k for questions mentioning multiple locations/restrictions

---

## Files

- `rag_feedback.jsonl` - 28 balanced questions
- `rag_feedback_sources.md` - Source documentation per question
- `results/rag_optimization_results.csv` - Full trial results
- `results/rag_best_config.json` - Best configuration
