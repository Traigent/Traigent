# RAG Feedback Dataset - Source Documentation

This file documents which knowledge base documents are needed to answer each question in `rag_feedback.jsonl`.

## Dataset Overview

**28 questions total** - A balanced mix designed to test RAG system capabilities:
- Kept partial-success questions (most informative for optimization)
- Added 10 new variations of challenging patterns
- Removed some 100% success and 0% success questions to focus on differentiating questions

## Question Categories by Success Pattern

### Original Partial Success Questions (kept)
These questions showed variable performance (some trials pass, some fail) - most useful for optimization:

| # | Question | Previous Pass Rate | Pattern |
|---|----------|-------------------|---------|
| 4 | Physical gift card to Canada | 78% (7/9) | International + domestic confusion |
| 7 | $100 gift card + Afterpay | 67% (6/9) | Gift card + BNPL eligibility |
| 13 | Payment methods under $35 | 11% (1/9) | Listing all valid options |

### Original Zero Success Questions (kept 4, removed 2)
Kept the most challenging numerical reasoning and multi-constraint questions:

| # | Question | Pattern | Why Kept |
|---|----------|---------|----------|
| 3 | Klarna $25 gift card | Numerical: $25 < $35 | Core reasoning failure |
| 14 | Overnight at 1pm EST | Numerical: 1pm > 12pm | Core reasoning failure |
| 16 | PO Box + Canada overnight | Multi-constraint | Tests constraint combination |
| 18 | Split refund gift card + Apple Pay | 3-source synthesis | Tests multi-source |

Removed: Q3 (price match - ambiguous), Q23 (10-day refund - incomplete KB info)

### Original 100% Success Questions (kept ~10, removed ~11)
Kept representative samples, removed redundant ones:

**Kept:**
- Q1: Gift card return refund
- Q2: 3pm weekday + standard shipping
- Q5: Saturday 1pm + overnight
- Q6: PayPal return refund
- Q8: Defective swimwear return
- Q9: Multiple gift cards + Apple Pay
- Q10: Affirm return refund
- Q11: Lost gift card replacement
- Q12: Friday 3pm + express shipping
- Q15: Corporate bulk to Canada
- Q17: 25-day return refund

**Removed (redundant patterns):**
- Damaged item contact (similar to defective returns)
- $40 express shipping cost (simple calculation)
- 11am overnight shipping (similar to other timing Qs)
- Cheapest 2-3 day shipping (similar to shipping Qs)
- Personalized damaged item (same as defective pattern)
- E-gift card to UK (simple yes answer)
- Google Pay Final Sale (simple no answer)
- $60 standard shipping (simple free shipping)
- $50 Klarna installments (simple answer)
- Track international order (simple answer)

### NEW Questions (10 added)
Based on partial success patterns - variations that should show mixed results:

| # | Question | Pattern | Expected Difficulty |
|---|----------|---------|---------------------|
| 19 | Affirm for $30 item | Numerical: $30 < $35 | High (numerical reasoning) |
| 20 | Physical gift card to Mexico overnight | International + overnight restriction | High (multi-constraint) |
| 21 | Afterpay for $40 order with gift card | BNPL + gift card exclusion | Medium-High |
| 22 | 2:30pm EST overnight timing | Numerical: 2:30pm > 2pm cutoff | High (numerical reasoning) |
| 23 | Shop Pay + gift card split refund | 3-source synthesis | High (like Q18) |
| 24 | Express to PO Box US | PO Box restriction only | Medium |
| 25 | Afterpay for $25 item | Numerical: $25 < $35 | High (like Q3) |
| 26 | Defective underwear/intimates | Exception + category | Medium (like Q8) |
| 27 | Google Pay + gift card split refund | 3-source synthesis | High (like Q18) |
| 28 | Standard shipping to international PO Box | International + PO Box | Medium-High |

## Question Details

### Numerical Reasoning Questions (5 questions)
Tests if model can compare numbers:
- Q3: $25 < $35 minimum → NO
- Q14: 1pm > 12pm cutoff → delayed
- Q19: $30 < $35 minimum → NO
- Q22: 2:30pm > 2pm cutoff → delayed
- Q25: $25 < $35 minimum → NO

### Multi-Constraint Questions (4 questions)
Tests combining multiple restrictions:
- Q16: PO Box + international + overnight → NO for multiple reasons
- Q20: International + overnight → NO
- Q24: PO Box + express → NO
- Q28: International + PO Box → depends on carrier

### Split Payment Refund Questions (3 questions)
Tests 3-source synthesis (gift card + payment + returns):
- Q18: Gift card + Apple Pay split → different destinations
- Q23: Gift card + Shop Pay split → different destinations
- Q27: Gift card + Google Pay split → different destinations

### Gift Card + BNPL Questions (3 questions)
Tests understanding of BNPL restrictions on gift cards:
- Q3: Klarna for $25 gift card → NO ($25 < $35)
- Q7: Afterpay for $100 gift card → depends on interpretation
- Q21: Afterpay for order WITH gift card → NO (gift cards excluded)

### Timing/Shipping Questions (5 questions)
Tests order processing and shipping timing:
- Q2: 3pm weekday + standard → next day ship
- Q5: Saturday 1pm + overnight → Tuesday arrival
- Q12: Friday 3pm + express → Monday ship, Wed-Thu arrive
- Q14: 1pm EST overnight → misses 12pm cutoff
- Q22: 2:30pm EST overnight → misses 2pm cutoff

### Exception Handling Questions (2 questions)
Tests understanding that defective overrides non-returnable:
- Q8: Defective swimwear → YES can return
- Q26: Defective intimates → YES can return

## Expected Outcome Distribution

Based on original patterns, we expect:
- ~6 questions: 100% pass rate (easy baseline)
- ~8 questions: 50-90% pass rate (partial success - most informative)
- ~8 questions: 10-50% pass rate (challenging but some success)
- ~6 questions: 0-10% pass rate (very hard - systematic failures)

This distribution should:
1. Better differentiate configurations
2. Reveal which improvements help which question types
3. Provide more signal for optimization

## Source Documents Reference

All questions draw from these knowledge base files:
- `returns_policy.md` - Return windows, refund process, exceptions
- `shipping_info.md` - Shipping options, costs, restrictions, timing
- `payment_methods.md` - Accepted payments, BNPL minimums, digital wallets
- `gift_cards.md` - Gift card types, usage, refund policies
- `store_hours.md` - Order processing times, cutoffs
- `contact_support.md` - Support contact info
- `price_matching.md` - Price match policy (removed from dataset - ambiguous)
