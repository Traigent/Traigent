# RAG Feedback Dataset - Source Documentation

This file documents which knowledge base documents are needed to answer each question in `rag_feedback.jsonl`.

## Dataset Overview

**All 30 questions are multi-source** - requiring synthesis of information from 2+ documents or multiple facts within documents. This makes the dataset challenging and tests the RAG system's ability to:
1. Retrieve relevant chunks from multiple documents
2. Synthesize information across different sources
3. Apply logical reasoning to combine facts

## Questions by Complexity

### 2-Source Questions

| # | Question | Source Documents | Reasoning Required |
|---|----------|------------------|-------------------|
| 1 | Gift card return refund | returns_policy.md, gift_cards.md | Return process + gift card refund policy |
| 2 | 3pm weekday order + standard shipping | store_hours.md, shipping_info.md | Order cutoff + shipping duration |
| 3 | Price match refund timing | price_matching.md, returns_policy.md | Price match process + refund timing |
| 4 | Klarna for $25 gift card | payment_methods.md, gift_cards.md | BNPL $35 min + $25 < $35 = NO |
| 5 | Damaged item contact + timeframe | returns_policy.md, contact_support.md | Damage reporting + contact info |
| 6 | Physical gift card to Canada | gift_cards.md, shipping_info.md | Gift card shipping + international rates |
| 7 | Saturday 1pm + overnight shipping | store_hours.md, shipping_info.md | Weekend processing + overnight timing |
| 8 | PayPal return refund | payment_methods.md, returns_policy.md | Payment method + refund timing |
| 9 | $40 order + express shipping cost | shipping_info.md | Express rate + free threshold check |
| 10 | $100 gift card + Afterpay | gift_cards.md, payment_methods.md | Gift card amount + BNPL availability |
| 11 | Defective swimwear return | returns_policy.md | Exception + defective override |
| 12 | 11am EST + overnight | store_hours.md, shipping_info.md | Cutoff time (12pm) + overnight rules |
| 13 | Multiple gift cards + Apple Pay | gift_cards.md, payment_methods.md | Stackable + digital wallet combo |
| 14 | Affirm return refund | payment_methods.md, returns_policy.md | BNPL payment + refund process |
| 15 | Cheapest 2-3 day shipping for $45 | shipping_info.md | Express vs standard timing + costs |
| 16 | Lost gift card replacement + online use | gift_cards.md, contact_support.md | Lost card process + flexible use |
| 17 | Friday 3pm + express shipping | store_hours.md, shipping_info.md | Weekend processing + express timing |
| 18 | Damaged personalized item | returns_policy.md | Personalized exception + defective override |
| 19 | E-gift card to UK | gift_cards.md, shipping_info.md | E-gift delivery + international |
| 20 | Payment methods under $35 | payment_methods.md | All methods minus BNPL restriction |
| 21 | Overnight at 1pm EST timing | shipping_info.md | 12pm cutoff violation + consequence |
| 22 | Corporate bulk to Canada | gift_cards.md, shipping_info.md | Bulk orders + international shipping |
| 23 | 10 business day credit card refund | returns_policy.md | Refund timing + credit card delay |
| 24 | Google Pay + Final Sale return | payment_methods.md, returns_policy.md | Payment method + Final Sale exception |
| 25 | $60 order + standard shipping cost | shipping_info.md | Free shipping threshold ($50) check |
| 26 | $50 e-gift card + Klarna installments | gift_cards.md, payment_methods.md | Gift card + Klarna payment options |
| 27 | Overnight to PO Box in Canada | shipping_info.md | PO Box restriction + international |
| 28 | 25-day return refund type | returns_policy.md | 30-day window check |
| 29 | Track international UK order | shipping_info.md | Tracking + international confirmation |
| 30 | Gift card + Apple Pay split refund | gift_cards.md, payment_methods.md, returns_policy.md | Split payment refund routing |

### 3-Source Questions

| # | Question | Source Documents |
|---|----------|------------------|
| 30 | Gift card + Apple Pay split refund | gift_cards.md, payment_methods.md, returns_policy.md |

## Question Categories

### Timing/Scheduling (7 questions)
Questions 2, 7, 12, 17, 21, 27, 28

### Payment + Policy Interaction (10 questions)
Questions 1, 3, 4, 8, 10, 13, 14, 20, 24, 26, 30

### Shipping + Product/Policy (6 questions)
Questions 6, 9, 15, 19, 22, 25

### Returns + Exceptions (5 questions)
Questions 5, 11, 18, 23, 29

### Lost/Issues Resolution (2 questions)
Questions 16, 27

## Difficulty Factors

### Logical Reasoning Required
- Q4: $25 < $35 minimum → NO
- Q21: 1pm > 12pm cutoff → delayed
- Q23: 10 days > expected 12 days → contact support
- Q28: 25 days < 30 day window → full refund

### Exception Handling
- Q11: Swimwear (non-returnable) BUT defective → returnable
- Q18: Personalized (non-returnable) BUT defective → returnable
- Q24: Any payment BUT Final Sale → non-returnable

### Multi-Step Calculations
- Q7: Saturday → Monday ship → Tuesday arrive
- Q17: Friday 3pm → Monday ship → Wed-Thu arrive
- Q15: Cost comparison between express and standard for time requirement

### Split/Combined Scenarios
- Q13: Multiple gift cards + digital wallet together
- Q30: Partial refund to different payment methods

## Purpose

This challenging dataset is designed to:
1. **Test retrieval breadth**: Higher `k` values should help retrieve relevant chunks from multiple documents
2. **Test LLM reasoning**: Models must combine facts, not just regurgitate single answers
3. **Differentiate configurations**: Simple questions pass with any config; these reveal which configs handle complexity better
4. **Identify failure patterns**: Track which question types fail to guide improvements
