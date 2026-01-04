# RAG Feedback Dataset - Source Documentation

This file documents which knowledge base documents are needed to answer each question in `rag_feedback.jsonl`.

## Single-Source Questions (1 document needed)

| # | Question | Source Document |
|---|----------|-----------------|
| 1 | How many days do I have to return an item? | returns_policy.md |
| 2 | What's the shipping cost for a $40 order? | shipping_info.md |
| 3 | How long does express shipping take? | shipping_info.md |
| 4 | Can I return swimwear? | returns_policy.md |
| 5 | How long until my refund shows up on my credit card? | returns_policy.md |
| 6 | Do you ship to Canada? | shipping_info.md |
| 7 | What's the cost for overnight delivery? | shipping_info.md |
| 8 | Will you match Amazon third-party seller prices? | price_matching.md |
| 9 | How do I report a damaged item? | returns_policy.md |
| 10 | What time do I need to order by for same-day processing? | store_hours.md |
| 11 | Can I get a price match after I already bought something? | price_matching.md |
| 12 | Do you match Costco prices? | price_matching.md |
| 13 | What happens to my gift card balance if I don't use it all? | gift_cards.md |
| 14 | When is phone support available? | contact_support.md |
| 15 | What's the minimum order for free shipping? | shipping_info.md |
| 16 | Do you accept Apple Pay? | payment_methods.md |
| 17 | What's the minimum order for buy now pay later? | payment_methods.md |
| 18 | What's your support email address? | contact_support.md |
| 19 | What's the average wait time for phone support? | contact_support.md |

## Multi-Source Questions (2+ documents needed)

| # | Question | Source Documents | Notes |
|---|----------|------------------|-------|
| 20 | If I return an item I paid for with a gift card, how do I get my money back? | returns_policy.md, gift_cards.md | Needs return process + gift card refund policy |
| 21 | I ordered at 3pm on a weekday - when will it ship and how long until it arrives with standard shipping? | store_hours.md, shipping_info.md | Needs order cutoff time + shipping duration |
| 22 | If I find a lower price after buying, how do I request a refund and how long until I get it? | price_matching.md, returns_policy.md | Needs price match process + refund timing |
| 23 | Can I use Klarna to buy a $25 gift card? | payment_methods.md, gift_cards.md | Needs BNPL minimum + gift card amounts |
| 24 | If my item arrives damaged, who do I contact and within what timeframe? | returns_policy.md, contact_support.md | Needs damage reporting + contact info |
| 25 | I want to buy a physical gift card for someone in Canada - can I ship it there and how much will it cost? | gift_cards.md, shipping_info.md | Needs gift card shipping + international rates |
| 26 | If I order at 1pm on Saturday and choose overnight shipping, when will it arrive? | store_hours.md, shipping_info.md | Needs weekend processing + overnight timing |
| 27 | I paid with PayPal and returned an item - when will I see the refund? | payment_methods.md, returns_policy.md | Needs payment method + refund timing |
| 28 | What's the total cost to ship a $40 order with express shipping? | shipping_info.md, shipping_info.md | Needs express rate + free shipping threshold (same doc, multiple facts) |

## Summary

- **Single-source questions**: 19 (questions 1-19)
- **Multi-source questions**: 9 (questions 20-28)
- **Total questions**: 28

## Purpose

Multi-source questions are designed to test whether the RAG system can:
1. Retrieve relevant chunks from multiple documents
2. Synthesize information across different sources
3. Provide accurate answers that require combining facts

Higher `k` values (more retrieved chunks) should improve performance on multi-source questions.
