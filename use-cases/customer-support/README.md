# Customer Support Agent Use Case

This use case demonstrates optimizing a **customer support agent** for ShopEasy, an e-commerce platform.

## Overview

The customer support agent:
1. Receives customer inquiries with context (tier, sentiment, order status)
2. Generates helpful, empathetic responses
3. Decides whether to escalate to a supervisor

It optimizes for:
- **Resolution Accuracy** - Did the response correctly address the issue?
- **Tone Quality** - Empathy, clarity, and professionalism scores
- **Escalation Accuracy** - Correct escalation decisions (precision/recall)

## Quick Start

```bash
# From project root
cd /path/to/Traigent

# Enable mock mode (recommended for testing)
export TRAIGENT_MOCK_MODE=true

# Run the agent optimization
python use-cases/customer-support/agent/support_agent.py
```

## Configuration Space

| Parameter | Values | Description |
|-----------|--------|-------------|
| `model` | gpt-3.5-turbo, gpt-4o-mini, gpt-4o | LLM model selection |
| `temperature` | 0.3, 0.5, 0.7 | Higher for more varied responses |
| `tone` | professional, friendly, empathetic | Response tone style |
| `empathy_level` | moderate, high, very_high | Degree of empathy shown |
| `escalation_threshold` | conservative, moderate, aggressive | When to escalate issues |

## Dataset

The evaluation dataset (`datasets/support_tickets.jsonl`) contains 50+ support scenarios including:

- Refund requests (damaged items, wrong items, changed mind)
- Order tracking inquiries
- Cancellation requests
- Complaints and escalation scenarios
- Account issues
- Product questions

### Sample Entry

```json
{
  "input": {
    "query": "I received a damaged laptop and I want a full refund immediately",
    "customer_context": {
      "customer_tier": "gold",
      "sentiment": "negative",
      "order_status": "delivered",
      "previous_interactions": 0
    }
  },
  "expected": {
    "gold_response": "Apologize for inconvenience, offer refund/replacement options...",
    "should_escalate": false,
    "resolution_type": "refund"
  }
}
```

## Evaluation Metrics

### Resolution Accuracy (40% weight)

Uses LLM-as-judge with calibrated rubric:

- **Accuracy Score (1-5)**: How well the response addresses the issue
- **Resolution Match**: Did it match the expected resolution type?
- **Policy Correct**: Were company policies correctly applied?
- **Next Steps Clear**: Were actionable next steps provided?

### Tone Quality (35% weight)

Three sub-dimensions evaluated:

- **Empathy (1-5)**: Acknowledgment of customer feelings
- **Clarity (1-5)**: Clear communication and structure
- **Professionalism (1-5)**: Appropriate language and brand voice

### Escalation Accuracy (25% weight)

Deterministic evaluation:

- **Precision**: Of escalated cases, how many should have been?
- **Recall**: Of cases that should escalate, how many did?
- **F1 Score**: Harmonic mean of precision and recall

## Escalation Criteria

The agent should escalate when:
- Customer explicitly requests supervisor/manager
- Legal threats or serious complaints
- Safety or security issues
- Requests beyond agent authority (e.g., refunds > $500)
- Repeated failed resolution attempts
- VIP customers with unresolved high-severity issues

## Files

```
customer-support/
├── agent/
│   └── support_agent.py          # Customer support agent
├── datasets/
│   └── support_tickets.jsonl     # 50+ support scenarios
├── eval/
│   └── evaluator.py              # Resolution + tone + escalation evaluator
└── README.md
```

## Expected Results

After optimization, you should see results like:

```
Best Configuration:
  model: gpt-4o-mini
  temperature: 0.5
  tone: empathetic
  empathy_level: high
  escalation_threshold: moderate

Best Score: 0.78
```

## Company Policies (for evaluation)

ShopEasy support policies:
- Returns accepted within 30 days with original receipt
- Refunds processed within 5-7 business days
- VIP customers (gold/platinum) get priority support
- Order modifications possible until shipping begins
- Damaged items: immediate replacement or refund option

## Customization

### Adding Support Scenarios

Add entries to `datasets/support_tickets.jsonl`:

```json
{
  "input": {
    "query": "Your customer query here",
    "customer_context": {
      "customer_tier": "standard|gold|platinum",
      "sentiment": "positive|neutral|negative|very_negative",
      "order_status": "pending|shipped|delivered|cancelled",
      "previous_interactions": 0
    }
  },
  "expected": {
    "gold_response": "Expected response guidelines",
    "should_escalate": false,
    "resolution_type": "refund|replacement|information|escalated|resolved"
  }
}
```

### Modifying Evaluation

Edit `eval/evaluator.py` to adjust:
- Resolution accuracy rubric
- Tone quality dimensions
- Escalation decision criteria
- Metric weights

### Testing the Evaluator

```bash
python use-cases/customer-support/eval/evaluator.py
```

## Tone Quality Rubric

The LLM-as-judge uses this calibrated rubric:

**Empathy (1-5)**:
- 1 = Dismissive, no acknowledgment
- 3 = Adequate, acknowledges concern
- 5 = Exceptional, truly connects with customer

**Clarity (1-5)**:
- 1 = Confusing, unclear next steps
- 3 = Adequately clear
- 5 = Crystal clear, perfectly explained

**Professionalism (1-5)**:
- 1 = Unprofessional, inappropriate
- 3 = Professional but generic
- 5 = Exemplary, perfect brand voice
