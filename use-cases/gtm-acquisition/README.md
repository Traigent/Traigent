# GTM & Acquisition Agent Use Case

> **Cold outreach → high-quality emails, auto-optimized**

<p align="center">
  <a href="demo/demo.cast">
    <img src="demo/demo.svg" alt="GTM Demo" width="600">
  </a>
</p>

This use case demonstrates optimizing a **Sales Development Representative (SDR) outbound message generator** using TraiGent.

## Overview

The GTM agent generates personalized outbound sales messages based on lead profiles. It optimizes for:

1. **Message Quality Score** - ICP fit, personalization depth, value proposition clarity, tone
2. **Compliance Pass Rate** - Spam score, banned phrases, professional tone

## Quick Start

```bash
# From project root
cd /path/to/Traigent

# Enable mock mode (recommended for testing)
export TRAIGENT_MOCK_MODE=true

# Run the agent optimization
python use-cases/gtm-acquisition/agent/gtm_agent.py
```

## Configuration Space

| Parameter | Values | Description |
|-----------|--------|-------------|
| `model` | gpt-3.5-turbo, gpt-4o-mini, gpt-4o | LLM model selection |
| `temperature` | 0.3, 0.5, 0.7, 0.9 | Creativity vs consistency |
| `personalization_depth` | basic, moderate, deep | How much research to incorporate |
| `tone` | professional, friendly, consultative | Message style |

## Dataset

The evaluation dataset (`datasets/leads_dataset.jsonl`) contains 218 lead profiles with:

- Lead information (name, title, company, industry, etc.)
- Recent company news for personalization
- Pain points relevant to the product
- Gold standard messages for comparison

### Sample Entry

```json
{
  "input": {
    "lead": {
      "name": "Sarah Chen",
      "title": "VP of Engineering",
      "company": "TechCorp Inc",
      "industry": "SaaS",
      "company_size": "200-500",
      "recent_news": "Just raised Series B funding of $25M",
      "pain_points": ["scaling infrastructure", "engineering hiring"]
    },
    "product": "AI-powered DevOps platform",
    "sender_name": "Alex from DevOpsAI"
  },
  "output": "Hi Sarah,\n\nCongratulations on TechCorp's Series B!..."
}
```

## Evaluation Metrics

### Message Quality (LLM-as-judge)

The evaluator scores messages on a 1-5 scale across dimensions:

| Dimension | Weight | Description |
|-----------|--------|-------------|
| ICP Fit | 30% | How well the message addresses industry/role |
| Personalization | 30% | Use of specific lead details |
| Value Proposition | 25% | Clarity of product benefits |
| Tone | 15% | Appropriate professional tone |

### Compliance

Deterministic checks for:
- Banned phrases (spam triggers)
- Excessive punctuation
- Message length bounds
- Professional formatting

## Files

```
gtm-acquisition/
├── agent/
│   └── gtm_agent.py      # Main agent with @traigent.optimize decorator
├── datasets/
│   └── leads_dataset.jsonl   # 218 lead profiles
├── eval/
│   └── evaluator.py      # LLM-as-judge implementation
└── README.md
```

## Expected Results

After optimization, you should see results like:

```
Best Configuration:
  model: gpt-4o-mini
  temperature: 0.5
  personalization_depth: moderate
  tone: friendly

Best Score: 0.85
```

## Customization

### Adding More Leads

Add entries to `datasets/leads_dataset.jsonl` following the JSON format above.

### Modifying Evaluation

Edit `eval/evaluator.py` to adjust:
- Quality dimension weights
- Banned phrase list
- Scoring rubrics

### Testing the Evaluator

```bash
python use-cases/gtm-acquisition/eval/evaluator.py
```

This runs the evaluator on sample messages to demonstrate scoring.
