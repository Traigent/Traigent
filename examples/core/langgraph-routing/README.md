# LangGraph Document Routing Demo

This example demonstrates **multi-agent optimization** with Traigent using a document processing pipeline that routes between Legal and Financial analysis branches.

## Overview

The demo implements a document processing system with three agents:

1. **Router Agent**: Classifies incoming documents as "legal" or "financial"
2. **Legal Analyzer**: Processes legal documents (contracts, NDAs, agreements)
3. **Financial Analyzer**: Processes financial documents (invoices, statements, reports)

```
Document Input
      │
      ▼
┌─────────────┐
│   Router    │ ─── Classifies document type
└─────────────┘
      │
      ├──── legal ────┐      ├──── financial ────┐
      ▼               │      ▼                   │
┌─────────────┐       │ ┌─────────────┐          │
│   Legal     │       │ │  Financial  │          │
│  Analyzer   │       │ │  Analyzer   │          │
└─────────────┘       │ └─────────────┘          │
      │               │      │                   │
      └───────────────┴──────┴───────────────────┘
                      │
                      ▼
              Structured Output
```

## What's Being Optimized

Traigent optimizes **each agent independently**:

| Agent | Parameters |
|-------|------------|
| Router | `router_model`, `router_temperature`, `router_prompt_style` |
| Legal Agent | `legal_agent_model`, `legal_agent_temperature`, `legal_agent_prompt_template` |
| Financial Agent | `financial_agent_model`, `financial_agent_temperature`, `financial_agent_prompt_template` |

### Configuration Space

```python
{
    # Router agent
    "router_model": ["gpt-4o-mini", "gpt-4o"],
    "router_temperature": [0.0, 0.1],
    "router_prompt_style": ["concise", "detailed", "few_shot"],

    # Legal agent
    "legal_agent_model": ["gpt-4o", "gpt-4o-mini"],
    "legal_agent_temperature": [0.1, 0.3, 0.5],
    "legal_agent_prompt_template": ["standard", "structured_extraction", "risk_focused"],

    # Financial agent
    "financial_agent_model": ["gpt-4o", "gpt-4o-mini"],
    "financial_agent_temperature": [0.1, 0.3, 0.5],
    "financial_agent_prompt_template": ["standard", "detailed_breakdown", "metric_extraction"],
}
```

## Objectives

The demo optimizes for two objectives:

1. **Routing Accuracy**: How often the router correctly classifies documents
2. **Processing Quality**: How well the analysis covers expected document elements

## Quick Start

### Mock Mode (No API Costs)

```bash
cd examples/core/langgraph-routing
TRAIGENT_MOCK_LLM=true python run.py
```

### Real LLM Mode

```bash
export OPENAI_API_KEY=sk-...
python run.py
```

## Files

```
examples/core/langgraph-routing/
├── run.py              # Main entry point with Traigent optimization
├── evaluators.py       # Custom metric functions
├── prompts/
│   ├── __init__.py
│   ├── router_prompts.py    # Classification prompts
│   ├── legal_prompts.py     # Legal analysis prompts
│   └── financial_prompts.py # Financial analysis prompts
└── README.md

examples/datasets/langgraph-routing/
└── evaluation_set.jsonl     # 22 sample documents (10 legal, 12 financial)
```

## Evaluation Dataset

The dataset includes 22 realistic document samples (10 legal, 12 financial):

**Legal Documents (10):**
- Service agreements
- Non-disclosure agreements (NDAs)
- Software license agreements
- Employment agreements
- Termination notices
- Lease amendments
- Settlement agreements
- Powers of attorney
- Partnership agreements
- Indemnification agreements

**Financial Documents (12):**
- Invoices
- Quarterly financial reports
- Bank statements
- Purchase orders
- Expense reports
- Budget proposals
- Accounts receivable aging reports
- Credit memos
- Profit & loss statements
- Wire transfer confirmations
- Tax documents (1099-NEC)
- Cash flow statements

## Prompt Templates

### Router Prompts

| Style | Description |
|-------|-------------|
| `concise` | Minimal classification prompt |
| `detailed` | Full category descriptions |
| `few_shot` | Includes 2-3 classification examples |
| `chain_of_thought` | Step-by-step reasoning approach |

### Legal Prompts

| Template | Focus |
|----------|-------|
| `standard` | Comprehensive document summary |
| `structured_extraction` | Extract structured elements |
| `risk_focused` | Risk analysis perspective |
| `compliance_oriented` | Compliance review |

### Financial Prompts

| Template | Focus |
|----------|-------|
| `standard` | General financial summary |
| `detailed_breakdown` | Itemized financial analysis |
| `summary_focused` | Executive summary |
| `metric_extraction` | Extract specific metrics |

## Custom Evaluators

### `routing_accuracy_scorer`

Scores routing correctness:
- Returns `1.0` if document routed to correct type
- Returns `0.0` if incorrectly routed

### `processing_quality_scorer`

Scores analysis quality based on expected element coverage:
- Checks if analysis mentions expected elements
- Applies fuzzy matching with element variations
- Penalizes very short responses

## Example Output

```
======================================================================
OPTIMIZATION RESULTS
======================================================================

Best Score: 0.8542
Total Trials: 20

Best Configuration by Agent:
--------------------------------------------------
ROUTER:
  Model: gpt-4o-mini
  Temperature: 0.0
  Prompt Style: few_shot

LEGAL ANALYZER:
  Model: gpt-4o
  Temperature: 0.3
  Template: structured_extraction

FINANCIAL ANALYZER:
  Model: gpt-4o-mini
  Temperature: 0.1
  Template: detailed_breakdown
```

## Extending This Example

### Add New Document Types

1. Add samples to `evaluation_set.jsonl`
2. Update router prompts to include new category
3. Create new analyzer prompts
4. Add to configuration space

### Add New Metrics

Edit `evaluators.py` to add custom scorers:

```python
def my_custom_metric(output, expected, input_data=None, config=None, llm_metrics=None):
    # Your scoring logic
    return score  # 0.0 to 1.0
```

Then add to the decorator:

```python
@traigent.optimize(
    metric_functions={
        "routing_accuracy": routing_accuracy_scorer,
        "my_metric": my_custom_metric,
    },
    objectives=["routing_accuracy", "my_metric"],
)
```

## Related Examples

- `examples/core/structured-output-json/` - JSON extraction optimization
- `examples/quickstart/02_customer_support_rag.py` - RAG optimization
- `examples/advanced/ai-engineering-tasks/` - Advanced patterns
