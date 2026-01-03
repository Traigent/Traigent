# Traigent Use Case Demo Plan

This document tracks the implementation of all 5 agent types from the Traigent Agent Optimization Guide.

## Overview

Based on the Traigent Agent Optimization Guide, we have implemented the following use cases:

| Use Case | Agent Type | Primary Metrics | Status |
|----------|------------|-----------------|--------|
| 1. GTM & Acquisition | Sales/Marketing Agent | Message Quality Score, Compliance Pass Rate | [x] Complete |
| 2. Operations | Workflow Automation Agent | Action Sequence Accuracy, Decision Accuracy | [x] Complete |
| 3. Knowledge & RAG | Document QA Agent | Grounded Accuracy, Abstention F1 | [x] Complete |
| 4. Product & Technical | Code Generation Agent | Weighted Test Pass, Quality Score | [x] Complete |
| 5. Customer Support | Support Bot Agent | Resolution Accuracy, Escalation Accuracy | [x] Complete |

---

## Quick Start

```bash
# Activate virtual environment
source venv/bin/activate

# Set mock mode (REQUIRED for testing without API costs)
export TRAIGENT_MOCK_LLM=true

# Run any agent evaluator to test
python use-cases/gtm-acquisition/eval/evaluator.py
python use-cases/operations/eval/evaluator.py
python use-cases/knowledge-rag/eval/evaluator.py
python use-cases/product-technical/eval/evaluator.py
python use-cases/customer-support/eval/evaluator.py
```

---

## 1. GTM & Acquisition Agent

**Folder:** `use-cases/gtm-acquisition/`

### Description
A sales outbound message generation agent that creates personalized outreach messages for leads based on ICP (Ideal Customer Profile) criteria.

### Implementation
- **Agent:** `agent/gtm_agent.py` - SDR outbound message generator
- **Dataset:** `datasets/leads_dataset.jsonl` - 218 lead profiles with gold standard messages
- **Evaluator:** `eval/evaluator.py` - LLM-as-judge for message quality + compliance checks

### Configuration Space
| Parameter | Values | Description |
|-----------|--------|-------------|
| `model` | gpt-3.5-turbo, gpt-4o-mini, gpt-4o | LLM model selection |
| `temperature` | 0.3, 0.5, 0.7, 0.9 | Response creativity |
| `personalization_depth` | basic, moderate, deep | How much to personalize |
| `tone` | professional, friendly, consultative | Message tone |

### Objectives & Metrics
1. **Message Quality Score** (LLM-as-judge)
   - ICP fit (1-5)
   - Personalization depth (1-5)
   - Value proposition clarity (1-5)
   - Appropriate tone (1-5)

2. **Compliance Pass Rate** (Deterministic + LLM)
   - Spam score check
   - No banned phrases (e.g., "act now", "limited time")
   - Professional tone verification

### Test Results
```
Evaluating good message:
  ICP Fit: 1.5/5
  Personalization: 4.0/5
  Value Proposition: 5.0/5
  Tone: 3.0/5
  Overall Quality: 3.35/5
  Compliance: PASSED

Evaluating bad message:
  Overall Quality: 1.12/5
  Compliance: FAILED (13 issues detected)
```

---

## 2. Operations Agent

**Folder:** `use-cases/operations/`

### Description
A workflow automation agent that processes task requests and generates appropriate action sequences for operational tasks.

### Implementation
- **Agent:** `agent/operations_agent.py` - Task execution planner
- **Dataset:** `datasets/tasks_dataset.jsonl` - 209 operational task scenarios
- **Evaluator:** `eval/evaluator.py` - Action sequence and escalation evaluator

### Configuration Space
| Parameter | Values | Description |
|-----------|--------|-------------|
| `model` | gpt-3.5-turbo, gpt-4o-mini, gpt-4o | LLM model selection |
| `temperature` | 0.1, 0.3, 0.5 | Lower for consistency |
| `autonomy_level` | conservative, moderate, aggressive | Escalation behavior |
| `validation_strictness` | lenient, standard, strict | Validation checks |

### Objectives & Metrics
1. **Action Sequence Accuracy** - Match rate against ground truth
2. **Escalation Accuracy** - Precision/Recall/F1 for routing decisions
3. **Execution Efficiency** - Action Economy = min_steps / agent_steps

### Test Results
```
Test 1: Exact Match - Overall: 1.00
Test 2: Partial Match - Overall: 0.90
Test 3: Wrong Escalation - Overall: 0.32
Test 4: Over-engineered - Overall: 0.74
```

---

## 3. Knowledge & RAG Agent

**Folder:** `use-cases/knowledge-rag/`

### Description
A document Q&A agent that retrieves information from a knowledge base and generates grounded answers with appropriate citations and abstention behavior.

### Implementation
- **Agent:** `agent/rag_agent.py` - RAG Q&A agent
- **Knowledge Base:** `datasets/knowledge_base/cloudstack_docs.json` - 40+ documentation chunks
- **Dataset:** `datasets/qa_dataset.jsonl` - 203 Q&A pairs (including 56 unanswerable)
- **Evaluator:** `eval/evaluator.py` - Grounding, retrieval, and abstention evaluator

### Configuration Space
| Parameter | Values | Description |
|-----------|--------|-------------|
| `model` | gpt-3.5-turbo, gpt-4o-mini, gpt-4o | LLM model selection |
| `temperature` | 0.0, 0.1, 0.3 | Low for factual accuracy |
| `top_k` | 3, 5, 7, 10 | Documents to retrieve |
| `confidence_threshold` | 0.5, 0.7, 0.85 | Abstention trigger |

### Objectives & Metrics
1. **Grounded Accuracy** - Correctness AND faithfulness to sources
2. **Retrieval Quality** - MRR, Retrieval Success Rate
3. **Abstention F1** - Balance of knowing when to abstain

### Test Results
```
Test 1: Good Answer - Overall: 0.84
Test 2: Correct Abstention - Overall: 0.87
Test 3: False Confidence - Overall: 0.42
Test 4: Wrong Sources - Overall: 0.30
```

---

## 4. Product & Technical Agent

**Folder:** `use-cases/product-technical/`

### Description
A code generation agent that writes Python functions based on specifications, optimized for correctness and code quality.

### Implementation
- **Agent:** `agent/code_agent.py` - Python code generator
- **Dataset:** `datasets/coding_tasks.jsonl` - 126 coding tasks with test cases
- **Evaluator:** `eval/evaluator.py` - Test execution + code quality analyzer

### Configuration Space
| Parameter | Values | Description |
|-----------|--------|-------------|
| `model` | gpt-3.5-turbo, gpt-4o-mini, gpt-4o | LLM model selection |
| `temperature` | 0.0, 0.2, 0.4 | Low for deterministic code |
| `coding_style` | concise, verbose, documented | Code style preference |
| `approach` | direct, test_first | Problem-solving approach |

### Objectives & Metrics
1. **Test Pass Rate** - Actual code execution against test cases
2. **Code Quality** - AST analysis, complexity, style checks
3. **Solution Efficiency** - Conciseness vs reference solution

### Test Results
```
Test 1: Correct Solution - Overall: 0.90
Test 2: Incorrect Solution - Overall: 0.79
Test 3: Syntax Error - Overall: 0.17
Test 4: Verbose Solution - Overall: 0.88
```

---

## 5. Customer Support Agent

**Folder:** `use-cases/customer-support/`

### Description
A customer support agent that handles inquiries, provides resolutions, and makes appropriate escalation decisions.

### Implementation
- **Agent:** `agent/support_agent.py` - ShopEasy support bot
- **Dataset:** `datasets/support_tickets.jsonl` - 309 support scenarios (65 escalations)
- **Evaluator:** `eval/evaluator.py` - Resolution + tone + escalation evaluator

### Configuration Space
| Parameter | Values | Description |
|-----------|--------|-------------|
| `model` | gpt-3.5-turbo, gpt-4o-mini, gpt-4o | LLM model selection |
| `temperature` | 0.3, 0.5, 0.7 | Moderate for natural responses |
| `tone` | professional, friendly, empathetic | Response tone |
| `empathy_level` | moderate, high, very_high | Empathy expression |
| `escalation_threshold` | conservative, moderate, aggressive | When to escalate |

### Objectives & Metrics
1. **Resolution Accuracy** - Correctly addresses customer issue (LLM-as-judge)
2. **Tone Quality** - Empathy, clarity, professionalism (LLM-as-judge)
3. **Escalation Accuracy** - Precision/Recall/F1 for escalation decisions

### Test Results
```
Test 1: Good Response - Overall: 0.92
Test 2: Poor Response - Overall: 0.62
Test 3: Wrong Escalation - Overall: 0.54
Test 4: Correct Escalation - Overall: 0.81
```

---

## Implementation Progress Tracker

### Phase 1: Setup
- [x] Install Traigent SDK
- [x] Create folder structure
- [x] Create planning document

### Phase 2: GTM & Acquisition Agent
- [x] Generate leads dataset (218 entries)
- [x] Implement agent with @traigent.optimize decorator
- [x] Implement LLM-as-judge evaluator
- [x] Create README documentation

### Phase 3: Operations Agent
- [x] Generate tasks dataset (209 entries)
- [x] Implement agent
- [x] Implement action sequence evaluator
- [x] Create README documentation

### Phase 4: Knowledge & RAG Agent
- [x] Generate knowledge base (40+ docs)
- [x] Generate Q&A dataset (203 entries with 56 abstention cases)
- [x] Implement RAG agent
- [x] Implement grounding + abstention evaluator
- [x] Create README documentation

### Phase 5: Product & Technical Agent
- [x] Generate coding tasks dataset (126 entries)
- [x] Implement code agent
- [x] Implement test runner + quality evaluator
- [x] Create README documentation

### Phase 6: Customer Support Agent
- [x] Generate support tickets dataset (309 entries, 65 escalations)
- [x] Implement support agent
- [x] Implement resolution + tone + escalation evaluator
- [x] Create README documentation

---

## File Structure

```
use-cases/
├── use-case-demo-plan.md              # This file
├── gtm-acquisition/
│   ├── agent/
│   │   ├── __init__.py
│   │   └── gtm_agent.py               # SDR message generator
│   ├── datasets/
│   │   ├── __init__.py
│   │   └── leads_dataset.jsonl        # 218 lead profiles
│   ├── eval/
│   │   ├── __init__.py
│   │   └── evaluator.py               # Message quality evaluator
│   └── README.md
├── operations/
│   ├── agent/
│   │   ├── __init__.py
│   │   └── operations_agent.py        # Workflow automation agent
│   ├── datasets/
│   │   ├── __init__.py
│   │   └── tasks_dataset.jsonl        # 209 task scenarios
│   ├── eval/
│   │   ├── __init__.py
│   │   └── evaluator.py               # Action sequence evaluator
│   └── README.md
├── knowledge-rag/
│   ├── agent/
│   │   ├── __init__.py
│   │   └── rag_agent.py               # RAG Q&A agent
│   ├── datasets/
│   │   ├── __init__.py
│   │   ├── knowledge_base/
│   │   │   └── cloudstack_docs.json   # 40+ doc chunks
│   │   └── qa_dataset.jsonl           # 203 Q&A pairs
│   ├── eval/
│   │   ├── __init__.py
│   │   └── evaluator.py               # Grounding evaluator
│   ├── scripts/
│   │   ├── __init__.py
│   │   └── generate_knowledge_base.py
│   └── README.md
├── product-technical/
│   ├── agent/
│   │   ├── __init__.py
│   │   └── code_agent.py              # Code generation agent
│   ├── datasets/
│   │   ├── __init__.py
│   │   └── coding_tasks.jsonl         # 126 coding tasks
│   ├── eval/
│   │   ├── __init__.py
│   │   └── evaluator.py               # Test runner + quality
│   └── README.md
└── customer-support/
    ├── agent/
    │   ├── __init__.py
    │   └── support_agent.py           # Support bot agent
    ├── datasets/
    │   ├── __init__.py
    │   └── support_tickets.jsonl      # 309 support scenarios
    ├── eval/
    │   ├── __init__.py
    │   └── evaluator.py               # Resolution evaluator
    └── README.md
```

---

## Key Features Demonstrated

### 1. LLM-as-Judge Evaluation
- Calibrated rubrics with 1-5 scoring
- Multi-dimensional quality assessment
- Fallback heuristics for mock mode

### 2. Deterministic Evaluation
- Code execution with test cases (Product/Technical)
- Action sequence matching (Operations)
- Compliance checks with regex (GTM)

### 3. Classification Metrics
- Escalation decisions with precision/recall/F1
- Abstention F1 for RAG agents
- Policy compliance pass rates

### 4. Multi-Objective Optimization
- Quality vs Cost tradeoffs
- Efficiency vs Accuracy balance
- Pareto-optimal configuration search

### 5. Mock Mode Support
All agents work in mock mode without API keys:
- Uses heuristic evaluators
- Generates realistic mock responses
- Enables testing without incurring costs

---

## Running the Use Cases

```bash
# Set up environment
source venv/bin/activate
export TRAIGENT_MOCK_LLM=true

# Test each evaluator (no API keys needed)
python use-cases/gtm-acquisition/eval/evaluator.py
python use-cases/operations/eval/evaluator.py
python use-cases/knowledge-rag/eval/evaluator.py
python use-cases/product-technical/eval/evaluator.py
python use-cases/customer-support/eval/evaluator.py

# Run full agent optimization (requires API key without mock mode)
# python use-cases/gtm-acquisition/agent/gtm_agent.py
```

## Notes

- All datasets are synthetic but realistic
- LLM-as-judge implementations follow calibrated rubrics from the guide
- Each use case demonstrates a different aspect of Traigent's optimization capabilities
- Mock mode is used by default to avoid API costs during development
- All implementations include proper fallbacks for testing without API access
