# Operations Agent Use Case

> **Ops workflows that match ground truth - hands-off**

<p align="center">
  <a href="demo/demo.cast">
    <img src="demo/demo.svg" alt="Operations Demo" width="600">
  </a>
</p>

This use case demonstrates optimizing a **workflow automation agent** that processes operational tasks and generates appropriate action sequences.

## Overview

The Operations agent processes various task types (expense approvals, access requests, incident responses, etc.) and determines:

1. **Action Sequence** - The ordered list of steps to execute
2. **Escalation Decision** - Whether human intervention is needed

It optimizes for:
- **Action Sequence Accuracy** - Matching expected action sequences
- **Escalation Accuracy** - Correct escalate/don't escalate decisions
- **Execution Efficiency** - Minimizing unnecessary steps

## Quick Start

```bash
# From project root
cd /path/to/Traigent

# Enable mock mode (recommended for testing)
export TRAIGENT_MOCK_MODE=true

# Run the agent optimization
python use-cases/operations/agent/operations_agent.py
```

## Configuration Space

| Parameter | Values | Description |
|-----------|--------|-------------|
| `model` | gpt-3.5-turbo, gpt-4o-mini, gpt-4o | LLM model selection |
| `temperature` | 0.1, 0.3, 0.5 | Lower for consistent workflows |
| `autonomy_level` | conservative, moderate, aggressive | When to auto-approve vs escalate |
| `validation_strictness` | lenient, standard, strict | How thorough to validate |

## Dataset

The evaluation dataset (`datasets/tasks_dataset.jsonl`) contains 70+ operational task scenarios including:

- Expense approvals
- Purchase requests
- Access requests
- Time-off requests
- Vendor onboarding
- Incident response
- Employee onboarding
- Contract renewals
- Refund requests
- Security alerts
- And many more...

### Sample Entry

```json
{
  "input": {
    "task_type": "expense_approval",
    "description": "Process expense report #EXP-12345 for $2,500 from Marketing",
    "context": {
      "employee_level": "manager",
      "budget_remaining": "$5,000",
      "policy_limit": "$3,000",
      "department": "Marketing"
    }
  },
  "expected_actions": [
    "validate_amount",
    "check_policy_limit",
    "check_budget",
    "auto_approve",
    "notify_finance",
    "update_budget"
  ],
  "should_escalate": false
}
```

## Evaluation Metrics

### Action Sequence Accuracy

Measures how well the generated actions match expected sequences:
- Exact match bonus
- Jaccard similarity (set overlap)
- Category-based matching (validation, approval, notification)
- Order-aware scoring (validation before approval)

### Escalation Accuracy

Binary classification metrics:
- Correct escalation decisions
- Measures precision and recall for escalation class

### Execution Efficiency

Action Economy = `min_required_steps / agent_actual_steps`

Penalizes over-engineered solutions with unnecessary steps.

## Files

```
operations/
├── agent/
│   └── operations_agent.py   # Main agent with @traigent.optimize
├── datasets/
│   └── tasks_dataset.jsonl   # 70+ operational task scenarios
├── eval/
│   └── evaluator.py          # Action sequence and escalation evaluator
└── README.md
```

## Expected Results

After optimization, you should see results like:

```
Best Configuration:
  model: gpt-4o-mini
  temperature: 0.3
  autonomy_level: moderate
  validation_strictness: standard

Best Score: 0.78
```

## Task Types Covered

| Task Type | Description | Escalation Triggers |
|-----------|-------------|---------------------|
| `expense_approval` | Process expense reports | Over policy limit |
| `purchase_request` | Handle purchase requests | Over threshold |
| `access_request` | Grant system access | High sensitivity |
| `time_off_request` | Process PTO requests | Blackout periods |
| `vendor_onboarding` | Onboard new vendors | Compliance requirements |
| `incident_response` | Handle incidents | P1/P2 severity |
| `employee_onboarding` | Onboard new hires | C-level executives |
| `contract_renewal` | Manage renewals | Large contracts |
| `refund_request` | Process refunds | Policy exceptions |
| `security_alert` | Handle security events | Admin accounts, attacks |

## Customization

### Adding More Tasks

Add entries to `datasets/tasks_dataset.jsonl` following the JSON format above.

### Adjusting Evaluation

Edit `eval/evaluator.py` to modify:
- Metric weights
- Action category definitions
- Order rules for validation

### Testing the Evaluator

```bash
python use-cases/operations/eval/evaluator.py
```

This runs the evaluator on sample scenarios to demonstrate scoring.
