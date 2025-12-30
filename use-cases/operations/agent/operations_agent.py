#!/usr/bin/env python3
"""
Operations Agent - Workflow Automation Task Processor

This agent processes operational task requests and generates appropriate
action sequences for workflow automation. It optimizes for action accuracy
and correct escalation decisions.

Usage:
    export TRAIGENT_MOCK_MODE=true
    python use-cases/operations/agent/operations_agent.py
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Import evaluator from sibling directory
import importlib.util

import traigent
from traigent.api.decorators import EvaluationOptions, ExecutionOptions

_evaluator_path = Path(__file__).parent.parent / "eval" / "evaluator.py"
_spec = importlib.util.spec_from_file_location("operations_evaluator", _evaluator_path)
_evaluator_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_evaluator_module)
OperationsEvaluator = _evaluator_module.OperationsEvaluator


def format_task_context(
    task_type: str, description: str, context: dict[str, Any]
) -> str:
    """Format task information for the prompt."""
    context_str = "\n".join(f"  - {k}: {v}" for k, v in context.items())
    return f"""
Task Type: {task_type}
Description: {description}
Context:
{context_str}
"""


TASK_PROCESSING_PROMPT = """You are an operations workflow automation agent. Your job is to analyze task requests and determine the correct sequence of actions to take.

{task_context}

Autonomy Level: {autonomy_level}
Validation Strictness: {validation_strictness}

Based on the task and context, determine:
1. The sequence of actions to execute (in order)
2. Whether this task should be escalated to a human

Action Guidelines:
- Start with validation steps (validate_*, verify_*, check_*)
- Include appropriate approval/escalation steps based on thresholds
- End with notification and logging steps
- Consider the autonomy level when deciding to escalate

Common Actions Available:
- validate_amount, validate_request, validate_dates
- check_policy_limit, check_budget, check_threshold, check_balance
- verify_manager_approval, verify_identity, verify_authorization
- auto_approve, escalate_to_manager, escalate_to_director
- notify_*, send_*, log_*, update_*, create_*
- flag_*, alert_*, block_*

Output Format:
Return a JSON object with:
- "actions": list of action strings in execution order
- "should_escalate": boolean indicating if human escalation is needed
- "escalation_reason": string explaining why (if escalating)

Respond with only the JSON object:"""


@traigent.optimize(
    configuration_space={
        "model": ["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o"],
        "temperature": [0.1, 0.3, 0.5],
        "autonomy_level": ["conservative", "moderate", "aggressive"],
        "validation_strictness": ["lenient", "standard", "strict"],
    },
    objectives=["action_accuracy", "escalation_accuracy", "efficiency", "cost"],
    evaluation=EvaluationOptions(
        eval_dataset="use-cases/operations/datasets/tasks_dataset.jsonl",
        # OperationsEvaluator has scoring_function interface: (prediction, expected, input_data) -> dict
        scoring_function=OperationsEvaluator(),
    ),
    execution=ExecutionOptions(execution_mode="edge_analytics"),
)
def operations_workflow_agent(
    task_type: str,
    description: str,
    context: dict[str, Any],
) -> dict[str, Any]:
    """
    Process an operational task and generate action sequence.

    Args:
        task_type: Type of task (expense_approval, access_request, etc.)
        description: Human-readable description of the task
        context: Dictionary of contextual information

    Returns:
        Dictionary with 'actions', 'should_escalate', and optional 'escalation_reason'
    """
    # Get current configuration
    config = traigent.get_config()

    # Extract tuned variables with defaults
    model = config.get("model", "gpt-4o-mini")
    temperature = config.get("temperature", 0.3)
    autonomy_level = config.get("autonomy_level", "moderate")
    validation_strictness = config.get("validation_strictness", "standard")

    # Format the task context
    task_context = format_task_context(task_type, description, context)

    # Build the prompt
    prompt = TASK_PROCESSING_PROMPT.format(
        task_context=task_context,
        autonomy_level=autonomy_level,
        validation_strictness=validation_strictness,
    )

    # Use LangChain for LLM call
    try:
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(
            model=model,
            temperature=temperature,
        )
        response = llm.invoke(prompt)

        # Parse JSON response
        try:
            result = json.loads(response.content)
            return result
        except json.JSONDecodeError:
            # Fallback if response isn't valid JSON
            return generate_rule_based_response(task_type, context, autonomy_level)
    except ImportError:
        # Fallback for mock mode without LangChain
        return generate_rule_based_response(task_type, context, autonomy_level)


def generate_rule_based_response(
    task_type: str,
    context: dict[str, Any],
    autonomy_level: str,
) -> dict[str, Any]:
    """
    Generate a rule-based response when LLM is not available.
    This provides realistic mock responses for testing.
    """
    # Define escalation thresholds based on autonomy level
    thresholds = {
        "conservative": {"amount": 1000, "sensitivity": ["high", "medium"]},
        "moderate": {"amount": 5000, "sensitivity": ["high"]},
        "aggressive": {"amount": 10000, "sensitivity": ["high"]},
    }

    threshold = thresholds.get(autonomy_level, thresholds["moderate"])

    # Base actions by task type
    base_actions = {
        "expense_approval": ["validate_amount", "check_policy_limit", "check_budget"],
        "purchase_request": ["validate_request", "check_threshold", "check_budget"],
        "access_request": [
            "validate_request",
            "check_role_permissions",
            "verify_manager_approval",
        ],
        "time_off_request": ["validate_dates", "check_balance", "check_coverage"],
        "vendor_onboarding": ["create_vendor_profile", "initiate_risk_assessment"],
        "incident_response": ["create_incident_ticket", "notify_stakeholders"],
        "employee_onboarding": [
            "create_employee_record",
            "setup_email",
            "provision_equipment",
        ],
        "contract_renewal": ["flag_renewal_due", "pull_usage_report"],
        "refund_request": ["verify_order", "check_refund_eligibility"],
        "data_request": ["validate_request", "verify_authorization"],
    }

    actions = base_actions.get(task_type, ["validate_request", "check_policy"])

    # Determine if escalation is needed
    should_escalate = False
    escalation_reason = None

    # Check various escalation triggers
    if context.get("sensitivity_level") in threshold["sensitivity"]:
        should_escalate = True
        escalation_reason = f"Sensitivity level {context.get('sensitivity_level')} requires human review"

    # Check amount thresholds
    for amount_field in ["amount", "contract_value", "order_value"]:
        if amount_field in context:
            amount_str = str(context[amount_field]).replace("$", "").replace(",", "")
            try:
                amount = float(amount_str)
                if amount > threshold["amount"]:
                    should_escalate = True
                    escalation_reason = (
                        f"Amount ${amount} exceeds threshold ${threshold['amount']}"
                    )
            except ValueError:
                pass

    # Check for explicit escalation indicators
    if context.get("risk_assessment") in ["high", "medium"]:
        should_escalate = True
        escalation_reason = "Risk assessment requires human review"

    if (
        context.get("compliance_required")
        and len(context.get("compliance_required", [])) > 0
    ):
        should_escalate = True
        escalation_reason = "Compliance requirements need verification"

    if context.get("severity") in ["P1", "P2"]:
        should_escalate = True
        escalation_reason = (
            f"Severity {context.get('severity')} requires immediate attention"
        )

    # Add appropriate ending actions
    if should_escalate:
        actions.append("escalate_to_manager")
    else:
        actions.append("auto_approve")
        actions.append("notify_stakeholders")
        actions.append("log_completion")

    return {
        "actions": actions,
        "should_escalate": should_escalate,
        "escalation_reason": escalation_reason,
    }


async def run_optimization():
    """Run the Operations agent optimization."""
    print("=" * 60)
    print("Operations Agent - Traigent Optimization")
    print("=" * 60)

    # Check if mock mode is enabled
    mock_mode = os.environ.get("TRAIGENT_MOCK_MODE", "false").lower() == "true"
    print(f"\nMock Mode: {'Enabled' if mock_mode else 'Disabled'}")

    if not mock_mode:
        print("\nWARNING: Running without mock mode will incur API costs!")
        print("Set TRAIGENT_MOCK_MODE=true for testing.\n")

    print("\nStarting optimization...")
    print("Configuration Space:")
    print("  - Models: gpt-3.5-turbo, gpt-4o-mini, gpt-4o")
    print("  - Temperature: 0.1, 0.3, 0.5")
    print("  - Autonomy Level: conservative, moderate, aggressive")
    print("  - Validation Strictness: lenient, standard, strict")
    print("\nObjectives: action_accuracy, escalation_accuracy, efficiency, cost")
    print("-" * 60)

    # Run optimization
    results = await operations_workflow_agent.optimize(
        algorithm="random",
        max_trials=20,
    )

    # Display results
    print("\n" + "=" * 60)
    print("OPTIMIZATION RESULTS")
    print("=" * 60)
    print("\nBest Configuration:")
    for key, value in results.best_config.items():
        print(f"  {key}: {value}")
    print(f"\nBest Score: {results.best_score:.4f}")

    # Apply best config
    operations_workflow_agent.apply_best_config(results)
    print("\nBest configuration applied!")

    # Test with a sample task
    print("\n" + "-" * 60)
    print("Testing optimized agent with sample task...")
    print("-" * 60)

    sample_result = operations_workflow_agent(
        task_type="expense_approval",
        description="Process expense report #TEST-001 for $2,500 from Engineering",
        context={
            "employee_level": "senior_engineer",
            "budget_remaining": "$10,000",
            "policy_limit": "$3,000",
            "department": "Engineering",
            "expense_category": "equipment",
        },
    )

    print("\nGenerated Response:")
    print(f"  Actions: {sample_result['actions']}")
    print(f"  Should Escalate: {sample_result['should_escalate']}")
    if sample_result.get("escalation_reason"):
        print(f"  Reason: {sample_result['escalation_reason']}")

    return results


def main():
    """Main entry point."""
    asyncio.run(run_optimization())


if __name__ == "__main__":
    main()
