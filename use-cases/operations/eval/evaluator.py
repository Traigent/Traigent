#!/usr/bin/env python3
"""
Evaluator for Operations Agent

This evaluator scores workflow automation responses on:
1. Action Sequence Accuracy - How well the generated actions match expected
2. Escalation Accuracy - Precision/Recall of escalation decisions
3. Execution Efficiency - Action economy (min steps / actual steps)

Supports two modes:
- MOCK MODE (default): Uses heuristic rules for fast, free evaluation
- REAL MODE: Uses actual LLM calls for agent and LLM-as-judge evaluation

Usage:
  Mock mode: python evaluator.py  (default, uses heuristics)
  Real mode: TRAIGENT_MOCK_MODE=false python evaluator.py  (requires OPENAI_API_KEY)
"""

import json
import os
from dataclasses import dataclass
from typing import Any

# ============================================================================
# PROMPTS FOR REAL LLM MODE
# ============================================================================

# Prompt for the Operations agent to process tasks
AGENT_PROMPT = """You are an operations automation agent processing workflow requests.

TASK REQUEST:
{task}

CONTEXT:
- Requester: {requester}
- Department: {department}
- Amount (if applicable): {amount}
- Priority: {priority}

Available actions:
- validate_amount, validate_requester, verify_budget, check_policy
- auto_approve, approve_with_conditions, grant_access
- escalate_to_manager, escalate_to_finance, flag_for_review
- notify_requester, notify_finance, send_confirmation
- log_decision, create_ticket, update_records

Decide:
1. What sequence of actions to take
2. Whether to escalate to a human (for unusual amounts, policy violations, etc.)

Respond in EXACTLY this JSON format:
{{"actions": ["action1", "action2", ...], "should_escalate": true/false, "reason": "brief explanation"}}
"""

# Prompt for LLM-as-judge to evaluate action sequences
JUDGE_PROMPT = """You are evaluating an operations agent's decision.

TASK: {task}
CONTEXT: Requester={requester}, Amount={amount}, Priority={priority}

AGENT'S RESPONSE:
Actions: {predicted_actions}
Escalated: {predicted_escalate}

EXPECTED RESPONSE:
Actions: {expected_actions}
Should Escalate: {expected_escalate}

Evaluate:
1. ACTION_MATCH: How well do the actions match? (0.0-1.0)
2. ESCALATION_CORRECT: Was the escalation decision correct? (0 or 1)
3. ORDER_CORRECT: Are actions in logical order (validate before approve)? (0.0-1.0)

Respond in EXACTLY this format:
ACTION_MATCH: [0.0-1.0]
ESCALATION_CORRECT: [0 or 1]
ORDER_CORRECT: [0.0-1.0]
"""


@dataclass
class OperationsEvaluationResult:
    """Result of operations evaluation."""

    action_accuracy: float  # 0-1 scale
    step_level_accuracy: float  # 0-1 scale
    escalation_correct: bool
    escalation_precision: float  # 0-1
    escalation_recall: float  # 0-1
    action_economy: float  # min_steps / actual_steps
    overall_score: float


class OperationsEvaluator:
    """
    Evaluator for operations workflow automation agent.

    Evaluates action sequences, escalation decisions, and efficiency.
    """

    def __init__(self):
        """Initialize the evaluator."""
        self.weights = {
            "action_accuracy": 0.4,
            "escalation_accuracy": 0.4,
            "efficiency": 0.2,
        }

        # Action categories for partial matching
        self.action_categories = {
            "validation": ["validate_", "verify_", "check_"],
            "approval": ["approve", "auto_approve", "grant_"],
            "escalation": ["escalate_", "flag_", "alert_"],
            "notification": ["notify_", "send_", "email_"],
            "logging": ["log_", "record_", "archive_", "update_"],
            "creation": ["create_", "setup_", "provision_", "generate_"],
        }

    def __call__(
        self,
        prediction: dict[str, Any] | str,
        expected: dict[str, Any] | str | None,
        input_data: dict[str, Any],
    ) -> dict[str, float]:
        """
        Evaluate a generated response.

        Args:
            prediction: The generated response (dict with 'actions', 'should_escalate')
            expected: Expected output (from dataset)
            input_data: The input data

        Returns:
            Dictionary of metric scores
        """
        # Parse prediction if string
        if isinstance(prediction, str):
            try:
                import json

                prediction = json.loads(prediction)
            except (json.JSONDecodeError, TypeError):
                prediction = {"actions": [], "should_escalate": False}

        # Get expected values
        expected_actions = input_data.get("expected_actions", [])
        expected_escalate = input_data.get("should_escalate", False)

        # Get predicted values
        predicted_actions = prediction.get("actions", [])
        predicted_escalate = prediction.get("should_escalate", False)

        # Evaluate action accuracy
        action_accuracy = self._evaluate_action_sequence(
            predicted_actions, expected_actions
        )

        # Evaluate step-level accuracy
        step_accuracy = self._evaluate_step_level(predicted_actions, expected_actions)

        # Evaluate escalation decision
        escalation_correct = predicted_escalate == expected_escalate

        # Calculate action economy
        if predicted_actions and expected_actions:
            action_economy = min(
                len(expected_actions) / max(len(predicted_actions), 1), 1.0
            )
        else:
            action_economy = (
                1.0 if not predicted_actions and not expected_actions else 0.0
            )

        # Calculate overall score
        overall = (
            self.weights["action_accuracy"] * action_accuracy
            + self.weights["escalation_accuracy"] * (1.0 if escalation_correct else 0.0)
            + self.weights["efficiency"] * action_economy
        )

        return {
            "action_accuracy": action_accuracy,
            "step_accuracy": step_accuracy,
            "escalation_accuracy": 1.0 if escalation_correct else 0.0,
            "efficiency": action_economy,
            "overall": overall,
        }

    def _evaluate_action_sequence(
        self,
        predicted: list[str],
        expected: list[str],
    ) -> float:
        """
        Evaluate action sequence accuracy.

        Uses a combination of:
        - Exact match bonus
        - Jaccard similarity for action sets
        - Order-aware partial matching

        Returns:
            Score between 0 and 1
        """
        if not expected:
            return 1.0 if not predicted else 0.5

        if not predicted:
            return 0.0

        # Normalize actions
        predicted_normalized = [self._normalize_action(a) for a in predicted]
        expected_normalized = [self._normalize_action(a) for a in expected]

        # Exact match check
        if predicted_normalized == expected_normalized:
            return 1.0

        # Jaccard similarity (set overlap)
        predicted_set = set(predicted_normalized)
        expected_set = set(expected_normalized)

        intersection = len(predicted_set & expected_set)
        union = len(predicted_set | expected_set)

        jaccard = intersection / union if union > 0 else 0.0

        # Category-based matching (softer matching)
        category_score = self._evaluate_by_categories(predicted, expected)

        # Order-aware matching (for sequential dependencies)
        order_score = self._evaluate_order(predicted_normalized, expected_normalized)

        # Weighted combination
        return 0.4 * jaccard + 0.3 * category_score + 0.3 * order_score

    def _normalize_action(self, action: str) -> str:
        """Normalize an action string for comparison."""
        return action.lower().strip().replace("-", "_")

    def _evaluate_by_categories(
        self,
        predicted: list[str],
        expected: list[str],
    ) -> float:
        """
        Evaluate whether the right categories of actions are present.

        Returns:
            Score between 0 and 1
        """

        def get_categories(actions: list[str]) -> set[str]:
            categories = set()
            for action in actions:
                action_lower = action.lower()
                for category, prefixes in self.action_categories.items():
                    if any(action_lower.startswith(p) for p in prefixes):
                        categories.add(category)
                        break
            return categories

        predicted_cats = get_categories(predicted)
        expected_cats = get_categories(expected)

        if not expected_cats:
            return 1.0 if not predicted_cats else 0.5

        # Calculate category overlap
        intersection = len(predicted_cats & expected_cats)
        union = len(predicted_cats | expected_cats)

        return intersection / union if union > 0 else 0.0

    def _evaluate_order(
        self,
        predicted: list[str],
        expected: list[str],
    ) -> float:
        """
        Evaluate whether actions are in the correct relative order.

        Checks that validation comes before approval, etc.

        Returns:
            Score between 0 and 1
        """
        # Define ordering rules (earlier categories should come before later)
        order_rules = [
            (["validate_", "verify_", "check_"], ["approve", "grant_", "process_"]),
            (["validate_", "verify_"], ["escalate_", "flag_"]),
            (["approve", "process_"], ["notify_", "send_", "log_"]),
        ]

        if len(predicted) < 2:
            return 1.0

        violations = 0
        total_rules = 0

        for early_prefixes, late_prefixes in order_rules:
            # Find positions of early and late actions
            early_positions = []
            late_positions = []

            for i, action in enumerate(predicted):
                action_lower = action.lower()
                if any(action_lower.startswith(p) for p in early_prefixes):
                    early_positions.append(i)
                if any(action_lower.startswith(p) for p in late_prefixes):
                    late_positions.append(i)

            # Check if any late action comes before any early action
            if early_positions and late_positions:
                total_rules += 1
                min_early = min(early_positions)
                min_late = min(late_positions)
                if min_late < min_early:
                    violations += 1

        if total_rules == 0:
            return 1.0

        return 1.0 - (violations / total_rules)

    def _evaluate_step_level(
        self,
        predicted: list[str],
        expected: list[str],
    ) -> float:
        """
        Evaluate step-level accuracy.

        Counts how many expected steps are present in predicted.

        Returns:
            Score between 0 and 1
        """
        if not expected:
            return 1.0

        predicted_normalized = {self._normalize_action(a) for a in predicted}
        expected_normalized = [self._normalize_action(a) for a in expected]

        matches = sum(1 for e in expected_normalized if e in predicted_normalized)
        return matches / len(expected_normalized)


def is_mock_mode() -> bool:
    """Check if running in mock mode."""
    return os.environ.get("TRAIGENT_MOCK_MODE", "true").lower() == "true"


def process_task_with_llm(input_data: dict) -> dict:
    """Process a task using LLM (real mode only)."""
    try:
        from openai import OpenAI

        client = OpenAI()
        prompt = AGENT_PROMPT.format(
            task=input_data.get("task", "Unknown task"),
            requester=input_data.get("requester", "Unknown"),
            department=input_data.get("department", "Unknown"),
            amount=input_data.get("amount", "N/A"),
            priority=input_data.get("priority", "normal"),
        )

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )

        # Parse JSON response
        content = response.choices[0].message.content
        # Extract JSON from response
        import re

        json_match = re.search(r"\{.*\}", content, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        return {"actions": [], "should_escalate": False, "reason": "Parse error"}
    except Exception as e:
        return {"actions": [], "should_escalate": False, "reason": f"Error: {e}"}


def load_dataset() -> list[dict]:
    """Load the tasks dataset."""
    from pathlib import Path

    dataset_path = Path(__file__).parent.parent / "datasets" / "tasks_dataset.jsonl"
    if not dataset_path.exists():
        return []

    entries = []
    with open(dataset_path) as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))
    return entries


def run_optimization(num_configs: int = 5, num_examples: int = 10):
    """Run optimization testing different prompt configurations."""
    from openai import OpenAI

    client = OpenAI()
    dataset = load_dataset()[:num_examples]
    evaluator = OperationsEvaluator()

    configs = [
        {
            "name": "baseline",
            "temperature": 0.3,
            "instruction": "Process this task efficiently.",
        },
        {
            "name": "cautious",
            "temperature": 0.1,
            "instruction": "Be conservative. When in doubt, escalate.",
        },
        {
            "name": "autonomous",
            "temperature": 0.5,
            "instruction": "Handle as much as possible autonomously.",
        },
        {
            "name": "detailed",
            "temperature": 0.2,
            "instruction": "Include all relevant validation steps.",
        },
        {
            "name": "minimal",
            "temperature": 0.4,
            "instruction": "Use minimum necessary steps.",
        },
    ][:num_configs]

    print("\n" + "=" * 70)
    print("OPTIMIZATION RUN: Testing Different Agent Configurations")
    print("=" * 70)
    print(
        f"\nConfigs: {num_configs}, Examples: {num_examples}, Total calls: {num_configs * num_examples}"
    )

    results = []
    for config in configs:
        print(f"\n--- Config: {config['name']} (temp={config['temperature']}) ---")
        scores = []

        for i, entry in enumerate(dataset):
            input_data = entry.get("input", {})
            expected_actions = entry.get("expected_actions", [])
            should_escalate = entry.get("should_escalate", False)

            prompt = f"""{config['instruction']}

Task: {input_data.get('task', 'Unknown')}
Amount: {input_data.get('amount', 'N/A')}
Requester: {input_data.get('requester', 'Unknown')}

Available actions: validate_amount, check_policy, auto_approve, escalate_to_manager, notify_requester

Return JSON: {{"actions": ["action1", ...], "should_escalate": true/false}}"""

            try:
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=config["temperature"],
                )
                content = response.choices[0].message.content
                import re

                json_match = re.search(r"\{.*\}", content, re.DOTALL)
                pred = (
                    json.loads(json_match.group())
                    if json_match
                    else {"actions": [], "should_escalate": False}
                )

                result = evaluator(
                    pred,
                    None,
                    {
                        "expected_actions": expected_actions,
                        "should_escalate": should_escalate,
                    },
                )
                scores.append(result)
                print(
                    f"  [{i+1}/{num_examples}] action={result['action_accuracy']:.2f} esc={'✓' if result['escalation_accuracy']==1 else '✗'}"
                )
            except Exception as e:
                print(f"  [{i+1}/{num_examples}] Error: {e}")
                scores.append(
                    {"action_accuracy": 0, "escalation_accuracy": 0, "overall": 0}
                )

        avg_action = sum(s["action_accuracy"] for s in scores) / len(scores)
        avg_esc = sum(s["escalation_accuracy"] for s in scores) / len(scores)
        results.append(
            {
                "config": config["name"],
                "temp": config["temperature"],
                "action_acc": avg_action,
                "esc_acc": avg_esc,
                "overall": (avg_action + avg_esc) / 2,
            }
        )

    print("\n" + "=" * 70)
    print("RESULTS TABLE")
    print("=" * 70)
    print(
        f"\n{'Config':<12} {'Temp':<6} {'Action Acc':<12} {'Escalation':<12} {'Overall':<10}"
    )
    print("-" * 52)
    for r in sorted(results, key=lambda x: x["overall"], reverse=True):
        print(
            f"{r['config']:<12} {r['temp']:<6.1f} {r['action_acc']:.3f}        {r['esc_acc']*100:>5.0f}%       {r['overall']:.3f}"
        )

    best = max(results, key=lambda x: x["overall"])
    print("-" * 52)
    print(f"🏆 BEST: {best['config']} (score={best['overall']:.3f})")
    print("=" * 70)
    return results


def print_score_bar(label: str, score: float, max_score: float = 1.0, width: int = 20):
    """Print a visual score bar."""
    normalized = min(score / max_score, 1.0)
    filled = int(normalized * width)
    bar = "█" * filled + "░" * (width - filled)
    pct = score * 100 if max_score == 1.0 else score
    print(f"  {label:<20} {bar} {pct:.0f}%")


def demo_evaluator():
    """Demo the Operations Agent evaluator with clear input/output examples."""
    mock_mode = is_mock_mode()

    print("=" * 70)
    print("OPERATIONS AGENT - Evaluator Demo")
    print("=" * 70)
    print(
        f"\nMODE: {'MOCK (heuristic rules)' if mock_mode else 'REAL (using OpenAI API)'}"
    )

    print(
        """
WHAT THIS AGENT DOES:
  A workflow automation agent that processes operational requests (like
  expense approvals, access requests, etc.) and decides what actions to take.
  It either auto-processes the request or escalates to a human.

HOW IT'S EVALUATED:"""
    )
    if mock_mode:
        print("  MOCK MODE: Using deterministic comparison (fast, free, no API needed)")
    else:
        print("  REAL MODE: Using LLM agent + evaluation (requires API key)")
    print()

    # Load and show dataset info
    dataset = load_dataset()
    print(f"DATASET: {len(dataset)} task scenarios in tasks_dataset.jsonl")

    if dataset:
        escalation_count = sum(1 for e in dataset if e.get("should_escalate", False))
        print(f"  - {escalation_count} tasks require human escalation")
        print(f"  - {len(dataset) - escalation_count} tasks can be auto-processed")

        print("\n" + "-" * 70)
        print("SAMPLE DATA (first 2 entries):")
        print("-" * 70)
        for i, entry in enumerate(dataset[:2]):
            input_data = entry.get("input", {})
            task_desc = (
                input_data.get("task", "N/A")
                if isinstance(input_data, dict)
                else str(input_data)
            )
            should_escalate = entry.get("should_escalate", False)
            expected_actions = entry.get("expected_actions", [])

            print(f"\n[Entry {i+1}]")
            print("  INPUT (task request):")
            print(
                f'    "{task_desc[:80]}..."'
                if len(str(task_desc)) > 80
                else f'    "{task_desc}"'
            )
            print("\n  OUTPUT (expected response):")
            print(f"    Actions: {expected_actions}")
            print(f"    Escalate: {'Yes' if should_escalate else 'No'}")

    evaluator = OperationsEvaluator()

    print("\n" + "=" * 70)
    print("HOW SCORING WORKS:")
    print("=" * 70)
    print(
        """
The evaluator measures:

  - Action Accuracy:     Do the agent's actions match what was expected?
                         (Checked using overlap + order of actions)

  - Escalation Accuracy: Did the agent correctly decide to escalate or not?
                         (Binary: right or wrong)

  - Efficiency:          Did the agent use the minimum number of steps?
                         (Fewer steps = better, if still correct)
"""
    )

    print("=" * 70)
    print("EVALUATION EXAMPLES:")
    print("=" * 70)

    # Test case 1: Perfect match
    print("\n[PERFECT MATCH] - Agent got it exactly right")
    print("-" * 70)
    print('Task: "Process expense report #12345 for $2,500 from Marketing"')
    print("\nAgent Output:")
    print("  Actions: [validate_amount, check_budget, auto_approve, notify_finance]")
    print("  Escalate: No")
    print("\nExpected:")
    print("  Actions: [validate_amount, check_budget, auto_approve, notify_finance]")
    print("  Escalate: No")

    pred1 = {
        "actions": [
            "validate_amount",
            "check_budget",
            "auto_approve",
            "notify_finance",
        ],
        "should_escalate": False,
    }
    exp1 = {
        "expected_actions": [
            "validate_amount",
            "check_budget",
            "auto_approve",
            "notify_finance",
        ],
        "should_escalate": False,
    }
    result = evaluator(prediction=pred1, expected=None, input_data=exp1)
    print("\nScores:")
    print_score_bar("Action Accuracy", result["action_accuracy"])
    print_score_bar("Escalation", result["escalation_accuracy"])
    print_score_bar("Efficiency", result["efficiency"])
    print(f"  {'─' * 42}")
    print_score_bar("OVERALL", result["overall"])

    # Test case 2: Wrong order
    print("\n[WRONG ORDER] - Right actions but wrong sequence")
    print("-" * 70)
    print('Task: "Process expense for server equipment"')
    print("\nAgent Output:")
    print("  Actions: [auto_approve, validate_amount, check_budget]  <- Wrong order!")
    print("  Escalate: No")
    print("\nExpected:")
    print("  Actions: [validate_amount, check_budget, auto_approve]")
    print("  Escalate: No")

    pred2 = {
        "actions": ["auto_approve", "validate_amount", "check_budget"],
        "should_escalate": False,
    }
    exp2 = {
        "expected_actions": ["validate_amount", "check_budget", "auto_approve"],
        "should_escalate": False,
    }
    result = evaluator(prediction=pred2, expected=None, input_data=exp2)
    print("\nScores:")
    print_score_bar("Action Accuracy", result["action_accuracy"])
    print("    ^ Same actions but wrong order penalized")
    print_score_bar("Escalation", result["escalation_accuracy"])
    print_score_bar("Efficiency", result["efficiency"])
    print(f"  {'─' * 42}")
    print_score_bar("OVERALL", result["overall"])

    # Test case 3: Missed escalation
    print("\n[MISSED ESCALATION] - Should have escalated but didn't")
    print("-" * 70)
    print('Task: "Process $15,000 equipment purchase (over policy limit)"')
    print("\nAgent Output:")
    print("  Actions: [validate_amount, auto_approve, notify_finance]")
    print("  Escalate: No  <- WRONG!")
    print("\nExpected:")
    print("  Actions: [validate_amount, flag_over_limit, escalate_to_manager]")
    print("  Escalate: Yes")

    pred3 = {
        "actions": ["validate_amount", "auto_approve", "notify_finance"],
        "should_escalate": False,
    }
    exp3 = {
        "expected_actions": [
            "validate_amount",
            "flag_over_limit",
            "escalate_to_manager",
        ],
        "should_escalate": True,
    }
    result = evaluator(prediction=pred3, expected=None, input_data=exp3)
    print("\nScores:")
    print_score_bar("Action Accuracy", result["action_accuracy"])
    print_score_bar("Escalation", result["escalation_accuracy"])
    print("    ^ Critical failure! Over-limit request auto-approved!")
    print_score_bar("Efficiency", result["efficiency"])
    print(f"  {'─' * 42}")
    print_score_bar("OVERALL", result["overall"])

    # Test case 4: Extra unnecessary actions
    print("\n[INEFFICIENT] - Correct but too many steps")
    print("-" * 70)
    print('Task: "Simple expense report for $50 lunch"')
    print("\nAgent Output:")
    print("  Actions: [validate_amount, check_budget, verify_identity,")
    print("           check_policy_limit, auto_approve, notify_finance, log_action]")
    print("  Escalate: No")
    print("\nExpected:")
    print("  Actions: [validate_amount, auto_approve, notify_finance]")
    print("  Escalate: No")

    pred4 = {
        "actions": [
            "validate_amount",
            "check_budget",
            "verify_identity",
            "check_policy_limit",
            "auto_approve",
            "notify_finance",
            "log_action",
        ],
        "should_escalate": False,
    }
    exp4 = {
        "expected_actions": ["validate_amount", "auto_approve", "notify_finance"],
        "should_escalate": False,
    }
    result = evaluator(prediction=pred4, expected=None, input_data=exp4)
    print("\nScores:")
    print_score_bar("Action Accuracy", result["action_accuracy"])
    print_score_bar("Escalation", result["escalation_accuracy"])
    print_score_bar("Efficiency", result["efficiency"])
    print("    ^ Simple task didn't need 7 steps!")
    print(f"  {'─' * 42}")
    print_score_bar("OVERALL", result["overall"])

    # In real mode, run optimization
    if not mock_mode:
        run_optimization(num_configs=5, num_examples=10)

    print("\n" + "=" * 70)
    print("HOW TO RUN:")
    print("  Mock mode (heuristics): python evaluator.py  (default)")
    print(
        "  Real mode (LLM+optimize): TRAIGENT_MOCK_MODE=false OPENAI_API_KEY=sk-... python evaluator.py"
    )
    print("=" * 70)


if __name__ == "__main__":
    demo_evaluator()
