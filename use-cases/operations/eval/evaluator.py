#!/usr/bin/env python3
"""
Evaluator for Operations Agent

This evaluator scores workflow automation responses on:
1. Action Sequence Accuracy - How well the generated actions match expected
2. Escalation Accuracy - Precision/Recall of escalation decisions
3. Execution Efficiency - Action economy (min steps / actual steps)

Based on the Traigent Agent Optimization Guide specifications.
"""

from dataclasses import dataclass
from typing import Any


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
        step_accuracy = self._evaluate_step_level(
            predicted_actions, expected_actions
        )

        # Evaluate escalation decision
        escalation_correct = predicted_escalate == expected_escalate

        # Calculate action economy
        if predicted_actions and expected_actions:
            action_economy = min(len(expected_actions) / max(len(predicted_actions), 1), 1.0)
        else:
            action_economy = 1.0 if not predicted_actions and not expected_actions else 0.0

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


def load_dataset() -> list[dict]:
    """Load the tasks dataset."""
    import json
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


def demo_evaluator():
    """
    Demo the Operations Agent evaluator.

    This runs in MOCK MODE - no API calls are made.
    The evaluator uses deterministic rules to score workflow automation on:
    - Action Sequence Accuracy: Match rate against ground truth actions
    - Escalation Accuracy: Correct routing decisions (escalate vs auto-process)
    - Efficiency: Action economy (fewer steps = better, if correct)
    """
    print("=" * 60)
    print("OPERATIONS AGENT - Evaluator Demo")
    print("=" * 60)
    print("\nMODE: Mock (deterministic scoring, no API calls)")
    print("\nEVALUATOR: OperationsEvaluator")
    print("  - Compares predicted action sequences to ground truth")
    print("  - Evaluates escalation decisions (binary classification)")
    print("  - Measures action economy (min_steps / actual_steps)")

    # Load and show dataset info
    dataset = load_dataset()
    print(f"\nDATASET: tasks_dataset.jsonl ({len(dataset)} task scenarios)")

    if dataset:
        # Count escalations
        escalation_count = sum(1 for e in dataset if e.get("should_escalate", False))
        print(f"  - Tasks requiring escalation: {escalation_count}")
        print(f"  - Tasks for auto-processing: {len(dataset) - escalation_count}")

        print("\n" + "-" * 60)
        print("FIRST 3 TASKS FROM DATASET:")
        print("-" * 60)
        for i, entry in enumerate(dataset[:3]):
            task_type = entry.get("task_type", "unknown")
            should_escalate = entry.get("should_escalate", False)
            expected_actions = entry.get("expected_actions", [])
            print(f"\n  [{i+1}] Type: {task_type}")
            print(f"      Escalate: {'Yes' if should_escalate else 'No'}")
            print(f"      Expected Actions: {', '.join(expected_actions[:3])}{'...' if len(expected_actions) > 3 else ''}")

    evaluator = OperationsEvaluator()

    print("\n" + "=" * 60)
    print("EVALUATION EXAMPLES")
    print("=" * 60)

    # Test case 1: Exact match
    print("\n[EXAMPLE 1] Perfect action sequence match")
    print("-" * 60)
    pred1 = {"actions": ["validate_amount", "check_budget", "auto_approve", "notify_finance"], "should_escalate": False}
    exp1 = {"expected_actions": ["validate_amount", "check_budget", "auto_approve", "notify_finance"], "should_escalate": False}
    result = evaluator(prediction=pred1, expected=None, input_data=exp1)
    print(f"  Predicted: {pred1['actions']}")
    print(f"  Expected:  {exp1['expected_actions']}")
    print(f"\n  Scores:")
    print(f"    Action Accuracy:     {result['action_accuracy']:.2f}")
    print(f"    Escalation Accuracy: {result['escalation_accuracy']:.2f}")
    print(f"    Efficiency:          {result['efficiency']:.2f}")
    print(f"    ─────────────────────────")
    print(f"    Overall:             {result['overall']:.2f}")

    # Test case 2: Partial match
    print("\n[EXAMPLE 2] Partial match (similar but not exact)")
    print("-" * 60)
    pred2 = {"actions": ["validate_amount", "check_policy", "escalate_to_manager"], "should_escalate": True}
    exp2 = {"expected_actions": ["validate_amount", "check_policy_limit", "flag_over_limit", "escalate_to_manager"], "should_escalate": True}
    result = evaluator(prediction=pred2, expected=None, input_data=exp2)
    print(f"  Predicted: {pred2['actions']}")
    print(f"  Expected:  {exp2['expected_actions']}")
    print(f"\n  Scores:")
    print(f"    Action Accuracy:     {result['action_accuracy']:.2f}")
    print(f"    Escalation Accuracy: {result['escalation_accuracy']:.2f}")
    print(f"    Efficiency:          {result['efficiency']:.2f}")
    print(f"    ─────────────────────────")
    print(f"    Overall:             {result['overall']:.2f}")

    # Test case 3: Wrong escalation
    print("\n[EXAMPLE 3] Wrong escalation decision")
    print("-" * 60)
    pred3 = {"actions": ["validate_amount", "auto_approve", "notify_finance"], "should_escalate": False}
    exp3 = {"expected_actions": ["validate_amount", "escalate_to_manager"], "should_escalate": True}
    result = evaluator(prediction=pred3, expected=None, input_data=exp3)
    print(f"  Predicted escalate: {pred3['should_escalate']} | Expected: {exp3['should_escalate']}")
    print(f"\n  Scores:")
    print(f"    Action Accuracy:     {result['action_accuracy']:.2f}")
    print(f"    Escalation Accuracy: {result['escalation_accuracy']:.2f} ← WRONG!")
    print(f"    Efficiency:          {result['efficiency']:.2f}")
    print(f"    ─────────────────────────")
    print(f"    Overall:             {result['overall']:.2f}")

    print("\n" + "=" * 60)
    print("To run optimization with real API calls:")
    print("  export OPENAI_API_KEY=<your-key>")
    print("  unset TRAIGENT_MOCK_MODE")
    print("  python use-cases/operations/agent/operations_agent.py")
    print("=" * 60)


if __name__ == "__main__":
    demo_evaluator()
