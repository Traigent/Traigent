"""
Technical Support Request Classification Problem.

Classify technical support requests by issue type and severity,
helping prioritize and route support tickets effectively.
"""

import sys
from typing import Any, Callable, Dict, List, Optional

import traigent
from traigent.evaluators.base import Dataset, EvaluationExample

try:
    from langchain.chains import LLMChain
    from langchain.prompts import ChatPromptTemplate
    from langchain_openai import ChatOpenAI
except ImportError:
    print("Please install LangChain: pip install langchain langchain-openai")
    sys.exit(1)

from . import register_problem
from .base import BaseLangChainProblem, ProblemConfig, ProblemMetric


class TechSupportProblem(BaseLangChainProblem):
    """
    Technical support request classification problem.

    This problem tests the model's ability to:
    1. Identify the type of technical issue
    2. Assess severity and urgency
    3. Categorize for proper routing
    4. Distinguish between different technical domains
    """

    CATEGORIES = [
        "bug_report",
        "feature_request",
        "performance_issue",
        "security_concern",
        "documentation_update",
        "configuration_help",
        "integration_problem",
        "user_error",
    ]

    @classmethod
    def get_default_config(cls) -> ProblemConfig:
        """Get default configuration for this problem."""
        return ProblemConfig(
            name="tech_support",
            description="Classify technical support requests by issue type for effective routing and prioritization",
            difficulty_level="Advanced",
            dataset_size=30,
            model_configurations={
                "model": ["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o"],
                "temperature": [0.1, 0.3, 0.5],
                "max_tokens": [60, 100],
            },
            metrics=[
                ProblemMetric(
                    "accuracy", "Overall classification accuracy", True, 1.0, ".1%"
                ),
                ProblemMetric(
                    "severity_detection",
                    "Accuracy in detecting issue severity",
                    True,
                    0.9,
                    ".1%",
                ),
                ProblemMetric(
                    "category_precision",
                    "Precision across issue categories",
                    True,
                    0.8,
                    ".1%",
                ),
                ProblemMetric(
                    "routing_effectiveness",
                    "Effectiveness of ticket routing",
                    True,
                    0.7,
                    ".1%",
                ),
            ],
            optimization_objectives=["accuracy"],
            expected_model_ranking=["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"],
        )

    def __init__(self, config: Optional[ProblemConfig] = None):
        if config is None:
            config = self.get_default_config()
        super().__init__(config)

    def create_dataset(self) -> Dataset:
        """Create technical support classification dataset."""
        examples_data = [
            # Easy cases - clear issue types
            {
                "issue": "The login button doesn't work when I click it",
                "category": "bug_report",
                "severity": "medium",
                "difficulty": "easy",
                "reasoning": "Clear functionality failure",
            },
            {
                "issue": "Can you add dark mode to the settings?",
                "category": "feature_request",
                "severity": "low",
                "difficulty": "easy",
                "reasoning": "Direct feature request",
            },
            {
                "issue": "The app crashes every time I upload a large file",
                "category": "bug_report",
                "severity": "high",
                "difficulty": "easy",
                "reasoning": "Reproducible crash issue",
            },
            {
                "issue": "How do I configure email notifications?",
                "category": "configuration_help",
                "severity": "low",
                "difficulty": "easy",
                "reasoning": "Configuration question",
            },
            {
                "issue": "The API documentation is missing examples for authentication",
                "category": "documentation_update",
                "severity": "medium",
                "difficulty": "easy",
                "reasoning": "Documentation gap",
            },
            # Medium difficulty - requires analysis
            {
                "issue": "Database queries timeout during peak hours with concurrent users",
                "category": "performance_issue",
                "severity": "high",
                "difficulty": "medium",
                "reasoning": "Performance degradation under load",
            },
            {
                "issue": "Getting 403 errors when trying to access admin panel after update",
                "category": "bug_report",
                "severity": "high",
                "difficulty": "medium",
                "reasoning": "Access control issue after change",
            },
            {
                "issue": "Need better error messages for validation failures",
                "category": "feature_request",
                "severity": "medium",
                "difficulty": "medium",
                "reasoning": "UX improvement request",
            },
            {
                "issue": "Integration with Slack webhooks stopped working yesterday",
                "category": "integration_problem",
                "severity": "high",
                "difficulty": "medium",
                "reasoning": "Third-party integration failure",
            },
            {
                "issue": "Page load times increased by 300% after last deployment",
                "category": "performance_issue",
                "severity": "high",
                "difficulty": "medium",
                "reasoning": "Performance regression",
            },
            {
                "issue": "Can't figure out how to export data in CSV format",
                "category": "user_error",
                "severity": "low",
                "difficulty": "medium",
                "reasoning": "User needs guidance, feature exists",
            },
            {
                "issue": "Memory usage keeps growing, server needs restart every 2 days",
                "category": "bug_report",
                "severity": "high",
                "difficulty": "medium",
                "reasoning": "Memory leak issue",
            },
            # Hard cases - complex or ambiguous
            {
                "issue": "Cross-site scripting vulnerability in user-generated content",
                "category": "security_concern",
                "severity": "critical",
                "difficulty": "hard",
                "reasoning": "Security vulnerability",
            },
            {
                "issue": "Distributed transaction rollback fails in microservices architecture",
                "category": "bug_report",
                "severity": "critical",
                "difficulty": "hard",
                "reasoning": "Complex distributed system issue",
            },
            {
                "issue": "OAuth flow redirects to wrong URL in mobile app but works on web",
                "category": "bug_report",
                "severity": "high",
                "difficulty": "hard",
                "reasoning": "Platform-specific authentication issue",
            },
            {
                "issue": "Rate limiting seems inconsistent, some users hit limits others don't",
                "category": "bug_report",
                "severity": "medium",
                "difficulty": "hard",
                "reasoning": "Inconsistent behavior investigation needed",
            },
            {
                "issue": "Need GDPR compliance features for data deletion and export",
                "category": "feature_request",
                "severity": "high",
                "difficulty": "hard",
                "reasoning": "Compliance requirement",
            },
            {
                "issue": "API returns 200 but data is corrupted when payload > 10MB",
                "category": "bug_report",
                "severity": "high",
                "difficulty": "hard",
                "reasoning": "Silent data corruption issue",
            },
            {
                "issue": "Kubernetes pods randomly restart, no clear pattern in logs",
                "category": "performance_issue",
                "severity": "high",
                "difficulty": "hard",
                "reasoning": "Infrastructure stability issue",
            },
            {
                "issue": "SQL injection possible through search parameter",
                "category": "security_concern",
                "severity": "critical",
                "difficulty": "hard",
                "reasoning": "Critical security vulnerability",
            },
            # More examples to reach 30
            {
                "issue": "The mobile app freezes when switching between tabs quickly",
                "category": "bug_report",
                "severity": "medium",
                "difficulty": "medium",
                "reasoning": "Mobile-specific UI issue",
            },
            {
                "issue": "Can we add SSO support for enterprise customers?",
                "category": "feature_request",
                "severity": "medium",
                "difficulty": "easy",
                "reasoning": "Enterprise feature request",
            },
            {
                "issue": "Backup process fails with 'disk full' but there's 50GB free",
                "category": "bug_report",
                "severity": "high",
                "difficulty": "hard",
                "reasoning": "Misleading error, investigation needed",
            },
            {
                "issue": "Users report seeing other users' data occasionally",
                "category": "security_concern",
                "severity": "critical",
                "difficulty": "hard",
                "reasoning": "Critical data leak issue",
            },
            {
                "issue": "How do I set up two-factor authentication?",
                "category": "configuration_help",
                "severity": "low",
                "difficulty": "easy",
                "reasoning": "Security configuration help",
            },
            {
                "issue": "The dashboard graphs don't update in real-time anymore",
                "category": "bug_report",
                "severity": "medium",
                "difficulty": "medium",
                "reasoning": "Feature regression",
            },
            {
                "issue": "WebSocket connections drop after exactly 30 seconds",
                "category": "bug_report",
                "severity": "high",
                "difficulty": "medium",
                "reasoning": "Connection stability issue",
            },
            {
                "issue": "Search results take 10+ seconds for large datasets",
                "category": "performance_issue",
                "severity": "high",
                "difficulty": "medium",
                "reasoning": "Search performance issue",
            },
            {
                "issue": "Password reset emails not being received by Gmail users",
                "category": "integration_problem",
                "severity": "high",
                "difficulty": "medium",
                "reasoning": "Email delivery issue",
            },
            {
                "issue": "Can't delete my account, button does nothing",
                "category": "bug_report",
                "severity": "medium",
                "difficulty": "easy",
                "reasoning": "Account management bug",
            },
        ]

        examples = []
        for i, data in enumerate(examples_data):
            example = EvaluationExample(
                input_data={"issue_description": data["issue"]},
                expected_output=data["category"],
                metadata={
                    "difficulty": data["difficulty"],
                    "severity": data["severity"],
                    "reasoning": data["reasoning"],
                    "example_id": f"tech_{i + 1:03d}",
                },
            )
            examples.append(example)

        return Dataset(
            examples=examples,
            name="Technical Support Classification",
            description="Classify technical support requests by issue type and severity",
        )

    def create_function(self) -> Callable:
        """Create the base tech support classifier function."""

        def tech_support_classifier(issue_description: str) -> str:
            """Classify technical support requests."""
            llm = ChatOpenAI(
                model="gpt-4o-mini",
                temperature=0.3,
                model_kwargs={"max_tokens": 60},
            )

            prompt_template = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        """You are a technical support ticket classifier.
Classify support requests into exactly one of these categories:
bug_report, feature_request, performance_issue, security_concern, documentation_update, configuration_help, integration_problem, or user_error.

Consider the technical nature, severity, and required action when classifying.""",
                    ),
                    (
                        "human",
                        """Classify this technical support request:

Issue: {issue}

Category:""",
                    ),
                ]
            )

            chain = LLMChain(llm=llm, prompt=prompt_template)
            result = chain.invoke({"issue": issue_description})["text"]

            return self.clean_llm_output(result, self.CATEGORIES)

        return tech_support_classifier

    def create_optimized_function(self) -> Callable:
        """Create the optimized tech support classifier."""

        @traigent.optimize(
            eval_dataset=self.create_temporary_dataset_file(),
            objectives=self.get_optimization_objectives(),
            configuration_space=self.get_configuration_space(),
            auto_override_frameworks=True,
            framework_targets=["langchain_openai.ChatOpenAI"],
            execution_mode="edge_analytics",
        )
        def tech_support_classifier_optimized(issue_description: str) -> str:
            """Optimized technical support ticket classifier."""
            llm = ChatOpenAI(
                model="gpt-4o-mini",  # Will be overridden by TraiGent
                temperature=0.3,  # Will be overridden by TraiGent
                model_kwargs={"max_tokens": 60},  # Will be overridden by TraiGent
            )

            prompt_template = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        """You are a technical support ticket classifier.
Classify support requests into exactly one of these categories:
bug_report, feature_request, performance_issue, security_concern, documentation_update, configuration_help, integration_problem, or user_error.

Consider the technical nature, severity, and required action when classifying.""",
                    ),
                    (
                        "human",
                        """Classify this technical support request:

Issue: {issue}

Category:""",
                    ),
                ]
            )

            chain = LLMChain(llm=llm, prompt=prompt_template)
            result = chain.invoke({"issue": issue_description})["text"]

            return self.clean_llm_output(result, self.CATEGORIES)

        return tech_support_classifier_optimized

    def evaluate_custom_metrics(
        self,
        outputs: List[Any],
        expected_outputs: List[Any],
        errors: List[Optional[str]],
    ) -> Dict[str, float]:
        """Compute tech support specific metrics."""
        metrics = {}

        # Get dataset for metadata access
        dataset = self.get_dataset()

        # Standard accuracy
        correct = 0
        total = 0
        severity_correct = 0
        category_precision = {
            cat: {"tp": 0, "fp": 0, "fn": 0} for cat in self.CATEGORIES
        }
        routing_effective = 0

        for i, (output, expected, error) in enumerate(
            zip(outputs, expected_outputs, errors)
        ):
            if error is None and expected is not None:
                total += 1

                # Overall accuracy
                if output == expected:
                    correct += 1
                    category_precision[expected]["tp"] += 1
                else:
                    if output in self.CATEGORIES:
                        category_precision[output]["fp"] += 1
                    category_precision[expected]["fn"] += 1

                # Severity detection
                if i < len(dataset.examples):
                    severity = dataset.examples[i].metadata.get("severity", "medium")
                    # Consider high-severity issues correctly identified
                    if severity in ["high", "critical"]:
                        if output in [
                            "bug_report",
                            "security_concern",
                            "performance_issue",
                        ]:
                            severity_correct += 1
                    elif output == expected:
                        severity_correct += 1

                    # Routing effectiveness
                    # Security issues must be routed correctly
                    if expected == "security_concern" and output == "security_concern":
                        routing_effective += 1
                    # Critical bugs must be identified as bugs at least
                    elif severity == "critical" and output == "bug_report":
                        routing_effective += 1
                    # Other correct classifications
                    elif output == expected:
                        routing_effective += 1

        metrics["accuracy"] = correct / total if total > 0 else 0.0
        metrics["severity_detection"] = severity_correct / total if total > 0 else 0.0

        # Calculate average precision across categories
        precisions = []
        for _cat, counts in category_precision.items():
            if counts["tp"] + counts["fp"] > 0:
                precision = counts["tp"] / (counts["tp"] + counts["fp"])
                precisions.append(precision)

        metrics["category_precision"] = (
            sum(precisions) / len(precisions) if precisions else 0.0
        )
        metrics["routing_effectiveness"] = (
            routing_effective / total if total > 0 else 0.0
        )

        return metrics


# Register this problem
register_problem("tech_support", TechSupportProblem)
