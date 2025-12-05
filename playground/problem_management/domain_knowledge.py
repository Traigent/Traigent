"""
Domain Knowledge Management for Problem Generation.

This module provides domain-specific patterns, categories, and knowledge
for generating realistic and relevant examples across different domains.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


class DomainKnowledge:
    """
    Manages domain-specific knowledge and patterns for problem generation.

    Provides structured information about different domains including
    common categories, typical inputs/outputs, complexity patterns,
    and domain-specific terminology.
    """

    def __init__(self):
        """Initialize domain knowledge."""
        self.knowledge_dir = Path(__file__).parent / "domain_knowledge"
        self.knowledge_dir.mkdir(exist_ok=True)

        # Initialize built-in domain knowledge
        self._init_builtin_knowledge()

    def _init_builtin_knowledge(self):
        """Initialize built-in domain knowledge patterns."""

        # Customer Service Domain
        customer_service = {
            "categories": [
                "billing_issue",
                "technical_support",
                "product_inquiry",
                "shipping_inquiry",
                "return_request",
                "account_support",
                "policy_question",
                "order_modification",
            ],
            "input_patterns": {"query": "Customer inquiry or complaint text"},
            "output_patterns": {"category": "Classification of the customer inquiry"},
            "complexity_indicators": {
                "easy": ["clear intent", "single issue", "standard request"],
                "medium": ["some ambiguity", "multiple aspects", "context needed"],
                "hard": ["unclear intent", "complex situation", "edge case"],
                "very_hard": ["multi-faceted", "conflicting information", "nuanced"],
                "expert": [
                    "extremely ambiguous",
                    "multiple overlapping issues",
                    "edge cases",
                ],
            },
            "common_terms": [
                "order",
                "shipping",
                "return",
                "refund",
                "account",
                "password",
                "billing",
                "charge",
                "delivery",
                "product",
                "service",
                "support",
            ],
            "typical_scenarios": [
                "delayed shipment",
                "billing error",
                "product defect",
                "account access",
                "return request",
                "policy question",
                "technical issue",
                "order change",
            ],
        }

        # Legal Domain
        legal = {
            "categories": [
                "contract_review",
                "compliance_check",
                "risk_assessment",
                "liability_analysis",
                "regulatory_review",
                "ip_analysis",
            ],
            "input_patterns": {
                "contract_text": "Legal document or contract text",
                "analysis_type": "Type of legal analysis required",
            },
            "output_patterns": {
                "risks": "List of identified legal risks",
                "compliance_issues": "Compliance concerns",
                "recommendations": "Legal recommendations",
            },
            "complexity_indicators": {
                "easy": ["standard contract", "clear terms", "common clauses"],
                "medium": ["multiple parties", "some complexity", "industry-specific"],
                "hard": [
                    "complex terms",
                    "unusual clauses",
                    "regulatory considerations",
                ],
                "very_hard": [
                    "multi-jurisdictional",
                    "complex IP",
                    "regulatory overlap",
                ],
                "expert": [
                    "novel legal issues",
                    "conflicting laws",
                    "unprecedented clauses",
                ],
            },
            "common_terms": [
                "contract",
                "liability",
                "indemnification",
                "termination",
                "breach",
                "compliance",
                "regulatory",
                "jurisdiction",
                "governing law",
                "dispute",
            ],
            "contract_types": [
                "employment",
                "service",
                "purchase",
                "rental",
                "partnership",
                "licensing",
                "distribution",
                "consulting",
                "vendor",
            ],
        }

        # Medical Domain
        medical = {
            "categories": [
                "emergency",
                "routine_checkup",
                "specialist_referral",
                "prescription_request",
                "test_results",
                "chronic_care",
            ],
            "input_patterns": {
                "symptoms": "Patient symptoms description",
                "patient_history": "Medical history information",
                "test_results": "Laboratory or diagnostic results",
            },
            "output_patterns": {
                "diagnosis_category": "Medical diagnosis category",
                "urgency_level": "Level of medical urgency",
                "recommendations": "Medical recommendations",
            },
            "complexity_indicators": {
                "easy": ["clear symptoms", "common condition", "straightforward"],
                "medium": ["multiple symptoms", "requires history", "differential"],
                "hard": ["complex presentation", "multiple conditions", "rare disease"],
                "very_hard": [
                    "unusual symptoms",
                    "complex interactions",
                    "diagnostic challenge",
                ],
                "expert": [
                    "rare condition",
                    "multiple complications",
                    "research-level complexity",
                ],
            },
            "common_terms": [
                "symptoms",
                "diagnosis",
                "treatment",
                "medication",
                "condition",
                "patient",
                "doctor",
                "hospital",
                "clinic",
                "emergency",
            ],
            "specialties": [
                "cardiology",
                "neurology",
                "orthopedics",
                "dermatology",
                "psychiatry",
                "oncology",
                "pediatrics",
                "internal_medicine",
                "surgery",
            ],
        }

        # Technical Domain
        technical = {
            "categories": [
                "bug_report",
                "feature_request",
                "performance_issue",
                "security_concern",
                "documentation_update",
                "infrastructure",
            ],
            "input_patterns": {
                "code": "Source code to analyze",
                "language": "Programming language",
                "context": "Code context or description",
            },
            "output_patterns": {
                "issues": "List of identified issues",
                "severity": "Issue severity level",
                "suggestions": "Improvement suggestions",
            },
            "complexity_indicators": {
                "easy": ["simple bug", "clear issue", "standard practice"],
                "medium": [
                    "design issue",
                    "performance concern",
                    "moderate complexity",
                ],
                "hard": [
                    "architectural issue",
                    "security vulnerability",
                    "complex logic",
                ],
                "very_hard": [
                    "system-wide impact",
                    "complex interactions",
                    "optimization",
                ],
                "expert": ["advanced algorithms", "novel approaches", "research-level"],
            },
            "common_terms": [
                "code",
                "function",
                "variable",
                "algorithm",
                "database",
                "API",
                "security",
                "performance",
                "bug",
                "feature",
                "system",
                "architecture",
            ],
            "issue_types": [
                "logic_error",
                "null_pointer",
                "memory_leak",
                "race_condition",
                "security_vulnerability",
                "performance_bottleneck",
                "design_flaw",
            ],
        }

        # Financial Domain
        financial = {
            "categories": [
                "loan_application",
                "investment_advice",
                "fraud_detection",
                "credit_assessment",
                "insurance_claim",
                "tax_advice",
            ],
            "input_patterns": {
                "financial_data": "Financial information or statements",
                "transaction_details": "Transaction information",
                "risk_profile": "Risk assessment data",
            },
            "output_patterns": {
                "risk_level": "Financial risk assessment",
                "recommendation": "Financial recommendation",
                "approval_status": "Approval or rejection decision",
            },
            "complexity_indicators": {
                "easy": ["simple transaction", "standard product", "clear guidelines"],
                "medium": ["multiple factors", "some risk", "moderate complexity"],
                "hard": ["high risk", "complex product", "regulatory considerations"],
                "very_hard": [
                    "institutional level",
                    "complex derivatives",
                    "regulatory overlap",
                ],
                "expert": [
                    "novel instruments",
                    "systemic risk",
                    "regulatory uncertainty",
                ],
            },
            "common_terms": [
                "loan",
                "credit",
                "investment",
                "risk",
                "return",
                "portfolio",
                "insurance",
                "fraud",
                "compliance",
                "regulation",
                "audit",
            ],
            "product_types": [
                "checking_account",
                "savings_account",
                "credit_card",
                "mortgage",
                "personal_loan",
                "investment_account",
                "insurance_policy",
            ],
        }

        # Educational Domain
        educational = {
            "categories": [
                "course_content",
                "assessment",
                "student_support",
                "curriculum_design",
                "learning_outcome",
                "feedback",
            ],
            "input_patterns": {
                "content": "Educational content or material",
                "student_level": "Academic level of students",
                "subject": "Subject or topic area",
            },
            "output_patterns": {
                "learning_objective": "Educational learning objectives",
                "assessment_method": "Recommended assessment approach",
                "difficulty_level": "Content difficulty rating",
            },
            "complexity_indicators": {
                "easy": ["basic concepts", "introductory level", "clear objectives"],
                "medium": [
                    "intermediate concepts",
                    "some prerequisites",
                    "moderate depth",
                ],
                "hard": [
                    "advanced concepts",
                    "complex relationships",
                    "synthesis required",
                ],
                "very_hard": ["expert level", "research-based", "critical analysis"],
                "expert": [
                    "cutting-edge research",
                    "theoretical complexity",
                    "novel approaches",
                ],
            },
            "common_terms": [
                "learning",
                "student",
                "curriculum",
                "assessment",
                "objective",
                "course",
                "lesson",
                "assignment",
                "feedback",
                "evaluation",
            ],
            "subject_areas": [
                "mathematics",
                "science",
                "literature",
                "history",
                "language",
                "arts",
                "technology",
                "social_studies",
                "philosophy",
            ],
        }

        # Store built-in knowledge
        self.builtin_domains = {
            "customer_service": customer_service,
            "legal": legal,
            "medical": medical,
            "technical": technical,
            "financial": financial,
            "educational": educational,
        }

        # Save to files if they don't exist
        for domain, knowledge in self.builtin_domains.items():
            domain_file = self.knowledge_dir / f"{domain}.json"
            if not domain_file.exists():
                self.save_domain_knowledge(domain, knowledge)

    def get_domain_patterns(self, domain: str) -> Dict[str, Any]:
        """
        Get patterns and knowledge for a specific domain.

        Args:
            domain: Domain name

        Returns:
            Dictionary containing domain-specific patterns
        """
        # Try to load from file first
        domain_file = self.knowledge_dir / f"{domain}.json"
        if domain_file.exists():
            try:
                with open(domain_file) as f:
                    return json.load(f)
            except Exception:
                pass  # Fall back to built-in

        # Fall back to built-in knowledge
        if domain in self.builtin_domains:
            return self.builtin_domains[domain]

        # Return general pattern if domain not found
        return self._get_general_patterns()

    def save_domain_knowledge(self, domain: str, knowledge: Dict[str, Any]):
        """
        Save domain knowledge to file.

        Args:
            domain: Domain name
            knowledge: Domain knowledge dictionary
        """
        domain_file = self.knowledge_dir / f"{domain}.json"
        with open(domain_file, "w") as f:
            json.dump(knowledge, f, indent=2)

    def get_available_domains(self) -> List[str]:
        """Get list of available domains."""
        domains = set(self.builtin_domains.keys())

        # Add domains from files
        for domain_file in self.knowledge_dir.glob("*.json"):
            domains.add(domain_file.stem)

        return sorted(domains)

    def get_domain_categories(self, domain: str) -> List[str]:
        """Get categories for a specific domain."""
        patterns = self.get_domain_patterns(domain)
        return patterns.get("categories", [])

    def get_complexity_indicators(self, domain: str, difficulty: str) -> List[str]:
        """Get complexity indicators for a domain and difficulty level."""
        patterns = self.get_domain_patterns(domain)
        complexity = patterns.get("complexity_indicators", {})
        return complexity.get(difficulty, [])

    def get_common_terms(self, domain: str) -> List[str]:
        """Get common terms for a domain."""
        patterns = self.get_domain_patterns(domain)
        return patterns.get("common_terms", [])

    def suggest_input_structure(self, domain: str) -> Dict[str, str]:
        """Suggest input structure for a domain."""
        patterns = self.get_domain_patterns(domain)
        return patterns.get("input_patterns", {"input": "str"})

    def suggest_output_structure(self, domain: str) -> Dict[str, str]:
        """Suggest output structure for a domain."""
        patterns = self.get_domain_patterns(domain)
        return patterns.get("output_patterns", {"output": "str"})

    def _get_general_patterns(self) -> Dict[str, Any]:
        """Get general patterns for unknown domains."""
        return {
            "categories": ["category_a", "category_b", "category_c", "category_d"],
            "input_patterns": {"text": "Input text to classify"},
            "output_patterns": {"category": "Classification result"},
            "complexity_indicators": {
                "easy": ["clear", "simple", "straightforward"],
                "medium": ["moderate", "some complexity", "context needed"],
                "hard": ["complex", "ambiguous", "challenging"],
                "very_hard": ["very complex", "multiple factors", "edge cases"],
                "expert": ["extremely complex", "novel", "unprecedented"],
            },
            "common_terms": ["text", "content", "data", "information", "analysis"],
            "typical_scenarios": [
                "classification",
                "analysis",
                "evaluation",
                "assessment",
            ],
        }

    def add_custom_domain(
        self,
        domain: str,
        categories: List[str],
        input_patterns: Dict[str, str],
        output_patterns: Dict[str, str],
        common_terms: List[str],
        complexity_indicators: Optional[Dict[str, List[str]]] = None,
    ):
        """
        Add a custom domain with its patterns.

        Args:
            domain: Domain name
            categories: List of categories for classification
            input_patterns: Input structure patterns
            output_patterns: Output structure patterns
            common_terms: Common terminology
            complexity_indicators: Optional complexity indicators by difficulty
        """
        if complexity_indicators is None:
            complexity_indicators = {
                "easy": ["simple", "clear", "basic"],
                "medium": ["moderate", "some complexity"],
                "hard": ["complex", "challenging"],
                "very_hard": ["very complex", "multiple factors"],
                "expert": ["extremely complex", "expert-level"],
            }

        knowledge = {
            "categories": categories,
            "input_patterns": input_patterns,
            "output_patterns": output_patterns,
            "complexity_indicators": complexity_indicators,
            "common_terms": common_terms,
            "custom_domain": True,
        }

        self.save_domain_knowledge(domain, knowledge)

    def validate_domain_knowledge(self, domain: str) -> Dict[str, Any]:
        """
        Validate domain knowledge structure.

        Args:
            domain: Domain to validate

        Returns:
            Validation result with issues and suggestions
        """
        patterns = self.get_domain_patterns(domain)
        issues = []
        suggestions = []

        # Check required fields
        required_fields = ["categories", "input_patterns", "output_patterns"]
        for field in required_fields:
            if field not in patterns:
                issues.append(f"Missing required field: {field}")

        # Check categories
        categories = patterns.get("categories", [])
        if not categories:
            issues.append("No categories defined")
        elif len(categories) < 2:
            suggestions.append(
                "Consider adding more categories for better differentiation"
            )

        # Check complexity indicators
        complexity = patterns.get("complexity_indicators", {})
        expected_levels = ["easy", "medium", "hard", "very_hard", "expert"]
        missing_levels = [level for level in expected_levels if level not in complexity]
        if missing_levels:
            suggestions.append(
                f"Consider adding complexity indicators for: {', '.join(missing_levels)}"
            )

        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "suggestions": suggestions,
            "completeness_score": self._calculate_completeness(patterns),
        }

    def _calculate_completeness(self, patterns: Dict[str, Any]) -> float:
        """Calculate completeness score for domain knowledge."""
        required_fields = [
            "categories",
            "input_patterns",
            "output_patterns",
            "complexity_indicators",
            "common_terms",
        ]

        score = 0.0
        total_weight = len(required_fields)

        for field in required_fields:
            if field in patterns and patterns[field]:
                score += 1.0

        return score / total_weight if total_weight > 0 else 0.0
