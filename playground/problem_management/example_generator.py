"""
Example Generator for LangChain Optimization Problems.

This module generates realistic, challenging examples across different domains
and difficulty levels for LangChain optimization problems.
"""

import random
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .domain_knowledge import DomainKnowledge
from .intelligence import ProblemInsights


@dataclass
class GeneratedExample:
    """A generated example with metadata."""

    input_data: Dict[str, Any]
    expected_output: Any
    difficulty: str
    reasoning: str
    metadata: Dict[str, Any]


class ExampleGenerator:
    """
    Intelligent example generator for different problem types and domains.

    Generates realistic examples with appropriate difficulty progression
    and domain-specific knowledge.
    """

    def __init__(self):
        """Initialize the example generator."""
        self.domain_knowledge = DomainKnowledge()

        # Difficulty tier specifications
        self.difficulty_specs = {
            "easy": {
                "complexity": "low",
                "ambiguity": "none",
                "context_required": False,
                "edge_cases": False,
            },
            "medium": {
                "complexity": "moderate",
                "ambiguity": "low",
                "context_required": True,
                "edge_cases": False,
            },
            "hard": {
                "complexity": "moderate",
                "ambiguity": "moderate",
                "context_required": True,
                "edge_cases": False,
            },
            "very_hard": {
                "complexity": "high",
                "ambiguity": "high",
                "context_required": True,
                "edge_cases": True,
            },
            "expert": {
                "complexity": "very_high",
                "ambiguity": "very_high",
                "context_required": True,
                "edge_cases": True,
            },
        }

    async def generate_examples(
        self,
        problem_type: str,
        domain: str,
        description: str,
        count: int,
        difficulty: str = "Advanced",
        insights: Optional[ProblemInsights] = None,
    ) -> List[GeneratedExample]:
        """
        Generate examples for a new problem.

        Args:
            problem_type: Type of problem (classification, generation, etc.)
            domain: Problem domain
            description: Natural language description
            count: Number of examples to generate
            difficulty: Overall difficulty level
            insights: Optional problem insights

        Returns:
            List of generated examples
        """
        # Determine difficulty distribution
        difficulty_tiers = (
            insights.difficulty_tiers
            if insights
            else self._get_default_tiers(difficulty)
        )
        distribution = self._calculate_distribution(count, difficulty_tiers)

        examples = []

        for tier, tier_count in distribution.items():
            tier_examples = await self._generate_tier_examples(
                problem_type=problem_type,
                domain=domain,
                description=description,
                difficulty=tier,
                count=tier_count,
                insights=insights,
            )
            examples.extend(tier_examples)

        # Shuffle to avoid predictable ordering
        random.shuffle(examples)

        return examples

    async def generate_additional_examples(
        self,
        existing_problem: str,
        count: int,
        difficulty: Optional[str] = None,
        balance: bool = False,
        edge_cases: bool = False,
    ) -> List[GeneratedExample]:
        """
        Generate additional examples for an existing problem.

        Args:
            existing_problem: Name of existing problem
            count: Number of examples to add
            difficulty: Specific difficulty tier to focus on
            balance: Whether to distribute evenly across tiers
            edge_cases: Whether to focus on edge cases

        Returns:
            List of generated examples
        """
        # Analyze existing problem to understand patterns
        try:
            from ..langchain_problems import get_problem_class
        except ImportError:
            try:
                from langchain_problems import get_problem_class
            except ImportError:
                from examples.langchain_problems import get_problem_class

        problem_class = get_problem_class(existing_problem)
        problem_instance = problem_class()
        existing_dataset = problem_instance.get_dataset()

        # Analyze existing examples
        existing_analysis = await self._analyze_existing_examples(
            existing_dataset.examples
        )

        # Generate new examples following the patterns
        if balance:
            distribution = self._balance_distribution(
                count, existing_analysis["difficulty_distribution"]
            )
        elif difficulty:
            distribution = {difficulty: count}
        else:
            # Follow existing distribution pattern
            distribution = self._follow_existing_distribution(
                count, existing_analysis["difficulty_distribution"]
            )

        examples = []

        for tier, tier_count in distribution.items():
            if tier_count > 0:
                tier_examples = await self._generate_examples_matching_pattern(
                    existing_analysis=existing_analysis,
                    difficulty=tier,
                    count=tier_count,
                    edge_cases=edge_cases,
                )
                examples.extend(tier_examples)

        return examples

    def _get_default_tiers(self, difficulty: str) -> List[str]:
        """Get default difficulty tiers for a difficulty level."""
        tier_sets = {
            "Beginner": ["easy", "medium", "hard"],
            "Advanced": ["easy", "medium", "hard", "very_hard"],
            "Expert": ["easy", "medium", "hard", "very_hard", "expert"],
        }
        return tier_sets.get(difficulty, tier_sets["Advanced"])

    def _calculate_distribution(self, count: int, tiers: List[str]) -> Dict[str, int]:
        """Calculate how many examples to generate for each difficulty tier."""
        if len(tiers) == 0:
            return {"medium": count}

        # Standard distribution: more examples in middle tiers
        if len(tiers) == 3:  # easy, medium, hard
            ratios = [0.3, 0.4, 0.3]
        elif len(tiers) == 4:  # easy, medium, hard, very_hard
            ratios = [0.25, 0.35, 0.25, 0.15]
        elif len(tiers) == 5:  # easy, medium, hard, very_hard, expert
            ratios = [0.2, 0.3, 0.25, 0.15, 0.1]
        else:
            # Equal distribution for other cases
            ratios = [1.0 / len(tiers)] * len(tiers)

        distribution = {}
        remaining = count

        for i, tier in enumerate(tiers):
            if i == len(tiers) - 1:  # Last tier gets remainder
                distribution[tier] = remaining
            else:
                tier_count = int(count * ratios[i])
                distribution[tier] = tier_count
                remaining -= tier_count

        return distribution

    async def _generate_tier_examples(
        self,
        problem_type: str,
        domain: str,
        description: str,
        difficulty: str,
        count: int,
        insights: Optional[ProblemInsights],
    ) -> List[GeneratedExample]:
        """Generate examples for a specific difficulty tier."""
        examples = []
        difficulty_spec = self.difficulty_specs[difficulty]

        # Get domain-specific patterns
        domain_patterns = self.domain_knowledge.get_domain_patterns(domain)

        for _i in range(count):
            if problem_type == "classification":
                example = self._generate_classification_example(
                    domain,
                    description,
                    difficulty,
                    difficulty_spec,
                    domain_patterns,
                    insights,
                )
            elif problem_type == "generation":
                example = self._generate_generation_example(
                    domain,
                    description,
                    difficulty,
                    difficulty_spec,
                    domain_patterns,
                    insights,
                )
            elif problem_type == "analysis":
                example = self._generate_analysis_example(
                    domain,
                    description,
                    difficulty,
                    difficulty_spec,
                    domain_patterns,
                    insights,
                )
            elif problem_type == "structured":
                example = self._generate_structured_example(
                    domain,
                    description,
                    difficulty,
                    difficulty_spec,
                    domain_patterns,
                    insights,
                )
            else:
                # Default to classification
                example = self._generate_classification_example(
                    domain,
                    description,
                    difficulty,
                    difficulty_spec,
                    domain_patterns,
                    insights,
                )

            if example:
                examples.append(example)

        return examples

    def _generate_classification_example(
        self,
        domain: str,
        description: str,
        difficulty: str,
        difficulty_spec: Dict,
        domain_patterns: Dict,
        insights: Optional[ProblemInsights],
    ) -> GeneratedExample:
        """Generate a classification example."""
        # Get categories from insights or domain patterns
        categories = (
            insights.suggested_categories
            if insights
            else domain_patterns.get("categories", ["category_a", "category_b"])
        )

        if domain == "customer_service":
            return self._generate_customer_service_example(
                difficulty, difficulty_spec, categories
            )
        elif domain == "legal":
            return self._generate_legal_example(difficulty, difficulty_spec, categories)
        elif domain == "medical":
            return self._generate_medical_example(
                difficulty, difficulty_spec, categories
            )
        elif domain == "technical":
            return self._generate_technical_example(
                difficulty, difficulty_spec, categories
            )
        elif domain == "financial":
            return self._generate_financial_example(
                difficulty, difficulty_spec, categories
            )
        else:
            return self._generate_general_classification_example(
                difficulty, difficulty_spec, categories, description
            )

    def _generate_customer_service_example(
        self, difficulty: str, difficulty_spec: Dict, categories: List[str]
    ) -> GeneratedExample:
        """Generate customer service classification example."""
        templates = {
            "easy": [
                ("My order #{order_id} hasn't arrived yet", "shipping_inquiry"),
                ("I want to return this {product}, it doesn't fit", "return_request"),
                ("How do I reset my password?", "account_support"),
                ("What's your refund policy?", "policy_question"),
                ("The app crashes when I try to checkout", "technical_support"),
            ],
            "medium": [
                ("I was charged twice for order #{order_id}", "billing_issue"),
                ("Do you have this {product} in size {size}?", "product_inquiry"),
                ("My discount code isn't working", "technical_support"),
                ("Can I change my shipping address?", "order_modification"),
            ],
            "hard": [
                (
                    "I ordered last week but now it's cheaper, can you help?",
                    "billing_issue",
                ),
                ("The product page won't load and I can't buy it", "technical_support"),
                ("My package says delivered but I don't have it", "shipping_inquiry"),
            ],
            "very_hard": [
                (
                    "Your site keeps logging me out when I apply my employee discount",
                    "technical_support",
                ),
                (
                    "I need to update payment for a pending order that hasn't shipped",
                    "order_modification",
                ),
                (
                    "The tracking shows wrong address but my account is correct",
                    "shipping_inquiry",
                ),
            ],
            "expert": [
                (
                    "I ordered overseas but customs rejected it, can you help re-ship?",
                    "shipping_inquiry",
                ),
                (
                    "My account shows 3 items but I was charged for 2, should I worry?",
                    "billing_issue",
                ),
                (
                    "Store credit from previous return won't work in the system",
                    "technical_support",
                ),
            ],
        }

        template_data = templates.get(difficulty, templates["medium"])
        query_template, category = random.choice(template_data)

        # Fill in template variables
        query = query_template.format(
            order_id=random.randint(10000, 99999),
            product=random.choice(["shirt", "jacket", "shoes", "laptop", "phone"]),
            size=random.choice(["XS", "S", "M", "L", "XL"]),
        )

        reasoning = f"{difficulty.title()} customer service case requiring {category} classification"

        return GeneratedExample(
            input_data={"query": query},
            expected_output=category,
            difficulty=difficulty,
            reasoning=reasoning,
            metadata={"domain": "customer_service", "category": category},
        )

    def _generate_technical_example(
        self, difficulty: str, difficulty_spec: Dict, categories: List[str]
    ) -> GeneratedExample:
        """Generate technical support classification example."""
        templates = {
            "easy": [
                ("The application won't start after the update", "bug_report"),
                ("Can you add dark mode to the settings?", "feature_request"),
                ("The website loads slowly on mobile", "performance_issue"),
                ("The password reset email isn't working", "technical_support"),
                ("Documentation for the API is outdated", "documentation_update"),
            ],
            "medium": [
                (
                    "Memory usage spikes when processing large files",
                    "performance_issue",
                ),
                ("Users can't access the admin panel after login", "bug_report"),
                (
                    "Integration with third-party payment system fails randomly",
                    "bug_report",
                ),
                (
                    "Need better error messages for validation failures",
                    "feature_request",
                ),
            ],
            "hard": [
                (
                    "Database queries timeout during peak hours with concurrent users",
                    "performance_issue",
                ),
                (
                    "Race condition in payment processing causes duplicate charges",
                    "bug_report",
                ),
                (
                    "Need OAuth2 implementation with custom scope management",
                    "feature_request",
                ),
                (
                    "SSL certificate validation fails in production environment",
                    "security_concern",
                ),
            ],
            "very_hard": [
                (
                    "Memory leak in background worker causes gradual system degradation",
                    "performance_issue",
                ),
                (
                    "Cross-site scripting vulnerability in user-generated content",
                    "security_concern",
                ),
                (
                    "Distributed transaction rollback fails in microservices architecture",
                    "bug_report",
                ),
                (
                    "Need real-time collaboration features with conflict resolution",
                    "feature_request",
                ),
            ],
            "expert": [
                (
                    "Deadlock in distributed consensus algorithm under network partition",
                    "bug_report",
                ),
                (
                    "Zero-downtime migration strategy for breaking database schema changes",
                    "feature_request",
                ),
                (
                    "Advanced persistent threat detection in application layer",
                    "security_concern",
                ),
                (
                    "Optimize consensus protocol for Byzantine fault tolerance",
                    "performance_issue",
                ),
            ],
        }

        template_data = templates.get(difficulty, templates["medium"])
        description, category = random.choice(template_data)

        reasoning = (
            f"{difficulty.title()} technical issue requiring {category} classification"
        )

        return GeneratedExample(
            input_data={
                "issue_description": description,
                "context": "technical_support",
            },
            expected_output=category,
            difficulty=difficulty,
            reasoning=reasoning,
            metadata={"domain": "technical", "category": category},
        )

    def _generate_medical_example(
        self, difficulty: str, difficulty_spec: Dict, categories: List[str]
    ) -> GeneratedExample:
        """Generate medical classification example."""
        templates = {
            "easy": [
                (
                    "Need prescription refill for blood pressure medication",
                    "prescription_request",
                ),
                ("Annual checkup appointment scheduling", "routine_checkup"),
                ("Chest pain and shortness of breath", "emergency"),
                ("Lab results review for cholesterol test", "test_results"),
                ("Referral needed for dermatologist", "specialist_referral"),
            ],
            "medium": [
                (
                    "Persistent headaches with vision changes over 2 weeks",
                    "specialist_referral",
                ),
                (
                    "Medication side effects causing nausea and dizziness",
                    "prescription_request",
                ),
                ("Blood sugar levels elevated in recent home testing", "test_results"),
                ("Follow-up needed after minor surgery last month", "routine_checkup"),
            ],
            "hard": [
                (
                    "Complex symptom pattern: fatigue, joint pain, and cognitive issues",
                    "specialist_referral",
                ),
                (
                    "Medication interaction concerns with new cardiac prescription",
                    "prescription_request",
                ),
                (
                    "Abnormal imaging results requiring specialist interpretation",
                    "test_results",
                ),
                ("Severe allergic reaction requiring immediate attention", "emergency"),
            ],
            "very_hard": [
                (
                    "Multi-system symptoms suggesting autoimmune condition",
                    "specialist_referral",
                ),
                (
                    "Rare medication dosing for pediatric patient with comorbidities",
                    "prescription_request",
                ),
                (
                    "Complex lab panel interpretation for suspected metabolic disorder",
                    "test_results",
                ),
                ("Critical care consultation for multi-organ failure", "emergency"),
            ],
            "expert": [
                (
                    "Differential diagnosis for rare genetic syndrome presentation",
                    "specialist_referral",
                ),
                (
                    "Experimental treatment protocol for refractory condition",
                    "prescription_request",
                ),
                ("Research-level biomarker analysis interpretation", "test_results"),
                (
                    "Trauma case with multiple critical injuries requiring coordination",
                    "emergency",
                ),
            ],
        }

        template_data = templates.get(difficulty, templates["medium"])
        description, category = random.choice(template_data)

        reasoning = (
            f"{difficulty.title()} medical case requiring {category} classification"
        )

        return GeneratedExample(
            input_data={"patient_case": description, "urgency": "standard"},
            expected_output=category,
            difficulty=difficulty,
            reasoning=reasoning,
            metadata={"domain": "medical", "category": category},
        )

    def _generate_financial_example(
        self, difficulty: str, difficulty_spec: Dict, categories: List[str]
    ) -> GeneratedExample:
        """Generate financial classification example."""
        templates = {
            "easy": [
                (
                    "Personal loan application for debt consolidation",
                    "loan_application",
                ),
                ("Suspicious charge on credit card statement", "fraud_detection"),
                ("Car insurance claim for minor accident", "insurance_claim"),
                ("Investment advice for retirement planning", "investment_advice"),
                ("Credit score review and improvement tips", "credit_assessment"),
            ],
            "medium": [
                ("Small business loan for equipment purchase", "loan_application"),
                (
                    "Multiple unauthorized transactions across different merchants",
                    "fraud_detection",
                ),
                ("Property damage claim with disputed liability", "insurance_claim"),
                (
                    "Portfolio rebalancing for changing risk tolerance",
                    "investment_advice",
                ),
                ("Credit analysis for mortgage pre-approval", "credit_assessment"),
            ],
            "hard": [
                (
                    "Commercial real estate financing with multiple properties",
                    "loan_application",
                ),
                ("Identity theft with complex transaction patterns", "fraud_detection"),
                (
                    "Business interruption insurance claim calculation",
                    "insurance_claim",
                ),
                (
                    "Derivative instruments for institutional portfolio",
                    "investment_advice",
                ),
                (
                    "Complex debt restructuring for corporate client",
                    "credit_assessment",
                ),
            ],
            "very_hard": [
                (
                    "Structured financing for international acquisition",
                    "loan_application",
                ),
                ("Sophisticated money laundering scheme detection", "fraud_detection"),
                ("Catastrophic event insurance claim coordination", "insurance_claim"),
                (
                    "High-frequency trading algorithm risk assessment",
                    "investment_advice",
                ),
                ("Regulatory capital requirement calculation", "credit_assessment"),
            ],
            "expert": [
                (
                    "Cross-border leveraged buyout financing structure",
                    "loan_application",
                ),
                ("Advanced persistent financial threat detection", "fraud_detection"),
                (
                    "Systemic risk insurance modeling for market events",
                    "insurance_claim",
                ),
                (
                    "Quantitative strategies for alternative investments",
                    "investment_advice",
                ),
                ("Basel III compliance stress testing framework", "credit_assessment"),
            ],
        }

        template_data = templates.get(difficulty, templates["medium"])
        description, category = random.choice(template_data)

        reasoning = (
            f"{difficulty.title()} financial case requiring {category} classification"
        )

        return GeneratedExample(
            input_data={
                "request_description": description,
                "client_type": "individual",
            },
            expected_output=category,
            difficulty=difficulty,
            reasoning=reasoning,
            metadata={"domain": "financial", "category": category},
        )

    def _generate_legal_example(
        self, difficulty: str, difficulty_spec: Dict, categories: List[str]
    ) -> GeneratedExample:
        """Generate legal analysis example."""
        contract_types = ["employment", "service", "purchase", "rental", "partnership"]
        risk_types = [
            "liability",
            "compliance",
            "financial",
            "operational",
            "regulatory",
        ]

        templates = {
            "easy": [
                "Review this {contract_type} contract for basic compliance issues",
                "Identify standard liability clauses in this {contract_type} agreement",
                "Check if this contract meets basic legal requirements",
            ],
            "medium": [
                "Analyze potential {risk_type} risks in this {contract_type} contract",
                "Review termination clauses and identify potential issues",
                "Assess intellectual property protection in this agreement",
            ],
            "hard": [
                "Evaluate complex indemnification clauses and cross-jurisdictional implications",
                "Analyze force majeure provisions in context of recent regulatory changes",
                "Review multi-party liability allocation with conflicting governing laws",
            ],
            "very_hard": [
                "Assess regulatory compliance across multiple jurisdictions with conflicting requirements",
                "Analyze complex derivative liability chains in multi-tier service agreements",
                "Review contingent liability provisions with variable regulatory frameworks",
            ],
            "expert": [
                "Evaluate cross-border data privacy compliance with evolving GDPR interpretations",
                "Analyze sophisticated IP licensing with complex royalty calculation methods",
                "Review complex merger agreement with regulatory approval contingencies",
            ],
        }

        template = random.choice(templates.get(difficulty, templates["medium"]))
        contract_text = template.format(
            contract_type=random.choice(contract_types),
            risk_type=random.choice(risk_types),
        )

        # Generate expected analysis result
        analysis_result = {
            "risk_level": random.choice(["low", "medium", "high"]),
            "compliance_status": random.choice(
                ["compliant", "needs_review", "non_compliant"]
            ),
            "key_issues": [f"Issue related to {random.choice(risk_types)}"],
        }

        reasoning = f"{difficulty.title()} legal analysis requiring expertise in contract review"

        return GeneratedExample(
            input_data={
                "contract_text": contract_text,
                "analysis_type": "risk_assessment",
            },
            expected_output=analysis_result,
            difficulty=difficulty,
            reasoning=reasoning,
            metadata={
                "domain": "legal",
                "contract_type": random.choice(contract_types),
            },
        )

    def _generate_general_classification_example(
        self,
        difficulty: str,
        difficulty_spec: Dict,
        categories: List[str],
        description: str,
    ) -> GeneratedExample:
        """Generate a general classification example."""
        # Extract keywords from description for context
        keywords = re.findall(r"\b\w+\b", description.lower())
        content_words = [
            w
            for w in keywords
            if len(w) > 3
            and w
            not in {"this", "that", "with", "from", "they", "have", "will", "been"}
        ]

        category = random.choice(categories)

        # Generate text based on difficulty
        if difficulty in ["easy", "medium"]:
            text = f"This is a clear example of {category} in the context of {random.choice(content_words) if content_words else 'general topic'}."
        elif difficulty == "hard":
            text = f"This example involves {category} but also touches on aspects that might be confused with other categories, requiring careful analysis of {random.choice(content_words) if content_words else 'context'}."
        else:  # very_hard, expert
            text = f"A complex case that primarily represents {category} but contains multiple confounding factors including ambiguous indicators and edge cases that challenge standard classification approaches."

        reasoning = (
            f"{difficulty.title()} example designed to test {category} classification"
        )

        return GeneratedExample(
            input_data={"text": text},
            expected_output=category,
            difficulty=difficulty,
            reasoning=reasoning,
            metadata={"domain": "general", "category": category},
        )

    def _generate_generation_example(
        self,
        domain: str,
        description: str,
        difficulty: str,
        difficulty_spec: Dict,
        domain_patterns: Dict,
        insights: Optional[ProblemInsights],
    ) -> GeneratedExample:
        """Generate a text generation example."""

        # Domain-specific generation templates
        generation_templates = {
            "medical": {
                "easy": [
                    (
                        "Write a simple patient discharge instruction for a sprained ankle",
                        "Rest your ankle for the next 48 hours. Apply ice for 20 minutes every 2-3 hours. Keep your ankle elevated when sitting. Take over-the-counter pain medication as directed. Return if pain worsens or swelling increases.",
                    ),
                    (
                        "Create a brief medication reminder for daily blood pressure medicine",
                        "Take your blood pressure medication every morning with breakfast. Set a daily alarm to help you remember. Keep medication in a visible place. Track doses on a calendar. Never skip doses without consulting your doctor.",
                    ),
                ],
                "medium": [
                    (
                        "Generate a patient education summary about managing Type 2 diabetes",
                        "Type 2 diabetes management involves monitoring blood sugar levels regularly, following a balanced diet low in simple sugars, exercising at least 30 minutes daily, taking prescribed medications consistently, and attending regular check-ups. Work with your healthcare team to set personalized goals and track your progress.",
                    ),
                    (
                        "Write post-operative care instructions for minor surgery",
                        "Keep the incision site clean and dry for 48 hours. Change dressings daily using sterile gauze. Watch for signs of infection: increased redness, swelling, or discharge. Take prescribed pain medication as needed. Avoid strenuous activity for one week. Follow up with your surgeon in 7-10 days.",
                    ),
                ],
                "hard": [
                    (
                        "Create a comprehensive treatment plan summary for a patient with multiple chronic conditions",
                        "Integrated care plan for hypertension, diabetes, and COPD: Monitor blood pressure twice daily, maintaining below 130/80. Check blood glucose before meals, target 80-130 mg/dL. Use inhaler as prescribed, monitoring peak flow daily. Coordinate medications to avoid interactions. Low-sodium, diabetic-friendly diet with 6 small meals. Gentle exercise program with cardiac rehabilitation. Monthly lab work and quarterly specialist visits. Emergency action plan for exacerbations.",
                    )
                ],
            },
            "customer_service": {
                "easy": [
                    (
                        "Write a friendly response to a customer asking about store hours",
                        "Thank you for reaching out! Our store is open Monday through Friday from 9 AM to 8 PM, Saturday from 10 AM to 6 PM, and Sunday from 12 PM to 5 PM. We look forward to seeing you soon! Is there anything else I can help you with today?",
                    ),
                    (
                        "Create an order confirmation message",
                        "Your order #12345 has been confirmed! We'll process it within 1-2 business days and send you a tracking number once it ships. Expected delivery is 3-5 business days. Thank you for your purchase!",
                    ),
                ],
                "medium": [
                    (
                        "Draft a response to a customer complaint about delayed shipping",
                        "I sincerely apologize for the delay in your order delivery. I've checked your order #54321 and see it's been delayed due to unexpected high demand. I've expedited your shipment at no extra cost, and it should arrive within 2 business days. As an apology, I've added a 20% discount code to your account for your next purchase. Your satisfaction is important to us.",
                    ),
                    (
                        "Write a product replacement offer for a defective item",
                        "I'm sorry to hear about the issue with your product. We stand behind our quality and want to make this right. I can offer you: 1) A free replacement shipped today with expedited delivery, 2) A full refund upon return of the item, or 3) Store credit with an additional 15% bonus. Please let me know which option works best for you.",
                    ),
                ],
                "hard": [
                    (
                        "Compose a detailed response addressing multiple service failures and retention offer",
                        "I want to personally apologize for the series of issues you've experienced. After reviewing your account, I see the delayed shipment, incorrect item, and poor initial support response. This falls short of our standards. I've taken several actions: issued a full refund while you keep the item, applied a $50 credit to your account, upgraded you to VIP status with free expedited shipping for a year, and assigned a dedicated support representative. I'll personally monitor your future orders to ensure excellent service.",
                    )
                ],
            },
            "technical": {
                "easy": [
                    (
                        "Write a simple error message for a failed login attempt",
                        "Login failed. Please check your username and password and try again. If you've forgotten your password, click 'Reset Password' below.",
                    ),
                    (
                        "Create a success notification for file upload",
                        "File uploaded successfully! Your document 'report.pdf' (2.3 MB) has been saved to your Documents folder.",
                    ),
                ],
                "medium": [
                    (
                        "Generate API documentation for a user authentication endpoint",
                        "POST /api/auth/login - Authenticates user credentials and returns access token. Request body: {username: string, password: string}. Success response: {token: string, expires_in: number, user: {id: string, email: string}}. Error responses: 401 for invalid credentials, 429 for rate limiting. Token should be included in subsequent requests as Bearer token in Authorization header.",
                    ),
                    (
                        "Write a technical troubleshooting guide for connection issues",
                        "Connection Troubleshooting: 1) Check internet connectivity with ping test. 2) Verify firewall isn't blocking ports 80/443. 3) Clear DNS cache (ipconfig /flushdns). 4) Try alternate DNS servers (8.8.8.8). 5) Disable proxy settings if not required. 6) Check for VPN conflicts. 7) Review system logs for specific error codes. If issues persist, collect network trace and contact support.",
                    ),
                ],
                "hard": [
                    (
                        "Create comprehensive deployment documentation for a microservices architecture",
                        "Microservices Deployment Guide: Prerequisites: Kubernetes 1.24+, Helm 3.x, Docker Registry. Architecture: 5 services (Auth, API Gateway, Core Service, Analytics, Notification). Deployment steps: 1) Configure namespace and RBAC. 2) Deploy PostgreSQL and Redis using Helm charts. 3) Apply ConfigMaps and Secrets for each service. 4) Deploy services in order: Auth → Core → Analytics → Notification → Gateway. 5) Configure Ingress with TLS certificates. 6) Set up horizontal pod autoscaling. 7) Configure Prometheus monitoring and Grafana dashboards. 8) Implement circuit breakers and retry policies. 9) Set up distributed tracing with Jaeger. Post-deployment: Run smoke tests, verify health endpoints, check logs aggregation in ELK stack.",
                    )
                ],
            },
            "financial": {
                "easy": [
                    (
                        "Write a simple savings account balance notification",
                        "Your savings account balance as of today is $5,432.10. You've earned $12.35 in interest this month. Thank you for banking with us!",
                    ),
                    (
                        "Create a payment reminder message",
                        "Friendly reminder: Your credit card payment of $245.00 is due on March 15th. You can pay online, through our mobile app, or by calling 1-800-XXX-XXXX.",
                    ),
                ],
                "medium": [
                    (
                        "Generate an investment portfolio summary for a conservative investor",
                        "Portfolio Summary: Total Value: $75,000. Allocation: 60% Bonds ($45,000) with 3.2% average yield, 30% Blue-chip Stocks ($22,500) with 2.1% dividend yield, 10% Money Market ($7,500) at 2.5% APY. Year-to-date return: 4.8%. Risk level: Conservative. Recommended action: Consider rebalancing stocks portion as you're slightly overweight due to recent gains.",
                    ),
                    (
                        "Write a loan pre-approval letter with conditions",
                        "Congratulations! You're pre-approved for a personal loan up to $25,000 at 7.99% APR. This offer is valid for 30 days and subject to: verification of income documentation, maintenance of current credit score (740+), and debt-to-income ratio below 35%. Fixed monthly payments would be $486 for a 60-month term. No prepayment penalties.",
                    ),
                ],
                "hard": [
                    (
                        "Create a detailed financial planning recommendation for retirement",
                        "Retirement Planning Analysis: Based on your age (45), current savings ($450,000), and desired retirement at 65 with $80,000 annual income need: Recommended strategy: Maximize 401(k) contributions ($22,500/year), contribute $6,500 to Roth IRA, invest additional $15,000/year in taxable accounts. Asset allocation: Shift from 70/30 stocks/bonds to 60/40 over next 10 years. Target 7% average return. Consider tax-loss harvesting in taxable accounts. At current trajectory with recommendations, projected retirement savings: $1.8M, generating $72,000/year at 4% withdrawal rate. Gap of $8,000/year can be covered by part-time work or reduced expenses. Review annually and adjust for market conditions.",
                    )
                ],
            },
            "legal": {
                "easy": [
                    (
                        "Draft a simple confidentiality notice for email",
                        "CONFIDENTIALITY NOTICE: This email and any attachments are confidential and intended solely for the addressee. If you received this in error, please notify the sender and delete all copies. Unauthorized use or disclosure is prohibited.",
                    ),
                    (
                        "Write a basic disclaimer for a website",
                        "The information on this website is for general informational purposes only. We make no representations or warranties about the accuracy or completeness of any information on this site. Use of this site is at your own risk.",
                    ),
                ],
                "medium": [
                    (
                        "Generate a standard force majeure clause",
                        "Neither party shall be liable for any failure or delay in performance under this Agreement due to circumstances beyond its reasonable control, including but not limited to acts of God, natural disasters, war, terrorism, labor disputes, government actions, or pandemic. The affected party must promptly notify the other party and use reasonable efforts to mitigate the impact. If the force majeure event continues for more than 60 days, either party may terminate this Agreement upon written notice.",
                    ),
                    (
                        "Create a limitation of liability provision",
                        "To the maximum extent permitted by law, neither party shall be liable for any indirect, incidental, special, consequential, or punitive damages, including lost profits, lost revenue, or lost data, arising out of or related to this agreement, regardless of the theory of liability and whether or not such party has been advised of the possibility of such damages. Each party's total cumulative liability shall not exceed the total fees paid under this agreement in the twelve months preceding the claim.",
                    ),
                ],
                "hard": [
                    (
                        "Draft a complex intellectual property assignment clause with carve-outs",
                        "Employee hereby assigns to Company all right, title, and interest in any Inventions conceived, developed, or reduced to practice during employment that: (i) relate to Company's business or anticipated research; (ii) result from use of Company resources; or (iii) are developed during work hours. This assignment excludes: (a) Inventions developed entirely on Employee's own time without Company resources, provided they don't relate to Company's business; (b) Pre-existing Inventions listed in Exhibit A; (c) Inventions covered by California Labor Code Section 2870. Employee agrees to promptly disclose all Inventions, execute assignment documents, and assist in patent prosecution. Company grants Employee a non-exclusive license to academic publications derived from assigned Inventions, subject to Company's review for confidential information. Moral rights are waived to the extent permitted by law.",
                    )
                ],
            },
        }

        # Get domain-specific templates or use general ones
        domain_templates = generation_templates.get(
            domain,
            {
                "easy": [
                    (
                        f"Write a simple {domain} message",
                        f"This is a clear, concise {domain} message that addresses the basic requirements with appropriate terminology and tone for the {domain} field.",
                    ),
                    (
                        f"Create a brief {domain} summary",
                        f"A well-structured {domain} summary that captures the essential points while maintaining professional standards expected in {domain} communications.",
                    ),
                ],
                "medium": [
                    (
                        f"Generate a detailed {domain} report section",
                        f"This comprehensive {domain} report section provides in-depth analysis with specific details relevant to {domain} professionals. It includes key findings, supporting data, and actionable recommendations based on {domain} best practices.",
                    ),
                    (
                        f"Write a professional {domain} proposal",
                        f"A professionally crafted {domain} proposal that outlines objectives, methodology, timeline, and expected outcomes. The proposal demonstrates deep understanding of {domain} requirements and presents a compelling case for the recommended approach.",
                    ),
                ],
                "hard": [
                    (
                        f"Create a complex {domain} analysis with multiple considerations",
                        f"This sophisticated {domain} analysis examines multiple interconnected factors, weighs competing priorities, and synthesizes complex information into actionable insights. The analysis demonstrates advanced {domain} expertise by addressing nuanced challenges, considering edge cases, and providing strategic recommendations based on comprehensive evaluation of all relevant factors.",
                    )
                ],
            },
        )

        # Get templates for the difficulty level
        difficulty_templates = domain_templates.get(
            difficulty, domain_templates.get("medium", [])
        )

        if difficulty_templates:
            prompt, expected_output = random.choice(difficulty_templates)
        else:
            # Fallback
            prompt = f"Generate {domain} content for {description}"
            expected_output = f"Well-crafted {domain} content that addresses the specific requirements of {description} with appropriate complexity for {difficulty} level tasks."

        return GeneratedExample(
            input_data={"prompt": prompt},
            expected_output=expected_output,
            difficulty=difficulty,
            reasoning=f"{difficulty.title()} {domain} generation task with realistic expected output",
            metadata={"domain": domain, "type": "generation"},
        )

    def _generate_analysis_example(
        self,
        domain: str,
        description: str,
        difficulty: str,
        difficulty_spec: Dict,
        domain_patterns: Dict,
        insights: Optional[ProblemInsights],
    ) -> GeneratedExample:
        """Generate an analysis example."""

        # Domain-specific analysis examples
        analysis_templates = {
            "medical": {
                "easy": [
                    (
                        "Patient presents with fever (101°F), sore throat, and fatigue for 3 days. No other symptoms.",
                        {
                            "summary": "Likely viral upper respiratory infection",
                            "key_points": [
                                "Mild fever indicates infection",
                                "Localized symptoms suggest upper respiratory involvement",
                                "Short duration and limited symptoms suggest viral etiology",
                            ],
                            "recommendations": [
                                "Rest and hydration",
                                "Symptomatic treatment",
                                "Return if symptoms worsen",
                            ],
                        },
                    )
                ],
                "medium": [
                    (
                        "45-year-old patient with BMI 32, fasting glucose 118 mg/dL, blood pressure 138/88, sedentary lifestyle, family history of diabetes.",
                        {
                            "summary": "Pre-diabetic with metabolic syndrome risk factors",
                            "key_points": [
                                "Impaired fasting glucose (100-125 mg/dL range)",
                                "Stage 1 hypertension",
                                "Obesity class I",
                                "Multiple cardiovascular risk factors present",
                            ],
                            "recommendations": [
                                "Lifestyle modification program",
                                "Weight loss target 5-10%",
                                "Regular glucose monitoring",
                                "Consider metformin if lifestyle changes insufficient",
                            ],
                            "risk_assessment": "High risk for Type 2 diabetes development within 5 years",
                        },
                    )
                ],
                "hard": [
                    (
                        "Complex case: 68-year-old with CHF (EF 35%), CKD stage 3, diabetes with recent HbA1c 8.2%, new onset peripheral edema, creatinine increased from 1.8 to 2.3 over 2 weeks, K+ 5.1, on ACE-I, metformin, furosemide.",
                        {
                            "summary": "Acute on chronic kidney disease with multiple comorbidities requiring immediate intervention",
                            "key_points": [
                                "Worsening renal function suggests acute kidney injury on CKD",
                                "Hyperkalemia risk with ACE-I in setting of renal dysfunction",
                                "Metformin contraindicated with eGFR likely <30",
                                "Volume status complex given CHF and renal disease",
                            ],
                            "immediate_actions": [
                                "Hold ACE-I and metformin",
                                "Check volume status clinically",
                                "Renal ultrasound to rule out obstruction",
                                "Nephrology consultation",
                            ],
                            "differential": [
                                "Prerenal AKI from CHF",
                                "Medication-induced",
                                "Diabetic nephropathy progression",
                            ],
                            "monitoring": [
                                "Daily creatinine and electrolytes",
                                "Strict I/O",
                                "Daily weights",
                            ],
                        },
                    )
                ],
            },
            "financial": {
                "easy": [
                    (
                        "Monthly income: $5,000, Fixed expenses: $3,500, Variable expenses: $800, Current savings: $10,000",
                        {
                            "summary": "Positive cash flow with opportunity for increased savings",
                            "key_points": [
                                "Monthly surplus of $700",
                                "Expense ratio 86%",
                                "Emergency fund covers 2.5 months",
                            ],
                            "recommendations": [
                                "Increase emergency fund to 6 months",
                                "Automate savings of $500/month",
                                "Review variable expenses for optimization",
                            ],
                        },
                    )
                ],
                "medium": [
                    (
                        "Portfolio: $250k stocks (60%), $100k bonds (24%), $50k international (12%), $17k cash (4%). Age 52, risk tolerance moderate, retirement goal at 65.",
                        {
                            "summary": "Well-diversified portfolio slightly aggressive for age and timeline",
                            "key_points": [
                                "Equity allocation high for 13-year timeline",
                                "International exposure appropriate",
                                "Bond allocation below age-based recommendation",
                                "Adequate cash reserves",
                            ],
                            "recommendations": [
                                "Consider shifting 10% from stocks to bonds",
                                "Implement glide path strategy",
                                "Review expense ratios",
                                "Consider target-date fund option",
                            ],
                            "risk_metrics": {
                                "volatility": "moderate-high",
                                "expected_return": "6-8%",
                            },
                        },
                    )
                ],
                "hard": [
                    (
                        "Business valuation: Revenue $5M growing 15% YoY, EBITDA $800k, industry multiple 4-6x, comparable transaction $4.5M at 5.5x, strong customer concentration (top 3 = 60%), recurring revenue 40%, proprietary technology.",
                        {
                            "summary": "Business valued between $3.2M-$4.8M with adjustments for risk factors",
                            "key_points": [
                                "Base valuation $3.2M-$4.8M using EBITDA multiple",
                                "Customer concentration presents significant risk",
                                "Recurring revenue and proprietary tech add premium",
                                "Growth rate supports higher multiple within range",
                            ],
                            "valuation_breakdown": {
                                "base_value": "$4M (5x EBITDA)",
                                "adjustments": {
                                    "customer_concentration": "-15%",
                                    "recurring_revenue": "+10%",
                                    "growth_premium": "+10%",
                                    "proprietary_assets": "+5%",
                                },
                                "adjusted_value": "$4.4M",
                            },
                            "risks": [
                                "Customer dependency",
                                "Scalability questions",
                                "Tech obsolescence",
                            ],
                            "opportunities": [
                                "Diversify customer base",
                                "Increase recurring revenue",
                                "Geographic expansion",
                            ],
                        },
                    )
                ],
            },
            "technical": {
                "easy": [
                    (
                        "Error log shows: 'Connection timeout after 30s' occurring every hour at :15 mark. Database query taking 45s on average.",
                        {
                            "summary": "Database performance issue causing timeouts",
                            "key_points": [
                                "Query exceeds timeout threshold",
                                "Pattern suggests scheduled job",
                                "Consistent timing indicates specific trigger",
                            ],
                            "recommendations": [
                                "Index optimization needed",
                                "Increase timeout temporarily",
                                "Profile the specific query",
                            ],
                        },
                    )
                ],
                "medium": [
                    (
                        "System metrics: CPU spikes to 95% during peak hours, memory usage steady at 70%, response time increases from 200ms to 2s, concurrent users average 5000, database connections exhausted (100/100).",
                        {
                            "summary": "Database connection pooling bottleneck under load",
                            "key_points": [
                                "CPU spikes correlate with connection exhaustion",
                                "Memory usage indicates no memory leak",
                                "Response time degradation 10x under load",
                                "Connection pool size insufficient for user load",
                            ],
                            "recommendations": [
                                "Increase connection pool to 200",
                                "Implement connection multiplexing",
                                "Add caching layer to reduce DB hits",
                                "Consider read replicas for scaling",
                            ],
                            "metrics_analysis": {
                                "bottleneck": "database",
                                "scaling_needed": "horizontal",
                            },
                        },
                    )
                ],
                "hard": [
                    (
                        "Microservices issue: Service A (99.9% uptime) calls Service B (99.5% uptime) which calls Service C (99.8% uptime). End-to-end availability 97.2%. Cascade failures observed. Circuit breaker triggers but recovery slow. Distributed tracing shows 15% of requests have latency >5s due to retries.",
                        {
                            "summary": "Cascading failure pattern with insufficient resilience mechanisms",
                            "key_points": [
                                "Compound availability (99.9% × 99.5% × 99.8% = 99.2%) not matching observed 97.2%",
                                "Additional 2% loss from cascade failures and retry storms",
                                "Circuit breaker recovery time too conservative",
                                "Retry logic causing latency amplification",
                            ],
                            "root_cause_analysis": {
                                "primary": "Service B reliability below requirements",
                                "secondary": "Retry storm amplification",
                                "tertiary": "Circuit breaker configuration suboptimal",
                            },
                            "recommendations": [
                                "Implement bulkhead pattern to isolate failures",
                                "Add Service B redundancy or improve to 99.9%",
                                "Tune circuit breaker: faster recovery, gradual ramp",
                                "Implement adaptive retry with exponential backoff",
                                "Add request hedging for critical paths",
                                "Consider service mesh for advanced traffic management",
                            ],
                            "architecture_changes": [
                                "Add caching between A and B",
                                "Implement async patterns where possible",
                                "Create fallback mechanisms",
                            ],
                        },
                    )
                ],
            },
        }

        # Get domain-specific templates or use general ones
        domain_templates = analysis_templates.get(domain, {})

        if not domain_templates:
            # General analysis templates
            if difficulty == "easy":
                document = f"Simple {domain} report showing basic metrics and standard operations."
                analysis_result = {
                    "summary": f"Standard {domain} operations within normal parameters",
                    "key_points": [
                        f"All {domain} metrics within expected ranges",
                        "No anomalies detected",
                        "Routine patterns observed",
                    ],
                    "recommendations": [
                        "Continue standard monitoring",
                        "No immediate action required",
                    ],
                }
            elif difficulty == "medium":
                document = f"Detailed {domain} analysis with multiple data points showing some variations from baseline."
                analysis_result = {
                    "summary": f"Mixed {domain} indicators requiring attention",
                    "key_points": [
                        f"Several {domain} metrics showing deviation",
                        "Trend analysis indicates potential issues",
                        "Correlation found between multiple factors",
                        "Intervention recommended",
                    ],
                    "recommendations": [
                        "Implement corrective measures",
                        "Increase monitoring frequency",
                        "Review related processes",
                    ],
                    "trends": {"direction": "concerning", "confidence": "moderate"},
                }
            else:  # hard
                document = f"Complex {domain} scenario with multiple interrelated issues, conflicting indicators, and systemic challenges requiring expert analysis."
                analysis_result = {
                    "summary": f"Critical {domain} situation requiring immediate comprehensive intervention",
                    "key_points": [
                        f"Multiple cascading {domain} failures identified",
                        "Complex interdependencies affecting outcomes",
                        "Risk of system-wide impact",
                        "Several mitigation strategies needed simultaneously",
                        "Long-term structural changes required",
                    ],
                    "root_causes": [
                        "Systemic design limitations",
                        "Accumulated technical debt",
                        "Resource constraints",
                    ],
                    "recommendations": [
                        "Immediate stabilization measures",
                        "Phased remediation plan",
                        "Architecture redesign",
                        "Resource reallocation",
                    ],
                    "risk_assessment": "critical",
                    "timeline": "immediate action required",
                }
        else:
            # Use domain-specific template
            difficulty_templates = domain_templates.get(
                difficulty, domain_templates.get("medium", [])
            )
            if difficulty_templates:
                document, analysis_result = random.choice(difficulty_templates)
            else:
                document = f"{domain} document for analysis"
                analysis_result = {
                    "summary": f"{domain} analysis",
                    "key_points": ["Analysis complete"],
                }

        return GeneratedExample(
            input_data={"document": document, "analysis_type": "comprehensive"},
            expected_output=analysis_result,
            difficulty=difficulty,
            reasoning=f"{difficulty.title()} {domain} analysis task with detailed expected output",
            metadata={"domain": domain, "type": "analysis"},
        )

    def _generate_structured_example(
        self,
        domain: str,
        description: str,
        difficulty: str,
        difficulty_spec: Dict,
        domain_patterns: Dict,
        insights: Optional[ProblemInsights],
    ) -> GeneratedExample:
        """Generate a structured extraction example."""

        # Domain-specific structured extraction examples
        extraction_templates = {
            "medical": {
                "easy": [
                    (
                        "Patient John Smith, age 45, diagnosed with hypertension on March 15, 2024. Prescribed lisinopril 10mg daily.",
                        {
                            "entities": [
                                {"type": "patient", "value": "John Smith"},
                                {"type": "age", "value": "45"},
                                {"type": "condition", "value": "hypertension"},
                                {"type": "medication", "value": "lisinopril"},
                                {"type": "dosage", "value": "10mg daily"},
                            ],
                            "dates": [{"event": "diagnosis", "date": "2024-03-15"}],
                            "relationships": [
                                {
                                    "patient": "John Smith",
                                    "has_condition": "hypertension",
                                }
                            ],
                        },
                    )
                ],
                "medium": [
                    (
                        "Mrs. Johnson (DOB: 05/12/1978) presented with complaints of chest pain radiating to left arm, onset 2 hours ago. EKG shows ST elevation in leads II, III, aVF. Troponin elevated at 2.5 ng/mL. Administered aspirin 325mg, nitroglycerin 0.4mg sublingual, and started heparin drip. Cardiology consulted for urgent catheterization.",
                        {
                            "patient": {"name": "Mrs. Johnson", "dob": "1978-05-12"},
                            "symptoms": [
                                {
                                    "type": "chest pain",
                                    "characteristics": ["radiating to left arm"],
                                    "onset": "2 hours ago",
                                }
                            ],
                            "diagnostics": [
                                {
                                    "test": "EKG",
                                    "finding": "ST elevation",
                                    "location": "leads II, III, aVF",
                                },
                                {
                                    "test": "Troponin",
                                    "value": "2.5",
                                    "unit": "ng/mL",
                                    "status": "elevated",
                                },
                            ],
                            "medications_administered": [
                                {"drug": "aspirin", "dose": "325mg", "route": "oral"},
                                {
                                    "drug": "nitroglycerin",
                                    "dose": "0.4mg",
                                    "route": "sublingual",
                                },
                                {
                                    "drug": "heparin",
                                    "dose": "continuous",
                                    "route": "IV drip",
                                },
                            ],
                            "clinical_impression": "Acute inferior wall myocardial infarction",
                            "actions": [
                                {
                                    "type": "consultation",
                                    "service": "cardiology",
                                    "urgency": "urgent",
                                }
                            ],
                        },
                    )
                ],
            },
            "financial": {
                "easy": [
                    (
                        "Transaction: $1,250.00 paid to ABC Corporation on 01/15/2024 for invoice #INV-2024-001. Payment method: Wire transfer.",
                        {
                            "transaction": {
                                "amount": 1250.00,
                                "currency": "USD",
                                "date": "2024-01-15",
                                "type": "payment",
                            },
                            "parties": {
                                "payer": "unspecified",
                                "payee": "ABC Corporation",
                            },
                            "reference": {"invoice_number": "INV-2024-001"},
                            "payment_method": "wire transfer",
                        },
                    )
                ],
                "medium": [
                    (
                        "Q3 2024 Financial Summary: Total Revenue: $2.5M (up 12% YoY), Operating Expenses: $1.8M, EBITDA: $700K (28% margin). Major expense categories: Salaries $980K (54%), Marketing $360K (20%), R&D $280K (16%), Other $180K (10%). Cash position: $1.2M. Accounts receivable: $450K (45 days average).",
                        {
                            "period": {"quarter": "Q3", "year": 2024},
                            "revenue": {"amount": 2500000, "growth_yoy": 0.12},
                            "expenses": {
                                "total": 1800000,
                                "breakdown": [
                                    {
                                        "category": "Salaries",
                                        "amount": 980000,
                                        "percentage": 0.54,
                                    },
                                    {
                                        "category": "Marketing",
                                        "amount": 360000,
                                        "percentage": 0.20,
                                    },
                                    {
                                        "category": "R&D",
                                        "amount": 280000,
                                        "percentage": 0.16,
                                    },
                                    {
                                        "category": "Other",
                                        "amount": 180000,
                                        "percentage": 0.10,
                                    },
                                ],
                            },
                            "profitability": {"ebitda": 700000, "ebitda_margin": 0.28},
                            "balance_sheet": {
                                "cash": 1200000,
                                "accounts_receivable": 450000,
                                "days_sales_outstanding": 45,
                            },
                        },
                    )
                ],
            },
            "legal": {
                "easy": [
                    (
                        "This Agreement is entered into on January 1, 2024, between TechCorp Inc., a Delaware corporation ('Company'), and John Doe, an individual residing at 123 Main St, Anytown, ST 12345 ('Contractor').",
                        {
                            "document_type": "agreement",
                            "effective_date": "2024-01-01",
                            "parties": [
                                {
                                    "name": "TechCorp Inc.",
                                    "type": "corporation",
                                    "jurisdiction": "Delaware",
                                    "role": "Company",
                                },
                                {
                                    "name": "John Doe",
                                    "type": "individual",
                                    "address": "123 Main St, Anytown, ST 12345",
                                    "role": "Contractor",
                                },
                            ],
                        },
                    )
                ],
                "medium": [
                    (
                        "WHEREAS, Company desires to acquire substantially all assets of Target Corporation ('Target'), a California corporation, for a purchase price of $50,000,000 (the 'Purchase Price'), subject to adjustments for working capital as set forth in Section 2.3, with $5,000,000 held in escrow for 18 months to satisfy indemnification claims, and closing contingent upon regulatory approval from the FTC no later than June 30, 2024.",
                        {
                            "transaction_type": "asset_acquisition",
                            "buyer": "Company",
                            "seller": {
                                "name": "Target Corporation",
                                "type": "corporation",
                                "jurisdiction": "California",
                            },
                            "financial_terms": {
                                "purchase_price": 50000000,
                                "currency": "USD",
                                "adjustments": ["working capital"],
                                "escrow": {
                                    "amount": 5000000,
                                    "duration_months": 18,
                                    "purpose": "indemnification claims",
                                },
                            },
                            "conditions": [
                                {
                                    "type": "regulatory_approval",
                                    "authority": "FTC",
                                    "deadline": "2024-06-30",
                                }
                            ],
                            "references": [
                                {
                                    "section": "2.3",
                                    "topic": "working capital adjustments",
                                }
                            ],
                        },
                    )
                ],
            },
            "technical": {
                "easy": [
                    (
                        "GET /api/users/123 HTTP/1.1 Host: api.example.com Authorization: Bearer abc123token Response: 200 OK {id: 123, name: 'John Doe', email: 'john@example.com'}",
                        {
                            "request": {
                                "method": "GET",
                                "endpoint": "/api/users/123",
                                "protocol": "HTTP/1.1",
                                "host": "api.example.com",
                                "headers": {"Authorization": "Bearer abc123token"},
                            },
                            "response": {
                                "status_code": 200,
                                "status_text": "OK",
                                "body": {
                                    "id": 123,
                                    "name": "John Doe",
                                    "email": "john@example.com",
                                },
                            },
                        },
                    )
                ],
                "medium": [
                    (
                        "2024-01-15T10:30:45.123Z ERROR [payment-service] PaymentProcessor.processPayment() - Transaction failed for orderId=ORD-789, userId=456, amount=$99.99. Exception: PaymentGatewayException: Connection timeout after 30s. Stack trace: at PaymentGateway.charge(PaymentGateway.java:156) at PaymentProcessor.processPayment(PaymentProcessor.java:78). Retry attempt 2/3 scheduled for 10:31:15Z.",
                        {
                            "log_entry": {
                                "timestamp": "2024-01-15T10:30:45.123Z",
                                "level": "ERROR",
                                "service": "payment-service",
                                "class": "PaymentProcessor",
                                "method": "processPayment",
                            },
                            "error_details": {
                                "message": "Transaction failed",
                                "transaction": {
                                    "order_id": "ORD-789",
                                    "user_id": "456",
                                    "amount": 99.99,
                                    "currency": "USD",
                                },
                                "exception": {
                                    "type": "PaymentGatewayException",
                                    "message": "Connection timeout after 30s",
                                    "stack_trace": [
                                        {
                                            "class": "PaymentGateway",
                                            "method": "charge",
                                            "line": 156,
                                        },
                                        {
                                            "class": "PaymentProcessor",
                                            "method": "processPayment",
                                            "line": 78,
                                        },
                                    ],
                                },
                            },
                            "retry_info": {
                                "attempt": 2,
                                "max_attempts": 3,
                                "next_retry": "2024-01-15T10:31:15Z",
                            },
                        },
                    )
                ],
            },
        }

        # Get domain-specific templates or use general ones
        domain_templates = extraction_templates.get(domain, {})

        if not domain_templates:
            # General extraction templates
            if difficulty == "easy":
                text = f"The {domain} report was submitted by John Smith on January 15, 2024. It contains 3 main sections and 10 recommendations."
                extracted_data = {
                    "entities": [
                        {"type": "person", "value": "John Smith"},
                        {"type": "date", "value": "2024-01-15"},
                        {"type": "document", "value": f"{domain} report"},
                    ],
                    "quantities": [
                        {"item": "sections", "count": 3},
                        {"item": "recommendations", "count": 10},
                    ],
                    "relationships": [
                        {
                            "subject": "John Smith",
                            "action": "submitted",
                            "object": f"{domain} report",
                        }
                    ],
                }
            elif difficulty == "medium":
                text = f"The {domain} analysis conducted by the research team (Dr. Johnson, Dr. Smith, and Dr. Lee) from January to March 2024 identified 15 critical issues across 4 categories. Budget allocation was $250,000 with 70% spent on data collection and 30% on analysis tools."
                extracted_data = {
                    "project": {
                        "type": f"{domain} analysis",
                        "duration": {"start": "2024-01", "end": "2024-03"},
                        "team": ["Dr. Johnson", "Dr. Smith", "Dr. Lee"],
                    },
                    "findings": {"critical_issues": 15, "categories": 4},
                    "budget": {
                        "total": 250000,
                        "currency": "USD",
                        "allocation": [
                            {"category": "data collection", "percentage": 70},
                            {"category": "analysis tools", "percentage": 30},
                        ],
                    },
                }
            else:  # hard
                text = f"Complex {domain} framework implementation: Phase 1 (Q1 2024): Infrastructure setup by Team Alpha (8 engineers), budget $1.2M, delivered 15 microservices. Phase 2 (Q2 2024): Integration with legacy systems, Team Beta (12 engineers) + 3 consultants, budget $2.1M, migrated 50TB data, 99.9% uptime achieved. Phase 3 (Q3-Q4 2024): Optimization and scaling, combined teams, target 10x performance improvement, budget $3.5M. Key stakeholders: CTO Jane Williams, VP Engineering Robert Chen, External Auditor: PwC."
                extracted_data = {
                    "project_phases": [
                        {
                            "phase": 1,
                            "period": "Q1 2024",
                            "description": "Infrastructure setup",
                            "team": {
                                "name": "Team Alpha",
                                "size": 8,
                                "role": "engineers",
                            },
                            "budget": 1200000,
                            "deliverables": {"microservices": 15},
                        },
                        {
                            "phase": 2,
                            "period": "Q2 2024",
                            "description": "Integration with legacy systems",
                            "team": {
                                "primary": {
                                    "name": "Team Beta",
                                    "size": 12,
                                    "role": "engineers",
                                },
                                "support": {"consultants": 3},
                            },
                            "budget": 2100000,
                            "achievements": {
                                "data_migrated": {"amount": 50, "unit": "TB"},
                                "uptime": 0.999,
                            },
                        },
                        {
                            "phase": 3,
                            "period": "Q3-Q4 2024",
                            "description": "Optimization and scaling",
                            "team": "combined teams",
                            "budget": 3500000,
                            "targets": {"performance_improvement": "10x"},
                        },
                    ],
                    "stakeholders": [
                        {"name": "Jane Williams", "role": "CTO"},
                        {"name": "Robert Chen", "role": "VP Engineering"},
                        {"name": "PwC", "role": "External Auditor"},
                    ],
                    "total_budget": 6800000,
                    "project_type": f"{domain} framework implementation",
                }
        else:
            # Use domain-specific template
            difficulty_templates = domain_templates.get(
                difficulty, domain_templates.get("medium", [])
            )
            if difficulty_templates:
                text, extracted_data = random.choice(difficulty_templates)
            else:
                text = f"{domain} text for extraction"
                extracted_data = {
                    "entities": [f"{domain} entity"],
                    "metadata": {"source": "generated"},
                }

        return GeneratedExample(
            input_data={"text": text, "format": "structured"},
            expected_output=extracted_data,
            difficulty=difficulty,
            reasoning=f"{difficulty.title()} {domain} structured extraction task with realistic expected output",
            metadata={"domain": domain, "type": "structured"},
        )

    async def _analyze_existing_examples(self, examples: List[Any]) -> Dict[str, Any]:
        """Analyze existing examples to understand patterns."""
        analysis = {
            "total_count": len(examples),
            "difficulty_distribution": {},
            "input_patterns": set(),
            "output_patterns": set(),
            "domains": set(),
        }

        for example in examples:
            # Extract difficulty
            if hasattr(example, "metadata") and example.metadata:
                difficulty = example.metadata.get("difficulty", "unknown")
                analysis["difficulty_distribution"][difficulty] = (
                    analysis["difficulty_distribution"].get(difficulty, 0) + 1
                )

                # Extract domain
                domain = example.metadata.get("domain", "unknown")
                analysis["domains"].add(domain)

            # Analyze input patterns
            if hasattr(example, "input_data") and example.input_data:
                analysis["input_patterns"].update(example.input_data.keys())

            # Analyze output patterns
            if hasattr(example, "expected_output"):
                if isinstance(example.expected_output, dict):
                    analysis["output_patterns"].update(example.expected_output.keys())
                else:
                    analysis["output_patterns"].add(
                        type(example.expected_output).__name__
                    )

        # Convert sets to lists for JSON serialization
        analysis["input_patterns"] = list(analysis["input_patterns"])
        analysis["output_patterns"] = list(analysis["output_patterns"])
        analysis["domains"] = list(analysis["domains"])

        return analysis

    def _balance_distribution(
        self, count: int, existing_distribution: Dict[str, int]
    ) -> Dict[str, int]:
        """Create balanced distribution across existing tiers."""
        tiers = list(existing_distribution.keys())
        if not tiers:
            return {"medium": count}

        per_tier = count // len(tiers)
        remainder = count % len(tiers)

        distribution = {}
        for i, tier in enumerate(tiers):
            distribution[tier] = per_tier + (1 if i < remainder else 0)

        return distribution

    def _follow_existing_distribution(
        self, count: int, existing_distribution: Dict[str, int]
    ) -> Dict[str, int]:
        """Follow existing distribution pattern."""
        total_existing = sum(existing_distribution.values())
        if total_existing == 0:
            return {"medium": count}

        distribution = {}
        remaining = count

        for tier, existing_count in existing_distribution.items():
            ratio = existing_count / total_existing
            tier_count = int(count * ratio)
            distribution[tier] = tier_count
            remaining -= tier_count

        # Add remainder to most common tier
        if remaining > 0:
            most_common_tier = max(existing_distribution, key=existing_distribution.get)
            distribution[most_common_tier] = (
                distribution.get(most_common_tier, 0) + remaining
            )

        return distribution

    async def _generate_examples_matching_pattern(
        self,
        existing_analysis: Dict[str, Any],
        difficulty: str,
        count: int,
        edge_cases: bool = False,
    ) -> List[GeneratedExample]:
        """Generate examples that match existing patterns."""
        examples = []

        # Extract patterns from analysis
        domains = existing_analysis.get("domains", ["general"])
        input_patterns = existing_analysis.get("input_patterns", ["text"])

        primary_domain = domains[0] if domains else "general"

        for _i in range(count):
            # Generate example following the pattern
            if "query" in input_patterns:
                # Customer service style
                example = self._generate_customer_service_example(
                    difficulty,
                    self.difficulty_specs[difficulty],
                    ["category_1", "category_2", "category_3"],
                )
            elif "contract_text" in input_patterns:
                # Legal style
                example = self._generate_legal_example(
                    difficulty,
                    self.difficulty_specs[difficulty],
                    ["compliance", "risk", "review"],
                )
            else:
                # General pattern
                example = self._generate_general_classification_example(
                    difficulty,
                    self.difficulty_specs[difficulty],
                    ["option_a", "option_b", "option_c"],
                    f"{primary_domain} classification",
                )

            examples.append(example)

        return examples
