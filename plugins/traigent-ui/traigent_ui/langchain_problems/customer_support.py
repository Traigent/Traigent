"""
Customer Support Classification Problem.

A challenging classification problem that requires understanding context, intent,
and subtle differences between support categories. Designed to differentiate
between models based on reasoning capability.
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


class CustomerSupportProblem(BaseLangChainProblem):
    """
    Customer support ticket classification problem.

    This problem tests the model's ability to:
    1. Understand customer intent and context
    2. Distinguish between subtle category differences
    3. Handle ambiguous and edge cases
    4. Maintain consistency across similar scenarios
    """

    CATEGORIES = [
        "shipping_inquiry",
        "return_request",
        "account_support",
        "policy_question",
        "technical_support",
        "product_inquiry",
        "billing_issue",
        "order_modification",
    ]

    @classmethod
    def get_default_config(cls) -> ProblemConfig:
        """Get default configuration for this problem."""
        return ProblemConfig(
            name="customer_support",
            description="Multi-category customer support ticket classification with nuanced edge cases",
            difficulty_level="Advanced",
            dataset_size=30,
            model_configurations={
                "model": ["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o"],
                "temperature": [0.1, 0.7],
                "max_tokens": [50],
            },
            metrics=[
                ProblemMetric(
                    "accuracy", "Overall classification accuracy", True, 1.0, ".1%"
                ),
                ProblemMetric(
                    "category_f1",
                    "Average F1 score across categories",
                    True,
                    0.8,
                    ".3f",
                ),
                ProblemMetric(
                    "hard_case_accuracy",
                    "Accuracy on challenging examples",
                    True,
                    1.2,
                    ".1%",
                ),
                ProblemMetric(
                    "consistency_score",
                    "Consistency on similar cases",
                    True,
                    0.6,
                    ".1%",
                ),
                ProblemMetric(
                    "confidence_calibration",
                    "How well confidence matches accuracy",
                    True,
                    0.4,
                    ".3f",
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
        """Create challenging customer support dataset with graduated difficulty."""
        examples_data = [
            # Tier 1: Clear, unambiguous cases (should be easy for all models)
            {
                "query": "My order #12345 hasn't arrived yet, it's been 10 days since shipping",
                "category": "shipping_inquiry",
                "difficulty": "easy",
                "reasoning": "Clear shipping delay inquiry",
            },
            {
                "query": "I want to return this sweater, it doesn't fit properly",
                "category": "return_request",
                "difficulty": "easy",
                "reasoning": "Straightforward return request",
            },
            {
                "query": "How do I reset my account password?",
                "category": "account_support",
                "difficulty": "easy",
                "reasoning": "Direct account access issue",
            },
            {
                "query": "What's your refund policy for digital products?",
                "category": "policy_question",
                "difficulty": "easy",
                "reasoning": "Clear policy inquiry",
            },
            {
                "query": "The mobile app crashes when I try to checkout",
                "category": "technical_support",
                "difficulty": "easy",
                "reasoning": "Technical functionality issue",
            },
            # Tier 2: Moderate ambiguity (requires basic reasoning)
            {
                "query": "Do you have this shirt in size XL available?",
                "category": "product_inquiry",
                "difficulty": "medium",
                "reasoning": "Product availability question, not order modification",
            },
            {
                "query": "I was charged twice for the same order #98765",
                "category": "billing_issue",
                "difficulty": "medium",
                "reasoning": "Billing discrepancy, not shipping issue",
            },
            {
                "query": "When will the new summer collection be available?",
                "category": "product_inquiry",
                "difficulty": "medium",
                "reasoning": "Product availability, not policy question",
            },
            {
                "query": "My discount code SAVE20 isn't working at checkout",
                "category": "technical_support",
                "difficulty": "medium",
                "reasoning": "Technical issue with system functionality",
            },
            {
                "query": "Can I change my shipping address after placing the order?",
                "category": "order_modification",
                "difficulty": "medium",
                "reasoning": "Modifying existing order, not general policy",
            },
            # Tier 3: Contextual reasoning required (moderate challenge)
            {
                "query": "I ordered last week but now I see the same item is 30% cheaper",
                "category": "billing_issue",
                "difficulty": "hard",
                "reasoning": "Price adjustment request - billing concern, not policy question",
            },
            {
                "query": "The product page won't load and I can't complete my purchase",
                "category": "technical_support",
                "difficulty": "hard",
                "reasoning": "Technical issue preventing purchase, not product inquiry",
            },
            {
                "query": "I forgot my password and the reset email isn't coming through",
                "category": "account_support",
                "difficulty": "hard",
                "reasoning": "Account access issue, primary concern over email delivery",
            },
            {
                "query": "My package says delivered but I never received it, what should I do?",
                "category": "shipping_inquiry",
                "difficulty": "hard",
                "reasoning": "Delivery issue, not return request yet",
            },
            {
                "query": "Can I get a refund if I haven't opened the sealed product yet?",
                "category": "return_request",
                "difficulty": "hard",
                "reasoning": "Return initiation, not policy question despite 'can I' phrasing",
            },
            # Tier 4: Complex multi-faceted scenarios (challenging)
            {
                "query": "Your website keeps logging me out when I try to apply my employee discount during checkout",
                "category": "technical_support",
                "difficulty": "very_hard",
                "reasoning": "Session management issue is primary concern, not billing/discount issue",
            },
            {
                "query": "I need to update my payment method for a pending order that hasn't shipped yet",
                "category": "order_modification",
                "difficulty": "very_hard",
                "reasoning": "Order change request, not billing issue or account support",
            },
            {
                "query": "The tracking shows delivered to wrong address but my account address is correct",
                "category": "shipping_inquiry",
                "difficulty": "very_hard",
                "reasoning": "Shipping/delivery issue, not account issue despite address mention",
            },
            {
                "query": "I received the wrong color but right size, can I exchange without return shipping?",
                "category": "return_request",
                "difficulty": "very_hard",
                "reasoning": "Exchange request despite specific conditions mentioned",
            },
            {
                "query": "My subscription auto-renewed but I cancelled it last month, why was I charged?",
                "category": "billing_issue",
                "difficulty": "very_hard",
                "reasoning": "Billing discrepancy about subscription, not account support",
            },
            # Tier 5: Expert-level edge cases (very challenging)
            {
                "query": "I ordered a gift for overseas delivery but customs rejected it, can you help with re-shipping?",
                "category": "shipping_inquiry",
                "difficulty": "expert",
                "reasoning": "Complex shipping issue with customs, not return request",
            },
            {
                "query": "My account shows 3 items ordered but I was only charged for 2, should I be concerned?",
                "category": "billing_issue",
                "difficulty": "expert",
                "reasoning": "Billing discrepancy inquiry, not order modification",
            },
            {
                "query": "I'm trying to use store credit from a previous return but the system won't accept it",
                "category": "technical_support",
                "difficulty": "expert",
                "reasoning": "Technical system issue with payment processing, not billing issue",
            },
            {
                "query": "I ordered expedited shipping but the delivery estimate is same as standard shipping",
                "category": "shipping_inquiry",
                "difficulty": "expert",
                "reasoning": "Shipping service issue, not billing despite paying for upgrade",
            },
            {
                "query": "The item I want to return was a gift - do I need the original buyer's information?",
                "category": "return_request",
                "difficulty": "expert",
                "reasoning": "Return process question, though could seem like policy question",
            },
            # Additional expert cases to reach 30 examples
            {
                "query": "My order was split into multiple shipments but I was charged shipping for each one",
                "category": "billing_issue",
                "difficulty": "expert",
                "reasoning": "Billing concern about shipping charges, not shipping inquiry",
            },
            {
                "query": "I keep getting password reset emails but I didn't request them, is my account secure?",
                "category": "account_support",
                "difficulty": "expert",
                "reasoning": "Account security concern, not technical support",
            },
            {
                "query": "The promo code worked but I was charged full price, then got a partial refund automatically",
                "category": "billing_issue",
                "difficulty": "expert",
                "reasoning": "Complex billing flow issue, despite automatic resolution",
            },
            {
                "query": "I bought a digital product but the download link expired before I could use it",
                "category": "technical_support",
                "difficulty": "expert",
                "reasoning": "Technical access issue to digital product, not return request",
            },
            {
                "query": "My order history shows a different delivery address than what I selected at checkout",
                "category": "order_modification",
                "difficulty": "expert",
                "reasoning": "Order details discrepancy, not account support or shipping inquiry",
            },
            {
                "query": "What's your refund policy?",
                "category": "policy_question",
                "difficulty": "easy",
                "reasoning": "Generated example 1",
            },
            {
                "query": "Your site keeps logging me out when I apply my employee discount",
                "category": "technical_support",
                "difficulty": "very_hard",
                "reasoning": "Generated example 2",
            },
            {
                "query": "Your site keeps logging me out when I apply my employee discount",
                "category": "technical_support",
                "difficulty": "very_hard",
                "reasoning": "Generated example 3",
            },
            {
                "query": "Do you have this phone in size XL?",
                "category": "product_inquiry",
                "difficulty": "medium",
                "reasoning": "Generated example 4",
            },
            {
                "query": "I ordered last week but now it's cheaper, can you help?",
                "category": "billing_issue",
                "difficulty": "hard",
                "reasoning": "Generated example 5",
            },
            {
                "query": "Do you have this jacket in size XL?",
                "category": "product_inquiry",
                "difficulty": "medium",
                "reasoning": "Generated example 6",
            },
            {
                "query": "My discount code isn't working",
                "category": "technical_support",
                "difficulty": "medium",
                "reasoning": "Generated example 7",
            },
            {
                "query": "The product page won't load and I can't buy it",
                "category": "technical_support",
                "difficulty": "hard",
                "reasoning": "Generated example 8",
            },
            {
                "query": "I need to update payment for a pending order that hasn't shipped",
                "category": "order_modification",
                "difficulty": "very_hard",
                "reasoning": "Generated example 9",
            },
            {
                "query": "My order #10653 hasn't arrived yet",
                "category": "shipping_inquiry",
                "difficulty": "easy",
                "reasoning": "Generated example 10",
            },
            {
                "query": "I want to return this shirt, it doesn't fit",
                "category": "return_request",
                "difficulty": "easy",
                "reasoning": "Generated example 1",
            },
            {
                "query": "Your site keeps logging me out when I apply my employee discount",
                "category": "technical_support",
                "difficulty": "very_hard",
                "reasoning": "Generated example 2",
            },
            {
                "query": "Do you have this laptop in size XS?",
                "category": "product_inquiry",
                "difficulty": "medium",
                "reasoning": "Generated example 3",
            },
            {
                "query": "The tracking shows wrong address but my account is correct",
                "category": "shipping_inquiry",
                "difficulty": "very_hard",
                "reasoning": "Generated example 4",
            },
            {
                "query": "The product page won't load and I can't buy it",
                "category": "technical_support",
                "difficulty": "hard",
                "reasoning": "Generated example 5",
            },
            {
                "query": "My discount code isn't working",
                "category": "technical_support",
                "difficulty": "medium",
                "reasoning": "Generated example 6",
            },
            {
                "query": "I need to update payment for a pending order that hasn't shipped",
                "category": "order_modification",
                "difficulty": "very_hard",
                "reasoning": "Generated example 7",
            },
            {
                "query": "I ordered two items but the package arrived damaged and one product is missing. Should I return everything or can you just send the missing item? Also, I paid with a gift card - how will the refund work?",
                "category": "return_request",
                "difficulty": "hard",
                "reasoning": "Complex scenario involving damaged goods, missing items, and refund method questions - primarily about returning products",
            },
            {
                "query": "My subscription was supposed to auto-renew yesterday but I got charged twice. The app shows I'm on the free plan but my bank shows both charges went through.",
                "category": "billing_issue",
                "difficulty": "hard",
                "reasoning": "Double charging issue with conflicting account status - clear billing problem despite technical elements",
            },
            {
                "query": "Can I change the delivery address for order #78901? It already shipped but tracking shows it hasn't left the warehouse yet. Need it sent to my office instead.",
                "category": "order_modification",
                "difficulty": "medium",
                "reasoning": "Request to modify order details after placement but before actual shipment",
            },
            {
                "query": "What are the exact dimensions and weight capacity of the XR-2000 standing desk? The product page only shows assembled dimensions but I need to know if it'll fit through my doorway.",
                "category": "product_inquiry",
                "difficulty": "medium",
                "reasoning": "Specific product specification request not available on standard product page",
            },
            {
                "query": "I've been locked out after too many login attempts but the reset email isn't coming through. I've checked spam and tried different browsers. This is urgent as I have pending orders to track.",
                "category": "account_support",
                "difficulty": "hard",
                "reasoning": "Account access issue with failed recovery process - primary concern is regaining account access",
            },
        ]

        examples = []
        for i, data in enumerate(examples_data):
            example = EvaluationExample(
                input_data={"query": data["query"]},
                expected_output=data["category"],
                metadata={
                    "difficulty": data["difficulty"],
                    "reasoning": data["reasoning"],
                    "example_id": f"cs_{i + 1:03d}",
                    "category": data["category"],
                },
            )
            examples.append(example)

        return Dataset(
            examples=examples,
            name="Customer Support Classification",
            description=f"Customer support ticket classification with {len(examples)} examples across 5 difficulty tiers",
        )

    def create_function(self) -> Callable:
        """Create the base customer support classifier function."""

        def customer_support_classifier(query: str) -> str:
            """Classify customer support tickets into categories."""
            llm = ChatOpenAI(
                model="gpt-4o-mini",
                temperature=0.7,
                max_tokens=50,
            )

            prompt_template = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        "You are a customer support ticket classifier. Classify tickets into exactly one of these categories: shipping_inquiry, return_request, account_support, policy_question, technical_support, product_inquiry, billing_issue, or order_modification.",
                    ),
                    (
                        "human",
                        """Classify this customer support ticket:

Customer Query: {query}

Respond with only the category name, nothing else.""",
                    ),
                ]
            )

            chain = LLMChain(llm=llm, prompt=prompt_template)
            result = chain.invoke({"query": query})["text"]

            return self.clean_llm_output(result, self.CATEGORIES)

        return customer_support_classifier

    def create_optimized_function(self) -> Callable:
        """Create the optimized customer support classifier."""

        @traigent.optimize(
            eval_dataset=self.create_temporary_dataset_file(),
            objectives=self.get_optimization_objectives(),
            configuration_space=self.get_configuration_space(),
            auto_override_frameworks=True,
            framework_targets=["langchain_openai.ChatOpenAI"],
            execution_mode="edge_analytics",
        )
        def customer_support_classifier_optimized(query: str) -> str:
            """Optimized customer support ticket classifier."""
            llm = ChatOpenAI(
                model="gpt-4o-mini",  # Will be overridden by Traigent
                temperature=0.7,  # Will be overridden by Traigent
                max_tokens=50,  # Will be overridden by Traigent
            )

            prompt_template = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        "You are a customer support ticket classifier. Classify tickets into exactly one of these categories: shipping_inquiry, return_request, account_support, policy_question, technical_support, product_inquiry, billing_issue, or order_modification.",
                    ),
                    (
                        "human",
                        """Classify this customer support ticket:

Customer Query: {query}

Respond with only the category name, nothing else.""",
                    ),
                ]
            )

            chain = LLMChain(llm=llm, prompt=prompt_template)
            result = chain.invoke({"query": query})["text"]

            return self.clean_llm_output(result, self.CATEGORIES)

        return customer_support_classifier_optimized

    def evaluate_custom_metrics(
        self,
        outputs: List[Any],
        expected_outputs: List[Any],
        errors: List[Optional[str]],
    ) -> Dict[str, float]:
        """Compute customer support specific metrics."""
        metrics = {}

        # Get dataset for metadata access
        dataset = self.get_dataset()

        # Standard accuracy
        correct = 0
        total = 0
        for output, expected, error in zip(outputs, expected_outputs, errors):
            if error is None and expected is not None:
                if output == expected:
                    correct += 1
                total += 1

        metrics["accuracy"] = correct / total if total > 0 else 0.0

        # Category-wise F1 score
        from collections import defaultdict

        category_tp = defaultdict(int)
        category_fp = defaultdict(int)
        category_fn = defaultdict(int)

        for output, expected, error in zip(outputs, expected_outputs, errors):
            if error is None and expected is not None:
                if output == expected:
                    category_tp[expected] += 1
                else:
                    category_fp[output] += 1
                    category_fn[expected] += 1

        f1_scores = []
        for category in self.CATEGORIES:
            tp = category_tp[category]
            fp = category_fp[category]
            fn = category_fn[category]

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0
            )
            f1_scores.append(f1)

        metrics["category_f1"] = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0

        # Hard case accuracy (very_hard and expert difficulty)
        hard_correct = 0
        hard_total = 0
        for i, (output, expected, error) in enumerate(
            zip(outputs, expected_outputs, errors)
        ):
            if i < len(dataset.examples):
                difficulty = dataset.examples[i].metadata.get("difficulty", "")
                if difficulty in ["very_hard", "expert"]:
                    if error is None and expected is not None:
                        if output == expected:
                            hard_correct += 1
                        hard_total += 1

        metrics["hard_case_accuracy"] = (
            hard_correct / hard_total if hard_total > 0 else 0.0
        )

        # Consistency score (simplified - could be enhanced)
        # For now, just measure how often the model gives the same answer to similar queries
        metrics["consistency_score"] = metrics["accuracy"]  # Placeholder

        # Confidence calibration (simplified)
        # In a real implementation, you'd need confidence scores from the model
        metrics["confidence_calibration"] = metrics["accuracy"]  # Placeholder

        return metrics


# Register this problem
register_problem("customer_support", CustomerSupportProblem)
