"""
Georgian Cuisine Trivia Q&A Problem.

Answer trivia questions about Georgian food, including traditional dishes
like khachapuri and khinkali, regional specialties, ingredients, and
culinary traditions from Georgia.
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


class GeorgianCuisineTriviaQaProblem(BaseLangChainProblem):
    """
    Georgian cuisine trivia question answering problem.

    Tests the model's ability to answer questions about Georgian dishes,
    ingredients, cooking methods, and culinary traditions with accuracy
    and cultural sensitivity.
    """

    @classmethod
    def get_default_config(cls) -> ProblemConfig:
        """Get default configuration for this problem."""
        return ProblemConfig(
            name="georgian_cuisine_trivia_qa",
            description="Answer trivia questions about Georgian cuisine including traditional dishes, ingredients, and culinary traditions",
            difficulty_level="Medium",
            dataset_size=30,
            model_configurations={
                "model": ["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o"],
                "temperature": [0.1, 0.3, 0.5, 0.7],
                "max_tokens": [150, 250],
            },
            metrics=[
                ProblemMetric(
                    "accuracy", "Factual accuracy of answers", True, 1.0, ".1%"
                ),
                ProblemMetric("completeness", "Answer completeness", True, 0.9, ".1%"),
                ProblemMetric(
                    "cultural_accuracy", "Cultural details accuracy", True, 0.8, ".1%"
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
        """Create the evaluation dataset for Georgian cuisine trivia."""
        examples_data = [
            # Easy questions
            {
                "question": "What is khachapuri?",
                "expected_answer": "Khachapuri is a traditional Georgian cheese-filled bread. The main ingredients are bread dough, Georgian cheese (typically sulguni), eggs, and butter.",
                "difficulty": "easy",
                "topic": "Basic dishes",
            },
            {
                "question": "What is khinkali?",
                "expected_answer": "Khinkali are Georgian dumplings filled with spiced meat (beef, pork, or lamb), herbs, and broth. They're eaten by hand, holding the top knot.",
                "difficulty": "easy",
                "topic": "Basic dishes",
            },
            {
                "question": "What is the main cheese used in Georgian cuisine?",
                "expected_answer": "Sulguni is the main cheese used in Georgian cuisine. It's a brined cheese with elastic texture, perfect for khachapuri and other dishes.",
                "difficulty": "easy",
                "topic": "Ingredients",
            },
            {
                "question": "What is tkemali?",
                "expected_answer": "Tkemali is a Georgian sour plum sauce made from cherry plums, garlic, and herbs. It comes in red (ripe plums) and green (unripe plums) varieties.",
                "difficulty": "easy",
                "topic": "Sauces",
            },
            {
                "question": "What is chacha?",
                "expected_answer": "Chacha is a traditional Georgian grape brandy (pomace brandy) made from grape residue after winemaking. It's very strong, typically 40-65% alcohol.",
                "difficulty": "easy",
                "topic": "Beverages",
            },
            {
                "question": "What is kharcho?",
                "expected_answer": "Kharcho is a traditional Georgian beef soup with rice, cherry plum puree (tkemali), walnuts, and aromatic herbs. It's hearty and slightly sour.",
                "difficulty": "easy",
                "topic": "Soups",
            },
            # Medium questions
            {
                "question": "How is khinkali traditionally eaten in Georgia?",
                "expected_answer": "Khinkali is eaten by hand, holding the twisted top (kudi). You bite a small hole, sip the hot broth first, then eat the dumpling. The top knot is traditionally left on the plate.",
                "difficulty": "medium",
                "topic": "Eating customs",
            },
            {
                "question": "What is the difference between Adjarian and Imeretian khachapuri?",
                "expected_answer": "Adjarian khachapuri is boat-shaped with cheese, butter, and raw egg yolk in the center. Imeretian khachapuri is round and flat, filled only with cheese.",
                "difficulty": "medium",
                "topic": "Regional variations",
            },
            {
                "question": "What herbs are in khmeli suneli?",
                "expected_answer": "Khmeli suneli is a Georgian spice blend containing blue fenugreek, coriander, savory, black pepper, marigold petals, and sometimes basil, parsley, and mint.",
                "difficulty": "medium",
                "topic": "Spices",
            },
            {
                "question": "What is satsivi?",
                "expected_answer": "Satsivi is a walnut-based sauce served cold, traditionally with turkey or chicken during New Year's celebrations. It's made with walnuts, garlic, and spices.",
                "difficulty": "medium",
                "topic": "Sauces and holidays",
            },
            {
                "question": "What is the qvevri method?",
                "expected_answer": "Qvevri is an ancient Georgian winemaking method using large clay vessels buried underground. Grapes ferment with skins, stems, and seeds, creating amber-colored wines.",
                "difficulty": "medium",
                "topic": "Wine",
            },
            {
                "question": "What is lobio?",
                "expected_answer": "Lobio is a Georgian bean stew made with kidney beans, onions, garlic, walnuts, and herbs like coriander. It's seasoned with blue fenugreek and served hot or cold.",
                "difficulty": "medium",
                "topic": "Vegetarian dishes",
            },
            {
                "question": "What is pkhali?",
                "expected_answer": "Pkhali is a Georgian vegetable dish made from minced vegetables (spinach, beetroot, or beans) mixed with walnuts, garlic, herbs, and shaped into balls.",
                "difficulty": "medium",
                "topic": "Vegetarian dishes",
            },
            {
                "question": "What is the significance of bread (shotis puri) in Georgian culture?",
                "expected_answer": "Shotis puri is traditional Georgian bread baked in a deep circular clay oven (tone). Its boat shape is iconic, and bread is considered sacred in Georgian culture.",
                "difficulty": "medium",
                "topic": "Bread culture",
            },
            {
                "question": "What is matsoni?",
                "expected_answer": "Matsoni is a Georgian fermented milk product similar to yogurt but with a more elastic texture. It's used in cooking and eaten plain or with herbs.",
                "difficulty": "medium",
                "topic": "Dairy products",
            },
            {
                "question": "What is badrijani?",
                "expected_answer": "Badrijani is fried eggplant rolls filled with walnut paste (made from walnuts, garlic, herbs). It's garnished with pomegranate seeds.",
                "difficulty": "medium",
                "topic": "Appetizers",
            },
            {
                "question": "What is Megrelian khachapuri?",
                "expected_answer": "Megrelian khachapuri is similar to Imeretian but topped with extra cheese and egg mixture. It's from the Samegrelo region and is extra cheesy.",
                "difficulty": "medium",
                "topic": "Regional variations",
            },
            {
                "question": "What is ajika?",
                "expected_answer": "Ajika is a spicy Georgian paste made from hot peppers, garlic, herbs, and spices. The Abkhazian version is spicier than the Megrelian version.",
                "difficulty": "medium",
                "topic": "Condiments",
            },
            {
                "question": "What is Georgian tonis puri?",
                "expected_answer": "Tonis puri is bread baked in a tone (deep circular clay oven). The dough is slapped onto the hot oven walls and bakes quickly, creating a crispy crust.",
                "difficulty": "medium",
                "topic": "Bread",
            },
            {
                "question": "What is nigvziani badrijani?",
                "expected_answer": "Nigvziani badrijani is the full name for badrijani - fried eggplant rolls with walnut filling. 'Nigvziani' means 'with walnuts' in Georgian.",
                "difficulty": "medium",
                "topic": "Dishes",
            },
            # Hard questions
            {
                "question": "Name three traditional Georgian wines and their grape varieties.",
                "expected_answer": "Saperavi (red grape, dry red wine), Rkatsiteli (white grape for dry whites), and Kindzmarauli (semi-sweet red from Saperavi grapes in Kakheti region).",
                "difficulty": "hard",
                "topic": "Wine varieties",
            },
            {
                "question": "What is a supra and what role does the tamada play?",
                "expected_answer": "A supra is a traditional Georgian feast with multiple courses and toasts. The tamada (toastmaster) leads elaborate toasts, maintaining order and tradition throughout the celebration.",
                "difficulty": "hard",
                "topic": "Dining traditions",
            },
            {
                "question": "What is churchkhela and why is it called Georgian Snickers?",
                "expected_answer": "Churchkhela is a candle-shaped candy made by dipping strings of walnuts in thickened grape juice (tatara). It's called Georgian Snickers due to its nuts-and-sweet coating combination.",
                "difficulty": "hard",
                "topic": "Desserts",
            },
            {
                "question": "What are jonjoli?",
                "expected_answer": "Jonjoli are pickled sprouts from the bladdernut tree, a unique Georgian delicacy. They have a distinctive taste and are served as an appetizer or side dish.",
                "difficulty": "hard",
                "topic": "Unique ingredients",
            },
            {
                "question": "What is the traditional Georgian walnut oil called and how is it used?",
                "expected_answer": "Georgian walnut oil is traditionally cold-pressed and used for dressing salads, especially with fresh herbs. It's essential in many vegetable pkhali dishes.",
                "difficulty": "hard",
                "topic": "Cooking ingredients",
            },
            {
                "question": "What is the role of walnuts in Georgian cuisine?",
                "expected_answer": "Walnuts are essential in Georgian cuisine, used in satsivi, pkhali, badrijani, and many sauces. Georgia has unique endemic walnut varieties prized for their flavor.",
                "difficulty": "hard",
                "topic": "Key ingredients",
            },
            {
                "question": "What is chikhirtma?",
                "expected_answer": "Chikhirtma is a Georgian chicken soup thickened with egg yolks and flour, flavored with onions, garlic, and fresh herbs. It's traditionally a hangover cure.",
                "difficulty": "hard",
                "topic": "Soups",
            },
            {
                "question": "What are the main wine regions of Georgia?",
                "expected_answer": "The main wine regions are Kakheti (70% of production), Kartli, Imereti, Racha-Lechkhumi, and Samtskhe-Javakheti. Each has distinct terroir and grape varieties.",
                "difficulty": "hard",
                "topic": "Wine regions",
            },
            {
                "question": "What is mtsvadi and how does it differ from regular kebab?",
                "expected_answer": "Mtsvadi is Georgian grilled meat (usually pork or veal) marinated simply with onions and pomegranate juice. Unlike kebab, it uses minimal spices to highlight meat quality.",
                "difficulty": "hard",
                "topic": "Meat dishes",
            },
            {
                "question": "What is the significance of pomegranate in Georgian cuisine?",
                "expected_answer": "Pomegranate is widely used in Georgian cuisine for its juice in marinades, seeds as garnish, and symbolic meaning of fertility and abundance in Georgian culture.",
                "difficulty": "hard",
                "topic": "Ingredients and culture",
            },
        ]

        examples = []
        for i, data in enumerate(examples_data[:30]):  # Ensure exactly 30 examples
            example = EvaluationExample(
                input_data={"question": data["question"]},
                expected_output=data["expected_answer"],
                metadata={
                    "difficulty": data["difficulty"],
                    "topic": data["topic"],
                    "example_id": f"cuisine_{i + 1:03d}",
                },
            )
            examples.append(example)

        return Dataset(
            examples=examples,
            name="Georgian Cuisine Trivia",
            description="Trivia questions about Georgian food culture, traditional dishes, and culinary traditions",
        )

    def create_function(self) -> Callable:
        """Create the base Georgian cuisine trivia function."""

        def georgian_cuisine_trivia(question: str) -> str:
            """Answer trivia questions about Georgian cuisine."""
            llm = ChatOpenAI(
                model="gpt-4o-mini",
                temperature=0.3,
                model_kwargs={"max_tokens": 150},
            )

            prompt_template = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        """You are an expert on Georgian cuisine and food culture.
Provide accurate, informative answers about Georgian dishes, ingredients, cooking methods, and culinary traditions.
Be specific about regional variations and traditional preparations.""",
                    ),
                    (
                        "human",
                        "Question: {question}\n\nProvide a clear and informative answer:",
                    ),
                ]
            )

            chain = LLMChain(llm=llm, prompt=prompt_template)
            result = chain.invoke({"question": question})["text"]

            return result.strip()

        return georgian_cuisine_trivia

    def create_optimized_function(self) -> Callable:
        """Create the optimized Georgian cuisine trivia assistant."""

        @traigent.optimize(
            eval_dataset=self.create_temporary_dataset_file(),
            objectives=self.get_optimization_objectives(),
            configuration_space=self.get_configuration_space(),
            auto_override_frameworks=True,
            framework_targets=["langchain_openai.ChatOpenAI"],
            execution_mode="edge_analytics",
        )
        def georgian_cuisine_trivia_optimized(question: str) -> str:
            """Optimized Georgian cuisine trivia assistant."""
            llm = ChatOpenAI(
                model="gpt-4o-mini",  # Will be overridden by TraiGent
                temperature=0.3,  # Will be overridden by TraiGent
                model_kwargs={"max_tokens": 150},  # Will be overridden by TraiGent
            )

            prompt_template = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        """You are an expert on Georgian cuisine and food culture.
Provide accurate, informative answers about Georgian dishes, ingredients, cooking methods, and culinary traditions.
Be specific about regional variations and traditional preparations.""",
                    ),
                    (
                        "human",
                        "Question: {question}\n\nProvide a clear and informative answer:",
                    ),
                ]
            )

            chain = LLMChain(llm=llm, prompt=prompt_template)
            result = chain.invoke({"question": question})["text"]

            return result.strip()

        return georgian_cuisine_trivia_optimized

    def evaluate_custom_metrics(
        self,
        outputs: List[Any],
        expected_outputs: List[Any],
        errors: List[Optional[str]],
    ) -> Dict[str, float]:
        """Compute Georgian cuisine trivia specific metrics."""
        metrics = {}

        # Get dataset for metadata access
        self.get_dataset()

        # Standard accuracy (contains key facts)
        correct = 0
        total = 0
        complete = 0
        culturally_accurate = 0

        for _i, (output, expected, error) in enumerate(
            zip(outputs, expected_outputs, errors)
        ):
            if error is None and expected is not None and output:
                total += 1
                output_lower = str(output).lower()
                expected_lower = str(expected).lower()

                # Extract key terms from expected answer
                import re

                # Extract Georgian terms, dish names, ingredients
                georgian_terms = re.findall(r"\b[A-Za-z]+(?:i|a|o|e|u)\b", expected)
                key_ingredients = [
                    "cheese",
                    "walnut",
                    "bean",
                    "grape",
                    "wine",
                    "bread",
                    "meat",
                    "herb",
                ]

                # Check accuracy - key facts present
                facts_found = 0
                total_facts = 0

                # Check for Georgian terms
                for term in georgian_terms:
                    total_facts += 1
                    if term.lower() in output_lower:
                        facts_found += 1

                # Check for key ingredients mentioned in expected
                for ingredient in key_ingredients:
                    if ingredient in expected_lower:
                        total_facts += 1
                        if ingredient in output_lower:
                            facts_found += 1

                if total_facts > 0:
                    accuracy_score = facts_found / total_facts
                    if accuracy_score >= 0.6:
                        correct += 1

                # Completeness - answer addresses all parts
                question_parts = ["what", "how", "why", "when", "where"]
                addressed_parts = sum(
                    1 for part in question_parts if part in output_lower
                )
                if addressed_parts >= 2 or len(output.split()) >= 20:
                    complete += 1

                # Cultural accuracy - mentions traditions, regions, methods
                cultural_keywords = [
                    "traditional",
                    "georgian",
                    "region",
                    "method",
                    "culture",
                    "celebration",
                    "feast",
                ]
                if any(keyword in output_lower for keyword in cultural_keywords):
                    culturally_accurate += 1

        metrics["accuracy"] = correct / total if total > 0 else 0.0
        metrics["completeness"] = complete / total if total > 0 else 0.0
        metrics["cultural_accuracy"] = culturally_accurate / total if total > 0 else 0.0

        return metrics


# Register this problem
register_problem("georgian_cuisine_trivia_qa", GeorgianCuisineTriviaQaProblem)
