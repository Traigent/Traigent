"""
Multi-Language Sentiment Analysis Problem.

A challenging sentiment analysis problem that tests the model's ability to:
1. Handle multiple languages with cultural nuances
2. Detect subtle sentiment differences and sarcasm
3. Account for cultural context in sentiment interpretation
4. Maintain consistency across similar expressions
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


class SentimentAnalysisProblem(BaseLangChainProblem):
    """
    Multi-language sentiment analysis problem.

    Tests the model's ability to understand sentiment across languages and cultures,
    including handling of sarcasm, cultural context, and subtle emotional nuances.
    """

    SENTIMENTS = ["positive", "negative", "neutral"]
    CONFIDENCE_LEVELS = ["high", "medium", "low"]

    @classmethod
    def get_default_config(cls) -> ProblemConfig:
        """Get default configuration for this problem."""
        return ProblemConfig(
            name="sentiment_analysis",
            description="Multi-language sentiment analysis with cultural nuance and sarcasm detection",
            difficulty_level="Advanced",
            dataset_size=25,
            model_configurations={
                "model": ["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o"],
                "temperature": [0.2, 0.6],
                "max_tokens": [100],
            },
            metrics=[
                ProblemMetric(
                    "sentiment_accuracy",
                    "Overall sentiment classification accuracy",
                    True,
                    1.0,
                    ".1%",
                ),
                ProblemMetric(
                    "cross_language_consistency",
                    "Consistency across language translations",
                    True,
                    0.9,
                    ".1%",
                ),
                ProblemMetric(
                    "sarcasm_detection",
                    "Ability to detect sarcastic sentiment",
                    True,
                    1.1,
                    ".1%",
                ),
                ProblemMetric(
                    "cultural_sensitivity",
                    "Appropriate cultural context handling",
                    True,
                    0.8,
                    ".1%",
                ),
                ProblemMetric(
                    "confidence_calibration",
                    "Accuracy of confidence predictions",
                    True,
                    0.6,
                    ".2f",
                ),
            ],
            optimization_objectives=["sentiment_accuracy"],
            expected_model_ranking=["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"],
        )

    def __init__(self, config: Optional[ProblemConfig] = None):
        if config is None:
            config = self.get_default_config()
        super().__init__(config)

    def create_dataset(self) -> Dataset:
        """Create challenging multi-language sentiment analysis dataset."""
        examples_data = [
            # English - Straightforward Cases
            {
                "text": "I absolutely love this new restaurant! The food is amazing and the service is excellent.",
                "language": "English",
                "expected_sentiment": "positive",
                "expected_confidence": "high",
                "difficulty": "easy",
                "cultural_context": "Western dining culture",
            },
            {
                "text": "This movie was terrible. Worst acting I've ever seen and the plot made no sense.",
                "language": "English",
                "expected_sentiment": "negative",
                "expected_confidence": "high",
                "difficulty": "easy",
                "cultural_context": "General entertainment criticism",
            },
            {
                "text": "The weather today is cloudy with a chance of rain.",
                "language": "English",
                "expected_sentiment": "neutral",
                "expected_confidence": "high",
                "difficulty": "easy",
                "cultural_context": "Neutral weather reporting",
            },
            # English - Sarcasm and Irony
            {
                "text": "Oh great, another meeting that could have been an email. Just what I needed today!",
                "language": "English",
                "expected_sentiment": "negative",
                "expected_confidence": "medium",
                "difficulty": "hard",
                "cultural_context": "Modern workplace sarcasm",
            },
            {
                "text": "Sure, because staying up until 3 AM working on this project was exactly how I wanted to spend my weekend.",
                "language": "English",
                "expected_sentiment": "negative",
                "expected_confidence": "medium",
                "difficulty": "hard",
                "cultural_context": "Work-life balance frustration",
            },
            # Spanish - Cultural Context
            {
                "text": "¡Qué maravilloso! Me encanta pasar tiempo con mi familia los domingos.",
                "language": "Spanish",
                "expected_sentiment": "positive",
                "expected_confidence": "high",
                "difficulty": "medium",
                "cultural_context": "Hispanic family values",
            },
            {
                "text": "No me gusta nada este cambio en la empresa. Todo era mejor antes.",
                "language": "Spanish",
                "expected_sentiment": "negative",
                "expected_confidence": "high",
                "difficulty": "medium",
                "cultural_context": "Resistance to change",
            },
            {
                "text": "Claro, como si fuera tan fácil encontrar trabajo en estos tiempos...",
                "language": "Spanish",
                "expected_sentiment": "negative",
                "expected_confidence": "medium",
                "difficulty": "hard",
                "cultural_context": "Economic sarcasm",
            },
            # French - Nuanced Expressions
            {
                "text": "C'est vraiment magnifique, ce coucher de soleil sur la Seine!",
                "language": "French",
                "expected_sentiment": "positive",
                "expected_confidence": "high",
                "difficulty": "medium",
                "cultural_context": "French appreciation of beauty",
            },
            {
                "text": "Je suis déçu par cette nouvelle politique. C'est vraiment décevant.",
                "language": "French",
                "expected_sentiment": "negative",
                "expected_confidence": "high",
                "difficulty": "medium",
                "cultural_context": "Political disappointment",
            },
            {
                "text": "Ah oui, bien sûr, parce que c'est exactement ce qu'il nous fallait maintenant...",
                "language": "French",
                "expected_sentiment": "negative",
                "expected_confidence": "medium",
                "difficulty": "very_hard",
                "cultural_context": "French ironic expression",
            },
            # Mixed Language and Cultural Nuances
            {
                "text": "The pasta was okay, pero el servicio fue excelente. Overall not bad!",
                "language": "Mixed (English/Spanish)",
                "expected_sentiment": "positive",
                "expected_confidence": "medium",
                "difficulty": "hard",
                "cultural_context": "Bilingual restaurant review",
            },
            # Subtle Emotional Context
            {
                "text": "I suppose the presentation went well enough. People seemed to pay attention.",
                "language": "English",
                "expected_sentiment": "neutral",
                "expected_confidence": "low",
                "difficulty": "very_hard",
                "cultural_context": "Understated professional assessment",
            },
            {
                "text": "It's fine, I guess. Not what I expected but it's whatever.",
                "language": "English",
                "expected_sentiment": "negative",
                "expected_confidence": "low",
                "difficulty": "very_hard",
                "cultural_context": "Passive disappointment",
            },
            # Cultural-Specific Expressions
            {
                "text": "Ça va comme un lundi matin... encore une journée qui commence bien!",
                "language": "French",
                "expected_sentiment": "negative",
                "expected_confidence": "medium",
                "difficulty": "very_hard",
                "cultural_context": "French Monday morning blues with sarcasm",
            },
            {
                "text": "Está padrísimo este lugar, me la paso súper bien aquí.",
                "language": "Spanish (Mexican)",
                "expected_sentiment": "positive",
                "expected_confidence": "high",
                "difficulty": "medium",
                "cultural_context": "Mexican slang for enthusiasm",
            },
            # Professional Context with Subtle Sentiment
            {
                "text": "Thank you for your feedback. We will certainly take it into consideration for future improvements.",
                "language": "English",
                "expected_sentiment": "neutral",
                "expected_confidence": "medium",
                "difficulty": "hard",
                "cultural_context": "Professional diplomatic response",
            },
            {
                "text": "Je vous remercie pour cette proposition intéressante, nous allons y réfléchir.",
                "language": "French",
                "expected_sentiment": "neutral",
                "expected_confidence": "medium",
                "difficulty": "hard",
                "cultural_context": "French business politeness",
            },
            # Complex Emotional States
            {
                "text": "I'm happy for you, really I am, but I can't help feeling a bit jealous about your promotion.",
                "language": "English",
                "expected_sentiment": "neutral",
                "expected_confidence": "low",
                "difficulty": "very_hard",
                "cultural_context": "Mixed emotions - happiness and jealousy",
            },
            {
                "text": "Es bueno que tengamos estas opciones, aunque no estoy completamente convencido de ninguna.",
                "language": "Spanish",
                "expected_sentiment": "neutral",
                "expected_confidence": "low",
                "difficulty": "hard",
                "cultural_context": "Cautious optimism with reservations",
            },
            # Social Media Style
            {
                "text": "Just got my test results back... let's just say studying more would have been a good idea 😅",
                "language": "English",
                "expected_sentiment": "negative",
                "expected_confidence": "medium",
                "difficulty": "medium",
                "cultural_context": "Self-deprecating humor about failure",
            },
            {
                "text": "Nouveau job, nouveau défi! Hâte de voir ce que l'avenir me réserve ✨",
                "language": "French",
                "expected_sentiment": "positive",
                "expected_confidence": "high",
                "difficulty": "easy",
                "cultural_context": "Optimistic career change",
            },
            # Regional/Dialectal Variations
            {
                "text": "That concert was mint! Proper brilliant, like.",
                "language": "English (British)",
                "expected_sentiment": "positive",
                "expected_confidence": "high",
                "difficulty": "medium",
                "cultural_context": "British slang enthusiasm",
            },
            {
                "text": "No mames, está padrísimo este lugar, wey!",
                "language": "Spanish (Mexican slang)",
                "expected_sentiment": "positive",
                "expected_confidence": "high",
                "difficulty": "hard",
                "cultural_context": "Mexican casual excitement",
            },
            # Contextual Ambiguity
            {
                "text": "The deadline is tomorrow and I'm just getting started. Perfect timing as always!",
                "language": "English",
                "expected_sentiment": "negative",
                "expected_confidence": "medium",
                "difficulty": "hard",
                "cultural_context": "Stress and sarcastic self-criticism",
            },
            {
                "text": "Il pleut encore aujourd'hui. Décidément, l'été sera magnifique cette année...",
                "language": "French",
                "expected_sentiment": "negative",
                "expected_confidence": "medium",
                "difficulty": "hard",
                "cultural_context": "Weather disappointment with irony",
            },
        ]

        examples = []
        for i, data in enumerate(examples_data):
            example = EvaluationExample(
                input_data={
                    "text": data["text"],
                    "language": data["language"],
                    "context": data.get("cultural_context", ""),
                },
                expected_output={
                    "sentiment": data["expected_sentiment"],
                    "confidence": data["expected_confidence"],
                },
                metadata={
                    "difficulty": data["difficulty"],
                    "language": data["language"],
                    "cultural_context": data["cultural_context"],
                    "example_id": f"sent_{i+1:03d}",
                    "has_sarcasm": "sarcasm" in data["cultural_context"].lower()
                    or "irony" in data["cultural_context"].lower(),
                },
            )
            examples.append(example)

        return Dataset(
            examples=examples,
            name="Multi-Language Sentiment Analysis",
            description=f"Sentiment analysis with {len(examples)} examples across multiple languages and cultural contexts",
        )

    def create_function(self) -> Callable:
        """Create the base sentiment analysis function."""

        def sentiment_analyzer(text: str, language: str, context: str) -> str:
            """Analyze sentiment of text considering language and cultural context."""
            llm = ChatOpenAI(
                model="gpt-4o-mini",
                temperature=0.2,
                max_tokens=100,
            )

            prompt_template = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        """You are an expert sentiment analyst with deep understanding of multiple languages and cultures.

Analyze the sentiment as one of: positive, negative, or neutral.
Also assess confidence level as: high, medium, or low.

Consider:
- Cultural context and idioms
- Sarcasm and irony
- Mixed emotions
- Regional language variations

Respond in this exact format:
Sentiment: [positive/negative/neutral]
Confidence: [high/medium/low]""",
                    ),
                    (
                        "human",
                        """Analyze the sentiment of this text:

Text: {text}
Language: {language}
Context: {context}

Provide sentiment and confidence level.""",
                    ),
                ]
            )

            chain = LLMChain(llm=llm, prompt=prompt_template)
            result = chain.invoke(
                {"text": text, "language": language, "context": context}
            )["text"]

            return result.strip()

        return sentiment_analyzer

    def create_optimized_function(self) -> Callable:
        """Create the optimized sentiment analyzer."""

        @traigent.optimize(
            eval_dataset=self.create_temporary_dataset_file(),
            objectives=self.get_optimization_objectives(),
            configuration_space=self.get_configuration_space(),
            auto_override_frameworks=True,
            framework_targets=["langchain_openai.ChatOpenAI"],
            execution_mode="edge_analytics",
        )
        def sentiment_analyzer_optimized(text: str, language: str, context: str) -> str:
            """Optimized sentiment analyzer."""
            llm = ChatOpenAI(
                model="gpt-4o-mini",  # Will be overridden by Traigent
                temperature=0.2,  # Will be overridden by Traigent
                max_tokens=100,  # Will be overridden by Traigent
            )

            prompt_template = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        """You are an expert sentiment analyst with deep understanding of multiple languages and cultures.

Analyze the sentiment as one of: positive, negative, or neutral.
Also assess confidence level as: high, medium, or low.

Consider:
- Cultural context and idioms
- Sarcasm and irony
- Mixed emotions
- Regional language variations

Respond in this exact format:
Sentiment: [positive/negative/neutral]
Confidence: [high/medium/low]""",
                    ),
                    (
                        "human",
                        """Analyze the sentiment of this text:

Text: {text}
Language: {language}
Context: {context}

Provide sentiment and confidence level.""",
                    ),
                ]
            )

            chain = LLMChain(llm=llm, prompt=prompt_template)
            result = chain.invoke(
                {"text": text, "language": language, "context": context}
            )["text"]

            return result.strip()

        return sentiment_analyzer_optimized

    def evaluate_custom_metrics(
        self,
        outputs: List[Any],
        expected_outputs: List[Any],
        errors: List[Optional[str]],
    ) -> Dict[str, float]:
        """Compute sentiment analysis specific metrics."""
        metrics = {}

        dataset = self.get_dataset()

        # Extract sentiment and confidence from outputs
        def parse_output(output_str):
            """Parse sentiment and confidence from model output."""
            output_str = str(output_str).lower()

            sentiment = "neutral"  # default
            confidence = "medium"  # default

            # Extract sentiment
            if "sentiment:" in output_str:
                sent_line = [
                    line for line in output_str.split("\n") if "sentiment:" in line
                ]
                if sent_line:
                    sent_part = sent_line[0].split("sentiment:")[1].strip()
                    if any(s in sent_part for s in ["positive", "negative", "neutral"]):
                        for s in ["positive", "negative", "neutral"]:
                            if s in sent_part:
                                sentiment = s
                                break

            # Extract confidence
            if "confidence:" in output_str:
                conf_line = [
                    line for line in output_str.split("\n") if "confidence:" in line
                ]
                if conf_line:
                    conf_part = conf_line[0].split("confidence:")[1].strip()
                    if any(c in conf_part for c in ["high", "medium", "low"]):
                        for c in ["high", "medium", "low"]:
                            if c in conf_part:
                                confidence = c
                                break

            return sentiment, confidence

        # Sentiment accuracy
        correct_sentiment = 0
        total_sentiment = 0

        # Language-specific tracking for cross-language consistency
        language_results = {}

        # Sarcasm detection tracking
        sarcasm_correct = 0
        sarcasm_total = 0

        for i, (output, expected, error) in enumerate(
            zip(outputs, expected_outputs, errors)
        ):
            if (
                error is None
                and expected is not None
                and output
                and i < len(dataset.examples)
            ):
                total_sentiment += 1
                example_meta = dataset.examples[i].metadata

                predicted_sentiment, predicted_confidence = parse_output(output)
                expected_sentiment = expected.get("sentiment", "neutral")
                expected.get("confidence", "medium")

                # Sentiment accuracy
                if predicted_sentiment == expected_sentiment:
                    correct_sentiment += 1

                # Track by language for consistency
                language = example_meta.get("language", "unknown")
                if language not in language_results:
                    language_results[language] = {"correct": 0, "total": 0}
                language_results[language]["total"] += 1
                if predicted_sentiment == expected_sentiment:
                    language_results[language]["correct"] += 1

                # Sarcasm detection
                if example_meta.get("has_sarcasm", False):
                    sarcasm_total += 1
                    if predicted_sentiment == expected_sentiment:
                        sarcasm_correct += 1

        metrics["sentiment_accuracy"] = (
            correct_sentiment / total_sentiment if total_sentiment > 0 else 0.0
        )

        # Cross-language consistency (average accuracy across languages)
        language_accuracies = []
        for lang_data in language_results.values():
            if lang_data["total"] > 0:
                accuracy = lang_data["correct"] / lang_data["total"]
                language_accuracies.append(accuracy)

        metrics["cross_language_consistency"] = (
            sum(language_accuracies) / len(language_accuracies)
            if language_accuracies
            else 0.0
        )

        # Sarcasm detection
        metrics["sarcasm_detection"] = (
            sarcasm_correct / sarcasm_total if sarcasm_total > 0 else 1.0
        )

        # Cultural sensitivity (simplified - same as overall accuracy for now)
        metrics["cultural_sensitivity"] = metrics["sentiment_accuracy"]

        # Confidence calibration (simplified)
        metrics["confidence_calibration"] = (
            metrics["sentiment_accuracy"] * 0.9
        )  # Placeholder

        return metrics


# Register this problem
register_problem("sentiment_analysis", SentimentAnalysisProblem)
