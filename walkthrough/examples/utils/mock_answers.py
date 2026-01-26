"""Shared mock answers for walkthrough examples.

Maps to datasets in walkthrough/examples/datasets/:
- ANSWERS: simple_questions.jsonl (Easy 10 / Medium 5 / Hard 5)
- CLASSIFICATION_LABELS: classification.jsonl (Clear 10 / Ambiguous 10)
- RAG_ANSWERS: rag_questions.jsonl (Simple 10 / Complex 10)
"""

from __future__ import annotations


def normalize_text(text: str) -> str:
    return text.strip().lower()


def configure_mock_notice(example_file: str) -> None:
    """Customize mock-mode messaging for walkthrough examples."""
    import logging

    logger = logging.getLogger("traigent.evaluators.local")

    class _Filter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            message = record.getMessage()
            if "MOCK LLM MODE: Metrics are simulated" in message:
                return False
            return True

    logger.addFilter(_Filter())
    print(
        "MOCK MODE: Metrics are simulated for "
        f"{example_file}. Best configuration and accuracy are based on "
        "reference runs from walkthrough/examples/real/. "
        f"To run real evaluations, use walkthrough/examples/real/{example_file}."
    )


# simple_questions.jsonl: Easy (10) / Medium (5) / Hard (5)
ANSWERS = {
    # --- Easy (10): Factual single-word answers ---
    "what is 2+2?": "4",
    "what is the capital of france?": "Paris",
    "what color is the sky?": "blue",
    "how many days are in a week?": "7",
    "what is the largest planet?": "Jupiter",
    "what year did world war ii end?": "1945",
    "what is h2o commonly called?": "water",
    "how many continents are there?": "7",
    "what is the speed of light in km/s?": "299792",
    "who wrote romeo and juliet?": "Shakespeare",
    # --- Medium (5): Brief explanations ---
    "what causes ocean tides?": "The gravitational pull of the moon and sun",
    "why is the sky blue?": "Rayleigh scattering of sunlight by atmosphere",
    "how does photosynthesis work?": "Plants convert light energy to chemical energy using chlorophyll",
    "what is machine learning?": "A method where computers learn patterns from data",
    "why do we have seasons?": "Earth's axial tilt causes varying sun angles throughout the year",
    # --- Hard (5): Nuanced reasoning ---
    "explain opportunity cost with an example": "The value of the next best alternative foregone when making a choice",
    "what is the difference between weather and climate?": "Weather is short-term atmospheric conditions; climate is long-term patterns",
    "how does a vaccine work?": "It trains the immune system to recognize and fight pathogens",
    "what is cognitive bias?": "Systematic patterns of deviation from rational judgment",
    "explain the concept of compound interest": "Interest calculated on both principal and accumulated interest",
}

# classification.jsonl: Clear (10) / Ambiguous (10)
CLASSIFICATION_LABELS = {
    # --- Clear positive (5) ---
    "this product is absolutely amazing! best purchase ever!": "positive",
    "i love this service, highly recommend to everyone!": "positive",
    "exceeded my expectations, will definitely buy again.": "positive",
    "perfect quality and fast shipping. 5 stars!": "positive",
    "great value for money, very satisfied customer.": "positive",
    # --- Clear negative (5) ---
    "complete waste of money, don't buy this.": "negative",
    "terrible experience, worst customer service ever.": "negative",
    "product broke after one day. very disappointed.": "negative",
    "horrible quality, nothing like the pictures showed.": "negative",
    "never ordering from here again. total scam.": "negative",
    # --- Ambiguous/neutral (10) ---
    "it's okay, nothing special but gets the job done.": "neutral",
    "not bad, not great. average experience overall.": "neutral",
    "frustrating setup but eventually worked fine.": "neutral",
    "good product but overpriced for what you get.": "neutral",
    "would have been better with more features.": "neutral",
    "mixed feelings. some parts great, others not so much.": "neutral",
    "it works as advertised, nothing more nothing less.": "neutral",
    "decent quality but shipping took forever.": "neutral",
    "happy with the purchase but expected faster delivery.": "neutral",
    "the product is fine but customer support was slow.": "neutral",
}

# rag_questions.jsonl: Simple (10) / Complex (10)
RAG_ANSWERS = {
    # --- Simple (10): Direct factual ---
    "what does traigent do?": "Traigent optimizes AI applications without code changes",
    "what is seamless mode?": "Seamless mode intercepts and overrides hardcoded LLM parameters",
    "what is parameter mode?": "Parameter mode provides explicit configuration control via function parameters",
    "what execution modes does traigent support?": "Local, cloud, and hybrid execution modes",
    "how does local mode work?": "Local mode keeps all data on your machine for complete privacy",
    "what is edge_analytics mode?": "Edge analytics mode runs optimization locally while sending analytics",
    "what optimization algorithms does traigent support?": "Grid search, random search, and Bayesian optimization",
    "what is the @traigent.optimize decorator?": "A decorator that enables automatic optimization of LLM functions",
    "how do you define objectives in traigent?": "Using the objectives parameter with metrics like accuracy, cost, latency",
    "what is a configuration space?": "A dictionary defining the hyperparameters and their possible values to optimize",
    # --- Complex (10): Synthesis required ---
    "how does traigent balance multiple objectives?": "Through weighted objective definitions using ObjectiveSchema with maximize/minimize orientations",
    "explain hybrid mode with privacy": "Hybrid mode runs LLM calls locally but uses cloud for optimization intelligence with privacy enabled",
    "how does traigent handle rag optimization?": "It can optimize both retrieval parameters like k and method, plus generation parameters like model and temperature",
    "what is the difference between seamless and context injection?": "Seamless mode auto-overrides LLM calls; context mode adds config to prompts",
    "how do custom evaluators work?": "Custom evaluators let you define your own scoring logic for specialized use cases",
    "can traigent optimize cost and accuracy simultaneously?": "Yes, through multi-objective optimization with configurable weights for each metric",
    "what privacy features does traigent offer?": "Local storage, privacy_enabled flag, and edge_analytics mode for data control",
    "how does traigent integrate with langchain?": "Via adapters that intercept LangChain LLM calls and inject optimized configurations",
    "what is the eval_dataset parameter?": "A JSONL file or Dataset object containing input/output pairs for evaluation",
    "how does traigent determine the best configuration?": "By running trials, evaluating against objectives, and selecting the config with best weighted score",
}
