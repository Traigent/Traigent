"""Shared mock answers for walkthrough examples.

Maps to datasets in walkthrough/datasets/:
- ANSWERS: simple_questions.jsonl (Easy 10 / Medium 5 / Hard 5)
- CLASSIFICATION_LABELS: classification.jsonl (Clear 10 / Ambiguous 10)
- RAG_ANSWERS: rag_questions.jsonl (Simple 10 / Complex 10)
"""

from __future__ import annotations

from contextvars import ContextVar

# Default model for mock trials
DEFAULT_MOCK_MODEL = "gpt-3.5-turbo"

# Context variable for current mock trial model (works with async/parallel execution)
_current_mock_model: ContextVar[str] = ContextVar("current_mock_model", default=DEFAULT_MOCK_MODEL)


def set_mock_model(model: str) -> None:
    """Set the current model for mock trial context.

    Call this from the decorated function to track which model is being tested.
    The scoring function can then use get_current_mock_model() to get the model.

    Note: Uses contextvars.ContextVar which properly isolates values across
    concurrent/parallel execution (unlike threading.local).
    """
    _current_mock_model.set(model)


def get_current_mock_model() -> str:
    """Get the current model from mock trial context.

    Returns:
        The model set by set_mock_model(), or DEFAULT_MOCK_MODEL as default.
    """
    return _current_mock_model.get()


# Model-dependent accuracy for mock mode (more capable models score higher)
MOCK_MODEL_ACCURACY = {
    # OpenAI Models (as of Jan 2026)
    "gpt-4o": 0.90,
    "gpt-4o-mini": 0.82,
    "gpt-3.5-turbo": 0.75,
    "gpt-4": 0.88,
    "gpt-4-turbo": 0.87,
    "gpt-4.1-nano": 0.70,       # Released Apr 2025, optimized for speed
    "gpt-5-nano": 0.73,         # Released Aug 2025, improved over 4.1-nano
    "gpt-5.1": 0.91,            # Released Nov 2025, strong reasoning
    "gpt-5.2": 0.93,            # Released Dec 2025, best accuracy
    # Anthropic models
    "claude-3-opus-20240229": 0.92,
    "claude-sonnet-4-20250514": 0.88,
    "claude-3-5-sonnet-20241022": 0.87,
    "claude-3-sonnet-20240229": 0.85,
    "claude-3-5-haiku-20241022": 0.78,
    # Google Gemini models
    "gemini-1.5-flash": 0.80,         # Fast and capable
    "gemini-1.5-pro": 0.89,           # High accuracy model
    "gemini-2.0-flash-exp": 0.85,     # Experimental next-gen flash
}

# Costs per 1K tokens (synced with cost_estimator.py, as of Jan 2026)
MOCK_MODEL_COSTS = {
    "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    "gpt-4o": {"input": 0.0025, "output": 0.01},
    "gpt-4.1-nano": {"input": 0.0001, "output": 0.0004},  # Released Apr 2025
    "gpt-5-nano": {"input": 0.00008, "output": 0.0003},   # Released Aug 2025
    "gpt-5.1": {"input": 0.002, "output": 0.008},         # Released Nov 2025
    "gpt-5.2": {"input": 0.003, "output": 0.012},         # Released Dec 2025
    "gpt-4": {"input": 0.03, "output": 0.06},
    "gpt-4-turbo": {"input": 0.01, "output": 0.03},
    # Corrected: 3.5 Haiku pricing ($0.80/$4.00 per 1M tokens)
    "claude-3-5-haiku-20241022": {"input": 0.0008, "output": 0.004},
    "claude-3-5-sonnet-20241022": {"input": 0.003, "output": 0.015},
    "claude-3-sonnet-20240229": {"input": 0.003, "output": 0.015},
    "claude-3-opus-20240229": {"input": 0.015, "output": 0.075},
    "claude-sonnet-4-20250514": {"input": 0.003, "output": 0.015},
    # Google Gemini models (Standard tier <= 128k context)
    # Note: Prices double if context length > 128k tokens
    "gemini-1.5-flash": {"input": 0.000075, "output": 0.0003},
    "gemini-1.5-pro": {"input": 0.00125, "output": 0.005},
    "gemini-2.0-flash-exp": {"input": 0.000075, "output": 0.0003},  # Often free in AI Studio
}

# Average tokens by task type
MOCK_TASK_TOKENS = {
    "simple_qa": {"input": 50, "output": 30},
    "classification": {"input": 80, "output": 10},
    "rag_qa": {"input": 2000, "output": 100},
    "code_generation": {"input": 150, "output": 300},
    "privacy_test": {"input": 100, "output": 50},
}

# Latency estimates per LLM API call in seconds (based on typical response times)
MOCK_MODEL_LATENCY = {
    # OpenAI Models (as of Jan 2026)
    "gpt-4o": 0.45,           # Fast, newer architecture (~0.4-0.5s avg)
    "gpt-4o-mini": 0.25,      # Extremely fast, comparable to Haiku
    "gpt-3.5-turbo": 0.35,    # Fast, but 4o-mini is often faster now
    "gpt-4-turbo": 0.8,       # Slower than 4o ("Turbo" is older tech now)
    "gpt-4": 1.5,             # Legacy GPT-4 is notoriously slow
    "gpt-4.1-nano": 0.15,     # Ultra-low latency, released Apr 2025
    "gpt-5-nano": 0.12,       # Released Aug 2025, faster than 4.1-nano
    "gpt-5.1": 0.5,           # Released Nov 2025, improved reasoning
    "gpt-5.2": 0.6,           # Released Dec 2025, complex reasoning model
    # Anthropic Models
    "claude-sonnet-4-20250514": 0.9,  # Slower TTFT than GPT-4o (~1.0s avg)
    "claude-3-5-sonnet-20241022": 0.85,  # Similar to Sonnet 4
    "claude-3-sonnet-20240229": 0.8,
    "claude-3-opus-20240229": 1.8,  # Heavy thinker, very slow start
    "claude-3-5-haiku-20241022": 0.3,  # Very fast
    # Google Gemini Models
    "gemini-1.5-flash": 0.2,            # Very fast, optimized for speed
    "gemini-1.5-pro": 0.5,              # Balanced speed and quality
    "gemini-2.0-flash-exp": 0.15,       # Next-gen fast model
}


def get_mock_accuracy(
    model: str,
    task_type: str = "simple_qa",
    temperature: float | None = None,
    use_cot: bool | None = None,
) -> float:
    """Get model-dependent mock accuracy with config-based variation.

    Args:
        model: Model name
        task_type: Type of task (affects base accuracy)
        temperature: Optional temperature setting (lower = more accurate for factual)
        use_cot: Optional chain-of-thought flag (CoT typically improves accuracy)

    Returns:
        Mock accuracy score between 0 and 1
    """
    base_accuracy = MOCK_MODEL_ACCURACY.get(model, 0.75)

    # Task-specific adjustments (harder tasks reduce accuracy slightly)
    task_modifiers = {
        "simple_qa": 1.0,
        "classification": 0.98,
        "rag_qa": 0.95,
        "code_generation": 0.90,
        "privacy_test": 1.0,
    }
    modifier = task_modifiers.get(task_type, 1.0)

    # Temperature effect: lower temperature improves accuracy for factual tasks
    # Effect is smaller for creative tasks
    if temperature is not None:
        # temp 0.0 -> +3% accuracy, temp 1.0 -> -2% accuracy
        temp_effect = 0.03 - (temperature * 0.05)
        modifier += temp_effect

    # Chain-of-thought typically improves accuracy by 2-5%
    if use_cot is True:
        modifier += 0.03

    return min(base_accuracy * modifier, 1.0)


def get_mock_cost(
    model: str, task_type: str = "simple_qa", dataset_size: int = 20
) -> float:
    """Calculate realistic mock cost based on model pricing.

    Args:
        model: Model name
        task_type: Type of task (affects token estimates)
        dataset_size: Number of examples in dataset

    Returns:
        Estimated cost in dollars
    """
    costs = MOCK_MODEL_COSTS.get(model, {"input": 0.001, "output": 0.002})
    tokens = MOCK_TASK_TOKENS.get(task_type, {"input": 50, "output": 30})

    # Cost per example
    input_cost = (tokens["input"] / 1000) * costs["input"]
    output_cost = (tokens["output"] / 1000) * costs["output"]
    example_cost = input_cost + output_cost

    # Total cost for dataset
    return example_cost * dataset_size


def get_mock_latency(model: str, task_type: str = "simple_qa") -> float:
    """Get estimated latency for a model in mock mode.

    Args:
        model: Model name
        task_type: Type of task (affects latency estimate)

    Returns:
        Estimated latency in seconds per call
    """
    base_latency = MOCK_MODEL_LATENCY.get(model, 0.5)

    # Task-specific latency multipliers (more tokens = more time)
    task_multipliers = {
        "simple_qa": 1.0,
        "classification": 0.8,   # Short outputs
        "rag_qa": 2.5,           # Long context
        "code_generation": 3.0,  # Long outputs
        "privacy_test": 1.0,
    }
    multiplier = task_multipliers.get(task_type, 1.0)
    return base_latency * multiplier


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
        f"MOCK MODE: Running {example_file} with simulated LLM responses. "
        "Results demonstrate the optimization workflow but are not real metrics. "
        f"For real evaluations, use walkthrough/real/{example_file}."
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
    "what execution modes does traigent support?": "Edge analytics mode (local execution with anonymized metrics). Cloud and hybrid modes are planned for a future release",
    "how does local mode work?": "Local mode keeps all data on your machine for complete privacy",
    "what is edge_analytics mode?": "Edge analytics mode runs optimization locally while sending analytics",
    "what optimization algorithms does traigent support?": "Grid search, random search, and Bayesian optimization",
    "what is the @traigent.optimize decorator?": "A decorator that enables automatic optimization of LLM functions",
    "how do you define objectives in traigent?": "Using the objectives parameter with metrics like accuracy, cost, latency",
    "what is a configuration space?": "A dictionary defining the hyperparameters and their possible values to optimize",
    # --- Complex (10): Synthesis required ---
    "how does traigent balance multiple objectives?": "Through weighted objective definitions using ObjectiveSchema with maximize/minimize orientations",
    "explain hybrid mode with privacy": "Hybrid mode is planned for a future release. It will run LLM calls locally but use cloud for optimization intelligence with privacy enabled",
    "how does traigent handle rag optimization?": "It can optimize both retrieval parameters like k and method, plus generation parameters like model and temperature",
    "what is the difference between seamless and context injection?": "Seamless mode auto-overrides LLM calls; context mode adds config to prompts",
    "how do custom evaluators work?": "Custom evaluators let you define your own scoring logic for specialized use cases",
    "can traigent optimize cost and accuracy simultaneously?": "Yes, through multi-objective optimization with configurable weights for each metric",
    "what privacy features does traigent offer?": "Local storage, privacy_enabled flag, and edge_analytics mode for data control",
    "how does traigent integrate with langchain?": "Via adapters that intercept LangChain LLM calls and inject optimized configurations",
    "what is the eval_dataset parameter?": "A JSONL file or Dataset object containing input/output pairs for evaluation",
    "how does traigent determine the best configuration?": "By running trials, evaluating against objectives, and selecting the config with best weighted score",
}
