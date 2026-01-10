#!/usr/bin/env python
"""Combined DSPy + Tuned Variables Tutorial.

This tutorial demonstrates how to use Traigent's tuned variables system
together with DSPy prompt optimization for a complete optimization workflow.

The scenario: A RAG-based QA system where we want to optimize:
1. **Retriever selection** - Using TunedCallable pattern
2. **LLM parameters** - Using domain presets
3. **Prompt engineering** - Using DSPy integration

Requirements:
    pip install traigent[dspy] traigent-tuned-variables

Usage:
    # With mock LLM (for testing)
    TRAIGENT_MOCK_LLM=true TRAIGENT_OFFLINE_MODE=true python dspy_tuned_variables_combined.py

    # With real LLM
    OPENAI_API_KEY=your-key python dspy_tuned_variables_combined.py
"""

from __future__ import annotations

import os
from dataclasses import dataclass

# Check for mock mode
MOCK_MODE = os.environ.get("TRAIGENT_MOCK_LLM", "").lower() == "true"

# ============================================================================
# Part 1: Define Retriever Functions (to be discovered)
# ============================================================================

# These would typically be in a separate module (e.g., my_retrievers.py)


def traigent_callable(func):
    """Decorator to mark functions as tunable callables."""
    func.__traigent_callable__ = True
    return func


@traigent_callable
def similarity_search(query: str, k: int = 5) -> list[str]:
    """Similarity-based retrieval.

    Tags: dense, semantic
    """
    # Simulated retrieval - in practice, this would use a vector store
    return [f"[Similarity Result {i+1}] Info about {query}" for i in range(k)]


@traigent_callable
def mmr_search(query: str, k: int = 5, lambda_mult: float = 0.5) -> list[str]:
    """Maximal Marginal Relevance retrieval for diversity.

    Tags: dense, diverse
    """
    return [
        f"[MMR Result {i+1}, lambda={lambda_mult}] Diverse info about {query}"
        for i in range(k)
    ]


@traigent_callable
def bm25_search(query: str, k: int = 5) -> list[str]:
    """BM25 sparse retrieval.

    Tags: sparse, keyword
    """
    return [f"[BM25 Result {i+1}] Keyword match for {query}" for i in range(k)]


@traigent_callable
def hybrid_search(query: str, k: int = 5, alpha: float = 0.5) -> list[str]:
    """Hybrid dense+sparse retrieval.

    Tags: hybrid, fusion
    """
    return [
        f"[Hybrid Result {i+1}, alpha={alpha}] Combined info about {query}"
        for i in range(k)
    ]


# ============================================================================
# Part 2: Setup - Discover callables and create TunedCallable
# ============================================================================


def setup_retrievers():
    """Discover and configure retrievers using tuned variables."""
    import sys

    # Create a temporary module with our functions
    # (In practice, you'd import from an actual module)
    import types

    from traigent_tuned_variables import TunedCallable

    from traigent.api.parameter_ranges import Range
    from traigent.tuned_variables import discover_callables_by_decorator

    retriever_module = types.ModuleType("retriever_module")
    retriever_module.similarity_search = similarity_search
    retriever_module.mmr_search = mmr_search
    retriever_module.bm25_search = bm25_search
    retriever_module.hybrid_search = hybrid_search
    for fn in [similarity_search, mmr_search, bm25_search, hybrid_search]:
        fn.__module__ = "retriever_module"
    sys.modules["retriever_module"] = retriever_module

    # Discover callables by decorator
    discovered = discover_callables_by_decorator(retriever_module)
    print(f"Discovered {len(discovered)} retriever functions:")
    for name, info in discovered.items():
        print(f"  - {name}: {info.signature}")
        if info.tags:
            print(f"    Tags: {info.tags}")

    # Create TunedCallable with per-retriever parameters
    retrievers = TunedCallable(
        name="retriever",
        callables={name: info.callable for name, info in discovered.items()},
        parameters={
            "mmr_search": {"lambda_mult": Range(0.0, 1.0, default=0.5)},
            "hybrid_search": {"alpha": Range(0.0, 1.0, default=0.5)},
        },
        description="Retrieval strategy for RAG pipeline",
    )

    return retrievers


# ============================================================================
# Part 3: Setup DSPy Prompt Choices
# ============================================================================


def setup_prompt_choices():
    """Create prompt choices using DSPy integration."""
    from traigent.integrations.dspy_adapter import DSPyPromptOptimizer

    # Define base prompts to optimize over
    base_prompts = [
        "You are a helpful assistant. Answer questions based on the provided context.",
        "You are an expert researcher. Analyze the context carefully and provide accurate answers.",
        "Think step by step. First analyze the context, then formulate your answer.",
        "You are a concise assistant. Answer directly using only information from the context.",
    ]

    # Create prompt choices for Traigent optimization
    prompt_choices = DSPyPromptOptimizer.create_prompt_choices(
        base_prompts=base_prompts,
        name="system_prompt",
    )

    print(f"\nConfigured {len(base_prompts)} prompt choices for optimization")
    return prompt_choices


# ============================================================================
# Part 4: Create the Optimized RAG Pipeline
# ============================================================================


def create_rag_pipeline(retrievers, prompt_choices):
    """Create the optimized RAG pipeline."""
    from traigent_tuned_variables import LLMPresets, RAGPresets

    import traigent

    # Get full config space from TunedCallable, including per-callable params
    # This returns:
    #   "retriever": Choices(["similarity_search", "mmr_search", ...])
    #   "retriever.mmr_search.lambda_mult": Range(0.0, 1.0)
    #   "retriever.hybrid_search.alpha": Range(0.0, 1.0)
    retriever_space = retrievers.get_full_space()

    @traigent.optimize(
        # Retriever selection AND per-callable parameters (from TunedCallable)
        **retriever_space,
        # Additional RAG parameters
        k=RAGPresets.k_retrieval(max_k=8),
        # LLM parameters (from domain presets)
        temperature=LLMPresets.temperature(),
        model=LLMPresets.model(tier="balanced"),
        # Prompt selection (from DSPy integration)
        system_prompt=prompt_choices,
        # Objectives
        objectives=["accuracy", "cost"],
    )
    def rag_qa(question: str) -> str:
        """RAG-based question answering with tuned variables."""
        config = traigent.get_config()

        # 1. Retrieve context using selected retriever
        retriever_name = config["retriever"]
        k = config["k"]

        # Get retriever-specific parameters (e.g., lambda_mult for MMR)
        # extract_callable_params reads dotted keys like "retriever.mmr_search.lambda_mult"
        # and returns {"lambda_mult": 0.7} when retriever="mmr_search"
        retriever_params = retrievers.extract_callable_params(config)

        # Invoke retriever
        docs = retrievers.invoke(retriever_name, question, k=k, **retriever_params)
        context = "\n".join(docs)

        # 2. Generate answer using selected prompt and LLM settings
        system_prompt = config["system_prompt"]
        temperature = config["temperature"]
        model = config["model"]

        # Simulated LLM call (in practice, use actual LLM)
        if MOCK_MODE:
            # Show all config values in mock output for demo
            prompt_preview = (
                system_prompt[:30] + "..." if len(system_prompt) > 30 else system_prompt
            )
            answer = (
                f"[Mock Answer for '{question}' using {retriever_name}, "
                f"model={model}, temp={temperature:.2f}, prompt='{prompt_preview}', "
                f"context_len={len(context)}]"
            )
        else:
            # In production:
            # answer = llm.generate(
            #     system=system_prompt,
            #     user=f"Context:\n{context}\n\nQuestion: {question}",
            #     model=model,
            #     temperature=temperature,
            # )
            answer = f"[Answer for '{question}']"

        return answer

    return rag_qa


# ============================================================================
# Part 5: Analyze Optimization Results
# ============================================================================


def analyze_results(result):
    """Analyze optimization results using VariableAnalyzer."""
    from traigent_tuned_variables import VariableAnalyzer

    print("\n" + "=" * 60)
    print("POST-OPTIMIZATION ANALYSIS")
    print("=" * 60)

    # Create analyzer with objective directions
    analyzer = VariableAnalyzer(
        result,
        directions={
            "accuracy": "maximize",
            "cost": "minimize",
        },
    )

    # Analyze across both objectives
    analysis = analyzer.analyze_multi_objective(
        objectives=["accuracy", "cost"],
        aggregation="mean",
    )

    # Print variable importance
    print("\nVariable Importance (aggregate across objectives):")
    for var_name, var_analysis in sorted(
        analysis.variables.items(), key=lambda x: -x[1].aggregate_importance
    ):
        print(f"  {var_name}: {var_analysis.aggregate_importance:.3f}")
        for obj, imp in var_analysis.importance_by_objective.items():
            print(f"    - {obj}: {imp:.3f}")

    # Print elimination suggestions
    if analysis.elimination_suggestions:
        print("\nElimination Suggestions:")
        for suggestion in analysis.elimination_suggestions:
            print(f"  {suggestion.variable}: {suggestion.action.value}")
            print(f"    Reason: {suggestion.reason}")
            if suggestion.dominated_values:
                print(f"    Dominated values: {suggestion.dominated_values}")
            if suggestion.suggested_values:
                print(f"    Keep values: {suggestion.suggested_values}")

    # Print Pareto frontier values for categorical variables
    if analysis.pareto_frontier_values:
        print("\nPareto Frontier Values (non-dominated):")
        for var_name, values in analysis.pareto_frontier_values.items():
            print(f"  {var_name}: {values}")

    # Get refined configuration space
    refined_space = analyzer.get_refined_space(
        objectives=["accuracy", "cost"],
        prune_low_importance=True,
        prune_dominated_values=True,
        narrow_ranges=True,
    )

    print(f"\nRefined configuration space: {len(refined_space)} variables")
    for var_name, var_range in refined_space.items():
        print(f"  {var_name}: {var_range}")

    return refined_space


# ============================================================================
# Part 6: DSPy Prompt Optimization (Advanced)
# ============================================================================


def run_dspy_optimization(rag_qa, trainset=None):
    """Run DSPy prompt optimization on the best configuration.

    This is an advanced step that uses DSPy's MIPROv2 or BootstrapFewShot
    to further optimize the prompts in the best configuration.
    """
    try:
        import dspy

        from traigent.integrations.dspy_adapter import DSPyPromptOptimizer
    except ImportError:
        print("\nSkipping DSPy optimization (dspy not installed)")
        return None

    print("\n" + "=" * 60)
    print("DSPy PROMPT OPTIMIZATION")
    print("=" * 60)

    # Define a simple DSPy signature and module
    class QASignature(dspy.Signature):
        """Answer questions based on context."""

        context: str = dspy.InputField()
        question: str = dspy.InputField()
        answer: str = dspy.OutputField()

    class QAModule(dspy.Module):
        def __init__(self):
            super().__init__()
            self.qa = dspy.ChainOfThought(QASignature)

        def forward(self, context: str, question: str) -> str:
            return self.qa(context=context, question=question)

    # Create optimizer
    optimizer = DSPyPromptOptimizer(
        method="bootstrap",  # or "mipro" for more thorough optimization
        auto_setting="light",
    )

    # Create sample training data
    if trainset is None:
        trainset = [
            dspy.Example(
                context="The capital of France is Paris.",
                question="What is the capital of France?",
                answer="Paris",
            ).with_inputs("context", "question"),
            dspy.Example(
                context="Python was created by Guido van Rossum in 1991.",
                question="Who created Python?",
                answer="Guido van Rossum",
            ).with_inputs("context", "question"),
        ]

    # Define metric
    def exact_match(example, prediction):
        return float(example.answer.lower() in prediction.answer.lower())

    print(f"Optimizing prompts with {len(trainset)} training examples...")

    # Run optimization (only if DSPy LM is configured)
    try:
        result = optimizer.optimize_prompt(
            module=QAModule(),
            trainset=trainset,
            metric=exact_match,
            max_bootstrapped_demos=2,
        )

        print("Optimization complete!")
        print(f"  Method: {result.method}")
        print(f"  Demos: {result.num_demos}")
        print(f"  Best score: {result.best_score}")

        return result.optimized_module

    except Exception as e:
        print(f"DSPy optimization skipped: {e}")
        return None


# ============================================================================
# Main Execution
# ============================================================================


def main():
    """Run the complete DSPy + Tuned Variables workflow."""
    print("=" * 60)
    print("DSPy + Tuned Variables Combined Tutorial")
    print("=" * 60)

    # Step 1: Setup retrievers using callable discovery
    print("\n[Step 1] Setting up retrievers...")
    retrievers = setup_retrievers()

    # Step 2: Setup prompt choices using DSPy integration
    print("\n[Step 2] Setting up prompt choices...")
    prompt_choices = setup_prompt_choices()

    # Step 3: Create optimized RAG pipeline
    print("\n[Step 3] Creating optimized RAG pipeline...")
    rag_qa = create_rag_pipeline(retrievers, prompt_choices)

    # Step 4: Run optimization (mock mode for demo)
    print("\n[Step 4] Running optimization...")
    if MOCK_MODE:
        print("  (Running in mock mode - simulated optimization)")

        # Simulate a few calls to show the pipeline works
        test_questions = [
            "What is machine learning?",
            "How does RAG work?",
            "What are embeddings?",
        ]

        for q in test_questions:
            answer = rag_qa(q)
            print(f"  Q: {q}")
            print(f"  A: {answer}\n")

        # Create a mock result for analysis demo
        @dataclass
        class MockTrial:
            config: dict
            metrics: dict
            status: str = "completed"

        @dataclass
        class MockResult:
            trials: list
            configuration_space: dict
            objectives: list = None

        # Simulate some trial results WITH per-callable parameters
        # Note the dotted keys like "retriever.mmr_search.lambda_mult"
        mock_trials = [
            MockTrial(
                config={
                    "retriever": "similarity_search",
                    "k": 5,
                    "temperature": 0.7,
                    "model": "gpt-4o-mini",
                    # No per-callable params for similarity_search
                },
                metrics={"accuracy": 0.75, "cost": 0.001},
            ),
            MockTrial(
                config={
                    "retriever": "mmr_search",
                    "k": 5,
                    "temperature": 0.5,
                    "model": "gpt-4o",
                    # Per-callable param: lambda_mult for MMR diversity control
                    "retriever.mmr_search.lambda_mult": 0.7,
                },
                metrics={"accuracy": 0.82, "cost": 0.005},
            ),
            MockTrial(
                config={
                    "retriever": "hybrid_search",
                    "k": 8,
                    "temperature": 0.3,
                    "model": "gpt-4o",
                    # Per-callable param: alpha for dense/sparse fusion weight
                    "retriever.hybrid_search.alpha": 0.6,
                },
                metrics={"accuracy": 0.88, "cost": 0.006},
            ),
            MockTrial(
                config={
                    "retriever": "bm25_search",
                    "k": 3,
                    "temperature": 0.9,
                    "model": "gpt-4o-mini",
                    # No per-callable params for bm25_search
                },
                metrics={"accuracy": 0.65, "cost": 0.0005},
            ),
        ]

        mock_result = MockResult(
            trials=mock_trials,
            configuration_space={
                "retriever": [
                    "similarity_search",
                    "mmr_search",
                    "bm25_search",
                    "hybrid_search",
                ],
                "retriever.mmr_search.lambda_mult": (0.0, 1.0),  # Per-callable param
                "retriever.hybrid_search.alpha": (0.0, 1.0),  # Per-callable param
                "k": (3, 8),
                "temperature": (0.0, 1.0),
                "model": ["gpt-4o-mini", "gpt-4o"],
            },
        )

        # Step 5: Analyze results and get refined space for next iteration
        refined_space = analyze_results(mock_result)

        # The refined_space can be used for iterative optimization
        print(
            f"\nRefined space ready for next iteration: {len(refined_space)} variables"
        )

    else:
        # In production mode, run actual optimization
        # result = rag_qa.optimize()
        # refined_space = analyze_results(result)
        print("  (Set TRAIGENT_MOCK_LLM=true for demo mode)")

    # Step 6: Optional DSPy prompt optimization
    if not MOCK_MODE:
        run_dspy_optimization(rag_qa)

    print("\n" + "=" * 60)
    print("Tutorial Complete!")
    print("=" * 60)
    print(
        """
Key Takeaways:
1. Use `discover_callables()` to auto-discover functions from modules
2. Use `TunedCallable` to manage function-valued variables with per-callable params
3. Use `LLMPresets`, `RAGPresets` for domain-aware parameter ranges
4. Use `DSPyPromptOptimizer.create_prompt_choices()` for prompt optimization
5. Use `VariableAnalyzer` to understand what matters and refine the search space

Next Steps:
- Try with real LLM: unset TRAIGENT_MOCK_LLM and set OPENAI_API_KEY
- Add your own retriever implementations
- Customize the prompt choices for your domain
- Use refined_space for iterative optimization
"""
    )


if __name__ == "__main__":
    main()
