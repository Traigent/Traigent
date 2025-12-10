# 🚀 Welcome to TraiGent SDK - Your AI Optimization Journey Starts Here!

I'll walk you through TraiGent SDK step-by-step, as if you're completely new to it. TraiGent is a **zero-code-change optimization platform** that automatically finds the best AI configuration for your specific use case. Think of it as an intelligent tuner that tests different AI models and settings to maximize performance while minimizing costs.

---

## 📖 Chapter 1: Understanding What TraiGent Does

### The Core Problem TraiGent Solves

When building AI applications, you face countless decisions:

- Which model? (GPT-4, GPT-3.5, Claude, etc.)
- What temperature? (0.0 for factual, 0.9 for creative)
- How many retrieval results? (k=3 for speed, k=10 for thoroughness)
- What prompt style works best?

**TraiGent automatically tests all these combinations** to find what works best for YOUR specific task, without you changing your existing code!

### The Magic: Tuned Variables

TraiGent identifies **Tuned Variables** - parameters that affect your AI's behavior and can be optimized:

- ✅ **Model selection** (`gpt-4` vs `gpt-3.5-turbo`)
- ✅ **Temperature** (creativity level)
- ✅ **Token limits** (response length)
- ✅ **Retrieval depth** (RAG k parameter)

### 🧪 Try Example 1: See TraiGent in Action

Run the simple optimization example to see how TraiGent tests different configurations.

---

## 📖 Chapter 2: Installation & Setup

### Step 1: Install TraiGent

```bash
# Clone the repository (recommended for examples)
git clone https://github.com/Traigent/Traigent.git
cd Traigent

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install TraiGent with all integrations
pip install -r requirements/requirements.txt
pip install -r requirements/requirements-integrations.txt
pip install -e .
```

### Step 2: Verify Installation

```bash
# Quick test with mock mode (no API keys needed!)
TRAIGENT_MOCK_MODE=true python examples/core/hello-world/run.py
```

You should see optimization results showing different configurations being tested!

---

## 📖 Chapter 3: Your First TraiGent Optimization

### The Simplest Example - Zero Code Changes!

Here's your existing code:

```python
# your_existing_code.py
from langchain_openai import ChatOpenAI

def answer_question(question: str) -> str:
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
    response = llm.invoke(f"Answer this question concisely: {question}")
    return response.content
```

Now add TraiGent optimization - **your code stays exactly the same**:

```python
import traigent

@traigent.optimize(
    eval_dataset="questions.jsonl",  # Your test questions
    objectives=["accuracy", "cost"],  # What to optimize
    configuration_space={
        "model": ["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o"],
        "temperature": [0.1, 0.5, 0.9]
    }
)
def answer_question(question: str) -> str:
    # EXACT SAME CODE - NO CHANGES!
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
    response = llm.invoke(f"Answer this question concisely: {question}")
    return response.content
```

### Create Your Evaluation Dataset

Create `questions.jsonl`:

```json
{"input": {"question": "What is 2+2?"}, "expected_output": "4"}
{"input": {"question": "Capital of France?"}, "expected_output": "Paris"}
{"input": {"question": "What is machine learning?"}, "expected_output": "A method where computers learn from data"}
```

### Run Optimization

```python
import asyncio

async def main():
    # TraiGent tests different models and temperatures
    results = await answer_question.optimize(max_trials=10)

    print(f"✨ Best configuration found:")
    print(f"   Model: {results.best_config['model']}")
    print(f"   Temperature: {results.best_config['temperature']}")
    print(f"   Accuracy: {results.best_metrics['accuracy']:.2%}")
    print(f"   Cost per call: ${results.best_metrics['cost']:.6f}")

asyncio.run(main())
```

### 🧪 Try Example 2: Zero Code Changes Demo

Experience how TraiGent optimizes your existing code without any modifications!

---

## 📖 Chapter 4: Understanding TraiGent's Two Modes

### Mode 1: Seamless Mode (Default) - Zero Code Changes

TraiGent **automatically intercepts** your LLM calls and overrides parameters:

```python
@traigent.optimize(
    # injection_mode="seamless" is default!
    configuration_space={"model": ["gpt-3.5", "gpt-4"], "temperature": [0.1, 0.9]}
)
def my_agent(text):
    # TraiGent magically overrides these values during optimization
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
    return llm.invoke(text)
```

### Mode 2: Parameter Mode - Explicit Control

For new code or when you want full control:

```python
@traigent.optimize(
    injection_mode="parameter",
    configuration_space={"model": ["gpt-3.5", "gpt-4"], "temperature": [0.1, 0.9]}
)
def my_agent(text, config):  # Note: config parameter added
    # You explicitly use the configuration
    llm = ChatOpenAI(
        model=config.get("model"),
        temperature=config.get("temperature")
    )
    return llm.invoke(text)
```

### 🧪 Try Example 3: Parameter Mode

See how to use explicit configuration control for maximum flexibility.

---

## 📖 Chapter 5: Multi-Objective Optimization

TraiGent can optimize for multiple goals simultaneously:

```python
@traigent.optimize(
    objectives=["accuracy", "cost", "latency"],  # Optimize all three!
    configuration_space={
        "model": ["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o"],
        "temperature": [0.0, 0.5, 1.0],
        "max_tokens": [100, 500, 1000]
    }
)
def smart_agent(query):
    # TraiGent finds the best balance of accuracy, cost, and speed
    pass
```

### Understanding Trade-offs

- **Accuracy vs Cost**: Better models cost more
- **Speed vs Quality**: Faster responses may be less accurate
- **Complexity vs Maintainability**: Simpler configs are easier to manage

### 🧪 Try Example 4: Multi-Objective Optimization

Learn how TraiGent balances multiple competing objectives.

---

## 📖 Chapter 6: Privacy & Execution Modes

### Local Mode - Complete Privacy

Your data never leaves your machine:

```python
@traigent.optimize(
    execution_mode="edge_analytics",  # Everything stays on your computer
    local_storage_path="./my_results"
)
```

### Cloud Mode - Advanced Algorithms

Use TraiGent's cloud for smarter optimization:

```python
@traigent.optimize(
    execution_mode="cloud"  # Leverages Bayesian optimization
)
```

### Hybrid Mode - Best of Both

Local execution with cloud intelligence:

```python
@traigent.optimize(
    execution_mode="hybrid",
    privacy_enabled=True  # Data stays local, only metadata goes to cloud
)
```

### 🧪 Try Example 5: Privacy Modes

Experiment with different execution modes to understand privacy options.

---

## 📖 Chapter 7: Real-World Example with RAG

Here's a complete example with Retrieval-Augmented Generation:

```python
import traigent
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma

@traigent.optimize(
    eval_dataset="customer_queries.jsonl",
    objectives=["accuracy", "cost"],
    configuration_space={
        "model": ["gpt-3.5-turbo", "gpt-4o-mini"],
        "temperature": [0.1, 0.5, 0.9],
        "k": [3, 5, 10]  # Number of documents to retrieve
    }
)
def customer_support(query: str, knowledge_base: list) -> str:
    # Get optimized configuration
    config = traigent.get_trial_config()

    # Use optimized parameters
    llm = ChatOpenAI(
        model=config.get("model", "gpt-3.5-turbo"),
        temperature=config.get("temperature", 0.5)
    )

    # RAG retrieval with optimized k
    vectorstore = Chroma.from_texts(knowledge_base)
    docs = vectorstore.similarity_search(query, k=config.get("k", 5))

    context = "\n".join([doc.page_content for doc in docs])
    prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"

    return llm.invoke(prompt).content
```

### 🧪 Try Example 6: RAG Optimization

See how TraiGent optimizes both LLM and retrieval parameters together.

---

## 📖 Chapter 8: Custom Evaluation

### Define Your Own Success Metrics

```python
def custom_evaluator(output: str, expected: str) -> float:
    """Return score between 0.0 and 1.0"""
    # Your custom logic here
    if "error" in output.lower():
        return 0.0
    similarity = calculate_similarity(output, expected)
    return similarity

@traigent.optimize(
    custom_evaluator=custom_evaluator,
    eval_dataset="custom_data.jsonl"
)
def my_function(input_text):
    return process(input_text)
```

### Common Custom Metrics

- **Semantic Similarity**: Meaning-based comparison
- **Exact Match**: Precise string matching
- **Regex Patterns**: Pattern-based validation
- **Business Logic**: Domain-specific rules

### 🧪 Try Example 7: Custom Evaluator

Create your own evaluation logic for specialized use cases.

---

## 📖 Chapter 9: Performance & Cost Control

### Parallel Execution

Speed up optimization with parallelization:

```python
await my_agent.optimize(
    parallel_config={"trial_concurrency": 4}  # Test 4 configurations simultaneously
)
```

### Cost Budgets

Control your spending:

```python
@traigent.optimize(
    optimization_strategy={
        "max_cost_budget": 10.0,  # Stop after spending $10
        "adaptive_sample_size": True,  # Smart subset selection
        "early_stopping": True  # Stop if no improvement
    }
)
```

### Performance Tips

- Start with small datasets (10-20 examples)
- Use mock mode for initial testing
- Enable parallel execution for speed
- Set cost budgets to control spending

### 🧪 Try Example 8: Performance Optimization

Learn to optimize faster while controlling costs.

---

## 📖 Chapter 10: Complete Real-World Application

### Building a Production-Ready Agent

Let's combine everything into a real application:

```python
import traigent
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@traigent.optimize(
    eval_dataset="production_data.jsonl",
    objectives=["accuracy", "cost", "latency"],
    configuration_space={
        "model": ["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o"],
        "temperature": [0.0, 0.3, 0.7],
        "max_tokens": [150, 300, 500],
        "k": [3, 5, 7],
        "use_cache": [True, False]
    },
    execution_mode="hybrid",
    privacy_enabled=True,
    optimization_strategy={
        "max_cost_budget": 50.0,
        "early_stopping": True,
        "adaptive_sample_size": True
    }
)
def production_agent(
    query: str,
    context: dict,
    knowledge_base: list
) -> dict:
    """Production-ready agent with full optimization."""

    # Get optimized configuration
    config = traigent.get_trial_config()

    # Log configuration for monitoring
    logger.info(f"Using config: {config}")

    # Initialize LLM with optimized parameters
    llm = ChatOpenAI(
        model=config.get("model", "gpt-3.5-turbo"),
        temperature=config.get("temperature", 0.3),
        max_tokens=config.get("max_tokens", 300)
    )

    # RAG retrieval if knowledge base provided
    retrieved_context = ""
    if knowledge_base and len(knowledge_base) > 0:
        vectorstore = Chroma.from_texts(knowledge_base)
        docs = vectorstore.similarity_search(
            query,
            k=config.get("k", 5)
        )
        retrieved_context = "\n".join([d.page_content for d in docs])

    # Build prompt with context
    prompt = f"""
    User Context: {context}
    Retrieved Information: {retrieved_context}

    Question: {query}

    Provide a helpful, accurate answer:
    """

    # Get response
    response = llm.invoke(prompt).content

    # Return structured output
    return {
        "answer": response,
        "config_used": config,
        "model": config.get("model"),
        "confidence": calculate_confidence(response)
    }

def calculate_confidence(response: str) -> float:
    """Calculate confidence score for the response."""
    # Simple heuristic - replace with your logic
    if len(response) < 10:
        return 0.3
    elif "I don't know" in response:
        return 0.4
    elif "I think" in response or "possibly" in response:
        return 0.7
    else:
        return 0.9

# Production deployment
async def deploy_optimized_agent():
    """Deploy the agent with optimal configuration."""

    # Run optimization
    results = await production_agent.optimize(
        algorithm="bayesian",
        max_trials=100,
        parallel_config={"trial_concurrency": 4},
    )

    # Apply best configuration
    production_agent.apply_config(results.best_config)

    # Log results
    logger.info(f"Optimization complete!")
    logger.info(f"Best config: {results.best_config}")
    logger.info(f"Performance: {results.best_metrics}")

    # Save configuration for production
    results.save_config("optimal_config.json")

    return production_agent
```

### 🧪 Try Example 10: Complete Application

Run a full production-ready example with all TraiGent features combined.

---

## 🎯 Quick Reference

### Essential Commands

```bash
# Install TraiGent
pip install -e .

# Run with mock mode
TRAIGENT_MOCK_MODE=true python my_script.py

# Check installation
python scripts/verify_installation.py

# Launch interactive UI
python scripts/launch_control_center.py
```

### Common Patterns

```python
# Basic optimization
@traigent.optimize(
    eval_dataset="data.jsonl",
    objectives=["accuracy"]
)

# Multi-objective
objectives=["accuracy", "cost", "latency"]

# Privacy mode
execution_mode="edge_analytics"

# Cost control
optimization_strategy={"max_cost_budget": 10.0}

# Custom evaluation
custom_evaluator=my_evaluator_function
```

---

## 💡 Key Takeaways

1. **Start Simple**: Use the decorator on existing code - zero changes needed!
2. **Mock Mode First**: Test without API keys using `TRAIGENT_MOCK_MODE=true`
3. **Small Datasets**: Start with 10-20 examples in your evaluation dataset
4. **Multiple Objectives**: Optimize for accuracy AND cost together
5. **Local Privacy**: Use `execution_mode="edge_analytics"` to keep everything on your machine
6. **Interactive UI**: Use the playground for visual experimentation

---

## 🚀 Next Steps

1. Complete all the examples in this walkthrough
2. Explore the `/examples` directory for more use cases
3. Read the architecture guide for deep understanding
4. Join the community to share your experiences

**Congratulations!** You've completed the TraiGent walkthrough. You're now ready to optimize your AI applications with data-driven decisions instead of guesswork!

Remember: TraiGent transforms the way you build AI applications - from guessing parameters to data-driven optimization. Start with your existing code, add the decorator, and watch TraiGent find the perfect configuration for YOUR specific needs! 🎯
