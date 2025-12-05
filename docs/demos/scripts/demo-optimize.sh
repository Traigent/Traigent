#!/bin/bash
# Demo: Traigent LLM Agent Optimization
set -e
cd "$(dirname "$0")/.."
export TERM="${TERM:-xterm-256color}"

clear
echo "# Traigent: Evaluation-Driven LLM Optimization"
echo ""
sleep 1

echo "# Step 1: Define your LLM agent with tuned variables"
echo ""
sleep 0.5

cat << 'PYTHON'
import traigent
from langchain_openai import ChatOpenAI

@traigent.optimize(
    # Define TUNED VARIABLES - parameters to optimize
    configuration_space={
        "model": ["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o"],
        "temperature": [0.1, 0.5, 0.9],
        "max_tokens": [500, 1000, 2000]
    },

    # Define OBJECTIVES - what to optimize for
    objectives=["accuracy", "cost"],

    # Evaluation dataset for testing
    eval_dataset="qa_samples.jsonl"
)
def qa_agent(question: str) -> str:
    """Q&A agent with tunable parameters"""
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",    # Traigent will tune this
        temperature=0.7           # Traigent will tune this
    )
    return llm.invoke(question).content
PYTHON
sleep 3

echo ""
echo "# Step 2: Prepare evaluation dataset (qa_samples.jsonl)"
echo ""
sleep 0.5

cat << 'JSONL'
{"input": {"question": "What is Python?"}, "expected": "A programming language"}
{"input": {"question": "What is 2+2?"}, "expected": "4"}
{"input": {"question": "Capital of France?"}, "expected": "Paris"}
JSONL
sleep 2

echo ""
echo "# Step 3: Run the optimization"
echo ""
sleep 0.5

echo '$ python -c "import asyncio; asyncio.run(qa_agent.optimize())"'
sleep 0.5
echo ""
echo "Starting optimization with grid search..."
echo "Objectives: accuracy, cost"
echo "Configuration space: 27 combinations (3 models x 3 temps x 3 tokens)"
echo ""
sleep 1

echo "[1/27] Testing: gpt-3.5-turbo, temp=0.1, tokens=500"
echo "       Score: accuracy=0.72, cost=\$0.002"
sleep 0.3
echo "[2/27] Testing: gpt-3.5-turbo, temp=0.1, tokens=1000"
echo "       Score: accuracy=0.75, cost=\$0.003"
sleep 0.3
echo "[3/27] Testing: gpt-3.5-turbo, temp=0.5, tokens=500"
echo "       Score: accuracy=0.68, cost=\$0.002"
sleep 0.3
echo "..."
sleep 0.5
echo "[15/27] Testing: gpt-4o-mini, temp=0.1, tokens=1000"
echo "       Score: accuracy=0.89, cost=\$0.008  <- New best!"
sleep 0.3
echo "..."
sleep 0.5
echo "[27/27] Testing: gpt-4o, temp=0.9, tokens=2000"
echo "       Score: accuracy=0.91, cost=\$0.045"
echo ""
sleep 1

echo "Optimization complete!"
echo ""
echo "BEST CONFIGURATION FOUND:"
echo "  model: gpt-4o-mini"
echo "  temperature: 0.1"
echo "  max_tokens: 1000"
echo ""
echo "METRICS:"
echo "  accuracy: 0.89 (89%)"
echo "  cost: \$0.008 per query"
echo ""
echo "Compared to default (gpt-3.5-turbo, temp=0.7):"
echo "  +17% accuracy improvement"
echo "  Best cost/accuracy trade-off"
sleep 3

echo ""
echo "# Step 4: Apply best configuration"
echo ""
sleep 0.5

cat << 'PYTHON'
# Apply the optimized configuration
results = await qa_agent.optimize()
qa_agent.apply_best_config(results)

# Now qa_agent uses: gpt-4o-mini, temp=0.1, tokens=1000
answer = qa_agent("What is machine learning?")
PYTHON
sleep 2
