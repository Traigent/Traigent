# Onboarding Plan: Head of AI

**Welcome to TraiGent!**

This document is designed to get you up to speed with the TraiGent SDK, our mission, and the current state of our AI/Optimization stack.

## 1. Mission & Vision
TraiGent is a **zero-code LLM optimization SDK**. We allow developers to optimize their LLM calls (prompts, models, parameters) using simple Python decorators, without changing their core logic.

**Core Value Proposition:**
- **Decorate & Forget:** `@traigent.optimize` handles the complexity.
- **Multi-Objective:** Optimize for accuracy, cost, and latency simultaneously.
- **Local-First Execution:** Run locally (`edge_analytics`) with mock mode for demos; cloud/hybrid orchestration is a roadmap item.

## 2. Architecture Overview
The SDK is built on a few key pillars:
1.  **Interception:** We use Python decorators to intercept function calls.
2.  **Injection:** We inject optimized parameters into the function at runtime.
3.  **Optimization Loop:** We run experiments (trials) to find the best parameters.
4.  **Backend Sync:** We sync results to the TraiGent Cloud (or local storage in dev mode).

### Key Directories
- `traigent/api/`: Public facing decorators (`@optimize`).
- `traigent/core/`: Core logic (`OptimizedFunction`, `Orchestrator`).
- `traigent/optimizers/`: Optimization algorithms (The "Brain").
- `traigent/integrations/`: Adapters for LangChain, OpenAI, etc.

## 3. Current Status: AI Capabilities
Here is a technical breakdown of our current optimization capabilities:

| Capability | Status | Implementation Details |
| :--- | :--- | :--- |
| **Grid Search** | Stable | `traigent/optimizers/grid.py` - Simple, robust, exhaustive. |
| **Random Search** | Stable | `traigent/optimizers/random.py` - Good baseline. |
| **Bayesian Opt (Native)** | **Legacy / Beta** | `traigent/optimizers/bayesian.py` - Custom Gaussian Process implementation. **Known limitations** with categorical variables and multi-objective optimization. Scheduled for replacement. |
| **Optuna Integration** | **Mature / Adapter** | `traigent/optimizers/optuna_adapter.py` - Wraps the industry-standard Optuna library. Supports TPE, CMA-ES, and NSGA-II. Currently available as an alternative backend; planned to become the default. |
| **Smart Sampling** | Beta | `traigent/analytics/subset_selection.py` - TF-IDF & Clustering to select representative test sets. |
| **Meta-Learning** | Alpha | `traigent/analytics/meta_learning.py` - Logic to recommend algorithms based on problem shape. |

## 4. Technical Observations (The "Real" State)
*Based on a recent code audit:*

1.  **Bayesian Optimizer Limitations:** The native `bayesian.py` implementation assumes a maximization problem and has hardcoded convergence limits. It struggles with purely categorical search spaces (e.g., choosing between 'gpt-4' and 'claude-3').
2.  **Optuna Strategy:** We have a robust adapter for Optuna (`optuna_adapter.py`) which solves the categorical variable issue using TPE (Tree-structured Parzen Estimator). There is an active plan (`docs/plans/optuna_integration_plan.md`) to migrate this to be the *core* optimizer, replacing the native Bayesian implementation.
3.  **Async Architecture:** The cloud sync operations are async, but the core optimization loop in `Orchestrator` often runs synchronously in local mode.
4.  **Mock Mode:** For development, we rely heavily on `TRAIGENT_MOCK_MODE=true` to avoid burning API credits.

## 5. Immediate Priorities for You
1.  **Review the Optuna Migration Plan:** Read `docs/plans/optuna_integration_plan.md`. This is a critical infrastructure upgrade.
2.  **Audit the Meta-Learning Module:** `traigent/analytics/meta_learning.py` is ambitious but needs validation against real-world data.
3.  **Standardize Evaluators:** Our evaluation logic (scoring the output of LLMs) needs to be more pluggable.

## 6. Getting Started Guide (Step-by-Step)

### Phase 1: Environment Setup
1.  **Prerequisites**: Python 3.11+ (matches SDK target).
2.  **Installation**:
    ```bash
    git clone https://github.com/Traigent/Traigent.git
    cd Traigent
    python3 -m venv .venv && source .venv/bin/activate
    pip install -e ".[dev,integrations]"
    ```
    *   *Success Criteria*: `python -c "import traigent; print('✅ Installed')"` prints the success message.
3.  **Verification**:
    ```bash
    make test-unit
    ```
    *   *Success Criteria*: All unit tests pass (green).

### Phase 2: Your First Optimization Run
1.  **Navigate**: Go to the simple prompt example.
    ```bash
    cd examples/core/simple-prompt
    ```
2.  **Run in Mock Mode**:
    ```bash
    TRAIGENT_MOCK_MODE=true python run.py
    ```
    *   *Why Mock Mode?* To test the flow without needing API keys immediately.
    *   *Success Criteria*: Script runs, logs show "Optimization started", and a "Best config" is printed at the end.

### Phase 3: Explore the Codebase
1.  **Key File**: Open `traigent/optimizers/optuna_adapter.py`.
    *   *Task*: Understand how we map TraiGent's `ConfigurationSpace` to Optuna's `Trial` object.
2.  **Key File**: Open `traigent/core/orchestrator.py`.
    *   *Task*: Trace the `run_optimization_loop` method to see how trials are executed.
