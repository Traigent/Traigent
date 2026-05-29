# Traigent SDK Architecture

## How it Works

Traigent is a zero-code LLM optimization SDK that automatically finds the best configuration for your AI workflows. It uses advanced optimization algorithms to explore parameter spaces while respecting constraints and budgets.

### Key Capabilities

- **Parameter Optimization**: Automatically tune model selection, temperature, prompts, and other parameters
- **Multi-Objective Optimization**: Balance accuracy, cost, latency, and other metrics simultaneously
- **Framework Integration**: Native support for LangChain, DSPy, OpenAI, Anthropic, and more
- **Flexible Execution**: Run locally (edge analytics) or in the cloud
- **TVL Specifications**: Define optimization specs in YAML for version control and collaboration

---

## High-Level Architecture

```mermaid
flowchart TB
    subgraph UserCode["Your Application"]
        decorator["@traigent.optimize()"]
        func["LLM Function/Agent"]
        dataset["Evaluation Dataset"]
    end

    subgraph TraigentSDK["Traigent SDK"]
        subgraph API["Public API Layer"]
            ranges["Parameter Ranges<br/>(Range, Choices, IntRange)"]
            constraints["Constraints<br/>(TVL Expressions)"]
            objectives["Objectives<br/>(accuracy, cost, latency)"]
        end

        subgraph Core["Optimization Core"]
            orchestrator["Optimization<br/>Orchestrator"]
            optimized_func["Optimized<br/>Function"]
            trial_lifecycle["Trial<br/>Lifecycle"]
            cost_enforcer["Cost<br/>Enforcer"]
        end

        subgraph Optimizers["Optimization Algorithms"]
            tpe["TPE<br/>(Bayesian)"]
            grid["Grid<br/>Search"]
            random["Random<br/>Search"]
            nsga["NSGA-II<br/>(Multi-Obj)"]
            cmaes["CMA-ES"]
        end

        subgraph Evaluation["Evaluation Engine"]
            evaluator["Local<br/>Evaluator"]
            metrics_tracker["Metrics<br/>Tracker"]
            custom_eval["Custom<br/>Evaluators"]
        end

        subgraph Config["Configuration"]
            injection["Config<br/>Injection"]
            context["Trial<br/>Context"]
            providers["Config<br/>Providers"]
        end

        subgraph Integrations["Framework Integrations"]
            langchain["LangChain"]
            dspy["DSPy"]
            openai["OpenAI"]
            anthropic["Anthropic"]
            more["+ More"]
        end
    end

    subgraph Output["Results"]
        best_config["Best Configuration"]
        pareto["Pareto Front<br/>(Multi-Objective)"]
        insights["Optimization<br/>Insights"]
        analytics["Analytics &<br/>Telemetry"]
    end

    decorator --> optimized_func
    func --> optimized_func
    dataset --> evaluator

    ranges --> orchestrator
    constraints --> orchestrator
    objectives --> orchestrator

    optimized_func --> orchestrator
    orchestrator --> trial_lifecycle
    trial_lifecycle --> cost_enforcer

    orchestrator <--> tpe
    orchestrator <--> grid
    orchestrator <--> random
    orchestrator <--> nsga
    orchestrator <--> cmaes

    trial_lifecycle --> evaluator
    evaluator --> metrics_tracker
    evaluator --> custom_eval

    trial_lifecycle --> injection
    injection --> context
    context --> providers

    providers --> langchain
    providers --> dspy
    providers --> openai
    providers --> anthropic
    providers --> more

    orchestrator --> best_config
    orchestrator --> pareto
    orchestrator --> insights
    orchestrator --> analytics

    style UserCode fill:#e1f5fe,stroke:#01579b
    style TraigentSDK fill:#fff3e0,stroke:#e65100
    style Output fill:#e8f5e9,stroke:#2e7d32
    style Core fill:#fce4ec,stroke:#c2185b
    style Optimizers fill:#f3e5f5,stroke:#7b1fa2
    style Evaluation fill:#e8eaf6,stroke:#3f51b5
    style Integrations fill:#e0f2f1,stroke:#00695c
```

---

## Optimization Flow

```mermaid
sequenceDiagram
    participant User as Your Code
    participant OF as OptimizedFunction
    participant Orch as Orchestrator
    participant Opt as Optimizer
    participant Eval as Evaluator
    participant LLM as LLM Provider

    User->>OF: @traigent.optimize()
    User->>OF: func.optimize(dataset)

    OF->>Orch: Initialize optimization

    loop Until stopping condition
        Orch->>Opt: suggest_next_trial(history)
        Opt-->>Orch: config candidate

        Orch->>Orch: Acquire cost permit

        Orch->>Eval: evaluate(config, dataset)

        loop For each example
            Eval->>LLM: Execute with config
            LLM-->>Eval: Response + metrics
            Eval->>Eval: Run custom evaluators
        end

        Eval-->>Orch: TrialResult (metrics)
        Orch->>Opt: Update with result
        Orch->>Orch: Check stopping conditions
    end

    Orch-->>OF: OptimizationResult
    OF-->>User: Best config + insights
```

---

## Configuration Injection Modes

Traigent supports multiple ways to inject optimized configurations into your functions:

```mermaid
flowchart LR
    subgraph Modes["Injection Modes"]
        context["Context Mode<br/>(Default)"]
        param["Parameter Mode"]
        seamless["Seamless Mode<br/>(AST Transform)"]
    end

    subgraph Usage["How You Access Config"]
        context --> ctx_code["traigent.get_config()"]
        param --> param_code["def func(query, config):"]
        seamless --> seamless_code["Direct variable names<br/>(temperature, model)"]
    end

    style context fill:#e3f2fd
    style param fill:#f3e5f5
    style seamless fill:#e8f5e9
```

> A fourth function-attribute-based mode existed in v1.x but was removed in v2.x because it could not be made thread-safe under parallel trials. The wrapper still exposes the read-only `OptimizedFunction.current_config` property to inspect the active configuration after `apply_best_config()` — that is a separate surface, not an injection mode. See [User Guide / Section 4](../user-guide/injection_modes.md#4-attribute-mode-removed-in-v2x) for migration guidance.

---

## Execution Modes

```mermaid
flowchart TB
    subgraph EdgeAnalytics["Edge Analytics Mode"]
        direction TB
        ea_desc["Local optimization<br/>Data stays on-premise<br/>Optional telemetry"]
        ea_local["Local Storage"]
        ea_opt["Local Optimizer"]
    end

    subgraph CloudMode["Cloud Mode"]
        direction TB
        cloud_desc["Reserved future remote execution<br/>Not available today<br/>Fails closed"]
        cloud_backend["Future Traigent Cloud"]
        cloud_storage["Future Cloud Storage"]
    end

    subgraph HybridMode["Hybrid Mode"]
        direction TB
        hybrid_desc["Local trial execution<br/>Backend session/result tracking<br/>Portal-visible results"]
        hybrid_local["Local Execution"]
        hybrid_sync["Backend Sync"]
    end

    User["Your Application"] --> EdgeAnalytics
    User --> CloudMode
    User --> HybridMode

    style EdgeAnalytics fill:#e8f5e9,stroke:#2e7d32
    style CloudMode fill:#e3f2fd,stroke:#1565c0
    style HybridMode fill:#fff3e0,stroke:#ef6c00
```

---

## Multi-Objective Optimization

```mermaid
flowchart LR
    subgraph Objectives["Define Objectives"]
        acc["Accuracy<br/>(maximize)"]
        cost["Cost<br/>(minimize)"]
        latency["Latency<br/>(minimize)"]
    end

    subgraph Aggregation["Aggregation Modes"]
        weighted["Weighted Sum<br/>(Linear trade-offs)"]
        harmonic["Harmonic Mean<br/>(Penalize imbalance)"]
        chebyshev["Chebyshev<br/>(Worst-case focus)"]
    end

    subgraph Results["Output"]
        pareto["Pareto Front"]
        best["Best Trade-off"]
        insights["Trade-off Analysis"]
    end

    acc --> weighted
    cost --> weighted
    latency --> weighted

    acc --> harmonic
    cost --> harmonic
    latency --> harmonic

    acc --> chebyshev
    cost --> chebyshev
    latency --> chebyshev

    weighted --> pareto
    harmonic --> pareto
    chebyshev --> pareto

    pareto --> best
    pareto --> insights

    style Objectives fill:#ffebee
    style Aggregation fill:#e8eaf6
    style Results fill:#e8f5e9
```

---

## TVL Specification System

TVL (Tuned Variable Language) lets you define optimization specs in YAML:

```mermaid
flowchart TB
    subgraph TVLSpec["optimization.tvl.yaml"]
        tvars["tvars:<br/>- name: model<br/>- name: temperature"]
        objectives_tvl["objectives:<br/>- accuracy<br/>- cost"]
        constraints_tvl["constraints:<br/>- require: temp <= 0.7"]
        exploration["exploration:<br/>  budget: 50"]
    end

    subgraph Loader["TVL Loader"]
        parser["YAML Parser"]
        validator["Schema Validator"]
        compiler["Constraint Compiler"]
    end

    subgraph Runtime["Runtime"]
        config_space["Configuration Space"]
        constraint_eval["Constraint Evaluator"]
        objective_schema["Objective Schema"]
    end

    TVLSpec --> parser
    parser --> validator
    validator --> compiler

    compiler --> config_space
    compiler --> constraint_eval
    compiler --> objective_schema

    style TVLSpec fill:#fff8e1,stroke:#f57f17
    style Loader fill:#e3f2fd,stroke:#1565c0
    style Runtime fill:#f3e5f5,stroke:#7b1fa2
```

---

## Component Relationships

```mermaid
graph TB
    subgraph PublicAPI["Public API"]
        optimize["@optimize decorator"]
        configure["traigent.configure()"]
        get_config["traigent.get_config()"]
    end

    subgraph CoreEngine["Core Engine"]
        opt_func["OptimizedFunction"]
        orch["Orchestrator"]
        trial["TrialLifecycle"]
    end

    subgraph Strategies["Pluggable Strategies"]
        opt_alg["Optimizers<br/>(TPE, Grid, Random, NSGA-II)"]
        eval_strat["Evaluators<br/>(Local, Cloud)"]
        inject_mode["Injection<br/>(Context, Param, Seamless)"]
    end

    subgraph Infrastructure["Infrastructure"]
        cost["CostEnforcer"]
        budget["SampleBudget"]
        metrics["MetricsTracker"]
        trace["Tracing"]
    end

    subgraph External["External Systems"]
        llm_providers["LLM Providers"]
        observability["Observability<br/>(MLflow, W&B)"]
        backend["Traigent Backend"]
    end

    optimize --> opt_func
    configure --> orch
    get_config --> trial

    opt_func --> orch
    orch --> trial

    orch --> opt_alg
    trial --> eval_strat
    trial --> inject_mode

    trial --> cost
    trial --> budget
    eval_strat --> metrics
    orch --> trace

    eval_strat --> llm_providers
    trace --> observability
    orch --> backend

    style PublicAPI fill:#e1f5fe,stroke:#01579b
    style CoreEngine fill:#fce4ec,stroke:#c2185b
    style Strategies fill:#f3e5f5,stroke:#7b1fa2
    style Infrastructure fill:#fff3e0,stroke:#e65100
    style External fill:#e0f2f1,stroke:#00695c
```

---

## Quick Start Example

```python
import traigent
from traigent import Range, Choices

@traigent.optimize(
    # Define parameter search space
    model=Choices(["gpt-4o", "gpt-4o-mini", "claude-3-sonnet"]),
    temperature=Range(0.0, 1.0),

    # Set optimization objectives
    objectives=["accuracy", "cost"],

    # Configure evaluation
    evaluation={"eval_dataset": "test_cases.jsonl"},

    # Set budget
    exploration={"budget": 50}
)
def my_agent(query: str) -> str:
    config = traigent.get_config()
    # Your LLM logic here
    return response

# Run optimization
result = my_agent.optimize()
print(f"Best config: {result.best_config}")
print(f"Accuracy: {result.best_metrics['accuracy']:.2%}")
```

---

## Learn More

- [Examples](../../examples/)
