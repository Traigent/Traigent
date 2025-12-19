---
stepsCompleted: ['step-01-validate-prerequisites', 'step-02-design-epics', 'step-03-create-stories', 'step-04-final-validation']
inputDocuments: ['docs/PRD_Agentic_Workflow_Tuning_Haystack.docx']
validationStatus: 'PASSED'
totalEpics: 7
totalStories: 46
frsCovered: 28
nfrsCovered: 7
---

# Agentic Workflow Tuning: Haystack Integration - Epic Breakdown

## Overview

This document provides the complete epic and story breakdown for the Haystack Integration feature, decomposing the requirements from the PRD into implementable stories organized by the 7 milestones.

## Requirements Inventory

### Functional Requirements

**FR-100 Series: Pipeline Introspection**
- FR-101: System SHALL extract all components from a Haystack Pipeline object
- FR-102: System SHALL identify component types (Generator, Retriever, Router, etc.)
- FR-103: System SHALL extract init parameters from each component
- FR-104: System SHALL identify which parameters are tunable vs. fixed
- FR-105: System SHALL extract pipeline graph structure (edges between components)
- FR-106: System SHALL detect loops and extract max_runs_per_component settings
- FR-107: System SHALL support custom components decorated with @component

**FR-200 Series: Configuration Space Definition**
- FR-201: System SHALL allow users to specify tunable parameters via TVL syntax
- FR-202: System SHALL support categorical variables (e.g., model selection)
- FR-203: System SHALL support numerical variables (continuous and discrete)
- FR-204: System SHALL support conditional variables (e.g., if model=X, then max_tokens in range Y)
- FR-205: System SHALL allow users to fix certain parameters (exclude from search)
- FR-206: System SHALL validate configuration space for consistency
- FR-207: System SHALL provide auto-discovery mode with sensible defaults

**FR-300 Series: Optimization Execution**
- FR-301: System SHALL run pipeline with different configurations against eval dataset
- FR-302: System SHALL support configurable search strategies (grid, random, Bayesian, evolutionary)
- FR-303: System SHALL track cost (token usage, API calls) per experiment run
- FR-304: System SHALL track latency (p50, p95, p99) per experiment run
- FR-305: System SHALL compute user-defined quality metrics per run
- FR-306: System SHALL support early stopping when constraints are violated
- FR-307: System SHALL support parallel experiment execution
- FR-308: System SHALL checkpoint progress for resumable runs

**FR-400 Series: Results & Export**
- FR-401: System SHALL return Pareto frontier of optimal configurations
- FR-402: System SHALL rank configurations by user-specified primary objective
- FR-403: System SHALL export configurations as TVL files
- FR-404: System SHALL provide apply() function to update Pipeline with config
- FR-405: System SHALL export experiment history with all metrics
- FR-406: System SHALL provide per-node attribution scores

### NonFunctional Requirements

- NFR-1: Performance - Introspection latency <100ms for 20-component pipeline
- NFR-2: Performance - Optimization throughput: Support 100+ trials/hour with parallel=4
- NFR-3: Performance - Memory usage <2GB for 1000-run history
- NFR-4: Usability - Integration requires ≤10 lines of code change to existing pipelines
- NFR-5: Coverage - Support ≥90% of standard Haystack component types
- NFR-6: Quality - Achieve ≥10% improvement on primary metric within budget constraints
- NFR-7: Efficiency - Reduce time-to-optimize from weeks (manual) to hours (automated)

### Additional Requirements

**Dependencies:**
- haystack-ai>=2.0.0 (Pipeline integration, Apache 2.0)
- optuna>=3.0.0 (Bayesian optimization, MIT)
- networkx>=3.0 (Graph analysis, BSD)
- pandas>=2.0.0 (Results analysis, BSD)
- pydantic>=2.0.0 (Data validation, MIT)
- tiktoken>=0.5.0 (Token counting for OpenAI, MIT)
- anthropic>=0.20.0 (Token counting for Anthropic, MIT)

**Internal Dependencies:**
- traigent-core: TVL parser, config validation, base optimizer interfaces
- traigent-metrics: Standard metric implementations (F1, BLEU, semantic similarity)
- traigent-storage: Experiment storage, checkpointing

**Error Handling Requirements:**
- API Rate Limit: Exponential backoff with jitter, max 5 retries
- API Timeout: Retry once, then mark run as failed
- Invalid Config: Skip config, log warning
- Component Error: Catch exception, mark run failed with worst-case score
- Budget Exhausted: Stop optimization immediately, return best results
- >50% Runs Failed: Raise OptimizationError, abort with diagnostic info

**Async Support:**
- Must support both synchronous Pipeline and AsyncPipeline

### FR Coverage Map

| FR | Epic | Description |
|----|------|-------------|
| FR-101 | Epic 1 | Extract components from Pipeline |
| FR-102 | Epic 1 | Identify component types |
| FR-103 | Epic 1 | Extract init parameters |
| FR-104 | Epic 1 | Identify tunable vs. fixed params |
| FR-105 | Epic 1 | Extract pipeline graph structure |
| FR-106 | Epic 1 | Detect loops and max_runs |
| FR-107 | Epic 1 | Support @component custom components |
| FR-201 | Epic 2 | TVL syntax for tunable parameters |
| FR-202 | Epic 2 | Categorical variables support |
| FR-203 | Epic 2 | Numerical variables support |
| FR-204 | Epic 2 | Conditional variables support |
| FR-205 | Epic 2 | Fix parameters (exclude from search) |
| FR-206 | Epic 2 | Validate config space consistency |
| FR-207 | Epic 2 | Auto-discovery with defaults |
| FR-301 | Epic 3 | Run pipeline with configs |
| FR-302 | Epic 3/5 | Search strategies (basic in E3, advanced in E5) |
| FR-303 | Epic 4 | Track cost per run |
| FR-304 | Epic 4 | Track latency per run |
| FR-305 | Epic 3 | Compute quality metrics |
| FR-306 | Epic 4 | Early stopping on constraint violation |
| FR-307 | Epic 5 | Parallel experiment execution |
| FR-308 | Epic 5 | Checkpoint for resumable runs |
| FR-401 | Epic 5 | Pareto frontier of configs |
| FR-402 | Epic 5 | Rank by primary objective |
| FR-403 | Epic 7 | Export configs as TVL |
| FR-404 | Epic 3/7 | apply() function (basic in E3, production in E7) |
| FR-405 | Epic 7 | Export experiment history |
| FR-406 | Epic 6 | Per-node attribution scores |

## Epic List

### Epic 1: Pipeline Discovery & Analysis
ML Engineers can point Traigent at any Haystack pipeline and automatically discover all tunable parameters without manual inspection.

**FRs Covered:** FR-101, FR-102, FR-103, FR-104, FR-105, FR-106, FR-107
**NFRs Addressed:** NFR-1 (introspection <100ms), NFR-5 (≥90% component coverage)

---

### Epic 2: Configuration Space & TVL
ML Engineers can define, customize, and validate optimization search spaces using TVL syntax, with sensible auto-discovered defaults.

**FRs Covered:** FR-201, FR-202, FR-203, FR-204, FR-205, FR-206, FR-207
**NFRs Addressed:** NFR-4 (≤10 lines of code change)

---

### Epic 3: Basic Optimization Execution
ML Engineers can run optimization experiments against their evaluation dataset using grid or random search strategies, with basic apply() capability for testing.

**FRs Covered:** FR-301, FR-302 (grid/random), FR-305, FR-404 (basic)
**NFRs Addressed:** NFR-7 (reduce time-to-optimize)

---

### Epic 4: Cost, Latency & Constraints
ML Engineers can set cost/latency constraints and get production-viable configurations that meet their budget.

**FRs Covered:** FR-303, FR-304, FR-306
**NFRs Addressed:** NFR-6 (≥10% improvement within budget)

---

### Epic 5: Advanced Optimization & Pareto Analysis
ML Engineers can use Bayesian/evolutionary search and receive Pareto-optimal configurations balancing multiple objectives, with parallel execution and resumable runs.

**FRs Covered:** FR-302 (Bayesian/evolutionary), FR-307, FR-308, FR-401, FR-402
**NFRs Addressed:** NFR-2 (100+ trials/hour), NFR-3 (<2GB memory)

---

### Epic 6: Attribution & Insights
ML Engineers can identify which pipeline components are causing quality/cost/latency issues and get optimization recommendations.

**FRs Covered:** FR-406
**User Story Addressed:** US-3 (understand which components cause issues)

---

### Epic 7: Production Readiness & CI/CD
Platform Engineers can integrate Traigent into CI/CD pipelines with CLI, structured output, production-hardened apply(), and artifact storage.

**FRs Covered:** FR-403, FR-404 (production), FR-405
**User Story Addressed:** US-4 (CI/CD integration)

---

## Epic 1: Pipeline Discovery & Analysis

ML Engineers can point Traigent at any Haystack pipeline and automatically discover all tunable parameters without manual inspection.

### Story 1.1: Extract Components from Pipeline

**As an** ML Engineer,
**I want** to pass my Haystack Pipeline object to Traigent and get a list of all components,
**So that** I can see what's in my pipeline without manually inspecting the code.

**Acceptance Criteria:**

**Given** a valid Haystack Pipeline object with multiple components
**When** I call `from_pipeline(pipeline)`
**Then** the system returns a ConfigSpace object
**And** `config_space.components` contains all component names in the pipeline
**And** each component includes its class type (e.g., OpenAIGenerator, InMemoryBM25Retriever)

**Given** an empty Pipeline object
**When** I call `from_pipeline(pipeline)`
**Then** the system returns a ConfigSpace with empty components list without error

**FRs:** FR-101, FR-102

---

### Story 1.2: Extract Component Parameters

**As an** ML Engineer,
**I want** to see all init parameters for each component in my pipeline,
**So that** I know what values can potentially be tuned.

**Acceptance Criteria:**

**Given** a Pipeline with an OpenAIGenerator component
**When** I introspect the pipeline
**Then** the system extracts parameters: model, temperature, max_tokens, top_p, etc.
**And** each parameter includes its current value and Python type

**Given** a component with complex nested parameters
**When** I introspect the pipeline
**Then** the system extracts top-level parameters only (not nested objects)

**FR:** FR-103

---

### Story 1.3: Identify Tunable vs Fixed Parameters

**As an** ML Engineer,
**I want** the system to automatically identify which parameters are tunable vs fixed,
**So that** I don't waste time trying to optimize non-configurable settings.

**Acceptance Criteria:**

**Given** a component with various parameter types
**When** I introspect the pipeline
**Then** parameters with type hints (int, float, str, Literal) are marked as tunable
**And** parameters that are callables, objects, or document_store are marked as fixed
**And** the system provides sensible default ranges for tunable parameters

**Given** a parameter with `Literal["gpt-4o", "gpt-4o-mini"]` type hint
**When** I introspect the pipeline
**Then** the parameter is marked as categorical with those specific choices

**FR:** FR-104

---

### Story 1.4: Extract Pipeline Graph Structure

**As an** ML Engineer,
**I want** to see how components are connected in my pipeline,
**So that** I understand the data flow and can reason about optimization impact.

**Acceptance Criteria:**

**Given** a Pipeline with connected components (A → B → C)
**When** I introspect the pipeline
**Then** the system returns edge information showing connections
**And** the graph structure is accessible as a NetworkX DiGraph

**Given** a Pipeline with branching (A → B, A → C)
**When** I introspect the pipeline
**Then** the system correctly captures both branches

**FR:** FR-105

---

### Story 1.5: Detect Loops and Max Runs

**As an** ML Engineer,
**I want** the system to detect loops in my pipeline and extract max_runs_per_component,
**So that** optimization can account for bounded iteration.

**Acceptance Criteria:**

**Given** a Pipeline with a loop (e.g., agent retry logic)
**When** I introspect the pipeline
**Then** the system identifies the loop structure
**And** extracts max_runs_per_component settings

**Given** a Pipeline without loops
**When** I introspect the pipeline
**Then** the system reports no loops detected

**Given** a cyclic Pipeline without max_runs_per_component set
**When** I introspect the pipeline
**Then** the system raises ValueError indicating unbounded optimization is not supported

**FR:** FR-106

---

### Story 1.6: Support Custom @component Decorated Components

**As an** ML Engineer,
**I want** Traigent to recognize my custom components decorated with @component,
**So that** my custom logic is included in optimization.

**Acceptance Criteria:**

**Given** a Pipeline with a custom `@component` decorated class
**When** I introspect the pipeline
**Then** the system extracts the component and its __init__ parameters
**And** marks appropriate parameters as tunable based on type hints

**Given** a custom component without type hints
**When** I introspect the pipeline
**Then** the system includes the component but marks parameters as requiring manual specification

**FR:** FR-107

---

### Story 1.7: Introspection Performance and Component Coverage

**As an** ML Engineer,
**I want** pipeline introspection to be fast and support the majority of standard Haystack components,
**So that** auto-discovery is practical for real pipelines.

**Acceptance Criteria:**

**Given** a 20-component pipeline on the reference environment
**When** I call `from_pipeline(pipeline)`
**Then** introspection completes in under 100ms (NFR-1)

**Given** the v1 supported component list from the PRD
**When** I run component coverage tests
**Then** at least 90% of standard Haystack component types are auto-discovered (NFR-5)

**NFRs:** NFR-1, NFR-5

---

## Epic 2: Configuration Space & TVL

ML Engineers can define, customize, and validate optimization search spaces using TVL syntax, with sensible auto-discovered defaults.

### Story 2.1: Define ConfigSpace Data Model

**As an** ML Engineer,
**I want** a ConfigSpace object that holds all tunable parameters with their types and ranges,
**So that** I have a structured representation of the optimization search space.

**Acceptance Criteria:**

**Given** a ConfigSpace object
**When** I inspect its parameters
**Then** each parameter has: name, component, type, default value, and constraints (choices/range)

**Given** an introspected pipeline
**When** I call `from_pipeline(pipeline)`
**Then** the return value is a valid ConfigSpace object with all discovered parameters

**FR:** FR-201 (partial - data model foundation)

---

### Story 2.2: Support Categorical Variables

**As an** ML Engineer,
**I want** to define categorical parameters with specific choices (e.g., model selection),
**So that** the optimizer samples only from valid options.

**Acceptance Criteria:**

**Given** a parameter like `generator.model`
**When** I define it as categorical with choices `["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"]`
**Then** the ConfigSpace only samples from those exact values

**Given** a Literal type hint in the original component
**When** auto-discovery runs
**Then** the parameter is automatically typed as categorical with the Literal values as choices

**FR:** FR-202

---

### Story 2.3: Support Numerical Variables

**As an** ML Engineer,
**I want** to define numerical parameters with ranges (continuous floats, discrete integers),
**So that** the optimizer searches within meaningful bounds.

**Acceptance Criteria:**

**Given** a parameter like `generator.temperature`
**When** I define it as float with range [0.0, 2.0]
**Then** the ConfigSpace samples continuous values within that range

**Given** a parameter like `retriever.top_k`
**When** I define it as integer with range [1, 50]
**Then** the ConfigSpace samples only integer values within that range

**Given** a parameter with `log_scale=True`
**When** sampling occurs
**Then** values are sampled uniformly in log space (useful for learning rates, etc.)

**FR:** FR-203

---

### Story 2.4: Support Conditional Variables

**As an** ML Engineer,
**I want** to define conditional relationships between parameters,
**So that** certain parameters only apply when others have specific values.

**Acceptance Criteria:**

**Given** a conditional like "if model=gpt-4o, then max_tokens range is [100, 8192]"
**When** model=gpt-4o is sampled
**Then** max_tokens is sampled from [100, 8192]

**Given** a conditional like "if model=gpt-4o-mini, then max_tokens range is [100, 4096]"
**When** model=gpt-4o-mini is sampled
**Then** max_tokens is constrained to [100, 4096]

**Given** an invalid conditional configuration
**When** validation runs
**Then** the system reports the conflict clearly

**FR:** FR-204

---

### Story 2.5: Fix Parameters (Exclude from Search)

**As an** ML Engineer,
**I want** to explicitly fix certain parameters so they're excluded from optimization,
**So that** I can lock down values I don't want changed.

**Acceptance Criteria:**

**Given** a ConfigSpace with parameter `generator.model`
**When** I call `config_space.fix("generator.model", "gpt-4o")`
**Then** that parameter is removed from the search space
**And** all sampled configs use the fixed value

**Given** a fixed parameter
**When** I call `config_space.unfix("generator.model")`
**Then** the parameter is restored to the search space with its original range

**FR:** FR-205

---

### Story 2.6: Validate Configuration Space Consistency

**As an** ML Engineer,
**I want** the system to validate my ConfigSpace for consistency,
**So that** I catch configuration errors before running expensive experiments.

**Acceptance Criteria:**

**Given** a ConfigSpace with valid parameters and conditionals
**When** I call `config_space.validate()`
**Then** the system returns True

**Given** a ConfigSpace with conflicting conditionals
**When** I call `config_space.validate()`
**Then** the system raises ValidationError with clear description of the conflict

**Given** a ConfigSpace with a parameter range where low > high
**When** I call `config_space.validate()`
**Then** the system raises ValidationError identifying the invalid range

**Given** a ConfigSpace with zero tunable parameters
**When** I call `config_space.validate()`
**Then** the system raises ValueError indicating no tunable components were found

**FR:** FR-206

---

### Story 2.7: Search Space TVL Syntax

**As an** ML Engineer,
**I want** to specify parameter ranges and choices using Search Space TVL syntax,
**So that** I can customize and share search space definitions in a portable format.

**Acceptance Criteria:**

**Given** a Search Space TVL file with parameter range specifications
**When** I load it with `ConfigSpace.from_search_tvl("search_space.tvl")`
**Then** the ConfigSpace is configured according to the TVL definitions

**Given** a ConfigSpace object
**When** I call `config_space.to_search_tvl("search_space.tvl")`
**Then** the file contains a valid Search Space TVL representation (ranges, choices, conditionals)

**Given** a Search Space TVL file with syntax errors
**When** I try to load it
**Then** the system raises a clear parsing error with line number

**Note:** Search Space TVL (Epic 2) defines parameter *ranges* for optimization. Tuned Config TVL (Epic 7) captures *specific values* + metrics as artifacts. These are distinct schemas.

**FR:** FR-201

---

### Story 2.8: Auto-Discovery with Sensible Defaults

**As an** ML Engineer,
**I want** the system to provide sensible default ranges when auto-discovering parameters,
**So that** I can start optimizing immediately without manual configuration.

**Acceptance Criteria:**

**Given** an OpenAIGenerator with `temperature` parameter
**When** auto-discovery runs
**Then** the default range is [0.0, 2.0] (OpenAI's valid range)

**Given** a retriever with `top_k` parameter
**When** auto-discovery runs
**Then** the default range is [1, 50] with default value matching the component's current setting

**Given** a known model parameter (e.g., `generator.model`)
**When** auto-discovery runs with provider catalog available
**Then** the choices include common models from that provider's hardcoded catalog

**Given** a model parameter when provider catalog is unavailable (offline)
**When** auto-discovery runs
**Then** the parameter uses the component's current value as the single choice
**And** a warning is logged suggesting manual specification of model choices

**FR:** FR-207

---

### Story 2.9: Support Boolean and Optional Parameters

**As an** ML Engineer,
**I want** boolean and Optional parameters to be handled correctly during auto-discovery,
**So that** all common parameter types are supported without manual intervention.

**Acceptance Criteria:**

**Given** a component with a `bool` parameter (e.g., `scale_score: bool`)
**When** auto-discovery runs
**Then** the parameter is typed as categorical with choices `[True, False]`

**Given** a component with an `Optional[T]` parameter (e.g., `Optional[int]`)
**When** auto-discovery runs
**Then** the parameter includes `None` as a valid choice alongside the T-type range

**Given** a parameter with `Optional[str]` and no default
**When** auto-discovery runs
**Then** the parameter is typed as categorical with `[None]` plus any Literal choices if present

**FR:** FR-202, FR-203 (extended type coverage)

---

## Epic 3: Basic Optimization Execution

ML Engineers can run optimization experiments against their evaluation dataset using grid or random search strategies, with basic apply() capability for testing.

### Story 3.1: Define Evaluation Dataset Format

**As an** ML Engineer,
**I want** to provide my evaluation dataset in a simple format,
**So that** the optimizer can run my pipeline against test cases.

**Acceptance Criteria:**

**Given** a list of dictionaries with `input` and `expected` keys
**When** I pass it to `optimize(eval_dataset=data)`
**Then** the system accepts the dataset and validates the format

**Given** an evaluation dataset with missing keys
**When** I pass it to `optimize()`
**Then** the system raises a ValidationError with clear message about missing fields

**Given** an empty evaluation dataset
**When** I pass it to `optimize()`
**Then** the system raises a ValueError indicating at least one example is required

**FR:** FR-301 (partial - dataset handling)

---

### Story 3.2: Execute Pipeline with Configuration

**As an** ML Engineer,
**I want** the system to run my pipeline with a specific configuration against the eval dataset,
**So that** each configuration can be scored.

**Acceptance Criteria:**

**Given** a Pipeline, ConfigSpace, and eval dataset
**When** the optimizer samples a configuration
**Then** the system injects the config values into the pipeline
**And** runs the pipeline for each example in the dataset
**And** collects all outputs

**Given** a pipeline execution that raises an exception
**When** running an experiment
**Then** the system catches the error, logs it, and marks the run as failed

**FR:** FR-301

---

### Story 3.3: Basic Apply Function for Testing

**As an** ML Engineer,
**I want** to apply a configuration to my pipeline,
**So that** I can test configurations during development.

**Acceptance Criteria:**

**Given** a Pipeline and a configuration dict
**When** I call `apply_config(pipeline, config)`
**Then** the pipeline's component parameters are updated to match the config
**And** the modified pipeline is returned

**Given** a config with a parameter that doesn't exist in the pipeline
**When** I call `apply_config()`
**Then** the system raises KeyError with the invalid parameter name

**FR:** FR-404 (basic)

---

### Story 3.4: Grid Search Strategy

**As an** ML Engineer,
**I want** to use grid search to exhaustively explore a discretized configuration space,
**So that** I can systematically evaluate all combinations.

**Acceptance Criteria:**

**Given** a ConfigSpace with categorical and discretized numerical params
**When** I call `optimize(strategy='grid')`
**Then** the system generates all combinations of parameter values
**And** evaluates each combination

**Given** a grid search that would exceed n_trials
**When** running optimization
**Then** the system samples n_trials configurations from the grid

**FR:** FR-302 (grid)

---

### Story 3.5: Random Search Strategy

**As an** ML Engineer,
**I want** to use random search to sample configurations from the search space,
**So that** I can efficiently explore large configuration spaces.

**Acceptance Criteria:**

**Given** a ConfigSpace
**When** I call `optimize(strategy='random', n_trials=50)`
**Then** the system randomly samples 50 configurations
**And** evaluates each configuration

**Given** random search with a seed
**When** I run the same optimization twice with the same seed
**Then** the sampled configurations are identical (reproducibility)

**FR:** FR-302 (random)

---

### Story 3.6: User-Defined Quality Metrics

**As an** ML Engineer,
**I want** to provide my own metric function to score pipeline outputs,
**So that** optimization targets my specific quality criteria.

**Acceptance Criteria:**

**Given** a metric function `metric(output, expected) -> float`
**When** I pass it to `optimize(metric=my_metric)`
**Then** the system calls my function for each (output, expected) pair
**And** aggregates scores across the eval dataset

**Given** a metric function that returns NaN or raises an exception
**When** evaluating a run
**Then** the system treats that run as failed with worst-case score

**Given** no metric function provided
**When** I call `optimize()`
**Then** the system raises ValueError indicating metric is required

**FR:** FR-305

---

### Story 3.7: Basic OptimizationResult Container

**As an** ML Engineer,
**I want** the optimization to return a result object with the best configuration and scores,
**So that** I can inspect and use the results.

**Acceptance Criteria:**

**Given** a completed optimization run
**When** I access `result.best_config`
**Then** I get the configuration dict that achieved the highest score

**Given** a completed optimization run
**When** I access `result.best_score`
**Then** I get the float score achieved by the best configuration

**Given** a completed optimization run
**When** I call `result.apply(pipeline)`
**Then** the best configuration is applied to the pipeline

**Given** a ConfigSpace with only one valid configuration
**When** I call `optimize()`
**Then** the system skips search, evaluates the single config once, and returns it as `result.best_config`

**FR:** FR-301 (results), FR-404 (basic apply)

---

### Story 3.8: Execution Error Handling and Failure Thresholds

**As an** ML Engineer,
**I want** optimization runs to handle execution errors predictably,
**So that** failures do not corrupt results or waste compute.

**Acceptance Criteria:**

**Given** an API rate limit error during a trial
**When** the run executes
**Then** the system retries with exponential backoff and jitter up to 5 times

**Given** an API timeout during a trial
**When** the run executes
**Then** the system retries once and then marks the run as failed

**Given** an invalid configuration sampled by the optimizer
**When** the run starts
**Then** the system skips the config, logs a warning, and excludes the run from results

**Given** a component error during a run
**When** execution fails
**Then** the run is marked failed with worst-case score and recorded in history

**Given** more than 50% of runs fail in an optimization session
**When** optimization is still in progress
**Then** the system raises OptimizationError and aborts with diagnostic info

**FR:** FR-301 (execution robustness)

---

### Story 3.9: AsyncPipeline Support

**As an** ML Engineer,
**I want** to optimize AsyncPipeline instances,
**So that** asynchronous Haystack workflows are supported.

**Acceptance Criteria:**

**Given** an AsyncPipeline and evaluation dataset
**When** I call `optimize(pipeline=async_pipe, ...)`
**Then** the system executes using async pipeline runs and collects metrics

**Given** equivalent Pipeline and AsyncPipeline configurations
**When** I run optimization with the same config and metric
**Then** the resulting scores are equivalent within tolerance

**FR:** FR-301 (async execution support)

---

## Epic 4: Cost, Latency & Constraints

ML Engineers can set cost/latency constraints and get production-viable configurations that meet their budget.

### Story 4.1: Track Token Usage and Cost Per Run

**As an** ML Engineer,
**I want** the system to track token usage and compute cost for each experiment run,
**So that** I know how much each configuration costs.

**Acceptance Criteria:**

**Given** a pipeline with OpenAI/Anthropic generators
**When** an experiment run completes
**Then** the system records total input tokens, output tokens, and API calls
**And** computes cost using provider pricing (configurable)

**Given** a pipeline with multiple generator components
**When** an experiment run completes
**Then** the system aggregates costs across all components
**And** provides per-component cost breakdown in NodeMetrics

**Given** a custom component without token tracking
**When** an experiment run completes
**Then** the system records zero cost for that component with a warning

**FR:** FR-303

---

### Story 4.2: Track Latency Metrics Per Run

**As an** ML Engineer,
**I want** the system to track latency (p50, p95, p99) for each experiment run,
**So that** I can find configurations that meet my response time requirements.

**Acceptance Criteria:**

**Given** an experiment run with multiple eval examples
**When** the run completes
**Then** the system computes p50, p95, p99 latency across all examples

**Given** per-node tracing enabled
**When** an experiment run completes
**Then** each component's latency is tracked separately in NodeMetrics

**Given** a pipeline execution timeout
**When** latency exceeds timeout_seconds
**Then** the run is marked as failed with latency recorded as timeout value

**FR:** FR-304

---

### Story 4.3: Define Cost and Latency Constraints

**As an** ML Engineer,
**I want** to specify cost and latency constraints for optimization,
**So that** only production-viable configurations are considered.

**Acceptance Criteria:**

**Given** a constraint `Constraint('cost_per_query', '<=', 0.01)`
**When** passed to `optimize(constraints=[...])`
**Then** the system validates configs against this constraint

**Given** a constraint `Constraint('latency_p95_ms', '<=', 3000)`
**When** passed to `optimize()`
**Then** configs with p95 latency > 3000ms are marked as constraint-violating

**Given** multiple constraints
**When** optimizing
**Then** all constraints must be satisfied for a config to be considered valid

**FR:** FR-306 (partial - constraint definition)

---

### Story 4.4: Early Stopping on Constraint Violation

**As an** ML Engineer,
**I want** the optimizer to skip further evaluation when constraints are clearly violated,
**So that** I don't waste compute on non-viable configurations.

**Acceptance Criteria:**

**Given** a config that violates cost constraint on the first eval example
**When** running the experiment
**Then** the system may stop early and mark the run as constraint-violated

**Given** early stopping enabled
**When** >50% of examples exceed latency constraint
**Then** the run is stopped early with partial results recorded

**Given** early stopping disabled
**When** running optimization
**Then** all examples are evaluated regardless of constraint violations

**FR:** FR-306

---

### Story 4.5: Constraint Satisfaction in Results

**As an** ML Engineer,
**I want** to see which configurations satisfy my constraints,
**So that** I can choose from production-viable options.

**Acceptance Criteria:**

**Given** a completed optimization with constraints
**When** I access `result.best_config`
**Then** the best config is the highest-scoring config that satisfies ALL constraints

**Given** a completed optimization
**When** I access `run.constraints_satisfied` for any run
**Then** I get a boolean indicating whether all constraints were met

**Given** a run that violated constraints
**When** I access `run.constraint_violations`
**Then** I get a list of which constraints were violated and by how much

**FR:** FR-306

---

## Epic 5: Advanced Optimization & Pareto Analysis

ML Engineers can use Bayesian/evolutionary search and receive Pareto-optimal configurations balancing multiple objectives, with parallel execution and resumable runs.

### Story 5.1: Bayesian Optimization Strategy

**As an** ML Engineer,
**I want** to use Bayesian optimization to intelligently explore the search space,
**So that** I find good configurations faster than random search.

**Acceptance Criteria:**

**Given** a ConfigSpace
**When** I call `optimize(strategy='bayesian', n_trials=50)`
**Then** the system uses Optuna's TPE sampler to guide the search
**And** each trial informs subsequent sampling decisions

**Given** Bayesian optimization with conditional parameters
**When** optimizing
**Then** all sampled configurations satisfy the declared conditionals
**And** no invalid configurations are generated or evaluated

**Given** a ConfigSpace with only categorical parameters
**When** using Bayesian optimization
**Then** categorical params are sampled only from declared choices

**FR:** FR-302 (Bayesian)

---

### Story 5.2: Evolutionary Optimization Strategy (NSGA-II)

**As an** ML Engineer,
**I want** to use NSGA-II evolutionary algorithm for multi-objective optimization,
**So that** I can explore complex search spaces with many local optima.

**Acceptance Criteria:**

**Given** a ConfigSpace with multiple targets
**When** I call `optimize(strategy='evolutionary', n_trials=100)`
**Then** the system uses NSGA-II for multi-objective optimization

**Given** evolutionary optimization with population_size parameter
**When** optimizing
**Then** the system maintains a population of that size across generations

**Given** a single-objective optimization
**When** using evolutionary strategy
**Then** NSGA-II degrades gracefully to single-objective mode

**FR:** FR-302 (evolutionary)

---

### Story 5.3: Parallel Experiment Execution

**As an** ML Engineer,
**I want** to run multiple experiment trials in parallel,
**So that** I can complete optimization faster with available compute.

**Acceptance Criteria:**

**Given** `optimize(n_parallel=4)`
**When** running optimization
**Then** up to 4 configurations are evaluated concurrently

**Given** parallel execution with shared state
**When** using Bayesian optimization
**Then** trials are synchronized so the surrogate model stays updated

**Given** a trial that fails during parallel execution
**When** other trials are running
**Then** the failure is recorded without affecting other trials

**Given** n_parallel=4 on reference environment
**When** running 100 trials
**Then** throughput is ≥100 trials/hour (NFR-2)

**FR:** FR-307

---

### Story 5.4: Checkpoint and Resume Optimization

**As an** ML Engineer,
**I want** to checkpoint optimization progress and resume later,
**So that** I don't lose work if the process is interrupted.

**Acceptance Criteria:**

**Given** `optimize(checkpoint_path='./checkpoints/')`
**When** optimization runs
**Then** progress is saved after each completed trial

**Given** an interrupted optimization with checkpoint
**When** I call `optimize()` with the same checkpoint_path
**Then** the system resumes from the last saved state
**And** no trials are re-evaluated

**Given** a corrupted checkpoint file
**When** attempting to resume
**Then** the system raises a clear error with recovery suggestions

**FR:** FR-308

---

### Story 5.5: Multi-Objective Targets

**As an** ML Engineer,
**I want** to specify multiple optimization targets,
**So that** I can balance quality, cost, and latency trade-offs.

**Acceptance Criteria:**

**Given** multiple targets like `[Target('quality', 'maximize'), Target('cost', 'minimize')]`
**When** passed to `optimize(targets=[...])`
**Then** the system optimizes for all objectives simultaneously

**Given** conflicting targets (e.g., quality vs cost)
**When** optimization completes
**Then** the system returns trade-off configurations on the Pareto frontier

**FR:** FR-401 (partial - multi-objective setup)

---

### Story 5.6: Pareto Frontier Computation

**As an** ML Engineer,
**I want** to see all Pareto-optimal configurations,
**So that** I can choose the best trade-off for my use case.

**Acceptance Criteria:**

**Given** a completed multi-objective optimization
**When** I access `result.pareto_configs`
**Then** I get a list of all non-dominated configurations

**Given** Pareto configs with constraints defined
**When** I access `result.pareto_configs`
**Then** only configurations where `constraints_satisfied == True` are included

**Given** all evaluated configurations violate constraints
**When** I access `result.pareto_configs`
**Then** the list is empty and a warning is recorded in `result.warnings`

**Given** Pareto configs
**When** I access `result.pareto_scores`
**Then** each config has scores for all objectives as a dict

**Given** a single-objective optimization
**When** I access `result.pareto_configs`
**Then** I get a list containing only the best config

**FR:** FR-401

---

### Story 5.7: Rank Configurations by Primary Objective

**As an** ML Engineer,
**I want** configurations ranked by my primary objective,
**So that** I can easily find the best config for my main goal.

**Acceptance Criteria:**

**Given** targets with the first target as primary
**When** optimization completes
**Then** `result.best_config` is the config with best primary objective score (among constraint-satisfying configs)

**Given** `result.history`
**When** I iterate over runs
**Then** runs are in chronological order (by start time)

**Given** `result.ranked_runs`
**When** I iterate over runs
**Then** runs are ordered by primary objective score (descending for maximize, ascending for minimize)

**FR:** FR-402

---

### Story 5.8: Memory Efficiency for Large Run History

**As an** ML Engineer,
**I want** the system to handle large optimization runs efficiently,
**So that** I can run extensive searches without running out of memory.

**Acceptance Criteria:**

**Given** an optimization with 1000 completed trials
**When** the optimization completes
**Then** total memory usage is <2GB (NFR-3)

**Given** a long-running optimization
**When** accessing result.history
**Then** older runs can be lazily loaded from checkpoint if memory is constrained

**FR:** NFR-3 (memory efficiency)

---

### Story 5.9: Time-Budgeted Optimization

**As an** ML Engineer,
**I want** to set a time budget for optimization runs,
**So that** tuning completes within hours instead of days.

**Acceptance Criteria:**

**Given** `optimize(timeout_seconds=3600)`
**When** the time budget is exhausted
**Then** optimization stops immediately and returns the best results so far

**Given** a time-budgeted run with checkpointing enabled
**When** the time budget is exhausted
**Then** the latest checkpoint is saved and marked as complete

**NFR:** NFR-7 (time-to-optimize in hours)

---

### Story 5.10: Optimization Effectiveness Benchmark

**As an** ML Engineer,
**I want** a reference benchmark that demonstrates measurable optimization gains,
**So that** I can validate the system meets success criteria.

**Acceptance Criteria:**

**Given** the reference RAG pipeline and evaluation dataset
**When** I run optimization with constraints enabled
**Then** the best configuration improves the primary metric by at least 10% over baseline
**And** constraints remain satisfied

**NFR:** NFR-6 (>=10% improvement within budget constraints)

---

## Epic 6: Attribution & Insights

ML Engineers can identify which pipeline components are causing quality/cost/latency issues and get optimization recommendations.

### Story 6.1: Per-Node Metrics Collection (Per-Run)

**As an** ML Engineer,
**I want** to see metrics for each component in my pipeline for each run,
**So that** I can identify which components contribute most to cost/latency.

**Acceptance Criteria:**

**Given** a completed experiment run
**When** I access `run.node_metrics`
**Then** I get a dict mapping component names to NodeMetrics objects (per-run data)

**Given** NodeMetrics for a generator component
**When** I inspect it
**Then** it includes: invocation_count, total_latency_ms, total_cost, total_input_tokens, total_output_tokens, error_count

**Given** a component that wasn't invoked (e.g., conditional branch not taken)
**When** I access its NodeMetrics
**Then** all values are zero

**Note:** `run.node_metrics` is per-run data. `result.attribution` (Story 6.2-6.5) is aggregate analysis across all runs.

**FR:** FR-406 (partial - metrics collection)

---

### Story 6.2: Quality Contribution Analysis (Aggregate)

**As an** ML Engineer,
**I want** to understand each component's contribution to overall quality,
**So that** I know which components to focus optimization efforts on.

**Acceptance Criteria:**

**Given** a completed optimization with multiple runs
**When** I access `result.attribution[component_name].quality_contribution`
**Then** I get a score from -1 to 1 indicating impact on quality
**And** positive values indicate the component improves quality when tuned

**Given** attribution analysis
**When** the system computes quality_contribution
**Then** it uses permutation importance: measuring quality delta when component params are shuffled across runs

**Given** `result.attribution_ranked`
**When** I iterate over it
**Then** components are ordered by absolute quality_contribution (highest impact first)

**FR:** FR-406

---

### Story 6.3: Cost and Latency Contribution Analysis

**As an** ML Engineer,
**I want** to see each component's fraction of total cost and latency,
**So that** I can identify bottlenecks.

**Acceptance Criteria:**

**Given** a completed optimization
**When** I access `result.attribution[component_name].cost_contribution`
**Then** I get a float (0-1) representing fraction of total cost (averaged across runs)

**Given** a completed optimization
**When** I access `result.attribution[component_name].latency_contribution`
**Then** I get a float (0-1) representing fraction of total latency (averaged across runs)

**Given** all components
**When** I sum cost_contribution values
**Then** the sum equals 1.0 (within floating point tolerance)

**FR:** FR-406

---

### Story 6.4: Parameter Sensitivity Analysis

**As an** ML Engineer,
**I want** to know which parameters have the highest impact on outcomes,
**So that** I can focus on the most influential settings.

**Acceptance Criteria:**

**Given** a completed optimization
**When** I access `result.attribution[component_name].most_sensitive_param`
**Then** I get the parameter name with highest sensitivity_score

**Given** attribution for a component
**When** I access `sensitivity_scores`
**Then** I get a dict mapping each parameter to its impact score (computed via correlation with objective across runs)

**Given** sensitivity analysis
**When** a parameter has no variance in the optimization runs (e.g., was fixed)
**Then** its sensitivity_score is 0

**FR:** FR-406

---

### Story 6.5: Optimization Recommendations

**As an** ML Engineer,
**I want** the system to provide actionable optimization recommendations,
**So that** I know what to try next.

**Acceptance Criteria:**

**Given** attribution analysis for a component
**When** I access `optimization_opportunity`
**Then** I get 'high' (top 25% sensitivity), 'medium' (25-75%), or 'low' (bottom 25%)

**Given** a component with high optimization_opportunity
**When** I access `recommendation`
**Then** I get a human-readable suggestion (e.g., "Consider expanding temperature range" or "Model selection has high impact - try additional models")

**Given** a component with low optimization_opportunity
**When** I access `recommendation`
**Then** recommendation is None or suggests the component is well-optimized

**Given** all components
**When** ranking by optimization_opportunity
**Then** 'high' components appear in `result.attribution_ranked` at the top

**FR:** FR-406

---

## Epic 7: Production Readiness & CI/CD

Platform Engineers can integrate Traigent into CI/CD pipelines with CLI, structured output, production-hardened apply(), and artifact storage.

### Story 7.1: CLI Interface for Headless Optimization

**As a** Platform Engineer,
**I want** a CLI interface to run optimization without Python code,
**So that** I can integrate Traigent into CI/CD pipelines.

**Acceptance Criteria:**

**Given** a pipeline module and config file
**When** I run `traigent optimize --pipeline myapp.pipeline:pipe --config optimization.yaml`
**Then** the optimization runs headlessly and outputs results

**Given** CLI execution
**When** optimization completes successfully
**Then** exit code is 0

**Given** CLI execution
**When** optimization fails or no valid configs are found
**Then** exit code is non-zero and an error message is written to stderr

**Given** `--output-format json`
**When** optimization completes
**Then** results are printed as structured JSON to stdout

**FR:** US-4 (CI/CD integration)

---

### Story 7.2: YAML/JSON Configuration Files

**As a** Platform Engineer,
**I want** to configure optimization via YAML/JSON files,
**So that** I can version control and reproduce optimization runs.

**Acceptance Criteria:**

**Given** an optimization.yaml file with targets, constraints, and strategy
**When** it is loaded by the CLI config parser
**Then** the system configures optimization according to the file

**Given** a config file with invalid parameters
**When** loading the config
**Then** the system raises ConfigValidationError with a clear message and JSON path/line number

**Given** a config file referencing a Search Space TVL file
**When** loading the config
**Then** the system loads and applies the Search Space TVL definition

**FR:** US-4 (CI/CD integration)

---

### Story 7.3: Production-Hardened Apply Function

**As a** Platform Engineer,
**I want** a robust apply() function with validation and error handling,
**So that** I can safely deploy optimized configurations.

**Acceptance Criteria:**

**Given** `result.apply(pipeline, validate=True)`
**When** applying a configuration
**Then** the system validates the config against the current pipeline structure before applying
**And** raises ConfigMismatchError if the pipeline has changed

**Given** a validation failure
**When** apply is called
**Then** no changes are applied to the pipeline

**Given** `apply(backup=True)`
**When** applying a configuration
**Then** the system stores original parameter values for rollback

**Given** an applied configuration with backup enabled
**When** I call `result.rollback(pipeline)`
**Then** the pipeline is restored to the pre-apply state

**Given** rollback requested without a backup
**When** I call `result.rollback(pipeline)`
**Then** the system raises an error indicating rollback is unavailable

**Given** apply in production mode
**When** a parameter type doesn't match
**Then** the system raises TypeError with a clear description (no silent coercion)

**FR:** FR-404 (production)

---

### Story 7.4: Export Tuned Config TVL

**As a** Platform Engineer,
**I want** to export optimized configurations as Tuned Config TVL files,
**So that** I can store and version control the results.

**Acceptance Criteria:**

**Given** a completed optimization
**When** I call `result.to_tvl("optimized_config.tvl")`
**Then** the file contains the best configuration with all parameter values

**Given** a Tuned Config TVL file
**When** I inspect its contents
**Then** it includes: version, pipeline name, framework, target values, metrics (quality, cost, latency), constraints_satisfied, and optimization_metadata (strategy, n_trials, timestamp)

**Given** a Tuned Config TVL file
**When** I load it with `load_tuned_config("optimized_config.tvl")`
**Then** I get a config dict that can be passed to `apply_config()`

**FR:** FR-403

---

### Story 7.5: Export Experiment History

**As a** Platform Engineer,
**I want** to export the full experiment history,
**So that** I can analyze optimization runs and track experiments over time.

**Acceptance Criteria:**

**Given** a completed optimization
**When** I call `result.to_dataframe()`
**Then** I get a pandas DataFrame with all runs, configs, and metrics

**Given** a completed optimization
**When** I call `result.export_history("history.json")`
**Then** the file contains all ExperimentRun data in structured JSON format
**And** includes a schema version for forward compatibility

**Given** exported history
**When** I load it later
**Then** I can reconstruct the full OptimizationResult for analysis

**FR:** FR-405

---

### Story 7.6: Artifact Storage Integration

**As a** Platform Engineer,
**I want** to save optimization artifacts to configurable storage,
**So that** results persist across CI/CD runs.

**Acceptance Criteria:**

**Given** `optimize(artifact_path="./artifacts/")`
**When** optimization completes
**Then** the system saves: best_config.tvl, experiment_history.json, and summary.json to that path

**Given** artifact_path pointing to S3 or GCS (via fsspec)
**When** optimization completes
**Then** artifacts are uploaded to cloud storage

**Given** artifact_path pointing to S3 or GCS without fsspec installed
**When** optimization completes
**Then** the system raises a clear error indicating the optional dependency is required

**Given** a previous optimization artifact path
**When** I call `OptimizationResult.load("./artifacts/")`
**Then** I can reconstruct the result object for analysis or apply()

**FR:** FR-405 (extended - storage)

---

### Story 7.7: Structured Output for CI Integration

**As a** Platform Engineer,
**I want** structured output with clear success/failure indicators,
**So that** CI/CD pipelines can parse results programmatically.

**Acceptance Criteria:**

**Given** CLI with `--output-format json`
**When** optimization completes
**Then** output includes: status ("success" | "failure" | "no_valid_configs"), best_config, best_score, n_trials_completed, constraints_satisfied

**Given** optimization where no configs satisfy constraints
**When** CLI completes
**Then** status is "no_valid_configs" and exit code is 1
**And** output includes best_unconstrained_config if at least one run completed

**Given** `--ci` flag
**When** running optimization
**Then** progress output is suppressed and only final JSON is printed

**FR:** US-4 (CI/CD integration)

---

### Story 7.8: Minimal Integration Quickstart

**As a** Platform Engineer,
**I want** a minimal integration example,
**So that** teams can adopt Traigent with minimal code changes.

**Acceptance Criteria:**

**Given** the Quickstart example in docs
**When** I follow the steps
**Then** I can run optimization and apply results with 10 or fewer lines of code

**NFR:** NFR-4 (<=10 lines of code change)
