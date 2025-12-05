# TraiGent Execution Modes Guide

> **Current status:** Open-source builds only support `edge_analytics` (local) today. Cloud and hybrid orchestration are roadmap items; keep runs in local mode unless your environment explicitly wires a managed backend.

## Overview

TraiGent offers three execution modes that balance privacy, performance, and features based on your specific needs. Choose the mode that best fits your use case, infrastructure, and privacy requirements.

## Table of Contents

- [Execution Modes Comparison](#execution-modes-comparison)
- [Local Mode (edge_analytics)](#local-mode-edge_analytics)
- [Cloud Mode](#cloud-mode)
- [Hybrid Mode](#hybrid-mode)
- [Privacy-Safe Analytics](#privacy-safe-analytics)
- [Migration Path](#migration-path)
- [Best Practices](#best-practices)

## Execution Modes Comparison

| Feature | Local Mode | Cloud Mode | Hybrid Mode |
|---------|------------|------------|-------------|
| **Data Privacy** | ✅ Complete | ⚠️ Planned (metadata only) | ✅ Planned (I/O stays local) |
| **Optimization Algorithm** | Random/Grid | Planned Bayesian | Planned Bayesian guidance |
| **Speed** | Fast | Planned Fastest | Planned Fast |
| **Infrastructure** | Your servers | Planned TraiGent cloud | Planned Both |
| **Team Collaboration** | ❌ No | Planned | Planned |
| **Cost** | Free | Planned Usage-based | Planned Usage-based |
| **Air-Gapped Support** | ✅ Yes | ❌ No | ❌ No |
| **Trial Limits** | None | Planned Unlimited | Planned Unlimited |
| **Best For** | Sensitive data, testing | Roadmap | Roadmap |

## Local Mode (edge_analytics)

**🏠 Complete Privacy - Your Data Never Leaves Your Infrastructure**

### Overview

Local mode (also called `edge_analytics`) runs all optimization entirely on your infrastructure with zero external API calls for optimization logic.

### Configuration

```python
import traigent

@traigent.optimize(
    execution_mode="edge_analytics",  # Full privacy mode
    local_storage_path="./my_optimizations",
    minimal_logging=True,
    configuration_space={
        "model": ["gpt-4o-mini", "gpt-4o"],
        "temperature": [0.1, 0.5, 0.9]
    },
    objectives=["accuracy", "cost"]
)
def my_agent(query: str) -> str:
    # Your sensitive agent logic here
    return process_query(query)
```

### Features

✅ **Complete Data Privacy**
- Function code stays on your servers
- Input/output data never transmitted
- Configuration choices remain private
- Perfect for regulated industries (HIPAA, GDPR, SOC 2)

✅ **Local Storage**
- Results stored in filesystem (default: `~/.traigent/`)
- Customizable storage path
- No database dependencies
- Works offline/air-gapped environments

✅ **Fast Execution**
- No network latency
- Parallel trial execution
- Immediate results access

✅ **Zero External Dependencies**
- Works without internet (except LLM API calls)
- No cloud service registration required
- No authentication needed

### Limitations

⚠️ **Limited Optimization Algorithms**
- Random search
- Grid search
- No advanced Bayesian optimization (cloud-only)

⚠️ **No Team Features**
- Results not shared across team
- No central optimization history
- Manual result sharing required

### When to Use Local Mode

**Perfect for:**
- Sensitive data (healthcare, finance, legal)
- Air-gapped environments
- Development and testing
- POC and demos
- Compliance requirements (GDPR, HIPAA)
- Budget-conscious projects

**Example Use Cases:**
- Medical diagnosis agents (HIPAA-protected data)
- Financial analysis (SEC regulations)
- Legal document processing (attorney-client privilege)
- Internal security tools
- CI/CD testing pipelines

### Advanced Configuration

```python
import os
from traigent.api.decorators import EvaluationOptions, ExecutionOptions

# Custom storage location
os.makedirs("./secure_optimizations", exist_ok=True)

@traigent.optimize(
    execution=ExecutionOptions(
        execution_mode="edge_analytics",
        local_storage_path="./secure_optimizations",
        minimal_logging=True,  # Reduce logging for privacy
        cache_results=True     # Cache for faster iterations
    ),
    evaluation=EvaluationOptions(
        eval_dataset="sensitive_data.jsonl",
        custom_evaluator=my_secure_evaluator
    ),
    configuration_space={
        "model": ["gpt-4o-mini"],
        "temperature": [0.0, 0.3, 0.5]
    }
)
def hipaa_compliant_agent(patient_data: dict) -> str:
    # PHI stays completely local
    return process_patient_data(patient_data)
```

## Cloud Mode

**☁️ Roadmap: Managed orchestration (not available in OSS builds yet)**

> Cloud orchestration is in development. Use `edge_analytics` until a managed backend is provisioned.

### Overview

Cloud mode leverages TraiGent's cloud infrastructure for advanced Bayesian optimization, team collaboration, and scalable experimentation.

### Configuration

```python
import traigent

@traigent.optimize(
    execution_mode="edge_analytics",  # cloud is not yet available in OSS
    configuration_space={
        "model": ["gpt-4o-mini", "gpt-4o", "claude-3-sonnet"],
        "temperature": {"type": "float", "bounds": [0.0, 1.0]},
        "max_tokens": {"type": "int", "bounds": [100, 2000]}
    },
    objectives=["accuracy", "cost", "latency"]
)
def production_agent(query: str) -> str:
    return process_query(query)
```

### Features

✅ **Advanced Optimization**
- Bayesian optimization (better than grid/random)
- Multi-objective optimization (Pareto front)
- Intelligent trial scheduling
- Adaptive sampling strategies

✅ **Team Collaboration**
- Shared optimization history
- Team-wide best practices
- Centralized result tracking
- Collaborative experimentation

✅ **Scalability**
- Unlimited trials
- Distributed execution
- Cloud compute resources
- Automatic result archiving

✅ **Enhanced Analytics**
- Real-time dashboards
- Experiment comparisons
- Trend analysis
- ROI tracking

### Data Sharing (When Cloud is Active)

**What Gets Sent to Cloud:**
- Configuration space definitions
- Optimization parameters (models, temperatures, etc.)
- Aggregated performance metrics
- Trial results (scores, costs, latencies)

**What NEVER Leaves Your Servers:**
- Your function code
- Input/output data content
- API keys and credentials
- Proprietary business logic

### When to Use Cloud Mode

Planned for production deployments once the managed service is live. Today, keep workloads on `edge_analytics`.

## Hybrid Mode

**🔄 Roadmap: Local execution with cloud guidance**

> Hybrid mode depends on cloud infrastructure and is not available in the current open-source release.

### Overview

Hybrid mode executes your agent code locally while leveraging cloud-based Bayesian optimization for intelligent trial selection.

### Configuration

```python
import traigent

@traigent.optimize(
    execution_mode="edge_analytics",  # hybrid is not yet available in OSS
    privacy_enabled=True,  # Never transmit input/output/prompts
    configuration_space={
        "model": ["gpt-4o-mini", "gpt-4o"],
        "temperature": [0.1, 0.5, 0.9],
        "k": [3, 5, 10]
    }
)
def balanced_agent(query: str, knowledge_base: list) -> str:
    # Executes locally, optimization guided by cloud
    return rag_pipeline(query, knowledge_base)
```

### Features

✅ **Privacy + Performance**
- Code execution stays local
- Data never transmitted
- Cloud suggests optimal trials
- Bayesian optimization benefits

✅ **Flexible Privacy Controls**
```python
@traigent.optimize(
    execution_mode="hybrid",
    privacy_enabled=True,        # Never send input/output
    share_metrics=True,          # Share performance scores
    share_config_space=True      # Share parameter definitions
)
```

✅ **Gradual Adoption**
- Start with local mode
- Upgrade to hybrid when ready
- Full cloud for production

### When to Use Hybrid Mode

Planned for gradual cloud adoption; for now stay on `edge_analytics`.

## Privacy-Safe Analytics

### What Analytics Track

TraiGent includes optional privacy-safe analytics that help you understand optimization patterns without compromising data security.

#### ✅ What We Track (Aggregated Only)

```python
@traigent.optimize(
    enable_usage_analytics=True,  # Default: enabled
    eval_dataset="customer_support.jsonl"
)
```

**Tracked Metrics:**
- Total optimization sessions completed
- Average trials per optimization
- Performance improvement percentages (Δ accuracy)
- Days since first use
- Configuration space complexity (# parameters)
- Objective types (accuracy, cost, latency)

#### 🔒 What We NEVER Track

**Absolutely Private:**
- Function names or code content
- Actual parameter values or configurations
- API keys, credentials, or secrets
- File paths or directory structures
- Individual trial results or outputs
- Input/output data content
- Prompts or generated text

### Intelligent Upgrade Recommendations

Upgrade prompts are a roadmap item; the OSS CLI does not emit cloud upsell messages today.

### Disabling Analytics

```python
@traigent.optimize(
    enable_usage_analytics=False,  # Disable all analytics
    execution_mode="edge_analytics"
)
```

Or via environment variable:

```bash
export TRAIGENT_DISABLE_ANALYTICS=true
```

## Migration Path

### Step 1: Start Local (Week 1-2)

```python
# Begin with complete privacy
@traigent.optimize(
    execution_mode="edge_analytics",
    eval_dataset="test_data.jsonl"
)
def my_agent(query: str) -> str:
    return process(query)
```

**Goals:**
- Validate TraiGent value
- Test with production workloads
- Build team confidence
- Establish optimization patterns

### Step 2: Evaluate ROI (Week 3-4)

**Review Local Runs:**
- Use `traigent results` to list stored runs and `traigent plot <name>` to inspect progress.
- Compare objective gains and runtime locally; cloud ROI estimation will come with managed service availability.

### Step 3: Selective Cloud Adoption (Month 2)

```python
# Move non-sensitive workloads to cloud
@traigent.optimize(
    execution_mode="cloud",  # For blog generation
    eval_dataset="blog_examples.jsonl"
)
def blog_generator(topic: str) -> str:
    return generate_blog(topic)

# Keep sensitive workloads local
@traigent.optimize(
    execution_mode="edge_analytics",  # For customer data
    eval_dataset="customer_data.jsonl"
)
def customer_analyzer(data: dict) -> str:
    return analyze_customer(data)
```

### Step 4: Full Cloud (Month 3+)

```python
# Production deployment with cloud
@traigent.optimize(
    execution_mode="cloud",
    team_id="marketing-team",
    experiment_tags=["production", "v2.0"]
)
def production_agent(query: str) -> str:
    return process(query)
```

## Best Practices

### Security

```python
# ✅ Good: Use environment variables for credentials
import os

@traigent.optimize(
    api_key=os.getenv("TRAIGENT_API_KEY"),
    execution_mode="cloud"
)

# ❌ Bad: Hardcode API keys
@traigent.optimize(
    api_key="tg_abc123...",  # NEVER DO THIS
    execution_mode="cloud"
)
```

### Performance

```python
# ✅ Good: Enable parallel execution
@traigent.optimize(
    execution_mode="edge_analytics",
    parallel_config={
        "example_concurrency": 8,  # Evaluate examples in parallel
        "trial_concurrency": 4     # Run trials in parallel
    }
)

# ✅ Good: Use caching for iterative development
@traigent.optimize(
    execution_mode="edge_analytics",
    cache_results=True,  # Reuse previous trial results
    cache_ttl=3600       # Cache for 1 hour
)
```

### Privacy

```python
# ✅ Good: Explicit privacy controls
@traigent.optimize(
    execution_mode="hybrid",
    privacy_enabled=True,           # Never send I/O
    share_metrics=True,             # Share performance only
    minimal_logging=True,           # Reduce log verbosity
    enable_usage_analytics=False   # Disable analytics
)
```

### Development Workflow

```python
# Development: Use mock mode
if os.getenv("ENV") == "development":
    os.environ["TRAIGENT_MOCK_MODE"] = "true"

# Staging: Use local mode
elif os.getenv("ENV") == "staging":
    execution_mode = "edge_analytics"

# Production: Use cloud mode
else:
    execution_mode = "cloud"

@traigent.optimize(
    execution_mode=execution_mode,
    eval_dataset=get_dataset_for_env()
)
```

## Related Documentation

- [Quick Start Guide](quickstart.md)
- [Evaluation Guide](evaluation.md)
- [Privacy & Security](privacy-security.md)
- [API Reference](../api-reference/)

---

**Need Help?**
- [GitHub Issues](https://github.com/Traigent/Traigent/issues)
- [Discord Community](https://discord.gg/traigent)
- [Documentation](https://docs.traigent.ai)
