# Traigent Analytics Plugin

Advanced AI analytics capabilities for Traigent SDK.

## Features

- **Meta-Learning**: Algorithm selection based on problem characteristics
- **Predictive Analytics**: Cost and performance forecasting
- **Anomaly Detection**: Automated performance regression detection
- **Cost Optimization**: Budget allocation and resource optimization
- **Smart Scheduling**: Intelligent job scheduling

## Installation

```bash
pip install traigent-analytics
```

Or with the analytics bundle:

```bash
pip install traigent[analytics]
```

## Usage

```python
from traigent_analytics import (
    MetaLearningEngine,
    PredictiveAnalytics,
    AnomalyDetector,
    CostOptimizationAI,
    SmartScheduler,
)

# Meta-learning for algorithm selection
engine = MetaLearningEngine()
recommended_algo = engine.recommend_algorithm(problem_features)

# Predictive analytics for cost forecasting
analytics = PredictiveAnalytics()
forecast = analytics.forecast_costs(historical_data)

# Anomaly detection
detector = AnomalyDetector()
anomalies = detector.detect(metrics)

# Cost optimization
optimizer = CostOptimizationAI()
recommendations = optimizer.optimize(budget_constraints)

# Smart scheduling
scheduler = SmartScheduler()
schedule = scheduler.create_schedule(jobs, constraints)
```

## Components

### Meta-Learning (`meta_learning.py`)
- `MetaLearningEngine`: Main engine for algorithm selection
- `OptimizationHistory`: Historical optimization data storage
- `AlgorithmSelector`: Algorithm recommendation based on features
- `PerformancePredictor`: Predict performance of algorithms

### Predictive Analytics (`predictive.py`)
- `PredictiveAnalytics`: Combined forecasting interface
- `CostForecaster`: Predict future costs
- `PerformanceForecaster`: Predict performance trends
- `TrendAnalyzer`: Analyze optimization trends

### Anomaly Detection (`anomaly.py`)
- `AnomalyDetector`: Detect anomalies in metrics
- `PerformanceMonitor`: Monitor for regressions
- `RegressionDetector`: Detect performance regressions
- `AlertManager`: Manage and route alerts

### Cost Optimization (`cost_optimization.py`)
- `CostOptimizationAI`: AI-powered cost optimization
- `BudgetAllocator`: Allocate budget across resources
- `ResourceOptimizer`: Optimize resource usage

### Scheduling (`scheduling.py`)
- `SmartScheduler`: Intelligent job scheduling
- `ScheduledJob`: Job definition
- `SchedulingPolicy`: Scheduling policies

## Requirements

- Python 3.10+
- traigent >= 0.9.0
- numpy >= 1.21.0
- scipy >= 1.7.0

## License

MIT License - see LICENSE file for details.
