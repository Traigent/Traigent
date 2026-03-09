# Evaluation Guide

Both native and hybrid execution require evaluation rows through:

- `evaluation.data`
- or `evaluation.loadData`

The SDK uses those rows to construct `dataset_subset` information for each
trial. Trial logic should continue to read runtime configuration through:

- `TrialContext`
- `getTrialConfig()`
- `getTrialParam()`
