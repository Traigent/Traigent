# Cost Verification Tests

This directory contains manual tests to verify that Traigent correctly tracks and logs costs across different LLM providers.

## Overview

The test suite runs actual LLM calls through different SDKs and compares:
1. **Expected costs** - Calculated from known pricing tables
2. **SDK-reported costs** - What the LLM SDK reports
3. **Backend-logged costs** - What Traigent logs to the backend DB
4. **Langfuse costs** (optional) - What Langfuse records

## Supported Providers

- **OpenAI SDK** - Direct OpenAI API calls
- **Anthropic SDK** - Direct Anthropic API calls
- **LangChain** - LangChain with OpenAI and Anthropic
- **LiteLLM** - LiteLLM with OpenAI and Anthropic
- **Traigent** - Full Traigent @optimize decorator integration

## Prerequisites

1. **API Keys**: Set environment variables:
   ```bash
   export OPENAI_API_KEY="your-key"
   export ANTHROPIC_API_KEY="your-key"
   export TRAIGENT_API_KEY="your-key"
   export TRAIGENT_BACKEND_URL="http://localhost:5000"  # or your backend URL
   ```

2. **Optional Langfuse** (for comparison mode):
   ```bash
   export LANGFUSE_PUBLIC_KEY="pk-xxx"
   export LANGFUSE_SECRET_KEY="sk-xxx"
   ```

3. **Install dependencies**:
   ```bash
   pip install openai anthropic langchain-openai langchain-anthropic litellm aiohttp python-dotenv
   ```

## Usage

### Run all tests
```bash
cd tests/manual/cost_verification
python verify_cost_tracking.py
```

### Run specific provider tests
```bash
python verify_cost_tracking.py --provider openai
python verify_cost_tracking.py --provider openai --provider anthropic
```

### Enable Langfuse comparison
```bash
python verify_cost_tracking.py --with-langfuse
```

### Verbose output
```bash
python verify_cost_tracking.py -v
```

### Save report to file
```bash
python verify_cost_tracking.py --output report.json
```

## Test Flow

1. **SDK Tests**: Makes direct LLM calls and extracts token counts from responses
2. **Cost Calculation**: Computes expected costs using pricing tables
3. **Backend Check**: Queries Traigent backend API for logged costs
4. **Langfuse Check** (optional): Queries Langfuse for trace costs
5. **Report**: Generates comparison report showing discrepancies

## Expected Output

```
============================================================
TRAIGENT COST VERIFICATION TEST SUITE
============================================================
Time: 2025-01-20T12:00:00
Backend: http://localhost:5000
Langfuse: Disabled
Providers: openai, anthropic, langchain, litellm, traigent
============================================================

--- OpenAI SDK Tests ---
✓ OpenAI SDK test passed: 25 in, 3 out

--- Anthropic SDK Tests ---
✓ Anthropic SDK test passed: 18 in, 2 out

--- Cost Comparison ---
Test                      Expected     SDK          Diff %
------------------------------------------------------------
openai_sdk_direct         $0.000005    $0.000005    0.0%
anthropic_sdk_direct      $0.000012    $0.000012    0.0%

--- Backend Status ---
Reachable: True
Sessions found: 5
```

## Troubleshooting

### Backend not reachable
- Ensure the Traigent backend is running
- Check `TRAIGENT_BACKEND_URL` is correct
- Verify `TRAIGENT_API_KEY` is valid

### Missing API keys
- The script will skip tests for providers without API keys
- Check environment variables are exported

### LiteLLM not installed
- Install with: `pip install litellm`

### Cost discrepancies
- Pricing tables may be outdated - update `OPENAI_PRICING` and `ANTHROPIC_PRICING` in the script
- Check if the backend is using different pricing data

## Pricing Table Updates

The script includes pricing tables as of January 2025. Update these if needed:

```python
OPENAI_PRICING = {
    "gpt-4o": {"input": 0.0025, "output": 0.010},
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    # ... add more models
}

ANTHROPIC_PRICING = {
    "claude-3-5-sonnet-20241022": {"input": 0.003, "output": 0.015},
    # ... add more models
}
```

## Files

- `verify_cost_tracking.py` - Main test script
- `README.md` - This file
