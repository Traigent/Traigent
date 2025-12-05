# Experimental Features

⚠️ **WARNING: This module contains experimental features that are NOT part of the production TraiGent SDK.**

## Purpose

This module contains experimental implementations used for:

1. **Local Development & Testing**: When OptiGen backend services are under development
2. **Prototyping**: Testing new platform integrations before backend implementation
3. **Educational Examples**: Demonstrating how platform integrations might work

## What's Here

### `/simple_cloud/` - Naive Cloud Simulation

A simplified, **non-production** implementation of platform executors that simulates cloud execution locally.

**⚠️ IMPORTANT DISCLAIMERS:**
- **NOT the real TraiGent cloud implementation**
- **NOT suitable for production use**
- **Does NOT represent TraiGent's proprietary IP**
- **Only for local experimentation and testing**

## Production Implementation

The **real** cloud execution happens in the **OptiGen Backend**, which is:
- Proprietary TraiGent IP
- Not part of this open-source SDK
- Optimized for scale, performance, and advanced features
- Accessible only through TraiGent cloud services

## Usage

```python
# For local experimentation only
from traigent.experimental.simple_cloud import SimpleCloudSimulator

# This is NOT the real cloud - just for testing
simulator = SimpleCloudSimulator()
result = await simulator.test_platform_integration()
```

## Important Notes

1. **Do NOT use in production**
2. **Do NOT rely on this API** - it may change or be removed
3. **This is NOT TraiGent's cloud architecture**
4. **Real cloud features are much more advanced**

For production use, rely on:
- `@traigent.optimize()` decorator with `auto_override_frameworks=True`
- TraiGent cloud services (when available)
- OptiGen backend integration
