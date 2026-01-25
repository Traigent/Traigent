# Experimental Features

⚠️ **WARNING: This module contains experimental features that are NOT part of the production Traigent SDK.**

## Purpose

This module contains experimental implementations used for:

1. **Local Development & Testing**: When Traigent backend services are under development
2. **Prototyping**: Testing new platform integrations before backend implementation
3. **Educational Examples**: Demonstrating how platform integrations might work

## What's Here

### `/simple_cloud/` - Naive Cloud Simulation

A simplified, **non-production** implementation of platform executors that simulates cloud execution locally.

**⚠️ IMPORTANT DISCLAIMERS:**
- **NOT the real Traigent cloud implementation**
- **NOT suitable for production use**
- **Does NOT represent Traigent's proprietary IP**
- **Only for local experimentation and testing**

## Production Implementation

The **real** cloud execution happens in the **Traigent Backend**, which is:
- Proprietary Traigent IP
- Not part of this open-source SDK
- Optimized for scale, performance, and advanced features
- Accessible only through Traigent cloud services

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
3. **This is NOT Traigent's cloud architecture**
4. **Real cloud features are much more advanced**

For production use, rely on:
- `@traigent.optimize()` decorator with `auto_override_frameworks=True`
- Traigent cloud services (when available)
- Traigent backend integration
