"""
P0-1: Modern Structured Output Engineering
==========================================

This example demonstrates how Traigent can optimize structured output extraction
from unstructured text, addressing one of the most universal problems in AI engineering.

The optimization explores modern output strategies including JSON mode, function calling,
XML tags, and various validation approaches to achieve near-perfect parsing reliability
while maintaining extraction quality.

Key Goals:
- Achieve >99.9% valid, parseable outputs
- Maintain or improve extraction F1 score
- Keep latency within 110% of baseline
- Reduce or maintain token costs

This addresses the critical challenge where different models have different strengths:
GPT-4 excels with JSON mode, Claude performs better with XML tags, and open-source
models might need constrained decoding.
"""
