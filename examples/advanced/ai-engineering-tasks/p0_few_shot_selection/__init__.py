"""
P0-3: Few-Shot Example Selection Strategies
===========================================

This example demonstrates how TraiGent optimizes few-shot example selection strategies
to significantly improve task performance without additional model training.

The optimization explores modern selection strategies including semantic similarity,
diversity sampling, curriculum learning, and various ordering/formatting approaches
to maximize task performance within token budget constraints.

Key Goals:
- Increase task accuracy by 5-15% over random selection
- Reduce variance in model outputs by 30%
- Stay within token budget constraints
- Provide interpretable selection strategies
- Enable task-adaptive selection

This addresses the critical challenge where optimal selection varies by task:
classification benefits from diverse boundary examples, while generation tasks
might need similar examples with varied outputs.
"""
