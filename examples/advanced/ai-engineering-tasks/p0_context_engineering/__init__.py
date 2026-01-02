"""
P0-2: Context Engineering and RAG Optimization
==============================================

This example demonstrates how Traigent optimizes context engineering strategies
for RAG (Retrieval-Augmented Generation) systems - addressing one of the most
critical challenges in modern LLM applications.

The optimization explores retrieval strategies (sparse/dense/hybrid), chunk sizes,
reranking approaches, context ordering, and budget allocation to balance relevance,
completeness, and cost.

Key Goals:
- Improve answer quality by 15-25% over baseline retrieval
- Reduce context costs by 30-50% through smart allocation
- Maintain or improve latency despite additional processing
- Provide interpretable context assembly strategies
- Enable query-type-aware dynamic optimization

This addresses the challenge where optimal context composition varies dramatically
by query type: factual questions need different strategies than analytical questions,
and single-hop queries differ from multi-hop reasoning.
"""
