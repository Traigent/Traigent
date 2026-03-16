# Chunking Long Context

Optimize RAG chunking parameters for long document Q&A.

## Quick Start

```bash
export TRAIGENT_MOCK_LLM=true
python examples/core/chunking-long-context/run.py
```

## Configuration Space

| Parameter | Values | Purpose |
|-----------|--------|---------|
| `chunk_size` | 128, 192, 256 | Words per chunk |
| `overlap` | 8, 24 | Overlap between chunks |
| `top_k` | 1, 2 | Retrieved chunks |

## What It Optimizes

- Chunk size for context windows
- Overlap to preserve context boundaries
- Number of chunks to retrieve

## Expected Output

```
Best config: {'chunk_size': 192, 'overlap': 24, 'top_k': 2}
Best score: 0.85
```

## Key Concepts

- **Dynamic chunking**: Text split based on parameters
- **BM25 retrieval**: Keyword-based document ranking
- **Grid search**: Exhaustive parameter exploration

## Chunking Strategy

```python
def _chunk_text(text, chunk_size, overlap):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunks.append(" ".join(words[i:i + chunk_size]))
        i += max(1, chunk_size - overlap)
    return chunks
```

## Use Cases

- Long document Q&A
- Knowledge base retrieval
- RAG pipeline optimization

## Next Steps

- [hello-world](../hello-world/) - Basic RAG optimization
- [multi-objective-tradeoff](../multi-objective-tradeoff/) - Balance retrieval cost
