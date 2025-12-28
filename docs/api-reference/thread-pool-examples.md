# Thread Pool Usage with Context Propagation

This document provides examples of using thread pools within Traigent-optimized functions while properly propagating context.

## Overview

Python's `contextvars` (used by Traigent for configuration management) don't automatically propagate to `ThreadPoolExecutor` workers. This guide shows how to manually propagate Traigent's context to worker threads.

## When Do You Need This?

You need manual context propagation when:

1. **You create your own ThreadPoolExecutor** inside an optimized function
2. **You submit work to threads** that need access to `traigent.get_config()`
3. **You're parallelizing work** within a single trial

**Note**: Traigent's built-in parallel evaluator already handles context propagation automatically. You only need this when creating your own thread pools.

## Context Propagation API

Traigent provides utilities in `traigent.config.context`:

```python
from traigent.config.context import copy_context_to_thread

# Capture context snapshot before submitting to threads
snapshot = copy_context_to_thread()

# Restore context in worker thread
with snapshot.restore():
    config = traigent.get_config()  # Now works correctly!
    # ... use config ...
```

## Basic Example: Parallel Document Processing

```python
import traigent
from traigent.config.context import copy_context_to_thread
from concurrent.futures import ThreadPoolExecutor, as_completed

@traigent.optimize(
    objectives=["accuracy", "latency"],
    configuration_space={
        "model": ["gpt-4o-mini", "gpt-4"],
        "temperature": [0.1, 0.5, 0.9],
    },
    evaluation={"eval_dataset": "documents.jsonl"},
)
def process_documents(documents: list[str]) -> list[str]:
    """Process multiple documents in parallel with proper context propagation."""

    # Capture context BEFORE creating threads
    snapshot = copy_context_to_thread()

    def process_single_doc(doc_snapshot, doc):
        """Worker function that runs in thread pool."""
        # Restore context in worker thread
        with doc_snapshot.restore():
            # Now we can access the trial configuration
            config = traigent.get_config()

            # Use the configuration for this trial
            llm = OpenAI(
                model=config["model"],
                temperature=config["temperature"]
            )
            return llm.complete(f"Summarize: {doc}")

    # Process documents in parallel
    results = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        # Submit all tasks, passing the snapshot to each
        futures = [
            executor.submit(process_single_doc, snapshot, doc)
            for doc in documents
        ]

        # Collect results
        for future in as_completed(futures):
            results.append(future.result())

    return results
```

## Example: Batch API Calls with Rate Limiting

```python
import traigent
from traigent.config.context import copy_context_to_thread
from concurrent.futures import ThreadPoolExecutor
import time

@traigent.optimize(
    objectives=["accuracy", "cost", "latency"],
    configuration_space={
        "model": ["gpt-4o-mini", "gpt-4"],
        "batch_size": [10, 20, 50],
        "temperature": (0.0, 1.0),
    },
    evaluation={"eval_dataset": "queries.jsonl"},
)
def batch_process_queries(queries: list[str]) -> list[str]:
    """Process queries in batches with optimized batch size."""

    # Get configuration for this trial
    config = traigent.get_config()
    batch_size = config["batch_size"]

    # Split into batches
    batches = [queries[i:i + batch_size] for i in range(0, len(queries), batch_size)]

    # Capture context for thread workers
    snapshot = copy_context_to_thread()

    def process_batch(batch_snapshot, batch):
        """Process a single batch in a worker thread."""
        with batch_snapshot.restore():
            # Context is now available in worker thread
            config = traigent.get_config()

            llm = OpenAI(
                model=config["model"],
                temperature=config["temperature"]
            )

            results = []
            for query in batch:
                result = llm.complete(query)
                results.append(result)
                time.sleep(0.1)  # Rate limiting

            return results

    # Process batches in parallel
    all_results = []
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [
            executor.submit(process_batch, snapshot, batch)
            for batch in batches
        ]

        for future in futures:
            all_results.extend(future.result())

    return all_results
```

## Example: Multi-Agent System with Thread Pools

```python
import traigent
from traigent.config.context import copy_context_to_thread
from concurrent.futures import ThreadPoolExecutor

@traigent.optimize(
    objectives=["accuracy", "latency"],
    configuration_space={
        "researcher_model": ["gpt-4o-mini", "gpt-4"],
        "writer_model": ["gpt-4o-mini", "gpt-4"],
        "editor_model": ["gpt-4o-mini", "gpt-4"],
        "temperature": [0.1, 0.5, 0.9],
    },
    evaluation={"eval_dataset": "articles.jsonl"},
)
def multi_agent_content_creation(topic: str) -> str:
    """Use multiple agents in parallel to create content."""

    # Capture context once
    snapshot = copy_context_to_thread()

    def research_agent(ctx_snapshot, topic):
        """Research agent runs in parallel."""
        with ctx_snapshot.restore():
            config = traigent.get_config()
            llm = OpenAI(
                model=config["researcher_model"],
                temperature=config["temperature"]
            )
            return llm.complete(f"Research: {topic}")

    def writer_agent(ctx_snapshot, topic, research):
        """Writer agent runs after research."""
        with ctx_snapshot.restore():
            config = traigent.get_config()
            llm = OpenAI(
                model=config["writer_model"],
                temperature=config["temperature"]
            )
            return llm.complete(f"Write article about {topic} using: {research}")

    def editor_agent(ctx_snapshot, draft):
        """Editor agent runs after writing."""
        with ctx_snapshot.restore():
            config = traigent.get_config()
            llm = OpenAI(
                model=config["editor_model"],
                temperature=config["temperature"]
            )
            return llm.complete(f"Edit and improve: {draft}")

    # Execute agents in pipeline
    with ThreadPoolExecutor(max_workers=3) as executor:
        # Research phase
        research_future = executor.submit(research_agent, snapshot, topic)
        research_result = research_future.result()

        # Writing phase
        writer_future = executor.submit(writer_agent, snapshot, topic, research_result)
        draft = writer_future.result()

        # Editing phase
        editor_future = executor.submit(editor_agent, snapshot, draft)
        final_content = editor_future.result()

    return final_content
```

## Example: Map-Reduce Pattern

```python
import traigent
from traigent.config.context import copy_context_to_thread
from concurrent.futures import ThreadPoolExecutor, as_completed

@traigent.optimize(
    objectives=["accuracy", "cost"],
    configuration_space={
        "mapper_model": ["gpt-4o-mini", "gpt-4"],
        "reducer_model": ["gpt-4o-mini", "gpt-4"],
        "temperature": [0.1, 0.5, 0.9],
    },
    evaluation={"eval_dataset": "large_documents.jsonl"},
)
def analyze_large_document(document: str, chunk_size: int = 1000) -> str:
    """Analyze large document using map-reduce with thread pool."""

    # Split document into chunks
    chunks = [document[i:i + chunk_size] for i in range(0, len(document), chunk_size)]

    # Capture context
    snapshot = copy_context_to_thread()

    def map_chunk(ctx_snapshot, chunk):
        """Map phase: analyze each chunk independently."""
        with ctx_snapshot.restore():
            config = traigent.get_config()
            llm = OpenAI(
                model=config["mapper_model"],
                temperature=config["temperature"]
            )
            return llm.complete(f"Summarize key points: {chunk}")

    def reduce_summaries(ctx_snapshot, summaries):
        """Reduce phase: combine all chunk summaries."""
        with ctx_snapshot.restore():
            config = traigent.get_config()
            llm = OpenAI(
                model=config["reducer_model"],
                temperature=config["temperature"]
            )
            combined = "\n".join(summaries)
            return llm.complete(f"Create final summary from: {combined}")

    # Map phase: process chunks in parallel
    chunk_summaries = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [
            executor.submit(map_chunk, snapshot, chunk)
            for chunk in chunks
        ]

        for future in as_completed(futures):
            chunk_summaries.append(future.result())

    # Reduce phase: combine summaries
    with ThreadPoolExecutor(max_workers=1) as executor:
        final_future = executor.submit(reduce_summaries, snapshot, chunk_summaries)
        final_summary = final_future.result()

    return final_summary
```

## ContextSnapshot API Reference

### `copy_context_to_thread()`

Captures all Traigent context variables for propagation to worker threads.

```python
from traigent.config.context import copy_context_to_thread

snapshot = copy_context_to_thread()
```

**Returns**: `ContextSnapshot` containing all current context variables.

**What it captures**:
- Trial configuration (`config_context`)
- Applied configuration (`applied_config_context`)
- Configuration space (`config_space_context`)
- Trial metadata (`trial_context`)

### `ContextSnapshot.restore()`

Returns a context manager that restores the captured context in a worker thread.

```python
with snapshot.restore():
    # Context is now available
    config = traigent.get_config()
    # ... use config ...
```

**Returns**: `ContextRestorer` context manager.

### Thread Safety

The context propagation utilities are thread-safe:
- `copy_context_to_thread()` can be called from any thread
- `ContextSnapshot` can be passed to multiple worker threads
- Multiple workers can restore the same snapshot concurrently

## Common Pitfalls

### ❌ Incorrect: Creating threads without context propagation

```python
@traigent.optimize(...)
def process_items(items):
    def worker(item):
        # This will FAIL - context is not available
        config = traigent.get_config()  # Returns empty config!
        return process(item, config)

    with ThreadPoolExecutor() as executor:
        # Context is NOT propagated automatically
        results = list(executor.map(worker, items))
    return results
```

### ✅ Correct: Capture and restore context

```python
@traigent.optimize(...)
def process_items(items):
    # Capture context BEFORE creating threads
    snapshot = copy_context_to_thread()

    def worker(ctx_snapshot, item):
        # Restore context in worker thread
        with ctx_snapshot.restore():
            config = traigent.get_config()  # Now works!
            return process(item, config)

    with ThreadPoolExecutor() as executor:
        # Pass snapshot to each worker
        results = [
            executor.submit(worker, snapshot, item).result()
            for item in items
        ]
    return results
```

### ❌ Incorrect: Capturing context inside worker

```python
def worker(item):
    # Too late - context was lost when thread was created
    snapshot = copy_context_to_thread()  # Returns empty context!
    with snapshot.restore():
        config = traigent.get_config()  # Still empty!
        return process(item, config)
```

### ✅ Correct: Capture in main thread, restore in worker

```python
# In main thread
snapshot = copy_context_to_thread()  # Capture while context exists

def worker(ctx_snapshot, item):
    # Restore in worker thread
    with ctx_snapshot.restore():
        config = traigent.get_config()  # Now available!
        return process(item, config)

with ThreadPoolExecutor() as executor:
    future = executor.submit(worker, snapshot, item)
```

## Performance Considerations

### Snapshot Overhead

Creating a `ContextSnapshot` is lightweight:
- Only captures references to context variables
- No deep copying of configuration data
- Minimal memory overhead

Create one snapshot and reuse it for all workers:

```python
# Good: Create once, reuse many times
snapshot = copy_context_to_thread()
futures = [executor.submit(worker, snapshot, item) for item in items]

# Wasteful: Creating snapshot for each item
futures = [executor.submit(worker, copy_context_to_thread(), item) for item in items]
```

### Thread Pool Size

Consider your optimization's parallelism:

```python
@traigent.optimize(
    execution={
        "parallel_config": {
            "thread_workers": 4,  # Traigent's parallel trial execution
        }
    },
    ...
)
def my_function(data):
    # Your thread pool should be smaller to avoid oversubscription
    with ThreadPoolExecutor(max_workers=2) as executor:
        ...
```

## Async vs Threads

For async code, context propagates automatically:

```python
import asyncio
import traigent

@traigent.optimize(...)
async def process_items_async(items):
    async def worker(item):
        # Context propagates automatically to async tasks!
        config = traigent.get_config()  # Works without manual propagation
        return await async_process(item, config)

    # No need for copy_context_to_thread() with asyncio.create_task()
    tasks = [asyncio.create_task(worker(item)) for item in items]
    results = await asyncio.gather(*tasks)
    return results
```

**Use threads when**: You have blocking I/O or CPU-bound operations
**Use async when**: You have async I/O operations (async APIs, aiohttp, etc.)

## Related Documentation

- [Decorator Reference](./decorator-reference.md) - Configuration options
- [API Reference](./complete-function-specification.md) - Full API documentation
- [Parallel Configuration Guide](../guides/parallel-configuration.md) - Configuring parallelism
- [Context Management Source](../../traigent/config/context.py) - Implementation details
