"""
Dataset generation for context engineering and RAG optimization.

This module creates diverse QA datasets with document collections for testing
different retrieval strategies, chunk sizes, and context composition approaches.
"""

# ruff: noqa: E501
# Long lines are intentional - they contain educational content and example data

import random
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any


class QueryType(Enum):
    """Types of queries for RAG systems."""

    FACTUAL = "factual"  # Simple fact retrieval
    MULTI_HOP = "multi_hop"  # Requires multiple documents
    ANALYTICAL = "analytical"  # Requires reasoning over context
    TEMPORAL = "temporal"  # Time-sensitive information
    COMPARATIVE = "comparative"  # Comparing multiple entities


@dataclass
class Document:
    """A document in the knowledge base."""

    id: str
    title: str
    content: str
    metadata: dict[str, Any]
    relevance_score: float  # Ground truth relevance to queries
    chunk_size: int  # Optimal chunk size for this document


@dataclass
class DocumentChunk:
    """A chunk of a document."""

    doc_id: str
    chunk_id: str
    content: str
    start_pos: int
    end_pos: int
    metadata: dict[str, Any]


@dataclass
class RAGQuery:
    """A query for the RAG system."""

    question: str
    ground_truth_answer: str
    query_type: QueryType
    required_doc_ids: list[str]  # Documents needed for correct answer
    difficulty: float  # 0.0 (easy) to 1.0 (hard)
    metadata: dict[str, Any]


@dataclass
class RAGDataset:
    """Complete RAG evaluation dataset."""

    documents: list[Document]
    queries: list[RAGQuery]
    metadata: dict[str, Any]


def generate_technical_documents() -> list[Document]:
    """Generate technical documentation corpus."""
    documents = [
        # Machine Learning Documentation
        Document(
            id="ml_001",
            title="Introduction to Neural Networks",
            content="""Neural networks are computational models inspired by biological neural networks.
            They consist of interconnected nodes (neurons) organized in layers. The input layer receives data,
            hidden layers process information, and the output layer produces results. Each connection has a weight
            that is adjusted during training through backpropagation. Common architectures include feedforward networks,
            convolutional neural networks (CNNs) for image processing, and recurrent neural networks (RNNs) for sequential data.
            Training involves minimizing a loss function using optimization algorithms like gradient descent.
            Key hyperparameters include learning rate, batch size, and number of epochs. Modern frameworks like
            TensorFlow and PyTorch facilitate implementation. Applications span computer vision, natural language processing,
            and reinforcement learning. Challenges include overfitting, vanishing gradients, and computational requirements.""",
            metadata={
                "domain": "machine_learning",
                "year": 2024,
                "complexity": "intermediate",
            },
            relevance_score=0.9,
            chunk_size=256,
        ),
        Document(
            id="ml_002",
            title="Transformer Architecture Explained",
            content="""The Transformer architecture revolutionized NLP by replacing recurrence with self-attention mechanisms.
            Introduced in 'Attention is All You Need' (2017), it processes sequences in parallel rather than sequentially.
            The architecture consists of an encoder and decoder, each with multiple layers. Key components include
            multi-head attention, which allows the model to attend to different positions simultaneously, and positional
            encoding to maintain sequence order. Layer normalization and residual connections stabilize training.
            The attention mechanism computes Query, Key, and Value matrices, with attention scores calculated as
            softmax(QK^T/sqrt(d_k))V. BERT uses only the encoder for bidirectional understanding, while GPT uses
            only the decoder for autoregressive generation. Transformers scale efficiently with data and compute,
            enabling models with billions of parameters. They excel at capturing long-range dependencies and have
            achieved state-of-the-art results across NLP tasks.""",
            metadata={
                "domain": "machine_learning",
                "year": 2024,
                "complexity": "advanced",
            },
            relevance_score=0.95,
            chunk_size=256,
        ),
        # Software Engineering Documentation
        Document(
            id="se_001",
            title="Microservices Architecture Best Practices",
            content="""Microservices architecture decomposes applications into small, independent services that communicate
            via APIs. Each service owns its data and business logic, enabling autonomous development and deployment.
            Key principles include single responsibility, decentralized governance, and failure isolation.
            Services communicate through REST APIs, message queues, or event streaming. API gateways handle
            cross-cutting concerns like authentication and rate limiting. Service discovery enables dynamic
            service location. Circuit breakers prevent cascade failures. Each service should have its own database
            to ensure loose coupling. Containerization with Docker and orchestration with Kubernetes facilitate
            deployment. Monitoring requires distributed tracing (Jaeger, Zipkin) and centralized logging (ELK stack).
            Challenges include network latency, data consistency, and operational complexity. Benefits include
            scalability, technology diversity, and faster time-to-market.""",
            metadata={
                "domain": "software_engineering",
                "year": 2024,
                "complexity": "intermediate",
            },
            relevance_score=0.85,
            chunk_size=256,
        ),
        Document(
            id="se_002",
            title="Database Optimization Techniques",
            content="""Database optimization improves query performance and resource utilization. Indexing is fundamental -
            B-tree indexes for range queries, hash indexes for equality checks, and composite indexes for multi-column queries.
            Query optimization involves analyzing execution plans, avoiding N+1 queries, and using appropriate JOIN types.
            Denormalization can improve read performance at the cost of write complexity and storage. Partitioning
            distributes data across multiple tables or servers - horizontal partitioning (sharding) splits rows,
            vertical partitioning splits columns. Caching strategies include query result caching, application-level
            caching with Redis, and database buffer pools. Connection pooling reduces overhead. For writes, batch
            operations and bulk inserts improve throughput. VACUUM and ANALYZE maintain PostgreSQL performance.
            MySQL benefits from optimizing buffer pool size. NoSQL databases offer different trade-offs - MongoDB
            for document storage, Cassandra for wide-column storage, Redis for key-value caching.""",
            metadata={
                "domain": "software_engineering",
                "year": 2024,
                "complexity": "advanced",
            },
            relevance_score=0.8,
            chunk_size=256,
        ),
        # Cloud Computing Documentation
        Document(
            id="cloud_001",
            title="AWS Services Overview",
            content="""Amazon Web Services provides comprehensive cloud computing services. EC2 offers virtual servers
            with various instance types optimized for compute, memory, or GPU workloads. S3 provides object storage
            with different storage classes for cost optimization. RDS manages relational databases including MySQL,
            PostgreSQL, and Aurora. Lambda enables serverless computing with automatic scaling and pay-per-execution
            pricing. API Gateway creates and manages REST and WebSocket APIs. DynamoDB offers managed NoSQL with
            single-digit millisecond latency. CloudFormation enables infrastructure as code using JSON or YAML templates.
            ECS and EKS provide container orchestration. SQS and SNS handle message queuing and pub/sub messaging.
            CloudWatch monitors resources and applications. IAM controls access with users, groups, roles, and policies.
            VPC provides network isolation. Route 53 manages DNS. CloudFront CDN accelerates content delivery.
            Cost optimization involves Reserved Instances, Spot Instances, and right-sizing.""",
            metadata={
                "domain": "cloud_computing",
                "year": 2024,
                "complexity": "intermediate",
            },
            relevance_score=0.75,
            chunk_size=256,
        ),
        # Data Science Documentation
        Document(
            id="ds_001",
            title="Feature Engineering Techniques",
            content="""Feature engineering transforms raw data into meaningful inputs for machine learning models.
            Numerical features benefit from normalization (min-max scaling) or standardization (z-score).
            Categorical variables require encoding - one-hot for nominal, ordinal encoding for ordered categories,
            or target encoding for high cardinality. Missing data strategies include imputation (mean, median, mode,
            or advanced methods like KNN or MICE), or creating missingness indicators. Feature creation involves
            domain knowledge - polynomial features capture non-linear relationships, interaction terms model
            feature dependencies. Time series features include lags, rolling statistics, and seasonal decomposition.
            Text features use TF-IDF, word embeddings, or transformer-based representations. Dimensionality reduction
            techniques like PCA or autoencoders handle high-dimensional data. Feature selection methods include
            filter methods (correlation, chi-square), wrapper methods (RFE), and embedded methods (LASSO, tree importance).
            Validation ensures features generalize without leakage.""",
            metadata={
                "domain": "data_science",
                "year": 2024,
                "complexity": "intermediate",
            },
            relevance_score=0.82,
            chunk_size=256,
        ),
    ]

    return documents


def generate_queries(documents: list[Document]) -> list[RAGQuery]:
    """Generate queries for the document corpus."""
    queries = [
        # Factual queries - single document
        RAGQuery(
            question="What is backpropagation in neural networks?",
            ground_truth_answer="Backpropagation is the algorithm used to adjust connection weights during neural network training by propagating errors backward through the network.",
            query_type=QueryType.FACTUAL,
            required_doc_ids=["ml_001"],
            difficulty=0.2,
            metadata={"topic": "machine_learning"},
        ),
        RAGQuery(
            question="What is the attention formula in Transformers?",
            ground_truth_answer="The attention mechanism computes attention scores as softmax(QK^T/sqrt(d_k))V, where Q is Query, K is Key, V is Value, and d_k is the dimension of the key vectors.",
            query_type=QueryType.FACTUAL,
            required_doc_ids=["ml_002"],
            difficulty=0.3,
            metadata={"topic": "machine_learning"},
        ),
        # Multi-hop queries - require multiple documents
        RAGQuery(
            question="How can Lambda functions be monitored in AWS?",
            ground_truth_answer="Lambda functions can be monitored using CloudWatch, which is AWS's monitoring service that tracks resources and applications, providing metrics and logs for Lambda executions.",
            query_type=QueryType.MULTI_HOP,
            required_doc_ids=["cloud_001"],
            difficulty=0.5,
            metadata={"topic": "cloud_computing"},
        ),
        RAGQuery(
            question="What database optimization techniques are relevant for microservices?",
            ground_truth_answer="For microservices, key database optimizations include: each service having its own database for loose coupling, connection pooling to reduce overhead, caching with Redis, and using appropriate database types (SQL vs NoSQL) based on service needs.",
            query_type=QueryType.MULTI_HOP,
            required_doc_ids=["se_001", "se_002"],
            difficulty=0.6,
            metadata={"topic": "software_engineering"},
        ),
        # Analytical queries - require reasoning
        RAGQuery(
            question="Why are Transformers more efficient than RNNs for long sequences?",
            ground_truth_answer="Transformers are more efficient than RNNs for long sequences because they process sequences in parallel rather than sequentially, use self-attention to directly capture long-range dependencies without recursive computations, and scale efficiently with data and compute resources.",
            query_type=QueryType.ANALYTICAL,
            required_doc_ids=["ml_002"],
            difficulty=0.7,
            metadata={"topic": "machine_learning"},
        ),
        RAGQuery(
            question="What are the trade-offs between microservices and database normalization?",
            ground_truth_answer="Microservices favor database denormalization and service-specific databases for loose coupling and independent scaling, while traditional normalization reduces redundancy. The trade-off is between service autonomy (microservices) and data consistency (normalization).",
            query_type=QueryType.ANALYTICAL,
            required_doc_ids=["se_001", "se_002"],
            difficulty=0.8,
            metadata={"topic": "software_engineering"},
        ),
        # Comparative queries
        RAGQuery(
            question="Compare BERT and GPT architectures.",
            ground_truth_answer="BERT uses only the Transformer encoder for bidirectional understanding of context, while GPT uses only the decoder for autoregressive generation. BERT is better for understanding tasks, GPT for generation tasks.",
            query_type=QueryType.COMPARATIVE,
            required_doc_ids=["ml_002"],
            difficulty=0.6,
            metadata={"topic": "machine_learning"},
        ),
        RAGQuery(
            question="Compare EC2 and Lambda for compute workloads.",
            ground_truth_answer="EC2 provides virtual servers with full control and persistent compute, suitable for long-running applications. Lambda offers serverless computing with automatic scaling and pay-per-execution pricing, ideal for event-driven and short-duration workloads.",
            query_type=QueryType.COMPARATIVE,
            required_doc_ids=["cloud_001"],
            difficulty=0.5,
            metadata={"topic": "cloud_computing"},
        ),
        # Complex multi-hop analytical queries
        RAGQuery(
            question="How would you implement a machine learning pipeline using AWS services?",
            ground_truth_answer="Implement an ML pipeline using S3 for data storage, Lambda for data preprocessing, EC2 or SageMaker for model training, API Gateway and Lambda for model serving, CloudWatch for monitoring, and CloudFormation for infrastructure as code.",
            query_type=QueryType.ANALYTICAL,
            required_doc_ids=["cloud_001", "ml_001"],
            difficulty=0.9,
            metadata={"topic": "ml_engineering"},
        ),
        RAGQuery(
            question="What feature engineering would be needed for a text classification microservice?",
            ground_truth_answer="For a text classification microservice, use TF-IDF or transformer-based embeddings for text representation, handle missing data with appropriate strategies, implement proper API endpoints, ensure service isolation with its own database, and include monitoring for model performance.",
            query_type=QueryType.MULTI_HOP,
            required_doc_ids=["ds_001", "se_001"],
            difficulty=0.85,
            metadata={"topic": "ml_engineering"},
        ),
    ]

    return queries


def chunk_documents(
    documents: list[Document], chunk_size: int = 256, overlap: int = 50
) -> list[DocumentChunk]:
    """Chunk documents with specified size and overlap."""
    chunks = []

    for doc in documents:
        text = doc.content
        doc_chunks = []

        # Simple word-based chunking
        words = text.split()

        for i in range(0, len(words), chunk_size - overlap):
            chunk_words = words[i : i + chunk_size]
            chunk_text = " ".join(chunk_words)

            chunk = DocumentChunk(
                doc_id=doc.id,
                chunk_id=f"{doc.id}_chunk_{len(doc_chunks)}",
                content=chunk_text,
                start_pos=i,
                end_pos=min(i + chunk_size, len(words)),
                metadata={
                    "source_doc": doc.title,
                    "chunk_index": len(doc_chunks),
                    "total_chunks": (len(words) + chunk_size - 1)
                    // (chunk_size - overlap),
                },
            )
            chunks.append(chunk)
            doc_chunks.append(chunk)

    return chunks


def generate_baseline_configs() -> dict[str, dict[str, Any]]:
    """Generate baseline RAG configurations for comparison."""
    return {
        "simple_rag": {
            "retrieval_method": "bm25",
            "n_chunks": 3,
            "chunk_size": 256,
            "chunk_overlap": 0,
            "reranking": False,
            "query_expansion": False,
        },
        "standard_rag": {
            "retrieval_method": "dense",
            "n_chunks": 5,
            "chunk_size": 512,
            "chunk_overlap": 50,
            "reranking": False,
            "query_expansion": False,
        },
        "advanced_rag": {
            "retrieval_method": "hybrid",
            "n_chunks": 10,
            "chunk_size": 256,
            "chunk_overlap": 128,
            "reranking": True,
            "query_expansion": True,
        },
    }


def create_rag_dataset(num_documents: int = 20, num_queries: int = 50) -> RAGDataset:
    """Create a complete RAG evaluation dataset."""

    # Generate base documents
    base_docs = generate_technical_documents()

    # Expand document set if needed
    documents = base_docs.copy()
    while len(documents) < num_documents:
        # Create variations of existing documents
        base_doc = random.choice(base_docs)
        variant = Document(
            id=f"{base_doc.id}_v{len(documents)}",
            title=f"{base_doc.title} - Extended",
            content=base_doc.content
            + "\n\nAdditional information: "
            + base_doc.content[:200],
            metadata={**base_doc.metadata, "variant": True},
            relevance_score=base_doc.relevance_score * 0.8,
            chunk_size=base_doc.chunk_size,
        )
        documents.append(variant)

    # Generate queries
    base_queries = generate_queries(documents[: len(base_docs)])
    queries = base_queries.copy()

    # Generate additional queries if needed
    while len(queries) < num_queries:
        base_query = random.choice(base_queries)
        variant = RAGQuery(
            question=f"{base_query.question} (Variant {len(queries)})",
            ground_truth_answer=base_query.ground_truth_answer,
            query_type=base_query.query_type,
            required_doc_ids=base_query.required_doc_ids,
            difficulty=min(1.0, base_query.difficulty + random.uniform(0, 0.2)),
            metadata={**base_query.metadata, "variant": True},
        )
        queries.append(variant)

    return RAGDataset(
        documents=documents[:num_documents],
        queries=queries[:num_queries],
        metadata={
            "created_at": datetime.now().isoformat(),
            "num_documents": len(documents),
            "num_queries": len(queries),
            "domains": list({d.metadata.get("domain", "") for d in documents}),
        },
    )


def analyze_retrieval_quality(
    retrieved_chunks: list[DocumentChunk], query: RAGQuery, documents: list[Document]
) -> dict[str, float]:
    """Analyze the quality of retrieved chunks for a query."""

    # Get required document IDs
    required_ids = set(query.required_doc_ids)
    retrieved_ids = {chunk.doc_id for chunk in retrieved_chunks}

    # Calculate metrics
    precision = (
        len(required_ids & retrieved_ids) / len(retrieved_ids) if retrieved_ids else 0
    )
    recall = (
        len(required_ids & retrieved_ids) / len(required_ids) if required_ids else 0
    )
    f1 = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    # Calculate coverage - how much of required documents is covered
    coverage = (
        len(required_ids & retrieved_ids) / len(required_ids) if required_ids else 0
    )

    # Calculate redundancy - repeated information
    unique_docs = len({chunk.doc_id for chunk in retrieved_chunks})
    redundancy = 1 - (unique_docs / len(retrieved_chunks)) if retrieved_chunks else 0

    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "coverage": coverage,
        "redundancy": redundancy,
        "num_chunks": len(retrieved_chunks),
    }
