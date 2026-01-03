# Traigent Development Dockerfile
# Multi-stage build for development and testing

FROM python:3.12-slim as base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies and create non-root user for security
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 1000 traigent \
    && useradd --uid 1000 --gid 1000 --shell /bin/bash --create-home traigent

WORKDIR /app

# Development stage
FROM base as development

# Install uv for fast dependency management
RUN pip install uv

# Copy dependency files first for better caching
COPY pyproject.toml ./

# Install all development dependencies
RUN uv pip install --system -e ".[dev,integrations,analytics,security,test]"

# Copy the rest of the application
COPY . .

# Set default environment for development
ENV TRAIGENT_MOCK_LLM=true \
    TRAIGENT_LOG_LEVEL=DEBUG

# Run as non-root user
USER traigent

# Default command
CMD ["pytest", "tests/", "-v"]

# Testing stage (lighter, focused on test execution)
FROM base as testing

# Install only test dependencies
COPY pyproject.toml ./
RUN pip install -e ".[test]"

COPY . .

ENV TRAIGENT_MOCK_LLM=true

# Run as non-root user
USER traigent

CMD ["pytest", "tests/", "-v", "--tb=short"]

# Production stage (minimal footprint)
FROM base as production

COPY pyproject.toml ./
RUN pip install -e "."

COPY traigent/ ./traigent/

ENV TRAIGENT_MOCK_LLM=false

# Run as non-root user
USER traigent

CMD ["python", "-m", "traigent"]
