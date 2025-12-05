"""
Code Review System - Setup Configuration
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="code-review-system",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Comprehensive automated code review system for Python projects",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/code-review-system",
    packages=find_packages(where="automation"),
    package_dir={"": "automation"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Quality Assurance",
        "Topic :: Software Development :: Testing",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.8",
    install_requires=[
        "anthropic>=0.25.0",  # For Claude API
        "openai>=1.0.0",  # For OpenAI API (optional)
        "pydantic>=2.0.0",  # For validation
        "rich>=13.0.0",  # For pretty console output
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "ruff>=0.1.0",
            "mypy>=1.0.0",
        ],
        "all": [
            "anthropic>=0.25.0",
            "openai>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "code-review=run_all_validations:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": [
            "*.json",
            "*.md",
            "instructions.md",
            "required_checks.json",
        ],
    },
)
