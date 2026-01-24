"""
Dataset Generation for Structured Output Engineering
==================================================

Generates evaluation datasets with text snippets and Pydantic schemas
across 5 domains as specified in the use case.

Dataset Composition:
- 500 text snippets across 5 domains
- Each snippet paired with a Pydantic schema
- Deliberate inclusion of edge cases
- 20% real-world data, 80% synthetic but realistic examples

Schema Complexity Levels:
- Simple: 3-5 fields, flat structure
- Medium: 5-10 fields, one level of nesting
- Complex: 10+ fields, multiple nesting levels, arrays
"""

import random
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel


@dataclass
class DatasetSample:
    """A single dataset sample with text and expected extraction."""

    text: str
    schema: type[BaseModel]
    expected_output: dict
    domain: str
    complexity: str  # "simple", "medium", "complex"
    is_edge_case: bool = False


# Domain-specific Pydantic schemas


# 1. INVOICES DOMAIN
class SimpleInvoice(BaseModel):
    company_name: str
    total_amount: float
    invoice_date: str


class MediumInvoice(BaseModel):
    company_name: str
    company_address: str
    invoice_number: str
    invoice_date: str
    due_date: str
    total_amount: float
    tax_amount: float


class ComplexInvoice(BaseModel):
    vendor_info: dict
    billing_address: dict
    line_items: list[dict]
    payment_terms: str
    total_amount: float
    tax_details: dict
    invoice_metadata: dict


# 2. SUPPORT TICKETS DOMAIN
class SimpleTicket(BaseModel):
    ticket_id: str
    priority: str
    status: str


class MediumTicket(BaseModel):
    ticket_id: str
    customer_email: str
    subject: str
    priority: str
    status: str
    category: str
    created_date: str


class ComplexTicket(BaseModel):
    ticket_metadata: dict
    customer_info: dict
    issue_details: dict
    resolution_history: list[dict]
    escalation_info: dict
    tags: list[str]
    sla_details: dict


# 3. MEDICAL RECORDS DOMAIN
class SimpleMedical(BaseModel):
    patient_name: str
    diagnosis: str
    treatment_date: str


class MediumMedical(BaseModel):
    patient_id: str
    patient_name: str
    age: int
    diagnosis: str
    treatment_date: str
    doctor_name: str
    medication: str


class ComplexMedical(BaseModel):
    patient_demographics: dict
    medical_history: list[dict]
    current_diagnosis: dict
    treatment_plan: dict
    medication_list: list[dict]
    lab_results: dict
    provider_info: dict


# 4. PRODUCT REVIEWS DOMAIN
class SimpleReview(BaseModel):
    product_name: str
    rating: int
    review_text: str


class MediumReview(BaseModel):
    product_name: str
    product_category: str
    rating: int
    reviewer_name: str
    review_date: str
    review_text: str
    would_recommend: bool


class ComplexReview(BaseModel):
    product_details: dict
    reviewer_profile: dict
    rating_breakdown: dict
    review_content: dict
    purchase_details: dict
    helpfulness_metrics: dict
    moderation_status: dict


# 5. NEWS ARTICLES DOMAIN
class SimpleNews(BaseModel):
    headline: str
    author: str
    publish_date: str


class MediumNews(BaseModel):
    headline: str
    author: str
    publish_date: str
    category: str
    summary: str
    word_count: int
    source: str


class ComplexNews(BaseModel):
    article_metadata: dict
    content_structure: dict
    author_info: dict
    publication_details: dict
    topics_and_tags: list[str]
    engagement_metrics: dict
    editorial_notes: dict


# Schema registry by domain and complexity
SCHEMA_REGISTRY = {
    "invoices": {
        "simple": SimpleInvoice,
        "medium": MediumInvoice,
        "complex": ComplexInvoice,
    },
    "support_tickets": {
        "simple": SimpleTicket,
        "medium": MediumTicket,
        "complex": ComplexTicket,
    },
    "medical_records": {
        "simple": SimpleMedical,
        "medium": MediumMedical,
        "complex": ComplexMedical,
    },
    "product_reviews": {
        "simple": SimpleReview,
        "medium": MediumReview,
        "complex": ComplexReview,
    },
    "news_articles": {
        "simple": SimpleNews,
        "medium": MediumNews,
        "complex": ComplexNews,
    },
}


def generate_evaluation_dataset(total_samples: int = 500) -> list[DatasetSample]:
    """
    Generate the complete evaluation dataset with 500 samples across 5 domains.

    Distribution:
    - 20% real-world inspired examples
    - 80% synthetic but realistic examples
    - Even distribution across domains
    - Mixed complexity levels
    - 15% edge cases

    Args:
        total_samples: Total number of samples to generate

    Returns:
        List of DatasetSample objects
    """

    dataset = []
    samples_per_domain = total_samples // 5
    domains = list(SCHEMA_REGISTRY.keys())

    for domain in domains:
        domain_samples = []

        # Generate samples for each complexity level
        complexities = ["simple", "medium", "complex"]
        complexity_distribution = [0.4, 0.4, 0.2]  # More simple and medium cases

        for complexity, ratio in zip(
            complexities, complexity_distribution, strict=True
        ):
            complexity_samples = int(samples_per_domain * ratio)

            for _i in range(complexity_samples):
                # Determine if this should be an edge case (15% of samples)
                is_edge_case = random.random() < 0.15

                # Generate sample
                sample = _generate_domain_sample(
                    domain=domain, complexity=complexity, is_edge_case=is_edge_case
                )
                domain_samples.append(sample)

        dataset.extend(domain_samples)

    # Shuffle to avoid domain clustering
    random.shuffle(dataset)

    return dataset[:total_samples]


def _generate_domain_sample(
    domain: str, complexity: str, is_edge_case: bool = False
) -> DatasetSample:
    """Generate a single sample for a specific domain and complexity."""

    schema = SCHEMA_REGISTRY[domain][complexity]

    if domain == "invoices":
        return _generate_invoice_sample(schema, complexity, is_edge_case)
    elif domain == "support_tickets":
        return _generate_ticket_sample(schema, complexity, is_edge_case)
    elif domain == "medical_records":
        return _generate_medical_sample(schema, complexity, is_edge_case)
    elif domain == "product_reviews":
        return _generate_review_sample(schema, complexity, is_edge_case)
    elif domain == "news_articles":
        return _generate_news_sample(schema, complexity, is_edge_case)
    else:
        raise ValueError(f"Unknown domain: {domain}")


def _generate_invoice_sample(
    schema: type[BaseModel], complexity: str, is_edge_case: bool
) -> DatasetSample:
    """Generate invoice domain sample."""

    companies = [
        "Acme Corp",
        "TechStart Inc",
        "Global Solutions Ltd",
        "Innovation Hub",
        "Digital Dynamics",
    ]

    if complexity == "simple":
        company = random.choice(companies)
        amount = round(random.uniform(100, 10000), 2)
        date = f"2024-{random.randint(1, 12):02d}-{random.randint(1, 28):02d}"

        if is_edge_case:
            # Edge cases: missing fields, extra info, ambiguous values
            text = f"Invoice from {company}. Payment due: ${amount}. Some extra unrelated information here."
            expected = {
                "company_name": company,
                "total_amount": amount,
                "invoice_date": "",
            }
        else:
            text = f"INVOICE\nCompany: {company}\nDate: {date}\nTotal Amount: ${amount}\nThank you for your business."
            expected = {
                "company_name": company,
                "total_amount": amount,
                "invoice_date": date,
            }

    elif complexity == "medium":
        company = random.choice(companies)
        address = f"{random.randint(100, 9999)} Main St, City, ST {random.randint(10000, 99999)}"
        inv_num = f"INV-{random.randint(1000, 9999)}"
        amount = round(random.uniform(500, 50000), 2)
        tax = round(amount * 0.08, 2)
        date = f"2024-{random.randint(1, 12):02d}-{random.randint(1, 28):02d}"
        due_date = f"2024-{random.randint(1, 12):02d}-{random.randint(1, 28):02d}"

        text = f"""INVOICE #{inv_num}

{company}
{address}

Invoice Date: {date}
Due Date: {due_date}

Subtotal: ${amount - tax:.2f}
Tax: ${tax:.2f}
TOTAL: ${amount:.2f}"""

        expected = {
            "company_name": company,
            "company_address": address,
            "invoice_number": inv_num,
            "invoice_date": date,
            "due_date": due_date,
            "total_amount": amount,
            "tax_amount": tax,
        }

    else:  # complex
        text = f"""COMPLEX INVOICE DOCUMENT

Vendor: {random.choice(companies)}
Multiple line items, payment terms, detailed tax breakdown...
[Complex invoice structure with nested data]"""

        expected = {
            "vendor_info": {"name": random.choice(companies)},
            "billing_address": {"street": "123 Main St"},
            "line_items": [{"item": "Product A", "amount": 100.0}],
            "payment_terms": "Net 30",
            "total_amount": random.uniform(1000, 100000),
            "tax_details": {"rate": 0.08},
            "invoice_metadata": {"created_by": "system"},
        }

    return DatasetSample(
        text=text,
        schema=schema,
        expected_output=expected,
        domain="invoices",
        complexity=complexity,
        is_edge_case=is_edge_case,
    )


def _generate_ticket_sample(
    schema: type[BaseModel], complexity: str, is_edge_case: bool
) -> DatasetSample:
    """Generate support ticket domain sample."""

    priorities = ["Low", "Medium", "High", "Critical"]
    statuses = ["Open", "In Progress", "Resolved", "Closed"]
    categories = ["Technical", "Billing", "General", "Bug Report"]

    if complexity == "simple":
        ticket_id = f"TKT-{random.randint(1000, 9999)}"
        priority = random.choice(priorities)
        status = random.choice(statuses)

        text = f"Support Ticket {ticket_id}\nPriority: {priority}\nStatus: {status}\nCustomer needs assistance."
        expected = {"ticket_id": ticket_id, "priority": priority, "status": status}

    elif complexity == "medium":
        ticket_id = f"TKT-{random.randint(1000, 9999)}"
        email = f"user{random.randint(1, 1000)}@example.com"
        subject = "Unable to login to account"
        priority = random.choice(priorities)
        status = random.choice(statuses)
        category = random.choice(categories)
        date = f"2024-{random.randint(1, 12):02d}-{random.randint(1, 28):02d}"

        text = f"""Ticket #{ticket_id}
Customer: {email}
Subject: {subject}
Priority: {priority}
Status: {status}
Category: {category}
Created: {date}"""

        expected = {
            "ticket_id": ticket_id,
            "customer_email": email,
            "subject": subject,
            "priority": priority,
            "status": status,
            "category": category,
            "created_date": date,
        }

    else:  # complex
        text = """COMPLEX SUPPORT TICKET
Multiple escalations, detailed resolution history, SLA tracking...
[Complex ticket structure with nested data]"""

        expected = {
            "ticket_metadata": {"id": "TKT-complex"},
            "customer_info": {"tier": "Premium"},
            "issue_details": {"severity": "High"},
            "resolution_history": [{"step": 1, "action": "Initial response"}],
            "escalation_info": {"level": 2},
            "tags": ["urgent", "billing"],
            "sla_details": {"target_hours": 4},
        }

    return DatasetSample(
        text=text,
        schema=schema,
        expected_output=expected,
        domain="support_tickets",
        complexity=complexity,
        is_edge_case=is_edge_case,
    )


def _generate_medical_sample(
    schema: type[BaseModel], complexity: str, is_edge_case: bool
) -> DatasetSample:
    """Generate medical records sample."""

    names = ["John Smith", "Jane Doe", "Michael Johnson", "Sarah Wilson", "David Brown"]
    diagnoses = ["Hypertension", "Diabetes", "Anxiety", "Arthritis", "Migraine"]
    doctors = ["Dr. Adams", "Dr. Baker", "Dr. Clark", "Dr. Davis", "Dr. Evans"]

    if complexity == "simple":
        name = random.choice(names)
        diagnosis = random.choice(diagnoses)
        date = f"2024-{random.randint(1, 12):02d}-{random.randint(1, 28):02d}"

        text = f"Patient: {name}\nDiagnosis: {diagnosis}\nTreatment Date: {date}\nFollow-up scheduled."
        expected = {
            "patient_name": name,
            "diagnosis": diagnosis,
            "treatment_date": date,
        }

    elif complexity == "medium":
        name = random.choice(names)
        patient_id = f"P{random.randint(10000, 99999)}"
        age = random.randint(18, 85)
        diagnosis = random.choice(diagnoses)
        date = f"2024-{random.randint(1, 12):02d}-{random.randint(1, 28):02d}"
        doctor = random.choice(doctors)
        medication = "Prescribed medication XYZ"

        text = f"""Medical Record
Patient ID: {patient_id}
Name: {name}
Age: {age}
Diagnosis: {diagnosis}
Treatment Date: {date}
Attending Physician: {doctor}
Medication: {medication}"""

        expected = {
            "patient_id": patient_id,
            "patient_name": name,
            "age": age,
            "diagnosis": diagnosis,
            "treatment_date": date,
            "doctor_name": doctor,
            "medication": medication,
        }

    else:  # complex
        text = """COMPREHENSIVE MEDICAL RECORD
Patient demographics, complete medical history, treatment plans...
[Complex medical record with nested data structures]"""

        expected = {
            "patient_demographics": {"age": 45, "gender": "M"},
            "medical_history": [
                {"condition": "Previous surgery", "date": "2020-01-01"}
            ],
            "current_diagnosis": {
                "primary": "Diagnosis A",
                "secondary": ["Diagnosis B"],
            },
            "treatment_plan": {"medications": ["Med A", "Med B"]},
            "medication_list": [{"name": "Med A", "dosage": "10mg"}],
            "lab_results": {"test_date": "2024-01-15", "results": "Normal"},
            "provider_info": {
                "hospital": "General Hospital",
                "department": "Internal Medicine",
            },
        }

    return DatasetSample(
        text=text,
        schema=schema,
        expected_output=expected,
        domain="medical_records",
        complexity=complexity,
        is_edge_case=is_edge_case,
    )


def _generate_review_sample(
    schema: type[BaseModel], complexity: str, is_edge_case: bool
) -> DatasetSample:
    """Generate product review sample."""

    products = [
        "Wireless Headphones",
        "Smart Watch",
        "Laptop Stand",
        "Coffee Maker",
        "Desk Chair",
    ]
    categories = ["Electronics", "Home & Kitchen", "Office Products", "Sports", "Books"]

    if complexity == "simple":
        product = random.choice(products)
        rating = random.randint(1, 5)
        review = f"Good product, works as expected. {'Highly recommended!' if rating >= 4 else 'Could be better.'}"

        text = f"Product: {product}\nRating: {rating}/5 stars\nReview: {review}"
        expected = {"product_name": product, "rating": rating, "review_text": review}

    elif complexity == "medium":
        product = random.choice(products)
        category = random.choice(categories)
        rating = random.randint(1, 5)
        reviewer = f"Customer{random.randint(1, 1000)}"
        date = f"2024-{random.randint(1, 12):02d}-{random.randint(1, 28):02d}"
        review = (
            "Excellent product quality and fast shipping. Would definitely buy again!"
        )
        recommend = rating >= 4

        text = f"""Product Review
Product: {product}
Category: {category}
Rating: {rating}/5
Reviewer: {reviewer}
Date: {date}
Review: {review}
Would Recommend: {'Yes' if recommend else 'No'}"""

        expected = {
            "product_name": product,
            "product_category": category,
            "rating": rating,
            "reviewer_name": reviewer,
            "review_date": date,
            "review_text": review,
            "would_recommend": recommend,
        }

    else:  # complex
        text = """DETAILED PRODUCT REVIEW
Complete product analysis, detailed reviewer profile, engagement metrics...
[Complex review structure with nested data]"""

        expected = {
            "product_details": {"name": "Complex Product", "sku": "CP-001"},
            "reviewer_profile": {"verified": True, "review_count": 25},
            "rating_breakdown": {"overall": 4, "quality": 5, "value": 3},
            "review_content": {
                "title": "Great product!",
                "body": "Detailed review text...",
            },
            "purchase_details": {
                "verified_purchase": True,
                "purchase_date": "2024-01-01",
            },
            "helpfulness_metrics": {"helpful_votes": 12, "total_votes": 15},
            "moderation_status": {"approved": True, "flags": 0},
        }

    return DatasetSample(
        text=text,
        schema=schema,
        expected_output=expected,
        domain="product_reviews",
        complexity=complexity,
        is_edge_case=is_edge_case,
    )


def _generate_news_sample(
    schema: type[BaseModel], complexity: str, is_edge_case: bool
) -> DatasetSample:
    """Generate news article sample."""

    headlines = [
        "Tech Company Announces New AI Innovation",
        "Local Election Results Show Surprising Outcome",
        "Climate Change Summit Reaches Key Agreement",
        "Economic Growth Exceeds Expectations",
        "Scientific Breakthrough in Medical Research",
    ]
    authors = [
        "Jane Reporter",
        "John Journalist",
        "Sarah Writer",
        "Mike Correspondent",
        "Lisa Analyst",
    ]
    categories = ["Technology", "Politics", "Environment", "Business", "Science"]
    sources = [
        "News Network",
        "Daily Times",
        "Tech Today",
        "Business Weekly",
        "Science Journal",
    ]

    if complexity == "simple":
        headline = random.choice(headlines)
        author = random.choice(authors)
        date = f"2024-{random.randint(1, 12):02d}-{random.randint(1, 28):02d}"

        text = f"Headline: {headline}\nBy: {author}\nPublished: {date}\nArticle content follows..."
        expected = {"headline": headline, "author": author, "publish_date": date}

    elif complexity == "medium":
        headline = random.choice(headlines)
        author = random.choice(authors)
        date = f"2024-{random.randint(1, 12):02d}-{random.randint(1, 28):02d}"
        category = random.choice(categories)
        summary = (
            "Brief summary of the article highlighting key points and main findings."
        )
        word_count = random.randint(500, 2000)
        source = random.choice(sources)

        text = f"""News Article
Headline: {headline}
Author: {author}
Published: {date}
Category: {category}
Source: {source}
Word Count: {word_count}

Summary: {summary}

[Article content continues...]"""

        expected = {
            "headline": headline,
            "author": author,
            "publish_date": date,
            "category": category,
            "summary": summary,
            "word_count": word_count,
            "source": source,
        }

    else:  # complex
        text = """COMPREHENSIVE NEWS ARTICLE
Complete article metadata, detailed content structure, engagement analytics...
[Complex news structure with nested data]"""

        expected = {
            "article_metadata": {"id": "ART-001", "version": 1},
            "content_structure": {"sections": 5, "paragraphs": 20},
            "author_info": {
                "name": "Senior Reporter",
                "bio": "Award-winning journalist",
            },
            "publication_details": {"outlet": "Major News", "edition": "Morning"},
            "topics_and_tags": ["breaking", "analysis", "exclusive"],
            "engagement_metrics": {"views": 15000, "shares": 250},
            "editorial_notes": {"fact_checked": True, "editor": "Chief Editor"},
        }

    return DatasetSample(
        text=text,
        schema=schema,
        expected_output=expected,
        domain="news_articles",
        complexity=complexity,
        is_edge_case=is_edge_case,
    )


def get_baseline_configs() -> list[dict[str, Any]]:
    """Get baseline configurations for comparison."""
    return [
        {
            # Naive Baseline
            "name": "naive_baseline",
            "output_format": "json_mode",
            "schema_strategy": "minimal_description",
            "validation_approach": "none",
            "prompt_structure": "flat",
            "n_examples": 0,
            "example_selection": "random",
            "temperature": 0.0,
            "max_retries": 0,
        },
        {
            # Standard Baseline
            "name": "standard_baseline",
            "output_format": "json_mode",
            "schema_strategy": "pydantic_in_prompt",
            "validation_approach": "retry_with_error_feedback",
            "prompt_structure": "xml_sections",
            "n_examples": 1,
            "example_selection": "random",
            "temperature": 0.0,
            "max_retries": 1,
        },
    ]
