"""Custom evaluators for the LangGraph document routing demo."""

from __future__ import annotations

import json


def routing_accuracy_scorer(
    output: dict | str | None,
    expected: dict | None,
    input_data: dict | None = None,
    config: dict | None = None,
    llm_metrics: dict | None = None,
) -> float:
    """Score routing accuracy - 1.0 if correctly routed, 0.0 otherwise.

    Args:
        output: The pipeline output containing routing information
        expected: The expected output with expected_doc_type
        input_data: The input data (unused)
        config: The configuration used (unused)
        llm_metrics: LLM metrics (unused)

    Returns:
        1.0 if routing matches expected, 0.0 otherwise
    """
    if output is None or expected is None:
        return 0.0

    # Extract actual doc type from output
    actual_type = ""
    if isinstance(output, dict):
        routing = output.get("routing", {})
        if isinstance(routing, dict):
            actual_type = routing.get("type", "").lower().strip()
        elif isinstance(routing, str):
            actual_type = routing.lower().strip()
        # Also check top-level doc_type
        if not actual_type:
            actual_type = output.get("doc_type", "").lower().strip()
    elif isinstance(output, str):
        # Try to parse as JSON
        try:
            parsed = json.loads(output)
            if isinstance(parsed, dict):
                actual_type = parsed.get("routing", {}).get("type", "").lower().strip()
        except (json.JSONDecodeError, TypeError):
            # Extract from string response
            output_lower = output.lower()
            if "legal" in output_lower:
                actual_type = "legal"
            elif "financial" in output_lower:
                actual_type = "financial"

    # Extract expected doc type
    expected_type = ""
    if isinstance(expected, dict):
        expected_type = expected.get("expected_doc_type", "").lower().strip()
    elif isinstance(expected, str):
        expected_type = expected.lower().strip()

    return 1.0 if actual_type == expected_type else 0.0


def processing_quality_scorer(
    output: dict | str | None,
    expected: dict | None,
    input_data: dict | None = None,
    config: dict | None = None,
    llm_metrics: dict | None = None,
) -> float:
    """Score processing quality based on expected elements coverage.

    Evaluates how well the analysis covers the expected elements for the document.

    Args:
        output: The pipeline output containing analysis content
        expected: The expected output with expected_elements list
        input_data: The input data (unused)
        config: The configuration used (unused)
        llm_metrics: LLM metrics (unused)

    Returns:
        Score between 0.0 and 1.0 based on element coverage
    """
    if output is None or expected is None:
        return 0.0

    # Get expected elements
    expected_elements = []
    if isinstance(expected, dict):
        elements = expected.get("expected_elements", [])
        if isinstance(elements, list):
            expected_elements = [str(e).lower() for e in elements]

    if not expected_elements:
        # No expected elements defined, give partial credit if there's any output
        return 0.5 if output else 0.0

    # Extract content from output
    content = ""
    if isinstance(output, dict):
        # Check various possible content fields
        content = str(output.get("content", ""))
        if not content:
            content = str(output.get("analysis", ""))
        if not content:
            content = str(output.get("result", ""))
        # Also include routing info in content check
        routing = output.get("routing", {})
        if isinstance(routing, dict):
            content += " " + str(routing)
    elif isinstance(output, str):
        content = output

    content_lower = content.lower()

    # Count how many expected elements are mentioned in the output
    matched = 0
    for element in expected_elements:
        # Check for element or related terms
        element_variations = _get_element_variations(element)
        if any(var in content_lower for var in element_variations):
            matched += 1

    # Calculate coverage score
    coverage = matched / len(expected_elements)

    # Apply quality multiplier based on content length (penalize very short responses)
    length_factor = min(1.0, len(content) / 200)  # Full credit at 200+ chars

    return coverage * (0.7 + 0.3 * length_factor)


def _get_element_variations(element: str) -> list[str]:
    """Get variations of an element name for fuzzy matching.

    Args:
        element: The element name to get variations for

    Returns:
        List of string variations to match against
    """
    variations = [element]

    # Common element mappings - comprehensive synonyms for fuzzy matching
    element_mappings = {
        # Legal document elements
        "parties": [
            "parties",
            "party",
            "between",
            "contractor",
            "client",
            "provider",
            "company",
            "corporation",
            "llc",
            "inc",
        ],
        "effective_date": ["effective", "date", "commence", "starting", "as of"],
        "term": ["term", "duration", "period", "months", "years", "remain in effect"],
        "compensation": [
            "compensation",
            "payment",
            "salary",
            "fee",
            "price",
            "rate",
            "pay",
            "monthly",
            "annual",
        ],
        "confidentiality": [
            "confidential",
            "nda",
            "non-disclosure",
            "proprietary",
            "secret",
            "private",
        ],
        "confidentiality_scope": [
            "confidential",
            "information",
            "trade secret",
            "proprietary",
            "scope",
            "includes",
        ],
        "termination": ["terminat", "cancel", "end", "notice", "cease"],
        "obligations": [
            "obligation",
            "shall",
            "must",
            "agree",
            "responsible",
            "duties",
            "responsibilities",
        ],
        "purpose": ["purpose", "whereas", "intent", "objective", "goal"],
        "grant_of_license": ["grant", "license", "right", "permission", "authorize"],
        "restrictions": ["restrict", "limit", "prohibit", "may not", "shall not"],
        "intellectual_property": [
            "intellectual",
            "property",
            "ip",
            "patent",
            "copyright",
            "trademark",
            "ownership",
        ],
        "warranties": ["warrant", "guarantee", "represent", "assure"],
        "liability": ["liabil", "indemnif", "damages", "responsible"],
        "survival": ["surviv", "continue", "remain", "persist"],
        # Financial document elements
        "invoice_number": ["invoice", "inv-", "inv#", "number", "reference", "#"],
        "total": ["total", "amount", "due", "sum", "grand total"],
        "line_items": ["item", "description", "service", "product", "qty", "quantity"],
        "dates": ["date", "period", "due", "issued", "effective"],
        "vendor": ["vendor", "from", "seller", "supplier", "issuer"],
        "customer": ["customer", "bill to", "buyer", "client", "recipient"],
        "subtotal": ["subtotal", "sub-total", "before tax", "net amount"],
        "tax": ["tax", "vat", "gst", "sales tax", "%"],
        "payment_terms": ["payment", "terms", "net 30", "net 45", "due date"],
        "revenue": ["revenue", "sales", "income", "receipts"],
        "revenue_breakdown": [
            "revenue",
            "sales",
            "product",
            "service",
            "subscription",
            "breakdown",
            "category",
        ],
        "revenue_categories": ["revenue", "food", "beverage", "catering", "sales"],
        "expenses": ["expense", "cost", "operating", "expenditure"],
        "expenses_breakdown": ["expense", "cost", "category", "breakdown", "operating"],
        "net_income": ["net income", "net profit", "bottom line", "profit", "earnings"],
        "gross_margin": ["gross margin", "gross profit", "margin"],
        "gross_revenue": ["gross revenue", "total revenue", "gross sales"],
        "gross_profit": ["gross profit", "gross margin", "profit margin"],
        "operating_margin": [
            "operating margin",
            "operating income",
            "operating profit",
        ],
        "operating_expenses": [
            "operating expense",
            "opex",
            "overhead",
            "labor",
            "rent",
        ],
        "period": [
            "period",
            "quarter",
            "q1",
            "q2",
            "q3",
            "q4",
            "month",
            "year",
            "fiscal",
            "ended",
        ],
        "transactions": ["transaction", "deposit", "withdrawal", "transfer", "payment"],
        "balance": ["balance", "beginning", "ending", "cash"],
        "account": ["account", "acct", "acc#", "number"],
        "account_info": ["account", "holder", "number", "statement"],
        # Additional financial elements
        "po_number": ["po", "purchase order", "order number", "po-", "po #"],
        "buyer": ["buyer", "purchaser", "ordering", "from"],
        "quantities": ["qty", "quantity", "amount", "units"],
        "unit_prices": ["unit price", "price", "rate", "per unit", "$"],
        "shipping": ["shipping", "freight", "delivery", "transport"],
        "total_expenses": ["total expense", "total cost", "sum", "grand total"],
        "advance": ["advance", "prepaid", "deposit", "previous"],
        "amount_due": ["amount due", "balance due", "owed", "payable"],
        "employee": ["employee", "name", "submitted by", "staff"],
        "business_purpose": ["purpose", "business", "reason", "justification"],
        "expense_items": ["expense", "item", "category", "description"],
        "categories": [
            "category",
            "type",
            "classification",
            "airfare",
            "hotel",
            "meal",
        ],
        "amounts": ["amount", "$", "total", "sum", "value"],
        "department": ["department", "dept", "division", "team"],
        "fiscal_year": ["fiscal", "fy", "year", "annual", "budget"],
        "total_budget": ["total budget", "proposed", "budget", "allocation"],
        "category_breakdown": ["category", "breakdown", "allocation", "distribution"],
        "percentages": ["%", "percent", "proportion", "share"],
        "yoy_comparison": [
            "yoy",
            "year over year",
            "compared",
            "increase",
            "decrease",
            "growth",
            "change",
        ],
        "report_date": ["report date", "as of", "dated", "period"],
        "total_outstanding": ["outstanding", "receivable", "owed", "due"],
        "aging_buckets": [
            "aging",
            "0-30",
            "31-60",
            "61-90",
            "90+",
            "current",
            "overdue",
        ],
        "customer_detail": ["customer", "detail", "breakdown", "by customer"],
        "collection_notes": ["collection", "notes", "comments", "status"],
        "recommended_actions": ["recommend", "action", "suggest", "next steps"],
        "credit_memo_number": ["credit memo", "cm-", "credit #", "memo"],
        "original_invoice": ["original", "invoice", "reference", "related"],
        "reason": ["reason", "cause", "because", "due to", "returned"],
        "credit_items": ["credit", "item", "returned", "merchandise"],
        "merchandise_credit": ["merchandise", "credit", "goods", "product"],
        "total_credit": ["total credit", "credit amount", "refund"],
        "account_balance": ["account balance", "balance", "current", "new balance"],
        "cogs": ["cogs", "cost of goods", "cost of sales", "direct cost"],
        "margins": ["margin", "%", "percent", "ratio", "profit margin"],
        "transaction_reference": ["reference", "transaction", "id", "number", "wt-"],
        "originator": ["originator", "sender", "from", "payer"],
        "beneficiary": ["beneficiary", "recipient", "to", "payee"],
        "exchange_rate": ["exchange", "rate", "conversion", "forex", "eur", "usd"],
        "fees": ["fee", "charge", "cost", "wire", "transfer fee"],
        "total_debited": ["debited", "total", "withdrawn", "charged"],
        "tax_year": ["tax year", "year", "fiscal", "2024", "2023"],
        "payer": ["payer", "employer", "company", "from"],
        "recipient": ["recipient", "payee", "contractor", "freelancer"],
        "compensation_amount": ["compensation", "income", "earnings", "paid"],
        "federal_withholding": ["federal", "withhold", "tax withheld", "box 4"],
        "state_withholding": ["state", "withhold", "state tax", "box 5"],
        "quarterly_breakdown": ["quarterly", "q1", "q2", "q3", "q4", "breakdown"],
        # Cash flow elements
        "operating_activities": ["operating", "activities", "operations", "cash from"],
        "investing_activities": ["investing", "activities", "capital", "expenditure"],
        "financing_activities": ["financing", "activities", "dividend", "debt"],
        "cash_flow": ["cash flow", "net change", "cash", "flow"],
        "beginning_balance": ["beginning", "opening", "start", "initial"],
        "ending_balance": ["ending", "closing", "final", "end"],
    }

    # Add mappings if element matches
    for key, mapping_list in element_mappings.items():
        if element in mapping_list or key == element:
            variations.extend(mapping_list)

    # Add underscore/space variations
    variations.append(element.replace("_", " "))
    variations.append(element.replace("_", "-"))

    return list(set(variations))


def combined_scorer(
    output: dict | str | None,
    expected: dict | None,
    input_data: dict | None = None,
    config: dict | None = None,
    llm_metrics: dict | None = None,
) -> float:
    """Combined scorer that evaluates both routing accuracy and processing quality.

    Weights routing accuracy more heavily since incorrect routing leads to
    wrong analysis pipeline.

    Args:
        output: The pipeline output
        expected: The expected output
        input_data: The input data (unused)
        config: The configuration used (unused)
        llm_metrics: LLM metrics (unused)

    Returns:
        Weighted score between 0.0 and 1.0
    """
    routing_score = routing_accuracy_scorer(output, expected)
    processing_score = processing_quality_scorer(output, expected)

    # If routing is wrong, heavily penalize but give some credit for output
    if routing_score < 1.0:
        return 0.2 * processing_score

    # Routing correct: weight both scores
    return 0.4 * routing_score + 0.6 * processing_score


def latency_scorer(
    output: dict | str | None,
    expected: dict | None,
    input_data: dict | None = None,
    config: dict | None = None,
    llm_metrics: dict | None = None,
) -> float:
    """Extract latency from LLM metrics.

    Args:
        output: The pipeline output (unused)
        expected: The expected output (unused)
        input_data: The input data (unused)
        config: The configuration used (unused)
        llm_metrics: LLM metrics containing duration info

    Returns:
        Latency in milliseconds, or 0.0 if not available
    """
    if llm_metrics and isinstance(llm_metrics, dict):
        # Try various latency field names
        latency = llm_metrics.get("total_duration", 0)
        if not latency:
            latency = llm_metrics.get("duration_ms", 0)
        if not latency:
            latency = llm_metrics.get("latency", 0)
        return float(latency)
    return 0.0


def cost_scorer(
    output: dict | str | None,
    expected: dict | None,
    input_data: dict | None = None,
    config: dict | None = None,
    llm_metrics: dict | None = None,
) -> float:
    """Extract cost from LLM metrics.

    Args:
        output: The pipeline output (unused)
        expected: The expected output (unused)
        input_data: The input data (unused)
        config: The configuration used (unused)
        llm_metrics: LLM metrics containing cost info

    Returns:
        Cost in USD, or 0.0 if not available
    """
    if llm_metrics and isinstance(llm_metrics, dict):
        cost = llm_metrics.get("total_cost", 0)
        if not cost:
            cost = llm_metrics.get("cost", 0)
        return float(cost)
    return 0.0
