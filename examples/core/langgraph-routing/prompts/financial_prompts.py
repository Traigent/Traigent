"""Financial document analysis prompt templates."""

FINANCIAL_PROMPTS = {
    "standard": """Analyze this financial document and provide a summary.

Document:
{document}

Provide a clear analysis covering:
1. Document type (invoice, statement, report, etc.)
2. Key monetary amounts
3. Relevant dates and periods
4. Parties/entities involved
5. Important line items or transactions
6. Payment terms or due dates (if applicable)

Analysis:""",
    "detailed_breakdown": """Provide a detailed financial breakdown of this document.

Document:
{document}

DETAILED BREAKDOWN:

1. DOCUMENT IDENTIFICATION
   - Type: [invoice/statement/report/etc.]
   - Reference Number: [if present]
   - Date/Period: [relevant dates]

2. PARTIES
   - From/Issuer: [name and details]
   - To/Recipient: [name and details]
   - Account Numbers: [if present]

3. FINANCIAL SUMMARY
   | Category | Amount |
   |----------|--------|
   [itemized amounts]

4. LINE ITEM DETAILS
   [detailed breakdown of each item]

5. CALCULATIONS
   - Subtotal:
   - Taxes/Fees:
   - Discounts:
   - Total:

6. PAYMENT INFORMATION
   - Terms:
   - Due Date:
   - Methods Accepted:

7. NOTES AND OBSERVATIONS
   [any special terms or notable items]

Detailed Breakdown:""",
    "summary_focused": """Provide an executive summary of this financial document.

Document:
{document}

EXECUTIVE SUMMARY:

KEY FIGURES:
- Total Amount: [primary monetary value]
- Period/Date: [relevant timeframe]

BOTTOM LINE:
[One sentence summary of what this document represents]

HIGHLIGHTS:
- [Key point 1]
- [Key point 2]
- [Key point 3]

ACTION ITEMS:
- [Any required actions with deadlines]

TRENDS/OBSERVATIONS:
- [Notable patterns or changes, if applicable]

Executive Summary:""",
    "metric_extraction": """Extract specific financial metrics from this document.

Document:
{document}

EXTRACTED METRICS:

IDENTIFICATION:
- Document Type:
- Reference/ID:
- Date:

PRIMARY AMOUNTS:
- Gross Amount: $
- Net Amount: $
- Tax Amount: $
- Total Due/Paid: $

SECONDARY METRICS (if available):
- Revenue: $
- Expenses: $
- Profit/Loss: $
- Margin: %

COMPARATIVE METRICS (if available):
- YoY Change: %
- Period-over-Period: %
- Budget Variance: %

RATIOS (if calculable):
- Gross Margin: %
- Operating Margin: %
- Other:

TRANSACTION COUNT: [number of line items/transactions]

PAYMENT TERMS:
- Due Date:
- Terms:
- Status:

Extracted Metrics:""",
}
