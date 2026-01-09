"""Router prompt templates for document classification."""

ROUTER_PROMPTS = {
    "concise": """Classify this document as either "legal" or "financial".

Document:
{document}

Classification (respond with only "legal" or "financial"):""",
    "detailed": """You are a document classification expert. Analyze the document below and classify it into one of two categories:

LEGAL DOCUMENTS include:
- Contracts and agreements (service, employment, licensing)
- Non-disclosure agreements (NDAs)
- Legal notices and termination letters
- Leases and amendments
- Settlement agreements
- Powers of attorney
- Partnership agreements
- Any document with legal terms, parties, obligations, or clauses

FINANCIAL DOCUMENTS include:
- Invoices and receipts
- Financial statements and reports
- Bank statements
- Purchase orders
- Expense reports
- Budget proposals
- Accounts receivable/payable reports
- Tax documents
- Credit memos
- Wire transfer confirmations
- Any document with monetary amounts, transactions, or accounting information

Document to classify:
{document}

Based on the content, structure, and terminology, this document is (respond with only "legal" or "financial"):""",
    "few_shot": """Classify documents as "legal" or "financial".

Example 1:
Document: "This Agreement is entered into between Party A and Party B. WHEREAS the parties wish to establish terms... The parties agree to the following obligations..."
Classification: legal

Example 2:
Document: "Invoice #12345. Bill To: ABC Corp. Items: Consulting Services $5,000. Tax: $400. Total Due: $5,400. Payment Terms: Net 30."
Classification: financial

Example 3:
Document: "NON-DISCLOSURE AGREEMENT. The Receiving Party agrees to maintain confidentiality of all proprietary information disclosed by the Disclosing Party..."
Classification: legal

Example 4:
Document: "Q3 Financial Report. Revenue: $2.5M (+12% YoY). Operating Expenses: $1.8M. Net Income: $700K. Gross Margin: 45%."
Classification: financial

Now classify this document:
{document}

Classification:""",
    "chain_of_thought": """Analyze this document step by step to determine if it is a legal or financial document.

Document:
{document}

Step 1 - Identify key terms and phrases:
Look for legal terminology (agreement, parties, hereby, whereas, obligations, clause, indemnify) or financial terminology (invoice, amount, balance, revenue, expense, payment, account).

Step 2 - Determine the document's primary purpose:
Is it establishing rights/obligations between parties (legal) or recording/reporting monetary transactions (financial)?

Step 3 - Consider the document structure:
Legal documents typically have sections for parties, terms, and signatures. Financial documents typically have line items, amounts, and totals.

Based on this analysis, provide your classification.

Reasoning:""",
}
