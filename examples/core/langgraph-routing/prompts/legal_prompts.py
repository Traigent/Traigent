"""Legal document analysis prompt templates."""

LEGAL_PROMPTS = {
    "standard": """Analyze this legal document and provide a comprehensive summary.

Document:
{document}

Provide a clear analysis covering:
1. Document type and purpose
2. Parties involved
3. Key terms and conditions
4. Important dates and deadlines
5. Main obligations of each party
6. Notable clauses or provisions

Analysis:""",
    "structured_extraction": """Extract structured information from this legal document.

Document:
{document}

Extract the following elements (if present):

DOCUMENT_TYPE: [type of legal document]
EFFECTIVE_DATE: [when the document takes effect]
PARTIES:
- Party 1: [name and role]
- Party 2: [name and role]
TERM/DURATION: [how long the agreement lasts]
KEY_CLAUSES:
- [clause name]: [brief description]
OBLIGATIONS:
- [party]: [obligation description]
TERMINATION_CONDITIONS: [how the agreement can end]
GOVERNING_LAW: [jurisdiction]
SPECIAL_PROVISIONS: [any unique or notable terms]

Structured extraction:""",
    "risk_focused": """Perform a risk analysis of this legal document.

Document:
{document}

Analyze potential risks and concerns:

1. LIABILITY EXPOSURE
   - What liabilities does each party assume?
   - Are there caps on liability?
   - What indemnification provisions exist?

2. TERMINATION RISKS
   - What triggers termination?
   - What are the notice requirements?
   - What happens upon termination?

3. COMPLIANCE CONCERNS
   - Are there regulatory requirements?
   - What reporting obligations exist?
   - Are there audit rights?

4. AMBIGUOUS LANGUAGE
   - Are there terms that could be interpreted multiple ways?
   - Are definitions clear and complete?

5. MISSING PROTECTIONS
   - What standard clauses are absent?
   - What additional protections should be considered?

RISK LEVEL: [Low/Medium/High]
KEY RECOMMENDATIONS: [bullet points]

Risk Analysis:""",
    "compliance_oriented": """Review this legal document for compliance considerations.

Document:
{document}

COMPLIANCE REVIEW:

1. REGULATORY REQUIREMENTS
   - Identify applicable regulations (GDPR, CCPA, HIPAA, SOX, etc.)
   - Assess compliance with industry standards
   - Note any required disclosures

2. CONTRACTUAL COMPLIANCE
   - Are required elements present (offer, acceptance, consideration)?
   - Is the contract enforceable?
   - Are signature requirements met?

3. DATA AND PRIVACY
   - How is personal data handled?
   - Are data protection obligations addressed?
   - Is there a data breach notification clause?

4. CORPORATE GOVERNANCE
   - Are proper authorizations in place?
   - Is the signing authority appropriate?
   - Are there board approval requirements?

5. AUDIT AND REPORTING
   - What records must be maintained?
   - Are there audit rights?
   - What reporting is required?

COMPLIANCE STATUS: [Compliant/Needs Review/Non-Compliant]
REMEDIATION ITEMS: [if any]

Compliance Review:""",
}
