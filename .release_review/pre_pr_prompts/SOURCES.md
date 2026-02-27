# Sources Used To Build This Prompt Pack

## External Guidance

- Google Engineering Practices: Code Review - Reviewer Guide
  https://google.github.io/eng-practices/review/reviewer/

- OWASP Code Review Guide
  https://owasp.org/www-project-code-review-guide/

- OWASP Application Security Verification Standard (ASVS)
  https://owasp.org/www-project-application-security-verification-standard/

- OWASP Threat Modeling Cheat Sheet
  https://cheatsheetseries.owasp.org/cheatsheets/Threat_Modeling_Cheat_Sheet.html

- NIST SP 800-218: Secure Software Development Framework (SSDF)
  https://csrc.nist.gov/pubs/sp/800/218/final

- SLSA v1.0 Specification
  https://slsa.dev/spec/v1.0/

- Martin Fowler: The Practical Test Pyramid
  https://martinfowler.com/articles/practical-test-pyramid.html

- Python `doctest` documentation
  https://docs.python.org/3/library/doctest.html

- Google Developer Documentation Style Guide: Code Samples
  https://developers.google.com/style/code-samples

- Diataxis Documentation Framework
  https://diataxis.fr/

- RFC 2119: Key words for use in RFCs to indicate requirement levels
  https://www.rfc-editor.org/rfc/rfc2119

- RFC 3552: Guidelines for Writing RFC Text on Security Considerations
  https://www.rfc-editor.org/rfc/rfc3552

- Semantic Versioning 2.0.0
  https://semver.org/

## Additional Model Consultation

Prompt quality was additionally pressure-tested using local Claude Code CLI:

- Tool: `claude` (Claude Code CLI 2.1.58)
- Model: `claude-opus-4-6`
- Effort: `high` (CLI supports `low|medium|high`)
- Date: 2026-02-26
- Purpose: derive strong generic pre-PR prompt structure with explicit gates, severity rubric, and machine-parseable outputs.
