# Security Policy

## Supported Versions

Security fixes are applied to the current `main` branch.

Historical milestone branches and local worktree branches are not supported release lines.

## Reporting a Vulnerability

Do not open a public GitHub issue for suspected vulnerabilities.

Use one of these private channels:

- GitHub private vulnerability reporting:
  - https://github.com/traigent/traigent-js/security/advisories/new
- if you are reporting from an internal Traigent environment, use the internal security/support channel assigned to your team

When reporting a vulnerability, include:

- affected version or commit
- reproduction steps or proof of concept
- impact assessment
- any proposed mitigation

## Disclosure Expectations

- We will acknowledge receipt as soon as practical.
- We may ask for reproduction details or a minimal test case.
- Please keep reports private until a fix or mitigation is available.

## Scope Notes

- Experimental runtime seamless execution is trusted-local only and must be explicitly enabled.
- Remote cloud execution for arbitrary JS agents is out-of-scope by design for this SDK.
