# n8n Access Request Template (Yossi)

Use this template to request everything needed to start integration work without back-and-forth.

## Message Template

```text
Hi <owner>,

We are ready to start Traigent integration on n8n and need access details to unblock execution.

Please provide:
1) Workspace access
   - n8n URL:
   - invited users:
   - expected access date:

2) Runtime details
   - deployment type: cloud | self-hosted
   - version:
   - startup/runbook link:

3) Secrets and auth
   - required env vars / secret keys:
   - storage location (vault/env/other):
   - owner for rotation:

4) First workflow to optimize
   - workflow name:
   - tunable ID:
   - dataset source:
   - success metric(s):

5) Networking constraints
   - allow-list/proxy/VPN requirements:

Once shared, we will execute the runbook and start baseline + optimization.
```

## Completion Checklist

Mark complete when all are present:

1. Workspace URL and access granted.
2. Runtime type/version confirmed.
3. Secrets contract documented.
4. First workflow identified with tunable ID.
5. Network requirements validated.
