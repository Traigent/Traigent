# Contributing

## Setup

```bash
npm ci
npm run ci
```

## Expectations

- Keep the JS SDK aligned with the sibling Python SDK where the feature is part of the shared optimization contract.
- Prefer adding parity tests when you add or change optimization behavior.
- Keep walkthrough and examples runnable in offline/mock mode unless the example is explicitly under `walkthrough/real/`.

## Before Opening a PR

```bash
npm run ci
npm pack --dry-run
npm run smoke:examples
npm run smoke:walkthrough
```

