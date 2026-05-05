# TVL Website Source

The public TVL website is not maintained from this repository.

Canonical source lives in the standalone TVL repository:

- Repository: https://github.com/Traigent/tvl
- Website source: https://github.com/Traigent/tvl/tree/main/website
- Live site: https://tvl-lang.org/

That repository builds `website/dist/public/` and deploys it to the S3 and
CloudFront stack for `tvl-lang.org`.

Do not add a JavaScript package manifest or lockfile here. Keeping a stale copy
in `Traigent/Traigent` causes Dependabot to report vulnerabilities for a site
that is not served from this repo.
