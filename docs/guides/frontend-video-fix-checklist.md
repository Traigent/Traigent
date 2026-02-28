# Frontend Video Fix Checklist

Use this checklist to make video-related frontend bugs implementation-ready.

## Required Repro Data

1. Exact page/route where video appears.
2. Browser + version.
3. Device + OS.
4. Current behavior.
5. Expected behavior.
6. Step-by-step reproduction path.
7. Console/network errors (if any).
8. Screenshot or short capture.

## Triage Severity

Classify quickly:

1. `S1` - video does not load at all.
2. `S2` - video loads but playback is broken/intermittent.
3. `S3` - cosmetic or low-impact issues.

## Technical Checks

1. Source URL reachable and returns `200`.
2. Correct MIME type for media response.
3. CORS headers valid for hosting domain.
4. Frontend player receives non-empty source.
5. Autoplay/muted policy behavior verified by browser.
6. No CSP block on media resources.

## Acceptance Criteria

The issue can be closed when:

1. Repro case is validated on at least one affected browser.
2. Fix is verified on affected browser + one additional browser.
3. No regression on unaffected pages/components.
4. Before/after evidence is attached to the issue.
