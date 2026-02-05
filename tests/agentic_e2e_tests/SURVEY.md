# Building a comprehensive E2E testing system with video evidence and AI verification

The optimal stack for full-stack E2E testing (CLI → SDK → Backend → Frontend) combines **Playwright for video capture**, **Gemini 3 for native video analysis**, **docker-compose with pytest-docker for orchestration**, and **Allure for reporting with embedded video artifacts**. This combination provides production-ready tooling with strong documentation, active maintenance, and proven enterprise adoption. The key insight: AI verification works best as a complement to traditional pixel-based visual testing, not a replacement—use pixelmatch for precise validation and AI models for semantic verification of dynamic content.

## User story generation: BMAD excels at planning, not reverse-engineering

**BMAD (Breakthrough Method of Agile AI-Driven Development)** has emerged as a significant open-source framework with **32,900+ GitHub stars** and 21 specialized AI agents covering the full agile lifecycle. However, a critical clarification: BMAD generates user stories during the *planning phase before implementation*, not from existing tests. It uses agents like "Mary" (Business Analyst) and a Scrum Master for story preparation, outputting Gherkin-style acceptance criteria through commands like `/create-prd` and `/create-epics-and-stories`.

For teams wanting to extract user stories from existing test code, **no mature turnkey solution exists**. Recent academic research (arXiv:2509.19587) demonstrates that LLMs can achieve an **F1 score of 0.8** for recovering user stories from code up to 200 lines, with 8B parameter models matching 70B performance through one-shot prompting. The practical approach requires building custom pipelines using Claude or GPT-4 APIs to analyze test files.

**pytest-bdd** remains the recommended integration for pytest-based testing, supporting native Gherkin syntax with code generation (`pytest-bdd generate <feature>`). The key pattern is treating feature files as living documentation—the Gherkin scenarios themselves become the user stories. For legacy codebases, the most realistic path involves gradual migration to BDD format rather than automated extraction.

## Video evidence capture across the full stack

**Playwright dominates frontend video capture** with configuration as simple as `video: 'retain-on-failure'` in `playwright.config.ts`. Videos save as WebM format with configurable resolution, and the `retain-on-failure` mode automatically deletes passing test videos—essential for CI storage management. Playwright also provides trace files (`.zip` archives with screenshots, snapshots, and network logs) that complement video for debugging.

For Selenium environments, **Selenoid** provides the most mature Docker-based video recording solution with H.264 MP4 output, configurable frame rates up to 24fps, and REST API management. The setup requires Docker infrastructure but offers more granular control over video parameters than Playwright's simpler approach.

**CLI/terminal testing requires asciinema** with its CAST format—newline-delimited JSON capturing terminal output as timestamped text rather than pixels. This produces highly compressible recordings (down to 15% with zstd) where text remains searchable and selectable. The **asciinema-automation** Python package enables reproducible recordings by reading commands from text files and waiting for expected output before proceeding, making it suitable for pytest fixtures:

```python
@pytest.fixture
def cli_recording(artifact_dir, request):
    cast_path = artifact_dir / f"{request.node.name}.cast"
    # Use asciinema-automation for reproducible CLI captures
    yield cast_path
```

SDK execution traces integrate best through **OpenTelemetry instrumentation**, creating spans for each method call with attributes for parameters and results. Export traces to Jaeger or Grafana Tempo for visual timelines. For backend API recording during tests, **Playwright's built-in HAR recording** captures network traffic without additional tools, while **mitmproxy** provides more powerful interception capabilities when needed.

## AI-powered verification: Gemini leads for native video analysis

The most significant finding is that **Google Gemini 3 supports native video analysis**, accepting MP4, MOV, AVI, and WebM files directly without frame extraction. This provides a substantial workflow advantage over Claude Vision and GPT-4V, which require preprocessing video into individual frames. Gemini's configurable FPS (1-10) and agentic vision capabilities with think-act-observe loops make it particularly suited for test verification tasks.

Cost analysis reveals dramatic differences: **Gemini Flash costs approximately $0.40/million tokens** compared to Claude Sonnet at $3/million and GPT-4o at $5/million. For high-volume testing, this translates to roughly **$0.50 per 1,000 screenshots with Gemini Flash** versus $4.80 with Claude Sonnet. Self-hosted LLaVA eliminates API costs entirely but requires GPU infrastructure and delivers lower accuracy.

**Applitools Eyes remains the industry leader** for production AI visual testing, trained on 4 billion app screens with claimed 99.9999% accuracy. Unlike pixel-diff tools, it intelligently handles dynamic content like ads and personalized dashboards. Applitools integrates with Playwright, Cypress, and Selenium through SDKs, providing cross-browser testing via its Ultrafast Grid.

The recommended architecture combines approaches: use **pixelmatch for pixel-perfect validation** of critical UI elements and **AI models for semantic verification** of behavior and dynamic content. A practical pattern extracts frames at action boundaries (before/during/after user interactions) and passes them to AI with structured prompts requesting JSON responses for consistent parsing:

```python
result = await verifier.verify(screenshot, {
    "expected": "Dashboard visible with user greeting",
    "failure_indicators": ["error message", "login form still visible"]
})
expect(result.passed).toBe(True)
```

Handling false positives requires confidence thresholds, human review queues for ambiguous results, and multi-model consensus for critical checks. Scott Logic's January 2026 testing found Gemini 3 successfully identified broken images and data mismatches but occasionally hallucinated bugs in clean recordings—underscoring the need for hybrid approaches.

## Test orchestration patterns for multi-service environments

Docker-compose with health checks forms the foundation of reliable multi-service orchestration. The critical pattern uses **`depends_on` with `condition: service_healthy`** to ensure proper service ordering:

```yaml
backend:
  healthcheck:
    test: ["CMD-SHELL", "curl -f http://localhost:5000/health || exit 1"]
    interval: 5s
    timeout: 5s
    retries: 15
  depends_on:
    database:
      condition: service_healthy
```

**pytest-docker** provides the essential bridge between pytest fixtures and docker-compose, with `wait_until_responsive()` for polling service readiness. For programmatic container control, **testcontainers-python** offers disposable containers that work well for isolated tests, while docker-compose remains better suited for complex multi-service stacks.

Synchronization between test layers requires explicit wait strategies—never `time.sleep()`. Implement polling-based waits with condition callbacks:

```python
def wait_for_condition(check, timeout=30.0, pause=0.5):
    start = time.time()
    while time.time() - start < timeout:
        if check():
            return
        time.sleep(pause)
    raise TimeoutError(f"Condition not met after {timeout}s")
```

For message queue testing (Kafka, RabbitMQ), testcontainers provides purpose-built containers that integrate with consumer fixtures. The multi-layer coordination pattern flows through CLI → SDK → API → Frontend verification, ensuring consistency across the stack within a single test.

**pytest-xdist with `--dist loadscope`** groups tests by module to share fixtures efficiently during parallel execution. Agent interaction automation during tests benefits from recording and replay patterns—capture real agent responses during development, then replay deterministically in CI through mock fixtures.

## Reporting and evidence integration with Allure

**Allure Report provides the most mature video embedding** for pytest through `allure.attach.file()` with automatic attachment on test failure using pytest hooks. The combination of `@allure.story()`, `@allure.feature()`, and `@allure.link()` decorators enables rich traceability linking tests to user stories with embedded video evidence.

Setup requires `pip install allure-pytest` and `pytest --alluredir=allure-results`, with historical reports enabled through the Docker-based Allure Server (frankescobar/allure-docker-service) that maintains trend analysis across up to 20 previous runs. The critical integration for automatic video attachment:

```python
@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_teardown(item, nextitem):
    yield
    for file in artifacts_dir.iterdir():
        if file.suffix == ".webm":
            allure.attach.file(file, name=file.name,
                               attachment_type=allure.attachment_type.WEBM)
```

**ReportPortal** offers a stronger alternative for enterprise teams, providing AI-powered test analytics, real-time reporting, ML-based failure reason detection, and native integrations with 30+ frameworks. It handles video attachments in test logs and supports Sauce Labs plugin integration for embedded playback.

Historical dashboards pair **Grafana with InfluxDB** for time-series visualization of test metrics—pass/fail trends, duration analysis, and flakiness detection. ReportPortal provides equivalent capabilities through built-in widgets. For artifact storage, **S3 with lifecycle rules** handles retention policies automatically, with tiered retention keeping failed test videos for 30-90 days while passing test videos expire after 7-14 days.

Video compression through FFmpeg with VP9 encoding (`-c:v libvpx-vp9 -crf 35`) significantly reduces storage costs. The recommended approach filters uploads to include only failed test videos and compresses before upload.

## Production-ready implementations and templates

**Netflix's SafeTest framework** stands out as a notable open-source reference, combining E2E testing capabilities (Cypress/Playwright) with component-level testing. It provides automatic linking of video replays to test results and supports React, Vue, Svelte, Angular, and NextJS. For self-healing Selenium tests, **Healenium** uses ML to automatically replace broken locators at runtime, with an IntelliJ IDEA plugin for code updates.

The enterprise-grade Python boilerplate at [nirtal85/Playwright-Python-Example](https://github.com/nirtal85/Playwright-Python-Example) provides a production-ready foundation with Allure reports, BrowserStack integration, accessibility testing via axe, and pytest-split for parallel execution. For TypeScript, [akshayp7/playwright-typescript-playwright-test](https://github.com/akshayp7/playwright-typescript-playwright-test) covers web, API, mobile, visual, and load testing with Lighthouse metrics and SonarQube integration.

The recommended folder structure organizes artifacts by date and test suite:

```
project/
├── test-results/          # Playwright default output
│   └── <test-name>/
│       ├── video.webm
│       └── trace.zip
├── videos/
│   └── YYYY-MM-DD/
│       └── <test-suite>-<browser>/
└── allure-results/
```

Common pitfalls include recording everything (wasting storage), hardcoded test data instead of factories, test interdependence requiring shared state, and excessive UI testing that violates the testing pyramid. Address flaky tests immediately rather than ignoring them—track flakiness rates through ReportPortal or CircleCI Test Insights.

## Recommended tool combinations by use case

For **small teams (1-3 engineers)**, start with Playwright + pytest + pytest-html + GitHub Actions artifacts. This provides video capture, basic reporting, and artifact storage without infrastructure overhead.

For **medium teams (4-10 engineers)**, adopt Playwright + pytest + Allure + docker-compose + pytest-docker + S3 for artifacts. Add Grafana dashboards for historical tracking and consider BrowserStack for cross-browser coverage.

For **large teams (10+ engineers)**, implement Playwright + Applitools Eyes + Allure/ReportPortal + Healenium (if using Selenium) + Kubernetes for scaling + custom Gemini-based verification agents. Use SafeTest patterns for component integration testing and deploy dedicated test artifact infrastructure.

The critical success factors remain consistent across scales: adopt `retain-on-failure` video recording by default, implement the Page Object Model pattern from project start, establish video retention policies before storage costs spiral, and treat AI verification as a complement to traditional testing rather than a replacement. The most effective systems combine the precision of pixel-based comparison for known critical elements with the flexibility of AI analysis for semantic verification and dynamic content handling.
