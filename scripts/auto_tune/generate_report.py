#!/usr/bin/env python3
"""
Generate performance reports in markdown and HTML formats.
Creates detailed reports with metrics, plots, and recommendations.
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import yaml


def load_params() -> Dict[str, Any]:
    """Load parameters from params.yaml."""
    with open("params.yaml") as f:
        return yaml.safe_load(f)


def load_performance_report() -> Dict[str, Any]:
    """Load performance evaluation report."""
    with open("performance_report.json") as f:
        return json.load(f)


def load_optimization_results() -> Dict[str, Any]:
    """Load optimization results."""
    with open("optimization_results/latest.json") as f:
        return json.load(f)


def generate_recommendations(
    performance: Dict[str, Any], results: Dict[str, Any]
) -> List[str]:
    """
    Generate actionable recommendations based on performance.
    """
    recommendations = []

    # Check for regressions
    if performance.get("comparison", {}).get("has_regression"):
        recommendations.append(
            "⚠️ **Performance Regression Detected**: Review changes and consider reverting "
            "or investigating the cause of degradation."
        )

    # Check for improvements
    if performance.get("comparison", {}).get("has_improvement"):
        recommendations.append(
            "✅ **Performance Improved**: Consider updating the baseline to lock in improvements."
        )

    # Cost optimization
    current_cost = performance.get("current_metrics", {}).get("total_cost", 0)
    if current_cost > 10:
        recommendations.append(
            "💰 **High Cost Alert**: Consider using more cost-effective models or reducing trial count."
        )

    # Latency optimization
    avg_latency = performance.get("current_metrics", {}).get("avg_latency", 0)
    if avg_latency > 2.0:
        recommendations.append(
            "⏱️ **High Latency**: Consider using faster models or implementing caching strategies."
        )

    # Best configuration
    best_config = results.get("best_config", {})
    if best_config:
        model = best_config.get("model", "unknown")
        provider = best_config.get("provider", "unknown")
        recommendations.append(
            f"🎯 **Best Configuration**: {provider}/{model} showed the best performance. "
            "Consider making this the default."
        )

    # Trial efficiency
    trials_completed = performance.get("current_metrics", {}).get("trials_completed", 0)
    if trials_completed < 5:
        recommendations.append(
            "📊 **Limited Trials**: Consider increasing trial count for more comprehensive optimization."
        )

    return recommendations


def generate_markdown_report(
    performance: Dict[str, Any], results: Dict[str, Any], params: Dict[str, Any]
) -> str:
    """Generate markdown format report."""

    report = []
    report.append("# 🤖 TraiGent Auto-Tuning Report")
    report.append("")
    report.append(
        f"**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}"
    )
    report.append(f"**Environment:** {os.environ.get('CI_COMMIT_BRANCH', 'local')}")
    report.append(f"**Commit:** {os.environ.get('CI_COMMIT_SHA', 'local')[:8]}")
    report.append("")

    # Executive Summary
    report.append("## 📊 Executive Summary")
    report.append("")

    current = performance.get("current_metrics", {})
    report.append(f"- **Trials Completed:** {current.get('trials_completed', 0)}")
    report.append(f"- **Best Score:** {current.get('best_score', 0):.3f}")
    report.append(f"- **Total Cost:** ${current.get('total_cost', 0):.2f}")
    report.append(f"- **Average Latency:** {current.get('avg_latency', 0):.2f}s")
    report.append("")

    # Performance Comparison
    if performance.get("comparison"):
        report.append("## 📈 Performance Comparison")
        report.append("")
        report.append("| Metric | Baseline | Current | Change | Status |")
        report.append("|--------|----------|---------|--------|--------|")

        for metric, data in performance["comparison"]["metrics"].items():
            baseline = data.get("baseline", 0)
            current = data.get("current", 0)
            change = data.get("change_pct", 0)

            if data.get("is_improvement"):
                status = "✅ Improved"
            elif data.get("is_regression"):
                status = "❌ Regressed"
            else:
                status = "➡️ Stable"

            report.append(
                f"| {metric} | {baseline:.3f} | {current:.3f} | "
                f"{change:+.1f}% | {status} |"
            )
        report.append("")

    # Best Configuration
    best_config = results.get("best_config", {})
    if best_config:
        report.append("## 🎯 Best Configuration")
        report.append("")
        report.append("```json")
        report.append(json.dumps(best_config, indent=2))
        report.append("```")
        report.append("")

    # Top Performing Trials
    trials = results.get("trials", [])
    if trials:
        report.append("## 🏆 Top 5 Performing Trials")
        report.append("")
        report.append("| Rank | Model | Score | Latency | Cost |")
        report.append("|------|-------|-------|---------|------|")

        sorted_trials = sorted(trials, key=lambda x: x.get("score", 0), reverse=True)[
            :5
        ]
        for i, trial in enumerate(sorted_trials, 1):
            model = trial.get("config", {}).get("model", "unknown")
            score = trial.get("score", 0)
            latency = trial.get("latency", 0)
            cost = trial.get("cost", 0)
            report.append(
                f"| {i} | {model} | {score:.3f} | {latency:.2f}s | ${cost:.3f} |"
            )
        report.append("")

    # Recommendations
    recommendations = generate_recommendations(performance, results)
    if recommendations:
        report.append("## 💡 Recommendations")
        report.append("")
        for rec in recommendations:
            report.append(f"- {rec}")
        report.append("")

    # Configuration Details
    report.append("## ⚙️ Configuration")
    report.append("")
    report.append(f"- **Strategy:** {params['optimize']['strategy']}")
    report.append(f"- **Max Trials:** {params['optimize']['max_trials']}")
    report.append(f"- **Budget:** ${params['optimize']['budget']}")
    report.append(f"- **Threshold:** {params['evaluate']['threshold']*100}%")
    report.append("")

    # Footer
    report.append("---")
    report.append("*Generated by TraiGent Auto-Tuning Pipeline*")

    return "\n".join(report)


def generate_html_report(markdown_content: str) -> str:
    """Convert markdown report to HTML."""

    html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TraiGent Auto-Tuning Report</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }
        .container {
            background: white;
            border-radius: 10px;
            padding: 40px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.1);
        }
        h1 {
            color: #667eea;
            border-bottom: 3px solid #667eea;
            padding-bottom: 10px;
        }
        h2 {
            color: #764ba2;
            margin-top: 30px;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }
        th, td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        th {
            background: #f8f9fa;
            font-weight: 600;
        }
        tr:hover {
            background: #f8f9fa;
        }
        code {
            background: #f4f4f4;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: 'Courier New', monospace;
        }
        pre {
            background: #f4f4f4;
            padding: 15px;
            border-radius: 5px;
            overflow-x: auto;
        }
        ul {
            padding-left: 30px;
        }
        li {
            margin: 10px 0;
        }
        .metric-good { color: #28a745; font-weight: bold; }
        .metric-bad { color: #dc3545; font-weight: bold; }
        .metric-neutral { color: #6c757d; }
    </style>
</head>
<body>
    <div class="container">
        {content}
    </div>
</body>
</html>
"""

    # Simple markdown to HTML conversion (basic implementation)
    html_content = markdown_content

    # Convert headers
    html_content = html_content.replace("# ", "<h1>").replace("\n## ", "</h1>\n<h2>")
    html_content = html_content.replace("\n### ", "</h2>\n<h3>")

    # Convert emphasis
    html_content = html_content.replace("**", "<strong>").replace("**", "</strong>")

    # Convert lists
    lines = html_content.split("\n")
    in_list = False
    converted_lines = []

    for line in lines:
        if line.startswith("- "):
            if not in_list:
                converted_lines.append("<ul>")
                in_list = True
            converted_lines.append(f"<li>{line[2:]}</li>")
        else:
            if in_list:
                converted_lines.append("</ul>")
                in_list = False
            converted_lines.append(line)

    if in_list:
        converted_lines.append("</ul>")

    html_content = "\n".join(converted_lines)

    # Convert status symbols
    html_content = html_content.replace("✅", '<span class="metric-good">✅</span>')
    html_content = html_content.replace("❌", '<span class="metric-bad">❌</span>')
    html_content = html_content.replace("➡️", '<span class="metric-neutral">➡️</span>')

    return html_template.format(content=html_content)


def main():
    """Main report generation pipeline."""
    print("📝 Generating performance reports...")

    # Load data
    params = load_params()
    performance = load_performance_report()
    results = load_optimization_results()

    # Create reports directory
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)

    # Generate markdown report
    markdown_report = generate_markdown_report(performance, results, params)

    # Save markdown report
    with open(reports_dir / "performance_report.md", "w") as f:
        f.write(markdown_report)
    print(f"✅ Markdown report saved to {reports_dir}/performance_report.md")

    # Generate and save HTML report
    if params["report"].get("format") == "html" or params["report"].get(
        "include_plots"
    ):
        html_report = generate_html_report(markdown_report)
        with open(reports_dir / "performance_report.html", "w") as f:
            f.write(html_report)
        print(f"✅ HTML report saved to {reports_dir}/performance_report.html")

    print("✅ Report generation complete")

    return 0


if __name__ == "__main__":
    exit(main())
