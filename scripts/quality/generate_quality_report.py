#!/usr/bin/env python3
"""Generate comprehensive code quality report."""

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


class QualityReporter:
    """Generate code quality reports for various tools."""

    def __init__(self, base_path: str = "."):
        self.base_path = Path(base_path)
        self.reports_dir = self.base_path / "reports" / "code-quality"
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        # Use the current Python executable (should be the virtual environment)
        self.python_exe = sys.executable

    def run_flake8(self) -> dict:
        """Run flake8 and return results."""
        try:
            # First check if flake8 is available
            check_result = subprocess.run(
                [self.python_exe, "-c", "import flake8; print('available')"],
                capture_output=True,
                text=True,
                cwd=self.base_path,
            )
            if check_result.returncode != 0:
                return {"error": "flake8 not installed"}

            result = subprocess.run(
                [
                    self.python_exe,
                    "-m",
                    "flake8",
                    "--max-line-length=100",
                    "--extend-ignore=E203,W503",
                    "--exclude=__pycache__,venv,archive",
                    "--format=json",
                    "traigent/",
                ],
                capture_output=True,
                text=True,
                cwd=self.base_path,
                timeout=30,
            )

            if result.returncode in [0, 1]:  # 0 = no errors, 1 = found errors
                try:
                    # Try to parse JSON from stdout first, then stderr
                    for output_source in [result.stdout, result.stderr]:
                        if output_source.strip():
                            # Clean up any leading/trailing text that might interfere with JSON parsing
                            output = output_source.strip()
                            if output.startswith("json"):
                                output = output[4:].strip()
                            if output:
                                try:
                                    return json.loads(output)
                                except json.JSONDecodeError:
                                    continue
                    # If we get here, flake8 ran but didn't produce valid JSON
                    # This might be normal if there are no issues to report
                    return {}
                except Exception as e:
                    return {
                        "error": f"Failed to parse flake8 output: {e}",
                        "raw_output": result.stdout[:500],
                    }
            else:
                return {
                    "error": f"flake8 failed with code {result.returncode}",
                    "stderr": result.stderr,
                }
        except subprocess.TimeoutExpired:
            return {"error": "flake8 timed out"}
        except FileNotFoundError:
            return {"error": "flake8 not installed"}
        except Exception as e:
            return {"error": f"Unexpected error running flake8: {e}"}

    def run_ruff(self) -> dict:
        """Run ruff and return results."""
        try:
            # First check if ruff is available
            check_result = subprocess.run(
                [self.python_exe, "-c", "import ruff; print('available')"],
                capture_output=True,
                text=True,
                cwd=self.base_path,
            )

            if check_result.returncode != 0:
                return {"error": "ruff not installed"}

            result = subprocess.run(
                [
                    self.python_exe,
                    "-m",
                    "ruff",
                    "check",
                    "traigent/",
                    "--output-format=json",
                ],
                capture_output=True,
                text=True,
                cwd=self.base_path,
                timeout=30,
            )

            if result.returncode in [0, 1]:  # 0 = no errors, 1 = found errors
                try:
                    if result.stdout.strip():
                        return json.loads(result.stdout)
                    return []
                except json.JSONDecodeError as e:
                    return {
                        "error": f"Failed to parse ruff output: {e}",
                        "raw_output": result.stdout[:500],
                    }
            else:
                return {
                    "error": f"ruff failed with code {result.returncode}",
                    "stderr": result.stderr,
                }
        except subprocess.TimeoutExpired:
            return {"error": "ruff timed out"}
        except FileNotFoundError:
            return {"error": "ruff not installed"}
        except Exception as e:
            return {"error": f"Unexpected error running ruff: {e}"}

    def run_mypy(self) -> str:
        """Run mypy and return output."""
        try:
            # First check if mypy is available
            check_result = subprocess.run(
                [self.python_exe, "-c", "import mypy; print('available')"],
                capture_output=True,
                text=True,
                cwd=self.base_path,
            )

            if check_result.returncode != 0:
                return "mypy not installed"

            result = subprocess.run(
                [
                    self.python_exe,
                    "-m",
                    "mypy",
                    "traigent/",
                    "--ignore-missing-imports",
                    "--no-error-summary",
                    "--show-error-codes",
                ],
                capture_output=True,
                text=True,
                cwd=self.base_path,
                timeout=60,
            )

            output = result.stdout
            stderr = result.stderr

            if result.returncode not in [0, 1]:
                return f"mypy failed with code {result.returncode}\n{output}\n{stderr}"

            # If mypy ran successfully (even with errors), it means it's installed
            # Combine stdout and stderr for the final output, but check availability separately
            full_output = (output + stderr).strip()
            return full_output if full_output else "No mypy output"
        except subprocess.TimeoutExpired:
            return "mypy timed out"
        except FileNotFoundError:
            return "mypy not installed"
        except Exception as e:
            return f"Unexpected error running mypy: {e}"

    def generate_report(self) -> str:
        """Generate comprehensive quality report."""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        report_file = self.reports_dir / f"quality_report_{timestamp}.md"

        print("🔍 Running code quality analysis...")

        flake8_results = self.run_flake8()
        ruff_results = self.run_ruff()
        mypy_output = self.run_mypy()

        # Count issues and check tool availability
        flake8_available = not (
            isinstance(flake8_results, dict) and "error" in flake8_results
        )
        ruff_available = not (
            isinstance(ruff_results, dict) and "error" in ruff_results
        )
        mypy_available = mypy_output not in [
            "mypy not installed",
            "mypy timed out",
        ] and not mypy_output.startswith("mypy failed with code")

        flake8_count = (
            len(flake8_results)
            if isinstance(flake8_results, dict) and flake8_available
            else 0
        )
        ruff_count = (
            len(ruff_results)
            if isinstance(ruff_results, list) and ruff_available
            else 0
        )

        # Prepare rendered sections
        if flake8_available:
            flake8_section = f"```json\n{json.dumps(flake8_results, indent=2)}\n```"
        else:
            flake8_section = (
                "```\n"
                "❌ Flake8 not installed\n"
                "\n"
                "To install: pip install flake8\n"
                "```"
            )

        if ruff_available:
            ruff_section = f"```json\n{json.dumps(ruff_results, indent=2)}\n```"
        else:
            ruff_section = (
                "```\n"
                "❌ Ruff not installed\n"
                "\n"
                "To install: pip install ruff\n"
                "```"
            )

        mypy_section = (
            "❌ MyPy not installed\n\nTo install: pip install mypy\n"
            if not mypy_available
            else mypy_output
        )

        # Generate markdown report
        report_content = f"""# Code Quality Report
Generated: {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")}

## Tool Availability
- **Flake8**: {"✅ Available" if flake8_available else "❌ Not installed"}
- **Ruff**: {"✅ Available" if ruff_available else "❌ Not installed"}
- **MyPy**: {"✅ Available" if mypy_available else "❌ Not installed"}

## Summary
- **Flake8 Issues**: {flake8_count}
- **Ruff Issues**: {ruff_count}
- **Total Issues**: {flake8_count + ruff_count}

## Flake8 Results
{flake8_section}

## Ruff Results
{ruff_section}

## MyPy Output
```
{mypy_section}
```

## Installation Instructions

If any tools are missing, install them with:

```bash
# Install all development tools
pip install -r requirements/requirements-dev.txt

# Or install individually
pip install flake8 mypy ruff
```

## Recommendations

### High Priority
1. Fix undefined names and import errors
2. Address syntax errors
3. Resolve type annotation issues

### Medium Priority
1. Reduce cognitive complexity in large functions
2. Fix exception handling patterns
3. Remove unused imports

### Low Priority
1. Fix code formatting issues
2. Address documentation formatting
3. Clean up trailing whitespace

## Next Steps
1. Install missing quality tools if needed
2. Run `python scripts/quality/fix_common_issues.py` for automated fixes
3. Review and fix high-priority issues manually
4. Consider breaking down complex functions
5. Update SonarQube configuration if using server mode
"""

        with open(report_file, "w", encoding="utf-8") as f:
            f.write(report_content)

        print(f"✅ Report generated: {report_file}")
        return str(report_file)


def main():
    """Generate quality report."""
    reporter = QualityReporter()
    report_path = reporter.generate_report()
    print(f"📊 Quality report available at: {report_path}")


if __name__ == "__main__":
    main()
