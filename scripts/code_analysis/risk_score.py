"""Compute module risk scores from available metrics and signals."""

from __future__ import annotations

import argparse
import csv
import io
import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

from traigent.utils.secure_path import (
    PathTraversalError,
    safe_read_text,
    safe_write_text,
    validate_path,
)

try:  # pragma: no cover
    from .analysis_utils import load_coverage_map
except ImportError:  # pragma: no cover
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from scripts.code_analysis.analysis_utils import load_coverage_map


@dataclass
class ModuleSignals:
    complexity: float = 0.0
    fan_in: float = 0.0
    fan_out: float = 0.0
    violations: float = 0.0
    clones: float = 0.0
    churn: float = 0.0
    coverage: Optional[float] = None
    owner_unknown: bool = False


@dataclass
class RiskWeights:
    complexity_z: float = 0.25
    fan_in_z: float = 0.20
    fan_out_z: float = 0.15
    violations_z: float = 0.15
    clones_z: float = 0.10
    churn_z: float = 0.10
    owner_unknown: float = 0.05


def load_risk_config(path: Path) -> RiskWeights:
    if not path.exists():
        return RiskWeights()
    weights: Dict[str, float] = {}
    current_section: Optional[str] = None
    validated_path = validate_path(path, path.parent, must_exist=True)
    content = safe_read_text(validated_path, path.parent)
    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.endswith(":"):
            current_section = line[:-1].strip()
            continue
        if ":" in line and current_section == "weights":
            key, value = line.split(":", 1)
            try:
                weights[key.strip()] = float(value.strip())
            except ValueError:
                continue
    return RiskWeights(
        complexity_z=float(weights.get("complexity_z", 0.25) or 0.0),
        fan_in_z=float(weights.get("fan_in_z", 0.2) or 0.0),
        fan_out_z=float(weights.get("fan_out_z", 0.15) or 0.0),
        violations_z=float(weights.get("violations_z", 0.15) or 0.0),
        clones_z=float(weights.get("clones_z", 0.1) or 0.0),
        churn_z=float(weights.get("churn_z", 0.1) or 0.0),
        owner_unknown=float(weights.get("owner_unknown", 0.05) or 0.0),
    )


def load_metrics(path: Path) -> Dict[str, ModuleSignals]:
    modules: Dict[str, ModuleSignals] = {}
    if not path.exists():
        return modules
    validated_path = validate_path(path, path.parent, must_exist=True)
    content = safe_read_text(validated_path, path.parent)
    reader = csv.DictReader(io.StringIO(content))
    for row in reader:
        module = row.get("module")
        if not module:
            continue
        signals = ModuleSignals(
            complexity=float(row.get("cyclomatic_total", 0.0) or 0.0),
            fan_in=float(row.get("fan_in", 0.0) or 0.0),
            fan_out=float(row.get("fan_out", 0.0) or 0.0),
        )
        modules[module] = signals
    return modules


def load_lint_counts(path: Path) -> Dict[str, int]:
    if not path.exists():
        return {}
    try:
        validated_path = validate_path(path, path.parent, must_exist=True)
        data = json.loads(safe_read_text(validated_path, path.parent))
    except (json.JSONDecodeError, OSError):
        return {}
    counts: Dict[str, int] = {}
    if isinstance(data, list):
        for entry in data:
            filename = entry.get("filename") if isinstance(entry, dict) else None
            if not filename:
                continue
            counts[filename] = counts.get(filename, 0) + 1
    elif isinstance(data, dict):
        files = data.get("files")
        if isinstance(files, list):
            for entry in files:
                filename = entry.get("filename") if isinstance(entry, dict) else None
                if not filename:
                    continue
                messages = entry.get("messages")
                if isinstance(messages, list):
                    counts[filename] = len(messages)
                else:
                    counts[filename] = counts.get(filename, 0) + 1
    return counts


def load_clones(path: Path) -> Dict[str, int]:
    if not path.exists():
        return {}
    try:
        validated_path = validate_path(path, path.parent, must_exist=True)
        data = json.loads(safe_read_text(validated_path, path.parent))
    except (json.JSONDecodeError, OSError):
        return {}
    results: Dict[str, int] = {}
    duplicates = data.get("clones") if isinstance(data, dict) else None
    if isinstance(duplicates, list):
        for clone in duplicates:
            for instance in clone.get("sources", []):
                filename = instance.get("name")
                if filename:
                    results[filename] = results.get(filename, 0) + 1
    return results


def load_churn(path: Path) -> Dict[str, float]:
    churn: Dict[str, float] = {}
    if not path.exists():
        return churn
    validated_path = validate_path(path, path.parent, must_exist=True)
    content = safe_read_text(validated_path, path.parent)
    reader = csv.DictReader(io.StringIO(content))
    for row in reader:
        filename = row.get("path") or row.get("file")
        if not filename:
            continue
        try:
            value = float(row.get("changes", row.get("commits", 0.0)) or 0.0)
        except ValueError:
            value = 0.0
        churn[filename] = value
    return churn


def load_owners(path: Path) -> Dict[str, List[str]]:
    owners: Dict[str, List[str]] = {}
    if not path.exists():
        return owners
    validated_path = validate_path(path, path.parent, must_exist=True)
    content = safe_read_text(validated_path, path.parent)
    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        pattern = parts[0].lstrip("/")
        owners[pattern] = parts[1:]
    return owners


def match_owner(path: str, owners: Mapping[str, Sequence[str]]) -> bool:
    for pattern in owners:
        from fnmatch import fnmatch

        if fnmatch(path, pattern):
            return True
    return False


def map_files_to_modules(modules: Mapping[str, ModuleSignals]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for module in modules:
        filepath = module.replace(".", "/") + ".py"
        mapping[filepath] = module
    return mapping


def normalize_path(raw: str) -> str:
    try:
        return Path(raw).as_posix().lstrip("./")
    except Exception:
        return raw.replace("\\", "/")


def resolve_module_for_path(file_to_module: Mapping[str, str], raw_path: str) -> Optional[str]:
    normalized = normalize_path(raw_path)
    if normalized in file_to_module:
        return file_to_module[normalized]
    for candidate, module in file_to_module.items():
        if normalized.endswith(candidate):
            return module
    return None


def path_to_module(path: str) -> Optional[str]:
    normalized = normalize_path(path)
    if not normalized.endswith(".py"):
        return None
    if normalized.startswith("traigent/"):
        normalized = normalized[4:]
    parts = normalized.split("/")
    if not parts:
        return None
    leaf = parts[-1][:-3]
    if leaf == "__init__":
        parts = parts[:-1]
    else:
        parts[-1] = leaf
    if not parts:
        return "traigent"
    module = ".".join(parts)
    if not module:
        return None
    if not module.startswith("traigent."):
        module = f"traigent.{module}"
    return module


def z_scores(values: Sequence[float]) -> Dict[int, float]:
    scores: Dict[int, float] = {}
    if not values:
        return scores
    mean = statistics.mean(values)
    stdev = statistics.pstdev(values)
    if stdev == 0:
        return dict.fromkeys(range(len(values)), 0.0)
    for idx, value in enumerate(values):
        scores[idx] = (value - mean) / stdev
    return scores


def compute_risk(
    modules: Dict[str, ModuleSignals],
    weights: RiskWeights,
) -> Dict[str, Dict[str, object]]:
    complexity_values = [signals.complexity for signals in modules.values()]
    fan_in_values = [signals.fan_in for signals in modules.values()]
    fan_out_values = [signals.fan_out for signals in modules.values()]
    violations_values = [signals.violations for signals in modules.values()]
    clones_values = [signals.clones for signals in modules.values()]
    churn_values = [signals.churn for signals in modules.values()]

    complexity_z = z_scores(complexity_values)
    fan_in_z = z_scores(fan_in_values)
    fan_out_z = z_scores(fan_out_values)
    violations_z = z_scores(violations_values)
    clones_z = z_scores(clones_values)
    churn_z = z_scores(churn_values)

    coverage_values = [signals.coverage for signals in modules.values() if signals.coverage is not None]
    coverage_mean = statistics.mean(coverage_values) if coverage_values else None
    coverage_stdev = statistics.pstdev(coverage_values) if coverage_values else None

    results: Dict[str, Dict[str, object]] = {}
    for idx, (module, signals) in enumerate(modules.items()):
        coverage_penalty = 0.0
        coverage_reason = ""
        if signals.coverage is None:
            coverage_penalty = 0.5
            coverage_reason = "coverage unknown"
        else:
            if coverage_stdev and coverage_stdev > 0:
                z = (signals.coverage - coverage_mean) / coverage_stdev
            else:
                z = 0.0
            coverage_penalty = max(0.0, -z)
            if coverage_penalty > 0.1:
                coverage_reason = f"low coverage ({signals.coverage:.1f}%)"

        score = 0.0
        reasons: List[str] = []

        c_z = complexity_z.get(idx, 0.0)
        score += c_z * weights.complexity_z
        if c_z > 1.0:
            reasons.append(f"High complexity (+{c_z:.1f}σ)")

        fi_z = fan_in_z.get(idx, 0.0)
        score += fi_z * weights.fan_in_z
        if fi_z > 1.0:
            reasons.append(f"High fan-in (+{fi_z:.1f}σ)")

        fo_z = fan_out_z.get(idx, 0.0)
        score += fo_z * weights.fan_out_z
        if fo_z > 1.0:
            reasons.append(f"High fan-out (+{fo_z:.1f}σ)")

        vio_z_val = violations_z.get(idx, 0.0)
        score += vio_z_val * weights.violations_z
        if vio_z_val > 1.0:
            reasons.append(f"Lint/security issues (+{vio_z_val:.1f}σ)")

        clones_z_val = clones_z.get(idx, 0.0)
        score += clones_z_val * weights.clones_z
        if clones_z_val > 1.0:
            reasons.append(f"Duplicate code (+{clones_z_val:.1f}σ)")

        churn_z_val = churn_z.get(idx, 0.0)
        score += churn_z_val * weights.churn_z
        if churn_z_val > 1.0:
            reasons.append(f"Recent churn (+{churn_z_val:.1f}σ)")

        if coverage_penalty:
            score += coverage_penalty * 0.2
            if coverage_reason:
                reasons.append(coverage_reason)

        if signals.owner_unknown:
            score += weights.owner_unknown
            reasons.append("Owner unknown")

        results[module] = {
            "risk": round(score, 4),
            "complexity": signals.complexity,
            "fan_in": signals.fan_in,
            "fan_out": signals.fan_out,
            "violations": signals.violations,
            "clones": signals.clones,
            "churn": signals.churn,
            "coverage": signals.coverage if signals.coverage is not None else "unknown",
            "owner_unknown": signals.owner_unknown,
            "reasons": reasons,
        }
    return results


def write_heatmap_csv(path: Path, data: Mapping[str, Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "module",
        "risk",
        "complexity",
        "fan_in",
        "fan_out",
        "violations",
        "clones",
        "churn",
        "coverage",
        "owner_unknown",
        "reasons",
    ]
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()
    for module, metrics in sorted(
        data.items(), key=lambda item: (-item[1]["risk"], item[0])
    ):
        writer.writerow(
            {
                "module": module,
                **{
                    key: metrics.get(key)
                    for key in fieldnames
                    if key not in {"module", "reasons"}
                },
                "reasons": "; ".join(metrics.get("reasons", [])),
            }
        )
    validated_path = validate_path(path, path.parent)
    safe_write_text(validated_path, buffer.getvalue(), path.parent)


def write_top20(path: Path, data: Mapping[str, Mapping[str, object]]) -> None:
    sorted_modules = sorted(data.items(), key=lambda item: (-item[1]["risk"], item[0]))[:20]
    lines = ["# Top 20 High-Risk Modules", ""]
    for module, metrics in sorted_modules:
        reasons = metrics.get("reasons") or []
        reason_text = "; ".join(reasons) if reasons else "No major signals (score mostly baseline)."
        lines.append(f"- **{module}** — risk {metrics['risk']:.3f}: {reason_text}")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute risk scores per module")
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--lint", type=Path)
    parser.add_argument("--coverage", type=Path)
    parser.add_argument("--clones", type=Path)
    parser.add_argument("--churn", type=Path)
    parser.add_argument("--owners", type=Path)
    parser.add_argument("--config", type=Path, default=Path("config/risk.yaml"))
    parser.add_argument("--out", type=Path, required=True, help="Output directory for risk results")
    args = parser.parse_args()

    base_dir = Path.cwd()
    try:
        metrics_path = validate_path(args.metrics, base_dir, must_exist=True)
        lint_path = validate_path(args.lint, base_dir, must_exist=True) if args.lint else None
        coverage_path = (
            validate_path(args.coverage, base_dir, must_exist=True)
            if args.coverage
            else None
        )
        clones_path = validate_path(args.clones, base_dir, must_exist=True) if args.clones else None
        churn_path = validate_path(args.churn, base_dir, must_exist=True) if args.churn else None
        owners_path = validate_path(args.owners, base_dir, must_exist=True) if args.owners else None
        config_path = validate_path(args.config, base_dir)
        output_dir = validate_path(args.out, base_dir)
    except (PathTraversalError, FileNotFoundError) as exc:
        raise SystemExit(f"Error: {exc}") from exc

    weights = load_risk_config(config_path)
    modules = load_metrics(metrics_path)

    lint_counts = load_lint_counts(lint_path) if lint_path else {}
    clone_counts = load_clones(clones_path) if clones_path else {}
    churn_counts = load_churn(churn_path) if churn_path else {}
    owners = load_owners(owners_path) if owners_path else {}

    file_to_module = map_files_to_modules(modules)

    for path_str, count in lint_counts.items():
        module = resolve_module_for_path(file_to_module, path_str)
        if module:
            modules[module].violations += count
    for path_str, count in clone_counts.items():
        module = resolve_module_for_path(file_to_module, path_str)
        if module:
            modules[module].clones += count
    for path_str, value in churn_counts.items():
        module = resolve_module_for_path(file_to_module, path_str)
        if module:
            modules[module].churn += value

    if coverage_path and coverage_path.exists():
        coverage_map = load_coverage_signal(coverage_path)
    else:
        coverage_map = {}
    coverage_by_module: Dict[str, float] = {}
    for path_key, percent in coverage_map.items():
        module_name = path_to_module(path_key)
        if module_name and module_name in modules:
            coverage_by_module[module_name] = percent
    for module, signals in modules.items():
        signals.coverage = coverage_by_module.get(module)
        module_path = module.replace(".", "/") + ".py"
        if owners and not match_owner(module_path, owners):
            signals.owner_unknown = True

    risk_data = compute_risk(modules, weights)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_heatmap_csv(output_dir / "risk_heatmap.csv", risk_data)
    write_top20(output_dir / "top20.md", risk_data)


def load_coverage_signal(path: Path) -> Dict[str, float]:
    coverage_map: Dict[str, float] = {}
    if path.suffix == ".xml":
        coverage_map = load_coverage_map(path, Path.cwd())
    elif path.suffix == ".log":
        # Parse coverage.log fallback (expects command failure details)
        coverage_map = {}
    else:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            data = None
        if isinstance(data, dict):
            for module, percent in data.items():
                try:
                    coverage_map[module] = float(percent)
                except ValueError:
                    continue
    return coverage_map


if __name__ == "__main__":  # pragma: no cover
    main()
