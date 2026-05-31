"""Assertions for the public optimizer documentation surface."""

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]

PUBLIC_OPTIMIZER_DOCS = [
    "README.md",
    "docs/README.md",
    "docs/user-guide/README.md",
    "docs/getting-started/installation.md",
    "docs/examples/API_PATTERNS.md",
    "docs/api-reference/complete-function-specification.md",
    "docs/api-reference/telemetry.md",
    "docs/feature_matrices/optimizers.yml",
    "docs/requirements.yml",
    "docs/traceability/requirements.yml",
    "examples/catalog.yaml",
]

STALE_OPTUNA_SURFACE_PATTERNS = [
    "optuna_integration",
    "optuna integration",
    "traigent_optuna_enabled",
    'algorithm="optuna"',
    "[bayesian]",
    "optunametricsemitter",
    "optunaadapter",
    "optunacoordinator",
    "optunatpeoptimizer",
]


def _read(relative_path: str) -> str:
    return (REPO_ROOT / relative_path).read_text(encoding="utf-8")


def test_public_optimizer_docs_do_not_advertise_local_optuna_surface():
    """Public docs should not point users at removed Optuna install/API paths."""
    for relative_path in PUBLIC_OPTIMIZER_DOCS:
        content = _read(relative_path).lower()
        for pattern in STALE_OPTUNA_SURFACE_PATTERNS:
            assert pattern not in content, f"{relative_path} still contains {pattern!r}"


def test_public_examples_do_not_use_optuna_filenames():
    """User-facing example filenames should use backend-smart strategy names."""
    example_paths = [
        path.relative_to(REPO_ROOT).as_posix()
        for path in (REPO_ROOT / "examples").rglob("*")
        if path.is_file()
    ]
    assert not [path for path in example_paths if "optuna" in path.lower()]


def test_public_optimizer_import_surface_has_no_optuna_imports():
    """Local public optimizer modules must not import the removed SDK dependency."""
    public_modules = [
        "traigent/optimizers/__init__.py",
        "traigent/optimizers/base.py",
        "traigent/optimizers/grid.py",
        "traigent/optimizers/random.py",
        "traigent/optimizers/registry.py",
    ]

    for relative_path in public_modules:
        source = _read(relative_path).lower()
        assert "import optuna" not in source
        assert "from optuna" not in source


def test_sdk_package_does_not_ship_legacy_optuna_modules():
    """Removed local smart optimizer modules must not be included in the SDK tree."""
    forbidden_paths = [
        "traigent/config/seamless_optuna_adapter.py",
        "traigent/optimizers/benchmarking.py",
        "traigent/optimizers/optuna_adapter.py",
        "traigent/optimizers/optuna_checkpoint.py",
        "traigent/optimizers/optuna_coordinator.py",
        "traigent/optimizers/optuna_optimizer.py",
        "traigent/optimizers/optuna_utils.py",
        "traigent/optimizers/pruners.py",
        "traigent/telemetry/optuna_metrics.py",
    ]

    assert not [
        relative_path
        for relative_path in forbidden_paths
        if (REPO_ROOT / relative_path).exists()
    ]


def test_public_package_import_does_not_load_optuna():
    """A clean public import should not touch the removed Optuna dependency."""
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import traigent; print('optuna' in sys.modules)",
        ],
        capture_output=True,
        check=False,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip().splitlines()[-1] == "False"


def test_smart_optimization_guide_documents_current_strategy_boundary():
    guide = _read("docs/user-guide/smart_optimization.md")

    assert "Local optimization: `grid` and `random`." in guide
    assert (
        "Backend-routed smart optimization: `bayesian`, `tpe`, `hyperband`, "
        "and `frontier_scout`."
    ) in guide
    assert "does not expose Optuna as a Python dependency" in guide
    assert 'algorithm="optuna"' in guide
    assert 'pip install "traigent[bayesian]"' in guide
