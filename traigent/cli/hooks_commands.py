"""CLI commands for Traigent hooks management.

Provides commands to install, validate, and manage Git hooks
for agent configuration validation.
"""

# Traceability: CONC-Layer-API CONC-Quality-Usability CONC-Quality-Maintainability FUNC-API-ENTRY REQ-API-001

from __future__ import annotations

from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

console = Console()


@click.group()
def hooks() -> None:
    """Manage Traigent Git hooks for agent validation.

    Traigent hooks validate agent configurations against constraints
    defined in traigent.yml before allowing Git pushes.

    Examples:
        traigent hooks install         # Install pre-push hooks
        traigent hooks status          # Check hook status
        traigent hooks validate        # Validate current directory
        traigent hooks init            # Create traigent.yml
    """
    pass


@hooks.command()
@click.option("--force", "-f", is_flag=True, help="Overwrite existing hooks")
@click.option("--path", "-p", type=click.Path(exists=True), help="Git repository path")
def install(force: bool, path: str | None) -> None:
    """Install Traigent Git hooks.

    Installs pre-push and pre-commit hooks that validate agent
    configurations against traigent.yml constraints.
    """
    from traigent.hooks.installer import HooksInstaller, find_git_root

    repo_path = Path(path) if path else find_git_root()

    if repo_path is None:
        console.print("[red]Error: Not in a Git repository[/red]")
        console.print("Run this command from within a Git repository")
        raise SystemExit(1)

    console.print("\n[bold blue]Installing Traigent Git hooks[/bold blue]")
    console.print(f"Repository: {repo_path}\n")

    installer = HooksInstaller(repo_path)

    try:
        results = installer.install(force=force)

        # Display results
        for hook_name, success in results.items():
            if success:
                console.print(f"  [green]Installed[/green]: .git/hooks/{hook_name}")
            else:
                console.print(
                    f"  [yellow]Skipped[/yellow]: .git/hooks/{hook_name} (exists, use --force)"
                )

        if all(results.values()):
            console.print("\n[green]Hooks installed successfully![/green]")
        else:
            console.print("\n[yellow]Some hooks were skipped[/yellow]")

    except Exception as e:
        console.print(f"[red]Error installing hooks: {e}[/red]")
        raise SystemExit(1) from e


@hooks.command()
@click.option("--path", "-p", type=click.Path(exists=True), help="Git repository path")
def uninstall(path: str | None) -> None:
    """Uninstall Traigent Git hooks.

    Removes Traigent-installed hooks and restores any backed-up hooks.
    """
    from traigent.hooks.installer import HooksInstaller, find_git_root

    repo_path = Path(path) if path else find_git_root()

    if repo_path is None:
        console.print("[red]Error: Not in a Git repository[/red]")
        raise SystemExit(1)

    console.print("\n[bold blue]Uninstalling Traigent Git hooks[/bold blue]")

    installer = HooksInstaller(repo_path)

    try:
        results = installer.uninstall()

        for hook_name, success in results.items():
            if success:
                console.print(f"  [green]Removed[/green]: .git/hooks/{hook_name}")
            else:
                console.print(
                    f"  [yellow]Skipped[/yellow]: .git/hooks/{hook_name} (not a Traigent hook)"
                )

        console.print("\n[green]Hooks uninstalled![/green]")

    except Exception as e:
        console.print(f"[red]Error uninstalling hooks: {e}[/red]")
        raise SystemExit(1) from e


@hooks.command()
@click.option("--path", "-p", type=click.Path(exists=True), help="Git repository path")
def status(path: str | None) -> None:
    """Show status of Traigent Git hooks."""
    from traigent.hooks.config import find_config_file
    from traigent.hooks.installer import HooksInstaller, find_git_root

    repo_path = Path(path) if path else find_git_root()

    console.print("\n[bold blue]Traigent Hooks Status[/bold blue]\n")

    # Git repository status
    if repo_path is None:
        console.print("[yellow]Git Repository:[/yellow] Not found")
        return
    else:
        console.print(f"[green]Git Repository:[/green] {repo_path}")

    # Config file status
    config_path = find_config_file(repo_path)
    if config_path:
        console.print(f"[green]Configuration:[/green] {config_path}")
    else:
        console.print("[yellow]Configuration:[/yellow] Not found (using defaults)")

    # Hooks status
    console.print("\n[bold]Installed Hooks:[/bold]")

    installer = HooksInstaller(repo_path)
    hook_status = installer.status()

    table = Table(show_header=True, header_style="bold")
    table.add_column("Hook")
    table.add_column("Status")

    for hook_name, status_str in hook_status.items():
        if status_str == "installed (traigent)":
            status_display = "[green]Installed (Traigent)[/green]"
        elif status_str == "installed (other)":
            status_display = "[yellow]Installed (Other)[/yellow]"
        else:
            status_display = "[dim]Not installed[/dim]"

        table.add_row(hook_name, status_display)

    console.print(table)


@hooks.command()
@click.argument("target", default=".", type=click.Path(exists=True))
@click.option(
    "--config", "-c", type=click.Path(exists=True), help="Path to traigent.yml"
)
@click.option("--exit-code", is_flag=True, help="Exit with non-zero code on failure")
@click.option("--verbose", "-v", is_flag=True, help="Show detailed output")
def validate(target: str, config: str | None, exit_code: bool, verbose: bool) -> None:
    """Validate agent configurations against constraints.

    Scans TARGET (file or directory) for @traigent.optimize decorated
    functions and validates them against constraints in traigent.yml.

    Examples:
        traigent hooks validate              # Validate current directory
        traigent hooks validate my_agent.py  # Validate specific file
        traigent hooks validate --exit-code  # For use in CI/CD
    """
    from traigent.hooks.config import load_hooks_config
    from traigent.hooks.validator import AgentValidator

    target_path = Path(target)

    console.print("\n[bold blue]Traigent Agent Validation[/bold blue]")
    console.print(f"Target: {target_path}\n")

    # Load configuration
    try:
        hooks_config = load_hooks_config(config)
        if verbose:
            console.print(
                f"[dim]Configuration loaded from: {config or 'auto-detected'}[/dim]\n"
            )
    except Exception as e:
        console.print(f"[red]Error loading configuration: {e}[/red]")
        if exit_code:
            raise SystemExit(1) from e
        return

    # Check if validation is enabled
    if not hooks_config.enabled:
        console.print("[yellow]Traigent hooks are disabled in configuration[/yellow]")
        return

    # Run validation
    validator = AgentValidator(hooks_config)

    if target_path.is_file():
        results = validator.validate_file(target_path)
    else:
        results = validator.validate_directory(target_path)

    # No functions found
    if not results:
        console.print(
            "[yellow]No @traigent.optimize decorated functions found[/yellow]"
        )
        return

    # Display results
    console.print(f"Found {len(results)} decorated function(s)\n")

    has_failures = False

    for result in results:
        # Function header
        if result.is_valid:
            console.print(f"[green]✓[/green] [bold]{result.function_name}[/bold]")
        else:
            console.print(f"[red]✗[/red] [bold]{result.function_name}[/bold]")
            has_failures = True

        # Models
        if result.models_found:
            models_str = ", ".join(result.models_found)
            console.print(f"  Models: {models_str}")

        # Cost estimate
        if result.estimated_cost_per_query is not None:
            console.print(f"  Est. cost: ${result.estimated_cost_per_query:.4f}/query")

        # Issues
        for issue in result.issues:
            if issue.severity == "error":
                console.print(f"  [red]ERROR[/red]: {issue.message}")
            else:
                console.print(f"  [yellow]WARNING[/yellow]: {issue.message}")

            if verbose and issue.suggestion:
                console.print(f"    [dim]Suggestion: {issue.suggestion}[/dim]")

        # Warnings
        for warning in result.warnings:
            console.print(f"  [yellow]WARNING[/yellow]: {warning.message}")

        console.print()

    # Summary
    passed = sum(1 for r in results if r.is_valid)
    failed = len(results) - passed

    console.print("[bold]Summary:[/bold]")
    console.print(f"  [green]Passed[/green]: {passed}")
    console.print(f"  [red]Failed[/red]: {failed}")

    if has_failures:
        console.print("\n[red]Validation failed![/red]")
        if exit_code:
            raise SystemExit(1)
    else:
        console.print("\n[green]All validations passed![/green]")


@hooks.command()
@click.option("--quick", is_flag=True, help="Quick syntax check only")
def check(quick: bool) -> None:
    """Quick configuration check (for pre-commit hook).

    Performs a lightweight check suitable for pre-commit hooks.
    """
    from traigent.hooks.config import find_config_file, load_hooks_config

    # Just verify config can be loaded
    config_path = find_config_file()

    if config_path:
        try:
            load_hooks_config(config_path)
            console.print("[green]Configuration valid[/green]")
        except Exception as e:
            console.print(f"[red]Configuration error: {e}[/red]")
            raise SystemExit(1) from e
    else:
        if not quick:
            console.print("[yellow]No traigent.yml found (using defaults)[/yellow]")


@hooks.command()
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    default="traigent.yml",
    help="Output path for configuration file",
)
@click.option("--force", "-f", is_flag=True, help="Overwrite existing file")
def init(output: str, force: bool) -> None:
    """Create a default traigent.yml configuration file.

    Generates a traigent.yml with common constraints and sensible defaults.
    """
    from traigent.hooks.config import create_default_config

    output_path = Path(output)

    if output_path.exists() and not force:
        console.print(f"[yellow]File already exists: {output_path}[/yellow]")
        console.print("Use --force to overwrite")
        raise SystemExit(1)

    try:
        created_path = create_default_config(output_path)
        console.print(f"[green]Created configuration file: {created_path}[/green]")
        console.print("\nEdit this file to customize your agent constraints.")
        console.print("Then run: traigent hooks install")

    except Exception as e:
        console.print(f"[red]Error creating configuration: {e}[/red]")
        raise SystemExit(1) from e


@hooks.command()
@click.argument("module_path", type=click.Path(exists=True))
@click.option("--function", "-f", help="Specific function to show")
def show(module_path: str, function: str | None) -> None:
    """Show agent configuration details from a Python file.

    Displays the configuration space, objectives, and constraints
    for @traigent.optimize decorated functions.
    """
    from traigent.cli.function_discovery import discover_optimized_functions

    try:
        functions = discover_optimized_functions(
            module_path, [function] if function else None
        )
    except Exception as e:
        console.print(f"[red]Error loading module: {e}[/red]")
        raise SystemExit(1) from e

    if not functions:
        console.print(
            "[yellow]No @traigent.optimize decorated functions found[/yellow]"
        )
        return

    console.print(f"\n[bold blue]Agent Configurations in {module_path}[/bold blue]\n")

    for func_info in functions:
        console.print(f"[bold cyan]{func_info.name}[/bold cyan]")

        # Objectives
        objectives_str = ", ".join(func_info.objectives)
        console.print(f"  Objectives: {objectives_str}")

        # Dataset
        if func_info.eval_dataset:
            console.print(f"  Dataset: {func_info.eval_dataset}")

        # Configuration space
        config_space = func_info.decorator_config.get("configuration_space", {})
        if config_space:
            console.print("  Configuration Space:")
            for param, values in config_space.items():
                if isinstance(values, list):
                    console.print(f"    {param}: {values}")
                elif isinstance(values, tuple):
                    console.print(f"    {param}: [{values[0]} - {values[1]}]")
                else:
                    console.print(f"    {param}: {values}")

        console.print()
