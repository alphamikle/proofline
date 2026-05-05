from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, Optional

# macOS ML stacks can load libomp through more than one dependency
# (for example faiss plus torch/sentence-transformers). Without this,
# the OpenMP runtime can abort the process before Python can recover.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# Keep native math libraries from multiplying threads inside each Proofline
# worker. Users can override any of these before launching the CLI.
for _thread_env_var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_thread_env_var, "1")
del _thread_env_var

import typer
import yaml

from proofline.config import (
    CONFIG_ENV_VAR,
    DEFAULT_CONFIG,
    config_followup_warnings,
    config_shape_diff,
    default_config,
    ensure_dirs,
    load_config,
    upgrade_config_file,
)
from proofline.logging_utils import console, setup_logging
from proofline.repair import run_repair
from proofline.uninstall import run_uninstall, uninstall_plan
from proofline.upgrade import DEFAULT_REF, DEFAULT_REPO, UpgradeError, run_upgrade
from proofline.utils import json_dumps
from proofline.version import update_check, version_info

app = typer.Typer(help="Proofline CLI")


@app.callback()
def main(
    ctx: typer.Context,
    no_update_check: bool = typer.Option(False, "--no-update-check", help="Skip the Proofline update check."),
) -> None:
    if ctx.resilient_parsing or no_update_check:
        return
    if ctx.invoked_subcommand in {"upgrade", None}:
        return
    maybe_show_update_notice()


def config_path(config: Optional[str]) -> str:
    return config or os.getenv(CONFIG_ENV_VAR) or DEFAULT_CONFIG


def maybe_show_update_notice() -> None:
    if os.getenv("PROOFLINE_NO_UPDATE_CHECK") or not sys.stderr.isatty():
        return
    info = update_check()
    if not info.get("available") or not info.get("update_available"):
        return
    current = info.get("current_version") or "unknown"
    latest = info.get("latest_version") or "unknown"
    typer.echo(
        "\nProofline: a new version is available "
        f"({current} -> {latest}).",
        err=True,
    )
    typer.echo(f"To download and install it, run: {info.get('command')}\n", err=True)


def ensure_indexing_config_current(config: Optional[str]) -> None:
    path = Path(config_path(config))
    if not path.exists():
        return
    diff = config_shape_diff(path)
    if not diff["needs_migration"]:
        return
    missing = diff["missing_paths"]
    shown = ", ".join(missing[:10]) if missing else "config_version"
    if len(missing) > 10:
        shown += f" (+{len(missing) - 10} more)"
    typer.echo(
        "\nYou are using a newer Proofline version, but this project's config is outdated.",
        err=True,
    )
    typer.echo(f"Config: {path}", err=True)
    typer.echo(
        f"Config version: {diff.get('current_version') or 'unknown'} -> {diff['target_version']}",
        err=True,
    )
    typer.echo(f"Missing keys: {shown}", err=True)
    if not sys.stdin.isatty():
        typer.echo(
            f"Run this interactively to migrate it: pfl init --migrate --config {path}",
            err=True,
        )
        raise typer.Exit(1)
    if not typer.confirm("Update the config now? The old file will be saved next to it.", default=True):
        typer.echo(f"Stopped. You can migrate later with: pfl init --migrate --config {path}", err=True)
        raise typer.Exit(1)
    migrated, backup, _ = upgrade_config_file(path, use_agent=True, quiet=True)
    typer.echo(f"Previous config saved as: {backup}", err=True)
    typer.echo(f"New config written to: {path}\n", err=True)
    for warning in config_followup_warnings(migrated):
        typer.echo(f"Needs attention: {warning}", err=True)


def load_runtime(config: Optional[str]) -> tuple[dict[str, Any], KB]:
    from proofline.storage import KB

    cfg = load_config(config_path(config))
    ensure_dirs(cfg)
    return cfg, KB(cfg["storage"]["duckdb_path"])


def run_named_stage(name: str, config: Optional[str]) -> None:
    from proofline.pipeline.runner import STAGES, resolve_stage, run_stage

    ensure_indexing_config_current(config)
    setup_logging()
    cfg = load_config(config_path(config))
    ensure_dirs(cfg)
    stage_name = resolve_stage(name)
    if stage_name not in STAGES:
        raise typer.BadParameter(f"Unknown stage: {name}. Known: {', '.join(STAGES)}")
    run_stage(stage_name, cfg, STAGES[stage_name])


def run_stages(names: Iterable[str], config: Optional[str]) -> None:
    for name in names:
        run_named_stage(name, config)


def maybe_run_sync_stage(
    name: str,
    config: Optional[str],
    dry_run: bool,
    source: Optional[str] = None,
    skip_disabled: bool = False,
) -> None:
    if source and skip_disabled and not source_enabled(load_config(config_path(config)), source):
        log_disabled_source(source, config)
        return
    if dry_run:
        console.print(f"Would sync {name}")
        return
    run_named_stage(name, config)


def source_enabled(cfg: dict[str, Any], source: str) -> bool:
    return bool(cfg.get(source, {}).get("enabled", False))


def log_disabled_source(source: str, config: Optional[str]) -> None:
    console.print(f"[yellow]Source {source} is disabled in {config_path(config)}; skipping.[/yellow]")


def maybe_run_script(
    script_name: str,
    source: str,
    config: Optional[str],
    dry_run: bool,
    skip_disabled: bool,
) -> None:
    if skip_disabled and not source_enabled(load_config(config_path(config)), source):
        log_disabled_source(source, config)
        return
    run_script(script_name, config, dry_run)


def run_script(script_name: str, config: Optional[str], dry_run: bool = False) -> None:
    script = Path(__file__).resolve().parents[1] / "scripts" / script_name
    cmd = [sys.executable, str(script), "--config", config_path(config)]
    if dry_run:
        cmd.append("--dry-run")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise typer.Exit(result.returncode)


@app.command()
def init(
    config: Optional[str] = typer.Option(None, "--config", "-c"),
    force: bool = typer.Option(False, "--force"),
    non_interactive: bool = typer.Option(False, "--non-interactive", "--yes", help="Write defaults without the interactive survey."),
    migrate: bool = typer.Option(False, "--migrate", help="Update an existing config in place instead of failing when it exists."),
) -> None:
    """Create or update a proofline.yaml config and local working directories."""
    target = Path(config_path(config))
    if target.exists() and not force:
        if migrate:
            cfg, backup, added = upgrade_config_file(target, use_agent=True, quiet=True)
            ensure_dirs(load_config(target))
            if added:
                console.print(f"[green]Updated[/green] {target} with {len(added)} missing config keys.")
                if backup:
                    console.print(f"Previous config saved as {backup}")
            else:
                console.print(f"[green]Already up to date[/green] {target}")
            warnings = config_followup_warnings(cfg)
            for warning in warnings:
                console.print(f"[yellow]Needs attention:[/yellow] {warning}")
            return
        raise typer.BadParameter(f"{target} already exists. Pass --migrate to update it or --force to overwrite it.")
    cfg = default_config()
    interactive = (not non_interactive) and sys.stdin.isatty()
    if interactive:
        cfg = survey_config(cfg, target)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=False)
    cfg = load_config(target)
    ensure_dirs(cfg)
    console.print(f"Created {target}")
    warnings = config_followup_warnings(cfg)
    for warning in warnings:
        console.print(f"[yellow]Needs attention:[/yellow] {warning}")


def survey_config(cfg: dict[str, Any], target: Path) -> dict[str, Any]:
    console.print("[bold]Proofline config survey[/bold]")
    console.print(f"Target: {target}")

    console.print("\n[bold]Workspace[/bold]")
    cfg["workspace"] = typer.prompt("Workspace directory", default=str(cfg.get("workspace") or "./data"))
    cfg.setdefault("repos", {})
    cfg["repos"]["root"] = typer.prompt("Repositories directory", default=str(cfg["repos"].get("root") or "./repos"))
    cfg["repos"]["update_existing"] = typer.confirm("Fetch/update existing repos during sync?", default=bool(cfg["repos"].get("update_existing", True)))
    max_file_mb = typer.prompt("Max file size to index, MB", default=str(cfg["repos"].get("max_file_mb", 5)))
    cfg["repos"]["max_file_mb"] = float(max_file_mb)

    root = Path(cfg["workspace"])
    cfg.setdefault("storage", {})
    cfg["storage"]["duckdb_path"] = typer.prompt("DuckDB path", default=str(root / "kb.duckdb"))
    cfg["storage"]["sqlite_fts_path"] = typer.prompt("SQLite FTS path", default=str(root / "indexes" / "code_fts.sqlite"))
    cfg["storage"]["vector_index_path"] = typer.prompt("Vector index path", default=str(root / "indexes" / "code_vectors.faiss"))
    cfg["storage"]["vector_meta_path"] = typer.prompt("Vector metadata path", default=str(root / "indexes" / "code_vectors_meta.parquet"))

    console.print("\n[bold]Git History[/bold]")
    cfg.setdefault("git_history", {})
    gh = cfg["git_history"]
    gh["enabled"] = typer.confirm("Index Git history as Change Graph?", default=bool(gh.get("enabled", True)))
    gh["patch_hunks"] = typer.confirm("Index patch hunks?", default=bool(gh.get("patch_hunks", True)))
    gh["current_blame"] = typer.confirm("Build current blame index?", default=bool(gh.get("current_blame", True)))
    full_history = typer.confirm("Index full Git history? Choose no to set a commit limit.", default=gh.get("max_commits_per_repo") in (None, ""))
    gh["max_commits_per_repo"] = None if full_history else int(typer.prompt("Max commits per repo", default="5000"))
    gh["cochange_window_days"] = int(typer.prompt("Co-change window, days", default=str(gh.get("cochange_window_days") or 730)))

    console.print("\n[bold]Sources[/bold]")
    configure_source(cfg, "datadog", "Enable Datadog runtime ingestion?")
    configure_source(cfg, "bigquery", "Enable BigQuery metadata ingestion?")
    configure_source(cfg, "confluence", "Enable Confluence ingestion?")
    configure_source(cfg, "jira", "Enable Jira ingestion?")
    if cfg.get("datadog", {}).get("enabled"):
        cfg["datadog"]["site"] = typer.prompt("Datadog site", default=str(cfg["datadog"].get("site") or "datadoghq.com"))
    for key, label in [("confluence", "Confluence base URL"), ("jira", "Jira base URL")]:
        if cfg.get(key, {}).get("enabled"):
            default_url = str(cfg[key].get("base_url") or f"${{{key.upper()}_BASE_URL}}")
            cfg[key]["base_url"] = typer.prompt(label, default=default_url)

    console.print("\n[bold]Indexes[/bold]")
    cfg.setdefault("indexing", {}).setdefault("embeddings", {})
    cfg["indexing"]["lexical_fts"] = typer.confirm("Build lexical full-text search index?", default=bool(cfg["indexing"].get("lexical_fts", True)))
    emb = cfg["indexing"]["embeddings"]
    emb["enabled"] = typer.confirm("Build vector embeddings?", default=bool(emb.get("enabled", True)))
    if emb["enabled"]:
        emb["provider"] = typer.prompt("Embedding provider (sentence_transformers, openai, openai_compatible, cli)", default=str(emb.get("provider") or "sentence_transformers"))
        emb["model_name"] = typer.prompt("Embedding model", default=str(emb.get("model_name") or "Qwen/Qwen3-Embedding-0.6B"))
        if emb["provider"] == "sentence_transformers":
            emb["device"] = typer.prompt("Embedding device", default=str(emb.get("device") or "auto"))
        elif emb["provider"] in {"openai", "openai_compatible"}:
            emb["base_url"] = typer.prompt("Embedding base URL", default=str(emb.get("base_url") or ("https://api.openai.com/v1" if emb["provider"] == "openai" else "${OPENAI_BASE_URL}")))
            emb["api_key_env"] = typer.prompt("Embedding API key env", default=str(emb.get("api_key_env") or "OPENAI_API_KEY"))
        elif emb["provider"] == "cli":
            emb["command"] = typer.prompt("Embedding command", default=str(emb.get("command") or ""))

    console.print("\n[bold]Graph & Agent[/bold]")
    cfg.setdefault("graph_backend", {})
    cfg["graph_backend"]["enabled"] = typer.confirm("Enable external Neo4j graph backend?", default=bool(cfg["graph_backend"].get("enabled", True)))
    if cfg["graph_backend"]["enabled"]:
        cfg["graph_backend"]["uri"] = typer.prompt("Neo4j URI", default=str(cfg["graph_backend"].get("uri") or "bolt://localhost:7687"))
        cfg["graph_backend"]["username"] = typer.prompt("Neo4j username", default=str(cfg["graph_backend"].get("username") or "neo4j"))
    cfg.setdefault("agent", {})
    cfg["agent"]["provider"] = typer.prompt("Agent provider (none, cli, openai, openai_compatible, anthropic, anthropic_compatible)", default=str(cfg["agent"].get("provider") or "none"))
    if cfg["agent"]["provider"] != "none":
        cfg["agent"]["model"] = typer.prompt("Agent model", default=str(cfg["agent"].get("model") or ""))
    return cfg


def configure_source(cfg: dict[str, Any], key: str, prompt: str) -> None:
    cfg.setdefault(key, {})
    cfg[key]["enabled"] = typer.confirm(prompt, default=bool(cfg[key].get("enabled", False)))


@app.command()
def doctor(
    config: Optional[str] = typer.Option(None, "--config", "-c"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Check local setup and configured integrations."""
    checks: list[dict[str, Any]] = []
    path = Path(config_path(config))
    checks.append({"name": "config", "ok": path.exists(), "details": str(path)})
    try:
        cfg = load_config(path)
        ensure_dirs(cfg)
        checks.append({"name": "workspace", "ok": Path(cfg["workspace"]).exists(), "details": cfg["workspace"]})
        checks.append({"name": "repos", "ok": Path(cfg.get("repos", {}).get("root", "./repos")).exists(), "details": cfg.get("repos", {}).get("root", "./repos")})
        checks.append({"name": "datadog", "ok": bool(os.getenv("DD_API_KEY") and os.getenv("DD_APP_KEY")), "details": "DD_API_KEY/DD_APP_KEY"})
        checks.append({"name": "bigquery", "ok": bool(os.getenv("GOOGLE_APPLICATION_CREDENTIALS") or os.getenv("GOOGLE_CLOUD_PROJECT")), "details": "GOOGLE_APPLICATION_CREDENTIALS or GOOGLE_CLOUD_PROJECT"})
        checks.append({"name": "atlassian", "ok": bool(os.getenv("ATLASSIAN_BEARER_TOKEN") or (os.getenv("ATLASSIAN_EMAIL") and os.getenv("ATLASSIAN_API_TOKEN"))), "details": "ATLASSIAN_* credentials"})
        backend = cfg.get("graph_backend", {})
        if backend.get("enabled", False):
            checks.append({"name": "graph_backend", "ok": backend.get("provider", "neo4j") == "neo4j", "details": backend.get("provider", "neo4j")})
    except Exception as exc:
        checks.append({"name": "load_config", "ok": False, "details": str(exc)})
    if json_output:
        typer.echo(json_dumps({"checks": checks}))
        return
    for check in checks:
        marker = "ok" if check["ok"] else "missing"
        color = "green" if check["ok"] else "yellow"
        console.print(f"[{color}]{marker:7}[/{color}] {check['name']}: {check['details']}")


@app.command()
def repair(
    config: Optional[str] = typer.Option(None, "--config", "-c"),
    bin_dir: Optional[str] = typer.Option(None, "--bin-dir", help="Directory for proofline/pfl links. Default: ~/.local/bin or PROOFLINE_BIN_DIR."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be repaired without changing files."),
    skip_python_deps: bool = typer.Option(False, "--skip-python-deps", help="Skip reinstalling Proofline Python dependencies."),
    skip_cgc: bool = typer.Option(False, "--skip-cgc", help="Skip CGC, SCIP, Docker, and Neo4j repair."),
    skip_bin_links: bool = typer.Option(False, "--skip-bin-links", help="Skip relinking proofline/pfl into the bin directory."),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Repair local dependencies, config directories, CGC, and local Neo4j Docker runtime."""
    steps = run_repair(
        config_path=config_path(config),
        bin_dir=bin_dir,
        dry_run=dry_run,
        skip_python_deps=skip_python_deps,
        skip_cgc=skip_cgc,
        skip_bin_links=skip_bin_links,
    )
    if json_output:
        typer.echo(json_dumps({"steps": steps, "ok": all(step["ok"] for step in steps)}))
        if not all(step["ok"] for step in steps):
            raise typer.Exit(1)
        return
    for step in steps:
        marker = "ok" if step["ok"] else "failed"
        color = "green" if step["ok"] else "red"
        details = f": {step['details']}" if step.get("details") else ""
        console.print(f"[{color}]{marker:7}[/{color}] {step['name']}: {step['action']}{details}")
    if not all(step["ok"] for step in steps):
        raise typer.Exit(1)


@app.command()
def version(
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Show Proofline CLI version and install details."""
    info = version_info()
    if json_output:
        typer.echo(json_dumps(info))
        return
    console.print(f"Proofline {info['version']}")
    console.print(f"Package: {info['package_root']}")
    console.print(f"Python:  {info['python']}")
    git = info.get("git") or {}
    if git.get("available"):
        dirty = " dirty" if git.get("dirty") else ""
        console.print(f"Git:     {git.get('branch') or '?'}@{git.get('commit') or '?'}{dirty}")
        if git.get("remote"):
            console.print(f"Remote:  {git.get('remote')}")
    else:
        console.print("Git:     unavailable")


@app.command()
def upgrade(
    repo: Optional[str] = typer.Option(None, "--repo", help=f"Git repository to install from. Default: {DEFAULT_REPO} or PROOFLINE_REPO."),
    ref: Optional[str] = typer.Option(None, "--ref", help=f"Git branch/tag/ref to install. Default: {DEFAULT_REF} or PROOFLINE_REF."),
    install_dir: Optional[str] = typer.Option(None, "--dir", help="Proofline install directory. Default: current package root or PROOFLINE_DIR."),
    bin_dir: Optional[str] = typer.Option(None, "--bin-dir", help="Directory for proofline/pfl shims. Default: ~/.local/bin or PROOFLINE_BIN_DIR."),
    source_dir: Optional[str] = typer.Option(None, "--source-dir", help="Upgrade from a local checkout instead of remote git."),
    force: bool = typer.Option(False, "--force", help="Allow upgrading a dirty git checkout or skip non-git code backup."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be done without changing files."),
    skip_deps: bool = typer.Option(False, "--skip-deps", help="Skip pip dependency/package reinstall."),
) -> None:
    """Upgrade the Proofline CLI in place."""
    try:
        run_upgrade(
            repo=repo,
            ref=ref,
            install_dir=install_dir,
            bin_dir=bin_dir,
            source_dir=source_dir,
            force=force,
            dry_run=dry_run,
            skip_deps=skip_deps,
        )
    except UpgradeError as exc:
        print(f"Upgrade failed: {exc}", file=sys.stderr)
        raise typer.Exit(1)


@app.command()
def uninstall(
    install_dir: Optional[str] = typer.Option(None, "--dir", help="Proofline install directory. Default: current package root or PROOFLINE_DIR."),
    bin_dir: Optional[str] = typer.Option(None, "--bin-dir", help="Directory containing proofline/pfl shims. Default: ~/.local/bin or PROOFLINE_BIN_DIR."),
    include_cgc: bool = typer.Option(False, "--include-cgc", help="Also remove CGC/SCIP binaries and CGC venv/node tooling. CGC data and Docker volumes are preserved."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be removed without deleting anything."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Confirm uninstall without prompting."),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Uninstall Proofline CLI/runtime while preserving indexed data and configs."""
    plan = uninstall_plan(install_dir=install_dir, bin_dir=bin_dir, include_cgc=include_cgc)
    if json_output:
        typer.echo(json_dumps(plan))
        if dry_run:
            return
    else:
        console.print("[bold]Proofline uninstall plan[/bold]")
        console.print("\nWill remove:")
        for path in plan["remove"] or ["<nothing>"]:
            console.print(f"  {path}")
        console.print("\nWill preserve:")
        for path in plan["preserve"] or ["<nothing>"]:
            console.print(f"  {path}")
    if dry_run:
        return
    if not yes:
        if not sys.stdin.isatty():
            raise typer.BadParameter("Refusing to uninstall without --yes in a non-interactive shell.")
        if not typer.confirm("Remove the listed Proofline files?", default=False):
            console.print("Uninstall cancelled.")
            return
    result = run_uninstall(install_dir=install_dir, bin_dir=bin_dir, include_cgc=include_cgc, dry_run=False)
    if not json_output:
        console.print(f"[green]Removed {len(result['remove'])} paths.[/green]")
        console.print("Preserved data/config paths listed above.")


@app.command()
def run(
    config: Optional[str] = typer.Option(None, "--config", "-c"),
    from_stage: Optional[str] = typer.Option(None, "--from"),
    to_stage: Optional[str] = typer.Option(None, "--to"),
) -> None:
    """Run the main Proofline pipeline."""
    from proofline.pipeline.runner import run_order

    ensure_indexing_config_current(config)
    run_order(config_path(config), from_stage, to_stage)


@app.command("mcp")
def mcp_server(
    config: Optional[str] = typer.Option(None, "--config", "-c"),
    transport: str = typer.Option("stdio", "--transport"),
) -> None:
    """Start a read-only MCP server over the local Proofline KB."""
    allowed = {"stdio", "streamable-http", "sse"}
    if transport not in allowed:
        raise typer.BadParameter(f"Unsupported transport: {transport}. Expected one of: {', '.join(sorted(allowed))}")
    from proofline.mcp_server import serve

    serve(config_path(config), transport)


@app.command()
def stage(
    name: str = typer.Argument(...),
    config: Optional[str] = typer.Option(None, "--config", "-c"),
) -> None:
    """Run a single pipeline stage by public alias or internal name."""
    run_named_stage(name, config)


@app.command()
def sync(
    source: Optional[str] = typer.Argument(None),
    config: Optional[str] = typer.Option(None, "--config", "-c"),
    dry_run: bool = typer.Option(False, "--dry-run"),
) -> None:
    """Sync source facts such as repos, docs, runtime signals, or data metadata."""
    ensure_indexing_config_current(config)
    selected = (source or "all").replace("-", "_")
    skip_disabled = selected in {"all", "docs"}
    if selected in {"all", "repos", "repo"}:
        maybe_run_sync_stage("repos", config, dry_run)
    if selected in {"all", "docs", "confluence"}:
        maybe_run_script("download_confluence.py", "confluence", config, dry_run, skip_disabled)
    if selected in {"all", "docs", "jira"}:
        maybe_run_script("download_jira.py", "jira", config, dry_run, skip_disabled)
    if selected in {"all", "runtime", "datadog"}:
        maybe_run_sync_stage("runtime", config, dry_run, source="datadog", skip_disabled=skip_disabled)
    if selected in {"all", "data", "bigquery"}:
        maybe_run_sync_stage("data", config, dry_run, source="bigquery", skip_disabled=skip_disabled)
    if selected not in {"all", "repos", "repo", "docs", "confluence", "jira", "runtime", "datadog", "data", "bigquery"}:
        raise typer.BadParameter(f"Unknown sync source: {source}")


@app.command()
def build(
    target: Optional[str] = typer.Argument(None),
    config: Optional[str] = typer.Option(None, "--config", "-c"),
) -> None:
    """Build local indexes, entity resolution, graph, and capability maps."""
    ensure_indexing_config_current(config)
    selected = (target or "all").replace("-", "_")
    groups = {
        "all": ["history", "blame", "code-graph", "code", "embeddings", "api", "static", "identity", "graph", "endpoints", "capabilities", "visualization"],
        "history": ["history"],
        "change_history": ["history"],
        "blame": ["blame"],
        "code": ["code"],
        "embeddings": ["embeddings"],
        "api": ["api"],
        "code_graph": ["code-graph"],
        "static": ["static"],
        "identity": ["identity"],
        "graph": ["graph"],
        "endpoints": ["endpoints"],
        "endpoint_map": ["endpoints"],
        "capabilities": ["capabilities"],
        "visualization": ["visualization"],
        "visual": ["visualization"],
    }
    if selected not in groups:
        raise typer.BadParameter(f"Unknown build target: {target}")
    run_stages(groups[selected], config)


@app.command()
def publish(config: Optional[str] = typer.Option(None, "--config", "-c")) -> None:
    """Publish the local graph into the configured external graph backend."""
    ensure_indexing_config_current(config)
    run_named_stage("publish", config)


@app.command()
def visualize(
    config: Optional[str] = typer.Option(None, "--config", "-c"),
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8765, "--port"),
    refresh: bool = typer.Option(False, "--refresh", help="Rebuild visualization JSON before starting the UI."),
    no_browser: bool = typer.Option(False, "--no-browser", help="Do not open the browser automatically."),
) -> None:
    """Start the local Proofline visualization UI."""
    from proofline.visualization import serve_visualization

    setup_logging()
    cfg = load_config(config_path(config))
    ensure_dirs(cfg)
    serve_visualization(cfg, host=host, port=port, open_browser=not no_browser, rebuild=refresh)


@app.command()
def ui(
    config: Optional[str] = typer.Option(None, "--config", "-c"),
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8766, "--port"),
    no_browser: bool = typer.Option(False, "--no-browser", help="Do not open the browser automatically."),
) -> None:
    """Start the local Proofline control UI."""
    from proofline.ui.server import serve_ui

    if host not in {"127.0.0.1", "localhost", "::1"}:
        typer.echo(
            "Warning: Proofline UI can start local pipeline jobs and edit config. "
            f"Binding to {host!r} may expose it beyond this machine.",
            err=True,
        )
    serve_ui(config_path(config), host=host, port=port, open_browser=not no_browser)


@app.command()
def status(
    config: Optional[str] = typer.Option(None, "--config", "-c"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Show database and pipeline status."""
    cfg, kb = load_runtime(config)
    try:
        tables = [
            "repo_inventory",
            "repo_files",
            "git_commits",
            "git_file_changes",
            "git_semantic_changes",
            "git_cochange_edges",
            "code_chunks",
            "code_index_repo_status",
            "api_endpoints",
            "service_identity",
            "code_embedding_index",
            "code_embedding_repo_status",
            "pipeline_repo_status",
            "nodes",
            "edges",
            "endpoint_dependency_map",
            "data_capabilities",
            "visualization_exports",
        ]
        counts = {}
        for table in tables:
            counts[table] = int(kb.query_df(f"SELECT COUNT(*) AS n FROM {table}").iloc[0]["n"])
        runs = kb.query_df(
            """
            SELECT stage, max(finished_at) AS finished_at,
                   arg_max(status, finished_at) AS status,
                   arg_max(details, finished_at) AS details
            FROM pipeline_runs
            GROUP BY stage
            ORDER BY finished_at DESC
            """
        ).to_dict("records")
    finally:
        kb.close()
    payload = {"database": cfg["storage"]["duckdb_path"], "counts": counts, "runs": runs}
    if json_output:
        typer.echo(json_dumps(payload))
        return
    console.print(f"Database: {payload['database']}")
    for table, count in counts.items():
        console.print(f"{table:24} {count}")
    if runs:
        console.print("\nRecent stages:")
        for row in runs[:12]:
            console.print(f"{row['stage']:18} {row['status']:8} {row['finished_at']}")


@app.command()
def sql(
    query: str = typer.Argument(...),
    config: Optional[str] = typer.Option(None, "--config", "-c"),
) -> None:
    """Run a SQL query against the local Proofline store."""
    _, kb = load_runtime(config)
    try:
        console.print(kb.query_df(query).to_string(index=False))
    finally:
        kb.close()


@app.command()
def ask(
    question: str = typer.Argument(...),
    config: Optional[str] = typer.Option(None, "--config", "-c"),
    project: Optional[str] = typer.Option(None, "--project"),
    env: Optional[str] = typer.Option(None, "--env"),
    window_days: Optional[int] = typer.Option(None, "--window-days"),
    agent_name: Optional[str] = typer.Option(None, "--agent"),
    raw_context: bool = typer.Option(False, "--raw-context"),
    raw_trace: bool = typer.Option(False, "--raw-trace"),
    quiet: bool = typer.Option(False, "--quiet"),
) -> None:
    from proofline.agent import ask as ask_commands

    ask_commands.ask(question, config_path(config), project, env, window_days, raw_context, raw_trace, quiet, agent_name)


@app.command()
def impact(
    project: str = typer.Option(..., "--project", "-p"),
    feature: str = typer.Option(..., "--feature", "-f"),
    config: Optional[str] = typer.Option(None, "--config", "-c"),
    env: Optional[str] = typer.Option(None, "--env"),
    window_days: Optional[int] = typer.Option(None, "--window-days"),
    raw_context: bool = typer.Option(False, "--raw-context"),
) -> None:
    from proofline.agent import ask as ask_commands

    ask_commands.impact(project, feature, config_path(config), env, window_days, raw_context)


@app.command("data-source")
def data_source(
    project: str = typer.Option(..., "--project", "-p"),
    feature: str = typer.Option(..., "--feature", "-f"),
    config: Optional[str] = typer.Option(None, "--config", "-c"),
    env: Optional[str] = typer.Option(None, "--env"),
    window_days: Optional[int] = typer.Option(None, "--window-days"),
    raw_context: bool = typer.Option(False, "--raw-context"),
) -> None:
    from proofline.agent import ask as ask_commands

    ask_commands.data_source(project, feature, config_path(config), env, window_days, raw_context)


@app.command("dependencies")
def dependencies(
    project: str = typer.Option(..., "--project", "-p"),
    config: Optional[str] = typer.Option(None, "--config", "-c"),
    env: Optional[str] = typer.Option(None, "--env"),
    window_days: Optional[int] = typer.Option(None, "--window-days"),
    raw_context: bool = typer.Option(False, "--raw-context"),
) -> None:
    from proofline.agent import ask as ask_commands

    ask_commands.dependency_report(project, config_path(config), env, window_days, raw_context)


@app.command()
def search(
    query: str = typer.Argument(...),
    config: Optional[str] = typer.Option(None, "--config", "-c"),
    repo: Optional[str] = typer.Option(None, "--repo", "--project", "-p"),
    limit: int = typer.Option(25, "--limit"),
) -> None:
    from proofline.agent import ask as ask_commands

    ask_commands.search(query, config_path(config), repo, limit)


@app.command()
def bootstrap() -> None:
    """Run the local bootstrap script."""
    root = Path(__file__).resolve().parents[1]
    subprocess.run([str(root / "scripts" / "bootstrap.sh")], cwd=str(root), check=True)


if __name__ == "__main__":
    app()
