from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, Optional

import typer

from proofline.agent import ask as ask_commands
from proofline.config import CONFIG_ENV_VAR, DEFAULT_CONFIG, ensure_dirs, load_config
from proofline.logging_utils import console, setup_logging
from proofline.pipeline.runner import STAGES, resolve_stage, run_order, run_stage
from proofline.storage import KB
from proofline.utils import json_dumps

app = typer.Typer(help="Proofline CLI")


def config_path(config: Optional[str]) -> str:
    return config or os.getenv(CONFIG_ENV_VAR) or DEFAULT_CONFIG


def load_runtime(config: Optional[str]) -> tuple[dict[str, Any], KB]:
    cfg = load_config(config_path(config))
    ensure_dirs(cfg)
    return cfg, KB(cfg["storage"]["duckdb_path"])


def run_named_stage(name: str, config: Optional[str]) -> None:
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
) -> None:
    """Create a proofline.yaml config and local working directories."""
    target = Path(config_path(config))
    if target.exists() and not force:
        raise typer.BadParameter(f"{target} already exists. Pass --force to overwrite it.")
    example = Path(__file__).resolve().parents[1] / "proofline.example.yaml"
    if not example.exists():
        raise typer.BadParameter(f"Template not found: {example}")
    shutil.copyfile(example, target)
    cfg = load_config(target)
    ensure_dirs(cfg)
    console.print(f"Created {target}")


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
def run(
    config: Optional[str] = typer.Option(None, "--config", "-c"),
    from_stage: Optional[str] = typer.Option(None, "--from"),
    to_stage: Optional[str] = typer.Option(None, "--to"),
) -> None:
    """Run the main Proofline pipeline."""
    run_order(config_path(config), from_stage, to_stage)


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
    selected = (target or "all").replace("-", "_")
    groups = {
        "all": ["code", "embeddings", "api", "code-graph", "static", "identity", "graph", "endpoints", "capabilities"],
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
    }
    if selected not in groups:
        raise typer.BadParameter(f"Unknown build target: {target}")
    run_stages(groups[selected], config)


@app.command()
def publish(config: Optional[str] = typer.Option(None, "--config", "-c")) -> None:
    """Publish the local graph into the configured external graph backend."""
    run_named_stage("publish", config)


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
            "code_chunks",
            "api_endpoints",
            "service_identity",
            "nodes",
            "edges",
            "endpoint_dependency_map",
            "data_capabilities",
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
    raw_context: bool = typer.Option(False, "--raw-context"),
) -> None:
    ask_commands.ask(question, config_path(config), project, env, window_days, raw_context)


@app.command()
def impact(
    project: str = typer.Option(..., "--project", "-p"),
    feature: str = typer.Option(..., "--feature", "-f"),
    config: Optional[str] = typer.Option(None, "--config", "-c"),
    env: Optional[str] = typer.Option(None, "--env"),
    window_days: Optional[int] = typer.Option(None, "--window-days"),
    raw_context: bool = typer.Option(False, "--raw-context"),
) -> None:
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
    ask_commands.data_source(project, feature, config_path(config), env, window_days, raw_context)


@app.command("dependencies")
def dependencies(
    project: str = typer.Option(..., "--project", "-p"),
    config: Optional[str] = typer.Option(None, "--config", "-c"),
    env: Optional[str] = typer.Option(None, "--env"),
    window_days: Optional[int] = typer.Option(None, "--window-days"),
    raw_context: bool = typer.Option(False, "--raw-context"),
) -> None:
    ask_commands.dependency_report(project, config_path(config), env, window_days, raw_context)


@app.command()
def search(
    query: str = typer.Argument(...),
    config: Optional[str] = typer.Option(None, "--config", "-c"),
    repo: Optional[str] = typer.Option(None, "--repo", "--project", "-p"),
    limit: int = typer.Option(25, "--limit"),
) -> None:
    ask_commands.search(query, config_path(config), repo, limit)


@app.command()
def bootstrap() -> None:
    """Run the local bootstrap script."""
    subprocess.run(["./scripts/bootstrap.sh"], check=True)


if __name__ == "__main__":
    app()
