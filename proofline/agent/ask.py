from __future__ import annotations

from typing import Any, Dict, Optional

import typer

from proofline.config import DEFAULT_CONFIG, load_config, ensure_dirs
from proofline.logging_utils import setup_logging
from proofline.storage import KB
from proofline.agent.tools import KBTools
from proofline.agent.compose import maybe_llm_answer, render_markdown
from proofline.utils import json_dumps

app = typer.Typer(help="Ask the local corporate knowledge graph")


def tools_for(config: str):
    cfg = load_config(config)
    ensure_dirs(cfg)
    kb = KB(cfg["storage"]["duckdb_path"])
    tools = KBTools(kb, cfg["storage"].get("sqlite_fts_path"), cfg)
    return cfg, kb, tools


def classify_question(q: str) -> str:
    low = q.lower()
    if any(x in low for x in [
        "dependency", "dependencies", "endpoint", "endpoints", "slow", "latency",
        "performance", "scary to deploy", "hard to deploy", "service graph",
        "downstream", "upstream", "calls", "depend on",
    ]):
        return "dependency_report"
    if any(x in low for x in [
        "source of truth", "data source", "pull data", "fetch data", "which service",
        "which api", "api request", "api example", "provider service", "where should i get",
    ]):
        return "data_source_recommendation"
    if any(x in low for x in [
        "impact", "affect", "break", "breaking", "compat", "backward", "backwards",
        "if i implement", "if i add", "if i change", "what can break",
    ]):
        return "impact_analysis"
    return "generic"


def extract_project_feature(question: str) -> tuple[str, str]:
    """Best-effort English extraction. Falls back to the full question as the feature text."""
    import re

    project = ""
    feature = question
    for pat in [
        r"project\s+[`\"']?([A-Za-z0-9_.\-/]+)",
        r"service\s+[`\"']?([A-Za-z0-9_.\-/]+)",
        r"repo(?:sitory)?\s+[`\"']?([A-Za-z0-9_.\-/]+)",
        r"--project\s+([A-Za-z0-9_.\-/]+)",
    ]:
        m = re.search(pat, question, re.I)
        if m:
            project = m.group(1).strip("`\"'.,:; ")
            break
    for pat in [
        r"feature\s+[`\"']?([^`\"']+)",
        r"implement\s+(.+?)\s+in\s+(?:project|service|repo|repository)",
        r"add\s+(.+?)\s+in\s+(?:project|service|repo|repository)",
        r"change\s+(.+?)\s+in\s+(?:project|service|repo|repository)",
        r"if i (?:implement|add|change)\s+(.+?)\s+in\s+(?:project|service|repo|repository)",
    ]:
        m = re.search(pat, question, re.I)
        if m:
            feature = m.group(1).strip("`\"'.,:; ")
            break
    return project, feature


def build_impact_context(tools: KBTools, project: str, feature: str, env: str, window_days: int) -> Dict[str, Any]:
    resolved = tools.resolve_project(project)
    sid = ((resolved.get("service") or {}).get("service_id") if resolved.get("found") else project)
    repo_id = (resolved.get("service") or {}).get("repo_id") if resolved.get("found") else None
    return {
        "question_type": "impact_analysis",
        "project_name": project,
        "feature_name": feature,
        "env": env,
        "window_days": window_days,
        "project": resolved,
        "profile": tools.get_service_profile(sid),
        "dependencies": tools.get_service_dependencies(sid, env, window_days),
        "dependents": tools.get_service_dependents(sid, env, window_days),
        "bq_usage": tools.get_bq_usage(sid, window_days),
        "capabilities": tools.search_capabilities(feature, limit=30),
        "graph_neighborhood": tools.get_graph_neighborhood(f"service:{sid}", limit=150),
        "repo_graph_neighborhood": tools.get_graph_neighborhood(f"repo:{repo_id}", limit=150) if repo_id else {},
        "change_history": tools.get_change_history(sid, feature, limit=60),
        "code_graph": tools.search_code_graph(feature, repo_id=repo_id, limit=40),
        "code_hits": tools.search_code(
            feature,
            repo_id=repo_id,
            limit=30,
        ),
    }


def build_data_source_context(tools: KBTools, project: str, feature: str, env: str, window_days: int) -> Dict[str, Any]:
    resolved = tools.resolve_project(project)
    sid = ((resolved.get("service") or {}).get("service_id") if resolved.get("found") else project)
    repo_id = (resolved.get("service") or {}).get("repo_id") if resolved.get("found") else None
    return {
        "question_type": "data_source_recommendation",
        "project_name": project,
        "feature_name": feature,
        "env": env,
        "window_days": window_days,
        "project": resolved,
        "profile": tools.get_service_profile(sid),
        "capabilities": tools.search_capabilities(feature, limit=50),
        "dependencies": tools.get_service_dependencies(sid, env, window_days),
        "graph_neighborhood": tools.get_graph_neighborhood(f"service:{sid}", limit=150),
        "change_history": tools.get_change_history(sid, feature, limit=60),
        "code_graph": tools.search_code_graph(feature, repo_id=repo_id, limit=40),
        "code_hits": tools.search_code(feature, repo_id=None, limit=30),
    }


def build_dependency_context(tools: KBTools, project: str, env: str, window_days: int) -> Dict[str, Any]:
    resolved = tools.resolve_project(project)
    sid = ((resolved.get("service") or {}).get("service_id") if resolved.get("found") else project)
    repo_id = (resolved.get("service") or {}).get("repo_id") if resolved.get("found") else None
    return {
        "question_type": "dependency_report",
        "project_name": project,
        "env": env,
        "window_days": window_days,
        "project": resolved,
        "profile": tools.get_service_profile(sid),
        "dependencies": tools.get_service_dependencies(sid, env, window_days),
        "dependents": tools.get_service_dependents(sid, env, window_days),
        "endpoint_dependencies": tools.get_endpoint_dependencies(sid, env, window_days),
        "bq_usage": tools.get_bq_usage(sid, window_days),
        "graph_neighborhood": tools.get_graph_neighborhood(f"service:{sid}", limit=200),
        "repo_graph_neighborhood": tools.get_graph_neighborhood(f"repo:{repo_id}", limit=200) if repo_id else {},
        "change_history": tools.get_change_history(sid, project, limit=80),
        "code_graph": tools.search_code_graph(project, repo_id=repo_id, limit=50),
    }


def emit_context_or_answer(ctx: Dict[str, Any], cfg: Dict[str, Any], raw_context: bool = False) -> None:
    if raw_context:
        typer.echo(json_dumps(ctx))
        return
    answer = maybe_llm_answer(ctx, cfg)
    typer.echo(answer or render_markdown(ctx))


@app.command("ask")
def ask(
    question: str = typer.Argument(..., help="Natural language question"),
    config: str = typer.Option(DEFAULT_CONFIG, "--config", "-c"),
    project: Optional[str] = typer.Option(None, "--project"),
    env: Optional[str] = typer.Option(None, "--env"),
    window_days: Optional[int] = typer.Option(None, "--window-days"),
    raw_context: bool = typer.Option(False, "--raw-context"),
):
    setup_logging()
    cfg, kb, tools = tools_for(config)
    try:
        env = env or cfg.get("agent", {}).get("default_env", "prod")
        window_days = window_days or int(cfg.get("agent", {}).get("default_window_days", 30))
        qtype = classify_question(question)
        p, f = extract_project_feature(question)
        p = project or p
        if qtype == "dependency_report":
            if not p:
                raise typer.BadParameter("Could not infer the project. Pass --project.")
            ctx = build_dependency_context(tools, p, env, window_days)
        elif qtype == "data_source_recommendation":
            if not p:
                raise typer.BadParameter("Could not infer the project. Pass --project.")
            ctx = build_data_source_context(tools, p, f, env, window_days)
        elif qtype == "impact_analysis":
            if not p:
                raise typer.BadParameter("Could not infer the project. Pass --project.")
            ctx = build_impact_context(tools, p, f, env, window_days)
        else:
            resolved = tools.resolve_project(p) if p else {}
            repo_id = (resolved.get("service") or {}).get("repo_id") if resolved.get("found") else None
            sid = (resolved.get("service") or {}).get("service_id") if resolved.get("found") else ""
            ctx = {
                "question_type": "generic",
                "question": question,
                "project": resolved,
                "graph_neighborhood": tools.get_graph_neighborhood(f"service:{sid}", limit=100) if sid else {},
                "repo_graph_neighborhood": tools.get_graph_neighborhood(f"repo:{repo_id}", limit=100) if repo_id else {},
                "change_history": tools.get_change_history(sid, question, limit=40) if sid else {},
                "capabilities": tools.search_capabilities(question, limit=25),
                "code_graph": tools.search_code_graph(question, repo_id=repo_id, limit=40),
                "code_hits": tools.search_code(question, repo_id=repo_id, limit=25),
            }
        emit_context_or_answer(ctx, cfg, raw_context)
    finally:
        kb.close()


@app.command("impact")
def impact(
    project: str = typer.Option(..., "--project", "-p"),
    feature: str = typer.Option(..., "--feature", "-f"),
    config: str = typer.Option(DEFAULT_CONFIG, "--config", "-c"),
    env: Optional[str] = typer.Option(None, "--env"),
    window_days: Optional[int] = typer.Option(None, "--window-days"),
    raw_context: bool = typer.Option(False, "--raw-context"),
):
    cfg, kb, tools = tools_for(config)
    try:
        env = env or cfg.get("agent", {}).get("default_env", "prod")
        window_days = window_days or int(cfg.get("agent", {}).get("default_window_days", 30))
        emit_context_or_answer(build_impact_context(tools, project, feature, env, window_days), cfg, raw_context)
    finally:
        kb.close()


@app.command("data-source")
def data_source(
    project: str = typer.Option(..., "--project", "-p"),
    feature: str = typer.Option(..., "--feature", "-f"),
    config: str = typer.Option(DEFAULT_CONFIG, "--config", "-c"),
    env: Optional[str] = typer.Option(None, "--env"),
    window_days: Optional[int] = typer.Option(None, "--window-days"),
    raw_context: bool = typer.Option(False, "--raw-context"),
):
    cfg, kb, tools = tools_for(config)
    try:
        env = env or cfg.get("agent", {}).get("default_env", "prod")
        window_days = window_days or int(cfg.get("agent", {}).get("default_window_days", 30))
        emit_context_or_answer(build_data_source_context(tools, project, feature, env, window_days), cfg, raw_context)
    finally:
        kb.close()


@app.command("dependencies")
def dependency_report(
    project: str = typer.Option(..., "--project", "-p"),
    config: str = typer.Option(DEFAULT_CONFIG, "--config", "-c"),
    env: Optional[str] = typer.Option(None, "--env"),
    window_days: Optional[int] = typer.Option(None, "--window-days"),
    raw_context: bool = typer.Option(False, "--raw-context"),
):
    cfg, kb, tools = tools_for(config)
    try:
        env = env or cfg.get("agent", {}).get("default_env", "prod")
        window_days = window_days or int(cfg.get("agent", {}).get("default_window_days", 30))
        emit_context_or_answer(build_dependency_context(tools, project, env, window_days), cfg, raw_context)
    finally:
        kb.close()


@app.command("search")
def search(
    query: str = typer.Argument(...),
    config: str = typer.Option(DEFAULT_CONFIG, "--config", "-c"),
    repo: Optional[str] = typer.Option(None, "--repo", "--project", "-p"),
    limit: int = typer.Option(25, "--limit"),
):
    cfg, kb, tools = tools_for(config)
    try:
        typer.echo(json_dumps({
            "code_hits": tools.search_code(query, repo_id=repo, limit=limit),
            "code_graph": tools.search_code_graph(query, repo_id=repo, limit=limit),
            "capabilities": tools.search_capabilities(query, limit=limit),
        }))
    finally:
        kb.close()


if __name__ == "__main__":
    app()
