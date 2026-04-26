from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path
from typing import Callable, Dict, Any, List, Optional

import pandas as pd
import typer

from corp_kb.config import load_config, ensure_dirs
from corp_kb.logging_utils import setup_logging, log_step, console
from corp_kb.storage import KB
from corp_kb.utils import now_iso, json_dumps
from corp_kb.extractors.repo import scan_all_repos
from corp_kb.extractors.code_index import build_chunks, build_sqlite_fts
from corp_kb.extractors.api_surface import parse_api_specs, extract_static_routes
from corp_kb.extractors.static_edges import extract_static_edges
from corp_kb.extractors.datadog import pull_service_dependencies, pull_service_definitions, search_spans, search_logs, build_runtime_edges_from_dd
from corp_kb.extractors.bigquery import pull_bq_jobs, build_table_usage
from corp_kb.extractors.entity_resolution import build_service_identity
from corp_kb.extractors.graph_build import build_graph
from corp_kb.extractors.endpoint_map import build_endpoint_dependency_map
from corp_kb.extractors.capabilities import build_capabilities
from corp_kb.extractors.compatibility import build_compatibility_index

app = typer.Typer(help="Local corporate code/runtime/data knowledge graph POC pipeline")


def stage_record(kb: KB, stage: str, started: str, status: str, details: str = "") -> None:
    kb.append_df("pipeline_runs", pd.DataFrame([{
        "stage": stage, "started_at": started, "finished_at": now_iso(), "status": status, "details": details,
    }]))


def run_stage(name: str, cfg: Dict[str, Any], func: Callable[[KB, Dict[str, Any]], None]) -> None:
    log_step(name)
    kb = KB(cfg["storage"]["duckdb_path"])
    started = now_iso()
    try:
        func(kb, cfg)
        stage_record(kb, name, started, "ok", "")
    except Exception as e:
        stage_record(kb, name, started, "error", str(e))
        raise
    finally:
        kb.close()


def maybe_clone_repos(cfg: Dict[str, Any]) -> None:
    repos_cfg = cfg.get("repos", {})
    urls_file = repos_cfg.get("clone_urls_file")
    if not urls_file:
        return
    urls_path = Path(urls_file)
    if not urls_path.exists():
        console.print(f"[yellow]clone_urls_file not found: {urls_path}; skipping clone[/yellow]")
        return
    root = Path(repos_cfg.get("root", "./repos"))
    root.mkdir(parents=True, exist_ok=True)
    for line in urls_path.read_text().splitlines():
        url = line.strip()
        if not url or url.startswith("#"):
            continue
        name = url.rstrip("/").split("/")[-1].replace(".git", "")
        target = root / name
        if target.exists():
            if repos_cfg.get("update_existing", False):
                console.print(f"Updating {name}")
                subprocess.run(["git", "fetch", "--all", "--prune"], cwd=target, check=False)
                subprocess.run(["git", "pull", "--ff-only"], cwd=target, check=False)
            continue
        console.print(f"Cloning {url} -> {target}")
        subprocess.run(["git", "clone", url, str(target)], check=False)


def stage_repo_ingest(kb: KB, cfg: Dict[str, Any]) -> None:
    maybe_clone_repos(cfg)
    dfs = scan_all_repos(cfg)
    for table, df in dfs.items():
        kb.replace_df(table, df)
    console.print(f"repos: {len(dfs['repo_inventory'])}, files: {len(dfs['repo_files'])}")


def stage_code_index(kb: KB, cfg: Dict[str, Any]) -> None:
    repo_files = kb.query_df("SELECT * FROM repo_files")
    chunks = build_chunks(repo_files, cfg)
    kb.replace_df("code_chunks", chunks)
    if cfg.get("indexing", {}).get("lexical_fts", True):
        build_sqlite_fts(chunks, cfg["storage"]["sqlite_fts_path"])
    console.print(f"chunks: {len(chunks)}")


def stage_api_surface(kb: KB, cfg: Dict[str, Any]) -> None:
    inv = kb.query_df("SELECT * FROM repo_inventory")
    files = kb.query_df("SELECT * FROM repo_files")
    contracts, endpoints1 = parse_api_specs(inv, files)
    endpoints2 = extract_static_routes(inv, files)
    endpoints = pd.concat([endpoints1, endpoints2], ignore_index=True) if not endpoints1.empty or not endpoints2.empty else pd.DataFrame()
    kb.replace_df("api_contracts", contracts)
    kb.replace_df("api_endpoints", endpoints)
    console.print(f"contracts: {len(contracts)}, endpoints: {len(endpoints)}")


def stage_static_edges(kb: KB, cfg: Dict[str, Any]) -> None:
    inv = kb.query_df("SELECT * FROM repo_inventory")
    files = kb.query_df("SELECT * FROM repo_files")
    edges = extract_static_edges(inv, files)
    kb.replace_df("static_edges", edges)
    console.print(f"static edges: {len(edges)}")


def stage_datadog(kb: KB, cfg: Dict[str, Any]) -> None:
    services, dd_edges = pull_service_dependencies(cfg)
    svc_defs = pull_service_definitions(cfg)
    if not svc_defs.empty:
        services = pd.concat([services, svc_defs], ignore_index=True) if not services.empty else svc_defs
    spans = search_spans(cfg)
    logs = search_logs(cfg)
    runtime_svc, runtime_ep = build_runtime_edges_from_dd(spans, logs, dd_edges, cfg.get("datadog", {}).get("windows_days", [30]))
    kb.replace_df("datadog_services", services)
    kb.replace_df("datadog_service_edges", dd_edges)
    kb.replace_df("datadog_spans", spans)
    kb.replace_df("datadog_logs", logs)
    kb.replace_df("runtime_service_edges", runtime_svc)
    kb.replace_df("runtime_endpoint_edges", runtime_ep)
    console.print(f"dd services rows: {len(services)}, dd service edges: {len(dd_edges)}, spans: {len(spans)}, logs: {len(logs)}, runtime edges: {len(runtime_svc)}, endpoint edges: {len(runtime_ep)}")


def stage_bigquery(kb: KB, cfg: Dict[str, Any]) -> None:
    jobs = pull_bq_jobs(cfg)
    usage = build_table_usage(jobs)
    kb.replace_df("bq_jobs", jobs)
    kb.replace_df("bq_table_usage", usage)
    console.print(f"bq jobs: {len(jobs)}, bq usage rows: {len(usage)}")


def stage_entity_resolution(kb: KB, cfg: Dict[str, Any]) -> None:
    inv = kb.query_df("SELECT * FROM repo_inventory")
    dd_services = kb.query_df("SELECT * FROM datadog_services")
    dd_edges = kb.query_df("SELECT * FROM datadog_service_edges")
    own = kb.query_df("SELECT * FROM ownership")
    api = kb.query_df("SELECT * FROM api_endpoints")
    static = kb.query_df("SELECT * FROM static_edges")
    bq = kb.query_df("SELECT * FROM bq_table_usage")
    service_identity, aliases, unresolved = build_service_identity(inv, dd_services, dd_edges, own, api, static, bq)
    kb.replace_df("service_identity", service_identity)
    kb.replace_df("entity_aliases", aliases)
    kb.replace_df("unresolved_entities", unresolved)
    console.print(f"services: {len(service_identity)}, aliases: {len(aliases)}, unresolved: {len(unresolved)}")


def stage_graph(kb: KB, cfg: Dict[str, Any]) -> None:
    nodes, edges, evidence = build_graph(
        kb.query_df("SELECT * FROM repo_inventory"),
        kb.query_df("SELECT * FROM service_identity"),
        kb.query_df("SELECT * FROM entity_aliases"),
        kb.query_df("SELECT * FROM api_endpoints"),
        kb.query_df("SELECT * FROM static_edges"),
        kb.query_df("SELECT * FROM runtime_service_edges"),
        kb.query_df("SELECT * FROM runtime_endpoint_edges"),
        kb.query_df("SELECT * FROM bq_table_usage"),
        kb.query_df("SELECT * FROM ownership"),
    )
    kb.replace_df("nodes", nodes)
    kb.replace_df("edges", edges)
    kb.replace_df("evidence", evidence)
    console.print(f"nodes: {len(nodes)}, edges: {len(edges)}, evidence: {len(evidence)}")


def stage_endpoint_map(kb: KB, cfg: Dict[str, Any]) -> None:
    df = build_endpoint_dependency_map(
        kb.query_df("SELECT * FROM api_endpoints"),
        kb.query_df("SELECT * FROM runtime_endpoint_edges"),
        kb.query_df("SELECT * FROM static_edges"),
        kb.query_df("SELECT * FROM service_identity"),
    )
    kb.replace_df("endpoint_dependency_map", df)
    console.print(f"endpoint dependency rows: {len(df)}")


def stage_capabilities(kb: KB, cfg: Dict[str, Any]) -> None:
    caps = build_capabilities(kb.query_df("SELECT * FROM api_endpoints"), kb.query_df("SELECT * FROM bq_table_usage"), kb.query_df("SELECT * FROM service_identity"))
    kb.replace_df("data_capabilities", caps)
    compat = build_compatibility_index(kb.query_df("SELECT * FROM api_endpoints"), kb.query_df("SELECT * FROM static_edges"), kb.query_df("SELECT * FROM runtime_service_edges"))
    kb.replace_df("compatibility_index", compat)
    console.print(f"capabilities: {len(caps)}, compatibility risk entities: {len(compat)}")


def stage_smoke(kb: KB, cfg: Dict[str, Any]) -> None:
    tables = ["repo_inventory", "service_identity", "api_endpoints", "edges", "endpoint_dependency_map", "data_capabilities"]
    rows = []
    for t in tables:
        n = kb.query_df(f"SELECT COUNT(*) AS n FROM {t}").iloc[0]["n"]
        rows.append({"table": t, "rows": int(n)})
    console.print(pd.DataFrame(rows).to_string(index=False))


STAGES: Dict[str, Callable[[KB, Dict[str, Any]], None]] = {
    "repo_ingest": stage_repo_ingest,
    "code_index": stage_code_index,
    "api_surface": stage_api_surface,
    "static_edges": stage_static_edges,
    "datadog": stage_datadog,
    "bigquery": stage_bigquery,
    "entity_resolution": stage_entity_resolution,
    "graph": stage_graph,
    "endpoint_map": stage_endpoint_map,
    "capabilities": stage_capabilities,
    "smoke": stage_smoke,
}

FULL_ORDER = [
    "repo_ingest", "code_index", "api_surface", "static_edges", "datadog", "bigquery",
    "entity_resolution", "graph", "endpoint_map", "capabilities", "smoke",
]


@app.command()
def full(config: str = typer.Option("config.yaml", "--config", "-c"), from_stage: Optional[str] = typer.Option(None), to_stage: Optional[str] = typer.Option(None)):
    setup_logging()
    cfg = load_config(config)
    ensure_dirs(cfg)
    order = FULL_ORDER[:]
    if from_stage:
        order = order[order.index(from_stage):]
    if to_stage:
        order = order[: order.index(to_stage) + 1]
    for s in order:
        run_stage(s, cfg, STAGES[s])


@app.command()
def stage(name: str, config: str = typer.Option("config.yaml", "--config", "-c")):
    setup_logging()
    cfg = load_config(config)
    ensure_dirs(cfg)
    if name not in STAGES:
        raise typer.BadParameter(f"Unknown stage: {name}. Known: {', '.join(STAGES)}")
    run_stage(name, cfg, STAGES[name])


if __name__ == "__main__":
    app()
