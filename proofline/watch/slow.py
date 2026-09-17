"""Slow lane for one repo: git delta + repo-scoped derived tables.

Fast lane (repo_ingest/code_index/embeddings/api/static) runs on every
batch. Slow lane (history/blame/identity/graph/endpoints/capabilities)
runs when HEAD moved or every ``slow_lane_minutes`` - it is heavier and
mostly commit-driven.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from proofline.extractors.api_surface import extract_static_routes, parse_api_specs
from proofline.extractors.capabilities import build_capabilities
from proofline.extractors.compatibility import build_compatibility_index
from proofline.extractors.endpoint_map import build_endpoint_dependency_map
from proofline.extractors.entity_resolution import build_service_identity
from proofline.extractors.git_history import extract_repo_git_blame, extract_repo_git_history
from proofline.extractors.graph_build import build_graph
from proofline.extractors.static_edges import extract_static_edges
from proofline.pipeline.repo_jobs import mark_repo_stage
from proofline.pipeline.runner import _append_repo_git_history
from proofline.utils import now_iso
from proofline.watch import EventCallback, emit
from proofline.watch.repos import repo_head


def head_changed(kb: Any, repo_path: Path, repo_id: str) -> bool:
    current = repo_head(repo_path)
    if not current:
        return True
    try:
        rows = kb.query_df("SELECT commit_sha FROM repo_inventory WHERE repo_id = ?", [repo_id])
    except Exception:
        return True
    if rows.empty:
        return True
    return str(rows.iloc[0].get("commit_sha") or "") != current


def refresh_git_history(kb, cfg, repo_path, repo_id, *, on_event=None):
    gh_cfg = dict(cfg.get("git_history") or {})
    if not gh_cfg.get("enabled", True):
        return {"repo_id": repo_id, "skipped": True}
    gh_cfg["current_blame"] = False
    existing = kb.query_df("SELECT commit_sha FROM git_commits WHERE repo_id = ?", [repo_id])
    shas = existing["commit_sha"].fillna("").astype(str).tolist() if not existing.empty else []
    if shas:
        gh_cfg["stop_commit_shas"] = set(shas)
    rows = extract_repo_git_history(repo_path, repo_id, gh_cfg)
    counts = _append_repo_git_history(kb, repo_id, rows)
    fp = str(repo_head(repo_path)) + ":watch"
    mark_repo_stage(kb, "git_history", repo_id, fp, "ok", item_count=counts.get("git_commits", 0))
    emit(on_event, "git_history", {"repo_id": repo_id, **counts})
    return {"repo_id": repo_id, **counts}
def refresh_git_blame(kb, cfg, repo_path, repo_id, *, on_event=None):
    gh_cfg = dict(cfg.get("git_history") or {})
    if not gh_cfg.get("enabled", True) or not gh_cfg.get("current_blame", True):
        return {"repo_id": repo_id, "skipped": True}
    rows = extract_repo_git_blame(repo_path, repo_id, gh_cfg).get("git_blame_current", [])
    kb.execute("DELETE FROM git_blame_current WHERE repo_id = ?", [repo_id])
    if rows:
        kb.append_df("git_blame_current", pd.DataFrame(rows))
    mark_repo_stage(kb, "git_blame", repo_id, str(repo_head(repo_path)) + ":blame:watch",
                    "ok", item_count=len(rows))
    emit(on_event, "git_blame", {"repo_id": repo_id, "rows": len(rows)})
    return {"repo_id": repo_id, "rows": len(rows)}


def _scoped_replace(kb, table, repo_id, df):
    """Replace one repo's slice of a derived table (all have repo_id)."""
    kb.execute(f"DELETE FROM {table} WHERE repo_id = ?", [repo_id])
    if df is not None and not df.empty:
        part = df[df["repo_id"].astype(str) == str(repo_id)] if "repo_id" in df.columns else df
        if not part.empty:
            kb.append_df(table, part)


def refresh_api_static(kb, cfg, repo_id, *, on_event=None):
    inv = kb.query_df("SELECT * FROM repo_inventory WHERE repo_id = ?", [repo_id])
    files = kb.query_df("SELECT * FROM repo_files WHERE repo_id = ?", [repo_id])
    contracts, endpoints1 = parse_api_specs(inv, files)
    endpoints2 = extract_static_routes(inv, files)
    endpoints = pd.concat([endpoints1, endpoints2], ignore_index=True) if not endpoints1.empty or not endpoints2.empty else pd.DataFrame()
    static = extract_static_edges(inv, files)
    _scoped_replace(kb, "api_contracts", repo_id, contracts)
    _scoped_replace(kb, "api_endpoints", repo_id, endpoints)
    _scoped_replace(kb, "static_edges", repo_id, static)
    mark_repo_stage(kb, "api_surface", repo_id, "watch", "ok", item_count=len(endpoints))
    mark_repo_stage(kb, "static_edges", repo_id, "watch", "ok", item_count=len(static))
    emit(on_event, "api_static",
         {"repo_id": repo_id, "endpoints": len(endpoints), "static_edges": len(static)})
    return {"repo_id": repo_id, "endpoints": len(endpoints), "static_edges": len(static)}


def refresh_derived(kb, cfg, repo_id, *, on_event=None):
    """Repo-scoped identity/graph/endpoints/capabilities.

    These builders are corpus-wide; we rebuild them fully but only when
    the slow lane fires (HEAD moved or timer), never on every keystroke.
    """
    inv = kb.query_df("SELECT * FROM repo_inventory")
    api = kb.query_df("SELECT * FROM api_endpoints")
    static = kb.query_df("SELECT * FROM static_edges")
    service_identity, aliases, unresolved = build_service_identity(
        inv, kb.query_df("SELECT * FROM datadog_services"),
        kb.query_df("SELECT * FROM datadog_service_edges"),
        kb.query_df("SELECT * FROM ownership"), api, static,
        kb.query_df("SELECT * FROM bq_table_usage"))
    _scoped_replace(kb, "service_identity", repo_id, service_identity)
    nodes, edges, evidence = build_graph(
        inv, kb.query_df("SELECT * FROM service_identity"),
        kb.query_df("SELECT * FROM entity_aliases"), api, static,
        kb.query_df("SELECT * FROM runtime_service_edges"),
        kb.query_df("SELECT * FROM runtime_endpoint_edges"),
        kb.query_df("SELECT * FROM bq_table_usage"),
        kb.query_df("SELECT * FROM ownership"),
        kb.query_df("SELECT * FROM code_graph_symbols"),
        kb.query_df("SELECT * FROM code_graph_edges"),
        kb.query_df("SELECT * FROM git_commits"),
        kb.query_df("SELECT * FROM git_file_changes"),
        kb.query_df("SELECT * FROM git_semantic_changes"),
        kb.query_df("SELECT * FROM git_cochange_edges"))
    kb.replace_df("nodes", nodes)
    kb.replace_df("edges", edges)
    kb.replace_df("evidence", evidence)
    epmap = build_endpoint_dependency_map(
        api, kb.query_df("SELECT * FROM runtime_endpoint_edges"), static,
        kb.query_df("SELECT * FROM service_identity"))
    kb.replace_df("endpoint_dependency_map", epmap)
    caps = build_capabilities(api, kb.query_df("SELECT * FROM bq_table_usage"),
                              kb.query_df("SELECT * FROM service_identity"))
    compat = build_compatibility_index(api, static, kb.query_df("SELECT * FROM runtime_service_edges"))
    kb.replace_df("data_capabilities", caps)
    kb.replace_df("compatibility_index", compat)
    for stage in ("entity_resolution", "graph", "endpoint_map", "capabilities"):
        mark_repo_stage(kb, stage, repo_id, "watch", "ok", details="watch slow lane")
    # aliases/unresolved are global maps rebuilt alongside identity.
    kb.replace_df("entity_aliases", aliases)
    kb.replace_df("unresolved_entities", unresolved)
    emit(on_event, "derived", {"repo_id": repo_id, "nodes": len(nodes), "edges": len(edges)})
    return {"repo_id": repo_id, "nodes": len(nodes), "edges": len(edges)}
