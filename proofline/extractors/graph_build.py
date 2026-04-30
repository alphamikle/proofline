from __future__ import annotations

from typing import Any, Dict, List, Tuple

import pandas as pd

from proofline.utils import normalize_name
from proofline.utils import stable_id, json_dumps, now_iso


def build_graph(
    repo_inventory: pd.DataFrame,
    service_identity: pd.DataFrame,
    aliases: pd.DataFrame,
    api_endpoints: pd.DataFrame,
    static_edges: pd.DataFrame,
    runtime_service_edges: pd.DataFrame,
    runtime_endpoint_edges: pd.DataFrame,
    bq_usage: pd.DataFrame,
    ownership: pd.DataFrame,
    code_graph_symbols: pd.DataFrame | None = None,
    code_graph_edges: pd.DataFrame | None = None,
    git_commits: pd.DataFrame | None = None,
    git_file_changes: pd.DataFrame | None = None,
    git_semantic_changes: pd.DataFrame | None = None,
    git_cochange_edges: pd.DataFrame | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    nodes: List[Dict[str, Any]] = []
    edges: List[Dict[str, Any]] = []
    evidence: List[Dict[str, Any]] = []
    alias_exact: Dict[str, str] = {}
    alias_norm: Dict[str, str] = {}
    if aliases is not None and not aliases.empty:
        for _, a in aliases.iterrows():
            alias = str(a.get("alias") or "")
            canonical = str(a.get("canonical_id") or "")
            if alias and canonical:
                alias_exact[alias.lower()] = canonical
                alias_norm[normalize_name(alias)] = canonical

    def canon(raw: str) -> str:
        if not raw:
            return ""
        hit = alias_exact.get(raw.lower())
        if hit:
            return hit
        return alias_norm.get(normalize_name(raw), raw)

    def add_node(node_id: str, node_type: str, display: str, source: str, confidence: float, props: Dict[str, Any] | None = None):
        nodes.append({"node_id": node_id, "node_type": node_type, "display_name": display, "source": source, "confidence": confidence, "properties": json_dumps(jsonable(props or {}))})

    def add_edge(from_node: str, to_node: str, edge_type: str, source: str, confidence: float, env: str = "", first_seen: str = "", last_seen: str = "", props: Dict[str, Any] | None = None, ev_refs: List[str] | None = None, edge_id_override: str | None = None):
        edge_id = edge_id_override or stable_id(from_node, to_node, edge_type, source, env)
        edges.append({"edge_id": edge_id, "from_node": from_node, "to_node": to_node, "edge_type": edge_type, "env": env, "source": source, "first_seen": first_seen, "last_seen": last_seen, "confidence": confidence, "evidence_refs": json_dumps(ev_refs or [edge_id]), "properties": json_dumps(jsonable(props or {}))})

    if repo_inventory is not None and not repo_inventory.empty:
        for _, r in repo_inventory.iterrows():
            repo_node = f"repo:{r['repo_id']}"
            add_node(repo_node, "repo", str(r["repo_id"]), "repo_inventory", 0.9, {"path": r.get("repo_path"), "type": r.get("probable_type")})
    if service_identity is not None and not service_identity.empty:
        for _, s in service_identity.iterrows():
            svc_node = f"service:{s['service_id']}"
            add_node(svc_node, "service", str(s.get("display_name") or s["service_id"]), "service_identity", float(s.get("confidence") or 0.6), {"datadog_service": s.get("datadog_service"), "owner_team": s.get("owner_team")})
            if str(s.get("repo_id") or ""):
                add_edge(f"repo:{s['repo_id']}", svc_node, "DEFINES_SERVICE", "service_identity", float(s.get("confidence") or 0.6))
            if str(s.get("owner_team") or ""):
                team_node = f"team:{s.get('owner_team')}"
                add_node(team_node, "team", str(s.get("owner_team")), "ownership", 0.6)
                add_edge(team_node, svc_node, "OWNS", "ownership", 0.6)
    if api_endpoints is not None and not api_endpoints.empty:
        for _, ep in api_endpoints.iterrows():
            svc = f"service:{ep['service_id']}"
            eid = f"endpoint:{ep['service_id']}:{ep.get('method')}:{ep.get('path')}"
            add_node(eid, "endpoint", f"{ep.get('method')} {ep.get('path')}", str(ep.get("source")), float(ep.get("confidence") or 0.5), {"operation_id": ep.get("operation_id"), "source_file": ep.get("source_file")})
            add_edge(svc, eid, "EXPOSES_ENDPOINT", str(ep.get("source")), float(ep.get("confidence") or 0.5))
    if static_edges is not None and not static_edges.empty:
        for _, e in static_edges.iterrows():
            from_node = canon(str(e.get("from_entity") or ""))
            if from_node.startswith("repo:"):
                # Repo-scoped static edge; keep as repo unless identity maps it later.
                pass
            to_node = canon(str(e.get("to_entity") or ""))
            ev_id = stable_id("evidence", e.get("edge_id"), e.get("file_path"), e.get("raw_match"))
            evidence.append({"evidence_id": ev_id, "evidence_type": "static", "source_system": e.get("source"), "source_ref": e.get("edge_id"), "repo_id": e.get("repo_id"), "file_path": e.get("file_path"), "line_start": e.get("line_start"), "line_end": e.get("line_end"), "raw_excerpt": e.get("raw_match"), "observed_at": now_iso(), "confidence": e.get("confidence")})
            add_edge(from_node, to_node, str(e.get("edge_type")), str(e.get("source")), float(e.get("confidence") or 0.3), ev_refs=[ev_id], props={"file_path": e.get("file_path"), "line": e.get("line_start")})
    if runtime_service_edges is not None and not runtime_service_edges.empty:
        for _, e in runtime_service_edges.iterrows():
            from_node = canon(str(e.get("from_service") or ""))
            if not from_node.startswith("service:"):
                from_node = f"service:{from_node}"
            to_raw = str(e.get("to_entity") or "")
            to_node = canon(to_raw)
            if not any(to_node.startswith(p) for p in ["service:", "bq_table:", "topic:", "database:", "host:"]):
                typ = str(e.get("to_type") or "entity")
                to_node = f"{typ}:{to_node}"
            ev_id = stable_id("runtime_evidence", e.get("edge_id"), e.get("source"))
            evidence.append({"evidence_id": ev_id, "evidence_type": "runtime", "source_system": e.get("source"), "source_ref": e.get("edge_id"), "repo_id": "", "file_path": "", "line_start": None, "line_end": None, "raw_excerpt": json_dumps({"count": e.get("count"), "p95_ms": e.get("p95_ms"), "error_rate": e.get("error_rate")}), "observed_at": str(e.get("last_seen") or now_iso()), "confidence": e.get("confidence")})
            add_edge(from_node, to_node, str(e.get("edge_type") or "CALLS"), str(e.get("source")), float(e.get("confidence") or 0.8), env=str(e.get("env") or ""), first_seen=str(e.get("first_seen") or ""), last_seen=str(e.get("last_seen") or ""), props={"count": e.get("count"), "p95_ms": e.get("p95_ms"), "error_rate": e.get("error_rate")}, ev_refs=[ev_id])
    if bq_usage is not None and not bq_usage.empty:
        for _, u in bq_usage.iterrows():
            if not str(u.get("referenced_table") or ""):
                continue
            principal = str(u.get("service_account") or u.get("principal_email") or "")
            from_node = canon(principal)
            if not from_node.startswith("service:"):
                from_node = f"principal:{principal}"
            table_node = f"bq_table:{u.get('referenced_table')}"
            add_node(table_node, "bq_table", str(u.get("referenced_table")), "bigquery", 0.8)
            ev_id = stable_id("bq_evidence", principal, u.get("referenced_table"), u.get("query_hash"))
            evidence.append({"evidence_id": ev_id, "evidence_type": "bigquery", "source_system": u.get("source"), "source_ref": u.get("query_hash"), "repo_id": "", "file_path": "", "line_start": None, "line_end": None, "raw_excerpt": json_dumps({"jobs": u.get("job_count"), "bytes": u.get("total_bytes_processed")}), "observed_at": str(u.get("last_seen") or ""), "confidence": u.get("confidence")})
            add_edge(from_node, table_node, "READS_TABLE", "bigquery", float(u.get("confidence") or 0.8), last_seen=str(u.get("last_seen") or ""), props={"job_count": u.get("job_count"), "total_bytes_processed": u.get("total_bytes_processed")}, ev_refs=[ev_id])
            if str(u.get("destination_table") or ""):
                dest_node = f"bq_table:{u.get('destination_table')}"
                add_node(dest_node, "bq_table", str(u.get("destination_table")), "bigquery", 0.8)
                add_edge(table_node, dest_node, "LINEAGE_READS_TO_WRITES", "bigquery", 0.75, ev_refs=[ev_id])
    if code_graph_symbols is not None and not code_graph_symbols.empty:
        for _, s in code_graph_symbols.iterrows():
            symbol_id = str(s.get("symbol_id") or "")
            if not symbol_id:
                continue
            display = str(s.get("name") or s.get("rel_path") or symbol_id)
            add_node(
                symbol_id,
                "code_symbol",
                display,
                "cgc",
                0.85,
                {
                    "repo_id": s.get("repo_id"),
                    "symbol_type": s.get("node_type"),
                    "file_path": s.get("file_path"),
                    "rel_path": s.get("rel_path"),
                    "line_start": s.get("line_start"),
                    "line_end": s.get("line_end"),
                    "language": s.get("language"),
                    "signature": s.get("signature"),
                },
            )
            if str(s.get("repo_id") or ""):
                add_edge(f"repo:{s.get('repo_id')}", symbol_id, "CONTAINS_CODE_SYMBOL", "cgc", 0.85)
    if code_graph_edges is not None and not code_graph_edges.empty:
        for _, e in code_graph_edges.iterrows():
            from_node = str(e.get("from_symbol_id") or "")
            to_node = str(e.get("to_symbol_id") or "")
            if not from_node or not to_node:
                continue
            ev_id = stable_id("cgc_evidence", e.get("edge_id"), e.get("file_path"), e.get("line_start"))
            evidence.append({
                "evidence_id": ev_id,
                "evidence_type": "code_graph",
                "source_system": "cgc",
                "source_ref": e.get("edge_id"),
                "repo_id": e.get("repo_id"),
                "file_path": e.get("file_path"),
                "line_start": e.get("line_start"),
                "line_end": e.get("line_start"),
                "raw_excerpt": e.get("properties"),
                "observed_at": now_iso(),
                "confidence": e.get("confidence"),
            })
            add_edge(
                from_node,
                to_node,
                str(e.get("edge_type") or "CODE_RELATES_TO"),
                "cgc",
                float(e.get("confidence") or 0.85),
                props={"repo_id": e.get("repo_id"), "file_path": e.get("file_path"), "rel_path": e.get("rel_path"), "line_start": e.get("line_start")},
                ev_refs=[ev_id],
                edge_id_override=str(e.get("edge_id") or ""),
            )
    if git_commits is not None and not git_commits.empty:
        for _, c in git_commits.iterrows():
            sha = str(c.get("commit_sha") or "")
            repo_id = str(c.get("repo_id") or "")
            if not sha:
                continue
            commit_node = f"commit:{sha}"
            author = str(c.get("author_email") or c.get("author_name") or "")
            add_node(commit_node, "commit", sha[:12], "git_history", 0.85, {
                "repo_id": repo_id,
                "subject": c.get("subject"),
                "commit_time": c.get("commit_time"),
                "is_revert": c.get("is_revert"),
                "is_hotfix": c.get("is_hotfix"),
            })
            if repo_id:
                add_edge(f"repo:{repo_id}", commit_node, "HAS_COMMIT", "git_history", 0.75, last_seen=str(c.get("commit_time") or ""))
            if author:
                person_node = f"person:{author}"
                add_node(person_node, "person", author, "git_history", 0.65)
                add_edge(person_node, commit_node, "AUTHORED", "git_history", 0.8, last_seen=str(c.get("commit_time") or ""))
            target = str(c.get("reverts_commit_sha") or "")
            if target:
                add_edge(commit_node, f"commit:{target}", "REVERTS", "git_history", 0.9, last_seen=str(c.get("commit_time") or ""))
    if git_file_changes is not None and not git_file_changes.empty:
        for _, fc in git_file_changes.iterrows():
            repo_id = str(fc.get("repo_id") or "")
            path = str(fc.get("new_path") or fc.get("old_path") or "")
            sha = str(fc.get("commit_sha") or "")
            if not repo_id or not path or not sha:
                continue
            file_node = f"file:{repo_id}:{path}"
            add_node(file_node, "file", path, "git_history", 0.65, {"repo_id": repo_id, "file_category": fc.get("file_category")})
            add_edge(f"commit:{sha}", file_node, "TOUCHED_FILE", "git_history", 0.75, props={
                "change_type": fc.get("change_type"),
                "added_lines": fc.get("added_lines"),
                "deleted_lines": fc.get("deleted_lines"),
                "is_rename": fc.get("is_rename"),
            })
    if git_semantic_changes is not None and not git_semantic_changes.empty:
        for _, sc in git_semantic_changes.iterrows():
            entity_type = str(sc.get("entity_type") or "change_entity")
            entity_id = str(sc.get("entity_id") or "")
            sha = str(sc.get("commit_sha") or "")
            if not entity_id or not sha:
                continue
            node_id = f"{entity_type}:{entity_id}"
            add_node(node_id, entity_type, entity_id, "git_semantic_change", float(sc.get("confidence") or 0.45), {
                "repo_id": sc.get("repo_id"),
                "breaking_risk": sc.get("breaking_risk"),
            })
            add_edge(f"commit:{sha}", node_id, str(sc.get("change_type") or "CHANGED"), "git_semantic_change", float(sc.get("confidence") or 0.45), props={
                "before": sc.get("before_value"),
                "after": sc.get("after_value"),
                "breaking_risk": sc.get("breaking_risk"),
                "evidence_id": sc.get("evidence_id"),
            })
    if git_cochange_edges is not None and not git_cochange_edges.empty:
        for _, ce in git_cochange_edges.iterrows():
            from_node = str(ce.get("from_entity") or "")
            to_node = str(ce.get("to_entity") or "")
            if not from_node or not to_node:
                continue
            add_edge(from_node, to_node, "CO_CHANGED_WITH", "git_history", float(ce.get("confidence") or 0.3), last_seen=str(ce.get("last_cochanged_at") or ""), props={
                "same_commit_count": ce.get("same_commit_count"),
                "same_jira_count": ce.get("same_jira_count"),
                "window_days": ce.get("window_days"),
                "entity_type": ce.get("entity_type"),
            })
    known_nodes = {str(n.get("node_id") or "") for n in nodes}
    for e in list(edges):
        for endpoint_key in ["from_node", "to_node"]:
            node_id = str(e.get(endpoint_key) or "")
            if not node_id or node_id in known_nodes:
                continue
            add_node(node_id, infer_node_type(node_id), node_id, "edge_endpoint", 0.3, {"inferred_from_edge": e.get("edge_id")})
            known_nodes.add(node_id)
    return dedupe(pd.DataFrame(nodes), ["node_id"]), dedupe(pd.DataFrame(edges), ["edge_id"]), dedupe(pd.DataFrame(evidence), ["evidence_id"])


def dedupe(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    if df.empty:
        return df
    return df.drop_duplicates(subset=keys, keep="first")


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [jsonable(v) for v in value]
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def infer_node_type(node_id: str) -> str:
    if ":" in node_id:
        return node_id.split(":", 1)[0]
    return "entity"
