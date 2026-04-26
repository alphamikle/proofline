from __future__ import annotations

from typing import Any, Dict, List, Tuple

import pandas as pd

from corp_kb.extractors.entity_resolution import canonicalize
from corp_kb.utils import stable_id, json_dumps, now_iso


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
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    nodes: List[Dict[str, Any]] = []
    edges: List[Dict[str, Any]] = []
    evidence: List[Dict[str, Any]] = []

    def add_node(node_id: str, node_type: str, display: str, source: str, confidence: float, props: Dict[str, Any] | None = None):
        nodes.append({"node_id": node_id, "node_type": node_type, "display_name": display, "source": source, "confidence": confidence, "properties": json_dumps(props or {})})

    def add_edge(from_node: str, to_node: str, edge_type: str, source: str, confidence: float, env: str = "", first_seen: str = "", last_seen: str = "", props: Dict[str, Any] | None = None, ev_refs: List[str] | None = None):
        edge_id = stable_id(from_node, to_node, edge_type, source, env)
        edges.append({"edge_id": edge_id, "from_node": from_node, "to_node": to_node, "edge_type": edge_type, "env": env, "source": source, "first_seen": first_seen, "last_seen": last_seen, "confidence": confidence, "evidence_refs": json_dumps(ev_refs or [edge_id]), "properties": json_dumps(props or {})})

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
            from_node = canonicalize(str(e.get("from_entity") or ""), aliases)
            if from_node.startswith("repo:"):
                # Repo-scoped static edge; keep as repo unless identity maps it later.
                pass
            to_node = canonicalize(str(e.get("to_entity") or ""), aliases)
            ev_id = stable_id("evidence", e.get("edge_id"), e.get("file_path"), e.get("raw_match"))
            evidence.append({"evidence_id": ev_id, "evidence_type": "static", "source_system": e.get("source"), "source_ref": e.get("edge_id"), "repo_id": e.get("repo_id"), "file_path": e.get("file_path"), "line_start": e.get("line_start"), "line_end": e.get("line_end"), "raw_excerpt": e.get("raw_match"), "observed_at": now_iso(), "confidence": e.get("confidence")})
            add_edge(from_node, to_node, str(e.get("edge_type")), str(e.get("source")), float(e.get("confidence") or 0.3), ev_refs=[ev_id], props={"file_path": e.get("file_path"), "line": e.get("line_start")})
    if runtime_service_edges is not None and not runtime_service_edges.empty:
        for _, e in runtime_service_edges.iterrows():
            from_node = canonicalize(str(e.get("from_service") or ""), aliases)
            if not from_node.startswith("service:"):
                from_node = f"service:{from_node}"
            to_raw = str(e.get("to_entity") or "")
            to_node = canonicalize(to_raw, aliases)
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
            from_node = canonicalize(principal, aliases)
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
    return dedupe(pd.DataFrame(nodes), ["node_id"]), dedupe(pd.DataFrame(edges), ["edge_id"]), dedupe(pd.DataFrame(evidence), ["evidence_id"])


def dedupe(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    if df.empty:
        return df
    return df.drop_duplicates(subset=keys, keep="first")
