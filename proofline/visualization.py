from __future__ import annotations

import json
import math
import os
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Iterable
from urllib.parse import parse_qs, urlparse

import pandas as pd

from proofline.utils import json_dumps, json_loads, normalize_name, now_iso, stable_id


def visualization_path(cfg: Dict[str, Any]) -> Path:
    viz = cfg.get("visualization", {}) or {}
    raw = str(viz.get("output_path") or "")
    if raw:
        path = Path(raw).expanduser()
        if path.is_absolute():
            return path
        config_dir = Path(str(cfg.get("_config_dir") or ".")).expanduser()
        return config_dir / path
    return Path(str(cfg.get("workspace") or "./data")) / "visualization" / "graph.json"


def build_visualization_artifacts(kb, cfg: Dict[str, Any]) -> pd.DataFrame:
    path = visualization_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        payload = build_visualization_payload(
            repo_inventory=kb.query_df("SELECT * FROM repo_inventory"),
            service_identity=kb.query_df("SELECT * FROM service_identity"),
            entity_aliases=kb.query_df("SELECT * FROM entity_aliases"),
            runtime_service_edges=kb.query_df("SELECT * FROM runtime_service_edges"),
            runtime_endpoint_edges=kb.query_df("SELECT * FROM runtime_endpoint_edges"),
            static_edges=kb.query_df("SELECT * FROM static_edges"),
            git_cochange_edges=kb.query_df("SELECT * FROM git_cochange_edges"),
            bq_table_usage=kb.query_df("SELECT * FROM bq_table_usage"),
            nodes=kb.query_df("SELECT * FROM nodes"),
            edges=kb.query_df("SELECT * FROM edges"),
        )
        path.write_text(json_dumps(payload), encoding="utf-8")
        total_nodes = sum(len(g.get("nodes", [])) for g in payload["projections"].values())
        total_edges = sum(len(g.get("edges", [])) for g in payload["projections"].values())
        return pd.DataFrame([{
            "exported_at": payload["metadata"]["generated_at"],
            "output_path": str(path),
            "projections": json_dumps(sorted(payload["projections"].keys())),
            "node_count": int(total_nodes),
            "edge_count": int(total_edges),
            "status": "ok",
            "details": "",
        }])
    except Exception as e:
        return pd.DataFrame([{
            "exported_at": now_iso(),
            "output_path": str(path),
            "projections": "[]",
            "node_count": 0,
            "edge_count": 0,
            "status": "error",
            "details": str(e)[:4000],
        }])


def build_visualization_payload(
    *,
    repo_inventory: pd.DataFrame,
    service_identity: pd.DataFrame,
    entity_aliases: pd.DataFrame,
    runtime_service_edges: pd.DataFrame,
    runtime_endpoint_edges: pd.DataFrame,
    static_edges: pd.DataFrame,
    git_cochange_edges: pd.DataFrame,
    bq_table_usage: pd.DataFrame,
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
) -> Dict[str, Any]:
    resolver = EntityResolver(repo_inventory, service_identity, entity_aliases)
    repo_graph = GraphProjection("repos")
    service_graph = GraphProjection("services")
    hybrid_graph = GraphProjection("hybrid")

    for repo in resolver.repos.values():
        repo_node = repo_visual_node(repo)
        repo_graph.add_node(repo_node)
        hybrid_graph.add_node(repo_node)

    for service in resolver.services.values():
        svc_node = service_visual_node(service)
        service_graph.add_node(svc_node)
        hybrid_graph.add_node(svc_node)
        repo_id = str(service.get("repo_id") or "")
        if repo_id:
            repo_node = f"repo:{repo_id}"
            service_node = f"service:{service['service_id']}"
            hybrid_graph.add_edge(repo_node, service_node, "DEFINES_SERVICE", "service_identity", 1.0, float(service.get("confidence") or 0.55), "service_identity")

    add_runtime_edges(runtime_service_edges, resolver, repo_graph, service_graph, hybrid_graph)
    add_endpoint_runtime_edges(runtime_endpoint_edges, resolver, repo_graph, service_graph, hybrid_graph)
    add_static_edges(static_edges, resolver, repo_graph, hybrid_graph)
    add_bq_edges(bq_table_usage, resolver, hybrid_graph)
    add_git_cochange_edges(git_cochange_edges, resolver, repo_graph)
    add_canonical_graph_edges(nodes, edges, resolver, hybrid_graph)

    projections = {
        "repos": repo_graph.to_payload(),
        "services": service_graph.to_payload(),
        "hybrid": hybrid_graph.to_payload(),
    }
    return {
        "metadata": {
            "generated_at": now_iso(),
            "projection_count": len(projections),
            "description": "Proofline visualization-ready repo/service graph projections.",
        },
        "projections": projections,
    }


class EntityResolver:
    def __init__(self, repo_inventory: pd.DataFrame, service_identity: pd.DataFrame, aliases: pd.DataFrame):
        self.repos: Dict[str, Dict[str, Any]] = {}
        self.services: Dict[str, Dict[str, Any]] = {}
        self.alias_exact: Dict[str, str] = {}
        self.alias_norm: Dict[str, str] = {}
        self.service_to_repo: Dict[str, str] = {}
        self.repo_to_service: Dict[str, str] = {}

        for row in _records(repo_inventory):
            repo_id = _s(row.get("repo_id"))
            if not repo_id:
                continue
            self.repos[repo_id] = row

        for row in _records(service_identity):
            service_id = normalize_name(_s(row.get("service_id"))) or _s(row.get("service_id"))
            if not service_id:
                continue
            row = dict(row)
            row["service_id"] = service_id
            self.services[service_id] = row
            repo_id = _s(row.get("repo_id"))
            if repo_id:
                self.service_to_repo[f"service:{service_id}"] = repo_id
                self.repo_to_service.setdefault(repo_id, service_id)
            self._add_alias(service_id, f"service:{service_id}")
            self._add_alias(row.get("display_name"), f"service:{service_id}")
            self._add_alias(row.get("datadog_service"), f"service:{service_id}")
            self._add_alias(repo_id, f"service:{service_id}")

        for row in _records(aliases):
            canonical = _s(row.get("canonical_id"))
            alias = _s(row.get("alias"))
            if canonical and alias:
                self._add_alias(alias, canonical)

    def _add_alias(self, alias: Any, canonical: str) -> None:
        alias_s = _s(alias)
        if not alias_s or not canonical:
            return
        self.alias_exact[alias_s.lower()] = canonical
        self.alias_norm[normalize_name(alias_s)] = canonical

    def resolve_service(self, raw: Any) -> str:
        value = _s(raw)
        if not value:
            return ""
        if value.startswith("service:"):
            sid = normalize_name(value.split(":", 1)[1])
            return f"service:{sid}" if sid else value
        hit = self.alias_exact.get(value.lower()) or self.alias_norm.get(normalize_name(value))
        if hit:
            return hit
        sid = normalize_name(value)
        return f"service:{sid}" if sid else value

    def repo_for_service_node(self, service_node: str) -> str:
        service_node = self.resolve_service(service_node)
        if service_node in self.service_to_repo:
            return self.service_to_repo[service_node]
        sid = service_node.split(":", 1)[1] if service_node.startswith("service:") else service_node
        service = self.services.get(sid)
        return _s(service.get("repo_id")) if service else ""

    def service_label(self, service_node: str) -> str:
        service_node = self.resolve_service(service_node)
        sid = service_node.split(":", 1)[1] if service_node.startswith("service:") else service_node
        service = self.services.get(sid)
        return _s(service.get("display_name")) if service else sid


class GraphProjection:
    def __init__(self, name: str):
        self.name = name
        self.nodes: Dict[str, Dict[str, Any]] = {}
        self.edges: Dict[str, Dict[str, Any]] = {}

    def add_node(self, node: Dict[str, Any]) -> None:
        node_id = _s(node.get("id"))
        if not node_id:
            return
        if node_id in self.nodes:
            existing = self.nodes[node_id]
            existing["size"] = max(float(existing.get("size") or 1), float(node.get("size") or 1))
            existing["confidence"] = max(float(existing.get("confidence") or 0), float(node.get("confidence") or 0))
            existing.setdefault("sources", [])
            for source in node.get("sources", []):
                if source not in existing["sources"]:
                    existing["sources"].append(source)
            existing.setdefault("properties", {}).update(node.get("properties") or {})
            return
        node = dict(node)
        node.setdefault("sources", [])
        node.setdefault("properties", {})
        self.nodes[node_id] = node

    def add_edge(
        self,
        source: str,
        target: str,
        edge_type: str,
        source_system: str,
        weight: float,
        confidence: float,
        evidence_ref: str = "",
        properties: Dict[str, Any] | None = None,
    ) -> None:
        if not source or not target or source == target:
            return
        edge_type = edge_type or "RELATED_TO"
        edge_id = stable_id(self.name, source, target, edge_type)
        edge = self.edges.get(edge_id)
        if edge is None:
            edge = {
                "id": edge_id,
                "source": source,
                "target": target,
                "type": edge_type,
                "weight": 0.0,
                "confidence": 0.0,
                "evidence_count": 0,
                "sources": [],
                "evidence_refs": [],
                "properties": {},
            }
            self.edges[edge_id] = edge
        edge["weight"] = round(float(edge["weight"]) + max(0.1, float(weight or 0.1)), 4)
        edge["confidence"] = max(float(edge["confidence"] or 0), float(confidence or 0))
        edge["evidence_count"] = int(edge["evidence_count"] or 0) + 1
        if source_system and source_system not in edge["sources"]:
            edge["sources"].append(source_system)
        if evidence_ref and evidence_ref not in edge["evidence_refs"]:
            edge["evidence_refs"].append(evidence_ref)
        if properties:
            edge["properties"].update({k: v for k, v in properties.items() if v not in (None, "")})

    def to_payload(self) -> Dict[str, Any]:
        nodes = list(self.nodes.values())
        edges = list(self.edges.values())
        degree: Dict[str, int] = {}
        for edge in edges:
            degree[edge["source"]] = degree.get(edge["source"], 0) + 1
            degree[edge["target"]] = degree.get(edge["target"], 0) + 1
        for node in nodes:
            node["degree"] = degree.get(node["id"], 0)
            node["size"] = round(max(float(node.get("size") or 4), 4 + math.log1p(node["degree"]) * 3), 3)
        nodes.sort(key=lambda n: (-int(n.get("degree") or 0), _s(n.get("label")).lower()))
        edges.sort(key=lambda e: (-float(e.get("weight") or 0), _s(e.get("type"))))
        return {"nodes": nodes, "edges": edges, "stats": {"nodes": len(nodes), "edges": len(edges)}}


def add_runtime_edges(runtime_edges: pd.DataFrame, resolver: EntityResolver, repo_graph: GraphProjection, service_graph: GraphProjection, hybrid_graph: GraphProjection) -> None:
    for row in _records(runtime_edges):
        src_service = resolver.resolve_service(row.get("from_service"))
        dst_service = resolver.resolve_service(row.get("to_entity"))
        if not src_service or not dst_service:
            continue
        src_label = resolver.service_label(src_service)
        dst_label = resolver.service_label(dst_service)
        service_graph.add_node(external_service_node(src_service, src_label, row.get("source")))
        service_graph.add_node(external_service_node(dst_service, dst_label, row.get("source")))
        hybrid_graph.add_node(external_service_node(src_service, src_label, row.get("source")))
        hybrid_graph.add_node(external_service_node(dst_service, dst_label, row.get("source")))
        count = _num(row.get("count"))
        confidence = _num(row.get("confidence"), 0.8)
        weight = max(1.0, math.log1p(count or 1)) * confidence
        props = {"count": count, "env": row.get("env"), "p95_ms": row.get("p95_ms"), "error_rate": row.get("error_rate")}
        service_graph.add_edge(src_service, dst_service, "RUNTIME_CALLS", _s(row.get("source")) or "runtime", weight, confidence, _s(row.get("edge_id")), props)
        hybrid_graph.add_edge(src_service, dst_service, "RUNTIME_CALLS", _s(row.get("source")) or "runtime", weight, confidence, _s(row.get("edge_id")), props)

        src_repo = resolver.repo_for_service_node(src_service)
        dst_repo = resolver.repo_for_service_node(dst_service)
        if src_repo and dst_repo:
            repo_graph.add_edge(f"repo:{src_repo}", f"repo:{dst_repo}", "RUNTIME_CALLS", _s(row.get("source")) or "runtime", weight, confidence, _s(row.get("edge_id")), props)


def add_endpoint_runtime_edges(runtime_edges: pd.DataFrame, resolver: EntityResolver, repo_graph: GraphProjection, service_graph: GraphProjection, hybrid_graph: GraphProjection) -> None:
    for row in _records(runtime_edges):
        src_service = resolver.resolve_service(row.get("service_id"))
        if not src_service:
            continue
        downstream = resolver.resolve_service(row.get("downstream_entity"))
        if downstream.startswith("service:"):
            count = _num(row.get("count"))
            confidence = _num(row.get("confidence"), 0.75)
            weight = max(1.0, math.log1p(count or 1)) * confidence
            props = {"count": count, "env": row.get("env"), "method": row.get("method"), "path": row.get("path"), "dependency_kind": row.get("dependency_kind")}
            service_graph.add_node(external_service_node(src_service, resolver.service_label(src_service), row.get("source")))
            service_graph.add_node(external_service_node(downstream, resolver.service_label(downstream), row.get("source")))
            hybrid_graph.add_node(external_service_node(src_service, resolver.service_label(src_service), row.get("source")))
            hybrid_graph.add_node(external_service_node(downstream, resolver.service_label(downstream), row.get("source")))
            service_graph.add_edge(src_service, downstream, "ENDPOINT_DEPENDS_ON", _s(row.get("source")) or "runtime", weight, confidence, _s(row.get("edge_id")), props)
            hybrid_graph.add_edge(src_service, downstream, "ENDPOINT_DEPENDS_ON", _s(row.get("source")) or "runtime", weight, confidence, _s(row.get("edge_id")), props)
            src_repo = resolver.repo_for_service_node(src_service)
            dst_repo = resolver.repo_for_service_node(downstream)
            if src_repo and dst_repo:
                repo_graph.add_edge(f"repo:{src_repo}", f"repo:{dst_repo}", "ENDPOINT_DEPENDS_ON", _s(row.get("source")) or "runtime", weight, confidence, _s(row.get("edge_id")), props)


def add_static_edges(static_edges: pd.DataFrame, resolver: EntityResolver, repo_graph: GraphProjection, hybrid_graph: GraphProjection) -> None:
    for row in _records(static_edges):
        source = _s(row.get("from_entity"))
        target = _s(row.get("to_entity"))
        if not source or not target:
            continue
        src_repo = source if source.startswith("repo:") else ""
        if not src_repo:
            continue
        target_node = target
        if target.startswith("service:"):
            target_node = resolver.resolve_service(target)
        elif target.startswith("repo:"):
            target_node = target
        hybrid_graph.add_node(external_node(target_node, target_node, _s(row.get("source")) or "static"))
        confidence = _num(row.get("confidence"), 0.35)
        weight = max(0.4, confidence)
        props = {"file_path": row.get("file_path"), "line": row.get("line_start"), "raw": row.get("raw_match")}
        hybrid_graph.add_edge(src_repo, target_node, _s(row.get("edge_type")) or "STATIC_REFERENCES", _s(row.get("source")) or "static", weight, confidence, _s(row.get("edge_id")), props)
        if target_node.startswith("repo:"):
            repo_graph.add_edge(src_repo, target_node, _s(row.get("edge_type")) or "STATIC_REFERENCES", _s(row.get("source")) or "static", weight, confidence, _s(row.get("edge_id")), props)


def add_bq_edges(bq_usage: pd.DataFrame, resolver: EntityResolver, hybrid_graph: GraphProjection) -> None:
    for row in _records(bq_usage):
        principal = _s(row.get("service_account") or row.get("principal_email"))
        table = _s(row.get("referenced_table"))
        if not principal or not table:
            continue
        src = resolver.resolve_service(principal)
        table_node = f"bq_table:{table}"
        hybrid_graph.add_node(external_service_node(src, resolver.service_label(src), "bigquery"))
        hybrid_graph.add_node(external_node(table_node, table, "bigquery", node_type="bq_table"))
        confidence = _num(row.get("confidence"), 0.7)
        weight = max(0.6, math.log1p(_num(row.get("job_count")) or 1) * confidence)
        hybrid_graph.add_edge(src, table_node, "READS_TABLE", "bigquery", weight, confidence, _s(row.get("query_hash")), {"job_count": row.get("job_count"), "last_seen": row.get("last_seen")})


def add_git_cochange_edges(cochange_edges: pd.DataFrame, resolver: EntityResolver, repo_graph: GraphProjection) -> None:
    repo_pairs: Dict[tuple[str, str], Dict[str, Any]] = {}
    for row in _records(cochange_edges):
        src_repo = repo_from_entity(row.get("from_entity"))
        dst_repo = repo_from_entity(row.get("to_entity"))
        if not src_repo or not dst_repo or src_repo == dst_repo:
            continue
        key = tuple(sorted([src_repo, dst_repo]))
        acc = repo_pairs.setdefault(key, {"count": 0, "confidence": 0.0, "last": ""})
        acc["count"] += int(_num(row.get("same_commit_count")) or 1)
        acc["confidence"] = max(float(acc["confidence"]), _num(row.get("confidence"), 0.3))
        acc["last"] = max(_s(acc.get("last")), _s(row.get("last_cochanged_at")))
    for (src_repo, dst_repo), acc in repo_pairs.items():
        src = f"repo:{src_repo}"
        dst = f"repo:{dst_repo}"
        if src not in repo_graph.nodes:
            repo_graph.add_node(repo_visual_node(resolver.repos.get(src_repo, {"repo_id": src_repo})))
        if dst not in repo_graph.nodes:
            repo_graph.add_node(repo_visual_node(resolver.repos.get(dst_repo, {"repo_id": dst_repo})))
        confidence = float(acc["confidence"])
        weight = max(0.2, math.log1p(float(acc["count"])) * confidence * 0.5)
        repo_graph.add_edge(src, dst, "CO_CHANGED_WITH", "git_history", weight, confidence, "", {"same_commit_count": acc["count"], "last_cochanged_at": acc["last"]})


def add_canonical_graph_edges(nodes: pd.DataFrame, edges: pd.DataFrame, resolver: EntityResolver, hybrid_graph: GraphProjection) -> None:
    allowed_node_types = {"team", "endpoint", "bq_table", "topic", "database", "host"}
    node_by_id = {_s(row.get("node_id")): row for row in _records(nodes)}
    for row in _records(nodes):
        node_id = _s(row.get("node_id"))
        node_type = _s(row.get("node_type"))
        if node_id and node_type in allowed_node_types:
            hybrid_graph.add_node(external_node(node_id, _s(row.get("display_name")) or node_id, _s(row.get("source")) or "graph", node_type=node_type, confidence=_num(row.get("confidence"), 0.4)))
    for row in _records(edges):
        src = _s(row.get("from_node"))
        dst = _s(row.get("to_node"))
        if not src or not dst:
            continue
        src_type = _s(node_by_id.get(src, {}).get("node_type"))
        dst_type = _s(node_by_id.get(dst, {}).get("node_type"))
        if src.startswith("code_symbol") or dst.startswith("code_symbol"):
            continue
        if src_type == "code_symbol" or dst_type == "code_symbol":
            continue
        if src not in hybrid_graph.nodes and not (src.startswith("repo:") or src.startswith("service:")):
            continue
        if dst not in hybrid_graph.nodes and not (dst.startswith("repo:") or dst.startswith("service:")):
            continue
        confidence = _num(row.get("confidence"), 0.4)
        hybrid_graph.add_edge(src, dst, _s(row.get("edge_type")) or "RELATED_TO", _s(row.get("source")) or "graph", confidence, confidence, _s(row.get("edge_id")), {"env": row.get("env")})


def repo_visual_node(row: Dict[str, Any]) -> Dict[str, Any]:
    repo_id = _s(row.get("repo_id"))
    return {
        "id": f"repo:{repo_id}",
        "label": repo_id,
        "type": "repo",
        "group": _s(row.get("probable_type")) or "repo",
        "size": max(5, math.log1p(_num(row.get("worktree_size_mb")) or _num(row.get("size_mb")) or 1) * 2),
        "confidence": 0.9,
        "sources": ["repo_inventory"],
        "properties": {
            "path": row.get("repo_path"),
            "language": row.get("primary_language"),
            "type": row.get("probable_type"),
            "size_mb": row.get("size_mb"),
        },
    }


def service_visual_node(row: Dict[str, Any]) -> Dict[str, Any]:
    service_id = _s(row.get("service_id"))
    return {
        "id": f"service:{service_id}",
        "label": _s(row.get("display_name")) or service_id,
        "type": "service",
        "group": _s(row.get("owner_team")) or _s(row.get("repo_id")) or "service",
        "size": 7,
        "confidence": _num(row.get("confidence"), 0.55),
        "sources": ["service_identity"],
        "properties": {
            "repo_id": row.get("repo_id"),
            "repo_path": row.get("repo_path"),
            "datadog_service": row.get("datadog_service"),
            "owner_team": row.get("owner_team"),
        },
    }


def external_service_node(node_id: str, label: str, source: Any) -> Dict[str, Any]:
    return {
        "id": node_id,
        "label": label or node_id,
        "type": "service",
        "group": "service",
        "size": 6,
        "confidence": 0.55,
        "sources": [_s(source) or "runtime"],
        "properties": {},
    }


def external_node(node_id: str, label: str, source: str, *, node_type: str | None = None, confidence: float = 0.35) -> Dict[str, Any]:
    typ = node_type or (node_id.split(":", 1)[0] if ":" in node_id else "entity")
    return {
        "id": node_id,
        "label": label.split(":", 1)[1] if label.startswith(f"{typ}:") else label,
        "type": typ,
        "group": typ,
        "size": 5,
        "confidence": confidence,
        "sources": [source],
        "properties": {},
    }


def repo_from_entity(value: Any) -> str:
    text = _s(value)
    if text.startswith("repo:"):
        return text.split(":", 1)[1]
    if text.startswith("file:"):
        parts = text.split(":", 2)
        return parts[1] if len(parts) > 1 else ""
    return ""


def _records(df: pd.DataFrame | None) -> Iterable[Dict[str, Any]]:
    if df is None or df.empty:
        return []
    return df.to_dict("records")


def _s(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return str(value).strip()


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


class VisualizationServer:
    def __init__(self, graph_path: Path):
        self.graph_path = graph_path

    def handler_class(self):
        graph_path = self.graph_path

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, fmt: str, *args: Any) -> None:
                return

            def do_GET(self) -> None:
                parsed = urlparse(self.path)
                if parsed.path in {"", "/"}:
                    self._send_text(INDEX_HTML, "text/html; charset=utf-8")
                    return
                if parsed.path == "/api/graph":
                    if not graph_path.exists():
                        self.send_error(404, f"Visualization graph not found: {graph_path}")
                        return
                    self._send_text(graph_path.read_text(encoding="utf-8"), "application/json; charset=utf-8")
                    return
                if parsed.path == "/api/projection":
                    query = parse_qs(parsed.query)
                    name = (query.get("name") or ["repos"])[0]
                    payload = json_loads(graph_path.read_text(encoding="utf-8")) if graph_path.exists() else {}
                    projection = (payload.get("projections") or {}).get(name)
                    if projection is None:
                        self.send_error(404, f"Unknown projection: {name}")
                        return
                    self._send_text(json_dumps(projection), "application/json; charset=utf-8")
                    return
                self.send_error(404)

            def _send_text(self, body: str, content_type: str) -> None:
                data = body.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(data)

        return Handler


def serve_visualization(cfg: Dict[str, Any], *, host: str, port: int, open_browser: bool = True, rebuild: bool = False) -> None:
    from proofline.storage import KB

    graph_path = visualization_path(cfg)
    if rebuild or not graph_path.exists():
        kb = KB(cfg["storage"]["duckdb_path"])
        try:
            result = build_visualization_artifacts(kb, cfg)
            kb.append_df("visualization_exports", result)
        finally:
            kb.close()

    server = ThreadingHTTPServer((host, port), VisualizationServer(graph_path).handler_class())
    url = f"http://{host}:{port}/"
    if open_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass
    print(f"Proofline visualization: {url}")
    print(f"Serving graph data: {graph_path}")
    print("Press Ctrl-C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Proofline Graph</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f7f7f4;
      --ink: #22252a;
      --muted: #696d73;
      --line: #d8d5cc;
      --panel: rgba(255,255,255,.9);
      --accent: #176d72;
      --accent-2: #a34b37;
      --accent-3: #5b6f2a;
      --accent-4: #7c5a9b;
    }
    * { box-sizing: border-box; }
    html, body { margin: 0; height: 100%; overflow: hidden; background: var(--bg); color: var(--ink); font: 13px/1.35 Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
    body { display: grid; grid-template-rows: auto 1fr; }
    header { height: 52px; display: flex; align-items: center; gap: 12px; padding: 8px 12px; border-bottom: 1px solid var(--line); background: var(--panel); backdrop-filter: blur(10px); }
    .brand { font-weight: 700; font-size: 15px; margin-right: 4px; white-space: nowrap; }
    select, input, button { height: 32px; border: 1px solid var(--line); background: #fff; color: var(--ink); border-radius: 6px; padding: 0 9px; font: inherit; }
    input[type="search"] { width: min(28vw, 360px); }
    input[type="range"] { width: 128px; padding: 0; accent-color: var(--accent); }
    button { cursor: pointer; }
    main { position: relative; min-height: 0; }
    canvas { position: absolute; inset: 0; width: 100%; height: 100%; display: block; }
    aside { position: absolute; top: 14px; right: 14px; width: min(360px, calc(100vw - 28px)); max-height: calc(100% - 28px); overflow: auto; background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 12px; box-shadow: 0 10px 30px rgba(0,0,0,.08); }
    aside[hidden] { display: none; }
    .stats { display: flex; gap: 12px; color: var(--muted); white-space: nowrap; margin-left: auto; }
    .row { display: flex; align-items: center; gap: 8px; }
    .label { color: var(--muted); }
    .title { font-weight: 700; font-size: 16px; margin-bottom: 6px; overflow-wrap: anywhere; }
    .kv { display: grid; grid-template-columns: 88px 1fr; gap: 6px 10px; margin-top: 10px; }
    .kv div:nth-child(odd) { color: var(--muted); }
    .pill { display: inline-flex; align-items: center; height: 22px; padding: 0 7px; border-radius: 999px; background: #ece8dd; margin: 2px 4px 2px 0; font-size: 12px; }
    @media (max-width: 760px) {
      header { height: auto; flex-wrap: wrap; }
      .stats { order: 5; width: 100%; margin-left: 0; }
      input[type="search"] { width: 100%; flex: 1 1 180px; }
    }
  </style>
</head>
<body>
  <header>
    <div class="brand">Proofline</div>
    <select id="projection" title="Projection"></select>
    <input id="search" type="search" placeholder="Search nodes" />
    <div class="row"><span class="label">Weight</span><input id="weight" type="range" min="0" max="10" step="0.1" value="0" /></div>
    <button id="reset" type="button">Reset</button>
    <div class="stats" id="stats"></div>
  </header>
  <main>
    <canvas id="graph"></canvas>
    <aside id="details" hidden></aside>
  </main>
  <script>
    const canvas = document.getElementById('graph');
    const ctx = canvas.getContext('2d');
    const projectionSelect = document.getElementById('projection');
    const searchInput = document.getElementById('search');
    const weightInput = document.getElementById('weight');
    const resetButton = document.getElementById('reset');
    const details = document.getElementById('details');
    const stats = document.getElementById('stats');
    const colors = { repo: '#176d72', service: '#a34b37', team: '#5b6f2a', endpoint: '#7c5a9b', bq_table: '#3769a3', package: '#7b6d21', host: '#6a6d75', topic: '#8a5b24', url: '#555d67', entity: '#4d6661' };
    let payload = null, graph = null, sim = null, selected = null, hovered = null, transform = { x: 0, y: 0, scale: 1 };
    let pointer = { x: 0, y: 0, down: false, moved: false, lastX: 0, lastY: 0 };

    fetch('/api/graph').then(r => r.json()).then(data => {
      payload = data;
      Object.keys(payload.projections).forEach(name => {
        const opt = document.createElement('option');
        opt.value = name;
        opt.textContent = name[0].toUpperCase() + name.slice(1);
        projectionSelect.appendChild(opt);
      });
      projectionSelect.value = payload.projections.services ? 'services' : Object.keys(payload.projections)[0];
      loadProjection();
    });

    function resize() {
      const dpr = window.devicePixelRatio || 1;
      canvas.width = Math.floor(canvas.clientWidth * dpr);
      canvas.height = Math.floor(canvas.clientHeight * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      if (graph) centerGraph(false);
    }
    window.addEventListener('resize', resize);

    function loadProjection() {
      const data = payload.projections[projectionSelect.value];
      const minWeight = Number(weightInput.value);
      const query = searchInput.value.trim().toLowerCase();
      const nodesById = new Map(data.nodes.map(n => [n.id, { ...n }]));
      let edges = data.edges.filter(e => (e.weight || 0) >= minWeight);
      if (query) {
        const hit = new Set(data.nodes.filter(n => (n.label || n.id).toLowerCase().includes(query)).map(n => n.id));
        edges = edges.filter(e => hit.has(e.source) || hit.has(e.target));
        for (const e of edges) { hit.add(e.source); hit.add(e.target); }
        for (const id of [...nodesById.keys()]) if (!hit.has(id)) nodesById.delete(id);
      } else {
        const connected = new Set();
        for (const e of edges) { connected.add(e.source); connected.add(e.target); }
        for (const id of [...nodesById.keys()]) if (!connected.has(id) && data.nodes.length > 100) nodesById.delete(id);
      }
      graph = { nodes: [...nodesById.values()], edges, nodesById };
      seedPositions();
      centerGraph(true);
      startSimulation();
      selected = null;
      details.hidden = true;
      stats.textContent = `${graph.nodes.length} nodes · ${graph.edges.length} edges`;
    }

    function seedPositions() {
      const w = canvas.clientWidth || 900, h = canvas.clientHeight || 600;
      graph.nodes.forEach((n, i) => {
        const angle = i * 2.399963;
        const radius = 30 + 8 * Math.sqrt(i);
        n.x = w / 2 + Math.cos(angle) * radius;
        n.y = h / 2 + Math.sin(angle) * radius;
        n.vx = 0; n.vy = 0;
      });
    }

    function startSimulation() {
      if (sim) cancelAnimationFrame(sim);
      let ticks = 0;
      const run = () => {
        tick(ticks++);
        draw();
        sim = ticks < 900 ? requestAnimationFrame(run) : null;
      };
      run();
    }

    function tick(t) {
      const nodes = graph.nodes, edges = graph.edges;
      const cx = canvas.clientWidth / 2, cy = canvas.clientHeight / 2;
      for (const e of edges) {
        const a = graph.nodesById.get(e.source), b = graph.nodesById.get(e.target);
        if (!a || !b) continue;
        const dx = b.x - a.x, dy = b.y - a.y;
        const dist = Math.max(1, Math.hypot(dx, dy));
        const desired = 70 + 120 / Math.max(1, e.weight || 1);
        const force = (dist - desired) * 0.0018;
        const fx = dx / dist * force, fy = dy / dist * force;
        a.vx += fx; a.vy += fy; b.vx -= fx; b.vy -= fy;
      }
      for (let i = 0; i < nodes.length; i++) {
        const a = nodes[i];
        for (let j = i + 1; j < nodes.length; j++) {
          const b = nodes[j];
          const dx = b.x - a.x, dy = b.y - a.y;
          const dist2 = Math.max(16, dx * dx + dy * dy);
          if (dist2 > 90000) continue;
          const force = 34 / dist2;
          a.vx -= dx * force; a.vy -= dy * force; b.vx += dx * force; b.vy += dy * force;
        }
      }
      for (const n of nodes) {
        n.vx += (cx - n.x) * 0.0007;
        n.vy += (cy - n.y) * 0.0007;
        n.vx *= 0.86; n.vy *= 0.86;
        n.x += n.vx; n.y += n.vy;
      }
    }

    function centerGraph(resetScale) {
      if (!graph || !graph.nodes.length) return;
      const xs = graph.nodes.map(n => n.x), ys = graph.nodes.map(n => n.y);
      const minX = Math.min(...xs), maxX = Math.max(...xs), minY = Math.min(...ys), maxY = Math.max(...ys);
      const scale = resetScale ? Math.min(1.8, Math.max(0.25, Math.min(canvas.clientWidth / Math.max(1, maxX - minX + 120), canvas.clientHeight / Math.max(1, maxY - minY + 120)))) : transform.scale;
      transform.scale = scale;
      transform.x = canvas.clientWidth / 2 - ((minX + maxX) / 2) * scale;
      transform.y = canvas.clientHeight / 2 - ((minY + maxY) / 2) * scale;
    }

    function draw() {
      if (!graph) return;
      ctx.clearRect(0, 0, canvas.clientWidth, canvas.clientHeight);
      ctx.save();
      ctx.translate(transform.x, transform.y);
      ctx.scale(transform.scale, transform.scale);
      const active = selected || hovered;
      const neighbors = active ? neighborSet(active.id) : null;
      ctx.lineCap = 'round';
      for (const e of graph.edges) {
        const a = graph.nodesById.get(e.source), b = graph.nodesById.get(e.target);
        if (!a || !b) continue;
        const on = !active || e.source === active.id || e.target === active.id;
        ctx.globalAlpha = on ? 0.55 : 0.08;
        ctx.strokeStyle = edgeColor(e);
        ctx.lineWidth = Math.min(7, 0.6 + Math.log1p(e.weight || 1));
        ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
      }
      ctx.globalAlpha = 1;
      for (const n of graph.nodes) {
        const on = !active || n.id === active.id || neighbors.has(n.id);
        ctx.globalAlpha = on ? 1 : 0.18;
        ctx.fillStyle = colors[n.type] || colors.entity;
        ctx.beginPath(); ctx.arc(n.x, n.y, radius(n), 0, Math.PI * 2); ctx.fill();
        if (n === selected) { ctx.strokeStyle = '#111'; ctx.lineWidth = 2.4 / transform.scale; ctx.stroke(); }
        if (n === hovered) { ctx.strokeStyle = '#fff'; ctx.lineWidth = 3 / transform.scale; ctx.stroke(); }
      }
      ctx.globalAlpha = 1;
      ctx.font = `${12 / transform.scale}px system-ui, sans-serif`;
      ctx.fillStyle = '#24272b';
      for (const n of graph.nodes) {
        if (radius(n) * transform.scale < 5 && n !== hovered && n !== selected) continue;
        if (active && n !== active && !neighbors.has(n.id)) continue;
        ctx.fillText(n.label || n.id, n.x + radius(n) + 3 / transform.scale, n.y + 4 / transform.scale);
      }
      ctx.restore();
    }

    function radius(n) { return Math.max(3, Math.min(18, n.size || 6)); }
    function edgeColor(e) {
      const s = (e.sources || [e.type])[0] || e.type;
      if (/datadog|runtime/i.test(s)) return '#a34b37';
      if (/git/i.test(s)) return '#5b6f2a';
      if (/bigquery/i.test(s)) return '#3769a3';
      if (/static|regex|package/i.test(s)) return '#7b6d21';
      return '#687078';
    }
    function neighborSet(id) {
      const set = new Set();
      for (const e of graph.edges) { if (e.source === id) set.add(e.target); if (e.target === id) set.add(e.source); }
      return set;
    }
    function screenToGraph(x, y) { return { x: (x - transform.x) / transform.scale, y: (y - transform.y) / transform.scale }; }
    function pick(x, y) {
      const p = screenToGraph(x, y);
      let best = null, bestD = Infinity;
      for (const n of graph.nodes) {
        const d = Math.hypot(n.x - p.x, n.y - p.y);
        if (d < radius(n) + 6 / transform.scale && d < bestD) { best = n; bestD = d; }
      }
      return best;
    }
    function showDetails(n) {
      if (!n) { details.hidden = true; return; }
      const related = graph.edges.filter(e => e.source === n.id || e.target === n.id).sort((a,b) => (b.weight || 0) - (a.weight || 0)).slice(0, 14);
      details.innerHTML = `<div class="title">${escapeHtml(n.label || n.id)}</div>
        <span class="pill">${escapeHtml(n.type || 'node')}</span><span class="pill">${escapeHtml(n.group || '')}</span>
        <div class="kv"><div>ID</div><div>${escapeHtml(n.id)}</div><div>Degree</div><div>${n.degree || 0}</div><div>Confidence</div><div>${Number(n.confidence || 0).toFixed(2)}</div></div>
        <div class="kv">${Object.entries(n.properties || {}).filter(([,v]) => v).slice(0, 8).map(([k,v]) => `<div>${escapeHtml(k)}</div><div>${escapeHtml(String(v))}</div>`).join('')}</div>
        <div class="title" style="margin-top:14px">Edges</div>
        ${related.map(e => `<div style="margin:8px 0"><b>${escapeHtml(e.type)}</b><br><span class="label">${escapeHtml(e.source === n.id ? e.target : e.source)}</span><br><span class="label">weight ${Number(e.weight || 0).toFixed(2)} · ${(e.sources || []).map(escapeHtml).join(', ')}</span></div>`).join('')}`;
      details.hidden = false;
    }
    function escapeHtml(s) { return String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])); }

    canvas.addEventListener('mousemove', e => {
      const r = canvas.getBoundingClientRect();
      if (pointer.down) {
        const dx = e.clientX - pointer.lastX, dy = e.clientY - pointer.lastY;
        transform.x += dx; transform.y += dy; pointer.lastX = e.clientX; pointer.lastY = e.clientY; pointer.moved = true; draw(); return;
      }
      hovered = pick(e.clientX - r.left, e.clientY - r.top);
      canvas.style.cursor = hovered ? 'pointer' : 'grab';
      draw();
    });
    canvas.addEventListener('mousedown', e => { pointer.down = true; pointer.moved = false; pointer.lastX = e.clientX; pointer.lastY = e.clientY; });
    window.addEventListener('mouseup', e => {
      if (!pointer.down) return;
      pointer.down = false;
      if (!pointer.moved) {
        const r = canvas.getBoundingClientRect();
        selected = pick(e.clientX - r.left, e.clientY - r.top);
        showDetails(selected);
      }
      draw();
    });
    canvas.addEventListener('wheel', e => {
      e.preventDefault();
      const r = canvas.getBoundingClientRect();
      const mx = e.clientX - r.left, my = e.clientY - r.top;
      const before = screenToGraph(mx, my);
      transform.scale *= Math.exp(-e.deltaY * 0.001);
      transform.scale = Math.max(0.08, Math.min(5, transform.scale));
      transform.x = mx - before.x * transform.scale;
      transform.y = my - before.y * transform.scale;
      draw();
    }, { passive: false });
    projectionSelect.addEventListener('change', loadProjection);
    searchInput.addEventListener('input', () => { clearTimeout(searchInput._t); searchInput._t = setTimeout(loadProjection, 120); });
    weightInput.addEventListener('input', loadProjection);
    resetButton.addEventListener('click', () => { searchInput.value = ''; weightInput.value = 0; loadProjection(); });
    resize();
  </script>
</body>
</html>
"""
