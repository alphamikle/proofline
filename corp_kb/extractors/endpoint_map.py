from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd

from corp_kb.utils import json_dumps, stable_id


def build_endpoint_dependency_map(api_endpoints: pd.DataFrame, runtime_endpoint_edges: pd.DataFrame, static_edges: pd.DataFrame, service_identity: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    if api_endpoints is None or api_endpoints.empty:
        return pd.DataFrame(rows)
    # Runtime endpoint map.
    if runtime_endpoint_edges is not None and not runtime_endpoint_edges.empty:
        for _, e in runtime_endpoint_edges.iterrows():
            service_id = str(e.get("service_id") or "")
            path = str(e.get("path") or e.get("endpoint_key") or "")
            method = str(e.get("method") or "")
            endpoint_id = f"endpoint:{service_id}:{method}:{path}"
            rows.append({
                "service_id": service_id,
                "endpoint_id": endpoint_id,
                "method": method,
                "path": path,
                "downstream_entity": str(e.get("downstream_entity") or ""),
                "downstream_type": str(e.get("downstream_type") or "entity"),
                "dependency_kind": str(e.get("dependency_kind") or "RUNTIME_DEPENDENCY"),
                "env": str(e.get("env") or ""),
                "runtime_count_7d": int(e.get("count") or 0) if int(e.get("window_days") or 0) <= 7 else None,
                "runtime_count_30d": int(e.get("count") or 0) if int(e.get("window_days") or 0) <= 30 else None,
                "p95_ms": e.get("p95_ms"),
                "error_rate": e.get("error_rate"),
                "static_evidence_count": 0,
                "runtime_evidence_count": 1,
                "sources": json_dumps([e.get("source")]),
                "confidence": float(e.get("confidence") or 0.75),
                "evidence_refs": str(e.get("evidence_refs") or "[]"),
            })
    # Static-only fallback: service-level static dependencies are associated with all endpoints with lower confidence.
    # This is intentionally conservative: it tells the agent static dependency exists but does not overclaim exact endpoint usage.
    if static_edges is not None and not static_edges.empty:
        service_by_repo = {str(s.get("repo_id")): str(s.get("service_id")) for _, s in service_identity.iterrows()} if service_identity is not None and not service_identity.empty else {}
        for _, se in static_edges.iterrows():
            repo_id = str(se.get("repo_id") or "")
            service_id = service_by_repo.get(repo_id, repo_id)
            if not service_id:
                continue
            eps = api_endpoints[api_endpoints["service_id"] == service_id]
            if eps.empty:
                continue
            # Only attach high-signal static edges to endpoints as service-level/static-only possible dependencies.
            if str(se.get("edge_type")) not in {"REFERENCES_URL", "USES_CONFIG_KEY", "REFERENCES_HOST", "REFERENCES_BQ_TABLE", "REFERENCES_TOPIC"}:
                continue
            for _, ep in eps.iterrows():
                endpoint_id = f"endpoint:{service_id}:{ep.get('method')}:{ep.get('path')}"
                rows.append({
                    "service_id": service_id,
                    "endpoint_id": endpoint_id,
                    "method": str(ep.get("method") or ""),
                    "path": str(ep.get("path") or ""),
                    "downstream_entity": str(se.get("to_entity") or ""),
                    "downstream_type": infer_type(str(se.get("to_entity") or "")),
                    "dependency_kind": "STATIC_SERVICE_LEVEL_POSSIBLE_DEPENDENCY",
                    "env": "",
                    "runtime_count_7d": None,
                    "runtime_count_30d": None,
                    "p95_ms": None,
                    "error_rate": None,
                    "static_evidence_count": 1,
                    "runtime_evidence_count": 0,
                    "sources": json_dumps([se.get("source")]),
                    "confidence": min(float(se.get("confidence") or 0.3), 0.4),
                    "evidence_refs": json_dumps([se.get("edge_id")]),
                })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # Merge duplicates; keep highest confidence and aggregate counts.
    agg = df.groupby(["service_id", "endpoint_id", "method", "path", "downstream_entity", "downstream_type", "dependency_kind", "env"], dropna=False).agg(
        runtime_count_7d=("runtime_count_7d", "sum"),
        runtime_count_30d=("runtime_count_30d", "sum"),
        p95_ms=("p95_ms", "max"),
        error_rate=("error_rate", "max"),
        static_evidence_count=("static_evidence_count", "sum"),
        runtime_evidence_count=("runtime_evidence_count", "sum"),
        sources=("sources", lambda x: json_dumps(sorted(set(sum([safe_list(v) for v in x], []))))),
        confidence=("confidence", "max"),
        evidence_refs=("evidence_refs", lambda x: json_dumps(sorted(set(sum([safe_list(v) for v in x], []))))),
    ).reset_index()
    return agg


def safe_list(s):
    import orjson
    if s is None or s == "":
        return []
    if isinstance(s, list):
        return s
    try:
        return orjson.loads(str(s))
    except Exception:
        return [str(s)]


def infer_type(entity: str) -> str:
    if entity.startswith("bq_table:"):
        return "bq_table"
    if entity.startswith("topic:"):
        return "topic"
    if entity.startswith("host:") or entity.startswith("url:"):
        return "host"
    if entity.startswith("package:"):
        return "package"
    return "entity"
