from __future__ import annotations

import re
from typing import Any, Dict, List

import pandas as pd

from corp_kb.utils import stable_id, json_dumps


def build_capabilities(api_endpoints: pd.DataFrame, bq_usage: pd.DataFrame, service_identity: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    owner_by_service = {str(s.get("service_id")): str(s.get("owner_team") or "") for _, s in service_identity.iterrows()} if service_identity is not None and not service_identity.empty else {}
    if api_endpoints is not None and not api_endpoints.empty:
        for _, ep in api_endpoints.iterrows():
            terms = extract_terms(" ".join(str(ep.get(k) or "") for k in ["path", "operation_id", "request_schema", "response_schema"]))
            service = str(ep.get("service_id") or "")
            capability = " ".join(terms[:6]) or str(ep.get("path") or "")
            rows.append({
                "capability_id": stable_id("api_cap", service, ep.get("method"), ep.get("path")),
                "provider_entity": f"service:{service}",
                "capability_name": capability,
                "fields": json_dumps(terms[:50]),
                "access_method": str(ep.get("method") or "API"),
                "docs_url": str(ep.get("source_file") or ""),
                "owner_team": owner_by_service.get(service, ""),
                "usage_count_30d": None,
                "confidence": float(ep.get("confidence") or 0.6),
                "evidence_refs": json_dumps([ep.get("endpoint_id")]),
            })
    if bq_usage is not None and not bq_usage.empty:
        agg = bq_usage.groupby("referenced_table", dropna=False).agg(job_count=("job_count", "sum"), last_seen=("last_seen", "max")).reset_index()
        for _, u in agg.iterrows():
            table = str(u.get("referenced_table") or "")
            if not table:
                continue
            terms = extract_terms(table)
            rows.append({
                "capability_id": stable_id("bq_cap", table),
                "provider_entity": f"bq_table:{table}",
                "capability_name": " ".join(terms[:6]) or table,
                "fields": json_dumps(terms[:50]),
                "access_method": "BigQuery",
                "docs_url": table,
                "owner_team": "",
                "usage_count_30d": int(u.get("job_count") or 0),
                "confidence": 0.55,
                "evidence_refs": json_dumps([table]),
            })
    return pd.DataFrame(rows)


def extract_terms(s: str) -> List[str]:
    s = re.sub(r"[^A-Za-z0-9_./-]+", " ", s)
    parts = []
    for p in re.split(r"[\s/._:-]+", s):
        p = p.strip().lower()
        if len(p) > 2 and p not in {"api", "get", "post", "put", "the", "and", "for", "with", "http", "https", "com"}:
            parts.append(p)
    # preserve order unique
    out = []
    seen = set()
    for p in parts:
        if p not in seen:
            out.append(p)
            seen.add(p)
    return out
