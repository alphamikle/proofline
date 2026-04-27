from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd


def build_compatibility_index(api_endpoints: pd.DataFrame, static_edges: pd.DataFrame, runtime_service_edges: pd.DataFrame) -> pd.DataFrame:
    # This table is intentionally generic: it lists entities whose changes require compatibility checks.
    rows: List[Dict[str, Any]] = []
    if api_endpoints is not None and not api_endpoints.empty:
        for _, ep in api_endpoints.iterrows():
            rows.append({
                "entity_id": f"endpoint:{ep.get('service_id')}:{ep.get('method')}:{ep.get('path')}",
                "entity_type": "endpoint",
                "service_id": ep.get("service_id"),
                "risk_kind": "API_CONTRACT_CHANGE",
                "checklist": "removed endpoint; changed request required fields; changed response fields; enum/type/nullability changes",
                "confidence": ep.get("confidence") or 0.7,
            })
    if static_edges is not None and not static_edges.empty:
        for _, e in static_edges.iterrows():
            if str(e.get("edge_type")) in {"REFERENCES_TOPIC", "REFERENCES_BQ_TABLE", "DEPENDS_ON_PACKAGE"}:
                rows.append({
                    "entity_id": e.get("to_entity"),
                    "entity_type": str(e.get("edge_type")),
                    "service_id": e.get("from_entity"),
                    "risk_kind": "STATIC_DEPENDENCY_CHANGE",
                    "checklist": "consumer compatibility; schema evolution; package versioning; migration safety",
                    "confidence": e.get("confidence") or 0.4,
                })
    return pd.DataFrame(rows)
