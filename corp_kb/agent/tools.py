from __future__ import annotations

import re
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from corp_kb.storage import KB
from corp_kb.utils import normalize_name, json_dumps


class KBTools:
    def __init__(self, kb: KB, sqlite_fts_path: str | Path | None = None):
        self.kb = kb
        self.sqlite_fts_path = Path(sqlite_fts_path) if sqlite_fts_path else None

    def resolve_project(self, name: str) -> Dict[str, Any]:
        q = name.strip()
        norm = normalize_name(q)
        # exact alias
        aliases = self.kb.query_df("SELECT * FROM entity_aliases")
        svc = pd.DataFrame()
        if not aliases.empty:
            hits = aliases[(aliases["alias"].str.lower() == q.lower()) | (aliases["alias"].apply(lambda x: normalize_name(str(x)) == norm))]
            if not hits.empty:
                sid = str(hits.iloc[0]["canonical_id"]).replace("service:", "")
                svc = self.kb.query_df("SELECT * FROM service_identity WHERE service_id = ?", [sid])
        if svc.empty:
            svc = self.kb.query_df(
                """
                SELECT * FROM service_identity
                WHERE lower(service_id) = lower(?) OR lower(display_name) = lower(?) OR lower(repo_id) = lower(?) OR lower(datadog_service) = lower(?)
                LIMIT 1
                """,
                [norm, q, q, q],
            )
        if svc.empty:
            svc = self.kb.query_df(
                """
                SELECT * FROM service_identity
                WHERE service_id ILIKE ? OR display_name ILIKE ? OR repo_id ILIKE ? OR datadog_service ILIKE ?
                ORDER BY confidence DESC LIMIT 5
                """,
                [f"%{norm}%", f"%{q}%", f"%{q}%", f"%{q}%"],
            )
        if svc.empty:
            return {"query": q, "found": False, "candidates": []}
        row = svc.iloc[0].to_dict()
        return {"query": q, "found": True, "service": row, "candidates": svc.head(5).to_dict("records")}

    def get_service_profile(self, service_id: str) -> Dict[str, Any]:
        sid = service_id.replace("service:", "")
        service = self.kb.query_df("SELECT * FROM service_identity WHERE service_id = ? LIMIT 1", [sid]).to_dict("records")
        endpoints = self.kb.query_df("SELECT method, path, source, source_file, confidence FROM api_endpoints WHERE service_id = ? ORDER BY method, path LIMIT 500", [sid]).to_dict("records")
        owners = self.kb.query_df("SELECT * FROM ownership WHERE entity_id = ? OR entity_id = ? LIMIT 20", [f"service:{sid}", f"repo:{sid}"]).to_dict("records")
        return {"service": service[0] if service else {"service_id": sid}, "endpoints": endpoints, "owners": owners}

    def get_service_dependencies(self, service_id: str, env: str = "prod", window_days: int = 30) -> List[Dict[str, Any]]:
        sid = service_id.replace("service:", "")
        names = [sid]
        si = self.kb.query_df("SELECT datadog_service, repo_id FROM service_identity WHERE service_id = ?", [sid])
        if not si.empty:
            for c in ["datadog_service", "repo_id"]:
                v = str(si.iloc[0].get(c) or "")
                if v:
                    names.append(v)
        rows = []
        for name in set(names):
            rows += self.kb.query_df(
                """
                SELECT * FROM runtime_service_edges
                WHERE from_service = ? AND (? = '' OR env = ? OR env = '')
                ORDER BY confidence DESC, count DESC NULLS LAST
                LIMIT 1000
                """,
                [name, env, env],
            ).to_dict("records")
        if not rows:
            rows = self.kb.query_df(
                """
                SELECT * FROM edges
                WHERE from_node IN (?, ?) AND edge_type IN ('OBSERVED_CALL','REFERENCES_URL','REFERENCES_HOST','USES_CONFIG_KEY','REFERENCES_BQ_TABLE','REFERENCES_TOPIC')
                ORDER BY confidence DESC LIMIT 1000
                """,
                [f"service:{sid}", f"repo:{sid}"],
            ).to_dict("records")
        return rows

    def get_service_dependents(self, service_id: str, env: str = "prod", window_days: int = 30) -> List[Dict[str, Any]]:
        sid = service_id.replace("service:", "")
        names = [sid]
        si = self.kb.query_df("SELECT datadog_service, repo_id FROM service_identity WHERE service_id = ?", [sid])
        if not si.empty:
            for c in ["datadog_service", "repo_id"]:
                v = str(si.iloc[0].get(c) or "")
                if v:
                    names.append(v)
        rows = []
        for name in set(names):
            rows += self.kb.query_df(
                """
                SELECT * FROM runtime_service_edges
                WHERE to_entity = ? AND (? = '' OR env = ? OR env = '')
                ORDER BY confidence DESC, count DESC NULLS LAST
                LIMIT 1000
                """,
                [name, env, env],
            ).to_dict("records")
        if not rows:
            rows = self.kb.query_df(
                """
                SELECT * FROM edges
                WHERE to_node IN (?, ?) ORDER BY confidence DESC LIMIT 1000
                """,
                [f"service:{sid}", sid],
            ).to_dict("records")
        return rows

    def get_endpoint_dependencies(self, service_id: str, env: str = "prod", window_days: int = 30) -> List[Dict[str, Any]]:
        sid = service_id.replace("service:", "")
        return self.kb.query_df(
            """
            SELECT * FROM endpoint_dependency_map
            WHERE service_id = ? AND (? = '' OR env = ? OR env = '')
            ORDER BY path, method, confidence DESC, runtime_count_30d DESC NULLS LAST
            LIMIT 5000
            """,
            [sid, env, env],
        ).to_dict("records")

    def get_bq_usage(self, service_id: str, window_days: int = 30) -> List[Dict[str, Any]]:
        sid = service_id.replace("service:", "")
        aliases = self.kb.query_df("SELECT alias FROM entity_aliases WHERE canonical_id = ? AND alias_type = 'service_account'", [f"service:{sid}"])
        accounts = aliases["alias"].tolist() if not aliases.empty else []
        if not accounts:
            return []
        placeholders = ",".join(["?"] * len(accounts))
        return self.kb.query_df(f"SELECT * FROM bq_table_usage WHERE service_account IN ({placeholders}) ORDER BY job_count DESC LIMIT 1000", accounts).to_dict("records")

    def search_capabilities(self, query: str, limit: int = 50) -> List[Dict[str, Any]]:
        terms = [t for t in re.split(r"\W+", query.lower()) if len(t) > 2]
        if not terms:
            terms = [query.lower()]
        where = " OR ".join(["lower(capability_name) LIKE ? OR lower(fields) LIKE ? OR lower(provider_entity) LIKE ?" for _ in terms])
        params = []
        for t in terms:
            like = f"%{t}%"
            params.extend([like, like, like])
        return self.kb.query_df(f"SELECT * FROM data_capabilities WHERE {where} ORDER BY confidence DESC, usage_count_30d DESC NULLS LAST LIMIT {int(limit)}", params).to_dict("records")

    def search_code(self, query: str, repo_id: str | None = None, limit: int = 25) -> List[Dict[str, Any]]:
        if not self.sqlite_fts_path or not self.sqlite_fts_path.exists():
            return []
        con = sqlite3.connect(str(self.sqlite_fts_path))
        con.row_factory = sqlite3.Row
        q = query.replace('"', ' ')
        try:
            if repo_id:
                rows = con.execute("SELECT * FROM chunks WHERE chunks MATCH ? AND repo_id = ? LIMIT ?", (q, repo_id, limit)).fetchall()
            else:
                rows = con.execute("SELECT * FROM chunks WHERE chunks MATCH ? LIMIT ?", (q, limit)).fetchall()
            return [dict(r) for r in rows]
        except Exception:
            return []
        finally:
            con.close()

    def get_evidence(self, evidence_ids: List[str]) -> List[Dict[str, Any]]:
        if not evidence_ids:
            return []
        placeholders = ",".join(["?"] * len(evidence_ids))
        return self.kb.query_df(f"SELECT * FROM evidence WHERE evidence_id IN ({placeholders})", evidence_ids).to_dict("records")
