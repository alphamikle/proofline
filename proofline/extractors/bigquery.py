from __future__ import annotations

import os
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

import pandas as pd

from proofline.utils import json_dumps, stable_id


def _client(project_id: str | None):
    from google.cloud import bigquery
    return bigquery.Client(project=project_id or os.getenv("GOOGLE_CLOUD_PROJECT"))


def pull_bq_jobs(cfg: Dict[str, Any]) -> pd.DataFrame:
    bq = cfg.get("bigquery", {})
    rows: List[Dict[str, Any]] = []
    if not bq.get("enabled", True) or not bq.get("pull_jobs", True):
        return pd.DataFrame(rows)
    try:
        client = _client(bq.get("project_id"))
    except Exception as e:
        return pd.DataFrame([_error_row(str(e))])
    max_results = bq.get("max_results")
    for region in bq.get("regions", ["region-us"]):
        for days in bq.get("windows_days", [30]):
            limit_sql = f"LIMIT {int(max_results)}" if max_results else ""
            sql = f"""
            SELECT
              creation_time,
              project_id,
              job_id,
              user_email,
              SAFE_CAST(query_info.query_hashes.normalized_literals AS STRING) AS query_hash,
              total_bytes_processed,
              total_slot_ms,
              destination_table.project_id AS dest_project,
              destination_table.dataset_id AS dest_dataset,
              destination_table.table_id AS dest_table,
              ARRAY(
                SELECT AS STRUCT rt.project_id, rt.dataset_id, rt.table_id
                FROM UNNEST(referenced_tables) AS rt
              ) AS referenced_tables
            FROM `{region}`.INFORMATION_SCHEMA.JOBS_BY_ORGANIZATION
            WHERE creation_time >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL {int(days)} DAY)
              AND job_type = 'QUERY'
              AND state = 'DONE'
            {limit_sql}
            """
            try:
                df = client.query(sql).result().to_dataframe(create_bqstorage_client=False)
            except Exception as e:
                rows.append(_error_row(f"{region}: {e}"))
                continue
            for _, r in df.iterrows():
                refs = []
                for item in r.get("referenced_tables") or []:
                    try:
                        refs.append(f"{item['project_id']}.{item['dataset_id']}.{item['table_id']}")
                    except Exception:
                        pass
                dest = ""
                if r.get("dest_project") and r.get("dest_dataset") and r.get("dest_table"):
                    dest = f"{r.get('dest_project')}.{r.get('dest_dataset')}.{r.get('dest_table')}"
                rows.append({
                    "job_id": str(r.get("job_id") or ""),
                    "project_id": str(r.get("project_id") or ""),
                    "user_email": str(r.get("user_email") or ""),
                    "creation_time": str(r.get("creation_time") or ""),
                    "query_hash": str(r.get("query_hash") or ""),
                    "referenced_tables": json_dumps(refs),
                    "destination_table": dest,
                    "total_bytes_processed": int(r.get("total_bytes_processed") or 0),
                    "total_slot_ms": int(r.get("total_slot_ms") or 0),
                    "raw": json_dumps({"region": region, "window_days": days}),
                })
    return pd.DataFrame(rows)


def _error_row(error: str) -> Dict[str, Any]:
    return {
        "job_id": "__error__", "project_id": "", "user_email": "", "creation_time": "",
        "query_hash": "", "referenced_tables": "[]", "destination_table": "",
        "total_bytes_processed": 0, "total_slot_ms": 0, "raw": json_dumps({"error": error}),
    }


def build_table_usage(jobs: pd.DataFrame) -> pd.DataFrame:
    if jobs is None or jobs.empty:
        return pd.DataFrame()
    rows: List[Dict[str, Any]] = []
    for _, j in jobs.iterrows():
        if str(j.get("job_id")) == "__error__":
            continue
        user = str(j.get("user_email") or "")
        refs = []
        try:
            import orjson
            refs = orjson.loads(str(j.get("referenced_tables") or "[]"))
        except Exception:
            refs = []
        dest = str(j.get("destination_table") or "")
        for ref in refs:
            rows.append({
                "principal_email": user,
                "service_account": user if user.endswith("gserviceaccount.com") else "",
                "referenced_table": ref,
                "destination_table": dest,
                "query_hash": str(j.get("query_hash") or ""),
                "job_count": 1,
                "last_seen": str(j.get("creation_time") or ""),
                "total_bytes_processed": int(j.get("total_bytes_processed") or 0),
                "source": "bq_information_schema_jobs_by_organization",
                "confidence": 0.8,
            })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    agg = df.groupby(["principal_email", "service_account", "referenced_table", "destination_table", "query_hash", "source"], dropna=False).agg(
        job_count=("job_count", "sum"),
        last_seen=("last_seen", "max"),
        total_bytes_processed=("total_bytes_processed", "sum"),
        confidence=("confidence", "max"),
    ).reset_index()
    return agg[["principal_email", "service_account", "referenced_table", "destination_table", "query_hash", "job_count", "last_seen", "total_bytes_processed", "source", "confidence"]]
