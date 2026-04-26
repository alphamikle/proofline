from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
import requests

from corp_kb.utils import dd_time_window, epoch_window, flatten_json, json_dumps, pick_first, stable_id

SITE_HOSTS = {
    "datadoghq.com": "https://api.datadoghq.com",
    "us3.datadoghq.com": "https://api.us3.datadoghq.com",
    "us5.datadoghq.com": "https://api.us5.datadoghq.com",
    "datadoghq.eu": "https://api.datadoghq.eu",
    "ap1.datadoghq.com": "https://api.ap1.datadoghq.com",
    "ap2.datadoghq.com": "https://api.ap2.datadoghq.com",
    "ddog-gov.com": "https://api.ddog-gov.com",
}


class DatadogClient:
    def __init__(self, cfg: Dict[str, Any]):
        dd = cfg.get("datadog", {})
        self.site = dd.get("site") or os.getenv("DD_SITE", "datadoghq.com")
        self.base = SITE_HOSTS.get(self.site, f"https://api.{self.site}")
        self.api_key = os.getenv("DD_API_KEY", "")
        self.app_key = os.getenv("DD_APP_KEY", "")
        self.timeout = 60

    @property
    def enabled(self) -> bool:
        return bool(self.api_key and self.app_key)

    def headers(self) -> Dict[str, str]:
        return {
            "DD-API-KEY": self.api_key,
            "DD-APPLICATION-KEY": self.app_key,
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

    def get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Any:
        url = self.base + path
        r = requests.get(url, headers=self.headers(), params=params, timeout=self.timeout)
        if r.status_code == 429:
            time.sleep(5)
            r = requests.get(url, headers=self.headers(), params=params, timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def post(self, path: str, body: Dict[str, Any]) -> Any:
        url = self.base + path
        r = requests.post(url, headers=self.headers(), json=body, timeout=self.timeout)
        if r.status_code == 429:
            time.sleep(5)
            r = requests.post(url, headers=self.headers(), json=body, timeout=self.timeout)
        r.raise_for_status()
        return r.json()


def pull_service_dependencies(cfg: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    client = DatadogClient(cfg)
    services_rows: List[Dict[str, Any]] = []
    edge_rows: List[Dict[str, Any]] = []
    if not cfg.get("datadog", {}).get("enabled", True) or not client.enabled:
        return pd.DataFrame(services_rows), pd.DataFrame(edge_rows)
    envs = cfg["datadog"].get("envs", ["prod"])
    windows = cfg["datadog"].get("windows_days", [30])
    for env in envs:
        for days in windows:
            start, end = epoch_window(int(days))
            try:
                data = client.get("/api/v1/service_dependencies", params={"env": env, "start": start, "end": end})
            except Exception as e:
                edge_rows.append({
                    "from_service": "__error__", "to_service": str(e), "env": env,
                    "window_days": int(days), "source": "datadog_service_dependencies_error",
                    "first_seen": "", "last_seen": "", "confidence": 0.0, "raw": json_dumps({"error": str(e)}),
                })
                continue
            for svc, meta in (data or {}).items():
                services_rows.append({
                    "datadog_service": svc, "env": env, "raw": json_dumps(meta),
                    "source": "datadog_service_dependencies", "pulled_at": datetime.now(timezone.utc).isoformat(),
                })
                for called in (meta or {}).get("calls", []) or []:
                    edge_rows.append({
                        "from_service": svc, "to_service": called, "env": env,
                        "window_days": int(days), "source": "datadog_service_dependencies",
                        "first_seen": "", "last_seen": "", "confidence": 0.85,
                        "raw": json_dumps(meta),
                    })
    return pd.DataFrame(services_rows), pd.DataFrame(edge_rows)


def pull_service_definitions(cfg: Dict[str, Any]) -> pd.DataFrame:
    client = DatadogClient(cfg)
    rows: List[Dict[str, Any]] = []
    if not cfg.get("datadog", {}).get("enabled", True) or not client.enabled:
        return pd.DataFrame(rows)
    page = 0
    while True:
        try:
            data = client.get("/api/v2/services/definitions", params={"page[size]": 100, "page[number]": page})
        except Exception as e:
            rows.append({"datadog_service": "__error__", "env": "", "raw": json_dumps({"error": str(e)}), "source": "datadog_service_definitions_error", "pulled_at": datetime.now(timezone.utc).isoformat()})
            break
        items = data.get("data", []) if isinstance(data, dict) else []
        if not items:
            break
        for item in items:
            attrs = item.get("attributes") or {}
            schema = attrs.get("schema") or {}
            name = schema.get("name") or attrs.get("name") or item.get("id") or ""
            rows.append({
                "datadog_service": name,
                "env": "",
                "raw": json_dumps(item),
                "source": "datadog_service_definitions",
                "pulled_at": datetime.now(timezone.utc).isoformat(),
            })
        page += 1
        if page > 1000:
            break
    return pd.DataFrame(rows)


def search_spans(cfg: Dict[str, Any]) -> pd.DataFrame:
    dd = cfg.get("datadog", {})
    client = DatadogClient(cfg)
    rows: List[Dict[str, Any]] = []
    spans_cfg = dd.get("spans", {})
    if not dd.get("enabled", True) or not spans_cfg.get("enabled", True) or not client.enabled:
        return pd.DataFrame(rows)
    aliases = dd.get("field_aliases", {})
    for days in dd.get("windows_days", [30]):
        start, end = dd_time_window(int(days))
        body = {
            "data": {
                "type": "search_request",
                "attributes": {
                    "filter": {"from": start, "to": end, "query": spans_cfg.get("query", "*")},
                    "page": {"limit": int(spans_cfg.get("page_limit", 1000))},
                    "sort": "timestamp",
                },
            }
        }
        cursor = None
        max_pages = int(spans_cfg.get("max_pages_per_window", 20))
        for _ in range(max_pages):
            if cursor:
                body["data"]["attributes"]["page"]["cursor"] = cursor
            try:
                data = client.post("/api/v2/spans/events/search", body)
            except Exception as e:
                rows.append(_span_error_row(str(e)))
                break
            for item in data.get("data", []) if isinstance(data, dict) else []:
                rows.append(normalize_span(item, aliases))
            cursor = ((data.get("meta") or {}).get("page") or {}).get("after") if isinstance(data, dict) else None
            if not cursor:
                break
    return pd.DataFrame(rows)


def search_logs(cfg: Dict[str, Any]) -> pd.DataFrame:
    dd = cfg.get("datadog", {})
    client = DatadogClient(cfg)
    rows: List[Dict[str, Any]] = []
    logs_cfg = dd.get("logs", {})
    if not dd.get("enabled", True) or not logs_cfg.get("enabled", True) or not client.enabled:
        return pd.DataFrame(rows)
    aliases = dd.get("field_aliases", {})
    for days in dd.get("windows_days", [30]):
        start, end = dd_time_window(int(days))
        body = {
            "filter": {"from": start, "to": end, "query": logs_cfg.get("query", "*")},
            "page": {"limit": int(logs_cfg.get("page_limit", 1000))},
            "sort": "timestamp",
        }
        cursor = None
        max_pages = int(logs_cfg.get("max_pages_per_window", 20))
        for _ in range(max_pages):
            if cursor:
                body["page"]["cursor"] = cursor
            try:
                data = client.post("/api/v2/logs/events/search", body)
            except Exception as e:
                rows.append(_log_error_row(str(e)))
                break
            for item in data.get("data", []) if isinstance(data, dict) else []:
                rows.append(normalize_log(item, aliases))
            cursor = ((data.get("meta") or {}).get("page") or {}).get("after") if isinstance(data, dict) else None
            if not cursor:
                break
    return pd.DataFrame(rows)


def normalize_span(item: Dict[str, Any], aliases: Dict[str, List[str]]) -> Dict[str, Any]:
    flat = flatten_json(item)
    attrs = item.get("attributes", {}) if isinstance(item, dict) else {}
    return {
        "span_id": str(pick_first(flat, aliases.get("span_id", [])) or item.get("id") or ""),
        "trace_id": str(pick_first(flat, aliases.get("trace_id", [])) or ""),
        "parent_id": str(pick_first(flat, aliases.get("parent_id", [])) or ""),
        "service": str(pick_first(flat, aliases.get("service", [])) or attrs.get("service") or ""),
        "env": str(pick_first(flat, aliases.get("env", [])) or ""),
        "resource": str(pick_first(flat, aliases.get("route", [])) or attrs.get("resource") or attrs.get("name") or ""),
        "operation": str(attrs.get("operation_name") or attrs.get("name") or ""),
        "route": str(pick_first(flat, aliases.get("route", [])) or ""),
        "method": str(pick_first(flat, aliases.get("method", [])) or ""),
        "url": str(pick_first(flat, aliases.get("url", [])) or ""),
        "peer_service": str(pick_first(flat, aliases.get("peer_service", [])) or ""),
        "host": str(pick_first(flat, aliases.get("host", [])) or ""),
        "db_name": str(pick_first(flat, aliases.get("db_name", [])) or ""),
        "messaging_destination": str(pick_first(flat, aliases.get("messaging_destination", [])) or ""),
        "duration_ms": to_float(pick_first(flat, aliases.get("duration_ms", []))),
        "error": bool(flat.get("attributes.error") or flat.get("attributes.attributes.error") or False),
        "timestamp": str(attrs.get("timestamp") or pick_first(flat, ["timestamp"]) or ""),
        "raw": json_dumps(item),
    }


def normalize_log(item: Dict[str, Any], aliases: Dict[str, List[str]]) -> Dict[str, Any]:
    flat = flatten_json(item)
    attrs = item.get("attributes", {}) if isinstance(item, dict) else {}
    message = attrs.get("message") or flat.get("attributes.message") or ""
    return {
        "log_id": str(item.get("id") or stable_id(json_dumps(item))),
        "trace_id": str(pick_first(flat, aliases.get("trace_id", [])) or ""),
        "span_id": str(pick_first(flat, aliases.get("span_id", [])) or ""),
        "service": str(pick_first(flat, aliases.get("service", [])) or ""),
        "env": str(pick_first(flat, aliases.get("env", [])) or ""),
        "route": str(pick_first(flat, aliases.get("route", [])) or ""),
        "method": str(pick_first(flat, aliases.get("method", [])) or ""),
        "status_code": str(pick_first(flat, aliases.get("status_code", [])) or ""),
        "url": str(pick_first(flat, aliases.get("url", [])) or ""),
        "host": str(pick_first(flat, aliases.get("host", [])) or ""),
        "peer_service": str(pick_first(flat, aliases.get("peer_service", [])) or ""),
        "db_name": str(pick_first(flat, aliases.get("db_name", [])) or ""),
        "messaging_destination": str(pick_first(flat, aliases.get("messaging_destination", [])) or ""),
        "duration_ms": to_float(pick_first(flat, aliases.get("duration_ms", []))),
        "timestamp": str(attrs.get("timestamp") or pick_first(flat, ["timestamp"]) or ""),
        "message": str(message)[:5000],
        "raw": json_dumps(item),
    }


def to_float(x: Any) -> Optional[float]:
    try:
        if x is None or x == "":
            return None
        v = float(x)
        # Some spans use nanoseconds.
        if v > 10_000_000:
            return v / 1_000_000.0
        return v
    except Exception:
        return None


def _span_error_row(error: str) -> Dict[str, Any]:
    return {"span_id": "__error__", "trace_id": "", "parent_id": "", "service": "", "env": "", "resource": "", "operation": "", "route": "", "method": "", "url": "", "peer_service": "", "host": "", "db_name": "", "messaging_destination": "", "duration_ms": None, "error": False, "timestamp": "", "raw": json_dumps({"error": error})}


def _log_error_row(error: str) -> Dict[str, Any]:
    return {"log_id": "__error__", "trace_id": "", "span_id": "", "service": "", "env": "", "route": "", "method": "", "status_code": "", "url": "", "host": "", "peer_service": "", "db_name": "", "messaging_destination": "", "duration_ms": None, "timestamp": "", "message": "", "raw": json_dumps({"error": error})}


def build_runtime_edges_from_dd(spans: pd.DataFrame, logs: pd.DataFrame, service_edges: pd.DataFrame, default_windows: List[int]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    service_rows: List[Dict[str, Any]] = []
    endpoint_rows: List[Dict[str, Any]] = []
    # Service dependency API edges.
    if not service_edges.empty:
        for _, e in service_edges.iterrows():
            if str(e.get("from_service")) == "__error__":
                continue
            edge_id = stable_id("runtime_service", e.get("from_service"), e.get("to_service"), e.get("env"), e.get("window_days"))
            service_rows.append({
                "edge_id": edge_id, "from_service": e.get("from_service"), "to_entity": e.get("to_service"),
                "to_type": "service", "edge_type": "OBSERVED_CALL", "env": e.get("env"),
                "source": e.get("source"), "window_days": int(e.get("window_days") or 0),
                "count": None, "p95_ms": None, "error_rate": None,
                "first_seen": e.get("first_seen"), "last_seen": e.get("last_seen"),
                "confidence": e.get("confidence") or 0.85, "evidence_refs": json_dumps([edge_id]),
            })
    # Spans direct fields.
    if spans is not None and not spans.empty:
        df = spans[spans["service"].fillna("") != ""].copy()
        # service -> peer/db/topic/host
        for target_col, typ, kind in [
            ("peer_service", "service", "SPAN_PEER_SERVICE"),
            ("db_name", "database", "SPAN_DB"),
            ("messaging_destination", "topic", "SPAN_MESSAGING"),
            ("host", "host", "SPAN_HOST"),
        ]:
            sub = df[df[target_col].fillna("") != ""]
            if sub.empty:
                continue
            grp = sub.groupby(["service", "env", target_col], dropna=False)
            for (svc, env, target), g in grp:
                durations = pd.to_numeric(g["duration_ms"], errors="coerce").dropna()
                edge_id = stable_id("span", svc, target_col, target, env)
                service_rows.append({
                    "edge_id": edge_id, "from_service": svc, "to_entity": str(target), "to_type": typ,
                    "edge_type": "OBSERVED_CALL" if typ == "service" else f"USES_{typ.upper()}",
                    "env": env, "source": "datadog_spans", "window_days": max(default_windows or [0]),
                    "count": int(len(g)), "p95_ms": float(durations.quantile(0.95)) if len(durations) else None,
                    "error_rate": float(g["error"].astype(bool).mean()) if "error" in g else None,
                    "first_seen": str(g["timestamp"].min()), "last_seen": str(g["timestamp"].max()),
                    "confidence": 0.95, "evidence_refs": json_dumps([edge_id]),
                })
        # endpoint -> downstream target.
        endpoint_col = "route" if "route" in df.columns else "resource"
        df["endpoint_key"] = df[endpoint_col].fillna("")
        df.loc[df["endpoint_key"] == "", "endpoint_key"] = df["resource"].fillna("")
        for target_col, typ, kind in [
            ("peer_service", "service", "CALLS"), ("db_name", "database", "READS_OR_WRITES"),
            ("messaging_destination", "topic", "PUBLISHES_OR_CONSUMES"), ("host", "host", "CALLS_HOST"),
        ]:
            sub = df[(df["endpoint_key"].fillna("") != "") & (df[target_col].fillna("") != "")]
            for (svc, ep, method, env, target), g in sub.groupby(["service", "endpoint_key", "method", "env", target_col], dropna=False):
                durations = pd.to_numeric(g["duration_ms"], errors="coerce").dropna()
                edge_id = stable_id("endpoint_span", svc, ep, target_col, target, env)
                endpoint_rows.append({
                    "edge_id": edge_id, "service_id": svc, "endpoint_key": ep, "method": method or "",
                    "path": ep, "downstream_entity": str(target), "downstream_type": typ,
                    "dependency_kind": kind, "env": env, "source": "datadog_spans",
                    "window_days": max(default_windows or [0]), "count": int(len(g)),
                    "p95_ms": float(durations.quantile(0.95)) if len(durations) else None,
                    "error_rate": float(g["error"].astype(bool).mean()) if "error" in g else None,
                    "first_seen": str(g["timestamp"].min()), "last_seen": str(g["timestamp"].max()),
                    "confidence": 0.95, "evidence_refs": json_dumps([edge_id]),
                })
    # Logs: lower confidence if no spans.
    if logs is not None and not logs.empty:
        df = logs[logs["service"].fillna("") != ""].copy()
        for target_col, typ, kind in [
            ("peer_service", "service", "LOG_PEER_SERVICE"), ("db_name", "database", "LOG_DB"),
            ("messaging_destination", "topic", "LOG_MESSAGING"), ("host", "host", "LOG_HOST"),
        ]:
            sub = df[df[target_col].fillna("") != ""]
            for (svc, env, target), g in sub.groupby(["service", "env", target_col], dropna=False):
                edge_id = stable_id("log", svc, target_col, target, env)
                service_rows.append({
                    "edge_id": edge_id, "from_service": svc, "to_entity": str(target), "to_type": typ,
                    "edge_type": "OBSERVED_CALL" if typ == "service" else f"USES_{typ.upper()}",
                    "env": env, "source": "datadog_logs", "window_days": max(default_windows or [0]),
                    "count": int(len(g)), "p95_ms": None, "error_rate": None,
                    "first_seen": str(g["timestamp"].min()), "last_seen": str(g["timestamp"].max()),
                    "confidence": 0.75, "evidence_refs": json_dumps([edge_id]),
                })
            sub2 = df[(df["route"].fillna("") != "") & (df[target_col].fillna("") != "")]
            for (svc, ep, method, env, target), g in sub2.groupby(["service", "route", "method", "env", target_col], dropna=False):
                edge_id = stable_id("endpoint_log", svc, ep, target_col, target, env)
                endpoint_rows.append({
                    "edge_id": edge_id, "service_id": svc, "endpoint_key": ep, "method": method or "",
                    "path": ep, "downstream_entity": str(target), "downstream_type": typ,
                    "dependency_kind": kind, "env": env, "source": "datadog_logs",
                    "window_days": max(default_windows or [0]), "count": int(len(g)),
                    "p95_ms": None, "error_rate": None,
                    "first_seen": str(g["timestamp"].min()), "last_seen": str(g["timestamp"].max()),
                    "confidence": 0.75, "evidence_refs": json_dumps([edge_id]),
                })
    return pd.DataFrame(service_rows), pd.DataFrame(endpoint_rows)
