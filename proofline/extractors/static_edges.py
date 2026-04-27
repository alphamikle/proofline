from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd
import yaml

from proofline.utils import safe_read_text, stable_id, normalize_name

URL_RE = re.compile(r"https?://[A-Za-z0-9_.:/\-{}$]+")
HOST_RE = re.compile(r"\b([a-z0-9][a-z0-9-]+(?:\.[a-z0-9][a-z0-9-]+){1,})(?::\d+)?\b", re.I)
ENV_RE = re.compile(r"\b([A-Z][A-Z0-9_]{2,}(?:URL|URI|HOST|SERVICE|TOPIC|QUEUE|TABLE|DATASET|DATABASE|DB))\b")
BQ_BACKTICK_TABLE_RE = re.compile(r"`([a-zA-Z0-9_\-]+\.[a-zA-Z0-9_\-]+\.[a-zA-Z0-9_\-]+)`")
BQ_SQL_TABLE_RE = re.compile(r"\b(?:FROM|JOIN|INTO|UPDATE|TABLE|MERGE\s+INTO)\s+`?([a-zA-Z0-9_\-]+\.[a-zA-Z0-9_\-]+\.[a-zA-Z0-9_\-]+)`?", re.I)
TOPIC_RE = re.compile(r"\b([a-z][a-z0-9_.\-]+\.(?:events?|commands?|topic|queue)\.?[a-z0-9_.\-]*)\b")
INTERNAL_PACKAGE_RE = re.compile(r"(@[A-Za-z0-9_.\-/]+/[A-Za-z0-9_.\-/]+|com\.[A-Za-z0-9_.\-]+|[A-Za-z0-9_.\-]+:[A-Za-z0-9_.\-]+)")


def extract_static_edges(repo_inventory: pd.DataFrame, repo_files: pd.DataFrame) -> pd.DataFrame:
    edges: List[Dict[str, Any]] = []
    inv_by_repo = {r["repo_id"]: r for _, r in repo_inventory.iterrows()}
    for _, f in repo_files.iterrows():
        repo_id = str(f["repo_id"])
        from_entity = f"repo:{repo_id}"
        rel = str(f["rel_path"])
        path = Path(str(f["path"]))
        text = safe_read_text(path)
        if not text:
            continue
        kind = str(f.get("kind") or "")
        if kind == "manifest":
            edges.extend(parse_manifest(repo_id, from_entity, rel, text))
        if kind in {"deploy_config", "source", "source_route_hint", "manifest", "dockerfile", "api_contract"}:
            include_hosts = kind not in {"source", "source_route_hint"}
            edges.extend(parse_refs(repo_id, from_entity, rel, text, include_hosts=include_hosts))
    return pd.DataFrame(edges)


def edge(repo_id: str, from_entity: str, to_entity: str, edge_type: str, source: str, rel: str, raw: str, line: int, confidence: float) -> Dict[str, Any]:
    return {
        "edge_id": stable_id(from_entity, to_entity, edge_type, source, rel, line, raw),
        "from_entity": from_entity,
        "to_entity": to_entity,
        "edge_type": edge_type,
        "source": source,
        "repo_id": repo_id,
        "file_path": rel,
        "line_start": line,
        "line_end": line,
        "raw_match": raw[:1000],
        "confidence": confidence,
    }


def line_no(text: str, pos: int) -> int:
    return text[:pos].count("\n") + 1


def parse_manifest(repo_id: str, from_entity: str, rel: str, text: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    name = Path(rel).name.lower()
    if name == "package.json":
        try:
            obj = json.loads(text)
            deps = {}
            for k in ["dependencies", "devDependencies", "peerDependencies"]:
                deps.update(obj.get(k) or {})
            for dep, ver in deps.items():
                out.append(edge(repo_id, from_entity, f"package:{dep}", "DEPENDS_ON_PACKAGE", "package.json", rel, f"{dep}@{ver}", 1, 0.6))
        except Exception:
            pass
    elif name in {"go.mod"}:
        for m in re.finditer(r"^\s*([A-Za-z0-9_.\-/]+)\s+v[0-9]", text, re.M):
            out.append(edge(repo_id, from_entity, f"package:{m.group(1)}", "DEPENDS_ON_PACKAGE", "go.mod", rel, m.group(0), line_no(text, m.start()), 0.6))
    elif name in {"requirements.txt"}:
        for m in re.finditer(r"^\s*([A-Za-z0-9_.\-]+)", text, re.M):
            dep = m.group(1)
            if dep and not dep.startswith("#"):
                out.append(edge(repo_id, from_entity, f"package:{dep}", "DEPENDS_ON_PACKAGE", "requirements", rel, m.group(0), line_no(text, m.start()), 0.55))
    elif name in {"pom.xml", "build.gradle", "gradle.lockfile"}:
        for m in INTERNAL_PACKAGE_RE.finditer(text):
            out.append(edge(repo_id, from_entity, f"package:{m.group(1)}", "DEPENDS_ON_PACKAGE", name, rel, m.group(0), line_no(text, m.start()), 0.45))
    return out


def parse_refs(repo_id: str, from_entity: str, rel: str, text: str, include_hosts: bool = True) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if "http://" in text or "https://" in text:
        for m in URL_RE.finditer(text):
            out.append(edge(repo_id, from_entity, f"url:{m.group(0)}", "REFERENCES_URL", "regex_url", rel, m.group(0), line_no(text, m.start()), 0.45))
    for m in ENV_RE.finditer(text):
        out.append(edge(repo_id, from_entity, f"config:{m.group(1)}", "USES_CONFIG_KEY", "regex_env", rel, m.group(1), line_no(text, m.start()), 0.4))
    if "." in text:
        seen_bq = set()
        for pat in (BQ_BACKTICK_TABLE_RE, BQ_SQL_TABLE_RE):
            for m in pat.finditer(text):
                table = m.group(1)
                if table in seen_bq:
                    continue
                seen_bq.add(table)
                out.append(edge(repo_id, from_entity, f"bq_table:{table}", "REFERENCES_BQ_TABLE", "regex_bq_table", rel, table, line_no(text, m.start()), 0.45))
        for m in TOPIC_RE.finditer(text):
            out.append(edge(repo_id, from_entity, f"topic:{m.group(1)}", "REFERENCES_TOPIC", "regex_topic", rel, m.group(1), line_no(text, m.start()), 0.35))
    # Hostnames are noisy and expensive in source files; keep them to config-like
    # surfaces where they are more likely to represent real dependencies.
    if include_hosts and "." in text:
        for m in HOST_RE.finditer(text):
            host = m.group(1)
            if any(host.endswith(x) for x in ["github.com", "google.com", "w3.org", "schema.org", "npmjs.com"]):
                continue
            out.append(edge(repo_id, from_entity, f"host:{host}", "REFERENCES_HOST", "regex_host", rel, host, line_no(text, m.start()), 0.25))
    return out
