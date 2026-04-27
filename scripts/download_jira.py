#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List
from urllib.parse import urlparse

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from proofline.config import ensure_dirs, load_config
from proofline.utils import now_iso


def safe_name(value: str) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in value)[:180] or "unnamed"


class JiraClient:
    def __init__(self, base_url: str, api_path: str, cfg: Dict[str, Any]):
        if not base_url or base_url.startswith("$"):
            raise SystemExit("Jira base_url is empty. Set JIRA_BASE_URL or jira.base_url.")
        self.base_url = base_url.rstrip("/")
        self.api_path = "/" + api_path.strip("/")
        self.timeout = int(cfg.get("request_timeout_seconds", 60))
        self.retry_count = int(cfg.get("retry_count", 3))
        self.retry_sleep = int(cfg.get("retry_sleep_seconds", 5))
        self.session = requests.Session()
        bearer = os.getenv(cfg.get("bearer_token_env", "ATLASSIAN_BEARER_TOKEN"), "")
        email = os.getenv(cfg.get("email_env", "ATLASSIAN_EMAIL"), "")
        token = os.getenv(cfg.get("api_token_env", "ATLASSIAN_API_TOKEN"), "")
        if bearer:
            self.session.headers.update({"Authorization": f"Bearer {bearer}"})
        elif email and token:
            self.session.auth = (email, token)
        else:
            raise SystemExit("Missing Atlassian auth. Set ATLASSIAN_EMAIL + ATLASSIAN_API_TOKEN or ATLASSIAN_BEARER_TOKEN.")
        self.session.headers.update({"Accept": "application/json"})

    def api_url(self, path: str) -> str:
        return f"{self.base_url}{self.api_path}/{path.lstrip('/')}"

    def get_json(self, url: str, params: Dict[str, Any] | None = None) -> Dict[str, Any]:
        for attempt in range(self.retry_count + 1):
            r = self.session.get(url, params=params, timeout=self.timeout)
            if r.status_code in {429, 500, 502, 503, 504} and attempt < self.retry_count:
                time.sleep(self.retry_sleep * (attempt + 1))
                continue
            r.raise_for_status()
            return r.json()
        raise RuntimeError("unreachable")

    def post_json(self, url: str, body: Dict[str, Any]) -> Dict[str, Any]:
        for attempt in range(self.retry_count + 1):
            r = self.session.post(url, json=body, timeout=self.timeout)
            if r.status_code in {429, 500, 502, 503, 504} and attempt < self.retry_count:
                time.sleep(self.retry_sleep * (attempt + 1))
                continue
            r.raise_for_status()
            return r.json()
        raise RuntimeError("unreachable")

    def download(self, url: str, target: Path) -> Dict[str, Any]:
        target.parent.mkdir(parents=True, exist_ok=True)
        for attempt in range(self.retry_count + 1):
            r = self.session.get(url, stream=True, timeout=self.timeout)
            if r.status_code in {429, 500, 502, 503, 504} and attempt < self.retry_count:
                time.sleep(self.retry_sleep * (attempt + 1))
                continue
            r.raise_for_status()
            with target.open("wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
            return {"path": str(target), "size_bytes": target.stat().st_size, "content_type": r.headers.get("content-type", "")}
        raise RuntimeError("unreachable")


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def jira_search(client: JiraClient, cfg: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
    start_at = 0
    page_limit = int(cfg.get("page_limit", 100))
    while True:
        body = {
            "jql": cfg.get("jql", "ORDER BY updated DESC"),
            "startAt": start_at,
            "maxResults": page_limit,
            "fields": cfg.get("fields", ["*all"]),
            "expand": cfg.get("expand", []),
        }
        data = client.post_json(client.api_url("search"), body)
        yield data
        issues = data.get("issues") or []
        if not issues or start_at + len(issues) >= int(data.get("total") or 0):
            break
        start_at += len(issues)


def paged_child(client: JiraClient, path: str, item_key: str, page_limit: int = 100) -> List[Dict[str, Any]]:
    start_at = 0
    rows: List[Dict[str, Any]] = []
    while True:
        data = client.get_json(client.api_url(path), {"startAt": start_at, "maxResults": page_limit})
        items = data.get(item_key) or data.get("values") or []
        rows.extend(items)
        total = int(data.get("total") or len(rows))
        if not items or start_at + len(items) >= total:
            return rows
        start_at += len(items)


def mirror_jira(config_path: str, dry_run: bool = False) -> None:
    cfg = load_config(config_path)
    ensure_dirs(cfg)
    jira = cfg.get("jira", {})
    if not jira.get("enabled", False) and not dry_run:
        raise SystemExit("jira.enabled is false. Enable it in proofline.yaml or run with --dry-run.")
    out = Path(jira.get("output_dir") or Path(cfg["workspace"]) / "raw" / "jira")
    if dry_run:
        print(json.dumps({"base_url": jira.get("base_url"), "output_dir": str(out), "jql": jira.get("jql")}, indent=2))
        return
    if jira.get("clean_output", True) and out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    client = JiraClient(jira.get("base_url", ""), jira.get("api_path", "/rest/api/3"), cfg.get("atlassian", {}))
    page_limit = int(jira.get("page_limit", 100))
    manifest: Dict[str, Any] = {"started_at": now_iso(), "issues": [], "attachments": []}

    if jira.get("download_metadata", True):
        metadata = {}
        for name, path in {
            "fields": "field",
            "statuses": "status",
            "priorities": "priority",
            "resolutions": "resolution",
            "issue_types": "issuetype",
            "projects": "project/search",
        }.items():
            try:
                metadata[name] = client.get_json(client.api_url(path))
            except Exception as e:
                metadata[name] = {"error": str(e)}
        write_json(out / "metadata.json", metadata)

    count = 0
    max_issues = jira.get("max_issues")
    for page in jira_search(client, jira):
        for issue in page.get("issues") or []:
            key = str(issue.get("key"))
            write_json(out / "issues" / f"{safe_name(key)}.json", issue)
            manifest["issues"].append({"key": key, "id": issue.get("id"), "summary": ((issue.get("fields") or {}).get("summary") or "")})
            if jira.get("include_comments", True):
                comments = paged_child(client, f"issue/{key}/comment", "comments", page_limit)
                write_json(out / "comments" / f"{safe_name(key)}.json", comments)
            if jira.get("include_changelog", True):
                changelog = paged_child(client, f"issue/{key}/changelog", "values", page_limit)
                write_json(out / "changelog" / f"{safe_name(key)}.json", changelog)
            if jira.get("include_worklogs", True):
                worklogs = paged_child(client, f"issue/{key}/worklog", "worklogs", page_limit)
                write_json(out / "worklogs" / f"{safe_name(key)}.json", worklogs)
            if jira.get("include_remote_links", True):
                try:
                    write_json(out / "remote_links" / f"{safe_name(key)}.json", client.get_json(client.api_url(f"issue/{key}/remotelink")))
                except Exception as e:
                    write_json(out / "remote_links" / f"{safe_name(key)}.json", {"error": str(e)})
            if jira.get("include_issue_properties", True):
                try:
                    props = client.get_json(client.api_url(f"issue/{key}/properties"))
                    full_props = {}
                    for p in props.get("keys", []):
                        prop_key = p.get("key")
                        if prop_key:
                            full_props[prop_key] = client.get_json(client.api_url(f"issue/{key}/properties/{prop_key}"))
                    write_json(out / "properties" / f"{safe_name(key)}.json", full_props)
                except Exception as e:
                    write_json(out / "properties" / f"{safe_name(key)}.json", {"error": str(e)})
            if jira.get("include_attachments", True) and jira.get("download_attachments", True):
                for att in ((issue.get("fields") or {}).get("attachment") or []):
                    url = att.get("content")
                    if not url:
                        continue
                    filename = safe_name(att.get("filename") or att.get("id") or key)
                    target = out / "files" / safe_name(key) / filename
                    info = client.download(url, target)
                    manifest["attachments"].append({"issue": key, "attachment_id": att.get("id"), "filename": att.get("filename"), **info})
            count += 1
            if max_issues and count >= int(max_issues):
                write_json(out / "manifest.json", {**manifest, "finished_at": now_iso(), "count": count})
                print(f"Downloaded {count} Jira issues into {out}")
                return
    write_json(out / "manifest.json", {**manifest, "finished_at": now_iso(), "count": count})
    print(f"Downloaded {count} Jira issues into {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Mirror Jira issues locally for proofline indexing.")
    parser.add_argument("--config", "-c", default="proofline.yaml")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    mirror_jira(args.config, args.dry_run)


if __name__ == "__main__":
    main()
