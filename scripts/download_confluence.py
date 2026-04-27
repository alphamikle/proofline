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

from corp_kb.config import ensure_dirs, load_config
from corp_kb.utils import now_iso, stable_id


def safe_name(value: str) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in value)[:180] or "unnamed"


class AtlassianClient:
    def __init__(self, base_url: str, api_path: str, cfg: Dict[str, Any]):
        if not base_url or base_url.startswith("$"):
            raise SystemExit("Confluence base_url is empty. Set CONFLUENCE_BASE_URL or confluence.base_url.")
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

    def resolve_link(self, link: str) -> str:
        if link.startswith("http://") or link.startswith("https://"):
            return link
        parsed = urlparse(self.base_url)
        origin = f"{parsed.scheme}://{parsed.netloc}"
        if link.startswith("/wiki/"):
            return origin + link
        if link.startswith("/"):
            return self.base_url + link
        return self.base_url + "/" + link

    def get_json(self, url: str, params: Dict[str, Any] | None = None) -> Dict[str, Any]:
        for attempt in range(self.retry_count + 1):
            r = self.session.get(url, params=params, timeout=self.timeout)
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


def confluence_cql(cfg: Dict[str, Any], content_type: str) -> str:
    parts = [f'type="{content_type}"', 'status="current"']
    spaces = [s for s in cfg.get("spaces", []) if s]
    if spaces:
        quoted = ",".join(f'"{s}"' for s in spaces)
        parts.append(f"space in ({quoted})")
    if cfg.get("cql"):
        parts.append(f"({cfg['cql']})")
    return " AND ".join(parts)


def paged_results(client: AtlassianClient, path: str, params: Dict[str, Any], limit_key: str = "limit") -> Iterator[Dict[str, Any]]:
    start = int(params.get("start", 0))
    limit = int(params.get(limit_key, 100))
    while True:
        params = dict(params, start=start, **{limit_key: limit})
        data = client.get_json(client.api_url(path), params=params)
        yield data
        results = data.get("results") or []
        if not results or len(results) < limit:
            break
        start += len(results)


def mirror_confluence(config_path: str, dry_run: bool = False) -> None:
    cfg = load_config(config_path)
    ensure_dirs(cfg)
    conf = cfg.get("confluence", {})
    if not conf.get("enabled", False) and not dry_run:
        raise SystemExit("confluence.enabled is false. Enable it in config.yaml or run with --dry-run.")
    out = Path(conf.get("output_dir") or Path(cfg["workspace"]) / "raw" / "confluence")
    if dry_run:
        print(json.dumps({"base_url": conf.get("base_url"), "output_dir": str(out), "content_types": conf.get("content_types", [])}, indent=2))
        return
    if conf.get("clean_output", True) and out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    client = AtlassianClient(conf.get("base_url", ""), conf.get("api_path", "/rest/api"), cfg.get("atlassian", {}))
    expand = ",".join(conf.get("expand", []))
    page_limit = int(conf.get("page_limit", 100))
    max_items = conf.get("max_items")
    manifest: Dict[str, Any] = {"started_at": now_iso(), "items": [], "attachments": []}
    count = 0
    for content_type in conf.get("content_types", ["page", "blogpost"]):
        params = {"cql": confluence_cql(conf, content_type), "expand": expand, "limit": page_limit}
        for page in paged_results(client, "content/search", params):
            for item in page.get("results") or []:
                content_id = str(item.get("id"))
                write_json(out / "content" / f"{content_id}.json", item)
                manifest["items"].append({"id": content_id, "type": item.get("type"), "title": item.get("title")})
                count += 1
                if conf.get("include_comments", True):
                    comments = []
                    for cp in paged_results(client, f"content/{content_id}/child/comment", {"expand": "body.storage,body.view,version,history", "limit": page_limit}):
                        comments.extend(cp.get("results") or [])
                    write_json(out / "comments" / f"{content_id}.json", comments)
                if conf.get("include_attachments", True):
                    attachments = []
                    for ap in paged_results(client, f"content/{content_id}/child/attachment", {"expand": "version,container,metadata", "limit": page_limit}):
                        attachments.extend(ap.get("results") or [])
                    write_json(out / "attachments" / f"{content_id}.json", attachments)
                    if conf.get("download_attachments", True):
                        for att in attachments:
                            link = ((att.get("_links") or {}).get("download") or "")
                            if not link:
                                continue
                            title = safe_name(att.get("title") or att.get("id") or stable_id(att))
                            target = out / "files" / content_id / title
                            info = client.download(client.resolve_link(link), target)
                            manifest["attachments"].append({"content_id": content_id, "attachment_id": att.get("id"), "title": att.get("title"), **info})
                if max_items and count >= int(max_items):
                    write_json(out / "manifest.json", {**manifest, "finished_at": now_iso(), "count": count})
                    print(f"Downloaded {count} Confluence content items into {out}")
                    return
    write_json(out / "manifest.json", {**manifest, "finished_at": now_iso(), "count": count})
    print(f"Downloaded {count} Confluence content items into {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Mirror Confluence content locally for corp-kb indexing.")
    parser.add_argument("--config", "-c", default="config.yaml")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    mirror_confluence(args.config, args.dry_run)


if __name__ == "__main__":
    main()
