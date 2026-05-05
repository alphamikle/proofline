from __future__ import annotations

import json
import mimetypes
import time
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import yaml

from proofline.config import (
    backup_config_path,
    config_followup_warnings,
    deep_merge,
    ensure_dirs,
    load_config,
    package_root,
)
from proofline.pipeline.runner import FULL_ORDER, STAGE_ALIASES, STAGES
from proofline.storage import KB
from proofline.ui.jobs import JobManager
from proofline.utils import json_dumps


STATIC_DIR = Path(__file__).resolve().parent / "static"


def serve_ui(
    config_path: str | Path,
    *,
    host: str = "127.0.0.1",
    port: int = 8766,
    open_browser: bool = True,
) -> None:
    config_path = str(Path(config_path).expanduser())
    manager = JobManager(config_path)
    server = ThreadingHTTPServer((host, port), UIHandler.factory(config_path, manager))
    url = f"http://{host}:{port}/"
    if open_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass
    print(f"Proofline UI: {url}")
    print(f"Config: {config_path}")
    print("Press Ctrl-C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


class UIHandler(BaseHTTPRequestHandler):
    config_path: str
    jobs: JobManager

    @classmethod
    def factory(cls, config_path: str, jobs: JobManager):
        class Handler(cls):
            pass

        Handler.config_path = config_path
        Handler.jobs = jobs
        return Handler

    def log_message(self, fmt: str, *args: Any) -> None:
        return

    def do_GET(self) -> None:
        try:
            parsed = urlparse(self.path)
            if parsed.path.startswith("/api/"):
                self._handle_api_get(parsed.path, parse_qs(parsed.query))
                return
            self._serve_static(parsed.path)
        except Exception as exc:
            self._send_error(exc)

    def do_POST(self) -> None:
        try:
            parsed = urlparse(self.path)
            if parsed.path == "/api/jobs":
                self._send_json(self.jobs.start(self._read_json()))
                return
            if parsed.path.startswith("/api/jobs/") and parsed.path.endswith("/cancel"):
                job_id = parsed.path.split("/")[3]
                job = self.jobs.cancel(job_id)
                if job is None:
                    self._send_json({"error": "Job not found"}, status=HTTPStatus.NOT_FOUND)
                    return
                self._send_json(job)
                return
            if parsed.path == "/api/config/patch":
                self._patch_config(self._read_json())
                return
            self._send_json({"error": "Not found"}, status=HTTPStatus.NOT_FOUND)
        except Exception as exc:
            self._send_error(exc)

    def do_PUT(self) -> None:
        try:
            parsed = urlparse(self.path)
            if parsed.path == "/api/config/raw":
                self._write_raw_config(self._read_text_or_json_text())
                return
            self._send_json({"error": "Not found"}, status=HTTPStatus.NOT_FOUND)
        except Exception as exc:
            self._send_error(exc)

    def _handle_api_get(self, path: str, query: dict[str, list[str]]) -> None:
        if path == "/api/health":
            self._send_json({"ok": True, "time": time.time()})
            return
        if path == "/api/stages":
            self._send_json(
                {
                    "stages": list(STAGES.keys()),
                    "full_order": FULL_ORDER,
                    "aliases": STAGE_ALIASES,
                    "sync_sources": ["all", "repos", "docs", "runtime", "data"],
                    "build_targets": [
                        "all",
                        "history",
                        "blame",
                        "code-graph",
                        "code",
                        "embeddings",
                        "api",
                        "static",
                        "identity",
                        "graph",
                        "endpoints",
                        "capabilities",
                        "visualization",
                    ],
                }
            )
            return
        if path == "/api/status":
            self._send_json(read_status(self.config_path))
            return
        if path == "/api/jobs":
            self._send_json({"jobs": self.jobs.list_jobs()})
            return
        if path.startswith("/api/jobs/"):
            job_id = path.split("/")[3]
            include_logs = (query.get("logs") or ["0"])[0] in {"1", "true", "yes"}
            job = self.jobs.get_job(job_id, include_logs=include_logs)
            if job is None:
                self._send_json({"error": "Job not found"}, status=HTTPStatus.NOT_FOUND)
                return
            self._send_json(job)
            return
        if path == "/api/config/raw":
            config_file = Path(self.config_path)
            self._send_json({"path": str(config_file), "text": config_file.read_text(encoding="utf-8")})
            return
        if path == "/api/config/parsed":
            cfg = load_config(self.config_path, quiet=True)
            self._send_json({"path": self.config_path, "config": _public_config(cfg), "warnings": config_followup_warnings(cfg)})
            return
        if path == "/api/config/schema":
            schema_path = package_root() / "config.schema.json"
            self._send_json(json.loads(schema_path.read_text(encoding="utf-8")))
            return
        self._send_json({"error": "Not found"}, status=HTTPStatus.NOT_FOUND)

    def _serve_static(self, path: str) -> None:
        if path in {"", "/"}:
            target = STATIC_DIR / "index.html"
        else:
            rel = path.lstrip("/")
            target = (STATIC_DIR / rel).resolve()
            if STATIC_DIR.resolve() not in target.parents and target != STATIC_DIR.resolve():
                self._send_json({"error": "Not found"}, status=HTTPStatus.NOT_FOUND)
                return
            if not target.exists():
                target = STATIC_DIR / "index.html"
        if not target.exists():
            self._send_json({"error": "UI assets are missing. Run `npm --prefix web run build` from the project checkout."}, status=HTTPStatus.NOT_FOUND)
            return
        content_type = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
        data = target.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store" if target.name == "index.html" else "public, max-age=31536000, immutable")
        self.end_headers()
        self.wfile.write(data)

    def _write_raw_config(self, text: str) -> None:
        parsed = yaml.safe_load(text) or {}
        if not isinstance(parsed, dict):
            raise ValueError("Config YAML must contain an object at the top level")
        path = Path(self.config_path)
        if path.exists():
            backup = backup_config_path(path)
            backup.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
        else:
            backup = None
            path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        cfg = load_config(path, quiet=True)
        ensure_dirs(cfg)
        self._send_json({"ok": True, "backup": str(backup) if backup else "", "config": _public_config(cfg), "warnings": config_followup_warnings(cfg)})

    def _patch_config(self, patch: dict[str, Any]) -> None:
        path = Path(self.config_path)
        current = yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else {}
        if not isinstance(current, dict):
            raise ValueError("Config YAML must contain an object at the top level")
        if not isinstance(patch, dict):
            raise ValueError("Patch must be an object")
        merged = deep_merge(current, patch)
        text = yaml.safe_dump(merged, sort_keys=False, allow_unicode=False)
        self._write_raw_config(text)

    def _read_json(self) -> dict[str, Any]:
        raw = self._read_body()
        if not raw:
            return {}
        payload = json.loads(raw.decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Expected a JSON object")
        return payload

    def _read_text_or_json_text(self) -> str:
        raw = self._read_body()
        content_type = self.headers.get("Content-Type", "")
        if "application/json" in content_type:
            payload = json.loads(raw.decode("utf-8"))
            if not isinstance(payload, dict) or not isinstance(payload.get("text"), str):
                raise ValueError("Expected JSON object with string field `text`")
            return payload["text"]
        return raw.decode("utf-8")

    def _read_body(self) -> bytes:
        try:
            length = int(self.headers.get("Content-Length") or "0")
        except ValueError:
            length = 0
        return self.rfile.read(length) if length > 0 else b""

    def _send_json(self, payload: Any, status: int | HTTPStatus = HTTPStatus.OK) -> None:
        data = json_dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _send_error(self, exc: Exception) -> None:
        self._send_json({"error": str(exc), "type": exc.__class__.__name__}, status=HTTPStatus.BAD_REQUEST)


def read_status(config_path: str | Path) -> dict[str, Any]:
    payload: dict[str, Any] = {"config_path": str(config_path), "ok": True}
    try:
        cfg = load_config(config_path, quiet=True)
        ensure_dirs(cfg)
        payload["config"] = _public_config(cfg)
        payload["warnings"] = config_followup_warnings(cfg)
    except Exception as exc:
        payload.update({"ok": False, "error": str(exc), "warnings": []})
        return payload

    kb = None
    try:
        kb = KB(cfg["storage"]["duckdb_path"])
        payload["table_counts"] = _table_counts(kb)
        payload["recent_pipeline_runs"] = _records(
            kb.query_df(
                """
                SELECT stage, started_at, finished_at, status, details
                FROM pipeline_runs
                ORDER BY finished_at DESC NULLS LAST, started_at DESC NULLS LAST
                LIMIT 50
                """
            )
        )
        payload["repo_status"] = _records(
            kb.query_df(
                """
                SELECT stage, repo_id, status, started_at, finished_at, item_count, details
                FROM pipeline_repo_status
                ORDER BY finished_at DESC NULLS LAST, started_at DESC NULLS LAST
                LIMIT 250
                """
            )
        )
    except Exception as exc:
        payload["database_error"] = str(exc)
        payload.setdefault("table_counts", {})
        payload.setdefault("recent_pipeline_runs", [])
        payload.setdefault("repo_status", [])
    finally:
        if kb is not None:
            kb.close()
    return payload


def _table_counts(kb: KB) -> dict[str, int]:
    tables = _records(
        kb.query_df(
            """
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'main' AND table_type = 'BASE TABLE'
            ORDER BY table_name
            """
        )
    )
    counts: dict[str, int] = {}
    for row in tables:
        table = str(row.get("table_name") or "")
        if not table:
            continue
        try:
            counts[table] = int(kb.query_df(f"SELECT COUNT(*) AS n FROM {table}").iloc[0]["n"])
        except Exception:
            counts[table] = 0
    return counts


def _records(df: Any) -> list[dict[str, Any]]:
    if df is None or getattr(df, "empty", False):
        return []
    return df.to_dict("records")


def _public_config(cfg: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in cfg.items() if not str(k).startswith("_")}

