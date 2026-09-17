"""Watch loop: watchdog observer + debounced single-writer worker.

One thread observes (watchdog), the main thread reindexes. DuckDB gets
exactly one writer at a time inside this process.
"""
from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from proofline.config import ensure_dirs, load_config
from proofline.logging_utils import console, setup_logging
from proofline.storage import KB
from proofline.utils import now_iso
from proofline.watch import (
    DebouncedQueue,
    EventCallback,
    WatchConfig,
    coalesce_events,
    describe_batch,
    emit,
    filter_indexable,
    should_ignore_path,
    watch_config_from_dict,
)
from proofline.watch.embeddings import refresh_embeddings
from proofline.watch.events import RepoEventHandler, start_observer
from proofline.watch.reindex import refresh_files, refresh_repo_ingest
from proofline.watch.repos import resolve_watched_repo
from proofline.watch.slow import (
    head_changed,
    refresh_api_static,
    refresh_derived,
    refresh_git_blame,
    refresh_git_history,
)


class Watcher:
    def __init__(self, config_path, *, once=False, full_on_start=False,
                 jsonl=False, quiet=False, on_event=None):
        self.config_path = config_path
        self.once = once
        self.full_on_start = full_on_start
        self.jsonl = jsonl
        self.quiet = quiet
        self.on_event = on_event
        self._stop = threading.Event()
        self.cfg: Dict[str, Any] = {}
        self.wcfg: Optional[WatchConfig] = None
        self.repo_id = ""
        self.repo_path = Path.cwd()
        self.queue: Optional[DebouncedQueue] = None
        self._embedder: Any = None
        self._last_slow = 0.0
        self.stats = {"batches": 0, "files": 0, "chunks": 0, "errors": 0}
    def stop(self) -> None:
        self._stop.set()

    def run(self) -> int:
        setup_logging()
        self.cfg = load_config(self.config_path)
        ensure_dirs(self.cfg)
        kb = KB(self.cfg["storage"]["duckdb_path"])
        try:
            resolved = resolve_watched_repo(kb, Path.cwd(), (self.cfg.get("repos") or {}).get("root"))
        finally:
            kb.close()
        self.repo_id = str(resolved["repo_id"])
        self.repo_path = Path(str(resolved["repo_path"]))
        self.wcfg = watch_config_from_dict(self.cfg, self.repo_path)
        self.queue = DebouncedQueue(self.wcfg.debounce_seconds)
        assert self.wcfg is not None and self.queue is not None
        if not resolved.get("matched"):
            self._say(f"[yellow]repo '{self.repo_id}' not in inventory yet; initial index will create it.[/yellow]")
        if self.full_on_start:
            self._full_index(reason="full-on-start")
            if self.once:
                return 0
        elif self.once:
            self._reindex_batch({"modified": [], "deleted": []}, reason="once")
            return 0
        handler = RepoEventHandler(self.repo_path, self.queue, self.wcfg.exclude_dirs)
        observer = start_observer(handler, self.repo_path)
        emit(self.on_event, "watch_started",
             {"repo_id": self.repo_id, "repo_path": str(self.repo_path),
              "debounce_seconds": self.wcfg.debounce_seconds})
        self._say(f"[green]watching {self.repo_path} (repo: {self.repo_id}) - Ctrl+C to stop[/green]")
        try:
            while not self._stop.is_set():
                assert self.queue is not None
                batch = self.queue.drain(timeout=1.0)
                if batch:
                    self._reindex_batch(coalesce_events(batch), reason="fs-events")
                else:
                    self._maybe_slow_lane(idle=True)
        except KeyboardInterrupt:
            pass
        finally:
            try:
                observer.stop()
                observer.join(timeout=10)
            except Exception:
                pass
        emit(self.on_event, "watch_stopped", {"repo_id": self.repo_id, "stats": dict(self.stats)})
        return 0
    def _kb(self) -> KB:
        return KB(self.cfg["storage"]["duckdb_path"])

    def _full_index(self, reason: str) -> None:
        emit(self.on_event, "full_index_start", {"repo_id": self.repo_id, "reason": reason})
        self._say(f"watch: full index ({reason}) for {self.repo_id}...")
        kb = self._kb()
        try:
            assert self.wcfg is not None
            refresh_repo_ingest(kb, self.cfg, self.repo_path, self.repo_id, on_event=self.on_event)
            refresh_files(kb, self.cfg, self.repo_path, self.repo_id,
                          self._all_indexable_rels(kb), [], on_event=self.on_event)
            if "api_surface" in self.wcfg.fast_lane or "static_edges" in self.wcfg.fast_lane:
                refresh_api_static(kb, self.cfg, self.repo_id, on_event=self.on_event)
            emb_cfg = (self.cfg.get("indexing", {}) or {}).get("embeddings", {}) or {}
            if "embeddings" in self.wcfg.fast_lane and emb_cfg.get("enabled", True):
                refresh_embeddings(kb, self.cfg, self.repo_id,
                                   embedder=self._embedder_cached(), on_event=self.on_event)
            self._run_slow_lane(kb)
        finally:
            kb.close()
        self._last_slow = time.monotonic()
        emit(self.on_event, "full_index_done", {"repo_id": self.repo_id})

    def _all_indexable_rels(self, kb: KB) -> List[str]:
        rows = kb.query_df("SELECT rel_path FROM repo_files WHERE repo_id = ?", [self.repo_id])
        if rows.empty:
            return []
        assert self.wcfg is not None
        return filter_indexable(
            rows["rel_path"].fillna("").astype(str).tolist(),
            exclude_dirs=self.wcfg.exclude_dirs,
            include_extensions=self.wcfg.include_extensions or None,
            repo_root=self.repo_path, max_file_mb=self.wcfg.max_file_mb,
        )
    def _reindex_batch(self, coalesced: Dict[str, Any], reason: str) -> None:
        assert self.wcfg is not None
        kb = self._kb()
        try:
            # repo_ingest first: it rescans the worktree (sha1/size/kind per
            # file), so refresh_files can diff fingerprints. For --once with
            # an empty batch this also picks up uncommitted edits.
            refresh_repo_ingest(kb, self.cfg, self.repo_path, self.repo_id, on_event=self.on_event)
            if reason == "once" and not coalesced.get("modified") and not coalesced.get("deleted"):
                modified, deleted = self._dirty_rels(kb)
            else:
                modified = filter_indexable(
                    coalesced.get("modified", []), exclude_dirs=self.wcfg.exclude_dirs,
                    include_extensions=self.wcfg.include_extensions or None,
                    repo_root=self.repo_path, max_file_mb=self.wcfg.max_file_mb,
                )
                deleted = [r for r in coalesced.get("deleted", []) if r not in set(modified)]
            if not modified and not deleted:
                self._say("watch: no changes -> 0 files, 0 chunks in 0.0s")
                return
            started = time.monotonic()
            emit(self.on_event, "batch_start",
                 {"repo_id": self.repo_id, "reason": reason,
                  "modified": modified[:50], "deleted": deleted[:50]})
            files_stat = refresh_files(kb, self.cfg, self.repo_path, self.repo_id,
                                       modified, deleted, on_event=self.on_event)
            if "api_surface" in self.wcfg.fast_lane or "static_edges" in self.wcfg.fast_lane:
                refresh_api_static(kb, self.cfg, self.repo_id, on_event=self.on_event)
            emb_cfg = (self.cfg.get("indexing", {}) or {}).get("embeddings", {}) or {}
            if "embeddings" in self.wcfg.fast_lane and emb_cfg.get("enabled", True):
                refresh_embeddings(kb, self.cfg, self.repo_id,
                                   embedder=self._embedder_cached(), on_event=self.on_event)
            self._maybe_slow_lane(kb=kb)
        except Exception as e:
            self.stats["errors"] += 1
            emit(self.on_event, "batch_error", {"repo_id": self.repo_id, "error": str(e)})
            self._say(f"[red]watch batch failed: {e}[/red]")
        else:
            self.stats["batches"] += 1
            self.stats["files"] += int(files_stat.get("files", 0))
            self.stats["chunks"] += int(files_stat.get("chunks", 0))
            elapsed = time.monotonic() - started
            emit(self.on_event, "batch_done",
                 {"repo_id": self.repo_id, "files": files_stat.get("files", 0),
                  "chunks": files_stat.get("chunks", 0), "elapsed_seconds": round(elapsed, 2)})
            self._say(f"watch: {describe_batch({'modified': modified, 'deleted': deleted})} "
                      f"-> {files_stat.get('files', 0)} files, "
                      f"{files_stat.get('chunks', 0)} chunks in {elapsed:.1f}s")
        finally:
            kb.close()

    def _dirty_rels(self, kb: KB) -> tuple[list, list]:
        """Diff repo_files against code_index_file_status fingerprints.

        Used by --once (no fs events): finds files whose content changed
        since the last index, plus indexed files missing from worktree.
        """
        from proofline.extractors.code_index import file_fingerprint
        from proofline.pipeline.runner import _graph_symbols_by_path, _load_code_index_graph_symbols

        assert self.wcfg is not None
        graph_by_path = _graph_symbols_by_path(_load_code_index_graph_symbols(kb, self.cfg, self.repo_id))
        files = kb.query_df("SELECT * FROM repo_files WHERE repo_id = ?", [self.repo_id])
        status = kb.query_df(
            "SELECT rel_path, file_fingerprint FROM code_index_file_status WHERE repo_id = ?", [self.repo_id])
        prev = {str(r["rel_path"]): str(r.get("file_fingerprint") or "")
                for r in status.to_dict("records")} if not status.empty else {}
        modified: list = []
        for row in files.to_dict("records") if not files.empty else []:
            rel = str(row.get("rel_path") or "")
            if should_ignore_path(rel, self.wcfg.exclude_dirs):
                continue
            if prev.get(rel) != file_fingerprint(row, self.cfg, graph_by_path.get(rel, [])):
                modified.append(rel)
        current = {str(r.get("rel_path")) for r in files.to_dict("records")} if not files.empty else set()
        deleted = [rel for rel in prev if rel not in current]
        return filter_indexable(
            modified, exclude_dirs=self.wcfg.exclude_dirs,
            include_extensions=self.wcfg.include_extensions or None,
            repo_root=self.repo_path, max_file_mb=self.wcfg.max_file_mb,
        ), deleted

    def _embedder_cached(self) -> Any:
        if self._embedder is None:
            from proofline.extractors.embeddings import load_embedder

            try:
                self._embedder = load_embedder(self.cfg.get("indexing", {}).get("embeddings", {}))
            except Exception as e:
                self._say(f"[yellow]embeddings unavailable, chunks+FTS still update: {e}[/yellow]")
                self._embedder = False
        return self._embedder if self._embedder is not False else None
    def _maybe_slow_lane(self, *, idle: bool = False, kb: Optional[KB] = None) -> None:
        assert self.wcfg is not None
        if not self.wcfg.slow_lane:
            return
        due = (time.monotonic() - self._last_slow) >= self.wcfg.slow_lane_minutes * 60
        own_kb = kb is None
        handle = kb if kb is not None else self._kb()
        try:
            moved = head_changed(handle, self.repo_path, self.repo_id)
            if moved or (due and idle):
                self._run_slow_lane(handle)
                self._last_slow = time.monotonic()
        finally:
            if own_kb:
                handle.close()

    def _run_slow_lane(self, kb: KB) -> None:
        assert self.wcfg is not None
        lane = set(self.wcfg.slow_lane)
        emit(self.on_event, "slow_lane_start", {"repo_id": self.repo_id, "stages": sorted(lane)})
        if "git_history" in lane:
            refresh_git_history(kb, self.cfg, self.repo_path, self.repo_id, on_event=self.on_event)
        if "git_blame" in lane:
            refresh_git_blame(kb, self.cfg, self.repo_path, self.repo_id, on_event=self.on_event)
        if {"entity_resolution", "graph", "endpoint_map", "capabilities"} & lane:
            refresh_derived(kb, self.cfg, self.repo_id, on_event=self.on_event)
        import pandas as pd

        kb.append_df("pipeline_runs", pd.DataFrame([{
            "stage": "watch_slow_lane", "started_at": now_iso(), "finished_at": now_iso(),
            "status": "ok", "details": f"repo={self.repo_id}",
        }]))
        emit(self.on_event, "slow_lane_done", {"repo_id": self.repo_id})

    def _say(self, message: str) -> None:
        if self.quiet:
            return
        if self.jsonl:
            sys.stdout.write(json.dumps({"event": "log", "message": message}) + "\n")
            sys.stdout.flush()
            return
        console.print(message)
