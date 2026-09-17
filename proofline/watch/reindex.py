"""Incremental reindex of exactly one repo: ingest + chunks + FTS.

Everything here is scoped by ``repo_id``. Mirrors the batch logic in
``pipeline/runner.py`` but only for the watched repo's changed files.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from proofline.extractors.code_index import (
    chunks_for_file,
    delete_sqlite_fts_file,
    ensure_sqlite_fts,
    file_fingerprint,
    insert_sqlite_fts,
    repo_files_fingerprint,
)
from proofline.extractors.repo import EXT_LANG, detect_kind, file_sha1, repo_source_fingerprint, scan_repo
from proofline.pipeline.repo_jobs import mark_repo_stage
from proofline.utils import now_iso
from proofline.watch import EventCallback, emit


def refresh_repo_ingest(kb, cfg, repo_path, repo_id, *, on_event=None):
    fingerprint = repo_source_fingerprint(repo_path, cfg)
    started = now_iso()
    inv, files, ownership, history = scan_repo(repo_path, cfg)
    inv["repo_id"] = repo_id
    for row in files:
        row["repo_id"] = repo_id
    for row in ownership:
        if str(row.get("entity_id") or "").startswith("repo:"):
            row["entity_id"] = f"repo:{repo_id}"
    for row in history:
        row["repo_id"] = repo_id
    mark_repo_stage(kb, "repo_ingest", repo_id, fingerprint, "running", started_at=started)
    kb.execute("DELETE FROM repo_inventory WHERE repo_id = ?", [repo_id])
    kb.execute("DELETE FROM repo_files WHERE repo_id = ?", [repo_id])
    kb.execute("DELETE FROM ownership WHERE entity_id = ?", [f"repo:{repo_id}"])
    kb.execute("DELETE FROM repo_git_history WHERE repo_id = ?", [repo_id])
    kb.append_df("repo_inventory", pd.DataFrame([inv]))
    if files:
        kb.append_df("repo_files", pd.DataFrame(files))
    if ownership:
        kb.append_df("ownership", pd.DataFrame(ownership))
    if history:
        kb.append_df("repo_git_history", pd.DataFrame(history))
    mark_repo_stage(kb, "repo_ingest", repo_id, fingerprint, "ok", started_at=started, item_count=len(files))
    emit(on_event, "repo_ingest", {"repo_id": repo_id, "files": len(files)})
    return {"repo_id": repo_id, "files": len(files), "fingerprint": fingerprint}
def _indexable_row(repo_path, repo_id, rel, cfg):
    abs_path = repo_path / rel
    try:
        if not abs_path.is_file():
            return None
        size = abs_path.stat().st_size
    except OSError:
        return None
    max_bytes = int(float(cfg.get("repos", {}).get("max_file_mb", 5)) * 1024 * 1024)
    if size > max_bytes:
        return None
    ext = abs_path.suffix.lower() or abs_path.name
    allowed = set(cfg.get("repos", {}).get("include_extensions", []) or [])
    if allowed and ext not in allowed and abs_path.name not in allowed:
        if ext not in EXT_LANG:
            return None
    return {
        "repo_id": repo_id,
        "path": str(abs_path),
        "rel_path": rel,
        "ext": ext,
        "size_bytes": size,
        "kind": detect_kind(abs_path, rel),
        "sha1": file_sha1(abs_path),
        "indexed_at": now_iso(),
    }


def refresh_files(kb, cfg, repo_path, repo_id, modified, deleted, *, on_event=None):
    indexing_cfg = cfg.get("indexing", {})
    use_fts = bool(indexing_cfg.get("lexical_fts", True))
    sqlite_fts_path = cfg["storage"]["sqlite_fts_path"]
    if use_fts:
        ensure_sqlite_fts(sqlite_fts_path)
    from proofline.pipeline.runner import _graph_symbols_by_path, _load_code_index_graph_symbols

    graph_by_path = _graph_symbols_by_path(_load_code_index_graph_symbols(kb, cfg, repo_id))
    for rel in deleted:
        kb.execute("DELETE FROM code_chunks WHERE repo_id = ? AND rel_path = ?", [repo_id, rel])
        kb.execute("DELETE FROM code_index_file_status WHERE repo_id = ? AND rel_path = ?", [repo_id, rel])
        if use_fts:
            try:
                delete_sqlite_fts_file(sqlite_fts_path, repo_id, rel)
            except Exception:
                pass
    if deleted:
        emit(on_event, "files_deleted", {"repo_id": repo_id, "count": len(deleted)})
    status_rows = kb.query_df(
        "SELECT rel_path, file_fingerprint, chunk_count FROM code_index_file_status WHERE repo_id = ?", [repo_id]
    )
    status_by_path = {str(r["rel_path"]): r for r in status_rows.to_dict("records")} if not status_rows.empty else {}
    changed = []
    for rel in modified:
        row = _indexable_row(repo_path, repo_id, rel, cfg)
        if row is None:
            continue
        fp = file_fingerprint(row, cfg, graph_by_path.get(rel, []))
        prev = status_by_path.get(rel)
        if prev and str(prev.get("file_fingerprint") or "") == fp:
            continue
        row["_graph_symbols"] = graph_by_path.get(rel, [])
        changed.append(row)
    refreshed = 0
    chunks_total = 0
    for row in changed:
        rel = str(row["rel_path"])
        chunks = chunks_for_file(
            {k: v for k, v in row.items() if not k.startswith("_")}, cfg, row.get("_graph_symbols", [])
        )
        kb.execute("DELETE FROM code_chunks WHERE repo_id = ? AND rel_path = ?", [repo_id, rel])
        kb.execute("DELETE FROM code_index_file_status WHERE repo_id = ? AND rel_path = ?", [repo_id, rel])
        if use_fts:
            try:
                delete_sqlite_fts_file(sqlite_fts_path, repo_id, rel)
            except Exception:
                pass
        if chunks:
            kb.append_df("code_chunks", pd.DataFrame(chunks))
            if use_fts:
                try:
                    insert_sqlite_fts(pd.DataFrame(chunks), sqlite_fts_path)
                except Exception:
                    pass
        kb.append_df("code_index_file_status", pd.DataFrame([{
            "repo_id": repo_id, "rel_path": rel,
            "file_fingerprint": file_fingerprint(row, cfg, row.get("_graph_symbols", [])),
            "status": "ok", "chunk_count": len(chunks),
            "indexed_at": now_iso(), "details": "watch",
        }]))
        refreshed += 1
        chunks_total += len(chunks)
    repo_files = kb.query_df("SELECT * FROM repo_files WHERE repo_id = ? ORDER BY rel_path", [repo_id])
    graph_symbols = _load_code_index_graph_symbols(kb, cfg, repo_id)
    fingerprint = repo_files_fingerprint(repo_files, cfg, graph_symbols)
    chunk_count = int(kb.query_df("SELECT COUNT(*) AS n FROM code_chunks WHERE repo_id = ?", [repo_id]).iloc[0]["n"])
    kb.execute("DELETE FROM code_index_repo_status WHERE repo_id = ?", [repo_id])
    kb.append_df("code_index_repo_status", pd.DataFrame([{
        "repo_id": repo_id, "source_fingerprint": fingerprint, "status": "ok",
        "file_count": len(repo_files), "chunk_count": chunk_count,
        "started_at": now_iso(), "finished_at": now_iso(), "details": "watch",
    }]))
    mark_repo_stage(kb, "code_index", repo_id, fingerprint, "ok", item_count=chunk_count, details="watch")
    emit(on_event, "code_index", {"repo_id": repo_id, "files": refreshed, "chunks": chunks_total})
    return {"repo_id": repo_id, "files": refreshed, "chunks": chunks_total}
