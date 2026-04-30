from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Callable, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import typer

from proofline.config import DEFAULT_CONFIG, load_config, ensure_dirs
from proofline.logging_utils import setup_logging, log_step, console
from proofline.storage import KB
from proofline.utils import now_iso
from proofline.extractors.repo import find_git_repos, repo_id_from_path, repo_source_fingerprint, scan_repo
from proofline.extractors.git_history import build_cochange_edges, extract_repo_git_blame, extract_repo_git_history
from proofline.extractors.code_index import (
    chunked_rows,
    chunks_for_file,
    delete_sqlite_fts_file,
    ensure_sqlite_fts,
    file_fingerprint,
    insert_sqlite_fts,
    repo_files_fingerprint,
)
from proofline.extractors.api_surface import parse_api_specs, extract_static_routes
from proofline.extractors.static_edges import extract_static_edges
from proofline.extractors.datadog import pull_service_dependencies, pull_service_definitions, search_spans, search_logs, build_runtime_edges_from_dd
from proofline.extractors.bigquery import pull_bq_jobs, build_table_usage
from proofline.extractors.entity_resolution import build_service_identity
from proofline.extractors.graph_build import build_graph
from proofline.extractors.endpoint_map import build_endpoint_dependency_map
from proofline.extractors.capabilities import build_capabilities
from proofline.extractors.compatibility import build_compatibility_index
from proofline.extractors.embeddings import build_code_embeddings
from proofline.extractors.graph_backend import publish_graph_backend
from proofline.extractors.code_graph import import_code_graph, run_code_graph_index
from proofline.pipeline.repo_jobs import mark_repo_stage, max_workers, repo_stage_done

app = typer.Typer(help="Proofline pipeline")


def stage_record(kb: KB, stage: str, started: str, status: str, details: str = "") -> None:
    kb.append_df("pipeline_runs", pd.DataFrame([{
        "stage": stage, "started_at": started, "finished_at": now_iso(), "status": status, "details": details,
    }]))


def run_stage(name: str, cfg: Dict[str, Any], func: Callable[[KB, Dict[str, Any]], None]) -> None:
    log_step(name)
    kb = KB(cfg["storage"]["duckdb_path"])
    started = now_iso()
    try:
        func(kb, cfg)
        stage_record(kb, name, started, "ok", "")
    except Exception as e:
        stage_record(kb, name, started, "error", str(e))
        raise
    finally:
        kb.close()


def maybe_clone_repos(cfg: Dict[str, Any]) -> None:
    repos_cfg = cfg.get("repos", {})
    urls_file = repos_cfg.get("clone_urls_file")
    if not urls_file:
        return
    urls_path = Path(urls_file)
    if not urls_path.exists():
        console.print(f"[yellow]clone_urls_file not found: {urls_path}; skipping clone[/yellow]")
        return
    root = Path(repos_cfg.get("root", "./repos"))
    root.mkdir(parents=True, exist_ok=True)
    for line in urls_path.read_text().splitlines():
        url = line.strip()
        if not url or url.startswith("#"):
            continue
        name = url.rstrip("/").split("/")[-1].replace(".git", "")
        target = root / name
        if target.exists():
            if repos_cfg.get("update_existing", False):
                console.print(f"Updating {name}")
                subprocess.run(["git", "fetch", "--all", "--prune"], cwd=target, check=False)
                subprocess.run(["git", "pull", "--ff-only"], cwd=target, check=False)
            continue
        console.print(f"Cloning {url} -> {target}")
        subprocess.run(["git", "clone", url, str(target)], check=False)


def stage_repo_ingest(kb: KB, cfg: Dict[str, Any]) -> None:
    maybe_clone_repos(cfg)
    repos = find_git_repos(Path(cfg["repos"]["root"]), cfg["repos"].get("exclude_dirs", []))
    repo_ids = {repo_id_from_path(repo) for repo in repos}
    existing_repo_ids = kb.query_df("SELECT repo_id FROM repo_inventory")
    for old_repo_id in existing_repo_ids["repo_id"].fillna("").astype(str).tolist() if not existing_repo_ids.empty else []:
        if old_repo_id and old_repo_id not in repo_ids:
            _delete_repo_ingest_rows(kb, old_repo_id)
    workers = max_workers(cfg.get("repos", {}), 1)
    scanned = 0
    skipped = 0
    file_count = 0

    try:
        from tqdm.auto import tqdm
    except Exception:
        tqdm = None

    def scan_one(repo_path: Path) -> tuple[str, str, dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
        repo_id = repo_id_from_path(repo_path)
        fingerprint = repo_source_fingerprint(repo_path, cfg)
        progress_desc = f"Scanning {repo_id}" if workers == 1 else None
        inv, fs, own, hist = scan_repo(repo_path, cfg, progress_desc=progress_desc)
        return repo_id, fingerprint, inv, fs, own, hist

    pending: list[Path] = []
    for repo in repos:
        repo_id = repo_id_from_path(repo)
        fingerprint = repo_source_fingerprint(repo, cfg)
        if repo_stage_done(kb, "repo_ingest", repo_id, fingerprint):
            existing_files = kb.query_df("SELECT COUNT(*) AS n FROM repo_files WHERE repo_id = ?", [repo_id])
            file_count += int(existing_files.iloc[0]["n"]) if not existing_files.empty else 0
            skipped += 1
            continue
        mark_repo_stage(kb, "repo_ingest", repo_id, fingerprint, "running")
        pending.append(repo)

    iterator: Any
    if workers > 1 and len(pending) > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(scan_one, repo) for repo in pending]
            iterator = as_completed(futures)
            if tqdm:
                iterator = tqdm(iterator, total=len(futures), desc="Scanning repositories", unit="repo", position=0)
            for future in iterator:
                repo_id, fingerprint, inv, fs, own, hist = future.result()
                _replace_repo_ingest_rows(kb, repo_id, inv, fs, own, hist)
                mark_repo_stage(kb, "repo_ingest", repo_id, fingerprint, "ok", item_count=len(fs))
                scanned += 1
                file_count += len(fs)
    else:
        iterator = pending
        if tqdm:
            iterator = tqdm(iterator, total=len(pending), desc="Scanning repositories", unit="repo", position=0)
        for repo in iterator:
            repo_id, fingerprint, inv, fs, own, hist = scan_one(repo)
            _replace_repo_ingest_rows(kb, repo_id, inv, fs, own, hist)
            mark_repo_stage(kb, "repo_ingest", repo_id, fingerprint, "ok", item_count=len(fs))
            scanned += 1
            file_count += len(fs)
    console.print(f"repos: {len(repos)}, files: {file_count}, scanned={scanned}, skipped={skipped}")


def _replace_repo_ingest_rows(
    kb: KB,
    repo_id: str,
    inv: dict[str, Any],
    files: list[dict[str, Any]],
    ownership: list[dict[str, Any]],
    history: list[dict[str, Any]],
) -> None:
    _delete_repo_ingest_rows(kb, repo_id)
    kb.append_df("repo_inventory", pd.DataFrame([inv]))
    if files:
        kb.append_df("repo_files", pd.DataFrame(files))
    if ownership:
        kb.append_df("ownership", pd.DataFrame(ownership))
    if history:
        kb.append_df("repo_git_history", pd.DataFrame(history))


def _delete_repo_ingest_rows(kb: KB, repo_id: str) -> None:
    kb.execute("DELETE FROM repo_inventory WHERE repo_id = ?", [repo_id])
    kb.execute("DELETE FROM repo_files WHERE repo_id = ?", [repo_id])
    kb.execute("DELETE FROM ownership WHERE entity_id = ?", [f"repo:{repo_id}"])
    kb.execute("DELETE FROM repo_git_history WHERE repo_id = ?", [repo_id])


def stage_git_history(kb: KB, cfg: Dict[str, Any]) -> None:
    gh_cfg = dict(cfg.get("git_history") or {})
    if not gh_cfg.get("enabled", True):
        console.print("git history: disabled")
        return
    gh_cfg["current_blame"] = False
    repos = kb.query_df("SELECT repo_id, repo_path, commit_sha FROM repo_inventory ORDER BY repo_id")
    workers = max_workers(gh_cfg, 1)
    totals = {"git_commits": 0, "git_file_changes": 0, "git_patch_hunks": 0, "git_semantic_changes": 0, "git_cochange_edges": 0}
    skipped = 0

    try:
        from tqdm.auto import tqdm
    except Exception:
        tqdm = None

    def repo_fp(row: dict[str, Any]) -> str:
        return str(row.get("commit_sha") or "") + f":patch={bool(gh_cfg.get('patch_hunks', True))}:rename={bool(gh_cfg.get('rename_detection', True))}"

    def build_one(row: dict[str, Any]) -> tuple[str, str, dict[str, list[dict[str, Any]]]]:
        repo_id = str(row.get("repo_id") or "")
        fingerprint = repo_fp(row)
        local_cfg = dict(gh_cfg)
        if row.get("_existing_shas"):
            local_cfg["stop_commit_shas"] = set(row["_existing_shas"])
        progress_desc = f"History {repo_id}" if workers == 1 else None
        rows = extract_repo_git_history(Path(str(row.get("repo_path") or "")), repo_id, local_cfg, progress_desc=progress_desc)
        return repo_id, fingerprint, rows

    pending = []
    for row in repos.to_dict("records"):
        repo_id = str(row.get("repo_id") or "")
        fingerprint = repo_fp(row)
        if repo_stage_done(kb, "git_history", repo_id, fingerprint):
            skipped += 1
            continue
        existing = kb.query_df("SELECT commit_sha FROM git_commits WHERE repo_id = ?", [repo_id])
        row["_existing_shas"] = existing["commit_sha"].fillna("").astype(str).tolist() if not existing.empty else []
        mark_repo_stage(kb, "git_history", repo_id, fingerprint, "running")
        pending.append(row)

    iterator: Any
    if workers > 1 and len(pending) > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(build_one, row) for row in pending]
            iterator = as_completed(futures)
            if tqdm:
                iterator = tqdm(iterator, total=len(futures), desc="Indexing git history", unit="repo", position=0)
            for future in iterator:
                repo_id, fingerprint, rows = future.result()
                counts = _append_repo_git_history(kb, repo_id, rows)
                mark_repo_stage(kb, "git_history", repo_id, fingerprint, "ok", item_count=counts.get("git_commits", 0))
                for key in totals:
                    totals[key] += counts.get(key, 0)
    else:
        iterator = pending
        if tqdm:
            iterator = tqdm(iterator, total=len(pending), desc="Indexing git history", unit="repo", position=0)
        for row in iterator:
            repo_id, fingerprint, rows = build_one(row)
            counts = _append_repo_git_history(kb, repo_id, rows)
            mark_repo_stage(kb, "git_history", repo_id, fingerprint, "ok", item_count=counts.get("git_commits", 0))
            for key in totals:
                totals[key] += counts.get(key, 0)

    console.print(
        "git history: "
        f"new_commits={totals['git_commits']}, "
        f"new_files={totals['git_file_changes']}, "
        f"new_hunks={totals['git_patch_hunks']}, "
        f"new_semantic={totals['git_semantic_changes']}, "
        f"cochange={totals['git_cochange_edges']}, skipped={skipped}"
    )


def _append_repo_git_history(kb: KB, repo_id: str, rows: dict[str, list[dict[str, Any]]]) -> dict[str, int]:
    counts = {key: len(value) for key, value in rows.items()}
    for table in ["git_commits", "git_file_changes", "git_patch_hunks", "git_detected_links", "git_reverts", "git_semantic_changes"]:
        values = rows.get(table) or []
        if values:
            if "commit_sha" in values[0]:
                shas = sorted({str(v.get("commit_sha") or "") for v in values if str(v.get("commit_sha") or "")})
                if shas:
                    placeholders = ",".join(["?"] * len(shas))
                    kb.execute(f"DELETE FROM {table} WHERE repo_id = ? AND commit_sha IN ({placeholders})", [repo_id] + shas)
            kb.append_df(table, pd.DataFrame(values))
    commits = kb.query_df("SELECT * FROM git_commits WHERE repo_id = ?", [repo_id]).to_dict("records")
    file_changes = kb.query_df("SELECT * FROM git_file_changes WHERE repo_id = ?", [repo_id]).to_dict("records")
    cochange = build_cochange_edges(repo_id, file_changes, {str(c.get("commit_sha") or ""): c for c in commits}, {"cochange_window_days": None, "cochange_min_count": 1})
    kb.execute("DELETE FROM git_cochange_edges WHERE from_entity LIKE ? OR to_entity LIKE ?", [f"file:{repo_id}:%", f"file:{repo_id}:%"])
    if cochange:
        kb.append_df("git_cochange_edges", pd.DataFrame(cochange))
    counts["git_cochange_edges"] = len(cochange)
    return counts


def stage_git_blame(kb: KB, cfg: Dict[str, Any]) -> None:
    gh_cfg = dict(cfg.get("git_history") or {})
    if not gh_cfg.get("enabled", True) or not gh_cfg.get("current_blame", True):
        console.print("git blame: disabled")
        return
    repos = kb.query_df("SELECT repo_id, repo_path, commit_sha FROM repo_inventory ORDER BY repo_id")
    skipped = 0
    rows_total = 0
    try:
        from tqdm.auto import tqdm
    except Exception:
        tqdm = None
    iterator = repos.to_dict("records")
    if tqdm:
        iterator = tqdm(iterator, total=len(repos), desc="Indexing git blame", unit="repo", position=0)
    for row in iterator:
        repo_id = str(row.get("repo_id") or "")
        fingerprint = str(row.get("commit_sha") or "") + ":blame"
        if repo_stage_done(kb, "git_blame", repo_id, fingerprint):
            skipped += 1
            continue
        started = now_iso()
        mark_repo_stage(kb, "git_blame", repo_id, fingerprint, "running", started_at=started)
        try:
            rows = extract_repo_git_blame(Path(str(row.get("repo_path") or "")), repo_id, gh_cfg, progress_desc=f"Blaming {repo_id}").get("git_blame_current", [])
            kb.execute("DELETE FROM git_blame_current WHERE repo_id = ?", [repo_id])
            if rows:
                kb.append_df("git_blame_current", pd.DataFrame(rows))
            rows_total += len(rows)
            mark_repo_stage(kb, "git_blame", repo_id, fingerprint, "ok", started_at=started, item_count=len(rows))
        except Exception as e:
            mark_repo_stage(kb, "git_blame", repo_id, fingerprint, "error", started_at=started, details=str(e))
            raise
    console.print(f"git blame: rows={rows_total}, skipped={skipped}")


def stage_code_index(kb: KB, cfg: Dict[str, Any]) -> None:
    repos = kb.query_df("SELECT repo_id FROM repo_inventory ORDER BY repo_id")
    if repos.empty:
        console.print("chunks: no repositories")
        return

    use_fts = bool(cfg.get("indexing", {}).get("lexical_fts", True))
    sqlite_fts_path = cfg["storage"]["sqlite_fts_path"]
    if use_fts:
        ensure_sqlite_fts(sqlite_fts_path)
    write_batch_size = max(1, int(cfg.get("indexing", {}).get("chunk_write_batch_size", 5000)))
    workers = max_workers(cfg.get("indexing", {}), 1)

    try:
        from tqdm.auto import tqdm
    except Exception:
        tqdm = None

    total_chunks = 0
    skipped = 0
    iterator = repos["repo_id"].fillna("").astype(str).tolist()
    if tqdm:
        iterator = tqdm(iterator, total=len(repos), desc="Indexing repositories", unit="repo", position=0)

    for repo_id in iterator:
        if not repo_id:
            continue
        repo_files = kb.query_df("SELECT * FROM repo_files WHERE repo_id = ? ORDER BY rel_path", [repo_id])
        fingerprint = repo_files_fingerprint(repo_files)
        existing = kb.query_df(
            """
            SELECT status, source_fingerprint, chunk_count
            FROM code_index_repo_status
            WHERE repo_id = ?
            ORDER BY finished_at DESC NULLS LAST
            LIMIT 1
            """,
            [repo_id],
        )
        if (
            not existing.empty
            and str(existing.iloc[0].get("status") or "") == "ok"
            and str(existing.iloc[0].get("source_fingerprint") or "") == fingerprint
        ):
            expected_chunks = int(existing.iloc[0].get("chunk_count") or 0)
            actual_chunks = int(kb.query_df("SELECT COUNT(*) AS n FROM code_chunks WHERE repo_id = ?", [repo_id]).iloc[0]["n"])
            fts_ready = (not use_fts) or Path(sqlite_fts_path).exists()
            if actual_chunks == expected_chunks and fts_ready:
                skipped += 1
                total_chunks += expected_chunks
                mark_repo_stage(kb, "code_index", repo_id, fingerprint, "ok", item_count=expected_chunks, details="cached")
                continue

        started = now_iso()
        mark_repo_stage(kb, "code_index", repo_id, fingerprint, "running", started_at=started, item_count=0)
        kb.execute("DELETE FROM code_index_repo_status WHERE repo_id = ?", [repo_id])
        kb.append_df("code_index_repo_status", pd.DataFrame([{
            "repo_id": repo_id,
            "source_fingerprint": fingerprint,
            "status": "running",
            "file_count": len(repo_files),
            "chunk_count": 0,
            "started_at": started,
            "finished_at": "",
            "details": "",
        }]))
        chunk_count = 0
        try:
            current_paths = set(repo_files["rel_path"].fillna("").astype(str).tolist()) if not repo_files.empty else set()
            old_paths = kb.query_df("SELECT rel_path FROM code_index_file_status WHERE repo_id = ?", [repo_id])
            for rel_path in old_paths["rel_path"].fillna("").astype(str).tolist() if not old_paths.empty else []:
                if rel_path not in current_paths:
                    kb.execute("DELETE FROM code_chunks WHERE repo_id = ? AND rel_path = ?", [repo_id, rel_path])
                    kb.execute("DELETE FROM code_index_file_status WHERE repo_id = ? AND rel_path = ?", [repo_id, rel_path])
                    if use_fts:
                        delete_sqlite_fts_file(sqlite_fts_path, repo_id, rel_path)

            changed_files: list[dict[str, Any]] = []
            existing_status = kb.query_df("SELECT rel_path, file_fingerprint, chunk_count FROM code_index_file_status WHERE repo_id = ?", [repo_id])
            status_by_path = {str(r["rel_path"]): r for r in existing_status.to_dict("records")} if not existing_status.empty else {}
            for row in repo_files.to_dict("records"):
                rel_path = str(row.get("rel_path") or "")
                fp = file_fingerprint(row)
                status = status_by_path.get(rel_path)
                if status and str(status.get("file_fingerprint") or "") == fp:
                    chunk_count += int(status.get("chunk_count") or 0)
                    continue
                changed_files.append(row)

            file_iterator: Any
            if workers > 1 and len(changed_files) > 1:
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    futures = [pool.submit(chunks_for_file, row, cfg) for row in changed_files]
                    file_iterator = zip(changed_files, futures)
                    if tqdm:
                        file_iterator = tqdm(file_iterator, total=len(changed_files), desc=f"Chunking {repo_id}", unit="file", position=1, leave=False)
                    for row, future in file_iterator:
                        chunks = future.result()
                        chunk_count += _replace_file_chunks(kb, row, chunks, sqlite_fts_path, use_fts)
            else:
                file_iterator = changed_files
                if tqdm:
                    file_iterator = tqdm(file_iterator, total=len(changed_files), desc=f"Chunking {repo_id}", unit="file", position=1, leave=False)
                for row in file_iterator:
                    chunks = chunks_for_file(row, cfg)
                    chunk_count += _replace_file_chunks(kb, row, chunks, sqlite_fts_path, use_fts)

            kb.execute("DELETE FROM code_index_repo_status WHERE repo_id = ?", [repo_id])
            kb.append_df("code_index_repo_status", pd.DataFrame([{
                "repo_id": repo_id,
                "source_fingerprint": fingerprint,
                "status": "ok",
                "file_count": len(repo_files),
                "chunk_count": chunk_count,
                "started_at": started,
                "finished_at": now_iso(),
                "details": "",
            }]))
            mark_repo_stage(kb, "code_index", repo_id, fingerprint, "ok", started_at=started, item_count=chunk_count)
            total_chunks += chunk_count
        except Exception as e:
            kb.execute("DELETE FROM code_index_repo_status WHERE repo_id = ?", [repo_id])
            kb.append_df("code_index_repo_status", pd.DataFrame([{
                "repo_id": repo_id,
                "source_fingerprint": fingerprint,
                "status": "error",
                "file_count": len(repo_files),
                "chunk_count": chunk_count,
                "started_at": started,
                "finished_at": now_iso(),
                "details": str(e),
            }]))
            mark_repo_stage(kb, "code_index", repo_id, fingerprint, "error", started_at=started, item_count=chunk_count, details=str(e))
            raise
    console.print(f"chunks: {total_chunks}, repos={len(repos)}, skipped={skipped}")


def _replace_file_chunks(
    kb: KB,
    row: dict[str, Any],
    chunks: list[dict[str, Any]],
    sqlite_fts_path: str,
    use_fts: bool,
) -> int:
    repo_id = str(row.get("repo_id") or "")
    rel_path = str(row.get("rel_path") or "")
    kb.execute("DELETE FROM code_chunks WHERE repo_id = ? AND rel_path = ?", [repo_id, rel_path])
    kb.execute("DELETE FROM code_index_file_status WHERE repo_id = ? AND rel_path = ?", [repo_id, rel_path])
    if use_fts:
        delete_sqlite_fts_file(sqlite_fts_path, repo_id, rel_path)
    if chunks:
        df = pd.DataFrame(chunks)
        for batch in chunked_rows(df.to_dict("records"), 5000):
            batch_df = pd.DataFrame(batch)
            kb.append_df("code_chunks", batch_df)
            if use_fts:
                insert_sqlite_fts(batch_df, sqlite_fts_path)
    kb.append_df("code_index_file_status", pd.DataFrame([{
        "repo_id": repo_id,
        "rel_path": rel_path,
        "file_fingerprint": file_fingerprint(row),
        "status": "ok",
        "chunk_count": len(chunks),
        "indexed_at": now_iso(),
        "details": "",
    }]))
    return len(chunks)


def stage_embeddings(kb: KB, cfg: Dict[str, Any]) -> None:
    meta, details = build_code_embeddings(kb, cfg)
    if not meta.empty:
        kb.replace_df("code_embedding_index", meta)
    console.print(f"embeddings: {details}")


def stage_api_surface(kb: KB, cfg: Dict[str, Any]) -> None:
    inv = kb.query_df("SELECT * FROM repo_inventory")
    files = kb.query_df("SELECT * FROM repo_files")
    contracts, endpoints1 = parse_api_specs(inv, files)
    endpoints2 = extract_static_routes(inv, files)
    endpoints = pd.concat([endpoints1, endpoints2], ignore_index=True) if not endpoints1.empty or not endpoints2.empty else pd.DataFrame()
    kb.replace_df("api_contracts", contracts)
    kb.replace_df("api_endpoints", endpoints)
    console.print(f"contracts: {len(contracts)}, endpoints: {len(endpoints)}")


def stage_code_graph(kb: KB, cfg: Dict[str, Any]) -> None:
    if cfg.get("code_graph", {}).get("clear_existing", True):
        kb.execute("DELETE FROM code_graph_runs")
        kb.execute("DELETE FROM code_graph_symbols")
        kb.execute("DELETE FROM code_graph_edges")
    runs = run_code_graph_index(kb, cfg)
    symbols, edges = import_code_graph(kb, cfg)
    kb.replace_df("code_graph_symbols", symbols)
    kb.replace_df("code_graph_edges", edges)
    counts = runs["status"].value_counts().to_dict() if not runs.empty else {}
    console.print(f"code graph runs: {counts}, symbols: {len(symbols)}, edges: {len(edges)}")


def stage_static_edges(kb: KB, cfg: Dict[str, Any]) -> None:
    inv = kb.query_df("SELECT * FROM repo_inventory")
    files = kb.query_df("SELECT * FROM repo_files")
    edges = extract_static_edges(inv, files)
    kb.replace_df("static_edges", edges)
    console.print(f"static edges: {len(edges)}")


def stage_datadog(kb: KB, cfg: Dict[str, Any]) -> None:
    services, dd_edges = pull_service_dependencies(cfg)
    svc_defs = pull_service_definitions(cfg)
    if not svc_defs.empty:
        services = pd.concat([services, svc_defs], ignore_index=True) if not services.empty else svc_defs
    spans = search_spans(cfg)
    logs = search_logs(cfg)
    runtime_svc, runtime_ep = build_runtime_edges_from_dd(spans, logs, dd_edges, cfg.get("datadog", {}).get("windows_days", [30]))
    kb.replace_df("datadog_services", services)
    kb.replace_df("datadog_service_edges", dd_edges)
    kb.replace_df("datadog_spans", spans)
    kb.replace_df("datadog_logs", logs)
    kb.replace_df("runtime_service_edges", runtime_svc)
    kb.replace_df("runtime_endpoint_edges", runtime_ep)
    console.print(f"dd services rows: {len(services)}, dd service edges: {len(dd_edges)}, spans: {len(spans)}, logs: {len(logs)}, runtime edges: {len(runtime_svc)}, endpoint edges: {len(runtime_ep)}")


def stage_bigquery(kb: KB, cfg: Dict[str, Any]) -> None:
    jobs = pull_bq_jobs(cfg)
    usage = build_table_usage(jobs)
    kb.replace_df("bq_jobs", jobs)
    kb.replace_df("bq_table_usage", usage)
    console.print(f"bq jobs: {len(jobs)}, bq usage rows: {len(usage)}")


def stage_entity_resolution(kb: KB, cfg: Dict[str, Any]) -> None:
    inv = kb.query_df("SELECT * FROM repo_inventory")
    dd_services = kb.query_df("SELECT * FROM datadog_services")
    dd_edges = kb.query_df("SELECT * FROM datadog_service_edges")
    own = kb.query_df("SELECT * FROM ownership")
    api = kb.query_df("SELECT * FROM api_endpoints")
    static = kb.query_df("SELECT * FROM static_edges")
    bq = kb.query_df("SELECT * FROM bq_table_usage")
    service_identity, aliases, unresolved = build_service_identity(inv, dd_services, dd_edges, own, api, static, bq)
    kb.replace_df("service_identity", service_identity)
    kb.replace_df("entity_aliases", aliases)
    kb.replace_df("unresolved_entities", unresolved)
    console.print(f"services: {len(service_identity)}, aliases: {len(aliases)}, unresolved: {len(unresolved)}")


def stage_graph(kb: KB, cfg: Dict[str, Any]) -> None:
    nodes, edges, evidence = build_graph(
        kb.query_df("SELECT * FROM repo_inventory"),
        kb.query_df("SELECT * FROM service_identity"),
        kb.query_df("SELECT * FROM entity_aliases"),
        kb.query_df("SELECT * FROM api_endpoints"),
        kb.query_df("SELECT * FROM static_edges"),
        kb.query_df("SELECT * FROM runtime_service_edges"),
        kb.query_df("SELECT * FROM runtime_endpoint_edges"),
        kb.query_df("SELECT * FROM bq_table_usage"),
        kb.query_df("SELECT * FROM ownership"),
        kb.query_df("SELECT * FROM code_graph_symbols"),
        kb.query_df("SELECT * FROM code_graph_edges"),
        kb.query_df("SELECT * FROM git_commits"),
        kb.query_df("SELECT * FROM git_file_changes"),
        kb.query_df("SELECT * FROM git_semantic_changes"),
        kb.query_df("SELECT * FROM git_cochange_edges"),
    )
    kb.replace_df("nodes", nodes)
    kb.replace_df("edges", edges)
    kb.replace_df("evidence", evidence)
    console.print(f"nodes: {len(nodes)}, edges: {len(edges)}, evidence: {len(evidence)}")


def stage_endpoint_map(kb: KB, cfg: Dict[str, Any]) -> None:
    df = build_endpoint_dependency_map(
        kb.query_df("SELECT * FROM api_endpoints"),
        kb.query_df("SELECT * FROM runtime_endpoint_edges"),
        kb.query_df("SELECT * FROM static_edges"),
        kb.query_df("SELECT * FROM service_identity"),
    )
    kb.replace_df("endpoint_dependency_map", df)
    console.print(f"endpoint dependency rows: {len(df)}")


def stage_capabilities(kb: KB, cfg: Dict[str, Any]) -> None:
    caps = build_capabilities(kb.query_df("SELECT * FROM api_endpoints"), kb.query_df("SELECT * FROM bq_table_usage"), kb.query_df("SELECT * FROM service_identity"))
    kb.replace_df("data_capabilities", caps)
    compat = build_compatibility_index(kb.query_df("SELECT * FROM api_endpoints"), kb.query_df("SELECT * FROM static_edges"), kb.query_df("SELECT * FROM runtime_service_edges"))
    kb.replace_df("compatibility_index", compat)
    console.print(f"capabilities: {len(caps)}, compatibility risk entities: {len(compat)}")


def stage_publish(kb: KB, cfg: Dict[str, Any]) -> None:
    result = publish_graph_backend(kb, cfg)
    kb.append_df("graph_backend_exports", result)
    row = result.iloc[0].to_dict() if not result.empty else {}
    details = str(row.get("details") or "")
    suffix = f", details={details[:240]}" if details and row.get("status") != "ok" else ""
    console.print(f"publish: {row.get('status')}, nodes={row.get('node_count')}, edges={row.get('edge_count')}{suffix}")


def stage_smoke(kb: KB, cfg: Dict[str, Any]) -> None:
    tables = [
        "repo_inventory",
        "git_commits",
        "git_file_changes",
        "git_semantic_changes",
        "git_cochange_edges",
        "service_identity",
        "api_endpoints",
        "code_embedding_index",
        "code_graph_symbols",
        "code_graph_edges",
        "edges",
        "endpoint_dependency_map",
        "data_capabilities",
    ]
    rows = []
    for t in tables:
        n = kb.query_df(f"SELECT COUNT(*) AS n FROM {t}").iloc[0]["n"]
        rows.append({"table": t, "rows": int(n)})
    console.print(pd.DataFrame(rows).to_string(index=False))


STAGES: Dict[str, Callable[[KB, Dict[str, Any]], None]] = {
    "repo_ingest": stage_repo_ingest,
    "git_history": stage_git_history,
    "git_blame": stage_git_blame,
    "code_index": stage_code_index,
    "embeddings": stage_embeddings,
    "api_surface": stage_api_surface,
    "code_graph": stage_code_graph,
    "static_edges": stage_static_edges,
    "datadog": stage_datadog,
    "bigquery": stage_bigquery,
    "entity_resolution": stage_entity_resolution,
    "graph": stage_graph,
    "endpoint_map": stage_endpoint_map,
    "capabilities": stage_capabilities,
    "publish": stage_publish,
    "smoke": stage_smoke,
}

FULL_ORDER = [
    "repo_ingest", "git_history", "git_blame", "code_index", "embeddings", "api_surface", "code_graph", "static_edges", "datadog", "bigquery",
    "entity_resolution", "graph", "endpoint_map", "capabilities", "publish", "smoke",
]

STAGE_ALIASES = {
    "repos": "repo_ingest",
    "repo": "repo_ingest",
    "history": "git_history",
    "change-history": "git_history",
    "change_history": "git_history",
    "blame": "git_blame",
    "code": "code_index",
    "api": "api_surface",
    "code-graph": "code_graph",
    "static": "static_edges",
    "runtime": "datadog",
    "data": "bigquery",
    "identity": "entity_resolution",
    "endpoints": "endpoint_map",
    "endpoint-map": "endpoint_map",
    "capability": "capabilities",
}


def resolve_stage(name: str) -> str:
    normalized = name.strip().replace("-", "_")
    return STAGE_ALIASES.get(name.strip(), STAGE_ALIASES.get(normalized, normalized))


def run_order(
    config: str = DEFAULT_CONFIG,
    from_stage: Optional[str] = None,
    to_stage: Optional[str] = None,
) -> None:
    setup_logging()
    cfg = load_config(config)
    ensure_dirs(cfg)
    order = FULL_ORDER[:]
    if from_stage:
        start = resolve_stage(from_stage)
        order = order[order.index(start):]
    if to_stage:
        end = resolve_stage(to_stage)
        order = order[: order.index(end) + 1]
    for s in order:
        run_stage(s, cfg, STAGES[s])


@app.command()
def full(
    config: str = typer.Option(DEFAULT_CONFIG, "--config", "-c"),
    from_stage: Optional[str] = typer.Option(None, "--from"),
    to_stage: Optional[str] = typer.Option(None, "--to"),
):
    run_order(config, from_stage, to_stage)


@app.command()
def stage(name: str, config: str = typer.Option(DEFAULT_CONFIG, "--config", "-c")):
    setup_logging()
    cfg = load_config(config)
    ensure_dirs(cfg)
    stage_name = resolve_stage(name)
    if stage_name not in STAGES:
        raise typer.BadParameter(f"Unknown stage: {name}. Known: {', '.join(STAGES)}")
    run_stage(stage_name, cfg, STAGES[stage_name])


if __name__ == "__main__":
    app()
