from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import pandas as pd
from rich.console import Console

from proofline.utils import json_dumps, now_iso, stable_id

console = Console()


def run_code_graph_index(kb, cfg: Dict[str, Any]) -> pd.DataFrame:
    cg = cfg.get("code_graph", {})
    if not cg.get("enabled", False):
        return pd.DataFrame([{
            "repo_id": "", "repo_path": "", "status": "disabled",
            "started_at": now_iso(), "finished_at": now_iso(),
            "command": "", "details": "",
        }])
    command_template = cg.get("command")
    if not command_template:
        return pd.DataFrame([{
            "repo_id": "", "repo_path": "", "status": "skipped",
            "started_at": now_iso(), "finished_at": now_iso(),
            "command": "", "details": "code_graph.command is not configured",
        }])

    repos = kb.query_df("SELECT repo_id, repo_path FROM repo_inventory ORDER BY repo_id")
    max_repos = cg.get("max_repos")
    if max_repos:
        repos = repos.head(int(max_repos))

    rows: List[Dict[str, Any]] = []
    timeout_seconds = int(cg.get("timeout_seconds", 1800))
    max_workers = max(1, int(cg.get("max_workers") or 1))
    retries = max(0, int(cg.get("retries") or 0))
    try:
        from tqdm.auto import tqdm
    except Exception:
        tqdm = None
    pending = []
    for _, repo in repos.reset_index(drop=True).iterrows():
        repo_id = str(repo.get("repo_id") or "")
        repo_path = str(repo.get("repo_path") or "")
        command = os.path.expandvars(command_template.format(repo_id=shlex.quote(repo_id), repo_path=shlex.quote(repo_path)))
        fingerprint = _code_graph_fingerprint(repo_id, repo_path, command)
        existing = kb.query_df(
            """
            SELECT status, fingerprint
            FROM pipeline_repo_status
            WHERE stage = 'code_graph' AND repo_id = ?
            LIMIT 1
            """,
            [repo_id],
        )
        if not existing.empty and str(existing.iloc[0].get("status") or "") == "ok" and str(existing.iloc[0].get("fingerprint") or "") == fingerprint:
            rows.append({
                "repo_id": repo_id,
                "repo_path": repo_path,
                "status": "cached",
                "started_at": now_iso(),
                "finished_at": now_iso(),
                "command": command,
                "details": "",
            })
            continue
        pending.append((repo_id, repo_path, command, fingerprint))

    def run_one(item: tuple[str, str, str, str]) -> Dict[str, Any]:
        repo_id, repo_path, command, fingerprint = item
        started = now_iso()
        if not tqdm:
            console.print(f"code graph: {repo_id}", highlight=False)
        status = "error"
        details = ""
        try:
            attempts: Iterable[int] = range(retries + 1)
            if tqdm and max_workers == 1:
                attempts = tqdm(attempts, total=retries + 1, desc=f"Code graph {repo_id}", unit="try", position=1, leave=False)
            for attempt in attempts:
                p = subprocess.run(
                    shlex.split(command),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=timeout_seconds,
                )
                status = "ok" if p.returncode == 0 else "error"
                details = (p.stdout + "\n" + p.stderr).strip()[:4000]
                if status == "ok" or attempt >= retries:
                    break
        except subprocess.TimeoutExpired as e:
            status = "timeout"
            details = f"timed out after {timeout_seconds}s: {e}"
        except Exception as e:
            status = "error"
            details = str(e)
        return {
            "repo_id": repo_id,
            "repo_path": repo_path,
            "status": status,
            "started_at": started,
            "finished_at": now_iso(),
            "command": command,
            "details": details,
            "_fingerprint": fingerprint,
        }

    if max_workers > 1 and len(pending) > 1:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [pool.submit(run_one, item) for item in pending]
            iterator: Iterable[Any] = as_completed(futures)
            if tqdm:
                iterator = tqdm(iterator, total=len(futures), desc="Indexing code graph", unit="repo", position=0)
            for future in iterator:
                row = future.result()
                rows.append(row)
                _record_code_graph_run(kb, row, quiet=bool(tqdm))
    else:
        iterator = pending
        if tqdm:
            iterator = tqdm(iterator, total=len(pending), desc="Indexing code graph", unit="repo", position=0)
        for item in iterator:
            row = run_one(item)
            rows.append(row)
            _record_code_graph_run(kb, row, quiet=bool(tqdm))
    return pd.DataFrame([{k: v for k, v in row.items() if not k.startswith("_")} for row in rows])


def _record_code_graph_run(kb, row: Dict[str, Any], *, quiet: bool = False) -> None:
    fingerprint = str(row.get("_fingerprint") or "")
    repo_id = str(row.get("repo_id") or "")
    status = str(row.get("status") or "")
    public_row = {k: v for k, v in row.items() if not k.startswith("_")}
    try:
        kb.append_df("code_graph_runs", pd.DataFrame([public_row]))
        kb.execute("DELETE FROM pipeline_repo_status WHERE stage = 'code_graph' AND repo_id = ?", [repo_id])
        kb.append_df("pipeline_repo_status", pd.DataFrame([{
            "stage": "code_graph",
            "repo_id": repo_id,
            "fingerprint": fingerprint,
            "status": status,
            "started_at": row.get("started_at") or "",
            "finished_at": row.get("finished_at") or "",
            "item_count": 1 if status == "ok" else 0,
            "details": str(row.get("details") or "")[:4000],
        }]))
    except Exception:
        pass
    if not quiet:
        console.print(f"code graph: {repo_id} -> {status}", highlight=False)


def _code_graph_fingerprint(repo_id: str, repo_path: str, command: str) -> str:
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo_path, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, timeout=30).stdout.strip()
    except Exception:
        head = ""
    return hashlib.sha1(f"{repo_id}\n{repo_path}\n{head}\n{command}".encode("utf-8", errors="ignore")).hexdigest()


def import_code_graph(kb, cfg: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cg = cfg.get("code_graph", {})
    import_cfg = cg.get("import", {})
    if not cg.get("enabled", False) or not import_cfg.get("enabled", True):
        return _symbol_df([]), _edge_df([])

    cgc = _cgc_executable(cg)
    page_size = int(import_cfg.get("page_size") or 5000)
    timeout = int(import_cfg.get("query_timeout_seconds") or max(int(cg.get("timeout_seconds", 300)), 300))
    max_symbols = import_cfg.get("max_symbols")
    max_edges = import_cfg.get("max_edges")
    repo_index = _repo_index(kb.query_df("SELECT repo_id, repo_path FROM repo_inventory ORDER BY length(repo_path) DESC"))

    symbol_rows: dict[str, Dict[str, Any]] = {}
    for node_type in import_cfg.get("node_types") or ["File", "Function", "Class", "Module", "Struct", "Enum"]:
        query = _symbol_query(str(node_type))
        for row in _paged_cgc_query(cgc, query, page_size, timeout, max_rows=max_symbols):
            sym = _symbol_row(row, repo_index)
            if sym["symbol_id"]:
                symbol_rows.setdefault(sym["symbol_id"], sym)

    edge_rows: dict[str, Dict[str, Any]] = {}
    for edge_type, query in _edge_queries().items():
        for row in _paged_cgc_query(cgc, query, page_size, timeout, max_rows=max_edges):
            src = _symbol_ref(row, "from", repo_index)
            dst = _symbol_ref(row, "to", repo_index)
            if src["symbol_id"]:
                symbol_rows.setdefault(src["symbol_id"], src)
            if dst["symbol_id"]:
                symbol_rows.setdefault(dst["symbol_id"], dst)
            edge = _edge_row(row, edge_type, src, dst, repo_index)
            if edge["edge_id"]:
                edge_rows.setdefault(edge["edge_id"], edge)

    return _symbol_df(symbol_rows.values()), _edge_df(edge_rows.values())


def _cgc_executable(cg: Dict[str, Any]) -> str:
    explicit = cg.get("query_command") or cg.get("executable")
    if explicit:
        return os.path.expandvars(str(explicit))
    command = os.path.expandvars(str(cg.get("command") or "cgc"))
    parts = shlex.split(command)
    return parts[0] if parts else "cgc"


def _repo_index(repos: pd.DataFrame) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for _, repo in repos.iterrows():
        repo_id = str(repo.get("repo_id") or "")
        repo_path = str(repo.get("repo_path") or "")
        if repo_id and repo_path:
            out.append((repo_id, str(Path(repo_path).expanduser().resolve())))
    return out


def _match_repo(path: str, repo_index: List[Tuple[str, str]]) -> Tuple[str, str]:
    if not path:
        return "", ""
    try:
        full = str(Path(path).expanduser().resolve())
    except Exception:
        full = path
    for repo_id, repo_path in repo_index:
        if full == repo_path:
            return repo_id, ""
        prefix = repo_path.rstrip("/") + "/"
        if full.startswith(prefix):
            return repo_id, full[len(prefix):]
    return "", full


def _symbol_query(node_type: str) -> str:
    return f"""
    MATCH (n:{node_type})
    RETURN '{node_type}' AS node_type,
           n.name AS name,
           n.path AS file_path,
           n.line_number AS line_start,
           n.end_line AS line_end,
           n.language AS language,
           n.signature AS signature
    """.strip()


def _edge_queries() -> Dict[str, str]:
    return {
        "CALLS": """
        MATCH (a)-[r:CALLS]->(b)
        RETURN labels(a) AS from_labels, a.name AS from_name, a.path AS from_path,
               a.line_number AS from_line, a.end_line AS from_end_line,
               labels(b) AS to_labels, b.name AS to_name, b.path AS to_path,
               b.line_number AS to_line, b.end_line AS to_end_line,
               r.line_number AS line_start, r.full_call_name AS full_call_name
        """.strip(),
        "IMPORTS": """
        MATCH (a)-[r:IMPORTS]->(b)
        RETURN labels(a) AS from_labels, a.name AS from_name, a.path AS from_path,
               a.line_number AS from_line, a.end_line AS from_end_line,
               labels(b) AS to_labels, b.name AS to_name, b.path AS to_path,
               b.line_number AS to_line, b.end_line AS to_end_line,
               r.source AS import_source
        """.strip(),
        "INHERITS": """
        MATCH (a)-[r:INHERITS]->(b)
        RETURN labels(a) AS from_labels, a.name AS from_name, a.path AS from_path,
               a.line_number AS from_line, a.end_line AS from_end_line,
               labels(b) AS to_labels, b.name AS to_name, b.path AS to_path,
               b.line_number AS to_line, b.end_line AS to_end_line
        """.strip(),
    }


def _paged_cgc_query(cgc: str, base_query: str, page_size: int, timeout: int, max_rows: Any = None) -> Iterable[Dict[str, Any]]:
    offset = 0
    yielded = 0
    max_n = int(max_rows) if max_rows not in (None, "", 0) else None
    while True:
        limit = page_size
        if max_n is not None:
            limit = min(limit, max_n - yielded)
            if limit <= 0:
                break
        query = f"{base_query} SKIP {offset} LIMIT {limit}"
        rows = _run_cgc_query(cgc, query, timeout)
        if not rows:
            break
        for row in rows:
            if isinstance(row, dict):
                yield row
                yielded += 1
                if max_n is not None and yielded >= max_n:
                    return
        if len(rows) < limit:
            break
        offset += limit


def _run_cgc_query(cgc: str, query: str, timeout: int) -> List[Dict[str, Any]]:
    env = os.environ.copy()
    env.setdefault("COLUMNS", "20000")
    env.setdefault("RICH_WIDTH", "20000")
    p = subprocess.run(
        [cgc, "query", query],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout,
        env=env,
    )
    if p.returncode != 0:
        raise RuntimeError((p.stdout + "\n" + p.stderr).strip()[:4000])
    return _parse_cgc_json(p.stdout)


def _parse_cgc_json(output: str) -> List[Dict[str, Any]]:
    clean = re.sub(r"\x1b\[[0-9;]*m", "", output)
    decoder = json.JSONDecoder()
    for idx, ch in enumerate(clean):
        if ch not in "[{":
            continue
        try:
            obj, _ = decoder.raw_decode(clean[idx:])
            if isinstance(obj, list):
                return [x for x in obj if isinstance(x, dict)]
            if isinstance(obj, dict):
                return [obj]
        except json.JSONDecodeError:
            continue
    raise RuntimeError(f"Could not parse cgc JSON output: {clean[:1000]}")


def _first_label(labels: Any, default: str) -> str:
    if isinstance(labels, list) and labels:
        return str(labels[0] or default)
    if isinstance(labels, str) and labels:
        return labels
    return default


def _int_or_none(value: Any) -> int | None:
    try:
        if value in (None, ""):
            return None
        return int(value)
    except Exception:
        return None


def _symbol_id(node_type: str, name: str, file_path: str, line_start: Any) -> str:
    return "code_symbol:" + stable_id(node_type, file_path, line_start, name)


def _symbol_row(raw: Dict[str, Any], repo_index: List[Tuple[str, str]]) -> Dict[str, Any]:
    node_type = str(raw.get("node_type") or "Code")
    name = str(raw.get("name") or raw.get("file_path") or "")
    file_path = str(raw.get("file_path") or "")
    repo_id, rel_path = _match_repo(file_path, repo_index)
    line_start = _int_or_none(raw.get("line_start"))
    line_end = _int_or_none(raw.get("line_end"))
    symbol_id = _symbol_id(node_type, name, file_path, line_start)
    return {
        "symbol_id": symbol_id,
        "repo_id": repo_id,
        "node_type": node_type,
        "name": name,
        "file_path": file_path,
        "rel_path": rel_path,
        "line_start": line_start,
        "line_end": line_end,
        "language": str(raw.get("language") or ""),
        "signature": str(raw.get("signature") or ""),
        "source": "cgc",
        "properties": json_dumps({k: v for k, v in raw.items() if k not in {"node_type", "name", "file_path", "line_start", "line_end", "language", "signature"}}),
    }


def _symbol_ref(raw: Dict[str, Any], prefix: str, repo_index: List[Tuple[str, str]]) -> Dict[str, Any]:
    labels = raw.get(f"{prefix}_labels")
    node_type = _first_label(labels, "Code")
    name = str(raw.get(f"{prefix}_name") or raw.get(f"{prefix}_path") or "")
    file_path = str(raw.get(f"{prefix}_path") or "")
    line_start = _int_or_none(raw.get(f"{prefix}_line"))
    return _symbol_row({
        "node_type": node_type,
        "name": name,
        "file_path": file_path,
        "line_start": line_start,
        "line_end": _int_or_none(raw.get(f"{prefix}_end_line")),
    }, repo_index)


def _edge_row(raw: Dict[str, Any], edge_type: str, src: Dict[str, Any], dst: Dict[str, Any], repo_index: List[Tuple[str, str]]) -> Dict[str, Any]:
    file_path = str(raw.get("file_path") or src.get("file_path") or dst.get("file_path") or "")
    repo_id, rel_path = _match_repo(file_path, repo_index)
    if not repo_id:
        repo_id = str(src.get("repo_id") or dst.get("repo_id") or "")
    if not rel_path:
        rel_path = str(src.get("rel_path") or dst.get("rel_path") or "")
    line_start = _int_or_none(raw.get("line_start") or src.get("line_start"))
    edge_id = "code_edge:" + stable_id(edge_type, src.get("symbol_id"), dst.get("symbol_id"), file_path, line_start, raw.get("full_call_name") or raw.get("import_source"))
    return {
        "edge_id": edge_id,
        "repo_id": repo_id,
        "from_symbol_id": str(src.get("symbol_id") or ""),
        "to_symbol_id": str(dst.get("symbol_id") or ""),
        "edge_type": f"CODE_{edge_type}",
        "file_path": file_path,
        "rel_path": rel_path,
        "line_start": line_start,
        "source": "cgc",
        "confidence": 0.85,
        "properties": json_dumps({k: v for k, v in raw.items() if k not in {"from_labels", "to_labels"}}),
    }


def _symbol_df(rows: Iterable[Dict[str, Any]]) -> pd.DataFrame:
    cols = ["symbol_id", "repo_id", "node_type", "name", "file_path", "rel_path", "line_start", "line_end", "language", "signature", "source", "properties"]
    return pd.DataFrame(list(rows), columns=cols)


def _edge_df(rows: Iterable[Dict[str, Any]]) -> pd.DataFrame:
    cols = ["edge_id", "repo_id", "from_symbol_id", "to_symbol_id", "edge_type", "file_path", "rel_path", "line_start", "source", "confidence", "properties"]
    return pd.DataFrame(list(rows), columns=cols)
