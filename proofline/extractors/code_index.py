from __future__ import annotations

import re
import sqlite3
import hashlib
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List

import pandas as pd

from proofline.utils import safe_read_text, stable_id, json_dumps

LANG_BY_EXT = {
    ".py": "python", ".js": "javascript", ".jsx": "javascript", ".ts": "typescript", ".tsx": "typescript",
    ".java": "java", ".kt": "kotlin", ".go": "go", ".rb": "ruby", ".php": "php", ".cs": "csharp",
    ".rs": "rust", ".scala": "scala", ".swift": "swift", ".sql": "sql", ".yaml": "yaml", ".yml": "yaml",
    ".json": "json", ".toml": "toml", ".xml": "xml", ".gradle": "gradle", ".proto": "protobuf",
    ".graphql": "graphql", ".md": "markdown", ".tf": "terraform", ".sh": "shell", ".dart": "dart",
}

SYMBOL_PATTERNS = [
    re.compile(r"^\s*(?:export\s+)?(?:async\s+)?function\s+([A-Za-z_$][\w$]*)", re.M),
    re.compile(r"^\s*(?:export\s+)?class\s+([A-Za-z_$][\w$]*)", re.M),
    re.compile(r"^\s*def\s+([A-Za-z_]\w*)\s*\(", re.M),
    re.compile(r"^\s*class\s+([A-Za-z_]\w*)", re.M),
    re.compile(r"^\s*(?:public|private|protected)?\s*(?:static\s+)?[\w<>\[\], ?]+\s+([A-Za-z_]\w*)\s*\(", re.M),
    re.compile(r"^\s*func\s+(?:\([^)]+\)\s*)?([A-Za-z_]\w*)\s*\(", re.M),
]

AST_CHUNK_KINDS = [
    "ast_symbol",
    "ast_function",
    "ast_method",
    "ast_class",
    "ast_interface",
    "ast_enum",
    "ast_module",
    "ast_struct",
    "ast_large_symbol_window",
]

DEFAULT_AST_CHUNKING = {
    "enabled": True,
    "source": "cgc",
    "fallback_regex": True,
    "keep_file_windows": True,
    "include_node_types": ["Function", "Class", "Method", "Module", "Struct", "Enum", "Interface"],
    "max_symbol_lines": 240,
    "symbol_window_lines": 160,
    "symbol_window_overlap": 30,
    "include_context_prefix": True,
    "dedupe_overlapping_chunks": True,
}

AST_KIND_BY_NODE_TYPE = {
    "class": "ast_class",
    "enum": "ast_enum",
    "function": "ast_function",
    "interface": "ast_interface",
    "method": "ast_method",
    "module": "ast_module",
    "struct": "ast_struct",
}


def ast_chunking_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    configured = cfg.get("indexing", {}).get("ast_chunking", {}) or {}
    out = dict(DEFAULT_AST_CHUNKING)
    out.update(configured)
    return out


def chunk_text(
    repo_id: str,
    path: str,
    rel_path: str,
    text: str,
    max_lines: int = 120,
    overlap: int = 20,
    *,
    cfg: Dict[str, Any] | None = None,
    graph_symbols: List[Dict[str, Any]] | None = None,
) -> List[Dict[str, Any]]:
    ext = Path(rel_path).suffix.lower()
    lang = LANG_BY_EXT.get(ext, "text")
    lines = text.splitlines()
    chunks: List[Dict[str, Any]] = []
    cfg = cfg or {}
    ast_cfg = ast_chunking_config(cfg)

    if ext == ".md":
        chunks.extend(chunk_text_markdown(repo_id, path, rel_path, lang, lines))
        if bool(ast_cfg.get("keep_file_windows", True)):
            chunks.extend(chunk_text_windows(repo_id, path, rel_path, lang, lines, max_lines=max_lines, overlap=overlap))
        if bool(ast_cfg.get("dedupe_overlapping_chunks", True)):
            return dedupe_chunks(chunks)
        return chunks

    ast_chunks: List[Dict[str, Any]] = []
    if ast_cfg.get("enabled", True) and str(ast_cfg.get("source") or "cgc").lower() == "cgc":
        ast_chunks = chunk_text_from_graph_symbols(repo_id, path, rel_path, lang, lines, graph_symbols or [], ast_cfg)
        chunks.extend(ast_chunks)

    if bool(ast_cfg.get("fallback_regex", True)):
        regex_chunks = chunk_text_regex_symbols(repo_id, path, rel_path, lang, text, lines, max_lines=max_lines)
        if not ast_chunks:
            chunks.extend(regex_chunks)
        else:
            chunks.extend(uncovered_regex_chunks(regex_chunks, ast_chunks))

    if bool(ast_cfg.get("keep_file_windows", True)):
        chunks.extend(chunk_text_windows(repo_id, path, rel_path, lang, lines, max_lines=max_lines, overlap=overlap))

    if bool(ast_cfg.get("dedupe_overlapping_chunks", True)):
        return dedupe_chunks(chunks)
    return chunks


def chunk_text_markdown(repo_id: str, path: str, rel_path: str, lang: str, lines: List[str]) -> List[Dict[str, Any]]:
    chunks: List[Dict[str, Any]] = []
    starts = [i for i, line in enumerate(lines) if line.startswith("#")]
    if not starts:
        starts = [0]
    starts.append(len(lines))
    for a, b in zip(starts, starts[1:]):
        if b <= a:
            continue
        chunk = "\n".join(lines[a:b])
        if chunk.strip():
            symbol = lines[a].strip()[:160] if a < len(lines) else "section"
            chunks.append(make_chunk(
                repo_id,
                path,
                rel_path,
                lang,
                "doc_section",
                symbol,
                a + 1,
                b,
                chunk,
                metadata={"chunk_source": "markdown", "line_start": a + 1, "line_end": b, "repo_id": repo_id, "rel_path": rel_path},
            ))
    return chunks


def chunk_text_regex_symbols(
    repo_id: str,
    path: str,
    rel_path: str,
    lang: str,
    text: str,
    lines: List[str],
    *,
    max_lines: int,
) -> List[Dict[str, Any]]:
    chunks: List[Dict[str, Any]] = []
    symbol_positions: List[tuple[int, str]] = []
    for pat in SYMBOL_PATTERNS:
        for m in pat.finditer(text):
            line = text[:m.start()].count("\n")
            symbol_positions.append((line, m.group(1)))
    symbol_positions = sorted(set(symbol_positions))
    if symbol_positions:
        for idx, (line, sym) in enumerate(symbol_positions):
            end = symbol_positions[idx + 1][0] if idx + 1 < len(symbol_positions) else min(len(lines), line + max_lines)
            end = min(end, line + max_lines)
            chunk = "\n".join(lines[line:end])
            if chunk.strip():
                chunks.append(make_chunk(
                    repo_id,
                    path,
                    rel_path,
                    lang,
                    "symbol",
                    sym,
                    line + 1,
                    end,
                    chunk,
                    metadata={
                        "chunk_source": "regex",
                        "node_type": "Symbol",
                        "symbol": sym,
                        "language": lang,
                        "line_start": line + 1,
                        "line_end": end,
                        "repo_id": repo_id,
                        "rel_path": rel_path,
                    },
                ))
    return chunks


def chunk_text_windows(
    repo_id: str,
    path: str,
    rel_path: str,
    lang: str,
    lines: List[str],
    *,
    max_lines: int = 120,
    overlap: int = 20,
) -> List[Dict[str, Any]]:
    chunks: List[Dict[str, Any]] = []
    if not lines:
        return chunks
    step = max_lines - overlap
    if step <= 0:
        step = max_lines
    for start in range(0, len(lines), step):
        end = min(len(lines), start + max_lines)
        chunk = "\n".join(lines[start:end])
        if chunk.strip():
            chunks.append(make_chunk(
                repo_id,
                path,
                rel_path,
                lang,
                "file_window",
                "",
                start + 1,
                end,
                chunk,
                metadata={
                    "chunk_source": "window",
                    "language": lang,
                    "line_start": start + 1,
                    "line_end": end,
                    "repo_id": repo_id,
                    "rel_path": rel_path,
                },
            ))
        if end == len(lines):
            break
    return chunks


def chunk_text_from_graph_symbols(
    repo_id: str,
    path: str,
    rel_path: str,
    lang: str,
    lines: List[str],
    graph_symbols: List[Dict[str, Any]],
    ast_cfg: Dict[str, Any],
) -> List[Dict[str, Any]]:
    symbols = normalize_graph_symbols(graph_symbols, len(lines), ast_cfg)
    if not symbols:
        return []
    symbols = annotate_parent_symbols(symbols)
    chunks: List[Dict[str, Any]] = []
    for sym in symbols:
        chunks.extend(make_ast_chunks_for_symbol(repo_id, path, rel_path, lang, lines, sym, symbols, ast_cfg))
    return chunks


def normalize_graph_symbols(graph_symbols: List[Dict[str, Any]], line_count: int, ast_cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    include = {str(t).lower() for t in ast_cfg.get("include_node_types") or []}
    max_symbol_lines = max(1, int(ast_cfg.get("max_symbol_lines") or DEFAULT_AST_CHUNKING["max_symbol_lines"]))
    out: List[Dict[str, Any]] = []
    seen: set[tuple[str, str, int, int]] = set()
    for raw in graph_symbols:
        node_type = str(raw.get("node_type") or "").strip()
        if not node_type or (include and node_type.lower() not in include):
            continue
        start = int_or_none(raw.get("line_start"))
        if start is None or start < 1 or start > max(line_count, 1):
            continue
        end = int_or_none(raw.get("line_end"))
        if end is None or end < start:
            end = min(line_count, start + max_symbol_lines - 1)
        end = min(end, line_count)
        name = str(raw.get("name") or raw.get("symbol") or "").strip()
        if not name:
            name = f"{node_type}@{start}"
        key = (node_type.lower(), name, start, end)
        if key in seen:
            continue
        seen.add(key)
        out.append({
            **raw,
            "node_type": node_type,
            "name": name,
            "line_start": start,
            "line_end": end,
        })
    return sorted(out, key=lambda s: (int(s["line_start"]), -(int(s["line_end"]) - int(s["line_start"])), str(s.get("name") or "")))


def annotate_parent_symbols(symbols: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    for sym in symbols:
        start = int(sym["line_start"])
        end = int(sym["line_end"])
        candidates = []
        for other in symbols:
            if other is sym:
                continue
            o_start = int(other["line_start"])
            o_end = int(other["line_end"])
            if o_start <= start and end <= o_end and (o_start, o_end) != (start, end):
                candidates.append(other)
        if candidates:
            parent = min(candidates, key=lambda s: int(s["line_end"]) - int(s["line_start"]))
            sym["parent_symbol"] = str(parent.get("name") or "")
            sym["parent_symbol_id"] = str(parent.get("symbol_id") or "")
    return symbols


def make_ast_chunks_for_symbol(
    repo_id: str,
    path: str,
    rel_path: str,
    lang: str,
    lines: List[str],
    sym: Dict[str, Any],
    all_symbols: List[Dict[str, Any]],
    ast_cfg: Dict[str, Any],
) -> List[Dict[str, Any]]:
    node_type = str(sym.get("node_type") or "Symbol")
    symbol = str(sym.get("name") or "")
    start = int(sym["line_start"])
    end = int(sym["line_end"])
    line_span = end - start + 1
    kind = AST_KIND_BY_NODE_TYPE.get(node_type.lower(), "ast_symbol")
    max_symbol_lines = max(1, int(ast_cfg.get("max_symbol_lines") or DEFAULT_AST_CHUNKING["max_symbol_lines"]))
    chunks: List[Dict[str, Any]] = []

    if node_type.lower() in {"class", "interface", "struct"} and line_span > max_symbol_lines and symbol_has_children(sym, all_symbols):
        header_end = class_context_end(sym, all_symbols, max_symbol_lines)
        text = slice_lines(lines, start, header_end)
        if text.strip():
            chunks.append(make_ast_chunk(repo_id, path, rel_path, lang, kind, symbol, start, header_end, text, sym, ast_cfg))
        return chunks

    text = slice_lines(lines, start, end)
    if not text.strip():
        return chunks
    if line_span <= max_symbol_lines:
        chunks.append(make_ast_chunk(repo_id, path, rel_path, lang, kind, symbol, start, end, text, sym, ast_cfg))
        return chunks

    chunks.extend(split_large_symbol_chunk(repo_id, path, rel_path, lang, lines, sym, ast_cfg))
    return chunks


def symbol_has_children(sym: Dict[str, Any], symbols: List[Dict[str, Any]]) -> bool:
    start = int(sym["line_start"])
    end = int(sym["line_end"])
    for other in symbols:
        if other is sym:
            continue
        o_start = int(other["line_start"])
        o_end = int(other["line_end"])
        if start < o_start and o_end <= end:
            return True
    return False


def class_context_end(sym: Dict[str, Any], symbols: List[Dict[str, Any]], max_symbol_lines: int) -> int:
    start = int(sym["line_start"])
    end = int(sym["line_end"])
    child_starts = [
        int(other["line_start"])
        for other in symbols
        if other is not sym and start < int(other["line_start"]) <= int(other["line_end"]) <= end
    ]
    if child_starts:
        return max(start, min(end, min(child_starts) - 1))
    return min(end, start + max_symbol_lines - 1)


def split_large_symbol_chunk(
    repo_id: str,
    path: str,
    rel_path: str,
    lang: str,
    lines: List[str],
    sym: Dict[str, Any],
    ast_cfg: Dict[str, Any],
) -> List[Dict[str, Any]]:
    start = int(sym["line_start"])
    end = int(sym["line_end"])
    window = max(1, int(ast_cfg.get("symbol_window_lines") or DEFAULT_AST_CHUNKING["symbol_window_lines"]))
    overlap = max(0, int(ast_cfg.get("symbol_window_overlap") or DEFAULT_AST_CHUNKING["symbol_window_overlap"]))
    step = max(1, window - overlap)
    chunks: List[Dict[str, Any]] = []
    current = start
    while current <= end:
        chunk_end = min(end, current + window - 1)
        text = slice_lines(lines, current, chunk_end)
        if text.strip():
            chunks.append(make_ast_chunk(
                repo_id,
                path,
                rel_path,
                lang,
                "ast_large_symbol_window",
                str(sym.get("name") or ""),
                current,
                chunk_end,
                text,
                sym,
                ast_cfg,
            ))
        if chunk_end >= end:
            break
        current += step
    return chunks


def make_ast_chunk(
    repo_id: str,
    path: str,
    rel_path: str,
    lang: str,
    kind: str,
    symbol: str,
    start: int,
    end: int,
    text: str,
    sym: Dict[str, Any],
    ast_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    node_type = str(sym.get("node_type") or "")
    signature = str(sym.get("signature") or "")
    parent_symbol = str(sym.get("parent_symbol") or "")
    metadata = {
        "chunk_source": str(sym.get("source") or "cgc"),
        "node_type": node_type,
        "symbol": symbol,
        "parent_symbol": parent_symbol,
        "signature": signature,
        "language": lang or str(sym.get("language") or ""),
        "line_start": start,
        "line_end": end,
        "repo_id": repo_id,
        "rel_path": rel_path,
        "graph_symbol_id": str(sym.get("symbol_id") or ""),
    }
    chunk_text_value = text
    if bool(ast_cfg.get("include_context_prefix", True)):
        chunk_text_value = context_prefix(rel_path, metadata["language"], symbol, parent_symbol, signature) + text
    return make_chunk(repo_id, path, rel_path, metadata["language"], kind, symbol, start, end, chunk_text_value, metadata=metadata)


def context_prefix(rel_path: str, language: str, symbol: str, parent_symbol: str, signature: str) -> str:
    display_symbol = f"{parent_symbol}.{symbol}" if parent_symbol and symbol and not symbol.startswith(parent_symbol) else symbol
    parts = [
        f"File: {rel_path}",
        f"Language: {language}",
        f"Symbol: {display_symbol}",
    ]
    if signature:
        parts.append(f"Signature: {signature}")
    return "\n".join(parts) + "\n\n"


def slice_lines(lines: List[str], start: int, end: int) -> str:
    if not lines:
        return ""
    start = max(1, start)
    end = min(len(lines), end)
    if end < start:
        return ""
    return "\n".join(lines[start - 1:end])


def int_or_none(value: Any) -> int | None:
    try:
        if value in (None, ""):
            return None
        return int(value)
    except Exception:
        return None


def dedupe_chunks(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    seen: set[tuple[str, str, int, int, str]] = set()
    for chunk in chunks:
        key = (
            str(chunk.get("kind") or ""),
            str(chunk.get("symbol") or ""),
            int(chunk.get("start_line") or 0),
            int(chunk.get("end_line") or 0),
            str(chunk.get("text") or ""),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(chunk)
    return out


def uncovered_regex_chunks(regex_chunks: List[Dict[str, Any]], ast_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    ast_starts = {int(c.get("start_line") or 0) for c in ast_chunks}
    ast_symbol_ranges = [
        (int(c.get("start_line") or 0), int(c.get("end_line") or 0))
        for c in ast_chunks
        if str(c.get("kind") or "") in {"ast_symbol", "ast_function", "ast_method", "ast_large_symbol_window"}
    ]
    out: List[Dict[str, Any]] = []
    for chunk in regex_chunks:
        start = int(chunk.get("start_line") or 0)
        if start in ast_starts:
            continue
        if any(a <= start <= b for a, b in ast_symbol_ranges):
            continue
        out.append(chunk)
    return out


def make_chunk(
    repo_id: str,
    path: str,
    rel_path: str,
    lang: str,
    kind: str,
    symbol: str,
    start: int,
    end: int,
    text: str,
    *,
    metadata: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    metadata = dict(metadata or {})
    metadata.setdefault("text_len", len(text))
    return {
        "chunk_id": stable_id(repo_id, rel_path, kind, symbol, start, end),
        "repo_id": repo_id,
        "file_path": path,
        "rel_path": rel_path,
        "language": lang,
        "kind": kind,
        "symbol": symbol,
        "start_line": start,
        "end_line": end,
        "text": text[:20000],
        "metadata": json_dumps(metadata),
    }


def build_chunks(repo_files: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(iter_file_chunks(repo_files, cfg))


def repo_files_fingerprint(
    repo_files: pd.DataFrame,
    cfg: Dict[str, Any] | None = None,
    graph_symbols: pd.DataFrame | None = None,
) -> str:
    if repo_files is None or repo_files.empty:
        return hashlib.sha1(b"").hexdigest()
    h = hashlib.sha1()
    h.update(json_dumps(fingerprint_chunking_config(cfg)).encode("utf-8", errors="ignore"))
    if graph_symbols is not None and not graph_symbols.empty:
        for row in graph_symbols.sort_values(["rel_path", "line_start", "line_end", "node_type", "name"], kind="stable").itertuples(index=False):
            h.update(
                f"{getattr(row, 'rel_path', '')}\0{getattr(row, 'node_type', '')}\0{getattr(row, 'name', '')}\0"
                f"{getattr(row, 'line_start', '')}\0{getattr(row, 'line_end', '')}\0{getattr(row, 'signature', '')}\n".encode("utf-8", errors="ignore")
            )
    for row in repo_files.sort_values(["repo_id", "rel_path"], kind="stable").itertuples(index=False):
        rel = str(getattr(row, "rel_path", "") or "")
        sha1 = str(getattr(row, "sha1", "") or "")
        size = str(getattr(row, "size_bytes", "") or "")
        h.update(f"{rel}\0{sha1}\0{size}\n".encode("utf-8", errors="ignore"))
    return h.hexdigest()


def fingerprint_chunking_config(cfg: Dict[str, Any] | None) -> Dict[str, Any]:
    if not cfg:
        return {}
    indexing = cfg.get("indexing", {}) or {}
    ast_cfg = ast_chunking_config(cfg)
    return {
        "ast_chunking": ast_cfg,
        "lexical_fts": bool(indexing.get("lexical_fts", True)),
    }


def iter_file_chunks(
    repo_files: pd.DataFrame,
    cfg: Dict[str, Any],
    *,
    show_progress: bool = False,
    desc: str | None = None,
) -> Iterator[Dict[str, Any]]:
    max_bytes = int(float(cfg["repos"].get("max_file_mb", 2)) * 1024 * 1024)
    allowed = set(cfg["repos"].get("include_extensions", []))
    iterator = repo_files.iterrows()
    if show_progress:
        try:
            from tqdm.auto import tqdm

            iterator = tqdm(iterator, total=len(repo_files), desc=desc or "REPO chunk files", unit="file", position=1, leave=False)
        except Exception:
            pass
    for _, f in iterator:
        rel = str(f["rel_path"])
        name = Path(rel).name
        ext = Path(rel).suffix
        if allowed and ext not in allowed and name not in allowed:
            continue
        text = safe_read_text(Path(f["path"]), max_bytes=max_bytes)
        if text is None:
            continue
        yield from chunk_text(str(f["repo_id"]), str(f["path"]), rel, text, cfg=cfg)


def file_fingerprint(
    row: pd.Series | Dict[str, Any],
    cfg: Dict[str, Any] | None = None,
    graph_symbols: List[Dict[str, Any]] | None = None,
) -> str:
    get = row.get if isinstance(row, dict) else row.get
    rel = str(get("rel_path") or "")
    sha1 = str(get("sha1") or "")
    size = str(get("size_bytes") or "")
    h = hashlib.sha1()
    h.update(f"{rel}\0{sha1}\0{size}\n".encode("utf-8", errors="ignore"))
    h.update(json_dumps(fingerprint_chunking_config(cfg)).encode("utf-8", errors="ignore"))
    for sym in graph_symbols or []:
        h.update(
            f"{sym.get('node_type', '')}\0{sym.get('name', '')}\0{sym.get('line_start', '')}\0"
            f"{sym.get('line_end', '')}\0{sym.get('signature', '')}\n".encode("utf-8", errors="ignore")
        )
    return h.hexdigest()


def chunks_for_file(
    file_row: Dict[str, Any],
    cfg: Dict[str, Any],
    graph_symbols: List[Dict[str, Any]] | None = None,
) -> List[Dict[str, Any]]:
    max_bytes = int(float(cfg["repos"].get("max_file_mb", 2)) * 1024 * 1024)
    allowed = set(cfg["repos"].get("include_extensions", []))
    rel = str(file_row["rel_path"])
    name = Path(rel).name
    ext = Path(rel).suffix
    if allowed and ext not in allowed and name not in allowed:
        return []
    text = safe_read_text(Path(str(file_row["path"])), max_bytes=max_bytes)
    if text is None:
        return []
    return chunk_text(str(file_row["repo_id"]), str(file_row["path"]), rel, text, cfg=cfg, graph_symbols=graph_symbols)


def chunked_rows(rows: Iterable[Dict[str, Any]], size: int) -> Iterator[List[Dict[str, Any]]]:
    batch: List[Dict[str, Any]] = []
    for row in rows:
        batch.append(row)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def build_sqlite_fts(chunks: pd.DataFrame, sqlite_path: str | Path) -> None:
    reset_sqlite_fts(sqlite_path)
    insert_sqlite_fts(chunks, sqlite_path)


def reset_sqlite_fts(sqlite_path: str | Path) -> None:
    sqlite_path = Path(sqlite_path)
    sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(sqlite_path))
    cur = con.cursor()
    cur.execute("DROP TABLE IF EXISTS chunks")
    cur.execute("CREATE VIRTUAL TABLE chunks USING fts5(chunk_id, repo_id, rel_path, language, kind, symbol, text)")
    con.commit()
    con.close()


def ensure_sqlite_fts(sqlite_path: str | Path) -> None:
    sqlite_path = Path(sqlite_path)
    sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(sqlite_path))
    cur = con.cursor()
    cur.execute("CREATE VIRTUAL TABLE IF NOT EXISTS chunks USING fts5(chunk_id, repo_id, rel_path, language, kind, symbol, text)")
    con.commit()
    con.close()


def delete_sqlite_fts_repo(sqlite_path: str | Path, repo_id: str) -> None:
    ensure_sqlite_fts(sqlite_path)
    con = sqlite3.connect(str(sqlite_path))
    cur = con.cursor()
    cur.execute("DELETE FROM chunks WHERE repo_id = ?", (repo_id,))
    con.commit()
    con.close()


def delete_sqlite_fts_file(sqlite_path: str | Path, repo_id: str, rel_path: str) -> None:
    ensure_sqlite_fts(sqlite_path)
    con = sqlite3.connect(str(sqlite_path))
    cur = con.cursor()
    cur.execute("DELETE FROM chunks WHERE repo_id = ? AND rel_path = ?", (repo_id, rel_path))
    con.commit()
    con.close()


def insert_sqlite_fts(chunks: pd.DataFrame, sqlite_path: str | Path) -> None:
    ensure_sqlite_fts(sqlite_path)
    sqlite_path = Path(sqlite_path)
    con = sqlite3.connect(str(sqlite_path))
    cur = con.cursor()
    if not chunks.empty:
        cur.executemany(
            "INSERT INTO chunks(chunk_id, repo_id, rel_path, language, kind, symbol, text) VALUES (?, ?, ?, ?, ?, ?, ?)",
            chunks[["chunk_id", "repo_id", "rel_path", "language", "kind", "symbol", "text"]].fillna("").itertuples(index=False, name=None),
        )
    con.commit()
    con.close()
