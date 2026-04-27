from __future__ import annotations

import re
import sqlite3
from pathlib import Path
from typing import Any, Dict, Iterable, List

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


def chunk_text(repo_id: str, path: str, rel_path: str, text: str, max_lines: int = 120, overlap: int = 20) -> List[Dict[str, Any]]:
    ext = Path(rel_path).suffix.lower()
    lang = LANG_BY_EXT.get(ext, "text")
    lines = text.splitlines()
    chunks: List[Dict[str, Any]] = []

    # Markdown: section chunks.
    if ext == ".md":
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
                chunks.append(make_chunk(repo_id, path, rel_path, lang, "doc_section", symbol, a + 1, b, chunk))
        return chunks

    # Try symbol/function chunks.
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
                chunks.append(make_chunk(repo_id, path, rel_path, lang, "symbol", sym, line + 1, end, chunk))

    # Also create rolling file chunks for search coverage.
    step = max_lines - overlap
    for start in range(0, len(lines), step):
        end = min(len(lines), start + max_lines)
        chunk = "\n".join(lines[start:end])
        if chunk.strip():
            chunks.append(make_chunk(repo_id, path, rel_path, lang, "file_window", "", start + 1, end, chunk))
        if end == len(lines):
            break
    return chunks


def make_chunk(repo_id: str, path: str, rel_path: str, lang: str, kind: str, symbol: str, start: int, end: int, text: str) -> Dict[str, Any]:
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
        "metadata": json_dumps({"text_len": len(text)}),
    }


def build_chunks(repo_files: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    max_bytes = int(float(cfg["repos"].get("max_file_mb", 2)) * 1024 * 1024)
    allowed = set(cfg["repos"].get("include_extensions", []))
    for _, f in repo_files.iterrows():
        rel = str(f["rel_path"])
        name = Path(rel).name
        ext = Path(rel).suffix
        if allowed and ext not in allowed and name not in allowed:
            continue
        text = safe_read_text(Path(f["path"]), max_bytes=max_bytes)
        if text is None:
            continue
        rows.extend(chunk_text(str(f["repo_id"]), str(f["path"]), rel, text))
    return pd.DataFrame(rows)


def build_sqlite_fts(chunks: pd.DataFrame, sqlite_path: str | Path) -> None:
    sqlite_path = Path(sqlite_path)
    sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(sqlite_path))
    cur = con.cursor()
    cur.execute("DROP TABLE IF EXISTS chunks")
    cur.execute("CREATE VIRTUAL TABLE chunks USING fts5(chunk_id, repo_id, rel_path, language, kind, symbol, text)")
    if not chunks.empty:
        cur.executemany(
            "INSERT INTO chunks(chunk_id, repo_id, rel_path, language, kind, symbol, text) VALUES (?, ?, ?, ?, ?, ?, ?)",
            chunks[["chunk_id", "repo_id", "rel_path", "language", "kind", "symbol", "text"]].fillna("").itertuples(index=False, name=None),
        )
    con.commit()
    con.close()
