from __future__ import annotations

import hashlib
import json
import re
import subprocess
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional

import orjson


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_id(*parts: Any) -> str:
    raw = "|".join("" if p is None else str(p) for p in parts)
    return hashlib.sha1(raw.encode("utf-8", errors="ignore")).hexdigest()[:24]


def normalize_name(s: str | None) -> str:
    if not s:
        return ""
    s = s.lower().strip()
    s = re.sub(r"https?://", "", s)
    s = re.sub(r"\.git$", "", s)
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = re.sub(r"-(api|service|svc|app|prod|staging|dev)$", "", s)
    return s.strip("-")


def run_cmd(cmd: List[str], cwd: str | Path | None = None, timeout: int = 30) -> str:
    try:
        p = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, timeout=timeout)
        return p.stdout.strip()
    except Exception:
        return ""


def json_dumps(obj: Any) -> str:
    return orjson.dumps(obj, option=orjson.OPT_SORT_KEYS).decode("utf-8")


def json_loads(s: str | bytes | None) -> Any:
    if not s:
        return None
    if isinstance(s, str):
        s = s.encode("utf-8")
    return orjson.loads(s)


def chunked(iterable: Iterable[Any], size: int) -> Iterator[List[Any]]:
    buf: List[Any] = []
    for item in iterable:
        buf.append(item)
        if len(buf) >= size:
            yield buf
            buf = []
    if buf:
        yield buf


def dd_time_window(days: int) -> tuple[str, str]:
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)
    return start.isoformat().replace("+00:00", "Z"), end.isoformat().replace("+00:00", "Z")


def epoch_window(days: int) -> tuple[int, int]:
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)
    return int(start.timestamp()), int(end.timestamp())


def safe_read_text(path: Path, max_bytes: int = 2_000_000) -> Optional[str]:
    try:
        if path.stat().st_size > max_bytes:
            return None
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None


def flatten_json(obj: Any, prefix: str = "") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            out.update(flatten_json(v, key))
    elif isinstance(obj, list):
        # Lists are stored as-is at the current prefix to avoid huge explosions.
        out[prefix] = obj
    else:
        out[prefix] = obj
    return out


def pick_first(flat: Dict[str, Any], aliases: Iterable[str]) -> Any:
    for a in aliases:
        if a in flat and flat[a] not in (None, ""):
            return flat[a]
        # Datadog often nests attributes under attributes.* or attributes.attributes.*
        for p in (f"attributes.{a}", f"attributes.attributes.{a}", f"meta.{a}"):
            if p in flat and flat[p] not in (None, ""):
                return flat[p]
    return None
