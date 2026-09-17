"""Single-repo filesystem watch: event intake, debounce, incremental reindex.

Scope: exactly one repository - the directory where ``pfl watch`` runs.
Even if the KB indexes hundreds of projects, the loop observes and
reindexes only that one repo. Backend: ``watchdog`` OS events only.
"""
from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

DEFAULT_DEBOUNCE_SECONDS = 2.0
DEFAULT_SLOW_LANE_MINUTES = 15.0
DEFAULT_FULL_REBUILD_MINUTES = 0.0

DEFAULT_IGNORE_NAMES = {
    ".DS_Store", ".git", ".hg", ".svn", "__pycache__", ".idea", ".vscode",
    # Proofline's own state dir (workspace ./.proofline) - never reindex it.
    ".proofline",
}

DEFAULT_IGNORE_SUFFIXES = ("~", ".swp", ".swo", ".swn", ".tmp", ".temp", ".bak", ".orig", ".rej")

FAST_LANE_STAGES = ("repo_ingest", "code_index", "embeddings", "api_surface", "static_edges")
SLOW_LANE_STAGES = ("git_history", "git_blame", "entity_resolution", "graph", "endpoint_map", "capabilities")


@dataclass
class WatchEvent:
    kind: str  # created | modified | deleted | moved
    rel_path: str = ""
    src_rel_path: str = ""
    dest_rel_path: str = ""


@dataclass
class WatchConfig:
    repo_root: Path
    debounce_seconds: float = DEFAULT_DEBOUNCE_SECONDS
    slow_lane_minutes: float = DEFAULT_SLOW_LANE_MINUTES
    full_rebuild_minutes: float = DEFAULT_FULL_REBUILD_MINUTES
    fast_lane: List[str] = field(default_factory=lambda: list(FAST_LANE_STAGES))
    slow_lane: List[str] = field(default_factory=lambda: list(SLOW_LANE_STAGES))
    exclude_dirs: Set[str] = field(default_factory=set)
    include_extensions: Set[str] = field(default_factory=set)
    max_file_mb: float = 5.0
def watch_defaults() -> Dict[str, Any]:
    return {
        "debounce_seconds": DEFAULT_DEBOUNCE_SECONDS,
        "slow_lane_minutes": DEFAULT_SLOW_LANE_MINUTES,
        "full_rebuild_minutes": DEFAULT_FULL_REBUILD_MINUTES,
        "fast_lane": list(FAST_LANE_STAGES),
        "slow_lane": list(SLOW_LANE_STAGES),
    }


def watch_config_from_dict(cfg: Dict[str, Any], repo_root: Path) -> WatchConfig:
    section = dict(cfg.get("watch", {}) or {})
    repos_cfg = cfg.get("repos", {}) or {}
    return WatchConfig(
        repo_root=repo_root,
        debounce_seconds=float(section.get("debounce_seconds", DEFAULT_DEBOUNCE_SECONDS)),
        slow_lane_minutes=float(section.get("slow_lane_minutes", DEFAULT_SLOW_LANE_MINUTES)),
        full_rebuild_minutes=float(section.get("full_rebuild_minutes", DEFAULT_FULL_REBUILD_MINUTES)),
        fast_lane=list(section.get("fast_lane", list(FAST_LANE_STAGES))),
        slow_lane=list(section.get("slow_lane", list(SLOW_LANE_STAGES))),
        exclude_dirs=set(repos_cfg.get("exclude_dirs", []) or []),
        include_extensions=set(repos_cfg.get("include_extensions", []) or []),
        max_file_mb=float(repos_cfg.get("max_file_mb", 5.0)),
    )


def should_ignore_path(rel: str, exclude_dirs: Set[str]) -> bool:
    if not rel:
        return True
    parts = Path(rel).parts
    if not parts:
        return True
    name = parts[-1]
    if name in DEFAULT_IGNORE_NAMES or parts[0] in DEFAULT_IGNORE_NAMES:
        return True
    for suffix in DEFAULT_IGNORE_SUFFIXES:
        if name.endswith(suffix):
            return True
    if set(parts) & set(exclude_dirs or ()):
        return True
    return False
class DebouncedQueue:
    """Intake: watchdog thread puts raw paths, worker drains debounced batches."""

    def __init__(self, debounce_seconds: float = DEFAULT_DEBOUNCE_SECONDS) -> None:
        self.debounce_seconds = max(0.1, float(debounce_seconds))
        self._queue: "queue.Queue[WatchEvent]" = queue.Queue()
        self._last_put = 0.0
        self._lock = threading.Lock()

    def put(self, event: WatchEvent) -> None:
        with self._lock:
            self._last_put = time.monotonic()
        self._queue.put(event)

    def drain(self, timeout: float = 0.5) -> List[WatchEvent]:
        try:
            first = self._queue.get(timeout=timeout)
        except queue.Empty:
            return []
        batch = [first]
        while True:
            with self._lock:
                quiet_for = time.monotonic() - self._last_put
            wait = self.debounce_seconds - quiet_for
            if wait <= 0:
                break
            try:
                batch.append(self._queue.get(timeout=wait))
            except queue.Empty:
                break
        while True:
            try:
                batch.append(self._queue.get_nowait())
            except queue.Empty:
                break
        return batch

    def drain_nowait(self) -> List[WatchEvent]:
        out: List[WatchEvent] = []
        while True:
            try:
                out.append(self._queue.get_nowait())
            except queue.Empty:
                return out

    def empty(self) -> bool:
        return self._queue.empty()


def coalesce_events(events: List[WatchEvent]) -> Dict[str, Any]:
    modified: Set[str] = set()
    deleted: Set[str] = set()
    for event in events:
        kind = (event.kind or "").lower()
        if kind == "moved":
            if event.src_rel_path:
                deleted.add(event.src_rel_path)
                modified.discard(event.src_rel_path)
            if event.dest_rel_path:
                modified.add(event.dest_rel_path)
                deleted.discard(event.dest_rel_path)
            continue
        rel = event.rel_path
        if not rel:
            continue
        if kind == "deleted":
            deleted.add(rel)
            modified.discard(rel)
        elif kind in ("created", "modified"):
            deleted.discard(rel)
            modified.add(rel)
    return {"modified": sorted(modified), "deleted": sorted(deleted), "total": len(events)}


def filter_indexable(
    rel_paths: List[str],
    *,
    exclude_dirs: Set[str],
    include_extensions: Optional[Set[str]] = None,
    repo_root: Optional[Path] = None,
    max_file_mb: float = 5.0,
) -> List[str]:
    out: List[str] = []
    for rel in rel_paths:
        if should_ignore_path(rel, exclude_dirs):
            continue
        name = Path(rel).name
        ext = Path(rel).suffix
        if include_extensions and ext not in include_extensions and name not in include_extensions:
            continue
        if repo_root is not None:
            try:
                size = (repo_root / rel).stat().st_size
            except OSError:
                continue
            if size > max_file_mb * 1024 * 1024:
                continue
        out.append(rel)
    return out


def describe_batch(coalesced: Dict[str, Any]) -> str:
    parts = []
    if coalesced.get("modified"):
        parts.append(f"{len(coalesced['modified'])} modified")
    if coalesced.get("deleted"):
        parts.append(f"{len(coalesced['deleted'])} deleted")
    return ", ".join(parts) or "no changes"


EventCallback = Callable[[str, Dict[str, Any]], None]


def emit(callback: Optional[EventCallback], event: str, payload: Optional[Dict[str, Any]] = None) -> None:
    if callback is None:
        return
    try:
        callback(event, dict(payload or {}))
    except Exception:
        pass
