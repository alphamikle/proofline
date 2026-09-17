"""watchdog event handler: single repo root -> DebouncedQueue of rel paths."""
from __future__ import annotations

from pathlib import Path
from typing import Set

from proofline.watch import DebouncedQueue, WatchEvent, should_ignore_path


class RepoEventHandler:
    """watchdog handler: repo root -> DebouncedQueue of rel paths."""

    def __init__(self, repo_root: Path, outbox: DebouncedQueue, exclude_dirs: Set[str]) -> None:
        self.repo_root = repo_root.resolve()
        self.outbox = outbox
        self.exclude_dirs = set(exclude_dirs or ())

    def _rel(self, path: str) -> str:
        try:
            return str(Path(path).resolve().relative_to(self.repo_root))
        except ValueError:
            return ""

    def _push(self, kind: str, rel: str) -> None:
        if not rel or should_ignore_path(rel, self.exclude_dirs):
            return
        self.outbox.put(WatchEvent(kind=kind, rel_path=rel))

    def dispatch(self, event) -> None:  # type: ignore[no-untyped-def]
        """Route watchdog events (FileSystemEventHandler-compatible entry)."""
        kind = getattr(event, "event_type", "")
        if kind == "moved":
            self.on_moved(event)
        elif kind == "created":
            self.on_created(event)
        elif kind == "deleted":
            self.on_deleted(event)
        elif kind == "modified":
            self.on_modified(event)

    # watchdog FileSystemEventHandler API
    def on_created(self, event) -> None:  # type: ignore[no-untyped-def]
        if getattr(event, "is_directory", False):
            return
        self._push("created", self._rel(event.src_path))

    def on_modified(self, event) -> None:  # type: ignore[no-untyped-def]
        if getattr(event, "is_directory", False):
            return
        self._push("modified", self._rel(event.src_path))

    def on_deleted(self, event) -> None:  # type: ignore[no-untyped-def]
        if getattr(event, "is_directory", False):
            return
        # Deleted file: push even if stat would fail (no size check downstream).
        rel = self._rel(event.src_path)
        if not rel or should_ignore_path(rel, self.exclude_dirs):
            return
        self.outbox.put(WatchEvent(kind="deleted", rel_path=rel))

    def on_moved(self, event) -> None:  # type: ignore[no-untyped-def]
        if getattr(event, "is_directory", False):
            return
        src = self._rel(getattr(event, "src_path", ""))
        dest = self._rel(getattr(event, "dest_path", ""))
        if src and not should_ignore_path(src, self.exclude_dirs):
            self.outbox.put(WatchEvent(kind="moved", src_rel_path=src, dest_rel_path=dest))
        elif dest and not should_ignore_path(dest, self.exclude_dirs):
            self.outbox.put(WatchEvent(kind="moved", src_rel_path=src, dest_rel_path=dest))


def start_observer(handler: RepoEventHandler, repo_root: Path):  # type: ignore[no-untyped-def]
    """Create + start a watchdog Observer. Raises RuntimeError if missing."""
    try:
        from watchdog.observers import Observer
    except ImportError as e:
        raise RuntimeError(
            "pfl watch needs the 'watchdog' package. Run: "
            ".venv/bin/python -m pip install -r requirements.txt"
        ) from e
    observer = Observer()
    observer.schedule(handler, str(repo_root), recursive=True)
    observer.start()
    return observer
