"""Resolve + validate the single watched repo.

``pfl watch`` must run inside exactly one git repository. That repo is
matched against ``repos.root`` inventory by normalized remote URL first,
then by absolute path. Resolution never scans or touches other repos.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from proofline.extractors.repo import repo_id_from_path
from proofline.utils import normalize_name, run_cmd


def find_enclosing_repo(start: Path) -> Optional[Path]:
    """Walk up from ``start`` to the nearest dir containing ``.git``."""
    current = start.resolve()
    while True:
        if (current / ".git").exists():
            return current
        parent = current.parent
        if parent == current:
            return None
        current = parent


def repo_remote_url(repo: Path) -> str:
    return run_cmd(["git", "config", "--get", "remote.origin.url"], cwd=repo)


def repo_head(repo: Path) -> str:
    return run_cmd(["git", "rev-parse", "HEAD"], cwd=repo)


def resolve_watched_repo(
    kb: Any,
    cwd: Path,
    repos_root: Optional[str] = None,
) -> Dict[str, Any]:
    """Return ``{repo_id, repo_path, matched}`` for the CWD repo.

    Raises RuntimeError when CWD is not inside a git repo.
    """
    enclosing = find_enclosing_repo(Path(cwd))
    if enclosing is None:
        raise RuntimeError(
            f"pfl watch must run inside a git repository (cwd: {cwd}). "
            "cd into the repo you want to track and retry."
        )
    repo_id = repo_id_from_path(enclosing)
    try:
        inv = kb.query_df("SELECT repo_id, repo_path, repo_url FROM repo_inventory")
    except Exception:
        inv = None
    matched_id = repo_id
    matched_via = "dirname"
    if inv is not None and not inv.empty:
        rows = inv.to_dict("records")
        # 1) match by normalized remote URL (handles different checkout paths).
        local_url = repo_remote_url(enclosing)
        if local_url:
            want = normalize_name(local_url)
            for row in rows:
                for candidate in (row.get("repo_url") or "", row.get("repo_id") or ""):
                    if candidate and normalize_name(str(candidate)) == want:
                        matched_id = str(row.get("repo_id"))
                        matched_via = "remote-url"
                        break
                if matched_via == "remote-url":
                    break
        # 2) match by resolved absolute path.
        if matched_via == "dirname":
            want_path = str(enclosing.resolve())
            for row in rows:
                try:
                    have = str(Path(str(row.get("repo_path") or "")).resolve())
                except Exception:
                    continue
                if have == want_path:
                    matched_id = str(row.get("repo_id"))
                    matched_via = "path"
                    break
    return {
        "repo_id": matched_id,
        "repo_path": str(enclosing),
        "matched": matched_via != "dirname" or _inventory_has(kb, matched_id),
        "matched_via": matched_via,
        "remote_url": repo_remote_url(enclosing),
        "head": repo_head(enclosing),
    }


def _inventory_has(kb: Any, repo_id: str) -> bool:
    try:
        rows = kb.query_df("SELECT repo_id FROM repo_inventory WHERE repo_id = ?", [repo_id])
        return not rows.empty
    except Exception:
        return False
