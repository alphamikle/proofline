from __future__ import annotations

import subprocess
import sys
from importlib import metadata
from pathlib import Path
from typing import Any, Dict

from proofline import __version__


def version_info() -> Dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    installed_version = _metadata_version()
    return {
        "version": installed_version or __version__,
        "package_version": __version__,
        "metadata_version": installed_version,
        "package_root": str(root),
        "python": sys.executable,
        "git": _git_info(root),
    }


def _metadata_version() -> str | None:
    try:
        return metadata.version("proofline")
    except metadata.PackageNotFoundError:
        return None


def _git_info(root: Path) -> Dict[str, Any]:
    if not (root / ".git").exists():
        return {"available": False}
    return {
        "available": True,
        "commit": _git(root, "rev-parse", "--short", "HEAD"),
        "branch": _git(root, "rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(_git(root, "status", "--porcelain")),
        "remote": _git(root, "remote", "get-url", "origin"),
    }


def _git(root: Path, *args: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), *args],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        return result.stdout.strip() if result.returncode == 0 else ""
    except Exception:
        return ""
