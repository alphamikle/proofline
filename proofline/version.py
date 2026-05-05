from __future__ import annotations

import re
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from typing import Any, Dict

from proofline import __version__
from proofline.upgrade import DEFAULT_REPO

SEMVER_RE = re.compile(r"^(\d+)\.(\d+)\.(\d+)$")


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


def update_check(timeout_seconds: float = 1.5) -> Dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    remote = _git(root, "remote", "get-url", "origin") or DEFAULT_REPO
    latest_version = _latest_remote_semver_tag(remote, timeout_seconds)
    if not latest_version:
        return {"available": False, "reason": "remote semver tag unavailable", "repo": remote, "ref": "main"}

    current_version = _metadata_version() or __version__ or ""
    current_semver = _parse_semver(current_version)
    latest_semver = _parse_semver(latest_version)
    update_available = latest_semver is not None and (current_semver is None or latest_semver > current_semver)
    return {
        "available": True,
        "update_available": update_available,
        "repo": remote,
        "ref": "main",
        "current_version": current_version or None,
        "latest_version": latest_version,
        "command": "pfl upgrade",
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


def _parse_semver(value: str | None) -> tuple[int, int, int] | None:
    if not value:
        return None
    match = SEMVER_RE.match(value.strip())
    if not match:
        return None
    if any(len(part) > 1 and part.startswith("0") for part in match.groups()):
        return None
    return tuple(int(part) for part in match.groups())


def _latest_remote_semver_tag(repo: str, timeout_seconds: float) -> str:
    try:
        result = subprocess.run(
            ["git", "ls-remote", "--tags", "--refs", repo],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=timeout_seconds,
        )
    except Exception:
        return ""
    if result.returncode != 0:
        return ""
    versions: list[tuple[tuple[int, int, int], str]] = []
    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        tag = parts[1].removeprefix("refs/tags/")
        parsed = _parse_semver(tag)
        if parsed is not None:
            versions.append((parsed, tag))
    if not versions:
        return ""
    return max(versions, key=lambda item: item[0])[1]
